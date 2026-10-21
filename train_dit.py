"""
Entraîne le Diffusion Transformer sur les latents FFHQ pré-calculés (prepare_latents.py).

Objectif : flow matching / rectified flow (comme Stable Diffusion 3)
  x_t = (1 - t) · x_0 + t · ε      v_cible = ε − x_0      perte = ||v_θ(x_t, t, a) − v_cible||²

Usage : python train_dit.py --model B --steps 150000
"""
import argparse
import copy
import math
import os
import time
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from dit import DiT, CONFIGS, AttributeEmbedder, sample
from vae import load_vae, decode


def attrs_from_probs(probs, low=0.3, high=0.7):
    """Probabilités CLIP → {0, 1, 2}. Entre les deux seuils, CLIP hésite : attribut "non spécifié"."""
    attrs = torch.full(probs.shape, AttributeEmbedder.UNKNOWN, dtype=torch.long)
    attrs[probs >= high] = 1
    attrs[probs <= low] = 0
    return attrs


def drop_attributes(attrs, p_all=0.1, p_each=0.15):
    """
    Classifier-free guidance : on masque tous les attributs (p_all) pour apprendre le modèle
    inconditionnel, et chaque attribut séparément (p_each) pour pouvoir n'en fixer que quelques-uns.
    """
    B, K = attrs.shape
    drop = torch.rand(B, K, device=attrs.device) < p_each
    drop |= (torch.rand(B, 1, device=attrs.device) < p_all)
    return attrs.masked_fill(drop, AttributeEmbedder.UNKNOWN)


@torch.no_grad()
def transfer_weights(model, path):
    """
    Initialise un DiT depuis un autre DiT de même taille, éventuellement avec une autre taille de patch.
    Les blocs Transformer et les embeddings (temps, attributs) sont copiés tels quels.
    Pour passer de patchs p×p à des patchs plus petits, on convertit l'entrée et la sortie :
      - entrée : somme du noyau p×p (réponse à une zone uniforme) → noyau 1×1 équivalent
      - sortie : moyenne des p×p prédictions de chaque token → une prédiction par pixel latent
    """
    src = torch.load(path, map_location="cpu")
    weights = src["ema"]
    p_src = src["config"].get("patch_size", 2)
    p_dst = model.patch_size
    state = model.state_dict()
    for name, value in weights.items():
        if name == "pos_embed":
            continue  # recalculé (sinus/cosinus) pour la nouvelle grille de tokens
        if name == "patch_embed.weight" and value.shape != state[name].shape:
            k = p_src // p_dst
            value = value.unflatten(2, (p_dst, k)).unflatten(4, (p_dst, k)).sum(dim=(3, 5))
        elif name.startswith("final.linear.") and value.shape != state[name].shape:
            c = model.in_channels
            k = p_src // p_dst
            rest = value.shape[1:]
            value = value.view(p_dst, k, p_dst, k, c, *rest).mean(dim=(1, 3)).reshape(-1, *rest)
        state[name] = value
    model.load_state_dict(state)
    print(f"Poids transférés depuis {path} (patch {p_src} → {p_dst}, étape {src['step']})")


@torch.no_grad()
def update_ema(ema, model, decay):
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.lerp_(p, 1 - decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", default="data/latents_ffhq128.pt")
    parser.add_argument("--model", default="B", choices=CONFIGS.keys())
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay_start", type=int, default=None,
                        help="étape à partir de laquelle le LR décroît (cosinus) jusqu'à --min_lr à la fin")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--ckpt_every", type=int, default=10000)
    parser.add_argument("--out_dir", default="runs/dit")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--patch_size", type=int, default=2, help="2 : 64 tokens de 16x16 px ; 1 : 256 tokens de 8x8 px")
    parser.add_argument("--init_from", default=None,
                        help="checkpoint d'un DiT déjà entraîné (même taille) dont on transfère les poids")
    parser.add_argument("--warmup", type=int, default=0, help="étapes de montée progressive du LR")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    os.makedirs(f"{args.out_dir}/samples", exist_ok=True)

    # Tout le dataset est en mémoire : pas de DataLoader, pas de lecture d'images.
    # En 128 (213 Mo) il tient sur le GPU ; en 256 (850 Mo) on le garde en RAM pour laisser la place au modèle.
    data = torch.load(args.latents)
    mean, std = data["mean"].view(1, -1, 1, 1).to(device), data["std"].view(1, -1, 1, 1).to(device)
    latents = torch.stack([data["latents"], data["latents_flip"]])  # (2, N, 4, H, W)
    latents = latents.to(device) if latents.shape[-1] <= 16 else latents.pin_memory()
    attrs_all = attrs_from_probs(data["attr_probs"]).to(device)
    N = latents.shape[1]
    num_attributes = attrs_all.shape[1]
    print(f"{N} visages, attributs : {data['attr_names']}")

    model = DiT(input_size=latents.shape[-1], patch_size=args.patch_size, num_attributes=num_attributes,
                **CONFIGS[args.model]).to(device)
    if args.init_from:
        transfer_weights(model, args.init_from)
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"DiT-{args.model}/{args.patch_size} : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M paramètres")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    step = 0
    if args.resume:
        # Chargé sur CPU puis libéré : sinon 2 Go restent sur le GPU et la mémoire déborde
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])  # les états Adam sont déplacés sur le GPU des paramètres
        step = ckpt["step"]
        del ckpt
        torch.cuda.empty_cache()
        print(f"Reprise à l'étape {step}")

    # Grille de suivi : mêmes bruits et mêmes attributs à chaque fois
    vae = load_vae(device)
    g = torch.Generator(device=device).manual_seed(0)
    fixed_attrs = attrs_all[torch.randperm(N, device=device, generator=g)[:16]]

    def save_checkpoint(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "config": dict(model=args.model, input_size=latents.shape[-1],
                                                 num_attributes=num_attributes, patch_size=args.patch_size),
                    "attr_names": data["attr_names"], "mean": data["mean"], "std": data["std"]}, path)

    def lr_at(step):
        """Montée linéaire (warmup), LR constant, puis décroissance en cosinus : le modèle "se pose"."""
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        if args.decay_start is None or step < args.decay_start:
            return args.lr
        progress = (step - args.decay_start) / max(1, args.steps - args.decay_start)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    model.train()
    t0 = time.time()
    while step < args.steps:
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        idx = torch.randint(0, N, (args.batch_size,), device=latents.device)
        flip = torch.randint(0, 2, (args.batch_size,), device=latents.device)
        x0 = (latents[flip, idx].to(device, non_blocking=True).float() - mean) / std
        attrs = drop_attributes(attrs_all[idx.to(device)])

        # t ~ logit-normal : plus d'échantillons aux niveaux de bruit intermédiaires (SD3)
        t = torch.sigmoid(torch.randn(args.batch_size, device=device))
        noise = torch.randn_like(x0)
        t_ = t.view(-1, 1, 1, 1)
        xt = (1 - t_) * x0 + t_ * noise

        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(xt, t, attrs)
        loss = F.mse_loss(v_pred.float(), noise - x0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema, model, args.ema_decay if step > 1000 else 0.0)
        step += 1

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"step {step}/{args.steps} | loss {loss.item():.4f} | lr {lr_at(step):.2e} | {100 / elapsed:.1f} it/s")
            t0 = time.time()

        if step % args.sample_every == 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = sample(ema, fixed_attrs, steps=50, cfg_scale=3.0,
                           generator=torch.Generator(device=device).manual_seed(0))
                images = decode(vae, z.float() * std + mean)
            vutils.save_image(images.float(), f"{args.out_dir}/samples/step_{step:06d}.png", nrow=4)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(f"{args.out_dir}/last.pt")

    print("Entraînement terminé !")


if __name__ == "__main__":
    main()
