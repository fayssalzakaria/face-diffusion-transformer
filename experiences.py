"""
Toutes les expériences de comparaison, en une commande, sur le modèle final :

  1. FID du DiT pour plusieurs valeurs de guidance (cfg)      → courbe "FID en fonction du cfg"
  2. FID du DiT pour plusieurs nombres d'étapes (au meilleur cfg)
  3. FID de la baseline WGAN-GP et de la reconstruction VAE (plafond)
  4. Temps de génération par image de chaque modèle
  5. Grilles visuelles : effet du cfg, effet du nombre d'étapes, chaque attribut, WGAN vs DiT

FID = distance entre les statistiques (moyenne, covariance) des features Inception
des images réelles et générées. Plus bas = meilleur. Les statistiques des images
réelles sont calculées UNE fois puis réutilisées pour toutes les mesures.

Les résultats déjà calculés sont gardés dans results/fid.json : si le script est
interrompu, on le relance et il reprend là où il s'était arrêté.

Usage : python experiences.py                    (≈ 1 h sur RTX 4070)
        python experiences.py --num 5000         (≈ 2x plus rapide, un peu moins précis)
"""
import argparse
import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from dataset import CustomDataset
from model import Generator
from pipeline import FaceGenerator
from train_dit import attrs_from_probs

DIT_COLOR = "#2a78d6"
REF_COLOR = "#6b6a63"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ffhq")
    parser.add_argument("--latents", default=None, help="par défaut data/latents_ffhq<résolution>.pt")
    parser.add_argument("--dit_ckpt", default="runs/dit/last.pt")
    parser.add_argument("--wgan_ckpt", default="checkpoints/wgan_generator.pt")
    parser.add_argument("--out", default="results")
    parser.add_argument("--num", type=int, default=10_000, help="images par mesure de FID")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--cfgs", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--steps_list", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--compare_ckpts", nargs="*", default=["runs/dit/step_080000_ema.pt"],
                        help="checkpoints intermédiaires à mesurer au meilleur cfg (l'entraînement a-t-il encore servi ?)")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class FIDBench:
    """Garde les stats des images réelles, mesure n'importe quel générateur, met les résultats en cache."""

    def __init__(self, args, device, img_size):
        self.args, self.device, self.img_size = args, device, img_size
        self.path = os.path.join(args.out, "fid.json")
        self.results = json.load(open(self.path)) if os.path.exists(self.path) else {}
        if self.results.get("_num") not in (None, args.num):
            print("Nombre d'images différent du cache : on repart de zéro.")
            self.results = {}
        self.results["_num"] = args.num
        self.fid = None

    def real_loader(self):
        s = self.img_size
        transform = transforms.Compose([transforms.Resize(s), transforms.CenterCrop(s), transforms.ToTensor()])
        real = CustomDataset(self.args.data_dir, transform=transform)
        g = torch.Generator().manual_seed(0)
        real.images = [real.images[i] for i in torch.randperm(len(real.images), generator=g)[:self.args.num]]
        return DataLoader(real, batch_size=self.args.batch_size,
                          num_workers=min(os.cpu_count() or 1, 8) if self.device.type == "cuda" else 0)

    def ensure_real_stats(self):
        if self.fid is not None:
            return
        # reset_real_features=False : reset() n'efface que le côté "images générées"
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(self.device)
        for images, _ in tqdm(self.real_loader(), desc="Stats des images réelles (une seule fois)"):
            self.fid.update(images.to(self.device), real=True)

    def measure(self, key, make_batch):
        """make_batch(b, offset) → b images dans [0, 1]. Renvoie (FID, secondes par image)."""
        if key in self.results:
            print(f"[cache] {key} : FID {self.results[key]['fid']:.2f}")
            return self.results[key]
        self.ensure_real_stats()
        self.fid.reset()
        gen_time, done = 0.0, 0
        with tqdm(total=self.args.num, desc=key) as bar:
            while done < self.args.num:
                b = min(self.args.batch_size, self.args.num - done)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                images = make_batch(b, done)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                gen_time += time.time() - t0
                self.fid.update(images.float().clamp(0, 1).to(self.device), real=False)
                done += b
                bar.update(b)
        self.results[key] = {"fid": self.fid.compute().item(), "sec_par_image": gen_time / self.args.num}
        json.dump(self.results, open(self.path, "w"), indent=2)
        print(f"{key} : FID {self.results[key]['fid']:.2f}")
        return self.results[key]


def save_labeled_grid(rows, path, title):
    """rows : liste de (étiquette, tenseur (n, 3, H, W)). Une ligne par réglage, étiquette à gauche."""
    n_cols = rows[0][1].shape[0]
    k = rows[0][1].shape[-1] / 128  # images 256 : figure 2x plus grande pour garder la résolution
    fig, axes = plt.subplots(len(rows), 1, figsize=(n_cols * 1.3 * k + 1.2, len(rows) * 1.35 * k + 0.5))
    axes = [axes] if len(rows) == 1 else axes
    for ax, (label, images) in zip(axes, rows):
        grid = vutils.make_grid(images.cpu().float(), nrow=n_cols, padding=2, pad_value=1.0)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10, color="#1f1e1c")
    fig.suptitle(title, fontsize=12, color="#1f1e1c")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))  # réserve la place du titre
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Grille : {path}")


def plot_fid_vs_cfg(cfgs, fids, refs, path):
    """Une seule série (DiT) + lignes de référence pointillées, étiquetées directement."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(cfgs, fids, color=DIT_COLOR, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    best = min(range(len(fids)), key=lambda i: fids[i])
    ax.annotate(f"meilleur : cfg {cfgs[best]:g}\nFID {fids[best]:.2f}", (cfgs[best], fids[best]),
                textcoords="offset points", xytext=(0, 14), ha="center", fontsize=9, color="#1f1e1c")
    for label, value in refs.items():
        ax.axhline(value, color=REF_COLOR, linestyle="--", linewidth=1, zorder=1)
        ax.text(cfgs[-1], value, f" {label} ({value:.2f})", va="bottom", ha="right", fontsize=9, color=REF_COLOR)
    ax.set_xlabel("Guidance (cfg)", color="#1f1e1c")
    ax.set_ylabel("FID (plus bas = meilleur)", color="#1f1e1c")
    ax.set_title("DiT : FID en fonction du guidance (50 étapes)", fontsize=12, color="#1f1e1c")
    ax.set_xticks(cfgs)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#55544f")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"Graphique : {path}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    gen = FaceGenerator(args.dit_ckpt, args.device)
    device = gen.device
    bench = FIDBench(args, device, gen.img_size)  # FID calculé à la résolution du DiT
    args.latents = args.latents or f"data/latents_ffhq{gen.img_size}.pt"
    print(f"Résolution : {gen.img_size}x{gen.img_size}")

    # Pour le FID, les attributs sont tirés de la distribution réelle : sinon le guidance n'a aucun effet
    attrs_real = attrs_from_probs(torch.load(args.latents)["attr_probs"]).to(device)

    def dit_batch(cfg, steps):
        def make(b, offset):
            idx = torch.randint(0, len(attrs_real), (b,), device=device,
                                generator=torch.Generator(device=device).manual_seed(offset))
            return gen.generate(attrs_real[idx], cfg=cfg, steps=steps, seed=1000 + offset)
        return make

    # --- Grilles visuelles (rapides) ---
    demo = {"souriant": 1, "lunettes": 1}
    save_labeled_grid([(f"cfg {c:g}", gen.generate(demo, n=8, cfg=c, seed=7)) for c in args.cfgs],
                      f"{args.out}/grille_cfg.png", "Effet du guidance — consigne : souriant + lunettes")
    save_labeled_grid([(f"{s} étapes", gen.generate(demo, n=8, cfg=3.0, steps=s, seed=7)) for s in args.steps_list],
                      f"{args.out}/grille_etapes.png", "Effet du nombre d'étapes (cfg 3)")
    save_labeled_grid([(f"{name}=1", gen.generate({name: 1}, n=8, cfg=3.0, seed=3)) for name in gen.attr_names],
                      f"{args.out}/grille_attributs.png", "Un attribut imposé par ligne, le reste libre")

    wgan = None
    if os.path.exists(args.wgan_ckpt):
        wgan = Generator(100, 3, 64, gen.img_size).to(device)
        wgan.load_state_dict(torch.load(args.wgan_ckpt, map_location=device))  # doit être à la même résolution
        wgan.eval()

    def wgan_batch(b, offset):
        g = torch.Generator(device=device).manual_seed(offset)
        with torch.no_grad():
            return (wgan(torch.randn(b, 100, 1, 1, device=device, generator=g)) + 1) / 2

    real_imgs = next(iter(bench.real_loader()))[0][:8]
    rows = [("Réel (FFHQ)", real_imgs)]
    if wgan is not None:
        rows.append(("WGAN-GP", wgan_batch(8, 0)))
    rows.append(("DiT", gen.generate(None, n=8, cfg=3.0, seed=11)))
    save_labeled_grid(rows, f"{args.out}/grille_comparaison.png", "Images réelles vs modèles (tirages non choisis)")

    # Morphing en GIF pour le README
    frames = gen.interpolate(1, 2, frames=24)
    frames = [transforms.functional.to_pil_image(f.cpu()) for f in frames]
    frames = frames + frames[::-1]
    frames[0].save(f"{args.out}/morphing.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)

    # --- Mesures FID (longues) ---
    def recon_batch_factory():
        it = iter(bench.real_loader())
        return lambda b, offset: gen.reconstruct(next(it)[0])
    refs = {"Reconstruction VAE": bench.measure("Reconstruction VAE", recon_batch_factory())["fid"]}
    if wgan is not None:
        refs["WGAN-GP"] = bench.measure("WGAN-GP", wgan_batch)["fid"]
    else:
        print(f"Pas de checkpoint WGAN-GP ({args.wgan_ckpt}) : baseline ignorée.")

    cfg_fids = [bench.measure(f"DiT cfg={c:g} steps=50", dit_batch(c, 50))["fid"] for c in args.cfgs]

    # Comparaison stricte avec le WGAN-GP : aucun attribut fourni, aucun guidance (génération inconditionnelle)
    bench.measure("DiT-inconditionnel cfg=1 steps=50",
                  lambda b, offset: gen.generate(None, n=b, cfg=1.0, steps=50, seed=1000 + offset))
    best_cfg = args.cfgs[min(range(len(cfg_fids)), key=lambda i: cfg_fids[i])]
    for s in args.steps_list:
        bench.measure(f"DiT cfg={best_cfg:g} steps={s}", dit_batch(best_cfg, s))

    # Convergence : même mesure sur un checkpoint intermédiaire
    final_gen = gen
    for path in args.compare_ckpts:
        if not os.path.exists(path):
            continue
        gen = FaceGenerator(path, args.device)
        ck = torch.load(path, map_location="cpu")
        label = f"DiT-{ck['config']['model']}/{ck['config'].get('patch_size', 2)}-{ck['step'] // 1000}k"
        bench.measure(f"{label} cfg={best_cfg:g} steps=50", dit_batch(best_cfg, 50))
        del gen
    gen = final_gen
    plot_fid_vs_cfg(args.cfgs, cfg_fids, refs, f"{args.out}/fid_vs_cfg.png")

    # --- Tableau récapitulatif (Markdown, à coller dans le README) ---
    lines = ["| Modèle | Réglage | FID ↓ | Temps / image |", "|---|---|---|---|"]
    for key, r in bench.results.items():
        if key.startswith("_"):
            continue
        name, _, setting = key.partition(" ")
        if key == "Reconstruction VAE":
            name, setting = key, "plafond du DiT"
        lines.append(f"| {name} | {setting or '—'} | {r['fid']:.2f} | {r['sec_par_image'] * 1000:.1f} ms |")
    table = "\n".join(lines)
    open(f"{args.out}/fid_tableau.md", "w", encoding="utf-8").write(table + "\n")
    print("\n" + table)
    print(f"\nMeilleur guidance : cfg = {best_cfg:g}")


if __name__ == "__main__":
    main()
