"""
Entraînement de la baseline WGAN-GP (128 ou 256 px).

Usage : python train.py --img_size 256 --epochs 57 --max_hours 12
        python train.py --resume checkpoints/wgan_complet.pt   (reprise après interruption)
"""
if __name__ == '__main__':  # indispensable sous Windows : les workers du DataLoader réimportent ce fichier
    import torch.multiprocessing as mp
    import torch
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch.amp import autocast, GradScaler
    import os
    from model import Generator, Critic
    import matplotlib.pyplot as plt
    import numpy as np
    from dataset import CustomDataset
    import matplotlib
    import time
    import argparse
    mp.set_start_method('spawn', force=True)
    matplotlib.use('Agg')  # pas de fenêtre : on sauvegarde seulement les figures

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=128, choices=[128, 256])
    parser.add_argument("--max_hours", type=float, default=None,
                        help="budget de temps : arrêt propre en fin d'époque une fois dépassé")
    parser.add_argument("--resume", default=None, help="checkpoint complet (checkpoints/wgan_complet.pt)")
    parser.add_argument("--max_images", type=int, default=None, help="sous-ensemble du dataset (tests rapides)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # précision mixte FP16 : ~2x plus rapide sur GPU
    print(f"Entraînement sur : {device}")
    print(f"Mixed Precision (AMP) : {'activé' if use_amp else 'désactivé (CPU)'}")

    img_size = args.img_size
    batch_size = 64
    lambda_gp = 15   # poids du gradient penalty (10 dans le papier, 15 réduisait les artefacts)
    n_critic = 5     # le critique s'entraîne 5 fois par pas du générateur
    epochs = args.epochs
    z_dim = 100
    channels = 3
    feature_maps = 64
    data_dir = "data/ffhq"

    if not os.path.exists(data_dir):
        raise RuntimeError("Dataset FFHQ introuvable dans data/ffhq : lance d'abord utils.py.")

    os.makedirs("generated_images", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)  # poids du générateur, utilisés par experiences.py

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # pixels dans [-1, 1], comme la sortie Tanh
    ])

    # Le décodage des PNG se fait sur CPU : plusieurs workers évitent de faire attendre le GPU
    num_workers = min(os.cpu_count() or 1, 8) if device.type == "cuda" else 0
    print(f"DataLoader : {num_workers} worker(s), prefetch_factor={'2' if num_workers > 0 else 'N/A'}")

    dataset = CustomDataset(root=data_dir, transform=transform, max_images=args.max_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),     # transfert CPU → GPU plus rapide
        persistent_workers=(num_workers > 0),   # pas de relance des workers à chaque époque
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,                         # batches de taille constante
    )

    generator = Generator(z_dim, channels, feature_maps, img_size).to(device)
    critic = Critic(channels, feature_maps, img_size).to(device)
    print("torch.compile : désactivé (Windows)")

    # Le critique apprend 2x moins vite que le générateur : sinon il domine et le générateur n'apprend plus
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_C = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-5)
    scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs, eta_min=1e-5)
    scaler_G = GradScaler(device=device.type, enabled=use_amp)
    scaler_C = GradScaler(device=device.type, enabled=use_amp)

    def gradient_penalty(critic, real, fake, device):
        """Pénalise le critique si son gradient s'éloigne de 1 entre images réelles et générées (Lipschitz)."""
        B, C, H, W = real.shape
        alpha = torch.rand(B, 1, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with autocast(device_type=device.type, enabled=False):  # en FP32 : le gradient du gradient est fragile en FP16
            score = critic(interpolated.float())
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def generate_and_save_images(epoch, fixed_noise):
        """Grille 4x4 générée depuis un bruit fixe, pour suivre l'évolution d'une époque à l'autre."""
        generator.eval()
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                fake_images = generator(fixed_noise).cpu()
        fake_images = (fake_images.float() + 1) / 2  # [-1, 1] → [0, 1], et FP16 → FP32 pour matplotlib

        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(np.transpose(fake_images[i].numpy(), (1, 2, 0)))
            ax.axis("off")
        plt.savefig(f"generated_images/epoch_{epoch}.png")
        plt.close(fig)
        plt.close()
        generator.train()

    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)

    # Reprise : on recharge tout l'état, pas seulement les poids, pour continuer exactement au même point
    start_epoch, elapsed_total = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        for obj, key in [(generator, "generator"), (critic, "critic"), (optimizer_G, "optimizer_G"),
                         (optimizer_C, "optimizer_C"), (scheduler_G, "scheduler_G"), (scheduler_C, "scheduler_C"),
                         (scaler_G, "scaler_G"), (scaler_C, "scaler_C")]:
            obj.load_state_dict(ckpt[key])
        fixed_noise = ckpt["fixed_noise"]
        start_epoch, elapsed_total = ckpt["epoch"], ckpt["elapsed_total"]
        print(f"Reprise à l'époque {start_epoch}, {elapsed_total / 3600:.2f} h déjà écoulées")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device, non_blocking=True)
            batch_size_cur = real_images.size(0)

            # Critique : distance de Wasserstein (score réel − score généré) + gradient penalty
            for _ in range(n_critic):
                noise = torch.randn(batch_size_cur, z_dim, 1, 1, device=device)
                optimizer_C.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_images = generator(noise)
                    score_real = critic(real_images)
                    score_fake = critic(fake_images.detach())
                    loss_C_w = -(score_real.mean() - score_fake.mean())
                gp = gradient_penalty(critic, real_images.detach(), fake_images.detach(), device)
                loss_C = loss_C_w + lambda_gp * gp
                scaler_C.scale(loss_C).backward()
                scaler_C.step(optimizer_C)
                scaler_C.update()

            # Générateur : maximiser le score que le critique donne à ses images
            optimizer_G.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                noise = torch.randn(batch_size_cur, z_dim, 1, 1, device=device)
                fake_images = generator(noise)
                loss_G = -critic(fake_images).mean()
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}] - "
                      f"Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}, GP: {gp.item():.4f}")

        scheduler_G.step()
        scheduler_C.step()
        elapsed = time.time() - t0
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"→ Epoch {epoch+1} terminée en {elapsed:.0f}s | LR: {current_lr:.6f}")
        generate_and_save_images(epoch+1, fixed_noise)

        # Deux sauvegardes : les poids du générateur (pour générer) et l'état complet (pour reprendre)
        torch.save(generator.state_dict(), "checkpoints/wgan_generator.pt")
        elapsed_total += elapsed
        torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(), "optimizer_C": optimizer_C.state_dict(),
                    "scheduler_G": scheduler_G.state_dict(), "scheduler_C": scheduler_C.state_dict(),
                    "scaler_G": scaler_G.state_dict(), "scaler_C": scaler_C.state_dict(),
                    "fixed_noise": fixed_noise, "epoch": epoch + 1, "elapsed_total": elapsed_total,
                    "img_size": img_size},
                   "checkpoints/wgan_complet.pt")
        print(f"Temps d'entraînement cumulé : {elapsed_total / 3600:.2f} h")
        if args.max_hours is not None and elapsed_total >= args.max_hours * 3600:
            print(f"Budget de {args.max_hours} h atteint : arrêt à l'époque {epoch + 1}")
            break

    print("Entraînement terminé !")
