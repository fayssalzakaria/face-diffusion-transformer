if __name__ == '__main__':  # Vérifie que le script est exécuté directement et non importé comme un module
    import torch.multiprocessing as mp
    # Importation des bibliothèques nécessaires
    import torch  # Framework de deep learning
    import torch.nn as nn  # Module contenant les couches de réseaux de neurones
    import torch.optim as optim  # Module pour les algorithmes d'optimisation
    import torchvision.transforms as transforms  # Module pour transformer les images
    from torch.utils.data import DataLoader  # Classe pour charger les données de manière efficace
    from torch.amp import autocast, GradScaler
    import os  # Module pour l'interaction avec le système de fichiers
    from model import Generator, Critic # Importation des modèles de génération et de discrimination
    import matplotlib.pyplot as plt  # Module pour l'affichage des images
    import numpy as np  # Bibliothèque pour la manipulation de matrices et tableaux
    from dataset import CustomDataset  # Importation du dataset personnalisé
    import matplotlib
    import time
    mp.set_start_method('spawn', force=True)  # Force un démarrage propre des processus
    matplotlib.use('Agg')  # Désactive Tkinter

    # **Configuration des paramètres d'entraînement**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Sélection du GPU si disponible, sinon CPU
    #Amp active
    print(f"Entraînement sur : {device}")
    use_amp = device.type == "cuda"

    print(f"Mixed Precision (AMP) : {'activé' if use_amp else 'désactivé (CPU)'}")
    img_size = 128  # Taille des images (64x64 pixels)
    batch_size = 64 # Nombre d'images par batch
    lambda_gp  = 15
    n_critic   = 5
    epochs = 100  # Nombre total d'époques d'entraînement
    z_dim = 100  # Taille du vecteur de bruit pour le générateur (latent space)
    channels = 3  # Nombre de canaux (RGB = 3)
    feature_maps = 64  # Nombre de cartes de caractéristiques dans le réseau
    data_dir = "data/ffhq"

    # Vérification de l'existence du dataset
    if not os.path.exists(data_dir):
        raise RuntimeError("Le dataset CelebA n'a pas été trouvé. Lance d'abord utils.py !")

    # Création d'un dossier pour stocker les images générées si non existant
    os.makedirs("generated_images", exist_ok=True)

    # **Définition des transformations à appliquer aux images**
    transform = transforms.Compose([
        transforms.Resize(img_size),  # Redimensionnement à 64x64 pixels
        transforms.CenterCrop(img_size),  # Recadrage centré à 64x64
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # Conversion en tenseur PyTorch
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation des pixels entre [-1,1]
    ])

    # **Calcul adaptatif du nombre de workers selon le matériel disponible**
    # On prend min(nb_CPU, 8) pour éviter la surcharge sur les petites machines,
    # et on tombe à 0 sur CPU-only (pas de gain à paralléliser sans GPU).
    num_workers = min(os.cpu_count() or 1, 8) if device.type == "cuda" else 0
    print(f"DataLoader : {num_workers} worker(s), prefetch_factor={'2' if num_workers > 0 else 'N/A'}")

 
    # **Chargement du dataset avec le DataLoader optimisé**
    dataset = CustomDataset(root=data_dir, transform=transform,max_images=None)  # Création du dataset personnalisé
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        # --- Optimisations DataLoader ---
        num_workers=num_workers,        # Adaptatif : évite les 4 workers inutiles sur CPU-only
        pin_memory=(device.type == "cuda"),  # Transfert CPU→GPU plus rapide via mémoire paginée
        persistent_workers=(num_workers > 0),  # Garde les workers vivants entre les époques (évite le fork à chaque époque)
        prefetch_factor=2 if num_workers > 0 else None,  # Pré-charge 2 batches en avance pendant le calcul GPU
        drop_last=True,  # Ignore le dernier batch incomplet pour des tailles stables (BCELoss plus stable)
    )
 

    # **Initialisation des modèles**
    generator = Generator(z_dim, channels, feature_maps).to(device)
    critic = Critic(channels, feature_maps).to(device)
    print("torch.compile : désactivé (Windows)")
    # **Définition de la fonction de perte et des optimisateurs**
    criterion = nn.BCEWithLogitsLoss()  # Fonction de perte pour la classification binaire (vrai/faux)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))  
    # - `lr=0.0001` : Taux d'apprentissage
    # - `betas=(0.5, 0.999)` : Paramètres de l'optimiseur Adam pour la stabilité

    optimizer_C = optim.Adam(critic.parameters(), lr=0.00005, betas=(0.5, 0.999))  
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=1e-5)
    scheduler_C = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=epochs, eta_min=1e-5)
    #Ajout des scalers
    scaler_G = GradScaler(device=device.type, enabled=use_amp)
    scaler_C = GradScaler(device=device.type, enabled=use_amp)
 
    def gradient_penalty(critic, real, fake, device):
        B, C, H, W = real.shape
        alpha = torch.rand(B, 1, 1, 1, device=device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        with autocast(device_type=device.type, enabled=False):
            score = critic(interpolated.float())
        gradients = torch.autograd.grad(
            outputs=score,
            inputs=interpolated,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(B, -1)
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp
    
    # **Fonction pour générer et sauvegarder des images à chaque époque**
    def generate_and_save_images(epoch, fixed_noise):
        """
        Génère et enregistre un lot d'images synthétiques à partir d'un bruit fixé.
        :param epoch: Numéro de l'époque en cours.
        :param fixed_noise: Vecteur de bruit constant pour suivre l'évolution du modèle.
        """
        generator.eval()  # Mode évaluation (désactive le dropout et la BN)
        with torch.no_grad():
            with autocast(device_type=device.type, enabled=use_amp):
                fake_images = generator(fixed_noise).cpu()
        fake_images = (fake_images.float() + 1) / 2  # .float() convertit FP16→FP32 pour matplotlib

        # Affichage des images générées sous forme de grille 4x4
        fig, axes = plt.subplots(4, 4, figsize=(6, 6))
        for i, ax in enumerate(axes.flatten()):
            img = np.transpose(fake_images[i].numpy(), (1, 2, 0))  # Réorganisation des dimensions (H, W, C)
            ax.imshow(img)  # Affichage de l'image
            ax.axis("off")  # Suppression des axes
        
        # Sauvegarde de l'image
        plt.savefig(f"generated_images/epoch_{epoch}.png")
        plt.close(fig)
        plt.close()
        generator.train()  # Retour en mode entraînement

    # **Création d'un vecteur de bruit fixe pour suivre la progression du modèle**
  # **Création d'un vecteur de bruit fixe pour suivre la progression du modèle**
    fixed_noise = torch.randn(16, z_dim, 1, 1, device=device)

    # **Boucle d'entraînement du GAN**
    for epoch in range(epochs):
        t0 = time.time()
        for batch_idx, (real_images, _) in enumerate(dataloader):
            # Récupération d'un batch d'images réelles
            real_images = real_images.to(device, non_blocking=True)  # non_blocking tire parti de pin_memory
            batch_size_cur = real_images.size(0)

            # **Entraînement du critique (n_critic fois)**
            for _ in range(n_critic):
                noise = torch.randn(batch_size_cur, z_dim, 1, 1, device=device)
                optimizer_C.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_images  = generator(noise)
                    score_real   = critic(real_images)
                    score_fake   = critic(fake_images.detach())
                    loss_C_w     = -(score_real.mean() - score_fake.mean())
                gp     = gradient_penalty(critic, real_images.detach(), fake_images.detach(), device)
                loss_C = loss_C_w + lambda_gp * gp
                scaler_C.scale(loss_C).backward()
                scaler_C.step(optimizer_C)
                scaler_C.update()

            # **Entraînement du générateur (1 fois)**
            optimizer_G.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                noise       = torch.randn(batch_size_cur, z_dim, 1, 1, device=device)
                fake_images = generator(noise)
                score_fake  = critic(fake_images)
                loss_G      = -score_fake.mean()
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            # Affichage des pertes toutes les 100 itérations
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}] - "
                      f"Loss C: {loss_C.item():.4f}, Loss G: {loss_G.item():.4f}, GP: {gp.item():.4f}")

        # Sauvegarde des images générées à la fin de chaque époque
        scheduler_G.step()
        scheduler_C.step()
        elapsed = time.time() - t0
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f"→ Epoch {epoch+1} terminée en {elapsed:.0f}s | LR: {current_lr:.6f}")
        generate_and_save_images(epoch+1, fixed_noise)

    print("Entraînement terminé !")