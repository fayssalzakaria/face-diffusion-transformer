import torch
import torch.nn as nn

# Paramètres de configuration
img_size = 128       # Résolution augmentée 64→128 pour des visages plus nets
z_dim = 100          # Dimension du vecteur latent
channels = 3         # Nombre de canaux (RGB)
feature_maps = 64    # Facteur multiplicatif pour les filtres

# Générateur DCGAN 128x128
# Architecture : z(100,1,1) → 4x4 → 8x8 → 16x16 → 32x32 → 64x64 → 128x128
class Generator(nn.Module):
    def __init__(self, z_dim, channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Entrée : vecteur latent (z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, feature_maps * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            # (feature_maps*16) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # (feature_maps*8) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # (feature_maps*4) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # (feature_maps*2) x 32 x 32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # (feature_maps) x 64 x 64
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Sortie : (channels) x 128 x 128
        )

    def forward(self, z):
        return self.net(z)


# Discriminateur DCGAN 128x128
# Spectral Normalization sur chaque couche Conv :
# - contrôle le gradient de Lipschitz du discriminateur
# - stabilise l'entraînement sans avoir besoin de WGAN-GP
# - remplace avantageusement BatchNorm dans le discriminateur
class Discriminator(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Discriminator, self).__init__()
        # Alias pour alléger l'écriture
        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            # Entrée : (channels) x 128 x 128
            SN(nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps) x 64 x 64
            SN(nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*2) x 32 x 32
            SN(nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*4) x 16 x 16
            SN(nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*8) x 8 x 8
            SN(nn.Conv2d(feature_maps * 8, feature_maps * 16, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(feature_maps * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # (feature_maps*16) x 4 x 4
            SN(nn.Conv2d(feature_maps * 16, 1, kernel_size=4, stride=1, padding=0, bias=False)),
            # Sortie : scalaire (pas de Sigmoid — intégré dans BCEWithLogitsLoss)
        )

    def forward(self, img):
        return self.net(img).view(-1, 1)