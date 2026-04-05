import torch
import torch.nn as nn

# Générateur WGAN-GP 128x128
class Generator(nn.Module):
    def __init__(self, z_dim, channels, feature_maps):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # (z_dim, 1, 1) → 4x4
            nn.ConvTranspose2d(z_dim, feature_maps * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(True),
            nn.Dropout2d(0.3),  # réduit les artefacts de couleur
            # 4x4 → 8x8
            nn.ConvTranspose2d(feature_maps * 16, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.Dropout2d(0.3),  # réduit le surapprentissage
            # 8x8 → 16x16
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# Critique WGAN-GP (pas de Sigmoid, pas de BatchNorm → remplacé par LayerNorm)
# Le critique prédit un score réel (pas une probabilité)
# LayerNorm à la place de BatchNorm : compatible avec le gradient penalty
class Critic(nn.Module):
    def __init__(self, channels, feature_maps):
        super(Critic, self).__init__()
        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            # 128x128 → 64x64
            SN(nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 → 32x32
            SN(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 → 16x16
            SN(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 → 8x8
            SN(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 → 4x4
            SN(nn.Conv2d(feature_maps * 8, feature_maps * 16, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(feature_maps * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # 4x4 → scalaire
            SN(nn.Conv2d(feature_maps * 16, 1, 4, 1, 0, bias=False)),
        )

    def forward(self, img):
        return self.net(img).view(-1)
