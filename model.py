import math
import torch
import torch.nn as nn

# WGAN-GP : la résolution (128 ou 256) fixe le nombre de couches.
# Chaque couche double (générateur) ou divise par deux (critique) la taille de l'image.
# En 128x128, l'architecture est identique à la version d'origine du projet ;
# en 256x256, une couche supplémentaire est ajoutée de chaque côté (feature_maps // 2 canaux).


def _channels(feature_maps, img_size):
    """Canaux à chaque résolution, de 4x4 jusqu'à img_size/2 : fm*16, fm*8, ..., fm, (fm/2)."""
    n = int(math.log2(img_size // 4))  # nombre de doublements de 4x4 à img_size
    return [max(feature_maps * 16 // 2 ** i, feature_maps // 2) for i in range(n)]


# Générateur WGAN-GP
class Generator(nn.Module):
    def __init__(self, z_dim, channels, feature_maps, img_size=128):
        super(Generator, self).__init__()
        ch = _channels(feature_maps, img_size)
        layers = [
            # (z_dim, 1, 1) → 4x4
            nn.ConvTranspose2d(z_dim, ch[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(True),
            nn.Dropout2d(0.3),  # réduit les artefacts de couleur
        ]
        for i in range(1, len(ch)):
            # résolution × 2
            layers += [nn.ConvTranspose2d(ch[i - 1], ch[i], 4, 2, 1, bias=False), nn.BatchNorm2d(ch[i]), nn.ReLU(True)]
            if i == 1:
                layers.append(nn.Dropout2d(0.3))  # réduit le surapprentissage
        # dernière couche → image RGB img_size x img_size
        layers += [nn.ConvTranspose2d(ch[-1], channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


# Critique WGAN-GP (pas de Sigmoid, InstanceNorm au lieu de BatchNorm : compatible avec le gradient penalty)
# Le critique prédit un score réel (pas une probabilité)
class Critic(nn.Module):
    def __init__(self, channels, feature_maps, img_size=128):
        super(Critic, self).__init__()
        SN = nn.utils.spectral_norm
        ch = _channels(feature_maps, img_size)[::-1]  # de img_size/2 jusqu'à 4x4
        # img_size → img_size/2 (pas de normalisation sur la première couche)
        layers = [SN(nn.Conv2d(channels, ch[0], 4, 2, 1, bias=False)), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(1, len(ch)):
            # résolution / 2
            layers += [SN(nn.Conv2d(ch[i - 1], ch[i], 4, 2, 1, bias=False)),
                       nn.InstanceNorm2d(ch[i], affine=True), nn.LeakyReLU(0.2, inplace=True)]
        # 4x4 → scalaire
        layers.append(SN(nn.Conv2d(ch[-1], 1, 4, 1, 0, bias=False)))
        self.net = nn.Sequential(*layers)

    def forward(self, img):
        return self.net(img).view(-1)
