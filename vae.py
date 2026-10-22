import torch
from diffusers import AutoencoderKL

# VAE pré-entraîné de Stable Diffusion, utilisé figé : divise la taille par 8 (256×256×3 → 32×32×4)
VAE_ID = "stabilityai/sd-vae-ft-ema"

# Attributs annotés par CLIP ViT-L/14 en zero-shot : (nom, prompts positifs, prompts négatifs).
# Les négatifs décrivent une alternative plutôt qu'une négation (« sans chapeau ») : CLIP comprend mal la négation.
# Annotations vérifiées à l'œil ; la couleur des cheveux a été retirée car trop peu fiable.
ATTRIBUTES = [
    ("homme", ["a photo of a man"], ["a photo of a woman"]),
    ("souriant", ["a photo of a person smiling", "a photo of a person laughing"],
                 ["a photo of a person with a serious face", "a photo of a person with a neutral face"]),
    ("lunettes", ["a photo of a person wearing glasses", "a photo of a person wearing sunglasses"],
                 ["a close-up photo of a face", "a photo of a person"]),
    ("barbe", ["a photo of a man with a beard", "a photo of a man with a mustache and beard"],
              ["a photo of a man with a shaved face", "a photo of a woman", "a photo of a child"]),
    ("age", ["a photo of an elderly person", "a photo of an old person with wrinkles"],
            ["a photo of a young adult", "a photo of a child", "a photo of a middle-aged person"]),
    ("chapeau", ["a photo of a person wearing a hat", "a photo of a person wearing a cap",
                 "a photo of a person wearing a beanie"],
                ["a photo of a person", "a photo of a person showing their hair"]),
    ("enfant", ["a photo of a child", "a photo of a baby"], ["a photo of an adult", "a photo of a teenager"]),
]
ATTRIBUTE_NAMES = [a[0] for a in ATTRIBUTES]


def load_vae(device):
    vae = AutoencoderKL.from_pretrained(VAE_ID).to(device).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae, images):
    """images dans [-1, 1] → latents bruts (non normalisés)."""
    return vae.encode(images).latent_dist.sample()


@torch.no_grad()
def decode(vae, latents):
    """latents bruts → images dans [0, 1]."""
    images = vae.decode(latents).sample
    return ((images + 1) / 2).clamp(0, 1)
