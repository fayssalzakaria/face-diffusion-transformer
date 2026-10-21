"""
Pré-calcule une fois pour toutes, sur tout FFHQ :
  1. les latents VAE (image 128x128 et sa version miroir) ;
  2. les attributs de chaque visage, annotés en zero-shot par CLIP ViT-L/14.
Le DiT s'entraîne ensuite uniquement sur ces latents (≈200 Mo en mémoire GPU),
sans relire les images ni repasser par le VAE : l'entraînement est bien plus rapide.

Usage : python prepare_latents.py --data_dir data/ffhq
"""
import argparse
import os
import torch
import torchvision.transforms as transforms
import open_clip
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CustomDataset
from vae import ATTRIBUTES, load_vae, encode

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class TwoViews:
    """Renvoie l'image 128x128 (pour le VAE) et l'image 224x224 (pour CLIP)."""

    def __init__(self, img_size):
        self.small = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size),
                                         transforms.ToTensor()])
        self.clip = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                        transforms.ToTensor()])

    def __call__(self, image):
        return self.small(image), self.clip(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/ffhq")
    parser.add_argument("--out", default="data/latents_ffhq128.pt")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CustomDataset(args.data_dir, transform=TwoViews(args.img_size), max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(os.cpu_count() or 1, 8), pin_memory=True)
    print(f"{len(dataset)} images trouvées dans {args.data_dir}")

    vae = load_vae(device)
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    with torch.no_grad():
        # Par attribut : embeddings des prompts positifs puis négatifs, et nombre de positifs
        text_feats = []
        for _, pos, neg in ATTRIBUTES:
            t = clip_model.encode_text(tokenizer(pos + neg).to(device)).float()
            text_feats.append((t / t.norm(dim=-1, keepdim=True), len(pos)))

    latents, latents_flip, probs = [], [], []
    with torch.no_grad():
        for (small, big), _ in tqdm(loader):
            small = small.to(device, non_blocking=True) * 2 - 1
            with torch.autocast(device.type, dtype=torch.bfloat16):
                latents.append(encode(vae, small).float().half().cpu())
                latents_flip.append(encode(vae, small.flip(-1)).float().half().cpu())

                big = (big.to(device) - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
                img_feats = clip_model.encode_image(big).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            # P(attribut) = masse de probabilité des prompts positifs (softmax, température CLIP = 100)
            probs.append(torch.stack([(100 * img_feats @ t.T).softmax(dim=-1)[:, :n_pos].sum(dim=-1)
                                      for t, n_pos in text_feats], dim=1).cpu())

    latents = torch.cat(latents)
    probs = torch.cat(probs)
    # Statistiques par canal pour normaliser les latents à moyenne 0, variance 1
    mean = latents.float().mean(dim=(0, 2, 3))
    std = latents.float().std(dim=(0, 2, 3))
    torch.save({
        "latents": latents,
        "latents_flip": torch.cat(latents_flip),
        "attr_probs": probs,
        "attr_names": [a[0] for a in ATTRIBUTES],
        "mean": mean,
        "std": std,
        "img_size": args.img_size,
        "files": dataset.images,
    }, args.out)

    print(f"Latents sauvegardés dans {args.out} : {tuple(latents.shape)}")
    print("Proportion de visages par attribut (seuil 0.5) :")
    for name, frac in zip([a[0] for a in ATTRIBUTES], (probs > 0.5).float().mean(0).tolist()):
        print(f"  {name:10s} {frac:.1%}")


if __name__ == "__main__":
    main()
