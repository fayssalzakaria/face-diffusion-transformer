"""
Génère des visages avec le DiT entraîné, en fixant les attributs voulus.
Les attributs non cités restent "non spécifiés" et le modèle les choisit librement.

Exemples :
  python sample.py --attrs souriant=1 lunettes=1
  python sample.py --attrs homme=0 chapeau=1 --cfg 4 --n 16
"""
import argparse
import torchvision.utils as vutils
from pipeline import FaceGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit256/last.pt")
    parser.add_argument("--attrs", nargs="*", default=[])
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="sample.png")
    args = parser.parse_args()

    gen = FaceGenerator(args.ckpt, args.device)
    attrs = {name: int(value) for name, value in (spec.split("=") for spec in args.attrs)}
    images = gen.generate(attrs, n=args.n, cfg=args.cfg, steps=args.steps, seed=args.seed)
    vutils.save_image(images, args.out, nrow=int(args.n ** 0.5))
    print(f"Image sauvegardée : {args.out}")


if __name__ == "__main__":
    main()
