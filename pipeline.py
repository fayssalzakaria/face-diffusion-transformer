"""
Charge le DiT entraîné + le VAE et expose les trois usages du modèle :
  - generate    : visages à partir d'attributs
  - edit        : modifier une vraie photo (SDEdit)
  - interpolate : morphing continu entre deux visages générés
Utilisé par sample.py, experiences.py et app.py.
"""
import contextlib
import torch
import torchvision.transforms as transforms
from dit import DiT, CONFIGS, AttributeEmbedder, sample
from vae import load_vae, encode, decode


class FaceGenerator:
    def __init__(self, ckpt_path="runs/dit/last.pt", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt["config"]
        self.model = DiT(input_size=cfg["input_size"], num_attributes=cfg["num_attributes"],
                         patch_size=cfg.get("patch_size", 2), **CONFIGS[cfg["model"]])
        self.model.load_state_dict(ckpt["ema"])  # poids EMA : plus stables pour générer
        self.model.to(self.device).eval()
        self.attr_names = ckpt["attr_names"]
        self.mean = ckpt["mean"].view(1, -1, 1, 1).to(self.device)
        self.std = ckpt["std"].view(1, -1, 1, 1).to(self.device)
        self.vae = load_vae(self.device)
        self.img_size = cfg["input_size"] * 8  # le VAE réduit la taille par 8 (16 → 128, 32 → 256)
        self.to_tensor = transforms.Compose([transforms.Resize(self.img_size), transforms.CenterCrop(self.img_size),
                                             transforms.ToTensor()])

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def attrs_tensor(self, attrs, n):
        """
        attrs : dict {nom: 0 ou 1}, ou tenseur (n, K) déjà prêt.
        Les attributs absents du dict restent "non spécifiés".
        """
        if torch.is_tensor(attrs):
            return attrs.to(self.device)
        t = torch.full((n, len(self.attr_names)), AttributeEmbedder.UNKNOWN, dtype=torch.long, device=self.device)
        for name, value in (attrs or {}).items():
            if name not in self.attr_names:
                raise ValueError(f"Attribut inconnu '{name}'. Disponibles : {self.attr_names}")
            t[:, self.attr_names.index(name)] = int(value)
        return t

    def noise(self, n, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(n, self.model.in_channels, self.model.input_size, self.model.input_size,
                           device=self.device, generator=g)

    # Le VAE travaille par petits paquets : en 256x256, décoder 100 images d'un coup dépasse les 12 Go du GPU
    VAE_CHUNK = 16

    @torch.no_grad()
    def to_latent(self, images):
        """images [0, 1] → latent normalisé (espace du DiT)."""
        with self.autocast():
            z = torch.cat([encode(self.vae, chunk.to(self.device) * 2 - 1).float()
                           for chunk in images.split(self.VAE_CHUNK)])
        return (z - self.mean) / self.std

    @torch.no_grad()
    def to_images(self, z):
        """latent normalisé → images [0, 1]."""
        z = z.float() * self.std + self.mean
        with self.autocast():
            return torch.cat([decode(self.vae, chunk).float() for chunk in z.split(self.VAE_CHUNK)])

    @torch.no_grad()
    def generate(self, attrs=None, n=16, cfg=3.0, steps=50, seed=0, x=None, solver="euler",
                 guidance_interval=(0.0, 1.0)):
        a = self.attrs_tensor(attrs, n)
        x = self.noise(len(a), seed) if x is None else x
        with self.autocast():
            z = sample(self.model, a, steps=steps, cfg_scale=cfg, x=x, solver=solver,
                       guidance_interval=guidance_interval)
        return self.to_images(z)

    @torch.no_grad()
    def edit(self, image, attrs, strength=0.6, cfg=3.0, steps=50, seed=0):
        """
        SDEdit : on bruite la photo jusqu'au niveau `strength` (0 = inchangée, 1 = bruit pur),
        puis on la débruite avec les nouveaux attributs. Plus `strength` est grand,
        plus la modification est forte, mais moins la personne reste ressemblante.
        """
        x0 = self.to_latent(self.to_tensor(image.convert("RGB")).unsqueeze(0))
        noise = self.noise(1, seed)
        xt = (1 - strength) * x0 + strength * noise  # même mélange qu'à l'entraînement
        a = self.attrs_tensor(attrs, 1)
        n_steps = max(1, round(steps * strength))  # on ne parcourt que la fin de la trajectoire
        with self.autocast():
            z = sample(self.model, a, steps=n_steps, cfg_scale=cfg, x=xt, t_start=strength)
        return self.to_images(z)

    @torch.no_grad()
    def interpolate(self, seed_a, seed_b, attrs=None, frames=16, cfg=3.0, steps=50):
        """Morphing : interpolation sphérique (slerp) entre deux bruits de départ."""
        a, b = self.noise(1, seed_a), self.noise(1, seed_b)
        omega = torch.acos((a * b).sum() / (a.norm() * b.norm()))
        alphas = torch.linspace(0, 1, frames, device=self.device).view(-1, 1, 1, 1)
        x = (torch.sin((1 - alphas) * omega) * a + torch.sin(alphas * omega) * b) / torch.sin(omega)
        return self.generate(attrs, n=frames, cfg=cfg, steps=steps, x=x)

    @torch.no_grad()
    def reconstruct(self, images):
        """Passage encode → decode du VAE seul (sert de plafond de qualité pour le FID)."""
        return self.to_images(self.to_latent(images))
