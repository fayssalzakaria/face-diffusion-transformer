import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Diffusion Transformer (DiT, Peebles & Xie 2023) en espace latent
# Entrée : latents VAE (4, 16, 16) → patchs 2x2 → 64 tokens
# Conditionnement : pas de temps + attributs du visage via adaLN-Zero


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Encode t ∈ [0, 1] en vecteur (embedding sinusoïdal + MLP)."""

    def __init__(self, hidden_size, freq_size=256):
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t):
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = (t.float() * 1000)[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


class AttributeEmbedder(nn.Module):
    """
    Un embedding par attribut avec 3 états : 0 = absent, 1 = présent, 2 = non spécifié.
    L'état "non spécifié" sert au classifier-free guidance : on peut conditionner
    sur un sous-ensemble d'attributs seulement (ex. juste "souriant").
    """

    UNKNOWN = 2

    def __init__(self, num_attributes, hidden_size):
        super().__init__()
        self.num_attributes = num_attributes
        self.tables = nn.Parameter(torch.randn(num_attributes, 3, hidden_size) * 0.02)

    def forward(self, attrs):
        # attrs : (B, K) entiers dans {0, 1, 2}
        idx = torch.arange(self.num_attributes, device=attrs.device)
        return self.tables[idx[None, :], attrs].sum(dim=1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # adaLN-Zero : 6 vecteurs (shift, scale, gate) × (attention, MLP)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def attention(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])  # Flash attention si dispo
        return self.proj(out.transpose(1, 2).reshape(B, N, C))

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate_a.unsqueeze(1) * self.attention(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_m.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, input_size=16, patch_size=2, in_channels=4, hidden_size=768,
                 depth=12, num_heads=12, num_attributes=8):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (input_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, patch_size)
        self.pos_embed = nn.Parameter(self._sincos_pos_embed(hidden_size), requires_grad=False)
        self.t_embed = TimestepEmbedder(hidden_size)
        self.attr_embed = AttributeEmbedder(num_attributes, hidden_size)
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final = FinalLayer(hidden_size, patch_size, in_channels)
        self._init_weights()

    def _sincos_pos_embed(self, dim):
        grid = self.input_size // self.patch_size
        coords = torch.arange(grid, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
        out = []
        for pos in (yy.flatten(), xx.flatten()):
            args = pos[:, None] * omega[None]
            out += [torch.sin(args), torch.cos(args)]
        return torch.cat(out, dim=1).unsqueeze(0)

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        # adaLN-Zero : chaque bloc démarre comme l'identité
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight)
        nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def unpatchify(self, x):
        p, g, c = self.patch_size, self.input_size // self.patch_size, self.in_channels
        x = x.view(x.shape[0], g, g, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, g * p, g * p)

    def forward(self, x, t, attrs):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        c = self.t_embed(t) + self.attr_embed(attrs)
        for block in self.blocks:
            x = block(x, c)
        return self.unpatchify(self.final(x, c))


# Tailles standard du papier DiT
CONFIGS = {
    "S": dict(hidden_size=384, depth=12, num_heads=6),
    "B": dict(hidden_size=768, depth=12, num_heads=12),
    "L": dict(hidden_size=1024, depth=24, num_heads=16),
}


@torch.no_grad()
def sample(model, attrs, steps=50, cfg_scale=3.0, generator=None, x=None, t_start=1.0,
           solver="euler", guidance_interval=(0.0, 1.0)):
    """
    Échantillonnage flow matching : intègre l'ODE de t=t_start à t=0 (données).
    Classifier-free guidance : v = v_uncond + w · (v_cond − v_uncond).

    solver : "euler" (1 évaluation par pas) ou "heun" (2 évaluations par pas : on prédit
             le pas, puis on le corrige avec la vitesse au point d'arrivée → moins d'erreurs
             géométriques à nombre d'évaluations égal).
    guidance_interval : (t_min, t_max), le guidance n'est appliqué que pour t dans cet intervalle
             (Kynkäänniemi et al., 2024). Au début (t≈1) il réduit la diversité, à la fin (t≈0)
             il abîme les détails fins ; ailleurs on utilise la prédiction conditionnelle seule.

    Par défaut on part d'un bruit pur (t_start=1). Pour modifier une image existante
    (SDEdit), on passe son latent partiellement bruité dans `x` avec t_start < 1.
    """
    B = attrs.shape[0]
    device = attrs.device
    if x is None:
        shape = (B, model.in_channels, model.input_size, model.input_size)
        x = torch.randn(shape, device=device, generator=generator)
    uncond = torch.full_like(attrs, AttributeEmbedder.UNKNOWN)
    t_min, t_max = guidance_interval

    def velocity(x, t_scalar):
        t = t_scalar.expand(B)
        w = cfg_scale if t_min <= t_scalar.item() <= t_max else 1.0
        if w == 1.0:
            return model(x, t, attrs)
        v_c, v_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([attrs, uncond])).chunk(2)
        return v_u + w * (v_c - v_u)

    ts = torch.linspace(t_start, 0.0, steps + 1, device=device)
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        v = velocity(x, ts[i])
        if solver == "heun":
            v_next = velocity(x + dt * v, ts[i + 1])
            v = (v + v_next) / 2
        x = x + dt * v
    return x
