"""
Interface web de démonstration (Gradio) :
  - Générer      : visages à partir d'attributs, réglage du guidance et du nombre d'étapes
  - Modifier     : changer les attributs d'une vraie photo (SDEdit)
  - Morphing     : transition continue entre deux visages

Usage : python app.py   puis ouvrir http://127.0.0.1:7860
"""
import argparse
import gradio as gr
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from pipeline import FaceGenerator

LABELS = {
    "homme": "Homme", "souriant": "Souriant", "lunettes": "Lunettes", "barbe": "Barbe",
    "age": "Personne âgée", "chapeau": "Chapeau", "enfant": "Enfant",
}
CHOICES = ["Indifférent", "Oui", "Non"]


def build_app(gen):
    def read_attrs(values):
        return {name: (1 if v == "Oui" else 0) for name, v in zip(gen.attr_names, values) if v != "Indifférent"}

    def attribute_controls():
        return [gr.Radio(CHOICES, value="Indifférent", label=LABELS.get(name, name)) for name in gen.attr_names]

    def to_pil_grid(images, nrow):
        return TF.to_pil_image(vutils.make_grid(images.cpu(), nrow=nrow, padding=2, pad_value=1.0))

    def on_generate(n, cfg, steps, seed, *values):
        images = gen.generate(read_attrs(values), n=int(n), cfg=cfg, steps=int(steps), seed=int(seed))
        return to_pil_grid(images, nrow=min(int(n), 4))

    def on_edit(image, strength, cfg, seed, *values):
        if image is None:
            raise gr.Error("Ajoute d'abord une photo.")
        original = TF.to_pil_image(gen.to_tensor(image.convert("RGB")))
        edited = gen.edit(image, read_attrs(values), strength=strength, cfg=cfg, seed=int(seed))
        return original, TF.to_pil_image(edited[0].cpu())

    def on_morph(seed_a, seed_b, frames, cfg, *values):
        images = gen.interpolate(int(seed_a), int(seed_b), read_attrs(values), frames=int(frames), cfg=cfg)
        pil = [TF.to_pil_image(f.cpu()) for f in images]
        pil = pil + pil[::-1]
        path = "morphing_demo.gif"
        pil[0].save(path, save_all=True, append_images=pil[1:], duration=80, loop=0)
        return path, to_pil_grid(images[:: max(1, len(images) // 8)], nrow=8)

    with gr.Blocks(title="Visages DiT") as app:
        gr.Markdown("# Génération de visages — Diffusion Transformer\n"
                    f"DiT-B entraîné sur FFHQ ({gen.img_size}×{gen.img_size}), "
                    "conditionné par 7 attributs annotés par CLIP.")

        with gr.Tab("Générer"):
            with gr.Row():
                with gr.Column(scale=1):
                    attrs = attribute_controls()
                    n = gr.Slider(1, 16, value=8, step=1, label="Nombre d'images")
                    cfg = gr.Slider(1, 7, value=3, step=0.5, label="Guidance (cfg) : force de la consigne")
                    steps = gr.Slider(5, 100, value=50, step=5, label="Étapes de débruitage")
                    seed = gr.Number(value=0, precision=0, label="Graine (même graine = mêmes visages)")
                    btn = gr.Button("Générer", variant="primary")
                out = gr.Image(label="Visages générés", scale=2)
            btn.click(on_generate, [n, cfg, steps, seed, *attrs], out)

        with gr.Tab("Modifier une photo"):
            gr.Markdown("La photo est bruitée puis débruitée avec les attributs choisis (**SDEdit**). "
                        "Intensité faible : la personne reste reconnaissable. Forte : le changement est net "
                        "mais le visage s'éloigne. Fonctionne mieux sur un visage centré et cadré serré.")
            with gr.Row():
                with gr.Column(scale=1):
                    photo = gr.Image(type="pil", label="Photo de départ")
                    e_attrs = attribute_controls()
                    strength = gr.Slider(0.1, 1.0, value=0.6, step=0.05, label="Intensité de la modification")
                    e_cfg = gr.Slider(1, 7, value=3, step=0.5, label="Guidance (cfg)")
                    e_seed = gr.Number(value=0, precision=0, label="Graine")
                    e_btn = gr.Button("Modifier", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        before = gr.Image(label=f"Avant ({gen.img_size}×{gen.img_size})")
                        after = gr.Image(label="Après")
            e_btn.click(on_edit, [photo, strength, e_cfg, e_seed, *e_attrs], [before, after])

        with gr.Tab("Morphing"):
            with gr.Row():
                with gr.Column(scale=1):
                    seed_a = gr.Number(value=1, precision=0, label="Visage de départ (graine)")
                    seed_b = gr.Number(value=2, precision=0, label="Visage d'arrivée (graine)")
                    frames = gr.Slider(8, 48, value=24, step=4, label="Nombre d'images")
                    m_cfg = gr.Slider(1, 7, value=3, step=0.5, label="Guidance (cfg)")
                    m_attrs = attribute_controls()
                    m_btn = gr.Button("Créer le morphing", variant="primary")
                with gr.Column(scale=2):
                    gif = gr.Image(label="Animation")
                    strip = gr.Image(label="Étapes intermédiaires")
            m_btn.click(on_morph, [seed_a, seed_b, frames, m_cfg, *m_attrs], [gif, strip])

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="runs/dit256/last.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true", help="lien public temporaire (72 h) pour une démo")
    args = parser.parse_args()
    build_app(FaceGenerator(args.ckpt, args.device)).launch(share=args.share)
