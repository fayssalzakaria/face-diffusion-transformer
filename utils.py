import kagglehub
import os
import shutil
 
print("Téléchargement de FFHQ...")
path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
print(f"Dataset téléchargé dans : {path}")
 
# Copie les images dans data/ffhq/
os.makedirs("data/ffhq", exist_ok=True)
print("Copie des images dans data/ffhq/...")
 
copied = 0
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith((".png", ".jpg", ".jpeg")):
            shutil.copy2(os.path.join(root, f), "data/ffhq/")
            copied += 1
            if copied % 1000 == 0:
                print(f"{copied} images copiées...")
 
print(f"Terminé ! {copied} images dans data/ffhq/")
 