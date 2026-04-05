import os  
from torch.utils.data import Dataset  
from PIL import Image  

class CustomDataset(Dataset):
    """
    Dataset personnalisé pour charger des images à partir d'un dossier.
    """

    def __init__(self, root, transform=None, max_images=None):
        """
        Initialise le dataset.

        :param root: Chemin du dossier contenant les images.
        :param transform: Transformations PyTorch à appliquer aux images (par défaut, aucune transformation).
        :param max_images: Nombre maximum d'images à charger (None = toutes).
        """
        self.root = root  # Stocke le chemin du dossier contenant les images
        self.transform = transform  # Stocke les transformations éventuelles
        # Liste tous les fichiers du dossier et ne garde que ceux avec une extension image valide
        self.images = [os.path.join(root, img) for img in os.listdir(root) 
                       if img.endswith(("jpg", "jpeg", "png"))]
        # Limite le nombre d'images si max_images est spécifié
        if max_images:
            self.images = self.images[:max_images]

    def __len__(self):
        """
        Retourne le nombre total d'images dans le dataset.

        :return: Nombre d'images.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Charge une image à l'index donné.

        :param idx: Index de l'image à charger.
        :return: L'image transformée et un label fictif (0).
        """
        img_path = self.images[idx]  # Récupère le chemin de l'image à l'index donné
        image = Image.open(img_path).convert("RGB")  # Ouvre l'image et la convertit en RGB
        image.load()
        if self.transform:
            image = self.transform(image)  # Applique les transformations si spécifiées
        return image, 0  # Retourne l'image et un label fictif (ici, toujours 0)