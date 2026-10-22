import os
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    """Images d'un dossier (jpg/png). max_images limite la taille pour des tests rapides."""

    def __init__(self, root, transform=None, max_images=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, img) for img in os.listdir(root)
                       if img.endswith(("jpg", "jpeg", "png"))]
        if max_images:
            self.images = self.images[:max_images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image.load()
        if self.transform:
            image = self.transform(image)
        return image, 0  # label fictif : le DataLoader attend un couple (image, label)
