import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CamVidDataset(Dataset):

    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Bildernamen und Pfad
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)

        # Maskennamen und Pfad
        mask_name = image_name.replace(".png", "_L.png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Konvertiere Bild von BGR zu RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Konvertiere Label von BGR zu Graustufen
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # In Tensoren umwandeln, Format umwandeln (Height, Width, Channels) -> (Channels, Height, Width)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

def get_dataloader(images_dir, masks_dir, batch_size=4, shuffle=True):
        # Erstellt eine Dataset-Instanz und verpackt ihn in einen PyTorch DataLoader
        dataset = CamVidDataset(images_dir, masks_dir)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataLoader
