import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Farb-Mapping zur Konvertierung von RGB-Masken in Klassen-IDs (CamVid Dataset)
CAMVID_COLORS = {
    (64, 128, 64): 0,  # Animal
    (192, 0, 128): 1,  # Archway
    (0, 128, 192): 2,  # Bicyclist
    (0, 128, 64): 3,  # Bridge
    (128, 0, 0): 4,  # Building
    (64, 0, 128): 5,  # Car
    (64, 0, 192): 6,  # CartLuggagePram
    (192, 128, 64): 7,  # Child
    (192, 192, 128): 8,  # Column_Pole
    (64, 64, 128): 9,  # Fence
    (128, 0, 192): 10,  # LaneMkgsDriv
    (192, 0, 64): 11,  # LaneMkgsNonDriv
    (128, 128, 64): 12,  # Misc_Text
    (192, 0, 192): 13,  # MotorcycleScooter
    (128, 64, 64): 14,  # OtherMoving
    (64, 192, 128): 15,  # ParkingBlock
    (64, 64, 0): 16,  # Pedestrian
    (128, 64, 128): 17,  # Road
    (128, 128, 192): 18,  # RoadShoulder
    (0, 0, 192): 19,  # Sidewalk
    (192, 128, 128): 20,  # SignSymbol
    (128, 128, 128): 21,  # Sky
    (64, 128, 192): 22,  # SUVPickupTruck
    (0, 0, 64): 23,  # TrafficCone
    (0, 64, 64): 24,  # TrafficLight
    (192, 64, 128): 25,  # Train
    (128, 128, 0): 26,  # Tree
    (192, 128, 192): 27,  # Truck_Bus
    (64, 0, 64): 28,  # Tunnel
    (192, 192, 0): 29,  # VegetationMisc
    (0, 0, 0): 30,  # Void
    (64, 192, 0): 31,  # Wall
}


class CamVidDataset(Dataset):
    """PyTorch Dataset für das Laden und Vorverarbeiten von CamVid-Bildern und

    zugehörigen Segmentierungsmasken.
    """

    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Dateipfade ermitteln
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)

        mask_name = image_name.replace(".png", "_L.png")
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Bilder mit OpenCV einlesen und Farbraum anpassen (BGR zu RGB)
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        target_size = (256, 256)

        # Bildgröße anpassen (Bilder bilinear, Maskennearest-neighbor, um Artefakte zu vermeiden)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # RGB-Maske in ein 2D-Array mit Klassen-IDs umwandeln
        label_mask = np.zeros(
            (target_size[1], target_size[0]), dtype=np.int64
        )
        for rgb_color, class_id in CAMVID_COLORS.items():
            matches = np.all(mask == rgb_color, axis=-1)
            label_mask[matches] = class_id

        # Normalisierung und Konvertierung in PyTorch Tensoren
        # Bild-Tensor: Permutation zu (Channels, Height, Width) und Skalierung auf [0.0, 1.0]
        image_tensor = (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )
        mask_tensor = torch.tensor(label_mask, dtype=torch.long)

        return image_tensor, mask_tensor


def get_dataloader(images_dir, masks_dir, batch_size=4, shuffle=True):
    """Hilfsfunktion zur Erstellung eines DataLoaders."""
    dataset = CamVidDataset(images_dir, masks_dir)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader