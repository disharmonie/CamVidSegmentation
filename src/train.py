import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CamVidDataset
from src.model import UNet


class AverageMeter:
    """Verwaltet und berechnet den laufenden Durchschnitt und die Summe

    einer Metrik (z. B. Loss), nützlich für das Tracking während des Trainings.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Setzt alle internen Zähler auf Null zurück."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Aktualisiert den Messwert mit einem neuen Wert und der Batch-Größe."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    # Hardware-Beschleunigung konfigurieren
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf: {device}\n" + "-" * 50)

    # Speicherort für Modell-Checkpoints erstellen
    os.makedirs("checkpoints", exist_ok=True)

    # Datensätze und DataLoader für Training und Validierung initialisieren
    train_dataset = CamVidDataset(
        images_dir="data/train", masks_dir="data/train_labels"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0
    )

    val_dataset = CamVidDataset(images_dir="data/val", masks_dir="data/val_labels")
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    # Modell, Verlustfunktion (Loss) und Optimizer definieren
    model = UNet(in_channels=3, out_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    best_val_loss = float("inf")

    # Trainingsschleife über alle Epochen
    for epoch in range(epochs):

        # --- TRAINING ---
        model.train()
        train_losses = AverageMeter()

        train_loop = tqdm(
            train_loader,
            desc=f"Epoche [{epoch+1:02d}/{epochs}] Train",
            leave=False,
        )

        for images, masks in train_loop:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            # Gradienten zurücksetzen, Vorwärtspass, Loss berechnen und optimieren
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), images.size(0))
            train_loop.set_postfix(
                loss=f"{train_losses.val:.4f}", avg=f"{train_losses.avg:.4f}"
            )

        # --- VALIDIERUNG ---
        model.eval()
        val_losses = AverageMeter()

        val_loop = tqdm(
            val_loader, desc=f"Epoche [{epoch+1:02d}/{epochs}] Val  ", leave=False
        )

        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_losses.update(loss.item(), images.size(0))
                val_loop.set_postfix(
                    loss=f"{val_losses.val:.4f}", avg=f"{val_losses.avg:.4f}"
                )

        # --- CHECKPOINT & TERMINAL AUSGABE ---
        save_msg = ""
        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            checkpoint_pfad = "checkpoints/best_model.pth"
            torch.save(model.state_dict(), checkpoint_pfad)
            save_msg = " --> [Neues bestes Modell gespeichert!]"

        print(
            f"Epoche {epoch+1:02d}/{epochs} | Train Loss: {train_losses.avg:.4f} | Val-Loss: {val_losses.avg:.4f}{save_msg}"
        )

    print("Training finished!")


if __name__ == "__main__":
    main()