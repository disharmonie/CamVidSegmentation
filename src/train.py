import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import UNet
from src.dataset import CamVidDataset

def main():
    # 1. Hardware prüfen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training auf: {device}")

    # 2. Den Dataloader aufbauen
    train_dataset = CamVidDataset(images_dir="data/train", 
                                  masks_dir="data/train_labels")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # 3. Den Validierungs-Datensatz laden
    val_dataset = CamVidDataset(images_dir="data/val", 
                                masks_dir="data/val_labels")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # 4. Das Modell aufbauen
    model = UNet(in_channels=3, out_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Die Trainingsschleife
    epochs = 50
    best_val_loss = float('inf')

    for epoch in range(epochs):

        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoche [{epoch+1}/{epochs}] Train")
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDIERUNG ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoche {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val-Loss: {avg_val_loss:.4f}")


        # --- Checkpoint speichern ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_pfad = "checkpoints/best_model.pth"
            torch.save(model.state_dict(), checkpoint_pfad)
            print("Best model saved!")

        # Zum Laden:
        # geladenes_modell = UNet(in_channels=3, out_channels=32)
        # geladenes_modell.load_state_dict(torch.load("experiments/checkpoints/best_model.pth"))
        # geladenes_modell.eval() # Direkt in den Prüfungsmodus versetzen!


if __name__ == "__main__":
    main()