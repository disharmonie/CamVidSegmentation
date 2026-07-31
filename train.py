import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNet
from dataset import CamVidDataset

def main():
    # 1. Hardware prüfen
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training auf: {device}")

    # 2. Den Dataloader aufbauen
    train_dataset = CamVidDataset(image_dir="data/CamVid/train/images", 
                                  mask_dir="data/CamVid/train/masks")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # 3. Den Validierungs-Datensatz laden
    val_dataset = CamVidDataset(image_dir="data/CamVid/val/images", 
                                mask_dir="data/CamVid/val/masks")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

    # 4. Das Modell aufbauen
    model = UNet(in_channels=3, out_channels=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Die Trainingsschleife
    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoche {epoch+1}/{epochs} - Fehler: {running_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()