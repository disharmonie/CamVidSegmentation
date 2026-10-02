import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp

from dataset import CamVidDataset
from model import UNet

class AverageMeter:
    """Verwaltet und berechnet den laufenden Durchschnitt einer Metrik."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def distillation_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    Kombiniert den harten Verlust (gegenüber CamVid-Daten) und 
    den weichen Verlust (gegenüber dem Teacher).
    """
    # 1. Hard Loss: Standard Cross-Entropy zwischen Student (UNet) und echten Pixel-Masken
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft Loss: KL-Divergenz zwischen aufgeweichten Teacher- und Student-Wahrscheinlichkeiten
    soft_log_probs = F.log_softmax(student_logits / T, dim=1)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    
    # Reduktion auf 'batchmean' ist Standard für KL-Divergenz in PyTorch
    soft_loss = F.kl_div(soft_log_probs, soft_targets, reduction='mean') * (T * T)
    
    # Gewichtete Kombination beider Verluste
    return (1. - alpha) * hard_loss + alpha * soft_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training auf: {device}\n" + "-" * 50)
    os.makedirs("checkpoints", exist_ok=True)

    # DataLoader initialisieren
    # HINWEIS: Skript muss aus dem Hauptverzeichnis via 'python src/train.py' gestartet werden
    train_dataset = CamVidDataset(images_dir="data/train", masks_dir="data/train_labels")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    val_dataset = CamVidDataset(images_dir="data/val", masks_dir="data/val_labels")
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

    # --- 1. TEACHER INITIALISIEREN (ResNet34) ---
    print("Lade Teacher-Modell (ResNet34)...")
    teacher = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=32
    ).to(device)
    
    teacher.eval() # Teacher wird dauerhaft in den Evaluierungsmodus versetzt
    for param in teacher.parameters():
        param.requires_grad = False # Einfrieren der Gewichte, spart enorm viel VRAM

    # --- 2. STUDENT INITIALISIEREN (Mini-UNet) ---
    print("Lade Student-Modell (Mini-UNet)...")
    student = UNet(in_channels=3, out_channels=32, features=[16, 32, 64]).to(device)
    
    # Der Optimizer kümmert sich NUR um die Parameter des Students (Variable korrigiert auf 'student')
    optimizer = optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-5)

    epochs = 50
    best_val_loss = float("inf")

    for epoch in range(epochs):

        # --- TRAINING ---
        student.train()
        train_losses = AverageMeter()

        train_loop = tqdm(train_loader, desc=f"Epoche [{epoch+1:02d}/{epochs}] Train", leave=False)

        for images, masks in train_loop:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            
            # Forward-Pass Teacher (ohne Gradientenberechnung)
            with torch.no_grad():
                teacher_logits = teacher(images)
                
            # Forward-Pass Student
            student_logits = student(images)
            
            # Loss berechnen
            loss = distillation_loss(student_logits, teacher_logits, masks, T=2.0, alpha=0.0)
            
            # Backpropagation für den Student
            loss.backward()
            optimizer.step()

            train_losses.update(loss.item(), images.size(0))
            train_loop.set_postfix(loss=f"{train_losses.val:.4f}", avg=f"{train_losses.avg:.4f}")

        # --- VALIDIERUNG (Wird nur für den Student durchgeführt) ---
        student.eval()
        val_losses = AverageMeter()
        val_loop = tqdm(val_loader, desc=f"Epoche [{epoch+1:02d}/{epochs}] Val  ", leave=False)

        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)

                student_logits = student(images)
                
                # In der Evaluierung betrachten wir nur den harten Cross-Entropy-Loss
                loss = F.cross_entropy(student_logits, masks)

                val_losses.update(loss.item(), images.size(0))
                val_loop.set_postfix(loss=f"{val_losses.val:.4f}", avg=f"{val_losses.avg:.4f}")

        # --- CHECKPOINT ---
        save_msg = ""
        if val_losses.avg < best_val_loss:
            best_val_loss = val_losses.avg
            torch.save(student.state_dict(), "checkpoints/best_baseline_model.pth")
            save_msg = " --> [Neuer bester Student gespeichert!]"

        print(f"Epoche {epoch+1:02d}/{epochs} | Train Loss (KD): {train_losses.avg:.4f} | Val-Loss: {val_losses.avg:.4f}{save_msg}")

if __name__ == "__main__":
    main()