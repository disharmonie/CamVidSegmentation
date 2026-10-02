import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CamVidDataset
from model import UNet

def calculate_intersections_and_unions(pred_mask, true_mask, num_classes=32):
    """Berechnet die Intersection und Union pro Klasse für den globalen mIoU"""
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()

    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)

    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        target_inds = (true_mask == cls)
        
        intersection = (pred_inds & target_inds).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        
        intersections[cls] = intersection
        unions[cls] = union
            
    correct = (pred_mask == true_mask).sum()
    total = true_mask.size
    
    return intersections, unions, correct, total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Auswertung auf: {device}\n" + "-" * 50)

    val_dataset = CamVidDataset(images_dir="data/val", masks_dir="data/val_labels")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("Lade trainiertes Modell...")
    model = UNet(in_channels=3, out_channels=32, features=[16, 32, 64]).to(device)
    
    # Pfad anpassen für Student oder Baseline
    checkpoint_path = "checkpoints/best_baseline_model.pth" 
    
    if not os.path.exists(checkpoint_path):
        print(f"Fehler: Checkpoint '{checkpoint_path}' nicht gefunden!")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    total_intersections = np.zeros(32)
    total_unions = np.zeros(32)
    total_correct = 0
    total_pixels = 0
    
    print("Berechne globale Metriken über das gesamte Validierungsset...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluiere"):
            images = images.to(device)
            masks = masks.to(device).cpu().numpy()

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            intersections, unions, correct, total = calculate_intersections_and_unions(preds, masks)
            total_intersections += intersections
            total_unions += unions
            total_correct += correct
            total_pixels += total

    print("\n--- KLASSENSPEZIFISCHE IoU ---")
    class_ious = []
    for cls in range(32):
        if total_unions[cls] > 0:
            iou = total_intersections[cls] / total_unions[cls]
            class_ious.append(iou)
            print(f"Klasse {cls:02d}: {iou * 100:>5.2f} %")
        else:
            print(f"Klasse {cls:02d}:   N/A (Nicht im Val-Set vorhanden)")

    global_miou = np.mean(class_ious)
    global_acc = total_correct / total_pixels

    print("\n--- GESAMTERGEBNISSE ---")
    print(f"Globale Pixel Accuracy: {global_acc * 100:.2f} %")
    print(f"Globaler mIoU:          {global_miou * 100:.2f} %\n")

    print("Generiere Beispiel-Plot...")
    sample_img, sample_mask = val_dataset[0]
    img_tensor = sample_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_logits = model(img_tensor)
        pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

    img_vis = sample_img.numpy().transpose(1, 2, 0)
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_vis)
    axes[0].set_title("Originalbild")
    axes[0].axis('off')
    
    axes[1].imshow(sample_mask.numpy(), cmap='tab20', vmin=0, vmax=31)
    axes[1].set_title("Ground Truth (Echte Maske)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=31)
    axes[2].set_title("Vorhersage")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("segmentation_result_plot.png")
    print("Plot gespeichert als 'baseline_result_plot.png'.")

if __name__ == "__main__":
    main()