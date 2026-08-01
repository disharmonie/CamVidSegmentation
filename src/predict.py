import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model import UNet


def predict_and_show(image_path, model_path, output_path="ergebnis.png"):
    """Lädt ein trainiertes U-Net Modell, führt eine Segmentierungs-Vorhersage

    für ein einzelnes Bild durch und speichert das Ergebnis als Visualisierung.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Gerät: {device}")

    # Modell initialisieren und trainierte Gewichte laden
    model = UNet(in_channels=3, out_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Eingabebild einlesen
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Bild für das Modell vorverarbeiten (Resize und Normalisierung)
    target_size = (256, 256)
    image_resized = cv2.resize(
        image_rgb, target_size, interpolation=cv2.INTER_LINEAR
    )

    image_tensor = (
        torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
    )
    image_tensor = image_tensor.unsqueeze(0).to(device)  

    # Inferenz (Vorhersage ohne Gradientenberechnung)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1)  # Klasse mit höchster Wahrscheinlichkeit wählen

    prediction_np = prediction.squeeze(0).cpu().numpy()

    # Ergebnisse plotten und speichern
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Eingabebild")
    plt.imshow(image_resized)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Modell-Vorhersage (Maske)")
    plt.imshow(prediction_np, cmap="nipy_spectral", vmin=0, vmax=31)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Erfolgreich gespeichert unter: {output_path}")


if __name__ == "__main__":
    TEST_BILD = "data/val/florian_straße.jpg"
    CHECKPOINT = "checkpoints/best_model.pth"

    predict_and_show(TEST_BILD, CHECKPOINT)