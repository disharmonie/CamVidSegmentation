# CamVidSegmentation

Dieses Repository enthält eine PyTorch-Implementierung eines U-Net-Modells zur semantischen Segmentierung von Straßenszenen. Das Modell wurde auf dem CamVid-Datensatz trainiert und klassifiziert Bildpixel präzise in 32 verschiedene Kategorien (u. a. Autos, Himmel, Gebäude, Straße, Fußgänger).

Dieses Projekt entsteht im Rahmen eines Semesterpraktikums an meiner Universität im Studiengang Informatik.

## 📂 Projektstruktur

- `src/model.py`: Definition der U-Net Architektur (3 Eingabekanäle, 32 Ausgabekanäle).
- `src/dataset.py`: PyTorch `Dataset` und `DataLoader`. Beinhaltet das Preprocessing und das Color-Mapping (RGB zu Klassen-IDs) für die 32 CamVid-Klassen.
- `src/train.py`: Trainingsschleife, Loss-Berechnung (Cross-Entropy) und automatische Speicherung der besten Gewichte.
- `src/predict.py`: Skript zur Anwendung des trainierten Modells auf ungesehene Testbilder inklusive Farb-Visualisierung der Segmentierungsmasken.
- `checkpoints/`: Speicherort für trainierte Modellgewichte (z. B. `best_model.pth`).
- `data/`: Ordnerstruktur für den CamVid-Datensatz (`train`, `val`, `test` und die dazugehörigen Masken).
## 🚀 Ergebnisse

Nach Behebung anfänglicher Class-Imbalance-Probleme im Data-Loading erreicht das aktuelle "Vanilla" U-Net Modell einen soliden **Validation Loss von 0.3879**. 
Das Modell zeigt auf den Testdaten eine starke Generalisierung bei der Erkennung grundlegender Straßengeometrien, verdeutlicht jedoch auch klassische Phänomene des *Domain Shifts*, wenn es auf moderne, ungesehene Straßenszenen außerhalb der britischen CamVid-Domäne angewendet wird.

## 🛠️ Nutzung

### 1. Training starten
Das Modell kann von Grund auf neu trainiert werden. Der beste Checkpoint wird automatisch unter `checkpoints/best_model.pth` gespeichert, sobald sich der Val-Loss verbessert.
```bash
python -m src.train