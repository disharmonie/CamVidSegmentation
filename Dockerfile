FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

# Arbeitsverzeichnis im Container
WORKDIR /workspace

# OpenCV benötigt einige zusätzliche Bibliotheken, die wir hier installieren
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Kopiere die requirements.txt und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Der restliche Code wird über docker-compose gemounted
# damit nicht bei jeder Code-Änderung neu gebaut werden muss.
