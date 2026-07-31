import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    # Ein wiederverwendbarer Block: Zwei Faltungsschichten (Convolutions) hintereinander. Jede Faltung wird von einer Batch-Normalisierung und einer ReLU-Aktivierung gefolgt.
       
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            # Erste Faltung: padding=1 sorgt dafür, dass sich die Bildhöhe/breite hier nicht ändert
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Zweite Faltung
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # in_channels=3, weil wir RGB-Bilder haben.
        # out_channels=32, weil CamVid 32 verschiedene Klassen hat.

        # --- ENCODER (Der Weg nach unten) ---
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        
        # MaxPool halbiert die Auflösung des Bildes
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK (Der tiefste Punkt des U) ---
        self.bottleneck = DoubleConv(256, 512)

        # --- DECODER (Der Weg nach oben) ---
        # ConvTranspose2d verdoppelt die Auflösung wieder (Upsampling)
        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(512, 256) # 512, weil wir später die Skip-Connection ankleben!

        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)

        # --- OUTPUT LAYER ---
        # Eine letzte Faltung, um von 64 auf unsere 32 Klassen zu kommen. 
        # kernel_size=1 verändert die Auflösung nicht mehr.
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. ENCODER-Phase
        x1 = self.down1(x)         # Herausziehen der ersten Merkmale
        p1 = self.pool(x1)         # Bild halbieren

        x2 = self.down2(p1)
        p2 = self.pool(x2)

        x3 = self.down3(p2)
        p3 = self.pool(x3)

        # 2. BOTTLENECK
        b = self.bottleneck(p3)

        # 3. DECODER-Phase mit SKIP-CONNECTIONS
        up1 = self.up_trans1(b)                # Bild wieder vergrößern
        concat1 = torch.cat([up1, x3], dim=1)  # SKIP-CONNECTION: Wir kleben die alten Details (x3) an!
        dec1 = self.up_conv1(concat1)          # Faltung über die zusammengeklebten Daten

        up2 = self.up_trans2(dec1)
        concat2 = torch.cat([up2, x2], dim=1)  # SKIP-CONNECTION mit x2
        dec2 = self.up_conv2(concat2)

        up3 = self.up_trans3(dec2)
        concat3 = torch.cat([up3, x1], dim=1)  # SKIP-CONNECTION mit x1
        dec3 = self.up_conv3(concat3)

        # 4. OUTPUT
        out = self.out_conv(dec3)
        return out