import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Wiederverwendbarer Baustein: Zwei aufeinanderfolgende Faltungsschichten,

    jeweils gefolgt von Batch Normalization und ReLU-Aktivierung.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1
            ),  # Erhält die räumliche Dimension
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net Architektur für die semantische Bildsegmentierung."""

    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        # in_channels=3 (RGB-Bild), out_channels=32 (Anzahl der CamVid-Klassen)

        # --- ENCODER (Absteigender Pfad zur Extraktion von Merkmalen) ---
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)

        # MaxPool halbiert jeweils die räumliche Auflösung (Höhe x Breite)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- BOTTLENECK (Tiefster Punkt des Netzwerks) ---
        self.bottleneck = DoubleConv(256, 512)

        # --- DECODER (Aufsteigender Pfad zur Wiederherstellung der Auflösung) ---
        self.up_trans1 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )  # Upsampling
        self.up_conv1 = DoubleConv(
            512, 256
        )  # 512 Kanäle wegen Skip-Connection (256 + 256)

        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(256, 128)

        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(128, 64)

        # --- OUTPUT LAYER ---
        # 1x1 Faltung zur Abbildung auf die Zielklassen (verändert die Auflösung nicht)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Encoder-Phase
        x1 = self.down1(x)
        p1 = self.pool(x1)

        x2 = self.down2(p1)
        p2 = self.pool(x2)

        x3 = self.down3(p2)
        p3 = self.pool(x3)

        # 2. Bottleneck
        b = self.bottleneck(p3)

        # 3. Decoder-Phase inklusive Skip-Connections
        up1 = self.up_trans1(b)
        concat1 = torch.cat([up1, x3], dim=1)  # Skip-Connection verknüpfen
        dec1 = self.up_conv1(concat1)

        up2 = self.up_trans2(dec1)
        concat2 = torch.cat([up2, x2], dim=1)
        dec2 = self.up_conv2(concat2)

        up3 = self.up_trans3(dec2)
        concat3 = torch.cat([up3, x1], dim=1)
        dec3 = self.up_conv3(concat3)

        # 4. Ausgabe-Layer
        out = self.out_conv(dec3)
        return out