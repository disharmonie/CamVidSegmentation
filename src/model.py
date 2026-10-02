import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Wiederverwendbarer Baustein: Zwei aufeinanderfolgende Faltungsschichten,
    jeweils gefolgt von Batch Normalization und ReLU-Aktivierung.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Dynamische U-Net Architektur für semantische Segmentierung und Knowledge Distillation."""
    
    def __init__(self, in_channels=3, out_channels=32, features=[64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- ENCODER ---
        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature

        # --- BOTTLENECK ---
        # Der Bottleneck hat standardmäßig doppelt so viele Kanäle wie die letzte Encoder-Schicht
        bottleneck_features = features[-1] * 2
        self.bottleneck = DoubleConv(features[-1], bottleneck_features)

        # --- DECODER ---
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # --- OUTPUT LAYER ---
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 1. Encoder-Phase
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 2. Bottleneck
        x = self.bottleneck(x)

        # 3. Decoder-Phase
        skip_connections = skip_connections[::-1]  # Reihenfolge für den Decoder umdrehen

        # In self.ups liegen abwechselnd ConvTranspose2d und DoubleConv
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)          # Upsampling
            skip_connection = skip_connections[i//2]
            
            concat_skip = torch.cat((skip_connection, x), dim=1) # Skip-Connection verknüpfen
            x = self.ups[i+1](concat_skip) # DoubleConv

        # 4. Ausgabe-Layer
        return self.out_conv(x)