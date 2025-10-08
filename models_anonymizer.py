# models_anonymizer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv/deconv helpers
def conv(in_ch, out_ch, k=4, s=2, p=1, bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, k, s, p)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)

def deconv(in_ch, out_ch, k=4, s=2, p=1, bn=True):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, k, s, p)]
    if bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class GeneratorUNet(nn.Module):
    """
    Encoder-decoder that:
     - forward(x) -> (out, z) where z is bottleneck latent (for EMA smoothing)
     - decode_from_latent(z) -> image (used at inference / to export)
    Output range: tanh -> [-1,1]
    """
    def __init__(self, in_ch=3, base=64, bottleneck_dim=512):
        super().__init__()
        # Encoder
        self.enc1 = conv(in_ch, base, bn=False)        # 256 -> 128
        self.enc2 = conv(base, base*2)                 # 128 -> 64
        self.enc3 = conv(base*2, base*4)               # 64 -> 32
        self.enc4 = conv(base*4, base*8)               # 32 -> 16
        self.enc5 = conv(base*8, base*8)               # 16 -> 8
        self.enc6 = conv(base*8, base*8)               # 8 -> 4

        # bottleneck projection
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base*8, bottleneck_dim)
        self.fc_inv = nn.Linear(bottleneck_dim, base*8*4*4)

        # Decoder (transpose conv)
        self.dec1 = deconv(base*8, base*8)  # 4 -> 8
        self.dec2 = deconv(base*8, base*8)  # 8 -> 16
        self.dec3 = deconv(base*8, base*4)  # 16 -> 32
        self.dec4 = deconv(base*4, base*2)  # 32 -> 64
        self.dec5 = deconv(base*2, base)    # 64 -> 128
        self.dec6 = nn.Sequential(
            nn.ConvTranspose2d(base, in_ch, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, return_latent=True):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)  # shape B x C x 4 x 4

        pooled = self.avg(e6).view(e6.size(0), -1)
        z = self.fc(pooled)  # B x bottleneck_dim

        feat = self.fc_inv(z).view(-1, e6.size(1), 4, 4)

        d1 = self.dec1(feat)        # 4->8
        d1 = d1 + e5                # skip
        d2 = self.dec2(d1)          # 8->16
        d2 = d2 + e4
        d3 = self.dec3(d2)          # 16->32
        d3 = d3 + e3
        d4 = self.dec4(d3)          # 32->64
        d4 = d4 + e2
        d5 = self.dec5(d4)          # 64->128
        d5 = d5 + e1
        out = self.dec6(d5)         # 128->256
        if return_latent:
            return out, z
        return out

    def decode_from_latent(self, z):
        """
        Decode from latent z (B, bottleneck_dim) to image [-1,1]
        """
        feat = self.fc_inv(z).view(-1, self.enc6[0].out_channels, 4, 4)
        d1 = self.dec1(feat)
        # we can't do skip by encoder features here; decode-only
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        d5 = self.dec5(d4)
        out = self.dec6(d5)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, base*8, 4, 1, 1), nn.BatchNorm2d(base*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*8, 1, 4, 1, 1)  # patch output
        )

    def forward(self, x):
        return self.model(x)
