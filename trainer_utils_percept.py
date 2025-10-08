# trainer_utils_percept.py
import torch
import torch.nn as nn
from torchvision import models
import lpips

class VGG16Perceptual(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for p in vgg.parameters(): p.requires_grad = False
        self.vgg = vgg
        self.layers = [3,8,15,22]

    def forward(self, x):
        feats = []
        cur = x
        for i, layer in enumerate(self.vgg):
            cur = layer(cur)
            if i in self.layers:
                feats.append(cur)
        return feats

def lpips_loss(device='cpu'):
    return lpips.LPIPS(net='vgg').to(device)
