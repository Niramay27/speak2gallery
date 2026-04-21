import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self, n_mels=80, embed_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.proj = nn.Linear(256, embed_dim)

    def forward(self, mel):
        # mel: [B, n_mels, T]
        x = mel.unsqueeze(1)      # [B, 1, n_mels, T]
        x = self.net(x)           # [B, 256, 1, 1]
        x = x.flatten(1)          # [B, 256]
        x = self.proj(x)          # [B, embed_dim]
        x = F.normalize(x, dim=-1)
        return x