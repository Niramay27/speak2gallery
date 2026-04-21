import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel


class WhisperAudioEncoder(nn.Module):
    def __init__(
        self,
        whisper_name="openai/whisper-medium",
        embed_dim=512,
        freeze_encoder=True,
    ):
        super().__init__()

        self.whisper = WhisperModel.from_pretrained(whisper_name)
        self.hidden_size = self.whisper.config.d_model

        if freeze_encoder:
            for p in self.whisper.encoder.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(self.hidden_size, embed_dim)

    def forward(self, mel):
        """
        mel: [B, 80, T]
        Whisper expects [B, 80, 3000] in the usual setup.
        So we pad or crop to T=3000.
        """
        b, n_mels, t = mel.shape

        if t < 3000:
            pad = 3000 - t
            mel = F.pad(mel, (0, pad))
        elif t > 3000:
            mel = mel[:, :, :3000]

        out = self.whisper.encoder(input_features=mel)
        x = out.last_hidden_state              # [B, T', H]
        x = x.mean(dim=1)                      # mean pool over time
        x = self.proj(x)                       # [B, 512]
        x = F.normalize(x, dim=-1)
        return x