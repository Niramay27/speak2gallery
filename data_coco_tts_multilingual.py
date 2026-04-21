import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class CocoTtsMultilingual(Dataset):
    def __init__(self, manifest_jsonl, clip_preprocess, clip_tokenizer,
                 sample_rate=16000, n_mels=80):
        self.rows = []
        with open(manifest_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sample_rate = sample_rate

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.rows)

    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        mel = self.melspec(wav)
        mel = self.db(mel).squeeze(0)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel

    def __getitem__(self, idx):
        row = self.rows[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = self.clip_preprocess(image)

        english_text_tokens = self.clip_tokenizer([row["english_caption"]])[0]
        mel = self.load_audio(row["audio_path"])

        return {
            "image": image,
            "text_tokens": english_text_tokens,
            "mel": mel,
            "lang": row["lang"],
            "caption": row["caption"],
            "english_caption": row["english_caption"],
            "image_path": row["image_path"],
            "audio_path": row["audio_path"],
        }


def collate_coco_tts_multilingual(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    text_tokens = torch.stack([x["text_tokens"] for x in batch], dim=0)

    mels = [x["mel"].transpose(0, 1) for x in batch]   # [T, n_mels]
    mels = pad_sequence(mels, batch_first=True)        # [B, T, n_mels]
    mels = mels.transpose(1, 2)                        # [B, n_mels, T]

    return {
        "image": images,
        "text_tokens": text_tokens,
        "mel": mels,
        "lang": [x["lang"] for x in batch],
        "caption": [x["caption"] for x in batch],
        "english_caption": [x["english_caption"] for x in batch],
        "image_path": [x["image_path"] for x in batch],
        "audio_path": [x["audio_path"] for x in batch],
    }