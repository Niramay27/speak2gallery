# Speak2Gallery — Language-Bridged Audio–Image Retrieval

A multilingual zero-shot speech-to-image retrieval system. Speak in English, Hindi, or Bengali to retrieve semantically matching images from a COCO-based gallery.

---

## Project Structure

```
speak2gallery/
│
├── data/
│   ├── make_coco_tts_manifest.py             # Step 1: Build English manifest and TTS
│   ├── make_multilingual_coco_tts_manifest.py  # Step 2: Translate and multilingual TTS
│   ├── split.py                              # Step 3: Split English dataset
│   └── split_multilingual_manifest.py        # Step 4: Split multilingual dataset
│
├── models/
│   ├── models_cnn.py                         # CNN audio encoder
│   └── models_whisper.py                     # Whisper audio encoder
│
├── training/
│   ├── train_bridge_multilingual_cnn.py      # Train CNN (multilingual)
│   └── train_bridge_multilingual_whisper.py  # Train Whisper (multilingual)
│
├── evaluation/
│   ├── eval_retrieval_multilingual_base.py   # Eval random baseline
│   ├── eval_retrieval_multilingual_cnn.py    # Eval CNN (multilingual)
│   └── eval_retrieval_multilingual_whisper.py # Eval Whisper (multilingual)
│
├── retrieval/
│   ├── build_image_index.py                  # Pre-build CLIP image index
│   ├── build_tag_index.py                    # Pre-build keyword tag index
│   ├── hybrid_retrieval.py                   # Hybrid semantic + tag retrieval
│   ├── query_processor.py                    # Query normalization + extraction
│   └── retrieve_from_audio.py               # Single audio file → top-K images
│
├── app/
│   ├── app.py                                # Flask API server
│   ├── static/
│   │   └── index.html                        # Web frontend
│   └── README_HTTPS.md                       # HTTPS setup notes
│
├── scripts/
│   ├── 01_build_dataset.sh                   # Dataset creation pipeline
│   ├── 02_train.sh                           # Training pipeline
│   ├── 03_build_index.sh                     # Index building
│   ├── 04_evaluate.sh                        # Evaluation pipeline
│   └── 05_run_app.sh                         # Launch web application
│
├── data_coco_tts.py                          # Dataset loader (English)
├── data_coco_tts_multilingual.py             # Dataset loader (multilingual)
├── requirements.txt
└── run_all.sh                                # End-to-end runner
```

All generated files (manifests, audio, model checkpoints, indexes) are saved **directly in the project root** — no extra output directory is needed.

---

## Prerequisites

### System Requirements

- Python 3.10+
- CUDA-capable GPU (recommended; CPU works but is slow for training)
- ~20 GB disk for COCO dataset + generated audio files

### Install Dependencies

```bash
pip install -r requirements.txt
```

For HTTPS support in the web app:
```bash
pip install flask flask-cors cryptography
```

For spaCy query extraction (recommended):
```bash
python -m spacy download en_core_web_sm
```

For microphone recording (CLI demo only):
```bash
pip install sounddevice soundfile
```

---

## COCO Dataset — Required Before Running

You must download the MS-COCO 2017 dataset and tell the pipeline where it lives.

### 1. Download COCO

```bash
# Create a directory for the dataset (any location you choose)
mkdir -p /path/to/your/coco_dataset

# Download train2017 images (~18 GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d /path/to/your/coco_dataset/

# Download annotations (~241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d /path/to/your/coco_dataset/
```

After extraction you should have:
```
/path/to/your/coco_dataset/
├── train2017/          ← ~118k JPEG images
└── annotations/
    └── captions_train2017.json
```

### 2. Set the Paths in the Dataset Script

Open `scripts/01_build_dataset.sh` and edit the two lines at the top of the CONFIG section:

```bash
# Path to your COCO train2017 images directory
COCO_IMAGES="/path/to/your/coco_dataset/train2017"

# Path to the COCO captions annotation JSON
COCO_CAPTIONS="/path/to/your/coco_dataset/annotations/captions_train2017.json"
```

You can also set the number of captions to use:

```bash
LIMIT=5000   # Use 5000 captions (good for quick experiments)
             # Set to 0 to use all ~591k captions (full dataset)
```

> **Tip:** Start with `LIMIT=1000` for a fast smoke-test, then increase for real training.

---

## Dynamic Output Directory

**All output files are saved automatically into the project root** — the directory that contains `run_all.sh`. You do not need to set or change any output path. The scripts detect their own location at runtime:

```
speak2gallery/                  ← project root (OUT_DIR)
├── coco_train_tts.jsonl
├── coco_train_tts_multilingual.jsonl
├── coco_train_tts_multilingual_train.jsonl
├── coco_train_tts_multilingual_val.jsonl
├── tts_wavs/                   ← English TTS audio
├── tts_wavs_multilingual/
│   ├── hi/                     ← Hindi TTS audio
│   └── bn/                     ← Bengali TTS audio
├── image_index.pt              ← CLIP image index (Stage 3)
├── coco_tags.json              ← Tag index (Stage 3)
└── audio_encoder_multilingual.pt  ← Trained model (Stage 2)
```

---

## End-to-End Pipeline

### Quick Start (Run Everything)

```bash
chmod +x run_all.sh scripts/*.sh

# Edit COCO_IMAGES and COCO_CAPTIONS in scripts/01_build_dataset.sh first!
./run_all.sh
```

Or run each stage individually:

---

### Stage 1 — Build Dataset

```bash
# Edit COCO paths inside the script first (see "COCO Dataset" section above)
./scripts/01_build_dataset.sh
```

What it does:
1. Generates English TTS audio for COCO captions using `edge-tts`
2. Creates the English manifest JSONL
3. Translates captions to Hindi + Bengali using NLLB-200
4. Generates Hindi and Bengali TTS audio
5. Splits all manifests into train/val sets

**Required config** inside `scripts/01_build_dataset.sh`:
```bash
COCO_IMAGES="/path/to/your/coco_dataset/train2017"
COCO_CAPTIONS="/path/to/your/coco_dataset/annotations/captions_train2017.json"
LIMIT=5000   # number of captions to use (0 = all)
```

---

### Stage 2 — Train the Audio Encoder

```bash
./scripts/02_train.sh             # Whisper encoder (default, best model)
MODEL=cnn ./scripts/02_train.sh   # CNN encoder (faster, lower accuracy)
```

Outputs saved to project root:
- `audio_encoder_multilingual.pt` — Whisper encoder
- `audio_encoder_cnn_multilingual.pt` — CNN encoder (if MODEL=cnn)

---

### Stage 3 — Build Retrieval Indexes

```bash
./scripts/03_build_index.sh
```

Outputs saved to project root:
- `image_index.pt` — CLIP image embeddings
- `coco_tags.json` — per-image keyword tags

---

### Stage 4 — Evaluate

```bash
MODEL=whisper ./scripts/04_evaluate.sh    # default
MODEL=cnn     ./scripts/04_evaluate.sh
MODEL=base    ./scripts/04_evaluate.sh    # random baseline
```

Reports R@1, R@5, R@10, MRR, and Median Rank per language (EN / HI / BN) and combined.

---

### Stage 5 — Launch Web Application

```bash
./scripts/05_run_app.sh
```

Open: `http://localhost:5000`

For HTTPS (required for microphone from remote browsers):
```bash
SSL_ADHOC=1 APP_PUBLIC_HOST=YOUR_SERVER_IP ./scripts/05_run_app.sh
```

Open: `https://YOUR_SERVER_IP:5000` and accept the self-signed certificate.

---

## Retrieve from a Single Audio File

```bash
python retrieval/retrieve_from_audio.py \
  --audio      query.wav \
  --index      image_index.pt \
  --checkpoint audio_encoder_multilingual.pt \
  --tag_index  coco_tags.json \
  --top_k 9 \
  --alpha 0.7
```

---

## Web Application

The Flask app (`app/app.py`) exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the web UI |
| `/api/search/text` | POST | JSON `{"query": "...", "top_k": 9}` |
| `/api/search/audio` | POST | Multipart form with `audio` file |
| `/api/status` | GET | Health check |

Run directly:
```bash
cd speak2gallery
python app/app.py
```

---

## Environment Variables (App)

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_DIR` | auto-detected | Override project root (where indexes and checkpoints live) |
| `COCO_DIRS` | `<project_root>/coco_dataset/train2017:<project_root>/coco_dataset/val2017` | Colon-separated list of COCO image directories |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `5000` | Port |
| `APP_PUBLIC_HOST` | `localhost` | Displayed in startup banner |
| `SSL_ADHOC` | `0` | `1` = self-signed HTTPS |
| `SSL_CERT_FILE` | — | Path to TLS certificate |
| `SSL_KEY_FILE` | — | Path to TLS private key |
| `FORCE_HTTPS` | `0` | `1` = redirect HTTP → HTTPS |
| `WHISPER_NAME` | `openai/whisper-medium` | Whisper model for audio search |

---

## Model Checkpoints

| File | Description |
|------|-------------|
| `audio_encoder_multilingual.pt` | **Main model** — Whisper (small) + projection, multilingual |
| `audio_encoder_cnn_multilingual.pt` | CNN encoder, multilingual |

---

## Results Summary

### Audio → Image Retrieval (multilingual val set)

| Model | R@1 | R@5 | R@10 | Median Rank |
|-------|-----|-----|------|-------------|
| Random Baseline | 0.48% | 2.42% | 4.83% | 102 |
| CNN (from scratch) | 10.63% | 32.37% | 45.89% | 12 |
| **Whisper (frozen + proj)** | **33.82%** | **70.05%** | **84.06%** | **3** |

### Image → Audio Retrieval (multilingual val set)

| Model | R@1 | R@5 | R@10 | Median Rank |
|-------|-----|-----|------|-------------|
| Random Baseline | 0.48% | 3.38% | 4.83% | 105 |
| CNN (from scratch) | 7.73% | 28.50% | 41.06% | 17 |
| **Whisper (frozen + proj)** | **26.09%** | **65.22%** | **78.26%** | **3** |
