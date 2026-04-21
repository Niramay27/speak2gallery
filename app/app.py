#!/usr/bin/env python3
"""
speak2gallery Flask API
=======================
Run from the project root (speak2gallery/):

    python app/app.py

Or with HTTPS (self-signed):
    SSL_ADHOC=1 APP_PUBLIC_HOST=YOUR_IP python app/app.py

Environment variables
---------------------
PROJECT_DIR   — override auto-detected project root (directory containing app/)
COCO_DIRS     — colon-separated list of COCO image directories
                default: <project_root>/coco_dataset/train2017:<project_root>/coco_dataset/val2017
HOST          — bind address (default 0.0.0.0)
PORT          — port (default 5000)
APP_PUBLIC_HOST — displayed in startup banner (default localhost)
SSL_ADHOC     — 1 = generate self-signed cert
SSL_CERT_FILE — path to TLS certificate
SSL_KEY_FILE  — path to TLS private key
FORCE_HTTPS   — 1 = redirect HTTP → HTTPS

Endpoints:
    GET  /                          -> serves index.html
    GET  /image/<token>             -> serves COCO images by absolute path
    POST /api/search/text           -> text query -> top-k images
    POST /api/search/audio          -> audio file -> top-k images
    GET  /api/status                -> health check
"""

import os
import sys
import json
import base64
import tempfile
import traceback
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory, abort, redirect
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

# ── Resolve project root (the directory that contains this app/ folder) ──────
_APP_DIR     = Path(__file__).resolve().parent          # …/speak2gallery/app
PROJECT_DIR  = Path(os.getenv("PROJECT_DIR", str(_APP_DIR.parent)))  # …/speak2gallery

# Add project root to sys.path so that retrieval/, models/, etc. are importable
for _p in [str(PROJECT_DIR), str(PROJECT_DIR / "retrieval"), str(PROJECT_DIR / "models")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

app = Flask(__name__, static_folder="static", template_folder="static")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# ── COCO image directories ────────────────────────────────────────────────────
_default_coco = ":".join([
    "/home/other/coco_dataset/train2017",
    "/home/other/coco_dataset/val2017",
])
COCO_DIRS = [Path(d) for d in os.getenv("COCO_DIRS", _default_coco).split(":") if d]

# ── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "image_index_path":  str(PROJECT_DIR / "image_index.pt"),
    "checkpoint_path":   str(PROJECT_DIR / "audio_encoder_multilingual.pt"),
    "tag_index_path":    str(PROJECT_DIR / "coco_tags.json"),
    "whisper_name":      os.getenv("WHISPER_NAME", "openai/whisper-small"),
    "embed_dim":         512,
    "n_mels":            80,
    "default_top_k":     9,
    "default_alpha":     0.7,
}

_retriever        = None
_audio_model      = None
_query_processor  = None
_device           = None


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_ssl_context():
    cert_file = os.getenv("SSL_CERT_FILE", "").strip()
    key_file  = os.getenv("SSL_KEY_FILE",  "").strip()
    use_adhoc = env_flag("SSL_ADHOC")

    if cert_file or key_file:
        if not cert_file or not key_file:
            raise RuntimeError("Both SSL_CERT_FILE and SSL_KEY_FILE must be set together.")
        cert_path = Path(cert_file)
        key_path  = Path(key_file)
        if not cert_path.exists():
            raise FileNotFoundError(f"SSL certificate not found: {cert_path}")
        if not key_path.exists():
            raise FileNotFoundError(f"SSL key not found: {key_path}")
        return (str(cert_path), str(key_path))

    if use_adhoc:
        return "adhoc"

    return None


def get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[app] Device: {_device}")
    return _device


def get_retriever():
    global _retriever
    if _retriever is None:
        print("[app] Loading HybridRetriever…")
        from hybrid_retrieval import HybridRetriever
        _retriever = HybridRetriever(
            image_index_path=CONFIG["image_index_path"],
            tag_index_path=CONFIG["tag_index_path"] if Path(CONFIG["tag_index_path"]).exists() else None,
            alpha=CONFIG["default_alpha"],
            device=get_device(),
        )
        print("[app] HybridRetriever ready.")
    return _retriever


def get_audio_model():
    global _audio_model
    if _audio_model is None:
        print("[app] Loading WhisperAudioEncoder…")
        import torch
        from models_whisper import WhisperAudioEncoder
        device = get_device()
        model  = WhisperAudioEncoder(
            whisper_name=CONFIG["whisper_name"],
            embed_dim=CONFIG["embed_dim"],
            freeze_encoder=True,
        ).to(device)
        state = torch.load(CONFIG["checkpoint_path"], map_location=device)
        model.load_state_dict(state)
        model.eval()
        _audio_model = model
        print("[app] WhisperAudioEncoder ready.")
    return _audio_model


def get_query_processor():
    global _query_processor
    if _query_processor is None:
        print("[app] Loading QueryProcessor…")
        from query_processor import QueryProcessor
        _query_processor = QueryProcessor(translate=False, use_spacy=True)
        print("[app] QueryProcessor ready.")
    return _query_processor


@app.before_request
def force_https_redirect_if_requested():
    if not env_flag("FORCE_HTTPS"):
        return None
    host = request.host.split(":")[0]
    if host in {"localhost", "127.0.0.1"}:
        return None
    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    if request.is_secure or forwarded_proto == "https":
        return None
    return redirect(request.url.replace("http://", "https://", 1), code=301)


@app.after_request
def add_security_headers(response):
    response.headers["Permissions-Policy"] = "microphone=(self)"
    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    if request.is_secure or forwarded_proto == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_and_preprocess_audio(audio_path, sample_rate=16000, n_mels=80):
    import torch
    import torchaudio
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=400, hop_length=160, n_mels=n_mels)
    db  = torchaudio.transforms.AmplitudeToDB(stype="power")
    mel = db(melspec(wav))
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel


def transcribe_audio(audio_path, model_size="small"):
    try:
        import whisper
        model  = whisper.load_model(model_size)
        result = model.transcribe(audio_path, fp16=False)
        return {"text": result["text"].strip(), "language": result.get("language", "en")}
    except ImportError:
        try:
            from transformers import pipeline as hf_pipeline
            import torchaudio
            pipe = hf_pipeline("automatic-speech-recognition",
                                model=f"openai/whisper-{model_size}", device=-1)
            wav, sr = torchaudio.load(audio_path)
            wav = wav.mean(0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            result = pipe({"raw": wav.numpy(), "sampling_rate": 16000})
            return {"text": result["text"].strip(), "language": "en"}
        except Exception as e:
            return {"text": "", "language": "en", "error": str(e)}
    except Exception as e:
        return {"text": "", "language": "en", "error": str(e)}


def format_results(raw_results):
    out = []
    for r in raw_results:
        img_path = r["image_path"]
        encoded  = base64.urlsafe_b64encode(img_path.encode()).decode()
        out.append({
            "rank":           r["rank"],
            "image_token":    encoded,
            "image_filename": Path(img_path).name,
            "semantic_score": round(r["semantic_score"], 6),
            "tag_score":      round(r["tag_score"],      6),
            "final_score":    round(r["final_score"],    6),
        })
    return out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/image/<token>")
def serve_image(token):
    try:
        img_path = base64.urlsafe_b64decode(token.encode()).decode()
        p = Path(img_path)
        if not p.exists():
            abort(404)
        allowed = any(str(p).startswith(str(d)) for d in COCO_DIRS)
        if not allowed:
            abort(403)
        return send_file(str(p), mimetype="image/jpeg")
    except Exception:
        abort(400)


@app.route("/api/status")
def status():
    import torch
    ssl_enabled = bool(request.is_secure or
                       request.headers.get("X-Forwarded-Proto") == "https")
    return jsonify({
        "status":             "ok",
        "device":             get_device(),
        "cuda":               torch.cuda.is_available(),
        "index_exists":       Path(CONFIG["image_index_path"]).exists(),
        "checkpoint_exists":  Path(CONFIG["checkpoint_path"]).exists(),
        "tag_index_exists":   Path(CONFIG["tag_index_path"]).exists(),
        "https":              ssl_enabled,
    })


@app.route("/api/search/text", methods=["POST"])
def search_text():
    try:
        data     = request.get_json()
        query    = (data.get("query") or "").strip()
        top_k    = int(data.get("top_k",  CONFIG["default_top_k"]))
        alpha    = float(data.get("alpha", CONFIG["default_alpha"]))
        src_lang = data.get("lang", "auto")

        if not query:
            return jsonify({"error": "query is required"}), 400

        qp = get_query_processor()
        qc = qp.process(query, src_lang=src_lang)

        retriever       = get_retriever()
        retriever.alpha = alpha
        raw             = retriever.retrieve(qc, top_k=top_k)

        return jsonify({
            "query":      query,
            "normalized": qc.normalized_text,
            "objects":    qc.objects,
            "attributes": qc.attributes,
            "relations":  qc.relations,
            "keywords":   qc.all_keywords,
            "results":    format_results(raw),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/search/audio", methods=["POST"])
def search_audio():
    try:
        import torch
        import torch.nn.functional as F

        top_k    = int(request.form.get("top_k",  CONFIG["default_top_k"]))
        alpha    = float(request.form.get("alpha", CONFIG["default_alpha"]))
        src_lang = request.form.get("lang", "auto")

        if "audio" not in request.files:
            return jsonify({"error": "audio file required"}), 400

        audio_file = request.files["audio"]
        suffix     = Path(audio_file.filename).suffix if audio_file.filename else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            asr           = transcribe_audio(tmp_path)
            transcribed   = asr.get("text", "")
            detected_lang = asr.get("language", "en") if src_lang == "auto" else src_lang

            qp = get_query_processor()
            qc = qp.process(transcribed, src_lang=detected_lang)

            mel = load_and_preprocess_audio(tmp_path, n_mels=CONFIG["n_mels"])
            mel = mel.to(get_device())

            audio_model = get_audio_model()
            with torch.no_grad():
                audio_emb = F.normalize(audio_model(mel), dim=-1)

            retriever       = get_retriever()
            retriever.alpha = alpha
            raw = retriever.retrieve_from_audio_embedding(audio_emb.cpu(), qc, top_k=top_k)

            return jsonify({
                "transcription": transcribed,
                "detected_lang": detected_lang,
                "normalized":    qc.normalized_text,
                "objects":       qc.objects,
                "attributes":    qc.attributes,
                "relations":     qc.relations,
                "keywords":      qc.all_keywords,
                "results":       format_results(raw),
            })
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    host        = os.getenv("HOST", "0.0.0.0")
    port        = int(os.getenv("PORT", "5000"))
    public_host = os.getenv("APP_PUBLIC_HOST", "localhost")
    ssl_context = get_ssl_context()
    scheme      = "https" if ssl_context else "http"

    print("=" * 60)
    print("  speak2gallery API server")
    print("=" * 60)
    print(f"  Project dir : {PROJECT_DIR}")
    print(f"  Index       : {CONFIG['image_index_path']}")
    print(f"  Checkpoint  : {CONFIG['checkpoint_path']}")
    print(f"  Tag index   : {CONFIG['tag_index_path']}")
    print(f"  COCO dirs   : {[str(d) for d in COCO_DIRS]}")
    print(f"  SSL enabled : {'yes' if ssl_context else 'no'}")
    print("=" * 60)
    print(f"  Open: {scheme}://{public_host}:{port}")
    print("=" * 60)

    app.run(host=host, port=port, debug=False, threaded=True, ssl_context=ssl_context)