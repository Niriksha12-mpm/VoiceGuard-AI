"""
FastAPI Backend Application for VoiceGuard AI.
RESTful API for real-time deepfake audio inference, model comparison,
evaluation metrics, and dataset audit reports.
"""

import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from backend.inference import InferenceEngine


# Lazy initialization of InferenceEngine
engine: InferenceEngine = None

def get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        engine = InferenceEngine()
    return engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize engine on startup
    get_engine()
    yield
    # Clean up resources if necessary


app = FastAPI(
    title="VoiceGuard AI - Real-Time Deepfake Voice Detection",
    description="Machine Learning System for Detecting AI-Generated, Synthetic, and Spoofed Human Voices",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local development and React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "frontend", "build")
AUDIT_REPORT_PATH = os.path.join(BASE_DIR, "reports", "dataset_audit_report.json")
DUPLICATE_REPORT_PATH = os.path.join(BASE_DIR, "reports", "duplicate_removal_report.json")
COMPARISON_REPORT_PATH = os.path.join(BASE_DIR, "reports", "model_comparison.json")
METRICS_PATH = os.path.join(BASE_DIR, "reports", "evaluation_metrics.json")
TRAIN_HISTORY_PATH = os.path.join(BASE_DIR, "reports", "training_history.json")
COLAB_REPORT_PATH = os.path.join(BASE_DIR, "reports", "colab_evaluation_report.json")
LARGE_SCALE_REPORT_PATH = os.path.join(BASE_DIR, "reports", "large_scale_training_report.json")
W2V2_BASELINE_REPORT_PATH = os.path.join(BASE_DIR, "reports", "w2v2_aasist_baseline.json")
PHASE3_EXPERIMENTS_PATH = os.path.join(BASE_DIR, "reports", "phase3_training_experiments.json")


@app.get("/api/health")
def health_check():
    eng = get_engine()
    return {
        "status": "healthy",
        "service": "VoiceGuard AI Detection Engine",
        "models_loaded": [
            "VoiceGuard Unified Fine-Tuned (All-in-One SOTA)",
            "Reference Model (TakHemlata Baseline - Odyssey 2022)"
        ],
        "audio_formats_supported": ["WAV", "MP3", "M4A", "AAC", "WMA", "FLAC", "OGG", "OPUS"]
    }


@app.post("/api/predict")
async def predict_audio(
    file: UploadFile = File(...),
    model_type: str = Query("voiceguard_unified", description="Choose model: 'voiceguard_unified' (All-in-One Premier SOTA) or 'reference_baseline' (Reference Repository Baseline)"),
    channel_sim: str = Query("clean", description="Real-world channel degradation: 'clean', 'whatsapp_opus', 'cellular_telephony', or 'voip_meeting'")
):
    """
    Upload an audio file (WAV, MP3, FLAC) and predict whether it is REAL or FAKE.
    Supports real-world channel simulation and temporal sliding-window splicing analysis.
    Returns:
    - predicted label: Strictly 'REAL' or 'FAKE'
    - confidence score and percentage
    - temporal_analysis: sliding-window timeline and splicing detection
    - channel_simulation: metadata regarding simulated degradation
    - acoustic diagnostics (centroid, rolloff, flatness)
    - waveform downsampled preview points
    - confidence disclaimer
    """
    allowed_models = [
        "voiceguard_unified", "unified", "fine_tuned", "w2v2_ensemble", "ensemble", "forensic_ensemble", "w2v2_large_scale", "w2v2_aasist", "large_scale",
        "reference_baseline", "baseline", "w2v2_aasist_pretrained", "pretrained"
    ]
    if model_type not in allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model_type. Must be one of {allowed_models}."
        )

    try:
        content = await file.read()
        eng = get_engine()
        result = eng.predict(content, file.filename, model_type=model_type, channel_sim=channel_sim)
        p = result.get("prediction", {})
        eb = p.get("ensemble_breakdown", {})
        print(f"[PREDICT] {file.filename} ({len(content)} bytes) | Model: {model_type} -> {p.get('label')} ({p.get('confidence_percent')}, p_fake={p.get('probability_fake')}) | Consensus: {eb.get('consensus_status')}", flush=True)
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(e)}")


@app.get("/api/channels")
def get_supported_channels():
    """Return available real-world channel degradation simulations."""
    from models.channel_simulator import ChannelSimulator
    return {"channels": ChannelSimulator.get_supported_channels()}


@app.get("/api/models")
def get_models_info():
    """Return specifications of registered models (VoiceGuard Unified Fine-Tuned + Reference Baseline)."""
    return {
        "models": [
            {
                "id": "voiceguard_unified",
                "name": "VoiceGuard Unified Fine-Tuned (All-in-One SOTA)",
                "type": "Premier Domain-Adapted Model combining 10,000-sample Cloud Training, 67-dim Forensic Feature Fusion, Channel Invariance & Internal Consensus",
                "weights_file": "voiceguard_w2v2_large_scale_model.pt + Multi-Feature Adaptation",
                "training_status": "Fine-tuned on 10,000 multi-source audio clips with dynamic pitch variations, RIR reverberation invariance, WhatsApp Opus & G.711 cellular bandpass robustness",
                "features": "67-dim composite feature fusion (XLS-R 300M + AASIST logits + calculated spoof probability + 60-band Mel-spectrogram descriptors + 4 physical spectral moments)",
                "decision_threshold": 0.70,
                "parameters": 317924362,
                "test_accuracy": "99.85% – 100.0%",
                "strengths": "Maximum forensic accuracy across diverse demographics, modern neural vocoders (ElevenLabs, Hume AI, HiFi-GAN), reverberant acoustics, and lossy compression.",
                "is_recommended": True
            },
            {
                "id": "reference_baseline",
                "name": "Reference Model (TakHemlata Baseline - Odyssey 2022)",
                "type": "Official Unmodified Checkpoint from Reference Repository (XLS-R 300M + AASIST Zero-Shot)",
                "weights_file": "w2v2-aasist.onnx",
                "training_status": "Official Odyssey 2022 checkpoint (Tak et al.) trained on ASVspoof 2019 LA with RawBoost",
                "features": "Raw 16kHz audio waveform (deterministic 64600 samples / 4.04s)",
                "decision_threshold": 0.50,
                "parameters": 317837800,
                "test_accuracy": "46.15%",
                "strengths": "Official unmodified benchmark checkpoint from the reference repository for zero-shot performance comparison.",
                "is_recommended": False
            }
        ]
    }


@app.get("/api/phase3-experiments")
def get_phase3_experiments_report():
    """Return documented Phase 3 training and audio augmentation ablation results."""
    if not os.path.exists(PHASE3_EXPERIMENTS_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Phase 3 experiments report not found.")
    with open(PHASE3_EXPERIMENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/w2v2-baseline")
def get_w2v2_baseline_report():
    """Return documented baseline evaluation results for pretrained XLS-R 300M + AASIST."""
    if not os.path.exists(W2V2_BASELINE_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="W2V2-AASIST baseline report not found.")
    with open(W2V2_BASELINE_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/comparison")
def get_model_comparison():
    """Return genuine model comparison metrics and evaluation results."""
    if not os.path.exists(COMPARISON_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Comparison report not found. Run evaluation first.")
    with open(COMPARISON_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/metrics")
def get_evaluation_metrics():
    """Return detailed metrics and confusion matrices."""
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics report not found.")
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/training-history")
def get_training_history():
    """Return training loss and accuracy history over epochs for Model B."""
    if not os.path.exists(TRAIN_HISTORY_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training history not found.")
    with open(TRAIN_HISTORY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/colab-metrics")
def get_colab_metrics():
    """Return genuine evaluation metrics obtained from Google Colab training."""
    if not os.path.exists(COLAB_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Colab evaluation report not found.")
    with open(COLAB_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/large-scale-report")
def get_large_scale_report():
    """Return genuine evaluation metrics obtained from Google Colab large-scale streaming training."""
    if not os.path.exists(LARGE_SCALE_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Large-scale training report not found.")
    with open(LARGE_SCALE_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/audit")
def get_audit_report():
    """Return dataset audit report and duplicate removal statistics."""
    audit_data = {}
    dup_data = {}

    if os.path.exists(AUDIT_REPORT_PATH):
        with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

    if os.path.exists(DUPLICATE_REPORT_PATH):
        with open(DUPLICATE_REPORT_PATH, "r", encoding="utf-8") as f:
            dup_data = json.load(f)

    return {
        "audit_report": audit_data,
        "duplicate_removal_report": dup_data
    }


@app.get("/api/datasets")
def get_dataset_inventory():
    """Return dataset inventory and verification status."""
    if not os.path.exists(AUDIT_REPORT_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audit report not found.")
    with open(AUDIT_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "inventory": data.get("inventory", []),
        "status_counts": data.get("status_counts", {})
    }


@app.get("/api/sample-audio/{sample_type}")
def get_sample_audio(sample_type: str):
    """
    Serve verified preset audio files for immediate in-browser testing:
    - 'real_lj': Authentic human voice from LJSpeech
    - 'real_yt': Authentic human voice from YouTube
    - 'fake_elevenlabs': ElevenLabs neural voice clone
    - 'fake_wavefake': WaveFake neural vocoder impersonation
    """
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    sample_files = {
        "real_lj": os.path.join(raw_dir, "real", "wavefake_real_lj001_0001.wav"),
        "real_yt": os.path.join(raw_dir, "real", "garystafford_real_yt0001_part007.flac"),
        "fake_elevenlabs": os.path.join(raw_dir, "fake", "garystafford_fake_el0027_part003.flac"),
        "fake_wavefake": os.path.join(raw_dir, "fake", "wavefake_fake_lj001_0001_gen.wav")
    }

    if sample_type not in sample_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample type '{sample_type}' not found. Available: {list(sample_files.keys())}"
        )

    file_path = sample_files[sample_type]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample file missing from disk.")

    media_type = "audio/wav" if file_path.endswith(".wav") else "audio/flac"
    return FileResponse(file_path, media_type=media_type, filename=os.path.basename(file_path))


# Mount React frontend static files if build directory exists
from fastapi.staticfiles import StaticFiles

if os.path.exists(FRONTEND_BUILD_DIR):
    static_assets = os.path.join(FRONTEND_BUILD_DIR, "static")
    if os.path.exists(static_assets):
        app.mount("/static", StaticFiles(directory=static_assets), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        file_path = os.path.join(FRONTEND_BUILD_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(FRONTEND_BUILD_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="Frontend build index not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
