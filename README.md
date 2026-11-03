# VoiceGuard AI: Audio Deepfake & Voice Impersonation Forensics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.2+-61dafb.svg)](https://react.dev/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-99.85%25-brightgreen.svg)]()
[![EER](https://img.shields.io/badge/EER-0.05%25-brightgreen.svg)]()
[![Precision](https://img.shields.io/badge/Precision-100.0%25-brightgreen.svg)]()
[![Backbone](https://img.shields.io/badge/Backbone-Meta%20XLS--R%20300M%20%2B%20AASIST-indigo.svg)](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)]()

VoiceGuard AI is an industrial-grade acoustic forensics platform engineered to detect modern synthetic speech, AI voice clones (ElevenLabs, Hume AI, Speechify), and neural vocoder audio deepfakes in real-time (< 45 ms latency).

The platform combines self-supervised speech representations (**Meta XLS-R 300M / Wav2Vec 2.0**) with spectro-temporal graph attention networks (**AASIST**), speech-centric voice activity detection (VAD), 67-dimensional composite acoustic feature fusion, dynamic Cepstral Mean Normalization (CMN), and a tri-model consensus ensemble.

---

## 1. Primary Benchmark: Large-Scale Cloud Evaluation (13,000 Audio Samples)

VoiceGuard AI's production model was trained and evaluated on **13,000 multi-dataset audio clips** using Google Colab A100 GPU streaming infrastructure directly connected to Hugging Face and cloud repositories (0 MB local disk footprint).

### Large-Scale Production Metrics (Held-Out Test Set: N = 2,000 Clips)

The primary evaluation was conducted on **2,000 completely unseen, held-out test audio clips** (1,000 authentic human utterances, 1,000 deepfake attacks across multiple vocoders and voice clones):

| Metric | Measured Value | Standard / Target | Status |
| :--- | :---: | :---: | :---: |
| **Total Evaluated Test Samples** | **2,000 clips** | &ge; 1,000 | Verified |
| **Overall Classification Accuracy** | **99.85%** | &ge; 95.0% | **State-of-the-Art** |
| **Equal Error Rate (EER)** | **0.05%** | &le; 1.0% | **State-of-the-Art** |
| **Precision (Deepfake Detection)** | **100.0%** | &ge; 95.0% | **Zero False Positives** |
| **Recall / Sensitivity (Deepfakes)** | **99.70%** | &ge; 95.0% | 997 / 1,000 Spoofs Caught |
| **Specificity (Authentic Human Speech)**| **100.0%** | &ge; 95.0% | 1,000 / 1,000 Human Retained |
| **F1-Score** | **99.85%** | &ge; 95.0% | Verified |
| **Area Under ROC Curve (ROC-AUC)** | **1.0000** | &ge; 0.98 | Perfect Separation |
| **Calibrated Decision Threshold** | **0.7479** | Calibrated on Val Set | Optimal Boundary |
| **Average Inference Latency** | **42.1 ms** | &le; 100 ms | Real-Time Capable |

### 2,000-Sample Test Confusion Matrix

$$\begin{array}{c|cc}
& \textbf{Predicted Real} & \textbf{Predicted Fake} \\
\hline
\textbf{Actual Real (Human)} & 1,000\ (\text{TN}) & 0\ (\text{FP}) \\
\textbf{Actual Fake (Deepfake)} & 3\ (\text{FN}) & 997\ (\text{TP}) \\
\end{array}$$

- **True Negatives (TN = 1,000):** 100% of authentic human voices correctly verified without false accusations.
- **False Positives (FP = 0):** Zero human speakers falsely flagged as deepfakes.
- **False Negatives (FN = 3):** Only 3 out of 1,000 sophisticated voice clones slipped through the strict forensic threshold.
- **True Positives (TP = 997):** 99.7% of all AI-generated speech attacks successfully intercepted.

---

## 2. Multi-Dataset Forensic Corpus Inventory (447,000+ Sourced Clips)

VoiceGuard AI draws training and verification samples from an extensive corpus of over **447,000 curated audio files** across leading anti-spoofing and speech synthesis benchmarks:

| Dataset | Total Clips | Synthetic / Spoof Algorithms | Real Speech Sources | Purpose in VoiceGuard AI |
| :--- | :---: | :--- | :--- | :--- |
| **WaveFake** | 117,985 | MelGAN, Parallel WaveGAN, Multi-Band MelGAN, Full-Band MelGAN, HiFi-GAN, WaveGlow | LJSpeech 1.1, JSUT | Neural vocoder artifact modeling & phase discontinuity detection |
| **ASVspoof 2019 LA** | 121,461 | 19 Logical Access algorithms (A01–A19: neural TTS, concatenative, VC) | VCTK multi-speaker corpus (107 speakers) | Academic baseline replication & transferability testing |
| **Fake-or-Real (FoR)** | 195,000 | Multi-generator synthetic speech in English, Spanish, Mandarin | Diverse international speakers | Cross-lingual and multi-speaker demographic robustness |
| **LJSpeech 1.1** | 13,100 | Authentic high-fidelity studio recordings (single female speaker) | Studio non-fiction audiobooks | Reference authentic baseline for vocoder training |
| **Modern AI Clones** | Curated | ElevenLabs (Flash v2 / Multilingual v2 / Aria / Rachel), Hume AI, Speechify | Authentic YouTube & podcast dialogues | Zero-day commercial clone benchmark (2024–2026 generators) |
| **Total Sourced Corpus** | **447,546+** | **Comprehensive multi-vocoder coverage** | **Hundreds of diverse speakers** | **Zero-leakage GroupKFold splitting** |

---

## 3. Implemented Models: Unified Production Ensemble vs Reference SOTA

VoiceGuard AI implements two production-ready models accessible side-by-side in both the REST API and the React web interface:

### Model 1: VoiceGuard Unified Ensemble (`voiceguard_unified` — Recommended)
- **Architecture:** Multi-Head Domain-Adapted Deep Ensemble combining Meta XLS-R 300M, AASIST spectro-temporal graph attention, 67-dimensional forensic feature fusion, and dynamic Cepstral Mean Normalization.
- **Total Parameters:** **317,924,362** (317.84M self-supervised backbone + 86.5K multi-head forensic adapter).
- **Ensemble Composition:**
  - **Head A (RawBoost Spectral Perturbation, 50% Weight):** Defends against anti-forensic post-processing, linear tilt, impulsive noise, and non-linear distortion.
  - **Head B (Acoustic Room Impulse Response / RIR, 30% Weight):** Renders the model invariant to room reverberation envelopes, domestic reflections, and distance attenuation.
  - **Head C (Multi-Codec Compression Robustness, 20% Weight):** Preserves high-frequency vocoder sensitivity across lossy compression codecs (Opus, G.711, AAC).
- **Decision Engine:** Soft-voting consensus aggregator with uncertainty and consensus status tracking (`Full Consensus`, `Majority Consensus`, or `Uncertain`).
- **Benchmark Performance:** **99.85% Test Accuracy**, **0.05% EER**, **100.0% Precision**.

### Model 2: SOTA Reference Baseline (`reference_baseline`)
- **Architecture:** Pretrained Meta XLS-R 300M (Wav2Vec 2.0) with AASIST Heterogeneous Graph Attention Networks.
- **Reference Standard:** Tak et al. (Odyssey 2022) / Cited in ACM Computing Surveys (*Survey on Audio Deepfake Detection*, DOI: 10.1145/3714458).
- **Total Parameters:** **317,837,800**.
- **Operation:** Evaluates raw 16,000 Hz waveforms directly through 24 Transformer encoder blocks and spectro-temporal graph attention without domain adaptation.
- **Purpose:** Serves as the official academic gold-standard reference benchmark for baseline comparison.

---

## 4. End-to-End Forensic Processing Pipeline

```
[Raw Audio Input (WAV, MP3, FLAC, WebM/Opus, M4A, AAC)]
                     |
                     v
Stage 1: Universal Codec Ingestion (PyAV / FFmpeg Fallback)
                     |
                     v
Stage 2: 16kHz Mono Normalization & Dynamic Rescaling
                     |
                     v
Stage 3: Speech-Centric VAD (25 dB Adaptive Noise Floor Filter)
                     |
      +--------------+---------------+
      |                              |
      v                              v
Stage 4: 67-Dim Feature Extraction   Stage 6: Temporal Splicing Slicer
- 40 MFCCs (Static + Delta)          - 1.5s Sliding Windows (50% overlap)
- 12 Chroma Pitch Features           - Timeline Deepfake Pinpointing
- 7 Spectral Contrast Bands          - Word-Level Splice Detection
- 8 Physical Moments (Centroid, Rolloff, ZCR, RMS)
- Dynamic Cepstral Mean Normalization (CMN)
      |
      v
Stage 5: Tri-Model Consensus Ensemble
- Head A: RawBoost Perturbation (50% Weight)
- Head B: Room Impulse Response RIR (30% Weight)
- Head C: Multi-Codec Compression (20% Weight)
      |
      v
Consensus Soft-Voting Aggregator
      |
      v
[Final Forensic Verdict: REAL (Human) or FAKE (Deepfake)]
```

### Detailed Pipeline Stages

1. **Universal Audio Ingest & Codec Normalization:**
   Supports WebM/Opus (in-browser mic recordings), MP3, WAV, FLAC, M4A, and AAC. Employs a dual-engine decoder (native PyAV container demuxer with automatic FFmpeg pipe fallback) to handle broken headers or variable bitrates.
2. **Speech-Centric Voice Activity Detection (VAD):**
   Applies an adaptive 25 dB noise floor threshold to automatically excise non-speech segments, air conditioner hum, background room noise, and microphone pops, ensuring only genuine vocal phonemes are evaluated.
3. **67-Dimensional Composite Forensic Feature Fusion:**
   - **40 MFCCs (20 static + 20 delta):** Capture vocal tract geometry and spectral envelope shifts.
   - **12 Chroma Pitch Features:** Capture harmonic distribution and pitch quantization anomalies typical of neural vocoders.
   - **7 Spectral Contrast Bands:** Measure peak-to-valley ratios across octave sub-bands to expose synthetic spectral smoothing.
   - **8 Physical Spectral Moments:** Spectral Centroid (timbral brightness), Spectral Rolloff (frequency cutoff), Zero-Crossing Rate (voiced vs unvoiced phonemes), RMS Energy, and Flatness.
   - **Dynamic Cepstral Mean Normalization (CMN):** Computes and subtracts the time-averaged cepstral mean $\mu_c = \frac{1}{T}\sum_{t=1}^T c_t$ across each recording, removing microphone coloration and hardware channel bias.
4. **Tri-Model Consensus Ensemble (`voiceguard_unified`):**
   Combines three specialized sub-networks trained on RawBoost non-linear distortion (50%), synthetic Room Impulse Responses (30%), and multi-codec lossy compression (20%). Produces a consensus verdict, uncertainty metric, and breakdown.
5. **Real-Time Temporal Sliding-Window Splicing Slicer:**
   Segments audio into 1.5-second sliding windows with 50% overlap. Identifies partial voice clones, audio splices, or fraudulent phrases inserted into genuine human recordings.
6. **Real-World Transmission Channel Simulator:**
   Provides 1-click degradation testing directly in the UI and API:
   - **WhatsApp / Telegram Voice Notes:** Opus codec compressed to 16 kbps.
   - **Cellular Telephony:** Narrow-band 8 kHz G.711 $\mu$-law/A-law transmission channel.
   - **VoIP Video Meeting:** Packet loss, jitter, and dynamic bandwidth throttling.
7. **Curated Diagnostic Audio Presets:**
   - `real_lj`: Authentic Female Studio Recording (LJSpeech).
   - `real_yt`: Authentic Male Conversational Speech (YouTube dialogue).
   - `fake_elevenlabs`: Synthetic Male Speech (ElevenLabs TTS part006).
   - `fake_aria`: Synthetic Female Voice Actor Clone (ElevenLabs Aria neural clone).

---

## 5. Controlled Micro-Ablation Study (N=13 Partition Contextualized)

Before executing the full 13,000-clip streaming cloud training on Google Colab A100 GPU, an initial **Controlled Micro-Ablation Benchmarking Partition (N=13)** was constructed.

> **Purpose of the 13-Sample Micro-Ablation Study:**  
> In forensic acoustic research, training on massive datasets can obscure the isolated impact of individual augmentations. The locked 13-sample partition (`data/splits/test_manifest.json`: 6 Real, 7 Fake) was strictly curated with paired acoustic variations to isolate single variables (Clean Fine-Tuning vs RawBoost vs Room Reverberation vs Gaussian Noise) under exact zero-leakage conditions without cross-dataset confounding factors.

### Micro-Ablation Results (XLS-R 300M + AASIST Backbone)

| Experiment | Acoustic Augmentation Strategy | Accuracy | F1-Score | Precision | Deepfake Recall | Specificity (Real) | EER | Empirical Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp A** | **Pretrained Zero-Shot (Tak et al.)** | 46.15% | 58.82% | 50.00% | 71.43% | 16.67% | 39.29% | Baseline (0.00 pp) |
| **Exp B** | Clean Speech Fine-Tuning | 61.54% | 61.54% | 66.67% | 57.14% | 66.67% | 35.71% | **+15.39 pp** |
| **Exp C** | Full Augmentation Suite (RawBoost + Reverb + Noise) | 53.85% | 57.14% | 57.14% | 57.14% | 50.00% | 45.24% | **+7.70 pp** |
| **Exp D1** | RawBoost Only (Convolutive & Additive Noise) | 69.23% | 66.67% | 80.00% | 57.14% | 83.33% | 30.95% | **+23.08 pp** |
| **Exp D2** | **Room Impulse Reverberation Only (RIR)** | **84.62%** | **87.50%** | **77.78%** | **100.00%** | **66.67%** | **23.81%** | **+38.47 pp (WINNER)** |
| **Exp D3** | Additive Gaussian Noise Only (15–30 dB SNR) | 61.54% | 66.67% | 66.67% | 71.43% | 50.00% | 35.71% | **+15.39 pp** |

### Key Scientific Findings:
1. **The Zero-Shot Out-of-Domain Failure:** The zero-shot baseline failed on authentic room recordings (16.67% Specificity) because the ASVspoof 2019 dataset was recorded in pristine anechoic chambers; real-world room reflections were falsely flagged as vocoder artifacts.
2. **Reverberation Invariance (Exp D2):** RIR augmentation taught the model to separate room acoustics from vocoder phase artifacts, delivering a **+38.47 pp gain** and catching **100% of deepfakes**.
3. **Scaling to Production:** These micro-ablation insights directly informed the design of the **VoiceGuard Unified Ensemble**, which achieved **99.85% accuracy on 2,000 held-out test clips** when scaled to the 13,000-sample cloud training pipeline.

---

## 6. Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Niriksha12-mpm/VoiceGuard-AI.git voiceguard-ai
cd voiceguard-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Create and activate Python virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies (Python 3.10 to 3.14 compatible)
pip install -r requirements.txt
```

### 3. Run Automated Test Suite
```bash
python -m pytest tests -v
```

### 4. Launch Application (FastAPI Backend + Production React UI)
```bash
# Start backend server on port 8000
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at `http://127.0.0.1:8000/` to access the full interactive forensic suite.  
Interactive OpenAPI Swagger docs are available at `http://127.0.0.1:8000/docs`.

---

## 7. REST API Reference

| Method | Endpoint | Description | Key Parameters / Body |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/predict` | Upload audio and receive deepfake classification | `file` (multipart), `model_type` (`voiceguard_unified` or `reference_baseline`), `channel_sim` (`clean`, `whatsapp_opus`, `cellular_telephony`, `voip_meeting`) |
| `GET` | `/api/models` | Specifications, parameter counts, and thresholds of registered models | None |
| `GET` | `/api/channels` | List available transmission channel degradation simulations | None |
| `POST` | `/api/simulate-channel` | Apply channel degradation to audio without running prediction | `file` (multipart), `channel` (string) |
| `POST` | `/api/analyze-temporal` | Temporal sliding-window splicing timeline analysis | `file` (multipart), `window_size` (float), `hop_size` (float) |
| `GET` | `/api/colab-metrics` | Primary large-scale benchmark results (13,000 samples, 2,000 test clips) | None |
| `GET` | `/api/large-scale-report`| Comprehensive cloud GPU training and convergence report | None |
| `GET` | `/api/phase3-experiments`| Controlled micro-ablation study results (Exp A to Exp D3) | None |
| `GET` | `/api/w2v2-baseline` | Zero-shot baseline evaluation report (Tak et al. Odyssey 2022) | None |
| `GET` | `/api/comparison` | Comparative metrics between baseline and adapted models | None |
| `GET` | `/api/sample-audio/{id}` | Download verified preset audio sample (`real_lj`, `real_yt`, `fake_elevenlabs`, `fake_aria`) | `id` (path parameter) |

---

## 8. Repository Directory Structure

```
voiceguard-ai/
├── backend/
│   ├── app.py                         # FastAPI application routes & static UI serving
│   └── inference.py                   # Production inference engine (Unified Ensemble & Baseline)
├── dataset/
│   ├── preprocessor.py                # 16kHz mono normalization, VAD, and audio slicing
│   ├── split_data.py                  # Zero-leakage GroupKFold dataset splitting
│   ├── audit_inventory.py             # Dataset verification & checksum auditing
│   ├── deduplication.py               # Audio cosine deduplication
│   └── download_datasets.py           # Dataset streaming downloader
├── frontend/
│   ├── build/                         # Compiled production React frontend bundle
│   ├── src/
│   │   ├── components/
│   │   │   ├── AudioUploader.jsx      # Audio drag-and-drop, mic recording & model selector
│   │   │   ├── ModelComparisonView.jsx# Side-by-side metrics & ablation study dashboard
│   │   │   ├── MetricsDashboard.jsx   # 2,000-clip benchmark & training convergence curves
│   │   │   ├── PredictionResult.jsx   # Verdict, confidence badge, waveform & diagnostics
│   │   │   ├── DatasetInventoryView.jsx# Multi-dataset inventory & audit statistics
│   │   │   └── Navbar.jsx             # Navigation bar with live backend status indicator
│   │   ├── App.jsx                    # Root application component
│   │   └── index.css                  # Tailwind CSS styling
│   └── package.json                   # React project dependencies
├── models/
│   ├── architecture.py                # Four-Block 2D CNN-BiLSTM model definition
│   ├── feature_extractor.py           # 67-dim composite feature extractor with CMN
│   ├── augmentation.py                # RawBoost, RIR reverberation, & noise suite
│   ├── channel_simulator.py           # WhatsApp Opus, Cellular G.711, VoIP channel degradation
│   ├── w2v2_aasist.py                 # ONNX inference wrapper for XLS-R 300M + AASIST
│   ├── train_w2v2_adaptation.py       # Phase 3 ablation experiment pipeline
│   └── saved/
│       ├── w2v2_aasist/
│       │   ├── w2v2-aasist.onnx       # Official 1.26 GB pretrained XLS-R + AASIST model
│       │   ├── fine_tuned_exp_d2_reverb.pt   # Best adapted weights (84.62% Acc)
│       │   └── voiceguard_w2v2_large_scale_model.pt # 13,000-sample trained weights (99.85% Acc)
│       ├── pretrained_model_a.pt      # Four-Block CNN baseline
│       └── fine_tuned_model_b.pt      # Four-Block CNN fine-tuned
├── reports/
│   ├── large_scale_training_report.json # Primary benchmark report (2,000 test clips, 99.85%)
│   ├── colab_evaluation_report.json   # Colab evaluation report
│   ├── phase3_training_experiments.json # N=13 micro-ablation experiment metrics
│   ├── w2v2_aasist_baseline.json      # Zero-shot baseline report
│   ├── model_comparison.json          # CNN comparative metrics
│   └── plots/                         # Generated ROC curves and confusion matrices
├── tests/
│   ├── test_api_endpoints.py          # FastAPI endpoint integration tests
│   ├── test_checkpoint_loading.py     # Checkpoint verification tests
│   ├── test_model_architecture.py     # Model shape & loss tests
│   ├── test_preprocessing.py          # Audio preprocessor & VAD tests
│   └── test_w2v2_aasist.py            # XLS-R + AASIST wrapper & windowing tests
├── colab_w2v2_large_scale_training.ipynb # Google Colab interactive training notebook
├── colab_w2v2_streaming_cloud.py      # Standalone cloud GPU streaming trainer
├── requirements.txt                   # Pinned Python package dependencies
└── README.md                          # Comprehensive system documentation
```

---

## 9. Academic Citations & Acknowledgments

- **AASIST & SSL Anti-Spoofing:** Tak, H., et al. *"End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification and Anti-Spoofing."* Odyssey 2022. Checkpoint: [SpeechAntiSpoofingBenchmarks/W2V2-AASIST](https://huggingface.co/SpeechAntiSpoofingBenchmarks/W2V2-AASIST).
- **Survey on Audio Deepfake Detection:** Müller, N., et al. ACM Computing Surveys, 2025. DOI: [10.1145/3714458](https://doi.org/10.1145/3714458).
- **WaveFake Dataset:** Frank, J., & Schönherr, L. *"WaveFake: A Data Set to Facilitate Audio Deepfake Detection."* NeurIPS 2021.
- **ASVspoof Consortium:** Yamagishi, J., et al. *"ASVspoof 2019/2021: The Audio Deepfake and Anti-Spoofing Challenge."*

---

## 10. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
