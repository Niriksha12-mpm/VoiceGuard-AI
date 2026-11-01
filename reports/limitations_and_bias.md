# Limitations and Potential Bias Analysis

## Project: Voice Impersonation Detection Using Machine Learning

---

### 1. Overview & Objective
This report provides an engineering and forensic audit of the **Voice Impersonation Detection System** (`VoiceGuard AI`). In accordance with strict experimental governance, this document details the architectural boundaries, potential demographic and acoustic biases, dataset constraints, and operational vulnerabilities of the machine learning models.

---

### 2. Dataset Distribution & Representation Constraints

#### 2.1 Language and Accent Representation
* **Current Corpus:** The primary training and evaluation corpora are composed of English speech samples from **LJSpeech** (single female speaker, standard North American reading accent) and **YouTube/Podcast interviews** (multi-speaker North American and British accents), paired with neural vocoders (WaveFake) and voice cloning platforms (ElevenLabs, Hume AI, Speechify).
* **Cross-Lingual Gap:** The model has not been extensively fine-tuned on tonal languages (e.g., Mandarin, Vietnamese) or highly inflectional dialects. When evaluating speech in non-English languages, the distribution of spectral centroids and pitch modulation can shift significantly, potentially degrading confidence calibration.
* **Mitigation:** Future iterations must ingest multi-lingual anti-spoofing benchmarks such as the multilingual ASVspoof extensions and Mozilla Common Voice subsets paired with multilingual TTS (e.g., Dhwani, SeamlessM4T).

#### 2.2 Gender and Speaker Acoustic Diversity
* **Gender Representation:** Both male and female voices are represented in the dataset (YouTube speakers, ElevenLabs Aria and Rachel clones). However, fundamental frequency ($F_0$) ranges differ substantially between male ($85 - 180\text{ Hz}$) and female voices ($165 - 255\text{ Hz}$).
* **Risk:** Models trained predominantly on higher-pitch voices risk overfitting to pitch-dependent harmonics rather than vocoder phase incoherence.
* **Mitigation:** Our feature extraction pipeline relies on 64-band Log-Mel filterbanks spanning $20\text{ Hz}$ to $8,000\text{ Hz}$, which captures spectral envelope shapes and high-frequency harmonics independent of speaker $F_0$.

---

### 3. Model Architecture & Domain Shift Limitations

#### 3.1 Neural Vocoder Generalization
* **In-Distribution vs. Out-of-Distribution Performance:**
  - **Model B (Fine-Tuned)** achieved **84.62% test accuracy** on held-out test audio from known vocoder families (WaveFake, ElevenLabs, Hume AI).
  - However, when presented with newly emergent voice cloning algorithms (e.g., zero-shot diffusion vocoders like Stable Audio or Flow-Matching models like Voicebox), detection accuracy may decrease because the specific high-frequency checkerboard artifacts differ.
* **Vocoder Signature Dependency:** Deepfake detection models can unintentionally overfit to the static spectral filters of specific training vocoders rather than the intrinsic biological traits of human vocal tract physiology.

#### 3.2 Pretrained Baseline (Model A) Failure Mode
* **Empirical Finding:** Model A (un-tuned baseline) achieved **46.15% accuracy** on the held-out test split, predicting all test samples as REAL (0% precision, 0% recall, 100% false negative rate).
* **Root Cause:** Audio neural networks trained solely on general audio classification (such as speaker identification or speech-to-text acoustic modeling) optimize for phonetic content while deliberately ignoring micro-level vocoder artifacts that humans cannot perceive. Without explicit fine-tuning on bonafide vs. synthetic contrasts, zero-shot audio models treat vocoded audio as legitimate speech.

---

### 4. Acoustic Environment & Channel Distortions

#### 4.1 Lossy Codec Compression
* Modern audio transmission over VoIP, cellular networks (GSM, AMR-WB), or messaging platforms (WhatsApp, Telegram) applies heavy lossy compression (Opus, AAC at 16–32 kbps).
* **Impact on Detection:** High-frequency compression rolls off acoustic energy above $7,000\text{ Hz}$, which is precisely where neural vocoder phase artifacts and metallic harmonic buzzing are most pronounced.
* **Recommendation:** Real-world deployments should augment training data with simulated codec compression (e.g., `torchaudio.transforms.Vad` and random MP3/Opus re-encoding).

#### 4.2 Room Acoustics and Reverberation
* Human recordings captured in reverberant spaces exhibit exponential room impulse response (RIR) decay, whereas synthetic speech rendered directly from vocoders often lacks natural spatial acoustics (unless post-processed with convolution reverb).
* **Risk:** The classifier could rely on environmental room ambience as a shortcut to classify audio as REAL, rather than analyzing vocal tract articulation.

---

### 5. Ethical Guidelines & Misuse Safeguards

1. **Non-Guarantee of Absolute Accuracy:** As shown on the web application dashboard, confidence scores are statistical estimates. No single classifier should be used as the sole deciding factor in criminal forensic analysis or legal identity determination.
2. **Biometric Security Defense:** In sensitive speaker verification systems (e.g., banking voice biometrics), voice impersonation detection must be combined with active liveness verification (such as randomized challenge-response phrase prompts).
3. **Data Leakage Safeguards:** In accordance with scientific reproducibility, all parent audio recordings and speaker identities were strictly partitioned into non-overlapping groups.
