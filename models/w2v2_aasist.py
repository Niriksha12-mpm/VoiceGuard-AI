"""
Pretrained XLS-R (Wav2Vec 2.0 300M) + AASIST Speech Anti-Spoofing Classifier.
Reference Paper: Tak et al., "Automatic speaker verification spoofing and deepfake
detection using wav2vec 2.0 and data augmentation", Odyssey 2022.
Reference Implementation: TakHemlata/SSL_Anti-spoofing
Official Pretrained Checkpoint: SpeechAntiSpoofingBenchmarks/W2V2-AASIST

Input: Raw 16 kHz mono waveform windowed to 64,600 samples (~4.0375 seconds).
Padding: Tile-repeated if shorter than 64,600 samples, truncated to first 64,600 samples if longer.
Output: 2-class logits [spoof, bona_fide].
Score: logit[1] (higher = more bona fide / genuine).
"""

import os
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import librosa
import soundfile as sf

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def pad_fixed(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    """
    Pad or truncate 1D audio waveform to exactly max_len samples.
    Deterministic protocol matching ASVspoof evaluation (TakHemlata/SSL_Anti-spoofing).
    If audio length >= max_len: take the first max_len samples.
    If audio length < max_len: repeat the audio tile-wise until length >= max_len, then truncate.
    """
    x = np.asarray(x, dtype=np.float32).flatten()
    x_len = x.shape[0]
    if x_len == 0:
        return np.zeros(max_len, dtype=np.float32)
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, num_repeats)[:max_len]
    return padded_x.astype(np.float32)


def preprocess_waveform(
    audio: Union[np.ndarray, str, bytes],
    sr: Optional[int] = None,
    target_sr: int = 16000,
    target_len: int = 64600
) -> np.ndarray:
    """
    Standardize audio input to 16 kHz mono waveform of shape (64600,).
    Accepts:
    - np.ndarray (already decoded audio samples)
    - filepath (str)
    - raw audio bytes
    """
    if isinstance(audio, (str, bytes)):
        import io
        if isinstance(audio, str):
            y, file_sr = sf.read(audio, dtype="float32")
        else:
            y, file_sr = sf.read(io.BytesIO(audio), dtype="float32")
        sr = file_sr
    else:
        y = np.asarray(audio, dtype=np.float32)

    # Convert to mono if multi-channel
    if y.ndim > 1:
        if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
            y = np.mean(y, axis=0)
        elif y.shape[1] <= 8:
            y = np.mean(y, axis=1)
        else:
            y = y.flatten()

    # Resample to 16 kHz if necessary
    if sr is not None and sr != target_sr and len(y) > 0:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

    # Apply deterministic 64600-sample window
    return pad_fixed(y, max_len=target_len)


class W2V2AASISTClassifier:
    """
    Inference engine for Pretrained Wav2Vec 2.0 (XLS-R 300M) + AASIST.
    Executes high-performance ONNX Runtime inference on CPU or CUDA.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        decision_threshold: float = 0.0,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize the W2V2-AASIST classifier.
        Args:
            model_path: Path to w2v2-aasist.onnx
            decision_threshold: Threshold on the bona fide logit (score). Default: 0.0
                               (since score = logit[1] - logit[0] or raw logit[1]).
            providers: ONNX execution providers (e.g. ['CPUExecutionProvider']).
        """
        if not HAS_ORT:
            raise ImportError("onnxruntime is required for W2V2AASISTClassifier. Run `pip install onnxruntime`.")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "saved", "w2v2_aasist", "w2v2-aasist.onnx")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"W2V2-AASIST model file not found at: {model_path}\n"
                f"Please ensure w2v2-aasist.onnx is downloaded to models/saved/w2v2_aasist/."
            )

        self.model_path = model_path
        self.decision_threshold = float(decision_threshold)

        if providers is None:
            available_providers = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in available_providers]
            if not providers:
                providers = ['CPUExecutionProvider']

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        print(f"[W2V2AASISTClassifier] Loaded session from {self.model_path}")
        print(f"  Input name: '{self.input_name}', shape: {self.input_shape}")
        print(f"  Output name: '{self.output_name}', Decision threshold: {self.decision_threshold:.4f}")

    def predict_batch_raw(self, batch_waveforms: np.ndarray) -> np.ndarray:
        """
        Execute forward pass on a batch of waveforms of shape (N, 64600).
        Returns raw 2-class logits array of shape (N, 2).
        """
        batch_waveforms = np.asarray(batch_waveforms, dtype=np.float32)
        if batch_waveforms.ndim == 1:
            batch_waveforms = np.expand_dims(batch_waveforms, axis=0)

        # Ensure correct sample length
        if batch_waveforms.shape[1] != 64600:
            padded = [pad_fixed(w, 64600) for w in batch_waveforms]
            batch_waveforms = np.stack(padded, axis=0)

        outputs = self.session.run([self.output_name], {self.input_name: batch_waveforms})
        return outputs[0]  # shape (N, 2)

    def get_score(self, audio: Union[np.ndarray, str, bytes], sr: Optional[int] = None) -> float:
        """
        Compute ASVspoof bona fide score for a single audio file or waveform.
        Score = output logit for class 1 (bona fide). Higher = more genuine.
        """
        waveform = preprocess_waveform(audio, sr=sr, target_sr=16000, target_len=64600)
        batch = np.expand_dims(waveform, axis=0)
        logits = self.predict_batch_raw(batch)[0]  # shape (2,)
        # score = logit[1] (bona fide class score)
        return float(logits[1])

    def predict_proba(
        self,
        audio: Union[np.ndarray, str, bytes],
        sr: Optional[int] = None,
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Run inference and return structured dictionary with probabilities and prediction.
        """
        waveform = preprocess_waveform(audio, sr=sr, target_sr=16000, target_len=64600)
        batch = np.expand_dims(waveform, axis=0)
        logits = self.predict_batch_raw(batch)[0]  # [spoof_logit, bonafide_logit]

        # Softmax probabilities over [spoof, bona_fide]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        prob_spoof = float(probs[0])
        prob_bonafide = float(probs[1])

        # Bona fide score (raw logit for class 1)
        score = float(logits[1])
        # Margin: logit[1] - logit[0] (relative bona fide log-odds)
        logit_diff = float(logits[1] - logits[0])

        threshold = custom_threshold if custom_threshold is not None else self.decision_threshold
        # Prediction decision:
        # A file is classified as GENUINE (bona fide) if its score exceeds threshold
        is_genuine = bool(score >= threshold)
        predicted_label = "REAL" if is_genuine else "FAKE"
        confidence = prob_bonafide if is_genuine else prob_spoof

        return {
            "prediction": predicted_label,
            "is_genuine": is_genuine,
            "confidence": confidence,
            "score": score,
            "logit_diff": logit_diff,
            "prob_bonafide": prob_bonafide,
            "prob_spoof": prob_spoof,
            "logits": [float(logits[0]), float(logits[1])],
            "threshold_used": threshold,
            "model_name": "XLS-R 300M + AASIST (SpeechAntiSpoofingBenchmarks)"
        }

    def calibrate_threshold(
        self,
        val_scores: np.ndarray,
        val_labels: np.ndarray,
        criterion: str = "youden"
    ) -> float:
        """
        Calibrate decision threshold on validation set scores.
        Args:
            val_scores: 1D array of bona fide scores (higher = bona fide)
            val_labels: 1D array of binary ground-truth labels (1 = bona fide / REAL, 0 = spoof / FAKE)
            criterion: 'youden' (maximize J = TPR - FPR) or 'eer' (equal error rate)
        Returns:
            Calibrated optimal threshold value.
        """
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(val_labels, val_scores, pos_label=1)

        if criterion == "youden":
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            self.decision_threshold = float(thresholds[best_idx])
        elif criterion == "eer":
            fnr = 1.0 - tpr
            eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
            self.decision_threshold = float(thresholds[eer_idx])
        else:
            raise ValueError(f"Unknown criterion '{criterion}'. Use 'youden' or 'eer'.")

        print(f"[W2V2AASISTClassifier] Calibrated threshold ({criterion}): {self.decision_threshold:.4f}")
        return self.decision_threshold
