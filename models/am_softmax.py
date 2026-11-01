"""
Additive Margin Softmax (AM-Softmax) and One-Class Softmax (OC-Softmax)
for VoiceGuard AI Anti-Spoofing Model Fine-Tuning.

References:
1. Wang et al., "CosFace: Large Margin Cosine Loss for Deep Feature Learning", CVPR 2018.
2. Tak et al., "End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification and Anti-Spoofing", Odyssey 2022.
3. Zhang et al., "One-Class Learning Towards Countermeasure for Voice Impersonation and Spoofing", IEEE/ACM TASLP 2021.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax (AM-Softmax / CosFace) Loss for Anti-Spoofing.
    Enforces a strict angular margin m between bona fide and spoofed voice embeddings
    on the unit hypersphere, shrinking intra-class variance and expanding inter-class separation.

    Args:
        in_features: Dimension of the input bottleneck embedding.
        num_classes: Number of target classes (2 for Real vs Fake).
        margin: Angular margin penalty m (default 0.35).
        scale: Hyper-spherical radius scaling factor s (default 30.0).
    """
    def __init__(
        self,
        in_features: int,
        num_classes: int = 2,
        margin: float = 0.35,
        scale: float = 30.0
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.m = float(margin)
        self.s = float(scale)

        # Weight matrix: [num_classes, in_features]
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute AM-Softmax loss and scaled cosine logits.
        Args:
            x: Input feature embeddings of shape [batch_size, in_features].
            target: Ground truth class indices [batch_size] (0: REAL, 1: FAKE).
        Returns:
            Tuple of (loss, eval_logits).
        """
        # 1. L2-normalize feature representations and class weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. Compute cosine similarity: cos(theta) = x_norm * w_norm.T
        cosine = F.linear(x_norm, w_norm)
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)

        # 3. Additive margin penalty applied strictly to target ground-truth class
        # phi(theta) = cos(theta) - m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1.0)

        margin_cosine = cosine - one_hot * self.m

        # 4. Scale by hyperspherical radius s
        scaled_logits = self.s * margin_cosine

        # 5. Cross-Entropy over scaled margin logits
        loss = F.cross_entropy(scaled_logits, target.long())

        # For validation / test evaluation, return un-penalized scaled cosine logits
        eval_logits = self.s * cosine
        return loss, eval_logits


class OCSoftmaxLoss(nn.Module):
    """
    One-Class Softmax (OC-Softmax) Loss for Anti-Spoofing.
    Models bona fide human voice as a compact hypersphere around a learned center w_real.
    All spoofed audio is driven away beyond margin radius r_fake.

    Args:
        in_features: Dimension of the input bottleneck embedding.
        r_real: Target cosine margin for genuine human voice (default 0.9).
        r_fake: Target cosine margin boundary for fake voice (default 0.2).
        alpha: Scale parameter (default 20.0).
    """
    def __init__(
        self,
        in_features: int,
        r_real: float = 0.9,
        r_fake: float = 0.2,
        alpha: float = 20.0
    ):
        super().__init__()
        self.in_features = in_features
        self.r_real = float(r_real)
        self.r_fake = float(r_fake)
        self.alpha = float(alpha)

        self.w_real = nn.Parameter(torch.randn(1, in_features))
        nn.init.xavier_normal_(self.w_real)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute OC-Softmax loss and cosine distance.
        Args:
            x: Input feature embeddings [batch_size, in_features].
            target: Ground truth indices [batch_size] (0: REAL, 1: FAKE).
        Returns:
            Tuple of (loss, cosine_scores).
        """
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.w_real, p=2, dim=1)
        cosine = torch.mm(x_norm, w_norm.t()).squeeze(-1)  # [batch_size]

        # For Real (0): penalize if cosine < r_real -> softplus(alpha * (r_real - cosine))
        # For Fake (1): penalize if cosine > r_fake -> softplus(alpha * (cosine - r_fake))
        score_real = self.alpha * (self.r_real - cosine)
        score_fake = self.alpha * (cosine - self.r_fake)

        loss_real = F.softplus(score_real)
        loss_fake = F.softplus(score_fake)

        loss = torch.where(target.long() == 0, loss_real, loss_fake).mean()
        return loss, cosine
