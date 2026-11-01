"""
Learnable Multi-Layer Self-Supervised (SSL) Representation Aggregation
for Wav2Vec 2.0 (XLS-R 300M) Anti-Spoofing Fine-Tuning.

References:
1. Wang et al., "Investigating Self-Supervised Front-Ends for Speech Anti-Spoofing", ASVspoof 2021.
2. Tak et al., "Automatic Speaker Verification Spoofing and Deepfake Detection Using SSL Representation Aggregation", 2022.
"""

from typing import List, Dict, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSLLayerAggregation(nn.Module):
    """
    Learnable Softmax Scalar Layer Weighting Module.
    Dynamically fuses hidden states across all transformer layers of the SSL backbone
    to weight acoustic/phase artifact layers (Layers 1-12) over high-level semantic layers (Layers 13-24).

    Args:
        num_layers: Total number of transformer layers (24 for XLS-R 300M, or custom K).
        temperature: Temperature parameter for softmax scaling (default 1.0).
        projection_dim: Optional projection dimension. If specified, applies linear projection post-aggregation.
    """
    def __init__(
        self,
        num_layers: int = 24,
        temperature: float = 1.0,
        projection_dim: Optional[int] = None,
        in_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = float(temperature)

        # Learnable layer scalar logits initialized uniformly
        self.weights = nn.Parameter(torch.zeros(num_layers))

        if projection_dim is not None and in_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.GELU()
            )
        else:
            self.projection = None

    def get_normalized_weights(self) -> Dict[str, float]:
        """Return human-readable dictionary of normalized layer contribution weights."""
        norm_weights = F.softmax(self.weights / self.temperature, dim=0).detach().cpu().numpy()
        return {f"layer_{i + 1}": round(float(w), 4) for i, w in enumerate(norm_weights)}

    def forward(
        self,
        layer_hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute weighted sum of layer representations.
        Args:
            layer_hidden_states:
                - Tensor of shape [batch_size, num_layers, seq_len, hidden_dim] OR [batch_size, num_layers, hidden_dim]
                - OR List of num_layers Tensors of shape [batch_size, ...]
        Returns:
            Fused representation tensor of shape [batch_size, ...].
        """
        # Calculate normalized scalar weights
        norm_weights = F.softmax(self.weights / self.temperature, dim=0)

        if isinstance(layer_hidden_states, list):
            # List of tensors
            assert len(layer_hidden_states) == self.num_layers, (
                f"Expected {self.num_layers} layer tensors, got {len(layer_hidden_states)}"
            )
            fused = sum(w * h for w, h in zip(norm_weights, layer_hidden_states))
        elif isinstance(layer_hidden_states, torch.Tensor):
            # Tensor with layer dimension at index 1
            assert layer_hidden_states.size(1) == self.num_layers, (
                f"Expected size(1)=={self.num_layers}, got {layer_hidden_states.size(1)}"
            )
            # Reshape weights to broadcast across remaining dimensions
            shape = [1, self.num_layers] + [1] * (layer_hidden_states.dim() - 2)
            w_broadcast = norm_weights.view(shape)
            fused = (layer_hidden_states * w_broadcast).sum(dim=1)
        else:
            raise TypeError(f"Unsupported input type for layer_hidden_states: {type(layer_hidden_states)}")

        if self.projection is not None:
            fused = self.projection(fused)

        return fused
