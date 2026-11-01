"""
Automated Unit and Integration Tests for Advanced Fine-Tuning:
1. AMSoftmaxLoss (Additive Margin metric learning)
2. OCSoftmaxLoss (One-Class hypersphere metric learning)
3. SSLLayerAggregation (Multi-layer learnable SSL aggregation)
4. W2V2DomainAdaptationHead (Embedding bottleneck and AM-Softmax training loop)
"""

import numpy as np
import pytest
import torch

from models.am_softmax import AMSoftmaxLoss, OCSoftmaxLoss
from models.layer_aggregation import SSLLayerAggregation
from models.train_w2v2_adaptation import W2V2DomainAdaptationHead, train_adaptation_head


def test_am_softmax_loss_and_gradients():
    """Verify AM-Softmax computes valid loss and propagates gradients."""
    in_features = 32
    batch_size = 8
    criterion = AMSoftmaxLoss(in_features=in_features, num_classes=2, margin=0.35, scale=30.0)

    x = torch.randn(batch_size, in_features, requires_grad=True)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)

    loss, eval_logits = criterion(x, target)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert eval_logits.shape == (batch_size, 2)

    # Verify backpropagation
    loss.backward()
    assert x.grad is not None
    assert criterion.weight.grad is not None
    assert not torch.isnan(x.grad).any()


def test_am_softmax_margin_effect():
    """Verify that margin m strictly penalizes target class logits compared to zero-margin."""
    in_features = 16
    x = torch.randn(4, in_features)
    target = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    am_loss_with_m = AMSoftmaxLoss(in_features=in_features, num_classes=2, margin=0.35, scale=20.0)
    am_loss_zero_m = AMSoftmaxLoss(in_features=in_features, num_classes=2, margin=0.0, scale=20.0)
    # Align weights for exact comparison
    am_loss_zero_m.weight.data.copy_(am_loss_with_m.weight.data)

    loss_m, _ = am_loss_with_m(x, target)
    loss_0, _ = am_loss_zero_m(x, target)

    # Margin penalty forces the model to work harder -> higher initial cross entropy
    assert loss_m.item() > loss_0.item()


def test_oc_softmax_loss():
    """Verify One-Class Softmax computes valid non-negative loss."""
    in_features = 32
    oc_loss = OCSoftmaxLoss(in_features=in_features, r_real=0.9, r_fake=0.2, alpha=20.0)

    x = torch.randn(6, in_features, requires_grad=True)
    target = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    loss, cosine_scores = oc_loss(x, target)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
    assert cosine_scores.shape == (6,)

    loss.backward()
    assert x.grad is not None
    assert oc_loss.w_real.grad is not None


def test_ssl_layer_aggregation_tensor_and_weights():
    """Verify SSLLayerAggregation forward pass across stacked layers and weight normalization."""
    num_layers = 24
    hidden_dim = 64
    batch_size = 4
    agg = SSLLayerAggregation(num_layers=num_layers, temperature=1.0)

    # Verify layer weights sum to 1.0
    weights_dict = agg.get_normalized_weights()
    assert len(weights_dict) == 24
    assert np.isclose(sum(weights_dict.values()), 1.0, atol=1e-3)

    # Forward pass with tensor [batch, num_layers, hidden_dim]
    stacked_layers = torch.randn(batch_size, num_layers, hidden_dim, requires_grad=True)
    fused = agg(stacked_layers)

    assert fused.shape == (batch_size, hidden_dim)
    assert not torch.isnan(fused).any()

    # Verify backward pass into layer scalar parameters
    loss = fused.sum()
    loss.backward()
    assert agg.weights.grad is not None
    assert stacked_layers.grad is not None


def test_ssl_layer_aggregation_list_input():
    """Verify SSLLayerAggregation accepts list of layer tensors."""
    num_layers = 12
    batch_size = 2
    hidden_dim = 32
    agg = SSLLayerAggregation(num_layers=num_layers)

    layers_list = [torch.randn(batch_size, hidden_dim) for _ in range(num_layers)]
    fused = agg(layers_list)
    assert fused.shape == (batch_size, hidden_dim)


def test_domain_adaptation_head_embedding():
    """Verify W2V2DomainAdaptationHead returns 32-dim embedding bottleneck when requested."""
    model = W2V2DomainAdaptationHead(in_features=66, hidden_dim=128, num_classes=2)
    x = torch.randn(5, 66)

    # Standard forward
    logits = model(x)
    assert logits.shape == (5, 2)

    # Bottleneck embedding forward
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (5, 2)
    assert emb.shape == (5, 32)


def test_train_adaptation_head_am_softmax_cycle():
    """Verify train_adaptation_head successfully completes an AM-Softmax training & calibration cycle."""
    np.random.seed(42)
    n_train = 20
    n_val = 10
    train_X = np.random.randn(n_train, 66).astype(np.float32)
    train_y = np.array([0, 1] * (n_train // 2), dtype=int)
    val_X = np.random.randn(n_val, 66).astype(np.float32)
    val_y = np.array([0, 1] * (n_val // 2), dtype=int)

    model, threshold, history = train_adaptation_head(
        train_X, train_y, val_X, val_y,
        epochs=3, lr=1e-3, loss_type="am_softmax", am_margin=0.25
    )

    assert isinstance(model, W2V2DomainAdaptationHead)
    assert hasattr(model, "metric_head")
    assert 0.05 <= threshold <= 0.95
    assert len(history) == 3
    assert "train_loss" in history[0]
