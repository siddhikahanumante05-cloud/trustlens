"""WavLM-base-plus Attentive Head for voice-clone / audio spoof detection.
Weighted layer sum + attentive statistics pooling + MLP classifier trained on ASVspoof 2019 LA.
Evaluates 1-second audio windows (16,000 samples).
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentiveStatisticsPooling(nn.Module):
    """Computes attention-weighted mean and standard deviation over temporal audio frames."""

    def __init__(self, in_features: int, hidden_dim: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, D)
        Output: (B, 2*D) concatenated weighted mean and std
        """
        # (B, T, 1)
        weights = F.softmax(self.attn(x), dim=1)
        weighted_mean = torch.sum(weights * x, dim=1)  # (B, D)
        # Weighted variance
        weighted_var = torch.sum(weights * (x - weighted_mean.unsqueeze(1)) ** 2, dim=1)
        weighted_std = torch.sqrt(torch.clamp(weighted_var, min=1e-6))  # (B, D)
        return torch.cat([weighted_mean, weighted_std], dim=-1)  # (B, 2*D)


class WavLMSpoofHead(nn.Module):
    """
    Lightweight classification head operating on WavLM layer representations.
    Input: (B, 12, T, 768) layer hidden states or pooled (B, T, 768)
    Output: (B, 1) spoof probability logit
    """

    def __init__(self, embed_dim: int = 768, num_layers: int = 12):
        super().__init__()
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.asp = AttentiveStatisticsPooling(in_features=embed_dim, hidden_dim=64)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, layer_feats: torch.Tensor) -> torch.Tensor:
        """
        layer_feats: (B, 12, T, 768) or (B, T, 768)
        Output: (B, 1) spoof logit
        """
        if layer_feats.ndim == 4:
            # Weighted layer combination
            w = F.softmax(self.layer_weights, dim=0).view(1, self.num_layers, 1, 1)
            fused = torch.sum(w * layer_feats, dim=1)  # (B, T, 768)
        else:
            fused = layer_feats  # (B, T, 768)

        pooled = self.asp(fused)  # (B, 1536)
        logit = self.mlp(pooled)  # (B, 1)
        return logit


def export_wavlm_head_onnx(model: WavLMSpoofHead, output_path: str = "models/wavlm_head.onnx"):
    """Export WavLM Spoof Head to ONNX."""
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 1 second of audio at 50Hz frame rate produces ~50 frames
    dummy_input = torch.randn(1, 50, 768, dtype=torch.float32)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=17,
            input_names=["audio_features"],
            output_names=["spoof_logit"],
            dynamic_axes={"audio_features": {0: "batch_size", 1: "time_steps"}, "spoof_logit": {0: "batch_size"}},
        )
        print(f"WavLM Spoof Head ONNX exported to: {output_path}")
    except Exception as e:
        print(f"Note: ONNX export requires 'onnx' and 'onnxscript' (available in Kaggle/Colab GPU env): {e}")
        with torch.no_grad():
            out = model(dummy_input)
            print(f"WavLM Spoof Head PyTorch forward pass test passed. Output shape: {out.shape}")


if __name__ == "__main__":
    head = WavLMSpoofHead()
    export_wavlm_head_onnx(head)
