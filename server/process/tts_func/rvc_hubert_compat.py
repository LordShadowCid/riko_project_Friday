"""
HuBERT compatibility shim for RVC inference.

Replaces the fairseq-based HuBERT loading with transformers-based loading,
avoiding the fairseq->omegaconf dependency hell on Python 3.13.

The wrapper exposes the same interface that rvc_infer.pipeline expects:
  model.extract_features(source=..., padding_mask=..., output_layer=...)
  model.final_proj(features)  # only for v1
"""

import logging
import os
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HuBERTWrapper(nn.Module):
    """
    Wraps a HuggingFace HuBERT model to expose the fairseq-compatible interface
    expected by rvc_infer's pipeline.
    """

    def __init__(self, hf_model, feature_dim: int = 768):
        super().__init__()
        self.model = hf_model
        # v1 models use a final_proj to go from 768 -> 256
        self.final_proj = nn.Linear(feature_dim, 256)

    def extract_features(self, source, padding_mask=None, output_layer=12):
        """
        Args:
            source: (batch, seq_len) raw waveform at 16kHz
            padding_mask: (batch, seq_len) bool mask (unused — kept for compat)
            output_layer: which transformer layer to extract from (9 for v1, 12 for v2)
        Returns:
            (features, padding_mask) — features shape: (batch, time, hidden_dim)
        """
        with torch.no_grad():
            outputs = self.model(
                source,
                output_hidden_states=True,
            )
            # hidden_states[0] is the embedding, [1] is layer 1, etc.
            # output_layer=12 means layer 12 (the last for base model)
            hidden_states = outputs.hidden_states
            # Clamp to available layers
            layer_idx = min(output_layer, len(hidden_states) - 1)
            features = hidden_states[layer_idx]
        return features, padding_mask

    def eval(self):
        self.model.eval()
        return super().eval()

    def half(self):
        self.model.half()
        return super().half()

    def float(self):
        self.model.float()
        return super().float()

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)


def load_hubert_transformers(model_path_or_name: str, device="cpu", is_half=False):
    """
    Load HuBERT using transformers instead of fairseq.

    Args:
        model_path_or_name: path to a local fairseq .pt file (ignored — we use HF hub)
                           or HuggingFace model name
        device: torch device
        is_half: whether to use fp16
    Returns:
        HuBERTWrapper with fairseq-compatible interface
    """
    from transformers import HubertModel

    hf_name = "facebook/hubert-base-ls960"
    logger.info("[RVC/HuBERT] Loading from transformers: %s", hf_name)

    hf_model = HubertModel.from_pretrained(hf_name)
    hf_model = hf_model.to(device)

    wrapper = HuBERTWrapper(hf_model)
    wrapper = wrapper.to(device)

    if is_half:
        wrapper = wrapper.half()
    else:
        wrapper = wrapper.float()

    return wrapper.eval()


def patch_rvc_infer():
    """
    Monkey-patch rvc_infer.modules to use our transformers-based HuBERT loader
    instead of the fairseq-based one.

    Call this BEFORE importing VoiceConverter or VC from rvc_infer.
    """
    import importlib
    import types

    # Create a fake fairseq.checkpoint_utils module
    fake_fairseq = types.ModuleType("fairseq")
    fake_checkpoint_utils = types.ModuleType("fairseq.checkpoint_utils")

    def load_model_ensemble_and_task(filenames, suffix=""):
        """Fake fairseq loader — not actually used after our patch."""
        raise NotImplementedError(
            "fairseq is not installed. Use load_hubert_transformers instead."
        )

    fake_checkpoint_utils.load_model_ensemble_and_task = load_model_ensemble_and_task
    fake_fairseq.checkpoint_utils = fake_checkpoint_utils

    import sys
    sys.modules["fairseq"] = fake_fairseq
    sys.modules["fairseq.checkpoint_utils"] = fake_checkpoint_utils

    # Now import rvc_infer.modules (it will get our fake fairseq)
    import rvc_infer.modules as rvc_modules

    # Replace the load_hubert function with our transformers-based version
    _original_load_hubert = rvc_modules.load_hubert

    def patched_load_hubert(hubert_model_path, config):
        return load_hubert_transformers(
            hubert_model_path,
            device=config.device,
            is_half=config.is_half,
        )

    rvc_modules.load_hubert = patched_load_hubert
    logger.info("[RVC/HuBERT] Patched rvc_infer to use transformers HuBERT")

    # Patch Pipeline.__init__ to lazy-load RMVPE instead of eagerly in constructor.
    # The default Pipeline tries RMVPE(...) immediately, which requires rmvpe.pt.
    # We only need it if f0_method is rmvpe/rmvpe+, so defer loading.
    from rvc_infer.pipeline import Pipeline

    _original_pipeline_init = Pipeline.__init__

    def _patched_pipeline_init(self, tgt_sr, config):
        import os
        # Temporarily set env var if missing so the constructor doesn't KeyError
        _had_rmvpe = "rmvpe_model_path" in os.environ
        _had_fcpe = "fcpe_model_path" in os.environ
        if not _had_rmvpe:
            os.environ["rmvpe_model_path"] = "rmvpe.pt"
        if not _had_fcpe:
            os.environ["fcpe_model_path"] = "fcpe.pt"

        # Run original init but catch RMVPE load failure (file not found)
        try:
            _original_pipeline_init(self, tgt_sr, config)
        except Exception as exc:
            # If RMVPE model file doesn't exist, init everything except model_rmvpe
            logger.warning("[RVC/Pipeline] RMVPE eager load failed (%s), using lazy init", exc)
            self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
                config.x_pad, config.x_query, config.x_center, config.x_max, config.is_half,
            )
            self.sr = 16000
            self.window = 160
            self.t_pad = self.sr * self.x_pad
            self.t_pad_tgt = tgt_sr * self.x_pad
            self.t_pad2 = self.t_pad * 2
            self.t_query = self.sr * self.x_query
            self.t_center = self.sr * self.x_center
            self.t_max = self.sr * self.x_max
            self.device = config.device
            self.model_rmvpe = None  # Will be loaded on first use if needed

    Pipeline.__init__ = _patched_pipeline_init
    logger.info("[RVC/Pipeline] Patched Pipeline for lazy RMVPE loading")
