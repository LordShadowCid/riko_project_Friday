"""
RVC (Retrieval-based Voice Conversion) post-processor.
Converts GPT-SoVITS output WAV through a trained RVC model for better voice fidelity.

Disabled by default (rvc.enabled: false in character_config.yaml).
Requires `rvc-infer` or compatible RVC inference package.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from server.annabeth_config import load_config, resolve_repo_path

logger = logging.getLogger(__name__)

_rvc_converter: Optional["RvcConverter"] = None
_rvc_initialized = False


class RvcConverter:
    """
    Thin wrapper around RVC's infer pipeline.
    Falls back silently if the rvc package is not installed.
    """

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
    ):
        self.model_path = str(model_path)
        # .index file is optional — improves voice similarity when present (MekaHime pattern)
        self.index_path = str(index_path) if index_path else ""
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self._pipeline = None
        self._load()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        """Try to import and initialize the RVC infer pipeline."""
        try:
            # rvc-infer exposes these after: pip install rvc-infer
            from rvc.infer.infer import VoiceConverter  # type: ignore
            self._pipeline = VoiceConverter()
            self._pipeline.load_model(self.model_path)
            logger.info("[RVC] Model loaded from %s", self.model_path)
            # Check rmvpe availability; fall back to harvest if not present
            rmvpe_path = Path(self.model_path).parent / "rmvpe.pt"
            if self.f0_method == "rmvpe" and not rmvpe_path.exists():
                logger.warning(
                    "[RVC] rmvpe.pt not found at %s — falling back to harvest. "
                    "Download from: https://huggingface.co/lj1995/VoiceConversionWebUI",
                    rmvpe_path,
                )
                self.f0_method = "harvest"
        except ImportError:
            logger.warning(
                "[RVC] rvc-infer package not installed — RVC conversion disabled. "
                "Install with: pip install rvc-infer"
            )
            self._pipeline = None
        except Exception as exc:
            logger.warning("[RVC] Could not load model: %s", exc)
            self._pipeline = None

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def convert(self, wav_path: str) -> str:
        """
        Convert *wav_path* through RVC and overwrite it in-place.

        Returns the same path so callers don't need special handling.
        Falls back to returning the original if RVC is unavailable.
        """
        if not self.available:
            return wav_path

        src = Path(wav_path)
        if not src.exists():
            logger.warning("[RVC] Source file not found: %s", wav_path)
            return wav_path

        tmp_out = src.with_suffix(".rvc_tmp.wav")

        try:
            self._pipeline.convert(
                input_path=str(src),
                output_path=str(tmp_out),
                f0up_key=self.pitch_shift,
                f0method=self.f0_method,
                # Pass index path for improved voice similarity (optional)
                index_path=self.index_path if self.index_path else None,
                index_rate=0.8 if self.index_path else 0.0,
                is_half=True,
                filter_radius=3,
                resample_sr=0,
                rms_mix_rate=0.8,
                protect=0.33,
            )

            if tmp_out.exists() and tmp_out.stat().st_size > 0:
                # Replace original with converted audio
                tmp_out.replace(src)
                logger.debug("[RVC] Converted %s (pitch_shift=%d)", src.name, self.pitch_shift)
            else:
                logger.warning("[RVC] Conversion produced empty output, keeping original")
                if tmp_out.exists():
                    tmp_out.unlink()

        except Exception as exc:
            logger.warning("[RVC] Conversion failed: %s — keeping original audio", exc)
            if tmp_out.exists():
                try:
                    tmp_out.unlink()
                except OSError:
                    pass

        return wav_path


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_rvc_converter() -> Optional[RvcConverter]:
    """
    Return the module-level RvcConverter singleton, or None if disabled / misconfigured.
    Safe to call repeatedly — initializes at most once.
    """
    global _rvc_converter, _rvc_initialized

    if _rvc_initialized:
        return _rvc_converter

    _rvc_initialized = True

    try:
        config = load_config()
        rvc_cfg = config.get("rvc", {})

        if not rvc_cfg.get("enabled", False):
            logger.debug("[RVC] Disabled in config (rvc.enabled: false)")
            _rvc_converter = None
            return None

        model_dir = Path(resolve_repo_path(rvc_cfg.get("model_dir", "character_files/rvc_models")))
        model_name = rvc_cfg.get("model_name", "annabeth")
        # Support both 'annabeth' and 'annabeth.pth' in config
        stem = model_name.removesuffix(".pth")
        model_path = model_dir / f"{stem}.pth"

        if not model_path.exists():
            logger.warning(
                "[RVC] Model file not found: %s — RVC disabled", model_path
            )
            _rvc_converter = None
            return None

        # Optional .index file alongside .pth (MekaHime pattern)
        index_path = model_dir / f"{stem}.index"
        index_path_str = str(index_path) if index_path.exists() else None
        if index_path_str:
            logger.info("[RVC] Index file found: %s", index_path)

        pitch_shift = int(rvc_cfg.get("pitch_shift", 0))
        f0_method = str(rvc_cfg.get("f0_method", "rmvpe"))

        _rvc_converter = RvcConverter(
            model_path=str(model_path),
            index_path=index_path_str,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
        )

        if not _rvc_converter.available:
            _rvc_converter = None

    except Exception as exc:
        logger.warning("[RVC] Initialisation error: %s — RVC disabled", exc)
        _rvc_converter = None

    return _rvc_converter
