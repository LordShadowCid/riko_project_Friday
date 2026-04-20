"""
RVC (Retrieval-based Voice Conversion) post-processor.
Converts GPT-SoVITS output WAV through a trained RVC model for better voice fidelity.

Disabled by default (rvc.enabled: false in character_config.yaml).
Uses rvc_infer (installed with --no-deps) + a transformers-based HuBERT shim
to avoid the fairseq dependency hell on Python 3.13.

Supports runtime voice switching via switch_voice(name) and list_voices().
"""

import os
import gc
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from server.annabeth_config import load_config, resolve_repo_path

logger = logging.getLogger(__name__)

_rvc_converter: Optional["RvcConverter"] = None
_rvc_initialized = False
_model_dir: Optional[Path] = None


class RvcConverter:
    """
    Thin wrapper around rvc_infer's VC class.
    Supports hot-swapping voice models at runtime.
    """

    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        pitch_shift: int = 0,
        f0_method: str = "harvest",
    ):
        self.model_path = str(model_path)
        self.index_path = str(index_path) if index_path else ""
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self._vc = None
        self._config = None
        self._lock = threading.Lock()
        self._current_voice = Path(model_path).stem
        self._load()

    def _load(self):
        """Try to import and initialize the RVC pipeline."""
        try:
            from server.process.tts_func.rvc_hubert_compat import patch_rvc_infer
            patch_rvc_infer()

            from rvc_infer.infer import Configs
            from rvc_infer.modules import VC

            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                device = "cuda:1"
            elif torch.cuda.is_available():
                device = "cuda:0"
            else:
                device = "cpu"

            self._config = Configs(device, is_half=True)
            self._vc = VC(self._config)
            self._vc.get_vc(self.model_path, 0.33, 0.5)
            logger.info("[RVC] Model loaded from %s on %s", self.model_path, device)

        except ImportError as e:
            logger.warning("[RVC] Missing dependency: %s — RVC disabled", e)
            self._vc = None
        except Exception as exc:
            logger.warning("[RVC] Could not load model: %s", exc)
            self._vc = None

    @property
    def available(self) -> bool:
        return self._vc is not None

    @property
    def current_voice(self) -> str:
        return self._current_voice

    def switch_model(self, model_path: str, index_path: Optional[str] = None) -> bool:
        """
        Hot-swap the loaded RVC model. Thread-safe.
        Returns True on success, False on failure (keeps previous model).
        """
        if not self._vc:
            return False

        with self._lock:
            old_model = self.model_path
            old_index = self.index_path
            try:
                self._vc.get_vc(model_path, 0.33, 0.5)
                self.model_path = model_path
                self.index_path = str(index_path) if index_path else ""
                self._current_voice = Path(model_path).stem
                gc.collect()
                logger.info("[RVC] Switched voice to %s", self._current_voice)
                return True
            except Exception as exc:
                logger.warning("[RVC] Switch failed (%s), restoring previous model", exc)
                try:
                    self._vc.get_vc(old_model, 0.33, 0.5)
                except Exception:
                    self._vc = None
                return False

    def convert(self, wav_path: str) -> str:
        """
        Convert *wav_path* through RVC and overwrite it in-place.
        Returns the same path. Falls back to original if RVC is unavailable.
        """
        if not self.available:
            return wav_path

        src = Path(wav_path)
        if not src.exists():
            logger.warning("[RVC] Source file not found: %s", wav_path)
            return wav_path

        with self._lock:
            try:
                import soundfile as sf

                (info, audio_result) = self._vc.vc_single_dont_save(
                    sid=0,
                    input_audio_path1=str(src),
                    f0_up_key=self.pitch_shift,
                    f0_method=self.f0_method,
                    file_index=self.index_path,
                    file_index2=self.index_path,
                    index_rate=0.8 if self.index_path else 0.0,
                    filter_radius=3,
                    resample_sr=0,
                    rms_mix_rate=0.8,
                    protect=0.33,
                    crepe_hop_length=128,
                    do_formant=False,
                    quefrency=0,
                    timbre=1,
                    f0_min="50",
                    f0_max="1100",
                    f0_autotune=False,
                    hubert_model_path="hubert_base.pt",
                )

                status = info[0] if isinstance(info, (list, tuple)) else str(info)
                if status == "Success." and audio_result and audio_result[0] is not None:
                    sr, audio_data = audio_result
                    sf.write(str(src), audio_data, sr)
                    logger.debug("[RVC] Converted %s (pitch_shift=%d)", src.name, self.pitch_shift)
                else:
                    logger.warning("[RVC] Conversion status: %s — keeping original", status)

            except Exception as exc:
                logger.warning("[RVC] Conversion failed: %s — keeping original audio", exc)

        return wav_path


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

def get_rvc_converter() -> Optional[RvcConverter]:
    """
    Return the module-level RvcConverter singleton, or None if disabled / misconfigured.
    Safe to call repeatedly — initializes at most once.
    """
    global _rvc_converter, _rvc_initialized, _model_dir

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

        _model_dir = Path(resolve_repo_path(rvc_cfg.get("model_dir", "character_files/rvc_models")))
        model_name = rvc_cfg.get("model_name", "annabeth")
        # Support both 'annabeth' and 'annabeth.pth' in config
        stem = model_name.removesuffix(".pth")
        model_path = _model_dir / f"{stem}.pth"

        if not model_path.exists():
            logger.warning(
                "[RVC] Model file not found: %s — RVC disabled", model_path
            )
            _rvc_converter = None
            return None

        # Optional .index file alongside .pth (MekaHime pattern)
        index_path = _model_dir / f"{stem}.index"
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


# ---------------------------------------------------------------------------
# Voice switching helpers
# ---------------------------------------------------------------------------

def list_voices() -> List[Dict[str, str]]:
    """
    Return a list of available voice models discovered in model_dir.
    Each entry: {"name": "annabeth", "label": "Annabeth (default)", "active": true/false}
    Labels come from character_config.yaml rvc.voices if defined,
    otherwise auto-generated from the .pth filename.
    """
    config = load_config()
    rvc_cfg = config.get("rvc", {})
    model_dir = Path(resolve_repo_path(rvc_cfg.get("model_dir", "character_files/rvc_models")))
    voice_cfg = rvc_cfg.get("voices", {})

    converter = get_rvc_converter()
    active_voice = converter.current_voice if converter else ""

    voices = []
    seen = set()
    for pth in sorted(model_dir.glob("*.pth")):
        name = pth.stem
        if name in seen:
            continue
        seen.add(name)
        label = name.replace("_", " ").title()
        if name in voice_cfg and "label" in voice_cfg[name]:
            label = voice_cfg[name]["label"]
        voices.append({
            "name": name,
            "label": label,
            "active": name == active_voice,
        })
    return voices


def switch_voice(voice_name: str) -> Dict[str, object]:
    """
    Switch the active RVC voice model by name (stem without .pth).
    Returns {"ok": bool, "voice": str, "error": str|None}
    """
    converter = get_rvc_converter()
    if converter is None:
        return {"ok": False, "voice": "", "error": "RVC not available"}

    config = load_config()
    rvc_cfg = config.get("rvc", {})
    model_dir = Path(resolve_repo_path(rvc_cfg.get("model_dir", "character_files/rvc_models")))

    stem = voice_name.removesuffix(".pth")
    pth_path = model_dir / f"{stem}.pth"
    if not pth_path.exists():
        return {"ok": False, "voice": converter.current_voice,
                "error": f"Model not found: {stem}.pth"}

    if converter.current_voice == stem:
        return {"ok": True, "voice": stem, "error": None}

    index_path = model_dir / f"{stem}.index"
    idx = str(index_path) if index_path.exists() else None

    ok = converter.switch_model(str(pth_path), idx)
    return {
        "ok": ok,
        "voice": converter.current_voice,
        "error": None if ok else "Model load failed",
    }
