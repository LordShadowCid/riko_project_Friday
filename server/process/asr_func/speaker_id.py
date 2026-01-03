"""
Speaker Identification module using Resemblyzer.
Creates voice embeddings for known speakers and identifies who is talking.
"""
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Tuple

# Lazy load resemblyzer to avoid import time
_encoder = None
_speaker_profiles: Dict[str, np.ndarray] = {}
_profiles_loaded = False


def _get_encoder():
    """Lazy-load the voice encoder (takes ~1-2s on first call)."""
    global _encoder
    if _encoder is None:
        from resemblyzer import VoiceEncoder
        # Use GPU if available, otherwise CPU
        _encoder = VoiceEncoder(device="cpu")  # CPU is fast enough for embeddings
        print("[Speaker ID] Voice encoder loaded")
    return _encoder


def _get_profiles_dir() -> Path:
    """Get the speaker profiles directory."""
    # Store profiles alongside character files
    base_dir = Path(__file__).parent.parent.parent.parent  # Go up to Anabeth root
    profiles_dir = base_dir / "speaker_profiles"
    profiles_dir.mkdir(exist_ok=True)
    return profiles_dir


def load_speaker_profiles() -> Dict[str, np.ndarray]:
    """Load all saved speaker profiles from disk."""
    global _speaker_profiles, _profiles_loaded
    
    if _profiles_loaded:
        return _speaker_profiles
    
    profiles_dir = _get_profiles_dir()
    _speaker_profiles = {}
    
    for profile_file in profiles_dir.glob("*.npy"):
        speaker_name = profile_file.stem
        embedding = np.load(profile_file)
        _speaker_profiles[speaker_name] = embedding
        print(f"[Speaker ID] Loaded profile for: {speaker_name}")
    
    if not _speaker_profiles:
        print("[Speaker ID] No speaker profiles found. Run enroll_speaker.py to create them.")
    
    _profiles_loaded = True
    return _speaker_profiles


def save_speaker_profile(name: str, embedding: np.ndarray):
    """Save a speaker profile to disk."""
    global _speaker_profiles
    
    profiles_dir = _get_profiles_dir()
    profile_path = profiles_dir / f"{name}.npy"
    np.save(profile_path, embedding)
    _speaker_profiles[name] = embedding
    print(f"[Speaker ID] Saved profile for: {name}")


def create_embedding_from_audio(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Create a voice embedding from audio data.
    
    Args:
        audio: Audio samples as numpy array (float32, mono)
        sample_rate: Sample rate of the audio (must be 16000)
    
    Returns:
        256-dimensional embedding vector
    """
    encoder = _get_encoder()
    
    # Resemblyzer expects float audio normalized to [-1, 1]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    # Audio should already be 16000Hz - skip librosa resampling
    # Just pass directly to encoder (avoids numba/librosa dependency issues)
    if sample_rate != 16000:
        raise ValueError(f"Audio must be 16000Hz, got {sample_rate}Hz")
    
    # Create embedding directly
    embedding = encoder.embed_utterance(audio)
    
    return embedding


def create_embedding_from_file(audio_path: str) -> np.ndarray:
    """Create a voice embedding from an audio file."""
    audio, sr = sf.read(audio_path)
    
    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Audio must be 16000Hz for our embedding function
    if sr != 16000:
        raise ValueError(f"Audio must be 16000Hz, got {sr}Hz")
    
    return create_embedding_from_audio(audio, 16000)


def identify_speaker(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.75,
    min_duration: float = 1.5
) -> Tuple[Optional[str], float]:
    """
    Identify which known speaker the audio belongs to.
    
    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate of the audio
        threshold: Minimum similarity score to consider a match (0-1)
        min_duration: Minimum audio duration in seconds for reliable ID
    
    Returns:
        Tuple of (speaker_name or None, similarity_score)
        Returns (None, 0.0) if no speaker matches above threshold
    """
    # Check minimum audio length for reliable identification
    duration = len(audio) / sample_rate
    if duration < min_duration:
        print(f"[Speaker ID] Audio too short ({duration:.1f}s < {min_duration}s), skipping")
        return None, 0.0
    
    profiles = load_speaker_profiles()
    
    if not profiles:
        return None, 0.0
    
    # Create embedding for the input audio
    try:
        embedding = create_embedding_from_audio(audio, sample_rate)
    except Exception as e:
        print(f"[Speaker ID] Error creating embedding: {e}")
        return None, 0.0
    
    # Compare against all known speakers
    best_match = None
    best_score = 0.0
    
    for name, profile_embedding in profiles.items():
        # Cosine similarity
        similarity = np.dot(embedding, profile_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(profile_embedding)
        )
        
        if similarity > best_score:
            best_score = similarity
            best_match = name
    
    # Only return a match if above threshold
    if best_score >= threshold:
        return best_match, float(best_score)
    else:
        return None, float(best_score)


def identify_speaker_from_file(
    audio_path: str,
    threshold: float = 0.75
) -> Tuple[Optional[str], float]:
    """Identify speaker from an audio file."""
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convert stereo to mono
    
    return identify_speaker(audio, sr, threshold)


# Convenience function for the ASR pipeline
def get_speaker_name(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """
    Get the speaker name for the audio, or "Unknown" if not recognized.
    This is the main function to call from the ASR pipeline.
    """
    speaker, score = identify_speaker(audio, sample_rate)
    
    if speaker:
        print(f"[Speaker ID] Identified: {speaker} (confidence: {score:.2f})")
        return speaker
    else:
        print(f"[Speaker ID] Unknown speaker (best score: {score:.2f})")
        return "Unknown"
