#!/usr/bin/env python3
"""
Speaker Enrollment Script for Annabeth Voice Assistant.

This script records voice samples and creates speaker profiles so Annabeth
can identify who is talking (e.g., distinguish between you and your daughter).

Usage:
    python enroll_speaker.py

The script will:
1. Ask for the speaker's name
2. Record multiple voice samples (about 5-10 seconds each)
3. Create and save a voice profile

WHAT TO SAY:
- Speak naturally, like you're having a conversation
- Say different things for each sample to capture voice variety
- Examples:
    "Hello Annabeth, I'm testing my voice profile today."
    "The quick brown fox jumps over the lazy dog."
    "How's the weather looking outside right now?"
    "I really love pizza and ice cream for dinner."
    "Can you tell me a joke or something funny?"
"""
import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path

# Add server to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configuration
SAMPLE_RATE = 16000
RECORDING_SECONDS = 8  # Seconds per sample
NUM_SAMPLES = 3  # Number of samples to record
SILENCE_BETWEEN = 1.5  # Pause between recordings


def get_input_device(config_path: str = "character_config.yaml") -> int:
    """Get the input device from config or use default."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device_name = config.get('audio', {}).get('input_device')
        if device_name:
            devices = sd.query_devices()
            for idx, d in enumerate(devices):
                if device_name.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                    print(f"Using input device: {d['name']}")
                    return idx
    except Exception as e:
        print(f"Could not load config, using default device: {e}")
    
    return None  # Use default


def record_audio(duration: float, device=None) -> np.ndarray:
    """Record audio for the specified duration."""
    print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(0.5)
    print("  GO! 🗣️")
    
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=device
    )
    sd.wait()
    
    print("  ✓ Recording complete!")
    return audio.flatten()


def play_audio(audio: np.ndarray, device=None):
    """Play back the recorded audio."""
    print("  🔊 Playing back...")
    sd.play(audio, SAMPLE_RATE, device=device)
    sd.wait()


def create_profile(samples: list) -> np.ndarray:
    """Create a speaker profile from multiple audio samples."""
    from server.process.asr_func.speaker_id import create_embedding_from_audio
    
    print("\n📊 Creating voice profile...")
    
    embeddings = []
    for i, audio in enumerate(samples):
        print(f"  Processing sample {i+1}/{len(samples)}...")
        embedding = create_embedding_from_audio(audio, SAMPLE_RATE)
        embeddings.append(embedding)
    
    # Average all embeddings for a robust profile
    profile = np.mean(embeddings, axis=0)
    
    # Normalize
    profile = profile / np.linalg.norm(profile)
    
    print("  ✓ Profile created!")
    return profile


def main():
    print("=" * 60)
    print("   🎙️  SPEAKER ENROLLMENT FOR ANNABETH  🎙️")
    print("=" * 60)
    print()
    print("This will create a voice profile so Annabeth can recognize you.")
    print(f"You'll record {NUM_SAMPLES} samples of about {RECORDING_SECONDS} seconds each.")
    print()
    print("TIPS FOR BEST RESULTS:")
    print("  • Speak naturally, like you're having a conversation")
    print("  • Say different things for each sample")
    print("  • Speak at your normal volume and pace")
    print("  • Try to minimize background noise")
    print()
    
    # Get speaker name
    name = input("Enter speaker name (e.g., 'Dad', 'Emma', 'Blake'): ").strip()
    if not name:
        print("❌ Name cannot be empty!")
        return
    
    # Clean up name for filename
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_name = safe_name.replace(' ', '_')
    
    # Get input device
    device = get_input_device()
    
    print(f"\n📝 Enrolling: {name}")
    print("-" * 40)
    
    # Suggested phrases
    phrases = [
        "Hello Annabeth, this is my voice profile recording.",
        "The quick brown fox jumps over the lazy dog.",
        "I'm recording this sample to help you recognize my voice."
    ]
    
    samples = []
    for i in range(NUM_SAMPLES):
        print(f"\n--- Sample {i+1} of {NUM_SAMPLES} ---")
        print(f"💡 Suggestion: \"{phrases[i % len(phrases)]}\"")
        print("   (Or say anything you like!)")
        
        input("Press ENTER when ready to record...")
        
        audio = record_audio(RECORDING_SECONDS, device)
        
        # Check if audio has content
        if np.abs(audio).max() < 0.01:
            print("  ⚠️  Recording seems silent. Let's try again.")
            i -= 1
            continue
        
        samples.append(audio)
        
        # Option to playback
        playback = input("Play back recording? (y/n): ").strip().lower()
        if playback == 'y':
            play_audio(audio)
        
        if i < NUM_SAMPLES - 1:
            print(f"\n⏳ Next sample in {SILENCE_BETWEEN} seconds...")
            time.sleep(SILENCE_BETWEEN)
    
    # Create and save profile
    profile = create_profile(samples)
    
    from server.process.asr_func.speaker_id import save_speaker_profile
    save_speaker_profile(safe_name, profile)
    
    # Also save the audio samples for reference
    samples_dir = Path("speaker_profiles") / f"{safe_name}_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for i, audio in enumerate(samples):
        sf.write(samples_dir / f"sample_{i+1}.wav", audio, SAMPLE_RATE)
    print(f"  💾 Audio samples saved to: {samples_dir}")
    
    print()
    print("=" * 60)
    print(f"  ✅ SUCCESS! Profile created for: {name}")
    print("=" * 60)
    print()
    print("Annabeth will now be able to recognize this voice!")
    print()
    
    # Test identification
    test = input("Would you like to test the profile? (y/n): ").strip().lower()
    if test == 'y':
        print("\n🧪 Recording test sample...")
        input("Press ENTER when ready...")
        
        test_audio = record_audio(5, device)
        
        from server.process.asr_func.speaker_id import identify_speaker
        speaker, score = identify_speaker(test_audio, SAMPLE_RATE)
        
        print(f"\n🎯 Result: {speaker or 'Unknown'} (confidence: {score:.1%})")
        
        if speaker == safe_name:
            print("✅ Correctly identified!")
        elif speaker:
            print(f"⚠️  Identified as different speaker. You may need more samples.")
        else:
            print("❌ Not recognized. Try enrolling again with clearer samples.")


if __name__ == "__main__":
    main()
