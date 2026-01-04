"""
Test WASAPI Loopback using PyAudioWPatch.
This should capture audio playing on your speakers (YouTube, Pandora, etc.)
"""
import pyaudiowpatch as pyaudio
import numpy as np
import time

def test_wasapi_loopback():
    print("=" * 60)
    print("Testing WASAPI Loopback Audio Capture")
    print("=" * 60)
    
    p = pyaudio.PyAudio()
    
    try:
        # Get default WASAPI loopback device
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(f"WASAPI Host API: {wasapi_info['name']}")
        
        # Get default loopback device (speakers)
        default_speakers = p.get_device_info_by_index(wasapi_info['defaultOutputDevice'])
        print(f"Default Speakers: {default_speakers['name']}")
        
        # Find loopback device for the speakers
        loopback_device = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get('isLoopbackDevice', False):
                if default_speakers['name'] in dev['name']:
                    loopback_device = dev
                    break
        
        if not loopback_device:
            # Try using the built-in method
            try:
                loopback_device = p.get_default_wasapi_loopback()
                print(f"Found loopback via get_default_wasapi_loopback()")
            except:
                pass
        
        if not loopback_device:
            print("\nNo loopback device found. Listing all devices:")
            for i in range(p.get_device_count()):
                dev = p.get_device_info_by_index(i)
                is_lb = dev.get('isLoopbackDevice', False)
                if is_lb or 'loopback' in dev['name'].lower():
                    print(f"  [{i}] {dev['name']} (loopback={is_lb})")
            return False
        
        print(f"\nUsing loopback: {loopback_device['name']}")
        print(f"  Channels: {int(loopback_device['maxInputChannels'])}")
        print(f"  Sample Rate: {int(loopback_device['defaultSampleRate'])}")
        
        # Open stream
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = int(loopback_device['maxInputChannels'])
        RATE = int(loopback_device['defaultSampleRate'])
        
        print(f"\n{'=' * 60}")
        print("PLAY SOME MUSIC! You should see bars moving below:")
        print("=" * 60)
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=loopback_device['index'],
            frames_per_buffer=CHUNK
        )
        
        # Capture and display for 10 seconds
        for _ in range(100):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.float32)
            
            # Calculate RMS
            rms = np.sqrt(np.mean(audio**2))
            
            # Simple frequency analysis
            if CHANNELS > 1:
                mono = audio.reshape(-1, CHANNELS).mean(axis=1)
            else:
                mono = audio
            
            fft = np.abs(np.fft.rfft(mono))
            bass = np.mean(fft[:10]) * 100
            mid = np.mean(fft[10:50]) * 100
            high = np.mean(fft[50:]) * 100
            
            # Display
            bar_len = 30
            bass_bar = '#' * min(int(bass * bar_len), bar_len)
            mid_bar = '#' * min(int(mid * bar_len), bar_len)
            high_bar = '#' * min(int(high * bar_len), bar_len)
            
            print(f"\rBass: {bass_bar:<{bar_len}} | Mid: {mid_bar:<{bar_len}} | High: {high_bar:<{bar_len}}", end='', flush=True)
            
            time.sleep(0.1)
        
        stream.stop_stream()
        stream.close()
        print("\n\nDone!")
        return True
        
    finally:
        p.terminate()


if __name__ == "__main__":
    test_wasapi_loopback()
