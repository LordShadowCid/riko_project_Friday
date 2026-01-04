"""Quick test of system audio capture."""
import soundcard as sc
import numpy as np
import time

# Find loopback device
mics = sc.all_microphones(include_loopback=True)
loopback = None
for m in mics:
    if m.isloopback and 'Realtek' in m.name:
        loopback = m
        break

if not loopback:
    # Fallback to first loopback
    for m in mics:
        if m.isloopback:
            loopback = m
            break

if loopback:
    print(f'Using: {loopback.name}')
    print('Recording for 5 seconds... PLAY SOME MUSIC!')
    print('-' * 60)
    
    with loopback.recorder(samplerate=44100, channels=2) as rec:
        for i in range(50):
            data = rec.record(numframes=4410)
            rms = np.sqrt(np.mean(data**2))
            bars = int(rms * 500)
            bar_str = '#' * min(bars, 50)
            print(f'Level: {bar_str:<50} RMS={rms:.4f}')
            time.sleep(0.1)
    
    print('\nDone! If you saw bars moving while music played, loopback works.')
else:
    print('No loopback device found!')
    print('Available devices:')
    for m in mics:
        print(f'  - {m.name} (loopback={m.isloopback})')
