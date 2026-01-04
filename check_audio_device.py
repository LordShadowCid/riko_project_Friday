"""Check what audio device Windows is actually using."""
import pyaudiowpatch as pa

p = pa.PyAudio()

print("=" * 60)
print("DEFAULT AUDIO DEVICES")
print("=" * 60)

try:
    output_info = p.get_default_output_device_info()
    print(f"Default OUTPUT: {output_info['name']}")
    print(f"  Index: {output_info['index']}")
except Exception as e:
    print(f"Error getting default output: {e}")

try:
    input_info = p.get_default_input_device_info()
    print(f"\nDefault INPUT: {input_info['name']}")
    print(f"  Index: {input_info['index']}")
except Exception as e:
    print(f"Error getting default input: {e}")

print("\n" + "=" * 60)
print("ALL LOOPBACK DEVICES (for capturing system audio)")
print("=" * 60)

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('isLoopbackDevice', False):
        print(f"[{i}] {dev['name']}")
        print(f"    Channels: {dev['maxInputChannels']}, Rate: {int(dev['defaultSampleRate'])}")

p.terminate()
