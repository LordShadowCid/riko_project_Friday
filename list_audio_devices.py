"""List all audio devices to find loopback devices."""
import pyaudiowpatch as pa

p = pa.PyAudio()

print("=" * 60)
print("Audio Devices:")
print("=" * 60)

loopback_devices = []

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    is_loopback = dev.get('isLoopbackDevice', False)
    name = dev.get('name', 'Unknown')
    
    if is_loopback:
        loopback_devices.append((i, name))
    
    if is_loopback or 'speaker' in name.lower() or 'realtek' in name.lower() or 'headphone' in name.lower():
        print(f"[{i}] {name}")
        print(f"    Loopback: {is_loopback}, Channels: {dev.get('maxInputChannels', 0)}")

print("\n" + "=" * 60)
print("All Loopback Devices:")
print("=" * 60)
for idx, name in loopback_devices:
    print(f"  [{idx}] {name}")

p.terminate()
