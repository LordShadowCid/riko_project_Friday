import requests
import json
import time

def test_gpt_sovits_connection():
    """Test connection to GPT-SoVITS Docker container"""
    url = "http://127.0.0.1:9880"
    
    print("🔍 Testing GPT-SoVITS connection...")
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ GPT-SoVITS health check passed!")
        else:
            print(f"⚠️  Health check returned status: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test TTS endpoint with a simple request
    try:
        test_payload = {
            "text": "Hello, this is a test of the GPT-SoVITS voice system.",
            "text_lang": "en",
            "ref_audio_path": "character_files/main_sample.wav",
            "prompt_text": "This is a sample voice for testing.",
            "prompt_lang": "en"
        }
        
        print("🎯 Testing TTS generation...")
        tts_response = requests.post(f"{url}/tts", json=test_payload, timeout=30)
        
        if tts_response.status_code == 200:
            print("✅ TTS test successful!")
            print(f"   Response size: {len(tts_response.content)} bytes")
            
            # Save test audio
            with open("test_output.wav", "wb") as f:
                f.write(tts_response.content)
            print("   Test audio saved as 'test_output.wav'")
            return True
        else:
            print(f"❌ TTS test failed with status: {tts_response.status_code}")
            print(f"   Response: {tts_response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ TTS test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 GPT-SoVITS Docker Test Script")
    print("=" * 40)
    
    # Wait a moment for the service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(5)
    
    success = test_gpt_sovits_connection()
    
    if success:
        print("\n🎉 GPT-SoVITS is working correctly!")
        print("You can now run the main Riko chat application.")
    else:
        print("\n❌ GPT-SoVITS test failed.")
        print("Please check the Docker logs: docker logs riko-gpt-sovits")