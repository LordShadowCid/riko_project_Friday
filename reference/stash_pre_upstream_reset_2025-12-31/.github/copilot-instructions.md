# Riko Project - AI Voice Assistant

This workspace contains Project Riko, an AI-powered voice assistant that combines:
- OpenAI's GPT for language processing
- GPT-SoVITS for voice synthesis
- Faster-Whisper for speech recognition

## Project Setup Status

- [x] Repository cloned from https://github.com/LordShadowCid/riko_project_Friday
- [x] Python virtual environment configured (Python 3.13.5)
- [x] Dependencies installed via uv (extra-req.txt and requirements.txt)
- [x] Project structure verified and imports tested
- [x] VS Code Python extensions available

## Project Structure

- `server/main_chat.py` - Main chat application
- `server/process/` - Core processing modules (ASR, LLM, TTS)
- `character_config.yaml` - Character configuration and API keys
- `requirements.txt` - Main Python dependencies
- `extra-req.txt` - Additional requirements

## Development Notes

- Virtual environment located at `.venv/`
- Use `C:/Users/blakd/OneDrive/Desktop/Anabeth/.venv/Scripts/python.exe` to run Python
- Main application requires OpenAI API key configuration
- Audio files stored in `audio/` directory (auto-created)

## Next Steps

1. Configure OpenAI API key in `character_config.yaml`
2. Set up GPT-SoVITS API server (external requirement)
3. Run `python server/main_chat.py` to start the voice assistant