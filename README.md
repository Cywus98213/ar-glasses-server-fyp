# AR Glasses Speaker Recognition System

A real-time speaker recognition and diarization system for AR glasses, featuring advanced AI models for speech-to-text conversion and speaker identification.

## Overview

This system enables AR glasses to identify and transcribe multiple speakers in real-time conversations. The audio captured by the glasses is sent to a server for processing using state-of-the-art AI models, and the results (speaker identification + transcribed text) are displayed back on the glasses.

## Features

- **Real-time Speaker Diarization**: Identifies who spoke when in multi-speaker conversations
- **Speaker Recognition**: Matches speakers to known profiles in the database
- **Speech-to-Text**: Accurate transcription using Whisper models
- **WebSocket Communication**: Low-latency real-time audio streaming
- **Easy Configuration**: Simple config file for network and model settings
- **Debug Support**: Saves audio and JSON output for troubleshooting
- **GPU Acceleration**: Supports CUDA for faster processing

## System Architecture

### Smart Glasses (Client Side)
- Captures audio via microphone
- Preprocesses audio (filtering, normalization)
- Sends audio to server via WebSocket
- Displays speaker identification and transcribed text on OLED screen

### Server Side
- **WebSocket Server**: Handles real-time communication
- **Speaker Diarization**: Identifies different speakers and their speech segments
- **Speech Recognition**: Converts audio to text using Whisper
- **Speaker Recognition**: Matches voice embeddings to known speakers
- **Database**: Stores speaker profiles and embeddings

## Requirements

### Hardware
- Server with recommended GPU (CUDA-capable) for faster processing
- AR glasses with microphone and display
- Network connection between glasses and server

### Software
- Python 3.8 or higher
- PyTorch with CUDA support (recommended)
- See `requirements.txt` for full list of dependencies

## Quick Start

### 1. Create Virtual Environment and Install Dependencies

**Create virtual environment:**
```bash
python -m venv venv
```

**Activate virtual environment:**
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Configure Server

Edit `config.env` file:

```env
# Your computer's IP address (find with: ipconfig on Windows, ifconfig on Mac/Linux)
SERVER_HOST=192.168.0.112

# Port number (use 8000 unless you have conflicts)
SERVER_PORT=8000

# Get your free token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here
```

**How to find your IP address:**
- Windows: Open Command Prompt, type `ipconfig`
- Mac/Linux: Open Terminal, type `ifconfig`
- Look for "IPv4 Address" (usually starts with 192.168.x.x)

**How to get Hugging Face token:**
1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Copy the token
4. Replace `your_token_here` in `config.env` with your token

### 3. Run Server

```bash
python ar_glasses_server.py
```

You should see:
```
[SERVER] Server running on ws://YOUR_IP:8000
```

### 4. Connect AR Glasses

Configure your AR glasses app to connect to:
```
ws://YOUR_IP:8000
```

## System Flow

```
AR Glasses                          Server
    |                                  |
    | 1. Capture Audio                 |
    |--------------------------------->|
    |                                  | 2. Decode Audio
    |                                  | 3. Diarization
    |                                  | 4. Speech-to-Text
    |                                  | 5. Speaker Recognition
    |<---------------------------------|
    | 6. Display Results               |
    |    (Speaker ID + Text)           |
```

## WebSocket API

### Client to Server Messages

#### Join Conversation
```json
{
  "type": "join_conversation"
}
```

#### Send Audio
```json
{
  "type": "audio_from_glasses",
  "chunk_id": "unique_chunk_id",
  "audio_data": "base64_encoded_audio",
  "sample_rate": 16000
}
```

#### Reset Session
```json
{
  "type": "reset_session"
}
```

### Server to Client Messages

#### Audio Received Confirmation
```json
{
  "type": "audio_received",
  "chunk_id": "unique_chunk_id",
  "status_code": 200,
  "timestamp": 1234567890.123
}
```

#### Segment Result
```json
{
  "type": "segment_result",
  "chunk_id": "unique_chunk_id",
  "segment": {
    "speaker_id": "SPEAKER_00",
    "text": "Hello, how are you?",
    "start": 0.0,
    "end": 2.5,
    "duration": 2.5,
    "confidence": 0.95
  },
  "timestamp": 1234567890.123
}
```

#### Processing Complete
```json
{
  "type": "audio_processed",
  "chunk_id": "unique_chunk_id",
  "total_segments": 3,
  "timestamp": 1234567890.123
}
```

## Configuration Options

### Server Settings
- `SERVER_HOST`: IP address of the server
- `SERVER_PORT`: Port number for WebSocket server

### Model Settings
- `HF_TOKEN`: Hugging Face token for model access
- `USE_ENHANCED_MODELS`: Enable enhanced AI models
- `FALLBACK_TO_SIMPLE`: Fallback to simpler models if enhanced fails

### Speaker Recognition Settings
- `SIMILARITY_THRESHOLD`: Minimum similarity score (0.0-1.0)
- `MIN_SEGMENT_DURATION`: Minimum speech segment duration in seconds
- `MAX_SEGMENT_DURATION`: Maximum speech segment duration in seconds

### Performance Settings
- `USE_CACHE`: Enable caching for faster processing
- `CACHE_SIZE`: Maximum number of cached items
- `USE_HALF_PRECISION`: Use half precision (FP16) for faster GPU processing

### Debug Settings
- `SAVE_DEBUG_AUDIO`: Save received audio files for debugging
- `SAVE_DEBUG_JSON`: Save JSON output for debugging
- `VERBOSE_LOGGING`: Enable detailed console logging

## Project Structure

```
FYP-serverBasedSpeakerRecon_draft/
├── ar_glasses_server.py          # Main WebSocket server
├── speaker_diarization.py        # Speaker diarization pipeline
├── speaker_recognition.py        # Speaker recognition module
├── config.env                    # Configuration file
├── requirements.txt              # Python dependencies
├── SETUP_GUIDE.md               # Detailed setup instructions
├── SYSTEM_BLOCK_DIAGRAM.md      # System architecture diagram
├── speakers.db                   # Speaker database
├── pretrained_models/           # AI model cache
├── debug_json_output/           # Debug JSON files
├── recorded_audio/              # Recorded audio samples
└── testing_audio/               # Test audio files
```

## AI Models Used

- **Speech-to-Text**: Faster-Whisper (OpenAI Whisper optimized)
- **Speaker Diarization**: PyAnnote Audio with segmentation models
- **Speaker Recognition**: SpeechBrain ECAPA-TDNN embeddings
- **Voice Activity Detection**: Silero VAD

## How It Works

1. **Audio Capture**: AR glasses capture audio through the microphone
2. **Preprocessing**: Audio is preprocessed and encoded as Base64
3. **Transmission**: Audio sent to server via WebSocket connection
4. **Diarization**: Server identifies different speakers and their segments
5. **Transcription**: Each segment is transcribed using Whisper
6. **Recognition**: Speaker embeddings are matched against known speakers
7. **Response**: Results sent back to glasses with speaker ID and text
8. **Display**: Glasses display speaker identification and transcribed text

## Troubleshooting

### Server won't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version is 3.8 or higher: `python --version`
- Check for port conflicts on port 8000

### Connection fails
- Verify SERVER_HOST is set to correct IP address
- Ensure server and glasses are on the same network
- Check firewall settings allow WebSocket connections

### Models don't load
- Verify HF_TOKEN is valid and set in config.env
- Check internet connection for initial model downloads
- Ensure sufficient disk space for model cache

### Poor recognition accuracy
- Adjust SIMILARITY_THRESHOLD in config.env
- Ensure good audio quality from microphone
- Check for background noise interference

### Slow processing
- Enable GPU/CUDA support for PyTorch
- Set USE_HALF_PRECISION=true in config.env
- Reduce audio chunk size from glasses

## Performance

- **Processing Time**: 2-5 seconds per audio chunk (with GPU)
- **Speaker Recognition**: 85-95% accuracy with clean audio
- **Speech-to-Text**: >90% accuracy using Whisper models
- **Latency**: <1 second WebSocket communication overhead

## Debug Output

When debug settings are enabled, the system saves:
- **Audio Files**: `debug_json_output/received_audio_*.wav`
- **Processing Results**: `debug_json_output/processing_result_*.json`
- **Segment Results**: `debug_json_output/segment_result_*.json`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for academic/research purposes.

## Support

For issues and questions, please check:
- Configuration in `config.env`
- Setup guide in `SETUP_GUIDE.md`
- System diagram in `SYSTEM_BLOCK_DIAGRAM.md`

## Acknowledgments

- OpenAI Whisper for speech-to-text
- PyAnnote Audio for speaker diarization
- SpeechBrain for speaker recognition
- Hugging Face for model hosting
