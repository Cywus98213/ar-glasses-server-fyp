# AR Glasses Server - Real-Time Speaker Diarization & Gesture Recognition

A comprehensive real-time audio processing and gesture recognition system for AR glasses, featuring speaker diarization, transcription, translation, and hand gesture recognition with text-to-speech feedback.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Core Modules](#core-modules)
- [WebSocket API](#websocket-api)
- [Functions & Methods](#functions--methods)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Details](#technical-details)

## 🎯 Overview

This system provides real-time processing capabilities for AR glasses applications:

- **Real-time Speaker Diarization**: Identifies and separates different speakers in audio streams
- **Speech Transcription**: Converts speech to text using Whisper Large-v3
- **Multi-language Translation**: Translates transcribed text to target languages
- **Gesture Recognition**: Recognizes hand gestures using MediaPipe
- **Text-to-Speech**: Plays gesture announcements through glasses speakers
- **Voice Registration**: Registers and identifies the glasses wearer's voice

## ✨ Features

### Audio Processing
- **Real-time Chunk Processing**: Processes audio in 3-second chunks for low latency
- **Speaker Identification**: Automatically identifies and tracks multiple speakers
- **Wearer Detection**: Identifies the AR glasses wearer using voice registration
- **Smart Deduplication**: Prevents duplicate transcriptions of similar speech
- **Multi-language Support**: Supports transcription and translation in multiple languages

### Gesture Recognition
- **Continuous Detection**: Real-time hand gesture recognition from camera feed
- **Multi-hand Support**: Detects up to 2 hands simultaneously
- **Confidence Scoring**: Provides confidence scores for each detected gesture
- **Automatic Orientation Correction**: Fixes camera orientation issues automatically
- **TTS Feedback**: Announces detected gestures through glasses speakers

### Performance
- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Parallel Processing**: Handles multiple connections and tasks concurrently
- **Memory Optimization**: Efficient memory management for extended operation
- **Connection Management**: Robust WebSocket connection handling with keep-alive

## 🏗️ Architecture

```
┌─────────────────┐
│  Android App    │
│  (AR Glasses)   │
└────────┬────────┘
         │ WebSocket
         │ (WSS/WS)
         ▼
┌─────────────────┐
│  Python Server  │
│                 │
│  ┌───────────┐  │
│  │ Diarization│  │
│  │ Pipeline  │  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │ Gesture   │  │
│  │ Recognizer│  │
│  └───────────┘  │
│                 │
│  ┌───────────┐  │
│  │ Translation│  │
│  │ Module    │  │
│  └───────────┘  │
└─────────────────┘
```

### Data Flow

1. **Audio Flow**:
   - Android app captures audio → Sends base64-encoded WAV chunks
   - Server processes with diarization pipeline → Identifies speakers
   - Server transcribes with Whisper → Translates if needed
   - Server sends results back → App displays in conversation UI

2. **Gesture Flow**:
   - Android app captures camera frames → Sends base64-encoded images
   - Server processes with MediaPipe → Recognizes gestures
   - Server sends gesture results → App displays and triggers TTS
   - TTS audio synthesized → Sent back to app → Played on glasses

## 📦 Core Modules

### 1. `ar_glasses_server.py` - Main Server

**Class: `ARGlassesServer`**

Main WebSocket server that handles all client connections and coordinates processing.

**Key Methods:**

- `__init__()`: Initializes server, loads models, sets up WebSocket server
- `handle_request()`: Main message handler for WebSocket connections
- `process_audio()`: Processes audio chunks with diarization
- `_process_gesture_async()`: Asynchronously processes gesture images
- `_safe_send_message()`: Thread-safe message sending
- `_cleanup_connection()`: Cleans up resources on disconnect

**Features:**
- Multi-connection support
- Parallel task execution
- Connection state tracking
- Stop processing control per connection

### 2. `speaker_diarization.py` - Audio Processing

**Class: `DiarizationPipeline`**

Handles speaker diarization, transcription, and translation.

**Key Methods:**

- `__init__()`: Initializes Whisper, pyannote, and speaker embedding models
- `process_audio()`: Main processing method for audio chunks
- `_extract_speaker_embedding()`: Extracts 192-dim speaker embeddings
- `_identify_speaker()`: Matches speakers using cosine similarity
- `_transcribe_segment()`: Transcribes audio segments with Whisper
- `_translate_text()`: Translates text to target language
- `_cleanup_memory()`: Memory management and cleanup

**Models Used:**
- **pyannote/speaker-diarization-3.1**: Speaker diarization
- **Whisper Large-v3**: Speech transcription
- **speechbrain/spkrec-ecapa-voxceleb**: Speaker embeddings (192-dim)

**Processing Pipeline:**
1. Diarization → Identifies speaker segments
2. Transcription → Converts speech to text
3. Speaker Matching → Matches to known speakers or creates new IDs
4. Translation → Translates if target language specified
5. Deduplication → Filters duplicate transcriptions

### 3. `gesture_recognition.py` - Gesture Processing

**Class: `GestureRecognizer`**

Handles hand gesture recognition using MediaPipe.

**Key Methods:**

- `__init__()`: Loads MediaPipe gesture recognition model
- `recognize_gesture()`: Processes image and returns gesture results
- `is_available()`: Checks if gesture recognition is available

**Features:**
- Supports up to 2 hands
- Configurable confidence thresholds
- Hand landmark detection
- Custom gesture model support

### 4. `translation_module.py` - Translation

**Class: `TranslationModule`**

Handles text translation between languages.

**Key Methods:**

- `__init__()`: Initializes translation model
- `translate()`: Translates text to target language
- `is_available()`: Checks if translation is available

## 🔌 WebSocket API

### Message Types

#### Client → Server Messages

**1. `join_conversation`**
```json
{
  "type": "join_conversation",
  "translation_language": "en"  // Optional
}
```
Initializes a new conversation session.

**2. `audio_from_glasses`**
```json
{
  "type": "audio_from_glasses",
  "chunk_id": "chunk_1234567890",
  "audio_data": "base64_encoded_wav_data",
  "sample_rate": 16000,
  "format": "wav",
  "is_chunk": true,  // true for real-time chunks
  "translation_language": "en"  // Optional
}
```
Sends audio data for processing.

**3. `gesture_from_glasses`**
```json
{
  "type": "gesture_from_glasses",
  "image_data": "base64_encoded_image",
  "timestamp": 1234567890.123
}
```
Sends camera image for gesture recognition.

**4. `register_voice`**
```json
{
  "type": "register_voice",
  "voice_id": "user_voice_001",
  "audio_data": "base64_encoded_wav",
  "sample_rate": 16000,
  "registration_method": "single"  // or "multi-sample"
}
```
Registers the wearer's voice for identification.

**5. `stop_processing`**
```json
{
  "type": "stop_processing"
}
```
Stops processing and discards remaining chunks.

**6. `reset_session`**
```json
{
  "type": "reset_session"
}
```
Resets the conversation session.

**7. `ping`**
```json
{
  "type": "ping"
}
```
Keep-alive ping message.

**8. `audio_to_glasses`**
```json
{
  "type": "audio_to_glasses",
  "audio_data": "base64_encoded_wav",
  "format": "wav",
  "sample_rate": 22050,
  "is_tts": true,
  "text": "Gesture name"
}
```
Sends TTS audio to be played on glasses.

#### Server → Client Messages

**1. `conversation_joined`**
```json
{
  "type": "conversation_joined",
  "status_code": 200,
  "message": "Successfully joined conversation",
  "translation_language": "en",
  "timestamp": 1234567890.123
}
```

**2. `segment_result`**
```json
{
  "type": "segment_result",
  "segment": {
    "speaker_id": "SPEAKER_00",
    "text": "Transcribed text",
    "start_time": 0.0,
    "end_time": 2.5,
    "is_wearer": false,
    "confidence": 0.95
  },
  "timestamp": 1234567890.123
}
```

**3. `processing_result`**
```json
{
  "type": "processing_result",
  "segments": [
    {
      "speaker_id": "SPEAKER_00",
      "text": "Transcribed text",
      "start_time": 0.0,
      "end_time": 2.5,
      "is_wearer": false
    }
  ],
  "timestamp": 1234567890.123
}
```

**4. `gesture_result`**
```json
{
  "type": "gesture_result",
  "status_code": 200,
  "gestures": [
    {
      "category_name": "Hello",
      "score": 0.95
    }
  ],
  "hand_landmarks": [...],
  "num_hands": 1,
  "timestamp": 1234567890.123
}
```

**5. `tts_audio`**
```json
{
  "type": "tts_audio",
  "audio_data": "base64_encoded_wav",
  "format": "wav",
  "sample_rate": 22050,
  "is_tts": true,
  "text": "Gesture name",
  "timestamp": 1234567890.123
}
```

**6. `processing_status`**
```json
{
  "type": "processing_status",
  "status": "Processing audio...",
  "timestamp": 1234567890.123
}
```

**7. `pong`**
```json
{
  "type": "pong",
  "status_code": 200,
  "timestamp": 1234567890.123
}
```

**8. `error`**
```json
{
  "type": "error",
  "error": "Error message",
  "timestamp": 1234567890.123
}
```

## 🔧 Functions & Methods

### ARGlassesServer Class

#### Initialization
```python
def __init__(self)
```
- Loads environment configuration
- Initializes diarization pipeline
- Initializes gesture recognizer
- Sets up WebSocket server
- Configures memory management

#### Connection Handling
```python
async def handle_request(self, websocket, path)
```
Main WebSocket handler that processes all incoming messages.

```python
def _cleanup_connection(self, websocket)
```
Cleans up resources when a connection is closed.

```python
async def _safe_send_message(self, websocket, message: Dict[str, Any])
```
Thread-safe message sending with connection state checking.

#### Audio Processing
```python
def process_audio(self, audio_array: np.ndarray, sample_rate: int, 
                  translation_language: str, is_chunk: bool = False, 
                  speaker_tracking: Dict[str, Any] = None) -> dict
```
Processes audio using diarization pipeline.

**Parameters:**
- `audio_array`: Audio data as numpy array
- `sample_rate`: Sample rate in Hz (typically 16000)
- `translation_language`: Target language for translation
- `is_chunk`: Whether this is a real-time chunk
- `speaker_tracking`: Speaker tracking state for the connection

**Returns:**
Dictionary with segments containing speaker_id, text, timestamps, etc.

#### Gesture Processing
```python
async def _process_gesture_async(self, image_data: str, websocket, timestamp: float)
```
Asynchronously processes gesture images.

**Parameters:**
- `image_data`: Base64-encoded image data
- `websocket`: WebSocket connection
- `timestamp`: Request timestamp

**Returns:**
Sends `gesture_result` message to client

#### Voice Registration
```python
def register_voice(self, voice_id: str, audio_data: str, 
                   sample_rate: int, registration_method: str = "single")
```
Registers a voice for wearer identification.

**Parameters:**
- `voice_id`: Unique identifier for the voice
- `audio_data`: Base64-encoded audio data
- `sample_rate`: Audio sample rate
- `registration_method`: "single" or "multi-sample"

### DiarizationPipeline Class

#### Processing
```python
def process_audio(self, audio_array: np.ndarray, sample_rate: int, 
                  translation_language: str = None, 
                  is_chunk: bool = False,
                  speaker_tracking: Dict[str, Any] = None,
                  registered_voices: Dict[str, Any] = None) -> Dict[str, Any]
```
Main audio processing method.

**Processing Steps:**
1. Diarization: Identifies speaker segments
2. Transcription: Converts speech to text per segment
3. Speaker Matching: Matches to known speakers or creates new IDs
4. Translation: Translates text if target language specified
5. Deduplication: Removes duplicate transcriptions

**Returns:**
```python
{
    "segments": [
        {
            "speaker_id": "SPEAKER_00",
            "text": "Transcribed text",
            "start_time": 0.0,
            "end_time": 2.5,
            "is_wearer": False,
            "confidence": 0.95
        }
    ],
    "speaker_tracking": {...},  # Updated speaker tracking state
    "processing_time": 1.23
}
```

#### Speaker Identification
```python
def _extract_speaker_embedding(self, audio_segment: np.ndarray, 
                                sample_rate: int) -> np.ndarray
```
Extracts 192-dimensional speaker embedding using ECAPA-TDNN.

```python
def _identify_speaker(self, embedding: np.ndarray, 
                     speaker_tracking: Dict[str, Any],
                     registered_voices: Dict[str, Any] = None,
                     similarity_threshold: float = 0.7) -> str
```
Identifies speaker by matching embedding to known speakers.

**Matching Logic:**
1. Check against registered wearer voice (if available)
2. Match against known speakers using cosine similarity
3. Create new speaker ID if no match found

#### Transcription
```python
def _transcribe_segment(self, audio_segment: np.ndarray, 
                        sample_rate: int, language: str = None) -> str
```
Transcribes audio segment using Whisper Large-v3.

#### Translation
```python
def _translate_text(self, text: str, target_language: str) -> str
```
Translates text to target language if translation module is available.

### GestureRecognizer Class

#### Recognition
```python
def recognize_gesture(self, image_data: np.ndarray) -> Dict[str, Any]
```
Recognizes gestures from image data.

**Parameters:**
- `image_data`: Image as numpy array (BGR or RGB format)

**Returns:**
```python
{
    "success": True,
    "gestures": [
        {
            "category_name": "Hello",
            "score": 0.95
        }
    ],
    "hand_landmarks": [...],
    "num_hands": 1
}
```

## 🚀 Installation

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed installation instructions.

**Quick Start:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure server
# Edit config.env with your settings

# Run server
python ar_glasses_server.py
```

## 📖 Usage

### Starting the Server

```bash
python ar_glasses_server.py
```

The server will:
1. Load diarization models (~2-3GB download on first run)
2. Load gesture recognition model
3. Start WebSocket server on configured port (default: 8000)

### Connecting from Android App

1. Update `SERVER_URL` in `MainActivity.java`
2. Launch the app
3. Click "Test Server" to connect
4. Grant microphone and camera permissions
5. Start recording or gesture detection

### Example: Processing Audio

```python
# Client sends:
{
    "type": "audio_from_glasses",
    "chunk_id": "chunk_001",
    "audio_data": "base64_wav_data...",
    "sample_rate": 16000,
    "is_chunk": true,
    "translation_language": "en"
}

# Server responds with:
{
    "type": "segment_result",
    "segment": {
        "speaker_id": "SPEAKER_00",
        "text": "Hello, how are you?",
        "is_wearer": false
    }
}
```

### Example: Gesture Recognition

```python
# Client sends:
{
    "type": "gesture_from_glasses",
    "image_data": "base64_image_data...",
    "timestamp": 1234567890.123
}

# Server responds with:
{
    "type": "gesture_result",
    "status_code": 200,
    "gestures": [
        {"category_name": "Hello", "score": 0.95}
    ],
    "num_hands": 1
}

# Client then synthesizes TTS and sends:
{
    "type": "audio_to_glasses",
    "audio_data": "base64_tts_audio...",
    "is_tts": true,
    "text": "Hello"
}

# Server forwards back as:
{
    "type": "tts_audio",
    "audio_data": "base64_tts_audio...",
    "text": "Hello"
}
```

## ⚙️ Configuration

### Environment Variables (`config.env`)

```bash
SERVER_HOST=0.0.0.0          # Server host (0.0.0.0 for all interfaces)
SERVER_PORT=8000             # WebSocket port
HF_TOKEN=your_token_here     # Hugging Face token (required)
TRANSCRIPTION_LANGUAGE=zh    # Default transcription language

# Speaker threshold tuning (optional)
# Tip: start from `config.env.example` and copy to `config.env`
SPEAKER_SIM_THRESHOLD=0.32
WEARER_SIM_THRESHOLD=0.65
DIAR_CLUSTERING_THRESHOLD=0.5
WEARER_SINGLE_THRESHOLD_CHUNK=0.65
SPEAKER_MATCH_THRESHOLD=0.32
SPEAKER_PROACTIVE_MERGE_THRESHOLD=0.35
```

### Model Configuration

**Diarization:**
- Model: `pyannote/speaker-diarization-3.1`
- Clustering threshold: Controlled by `DIAR_CLUSTERING_THRESHOLD` (default 0.5)
- Min cluster size: Controlled by `DIAR_MIN_CLUSTER_SIZE` (default 2)
- Min duration off: Controlled by `DIAR_MIN_DURATION_OFF` (default 0.5s)

**Transcription:**
- Model: Whisper Large-v3
- Compute type: int8 (memory efficient)
- Device: Auto-detect (CUDA if available)

**Gesture Recognition:**
- Model: Custom MediaPipe model (`.task` file)
- Max hands: 2
- Detection confidence: 0.5
- Tracking confidence: 0.5

## 🔬 Technical Details

### Audio Processing

**Format:**
- Sample rate: 16000 Hz
- Format: 16-bit PCM WAV
- Channels: Mono
- Encoding: Base64 for transmission

**Real-time Chunks:**
- Chunk size: 3 seconds (~48KB)
- Overlap: None (client handles chunking)
- Processing: Parallel per connection

**Speaker Embeddings:**
- Dimension: 192 (ECAPA-TDNN)
- Similarity metric: Cosine similarity
- Thresholds: Tunable via env (see `config.env.example` and `threshold_settings.py`)

### Gesture Recognition

**Image Format:**
- Format: JPEG (base64 encoded)
- Resolution: 640x480 (configurable)
- Color space: RGB (converted from BGR if needed)

**Processing:**
- Frame rate: ~2 FPS (throttled)
- Model: MediaPipe Gesture Recognizer
- Output: Gesture categories with confidence scores

### Performance

**Latency:**
- Audio processing: ~1-3 seconds per chunk
- Gesture recognition: ~200-500ms per image
- TTS synthesis: ~500ms-1s

**Memory:**
- Base memory: ~2-3GB (models loaded)
- Per connection: ~100-200MB
- Peak usage: ~4-5GB with multiple connections

**GPU Acceleration:**
- Automatic CUDA detection
- Falls back to CPU if GPU unavailable
- Significant speedup on GPU (3-5x faster)

### Error Handling

- Connection timeouts handled gracefully
- Model loading errors prevent server start
- Processing errors logged and sent to client
- Automatic cleanup on disconnect
- Memory management prevents OOM errors

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

---

**Note**: This README documents the current implementation. For setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md). For tunnel setup, see [LOCAL_TUNNEL_DEMO.md](LOCAL_TUNNEL_DEMO.md).
