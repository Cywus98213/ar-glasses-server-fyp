# AR Glasses Server Setup Guide

## Overview

This AR Glasses application provides real-time speaker diarization, gesture recognition, and text-to-speech audio playback. The system consists of:
- **Python Server**: Handles audio processing, speaker diarization, and gesture recognition
- **Android App**: Captures audio/video, sends to server, and plays TTS audio on glasses

## Quick Setup (3 steps)

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

**Note:** The first run will download ML models (~2-3GB). Ensure you have:
- Stable internet connection
- Sufficient disk space
- Valid Hugging Face token

### 2. Configure Server

Edit `config.env` file:
- `SERVER_HOST`: Set to `0.0.0.0` (listens on all interfaces) or your specific IP
- `SERVER_PORT`: Default is `8000` (change if needed)
- `HF_TOKEN`: Your Hugging Face token (required for model downloads)
- `TRANSCRIPTION_LANGUAGE`: Language code (e.g., `zh` for Chinese, `en` for English)

**Example config.env:**
```
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
HF_TOKEN=your_huggingface_token_here
TRANSCRIPTION_LANGUAGE=zh
```

### 3. Run Server

```bash
python ar_glasses_server.py
```

You should see:
```
[SERVER] Initializing AR Glasses Server...
[SERVER] Using device: cuda (or cpu)
[SERVER] Diarization pipeline loaded successfully
[SERVER] Gesture recognition model loaded successfully
[SERVER] Server running on ws://0.0.0.0:8000
```

## Android App Configuration

### Update Server URL

In `MainActivity.java`, update the `SERVER_URL` constant (around line 80):

```java
private static final String SERVER_URL = "wss://your-server-url.com";
```

**For local testing:**
- Use `ws://YOUR_IP:8000` (replace YOUR_IP with your computer's IP)
- Example: `ws://192.168.1.100:8000`

**For production (with tunnel):**
- Use `wss://your-ngrok-url.ngrok-free.dev` or similar
- Ensure the server supports WSS (secure WebSocket)

### Required Permissions

The app requires:
- **Microphone**: For audio recording and speaker diarization
- **Camera**: For gesture recognition

These are requested automatically on first launch.

## Features

### 1. Speaker Diarization
- Real-time audio processing
- Automatic speaker identification
- Conversation transcription with speaker labels
- Supports multiple languages

### 2. Gesture Recognition
- Continuous hand gesture detection
- Real-time gesture classification
- Automatic camera orientation correction
- Gesture display with confidence scores

### 3. Text-to-Speech (TTS) Audio Playback
- Automatic TTS when gestures are detected
- Audio plays through glasses speakers
- Cooldown period to prevent repetition
- Clean gesture name announcements

## Finding Your IP Address

**Windows:**
1. Open Command Prompt
2. Type: `ipconfig`
3. Look for "IPv4 Address" under your active network adapter (usually starts with 192.168.x.x)

**Mac/Linux:**
1. Open Terminal
2. Type: `ifconfig` or `ip addr`
3. Look for "inet" address (usually starts with 192.168.x.x)

**Note:** Use the IP address of the network interface connected to the same network as your AR glasses device.

## Getting Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Select "Read" access (minimum required)
4. Copy the token
5. Replace the token in `config.env` with your token

**Important:** Keep your token secure and never commit it to version control.

## Using a Tunnel (for Remote Access)

If you need to access the server from outside your local network:

### Option 1: ngrok
```bash
ngrok http 8000
```
Use the provided HTTPS URL (e.g., `wss://abc123.ngrok-free.dev`) in the Android app.

### Option 2: Cloudflare Tunnel
```bash
cloudflared tunnel --url http://localhost:8000
```

### Option 3: Other tunneling services
- LocalTunnel
- Serveo
- Your own reverse proxy

## Testing

### Server Test
After running the server, verify it's listening:
```bash
# Check if port 8000 is open
netstat -an | grep 8000  # Linux/Mac
netstat -an | findstr 8000  # Windows
```

### App Test
1. Launch the Android app
2. Click "Test Server" to connect
3. Grant microphone and camera permissions
4. Start gesture detection
5. Make a gesture - you should hear TTS audio on the glasses

## Troubleshooting

### Server Issues

**Server won't start?**
- Check that all dependencies are installed: `pip list`
- Verify Python version (3.8+ required)
- Check for port conflicts: `lsof -i :8000` (Mac/Linux) or `netstat -ano | findstr :8000` (Windows)

**Models don't load?**
- Verify your Hugging Face token is correct
- Check internet connection (first run downloads models)
- Ensure sufficient disk space (~3GB for models)

**Connection refused?**
- Check firewall settings (allow port 8000)
- Verify SERVER_HOST in config.env
- Ensure server is running before connecting from app

### Android App Issues

**Can't connect to server?**
- Verify SERVER_URL in MainActivity.java matches your server
- Check if server is accessible from device network
- For local network: ensure phone and server are on same WiFi
- For tunnel: verify tunnel URL is correct and active

**No audio playback?**
- Check logcat for TTS errors: `adb logcat | grep TTS`
- Verify MediaPlayer is starting: look for "MediaPlayer prepared" in logs
- Check audio permissions are granted
- Ensure glasses audio output is working

**Gestures appear upside down?**
- This should be fixed automatically (camera orientation correction)
- If still inverted, check device orientation settings
- Verify camera permissions are granted

**Gesture detection not working?**
- Ensure camera permission is granted
- Check if camera is being used by another app
- Verify gesture recognition model loaded on server (check server logs)

### Performance Issues

**Slow processing?**
- Use GPU if available (server will auto-detect CUDA)
- Reduce chunk size in MainActivity.java if needed
- Close other resource-intensive applications

**High memory usage?**
- Server uses optimized models for memory efficiency
- Restart server periodically if running for extended periods
- Monitor memory usage: `htop` (Linux/Mac) or Task Manager (Windows)

## Advanced Configuration

### Custom Audio Settings
In `MainActivity.java`, you can adjust:
- `SAMPLE_RATE`: Audio sample rate (default: 16000)
- `CHUNK_INTERVAL_MS`: Real-time chunk interval (default: 3000ms)
- `GESTURE_SEND_INTERVAL_MS`: Gesture image send rate (default: 500ms)

### Custom Server Settings
In `config.env`:
- Adjust `SERVER_PORT` if 8000 is in use
- Change `TRANSCRIPTION_LANGUAGE` for different languages
- Server auto-detects GPU/CPU usage

## Logging and Debugging

### Server Logs
Server logs show:
- Connection status
- Audio processing progress
- Gesture recognition results
- Error messages

### Android App Logs
Use Android Studio Logcat or:
```bash
adb logcat | grep AR_GLASSES_APP
```

Key log tags:
- `AR_GLASSES_APP`: Main application logs
- `TTS`: Text-to-speech related logs
- `Gesture`: Gesture detection logs
- `WebSocket`: Connection logs

## Support

For issues or questions:
1. Check server logs for error messages
2. Check Android logcat for app errors
3. Verify all dependencies are installed
4. Ensure network connectivity between devices
5. Review this guide's troubleshooting section
