# AR Glasses Server - Deployment Options

## 🚀 Two Server Versions Available

### **Full Version** - `ar_glasses_server.py`
**For:** Local development, powerful cloud servers, production with GPU

**Features:**
- ✅ Full AI speaker recognition
- ✅ Advanced diarization 
- ✅ Speech-to-text with Whisper
- ✅ SpeechBrain embeddings
- ⚠️ Requires 4-8GB RAM
- ⚠️ Heavy dependencies

**Run locally:**
```bash
pip install -r requirements.txt
python ar_glasses_server.py
```

### **Lite Version** - `ar_glasses_server_lite.py`
**For:** Railway, Render free tier, memory-limited deployments

**Features:**
- ✅ WebSocket server
- ✅ Audio reception/processing
- ✅ Basic audio analysis
- ✅ Works with <512MB RAM
- ⚠️ No AI speaker recognition (placeholder responses)

**Run locally:**
```bash
pip install -r requirements-lite.txt
python ar_glasses_server_lite.py
```

## 🔄 Railway Deployment

The Docker setup automatically uses the **Lite Version** for Railway:
- Uses `requirements-lite.txt` (minimal packages)
- Runs `ar_glasses_server_lite.py` 
- Fast build, low memory usage
- Perfect for testing/demos

## 🎯 For Your Groupmate

### **Railway URL (Lite):**
```
wss://your-app.railway.app
```
- Basic audio processing
- Good for testing WebSocket connection
- No AI features

### **Local/Powerful Server (Full):**
```
ws://your-ip:8000
```
- Full AI speaker recognition
- Complete feature set
- Requires powerful hardware

## 💡 Recommendation

1. **Use Railway Lite** for initial testing and demos
2. **Use Full Version locally** for development
3. **Deploy Full Version** on powerful cloud (AWS/GCP) for production
