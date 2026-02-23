# Local Server + Free Tunnel for School Demo 🎓

## 🎯 Solution: Host Locally + Free Tunnel Service

Since models are too big for free cloud tiers, we'll:
1. Run server on your PC (local)
2. Use **free tunnel service** to expose it to internet
3. App connects via tunnel URL (works from anywhere!)

---

## ⚡ Option 1: ngrok (Easiest - Recommended)

### ✅ Pros:
- **100% Free** (with limitations)
- Super easy setup (2 minutes)
- Works from anywhere
- Automatic HTTPS/WSS

### ⚠️ Free Tier Limitations:
- Random URL each time (changes on restart)
- 40 connections/minute limit
- Session timeout after 2 hours (just restart)

### 📋 Setup Steps:

#### 1. Install ngrok
```bash
# Windows: Download from https://ngrok.com/download
# Extract ngrok.exe to a folder (e.g., C:\ngrok)

# Or use Chocolatey:
choco install ngrok

# Mac/Linux:
brew install ngrok
# OR
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list && sudo apt update && sudo apt install ngrok
```

#### 2. Sign Up (Free)
1. Go to https://ngrok.com
2. Sign up (free account)
3. Get your authtoken from dashboard: https://dashboard.ngrok.com/get-started/your-authtoken

#### 3. Configure ngrok (One-Time Setup)
**Where to run:** In Command Prompt (Windows) or Terminal (Mac/Linux)

**Steps:**
1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Navigate to where ngrok is installed, OR if ngrok is in PATH, you can run from anywhere
3. Run this command (replace `YOUR_AUTH_TOKEN` with your actual token):

```bash
# Windows (Command Prompt or PowerShell)
ngrok config add-authtoken YOUR_AUTH_TOKEN

# Mac/Linux (Terminal)
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

#### 4. Start Your Local Server
```bash
# In your project directory
python ar_glasses_server.py
# Server should be running on localhost:8000
```

#### 5. Start ngrok Tunnel
```bash
# In a new terminal/command prompt
# IMPORTANT: Use port 8000 (not 80!) - this matches your server port
ngrok http 8000
```

You'll see:
```
Forwarding   https://abc123.ngrok-free.app -> http://localhost:8000
```

**Copy the HTTPS URL!** (e.g., `https://abc123.ngrok-free.app`)

#### 6. Update Android App
In `MainActivity.java`, line 57:
```java
private static final String SERVER_URL = "wss://abc123.ngrok-free.app";
```
**Note:** 
- Use `wss://` (secure WebSocket)
- Remove `https://` prefix, use `wss://` instead
- URL changes each time you restart ngrok (update app)

#### 7. Test Connection
- Open app
- Click "Connect"
- Should connect successfully!

### 🔄 For Demo Day:

**Before demo:**
1. Start local server: `python ar_glasses_server.py`
2. Start ngrok: `ngrok http 8000`
3. Copy the HTTPS URL
4. Update Android app with new URL
5. Rebuild and install app
6. Test connection

**During demo:**
- Keep both terminals open (server + ngrok)
- If ngrok disconnects, just restart it
- URL stays same during session

---

## 🚀 Option 2: Cloudflare Tunnel (Free, More Stable)

### ✅ Pros:
- **100% Free** (no limits!)
- Stable URL (doesn't change)
- No timeouts
- More reliable

### 📋 Setup:

#### 1. Install cloudflared
```bash
# Windows: Download from https://github.com/cloudflare/cloudflared/releases
# Extract cloudflared.exe

# Mac:
brew install cloudflare/cloudflare/cloudflared

# Linux:
# Download from GitHub releases
```

#### 2. Create Tunnel (One-time setup)
```bash
# Login to Cloudflare (free account)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create ar-glasses-demo

# Get tunnel ID (save this!)
# Example: abc12345-6789-0123-4567-890123456789
```

#### 3. Configure Tunnel
Create `config.yml` in your project:
```yaml
tunnel: abc12345-6789-0123-4567-890123456789  # Your tunnel ID
credentials-file: C:\path\to\.cloudflared\abc12345-6789-0123-4567-890123456789.json

ingress:
  - hostname: ar-glasses-demo.your-domain.com  # Optional: custom domain
    service: http://localhost:8000
  - service: http_status:404
```

**Or simpler (no config file):**
```bash
cloudflared tunnel --url http://localhost:8000
```

#### 4. Start Tunnel
```bash
# With config file:
cloudflared tunnel run ar-glasses-demo

# Or simple mode:
cloudflared tunnel --url http://localhost:8000
```

You'll get a URL like:
```
https://abc123.trycloudflare.com
```

#### 5. Update Android App
```java
private static final String SERVER_URL = "wss://abc123.trycloudflare.com";
```

---

## 🌐 Option 3: localtunnel (Simple Alternative)

### ✅ Pros:
- **100% Free**
- No signup required
- Very simple

### 📋 Setup:

#### 1. Install
```bash
npm install -g localtunnel
```

#### 2. Start Tunnel
```bash
# Start your server first
python ar_glasses_server.py

# In another terminal:
lt --port 8000
```

You'll get:
```
your url is: https://random-name.loca.lt
```

#### 3. Update Android App
```java
private static final String SERVER_URL = "wss://random-name.loca.lt";
```

**Note:** URL changes each time, but simpler than ngrok.

---

## 🎯 Recommended: ngrok for Quick Demo

**Why ngrok?**
- ✅ Fastest setup (2 minutes)
- ✅ Most reliable
- ✅ Free tier sufficient for demo
- ✅ Automatic HTTPS/WSS

**Just remember:**
- URL changes when you restart (update app)
- Keep both server and ngrok running
- Test before demo day!

---

## 📱 Complete Setup Workflow

### Before Demo Day:

1. **Prepare Local Server**
   ```bash
   # Make sure server works locally
   python ar_glasses_server.py
   # Test: Open browser to http://localhost:8000 (should see WebSocket error, that's OK)
   ```

2. **Set Up ngrok**
   ```bash
   # Install and configure (one time)
   ngrok config add-authtoken YOUR_TOKEN
   
   # Start tunnel
   ngrok http 8000
   ```

3. **Get Tunnel URL**
   - Copy the HTTPS URL from ngrok
   - Example: `https://abc123.ngrok-free.app`

4. **Update Android App**
   ```java
   // MainActivity.java, line 57
   private static final String SERVER_URL = "wss://abc123.ngrok-free.app";
   ```

5. **Rebuild App**
   - Build APK
   - Install on phone
   - Test connection

6. **Test Everything**
   - Test from home WiFi
   - Test from mobile data (different network)
   - Test voice registration
   - Test audio processing

### On Demo Day:

1. **Start Server** (Terminal 1)
   ```bash
   python ar_glasses_server.py
   ```

2. **Start ngrok** (Terminal 2)
   ```bash
   ngrok http 8000
   ```

3. **Copy URL** from ngrok output

4. **If URL Changed:**
   - Update `MainActivity.java` with new URL
   - Rebuild app (or use Android Studio's instant run)

5. **Keep Both Running:**
   - Don't close terminals
   - Keep laptop plugged in
   - Monitor server logs

---

## 🔧 Troubleshooting

### ngrok URL Changes
- **Problem:** URL changes when restarting ngrok
- **Solution:** 
  - Use paid ngrok for static URL ($8/month)
  - Or update app with new URL before demo
  - Or use Cloudflare Tunnel (free, stable URL)

### Connection Refused
- **Check:** Is local server running? (`python ar_glasses_server.py`)
- **Check:** Is ngrok running? (should show "Forwarding" message)
- **Check:** Firewall blocking port 8000? (Windows Firewall)

### ngrok Session Expired
- **Problem:** Free tier has 2-hour sessions
- **Solution:** Just restart ngrok before demo (takes 10 seconds)

### Slow Connection
- **Normal:** Tunnel adds ~100-200ms latency
- **Solution:** Use stable internet (WiFi, not mobile hotspot)

### App Can't Connect
- **Check:** Using `wss://` not `ws://`?
- **Check:** URL correct? (copy from ngrok exactly)
- **Check:** ngrok showing active connections?

---

## 💡 Pro Tips

1. **Pre-Demo Setup:**
   - Set up ngrok 1 day before
   - Test from different networks
   - Have backup plan (local WiFi if tunnel fails)

2. **During Demo:**
   - Keep laptop plugged in
   - Use stable WiFi (not mobile hotspot)
   - Have ngrok dashboard open (monitor connections)
   - Keep server logs visible

3. **Backup Options:**
   - If ngrok fails, use Cloudflare Tunnel
   - If both fail, use school WiFi + local IP (if same network)

4. **Performance:**
   - Tunnel adds minimal latency (~100ms)
   - Processing still happens on your PC (fast)
   - Models load from your PC (no cloud limits)

---

## 🎓 Quick Start (5 Minutes)

### First Time Setup (One-Time):

1. **Install ngrok**
   - Download from https://ngrok.com/download
   - Extract `ngrok.exe` (Windows) or `ngrok` (Mac/Linux)
   - Add to PATH, or place in your project folder

2. **Sign up and get token**
   - Go to https://ngrok.com → Sign up (free)
   - Go to https://dashboard.ngrok.com/get-started/your-authtoken
   - Copy your authtoken

3. **Configure ngrok (Command Prompt/Terminal)**
   ```bash
   # Open Command Prompt (Windows) or Terminal (Mac/Linux)
   # Navigate to where ngrok is, or if in PATH, run from anywhere:
   
   ngrok config add-authtoken YOUR_AUTH_TOKEN
   
   # Example:
   # ngrok config add-authtoken 2abc123def456ghi789jkl012mno345pq
   ```
   **This saves your token - you only do this once!**

### Every Time You Want to Demo:

```bash
# 1. Start server (Terminal/Command Prompt 1)
python ar_glasses_server.py

# 2. Start tunnel (Terminal/Command Prompt 2)
ngrok http 8000

# 3. Copy URL from ngrok output
# Example: https://abc123.ngrok-free.app

# 4. Update Android app
# SERVER_URL = "wss://abc123.ngrok-free.app"

# 5. Test!
```

---

## 📊 Comparison

| Service | Free? | Stable URL? | Setup Time | Best For |
|---------|-------|-------------|------------|----------|
| **ngrok** | ✅ Yes | ❌ No (changes) | 2 min | Quick demos |
| **Cloudflare** | ✅ Yes | ✅ Yes | 5 min | Stable demos |
| **localtunnel** | ✅ Yes | ❌ No | 1 min | Simple tests |

**Recommendation:** Use **ngrok** for quick setup, or **Cloudflare Tunnel** if you need stable URL.

---

## ✅ Pre-Demo Checklist

- [ ] ngrok installed and configured
- [ ] Local server tested and working
- [ ] ngrok tunnel tested
- [ ] Android app updated with tunnel URL
- [ ] Tested from different network (mobile data)
- [ ] Backup plan ready (Cloudflare Tunnel)
- [ ] Laptop charged / plugged in
- [ ] Stable internet connection ready

---

Perfect for FYP demo! No payment needed, works from anywhere! 🚀
