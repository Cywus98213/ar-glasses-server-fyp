# AR Glasses Server Setup Guide

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

### 2. Configure Server
Edit `config.env` file:
- Change `SERVER_HOST` to your computer's IP address
- Change `HF_TOKEN` to your Hugging Face token

### 3. Run Server
```bash
python ar_glasses_server.py
```

## Finding Your IP Address

**Windows:**
1. Open Command Prompt
2. Type: `ipconfig`
3. Look for "IPv4 Address" (usually starts with 192.168.x.x)

**Mac/Linux:**
1. Open Terminal
2. Type: `ifconfig`
3. Look for "inet" address (usually starts with 192.168.x.x)

## Getting Hugging Face Token

1. Go to: https://huggingface.co/settings/tokens
2. Click "New token"
3. Copy the token
4. Replace `your_token_here` in `config.env` with your token

## Testing

After running the server, you should see:
```
[SERVER] Server running on ws://YOUR_IP:8000
```

Use this address in your AR glasses app.

## Troubleshooting

- Server won't start? Check that all dependencies are installed
- Connection fails? Make sure IP address is correct
- Models don't load? Check your Hugging Face token
