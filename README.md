# WorldPlay AI Tutor

An immersive learning platform that combines AI tutoring with real-time 3D world generation for the most compelling educational experience.

## Overview

This project integrates:
- **[AI Tutor](https://github.com/GetFoolish/aitutor)** - Adaptive K-12 tutoring platform with Gemini AI
- **[HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay)** - Tencent's real-time 3D world generation

Students can explore immersive 3D environments while receiving AI-guided tutoring across multiple subjects including Physics, Biology, Chemistry, History, Geography, and Math.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   React Frontend │────│  WorldPlay       │────│  GPU Worker     │
│   + WorldViewer  │    │  Gateway (:8010) │    │  (Local Server) │
└────────┬────────┘     └────────┬─────────┘     └─────────────────┘
         │                       │
         │              ┌────────┴─────────┐
         │              │  Tutor Service   │
         └──────────────│  (Gemini Live)   │
                        │  (:8767)         │
                        └──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- NVIDIA GPU with 14GB+ VRAM (for world generation)
- CUDA 11.8+

### Setup

```bash
# Clone this repository (if not already done)
cd /path/to/world_play

# Run setup script
./setup.sh

# Edit environment variables
cp .env.example .env
# Add your API keys to .env

# Download HY-WorldPlay models (requires ~50GB)
cd HY-WorldPlay
pip install huggingface_hub
huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/hunyuan-video-1.5
huggingface-cli download tencent/HY-WorldPlay \
  --include "HY-World1.5-Autoregressive-480P-I2V/*" \
  --local-dir ./ckpts/worldplay
```

### Running the Application

**Option 1: Docker (Recommended)**

```bash
cd aitutor
docker-compose up -d
```

**Option 2: Manual**

```bash
# Terminal 1 - GPU Worker (on GPU machine)
cd worldplay-worker
source venv/bin/activate
python -m worker.main

# Terminal 2 - WorldPlay Gateway
cd worldplay-gateway
source venv/bin/activate
uvicorn app.main:app --port 8010

# Terminal 3 - AI Tutor services
cd aitutor
docker-compose up redis teaching_assistant tutor

# Terminal 4 - Frontend
cd aitutor/frontend
npm run dev
```

### Access

- Frontend: http://localhost:3000
- WorldPlay Gateway: http://localhost:8010
- WorldPlay API Docs: http://localhost:8010/docs

## Project Structure

```
world_play/
├── aitutor/                    # AI Tutor platform (cloned)
│   ├── frontend/               # React frontend
│   │   └── src/
│   │       ├── components/
│   │       │   └── WorldViewer/  # 3D world viewer components
│   │       ├── hooks/          # useWorldStream, useWorldControls
│   │       └── services/       # worldPlayApi.ts
│   ├── services/               # Backend services
│   └── docker-compose.yml
│
├── HY-WorldPlay/               # HY-WorldPlay (cloned)
│   ├── ckpts/                  # Model checkpoints
│   └── generate.py             # Generation script
│
├── worldplay-gateway/          # WorldPlay Gateway Service
│   ├── app/
│   │   ├── main.py            # FastAPI application
│   │   ├── routers/           # API endpoints
│   │   ├── services/          # Business logic
│   │   └── workers/           # GPU client
│   └── Dockerfile
│
├── worldplay-worker/           # GPU Worker Service
│   ├── worker/
│   │   ├── main.py            # Worker entry point
│   │   ├── inference_engine.py # HY-WorldPlay wrapper
│   │   ├── model_manager.py   # Model loading
│   │   └── frame_encoder.py   # Video encoding
│   └── requirements.txt
│
├── .env.example                # Environment template
├── setup.sh                    # Setup script
└── README.md
```

## Features

### Immersive Learning

The AI tutor analyzes student questions and automatically triggers 3D world generation when visual/spatial learning would be beneficial:

- **Physics**: Explore gravity in a space station, see electrical circuits from inside
- **Biology**: Journey inside a cell, explore ecosystems
- **Chemistry**: View molecular bonds at atomic scale
- **History**: Walk through ancient civilizations
- **Geography**: Witness geological processes
- **Math**: Interact with 3D shapes and coordinate systems

### Interactive Controls

- **WASD**: Move through the world
- **Mouse**: Look around
- **Click**: Enable pointer lock for immersive control
- **ESC**: Exit pointer lock

### AI Narration

The Gemini AI tutor guides exploration with:
- Real-time narration explaining what students see
- Guiding questions to encourage observation
- Highlights drawing attention to key elements
- Scene transitions to explore related concepts

## API Reference

### WorldPlay Gateway API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/worlds/generate` | POST | Generate new world |
| `/api/v1/worlds/{id}` | GET | Get session status |
| `/api/v1/worlds/{id}/pause` | POST | Pause generation |
| `/api/v1/worlds/{id}/resume` | POST | Resume generation |
| `/api/v1/worlds/{id}/transition` | POST | Transition scene |
| `/ws/worlds/{id}/stream` | WS | Video streaming |

### WebSocket Protocol

**Client → Server:**
```json
{
  "type": "action",
  "payload": {
    "keyboard": {"w": true, "a": false, "s": false, "d": false},
    "mouse": {"dx": 10, "dy": -5}
  }
}
```

**Server → Client:**
- Binary: H.264/JPEG encoded video frames
- JSON: Metadata, narration, control messages

## Configuration

See `.env.example` for all configuration options including:
- Google OAuth credentials
- Gemini API key
- MongoDB/Redis connections
- GPU worker host/port
- Model paths

## Troubleshooting

### CUDA Out of Memory
Enable model offloading or reduce batch size:
```bash
export WORLDPLAY_MODEL_OFFLOAD=true
```

### WebSocket Disconnections
Increase timeout in frontend config or check network stability.

### Slow Generation
Ensure you're using the distilled model with `--few-step` flag.

### Black Frames
Verify H.264 codec support in browser (Chrome/Firefox recommended).

## License

This project integrates:
- AI Tutor: [MIT License](https://github.com/GetFoolish/aitutor/blob/main/LICENSE)
- HY-WorldPlay: [Tencent Hunyuan License](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/main/LICENSE)

## Acknowledgments

- [Tencent Hunyuan Team](https://github.com/Tencent-Hunyuan) for HY-WorldPlay
- [GetFoolish](https://github.com/GetFoolish) for the AI Tutor platform
