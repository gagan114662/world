#!/bin/bash

# WorldPlay + AI Tutor Setup Script
# Run this script to set up the development environment

set -e  # Exit on error

echo "========================================"
echo "WorldPlay + AI Tutor Setup"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}✗ Python 3 is required but not installed${NC}"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓ Node.js: $NODE_VERSION${NC}"
else
    echo -e "${RED}✗ Node.js is required but not installed${NC}"
    exit 1
fi

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | tr -d ',')
    echo -e "${GREEN}✓ Docker: $DOCKER_VERSION${NC}"
else
    echo -e "${YELLOW}! Docker not found - you can still run services manually${NC}"
fi

# Check NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${GREEN}✓ GPU: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}! No NVIDIA GPU detected - WorldPlay Worker will run in simulation mode${NC}"
fi

# Create environment file
echo -e "\n${YELLOW}Setting up environment...${NC}"

if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from .env.example${NC}"
    echo -e "${YELLOW}! Please edit .env and add your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Install WorldPlay Gateway dependencies
echo -e "\n${YELLOW}Installing WorldPlay Gateway dependencies...${NC}"
cd worldplay-gateway
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
deactivate
echo -e "${GREEN}✓ WorldPlay Gateway dependencies installed${NC}"
cd ..

# Install WorldPlay Worker dependencies
echo -e "\n${YELLOW}Installing WorldPlay Worker dependencies...${NC}"
cd worldplay-worker
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -q -r requirements.txt
deactivate
echo -e "${GREEN}✓ WorldPlay Worker dependencies installed${NC}"
cd ..

# Install AI Tutor frontend dependencies
echo -e "\n${YELLOW}Installing AI Tutor frontend dependencies...${NC}"
cd aitutor/frontend
npm install --silent
echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
cd ../..

# Check if HY-WorldPlay models need to be downloaded
echo -e "\n${YELLOW}Checking HY-WorldPlay models...${NC}"
if [ ! -d "HY-WorldPlay/ckpts" ]; then
    echo -e "${YELLOW}! HY-WorldPlay models not found${NC}"
    echo ""
    echo "To download models, run:"
    echo "  cd HY-WorldPlay"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir ./ckpts/hunyuan-video-1.5"
    echo "  huggingface-cli download tencent/HY-WorldPlay --include \"HY-World1.5-Autoregressive-480P-I2V/*\" --local-dir ./ckpts/worldplay"
else
    echo -e "${GREEN}✓ HY-WorldPlay models directory found${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your API keys (GEMINI_API_KEY, GOOGLE_CLIENT_ID, etc.)"
echo "2. Download HY-WorldPlay models if you have a GPU (see commands above)"
echo "3. Start the services:"
echo ""
echo "   # Start with Docker:"
echo "   cd aitutor && docker-compose up -d"
echo ""
echo "   # Or start manually:"
echo "   # Terminal 1 - GPU Worker (on GPU machine):"
echo "   cd worldplay-worker && source venv/bin/activate && python -m worker.main"
echo ""
echo "   # Terminal 2 - WorldPlay Gateway:"
echo "   cd worldplay-gateway && source venv/bin/activate && uvicorn app.main:app --port 8010"
echo ""
echo "   # Terminal 3 - AI Tutor services:"
echo "   cd aitutor && docker-compose up redis teaching_assistant tutor"
echo ""
echo "   # Terminal 4 - Frontend:"
echo "   cd aitutor/frontend && npm run dev"
echo ""
echo "4. Open http://localhost:3000 in your browser"
echo ""
