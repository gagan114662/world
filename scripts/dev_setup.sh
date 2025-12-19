#!/bin/bash
# Development Setup Script for WorldPlay AI Tutor
# This script sets up the local development environment

set -e

echo "=============================================="
echo "WorldPlay AI Tutor - Development Setup"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo -e "\n${GREEN}1. Creating mock world data...${NC}"
python3 scripts/create_mock_worlds.py --output ./world_library

echo -e "\n${GREEN}2. Generating pose files for world generation...${NC}"
mkdir -p assets/pose
python3 scripts/pose_generator.py --output_dir ./assets/pose

echo -e "\n${GREEN}3. Creating placeholder reference images directory...${NC}"
mkdir -p assets/reference
echo "Place reference images here for world generation" > assets/reference/README.md

echo -e "\n${GREEN}4. Checking .env file...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env and add your API keys.${NC}"
else
    echo ".env file exists"
fi

echo -e "\n${GREEN}5. Checking frontend dependencies...${NC}"
if [ -d "aitutor/frontend/node_modules" ]; then
    echo "Frontend dependencies installed"
else
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd aitutor/frontend
    npm install
    cd "$PROJECT_ROOT"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To run the development environment:"
echo ""
echo "  Terminal 1 - World Library Server:"
echo "    python3 scripts/serve_worlds.py --dir ./world_library"
echo ""
echo "  Terminal 2 - Frontend Dev Server:"
echo "    cd aitutor/frontend && npm run dev"
echo ""
echo "Then open your browser to:"
echo "  Main App:    http://localhost:5173"
echo "  World Demo:  http://localhost:5173/?demo=world"
echo ""
echo "Note: Videos need to be generated on a GPU server using:"
echo "  python3 scripts/generate_world_library.py --world solar_system"
echo ""
