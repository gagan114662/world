#!/bin/bash
#
# RunPod World Generation Script
#
# This script is designed to be run on a RunPod H100 GPU instance.
# It downloads the necessary models and generates the world library.
#
# Usage:
#   1. Create a RunPod H100 pod with pytorch template
#   2. SSH into the pod
#   3. Copy this script and run it
#
# Estimated time: ~8-10 hours for all 10 worlds
# Estimated cost: ~$20-30 (H100 at $2.69/hr)
#

set -e

echo "=================================================="
echo "WorldPlay World Library Generator"
echo "=================================================="
echo ""

# Configuration
WORKSPACE="/workspace"
HYWORLD_DIR="$WORKSPACE/HY-WorldPlay"
OUTPUT_DIR="$WORKSPACE/world_library"
CKPTS_DIR="$HYWORLD_DIR/ckpts"

# Clone HY-WorldPlay if not exists
if [ ! -d "$HYWORLD_DIR" ]; then
    echo "Cloning HY-WorldPlay..."
    cd $WORKSPACE
    git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git
fi

cd $HYWORLD_DIR

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
pip install -q huggingface_hub

# Download models if not exists
if [ ! -d "$CKPTS_DIR/hunyuan-video-1.5" ]; then
    echo "Downloading HunyuanVideo-1.5 model..."
    huggingface-cli download tencent/HunyuanVideo-1.5 --local-dir $CKPTS_DIR/hunyuan-video-1.5
fi

if [ ! -d "$CKPTS_DIR/worldplay" ]; then
    echo "Downloading HY-WorldPlay model..."
    huggingface-cli download tencent/HY-WorldPlay \
        --include "HY-World1.5-Autoregressive-480P-I2V/*" \
        --local-dir $CKPTS_DIR/worldplay
fi

echo "Models downloaded successfully!"

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy generation scripts from world_play repo
if [ -f "/workspace/world_play/scripts/generate_world_library.py" ]; then
    cp /workspace/world_play/scripts/generate_world_library.py $WORKSPACE/
    cp /workspace/world_play/scripts/pose_generator.py $WORKSPACE/
fi

# Create reference images directory
mkdir -p $WORKSPACE/assets/reference

# Generate placeholder reference images (you should replace these with actual images)
echo "Creating placeholder reference images..."
for world in solar_system human_cell ancient_rome ocean_deep volcano_interior rainforest_canopy dna_helix medieval_castle atom_structure egyptian_pyramid; do
    # Use a simple colored placeholder (in production, use actual reference images)
    if [ ! -f "$WORKSPACE/assets/reference/${world}.jpg" ]; then
        convert -size 1280x720 xc:navy -fill white -gravity center -pointsize 48 -annotate 0 "$world" "$WORKSPACE/assets/reference/${world}.jpg" 2>/dev/null || echo "ImageMagick not installed, using fallback"
    fi
done

# Export environment variables
export HYWORLD_PATH=$HYWORLD_DIR
export MODEL_PATH=$CKPTS_DIR/hunyuan-video-1.5
export ACTION_CKPT=$CKPTS_DIR/worldplay
export OUTPUT_BASE=$OUTPUT_DIR

echo ""
echo "=================================================="
echo "Starting World Generation"
echo "=================================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Model path: $MODEL_PATH"
echo "Action checkpoint: $ACTION_CKPT"
echo ""

# Generate worlds one at a time (adjust as needed)
WORLDS=(
    "solar_system"
    "human_cell"
    "ancient_rome"
    "ocean_deep"
    "volcano_interior"
    "rainforest_canopy"
    "dna_helix"
    "medieval_castle"
    "atom_structure"
    "egyptian_pyramid"
)

# Parse command line arguments
WORLD_ARG=""
if [ ! -z "$1" ]; then
    WORLD_ARG=$1
fi

if [ ! -z "$WORLD_ARG" ]; then
    echo "Generating single world: $WORLD_ARG"
    python $WORKSPACE/generate_world_library.py \
        --world $WORLD_ARG \
        --output $OUTPUT_DIR \
        --depth 1  # Start with depth 1 for faster initial generation
else
    echo "Generating all worlds..."
    for world in "${WORLDS[@]}"; do
        echo ""
        echo "=================================================="
        echo "Generating: $world"
        echo "=================================================="

        python $WORKSPACE/generate_world_library.py \
            --world $world \
            --output $OUTPUT_DIR \
            --depth 1  # Start with depth 1

        echo "Completed: $world"
    done
fi

echo ""
echo "=================================================="
echo "Generation Complete!"
echo "=================================================="
echo ""
echo "Output saved to: $OUTPUT_DIR"
echo ""
echo "To download the generated worlds:"
echo "  rsync -avz root@\$POD_IP:$OUTPUT_DIR ./world_library"
echo ""
echo "Or upload to cloud storage:"
echo "  aws s3 sync $OUTPUT_DIR s3://your-bucket/world_library"
echo ""
