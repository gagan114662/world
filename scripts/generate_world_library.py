#!/usr/bin/env python3
"""
World Library Generator for Immersive AI Tutor

FULLY DYNAMIC - Generates ANY world, not just predefined ones.

Generates pre-rendered educational worlds with branching paths for interactive playback.
Each world has multiple navigation paths that students can choose from.

Usage:
    # Generate predefined world
    python generate_world_library.py --world solar_system

    # Generate ALL predefined worlds
    python generate_world_library.py --all

    # Generate CUSTOM world (any topic!)
    python generate_world_library.py --custom \
        --name "Black Hole Journey" \
        --subject "astrophysics" \
        --prompt "Flying toward a supermassive black hole, accretion disk glowing, gravitational lensing visible" \
        --reference path/to/image.jpg

    # Generate from JSON config
    python generate_world_library.py --config my_world.json

Requirements:
    - HY-WorldPlay installed and configured
    - GPU with sufficient VRAM (recommended: H100 80GB)
    - Model checkpoints downloaded
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import shutil

# Add parent directory to path for pose_generator import
sys.path.insert(0, str(Path(__file__).parent))
from pose_generator import generate_pose_sequence, save_pose_file


# Configuration
HYWORLD_PATH = os.environ.get("HYWORLD_PATH", "/workspace/HY-WorldPlay")
MODEL_PATH = os.environ.get("MODEL_PATH", f"{HYWORLD_PATH}/ckpts/hunyuan-video-1.5")
ACTION_CKPT = os.environ.get("ACTION_CKPT", f"{HYWORLD_PATH}/ckpts/worldplay")
OUTPUT_BASE = os.environ.get("OUTPUT_BASE", "/workspace/world_library")

# Video settings
NUM_LATENTS = 32  # (32-1)*4 + 1 = 125 frames = ~5 seconds at 25 FPS
VIDEO_LENGTH = 125
NUM_INFERENCE_STEPS = 50  # Full quality
HEIGHT = 480
WIDTH = 832

# Actions for branching paths
ACTIONS = ["idle", "forward", "backward", "left", "right", "turn_left", "turn_right"]


# Educational Worlds Definition
WORLDS: List[Dict] = [
    {
        "id": "solar_system",
        "name": "Solar System Explorer",
        "subject": "astronomy",
        "description": "Explore our solar system from Mercury to Neptune",
        "prompt": "Flying through outer space with stars and nebulae in the background, "
                  "approaching planets in our solar system, cinematic lighting, "
                  "photorealistic, high detail, 8K quality",
        "reference_image": "solar_system.jpg",
        "positions": {
            "start": "Floating in space near the Sun",
            "forward": "Approaching Mercury",
            "forward_forward": "Flying past Mercury toward Venus",
            "forward_left": "Veering toward the asteroid belt",
            "left": "Turning to view distant galaxies",
            "right": "Looking at Mars in the distance",
        }
    },
    {
        "id": "human_cell",
        "name": "Inside a Human Cell",
        "subject": "biology",
        "description": "Explore the interior of a living cell",
        "prompt": "Floating inside a human cell, translucent membrane visible, "
                  "mitochondria glowing with energy, nucleus in the center, "
                  "endoplasmic reticulum networks, ribosomes, soft bioluminescent lighting, "
                  "microscopic scale, photorealistic",
        "reference_image": "cell_interior.jpg",
        "positions": {
            "start": "Near the cell membrane",
            "forward": "Moving toward the nucleus",
            "forward_forward": "Approaching the nuclear envelope",
            "left": "Floating toward mitochondria",
            "right": "Approaching the endoplasmic reticulum",
        }
    },
    {
        "id": "ancient_rome",
        "name": "Ancient Roman Forum",
        "subject": "history",
        "description": "Walk through the heart of the Roman Empire",
        "prompt": "Walking through the ancient Roman Forum at its height, "
                  "magnificent marble temples, columns and arches, "
                  "Roman citizens in togas, sunny Mediterranean day, "
                  "photorealistic historical reconstruction, golden hour lighting",
        "reference_image": "roman_forum.jpg",
        "positions": {
            "start": "Entrance to the Forum",
            "forward": "Walking toward the Senate building",
            "left": "Approaching the Temple of Saturn",
            "right": "Heading toward the Colosseum",
        }
    },
    {
        "id": "ocean_deep",
        "name": "Deep Ocean Explorer",
        "subject": "marine_biology",
        "description": "Dive into the mysterious deep ocean",
        "prompt": "Submarine descending into the deep ocean, bioluminescent creatures, "
                  "giant squid in the distance, underwater canyon walls, "
                  "rays of light filtering from above, ethereal blue lighting, "
                  "photorealistic underwater photography style",
        "reference_image": "deep_ocean.jpg",
        "positions": {
            "start": "Descending through the twilight zone",
            "forward": "Going deeper toward the abyssal plain",
            "left": "Approaching a hydrothermal vent",
            "right": "Swimming toward a whale carcass ecosystem",
        }
    },
    {
        "id": "volcano_interior",
        "name": "Inside a Volcano",
        "subject": "geology",
        "description": "Explore the interior of an active volcano",
        "prompt": "Flying through an active volcano interior, magma chambers glowing, "
                  "lava flows, volcanic rock formations, extreme heat distortion, "
                  "orange and red dramatic lighting, photorealistic geological detail",
        "reference_image": "volcano_interior.jpg",
        "positions": {
            "start": "Near the volcano rim looking down",
            "forward": "Descending into the crater",
            "forward_forward": "Approaching the magma chamber",
            "left": "Viewing ancient lava tubes",
            "right": "Observing the volcanic vent",
        }
    },
    {
        "id": "rainforest_canopy",
        "name": "Amazon Rainforest Canopy",
        "subject": "ecology",
        "description": "Explore the world's most biodiverse ecosystem",
        "prompt": "Flying through the Amazon rainforest canopy, dense green foliage, "
                  "exotic birds and butterflies, monkeys swinging from vines, "
                  "sunlight filtering through leaves, mist and humidity visible, "
                  "photorealistic nature documentary style",
        "reference_image": "rainforest_canopy.jpg",
        "positions": {
            "start": "Emerging through the forest floor",
            "forward": "Rising up through the understory",
            "forward_forward": "Breaking through to the canopy",
            "left": "Following a river through the forest",
            "right": "Approaching a massive ceiba tree",
        }
    },
    {
        "id": "dna_helix",
        "name": "DNA Double Helix",
        "subject": "molecular_biology",
        "description": "Journey along the blueprint of life",
        "prompt": "Flying along a DNA double helix structure at molecular scale, "
                  "base pairs visible (adenine, thymine, guanine, cytosine), "
                  "sugar-phosphate backbone spiraling, protein enzymes nearby, "
                  "soft blue and purple scientific visualization lighting",
        "reference_image": "dna_helix.jpg",
        "positions": {
            "start": "Outside the DNA molecule",
            "forward": "Traveling along the major groove",
            "left": "Viewing the minor groove",
            "right": "Approaching a replication fork",
        }
    },
    {
        "id": "medieval_castle",
        "name": "Medieval Castle",
        "subject": "medieval_history",
        "description": "Explore a medieval fortress",
        "prompt": "Walking through a grand medieval castle, stone walls and towers, "
                  "banners and tapestries, knights in armor, great hall with fireplace, "
                  "torchlight and candlelight, photorealistic historical reconstruction",
        "reference_image": "medieval_castle.jpg",
        "positions": {
            "start": "Crossing the drawbridge",
            "forward": "Entering the main courtyard",
            "forward_forward": "Approaching the keep",
            "left": "Walking toward the armory",
            "right": "Heading to the great hall",
        }
    },
    {
        "id": "atom_structure",
        "name": "Inside an Atom",
        "subject": "physics",
        "description": "Explore the quantum realm of atoms",
        "prompt": "Flying inside an atom, electron cloud probability distributions, "
                  "nucleus with protons and neutrons visible, quantum orbital shells, "
                  "abstract scientific visualization, glowing particles, "
                  "deep blue and purple quantum aesthetic",
        "reference_image": "atom_structure.jpg",
        "positions": {
            "start": "Outside the electron cloud",
            "forward": "Passing through the electron shells",
            "forward_forward": "Approaching the nucleus",
            "left": "Viewing the s orbital",
            "right": "Observing the p orbital shapes",
        }
    },
    {
        "id": "egyptian_pyramid",
        "name": "Egyptian Pyramid Interior",
        "subject": "ancient_history",
        "description": "Explore the mysteries of the Great Pyramid",
        "prompt": "Walking through the interior passages of the Great Pyramid of Giza, "
                  "ancient stone blocks, hieroglyphics on walls, burial chamber ahead, "
                  "flickering torchlight, mysterious atmosphere, "
                  "photorealistic archaeological style",
        "reference_image": "egyptian_pyramid.jpg",
        "positions": {
            "start": "At the pyramid entrance",
            "forward": "Walking up the grand gallery",
            "forward_forward": "Approaching the King's Chamber",
            "left": "Entering the Queen's Chamber passage",
            "right": "Discovering a hidden passage",
        }
    },
]


def get_world_by_id(world_id: str) -> Optional[Dict]:
    """Get world configuration by ID"""
    for world in WORLDS:
        if world["id"] == world_id:
            return world
    return None


def generate_segment(
    world: Dict,
    action: str,
    pose_file: Path,
    output_path: Path,
    reference_image: Path,
) -> bool:
    """
    Generate a single video segment for a world+action combination.

    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Generating: {world['name']} - Action: {action}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    cmd = [
        "python", f"{HYWORLD_PATH}/generate.py",
        "--prompt", world["prompt"],
        "--image_path", str(reference_image),
        "--pose_json_path", str(pose_file),
        "--model_path", MODEL_PATH,
        "--action_ckpt", ACTION_CKPT,
        "--resolution", "480p",
        "--model_type", "ar",  # Autoregressive for I2V
        "--video_length", str(VIDEO_LENGTH),
        "--num_inference_steps", str(NUM_INFERENCE_STEPS),
        "--height", str(HEIGHT),
        "--width", str(WIDTH),
        "--output_path", str(output_path.parent),
        "--sr", "false",  # Skip super-resolution for now
        "--rewrite", "false",
        "--offloading", "true",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=HYWORLD_PATH,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per segment
        )

        if result.returncode != 0:
            print(f"ERROR generating {world['id']}/{action}:")
            print(result.stderr)
            return False

        # Rename output file to proper name
        generated_file = output_path.parent / "gen.mp4"
        if generated_file.exists():
            shutil.move(str(generated_file), str(output_path))
            print(f"SUCCESS: Saved {output_path}")
            return True
        else:
            print(f"ERROR: Output file not found at {generated_file}")
            return False

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT generating {world['id']}/{action}")
        return False
    except Exception as e:
        print(f"EXCEPTION generating {world['id']}/{action}: {e}")
        return False


def generate_world(
    world: Dict,
    output_base: Path,
    pose_dir: Path,
    reference_dir: Path,
    actions: List[str] = ACTIONS,
    depth: int = 2,
) -> Dict:
    """
    Generate all video segments for a world.

    Args:
        world: World configuration dictionary
        output_base: Base output directory
        pose_dir: Directory containing pose JSON files
        reference_dir: Directory containing reference images
        actions: List of actions to generate
        depth: How many levels of branching to generate

    Returns:
        Dictionary with generation status for each segment
    """
    world_dir = output_base / world["id"]
    world_dir.mkdir(parents=True, exist_ok=True)

    # Reference image
    ref_image = reference_dir / world["reference_image"]
    if not ref_image.exists():
        print(f"WARNING: Reference image not found: {ref_image}")
        print("Using placeholder - you should provide a proper reference image")
        # Create a simple placeholder message
        ref_image = reference_dir / "placeholder.jpg"

    results = {"world_id": world["id"], "segments": {}}

    # Generate starting segment (idle - establishes the scene)
    pose_file = pose_dir / f"idle_{NUM_LATENTS}_latents.json"
    if not pose_file.exists():
        generate_pose_sequence(NUM_LATENTS, "idle")
        save_pose_file(generate_pose_sequence(NUM_LATENTS, "idle"), pose_file)

    start_output = world_dir / "start.mp4"
    success = generate_segment(world, "idle", pose_file, start_output, ref_image)
    results["segments"]["start"] = success

    # Generate first-level branches (from start)
    for action in actions:
        pose_file = pose_dir / f"{action}_{NUM_LATENTS}_latents.json"
        if not pose_file.exists():
            save_pose_file(generate_pose_sequence(NUM_LATENTS, action), pose_file)

        segment_output = world_dir / f"from_start_{action}.mp4"

        # Use the last frame of start video as reference for continuity
        # For now, use the same reference image (in production, extract last frame)
        success = generate_segment(world, action, pose_file, segment_output, ref_image)
        results["segments"][f"from_start_{action}"] = success

    # Generate second-level branches (depth=2)
    if depth >= 2:
        for first_action in actions:
            for second_action in actions:
                segment_name = f"from_{first_action}_{second_action}"
                pose_file = pose_dir / f"{second_action}_{NUM_LATENTS}_latents.json"
                segment_output = world_dir / f"{segment_name}.mp4"

                success = generate_segment(world, second_action, pose_file, segment_output, ref_image)
                results["segments"][segment_name] = success

    # Save world metadata
    metadata = {
        "world": world,
        "generated_at": datetime.now().isoformat(),
        "settings": {
            "num_latents": NUM_LATENTS,
            "video_length": VIDEO_LENGTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "height": HEIGHT,
            "width": WIDTH,
        },
        "results": results,
    }

    metadata_path = world_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")

    return results


def create_custom_world(
    name: str,
    subject: str,
    prompt: str,
    description: str = None,
    reference_image: str = None,
) -> Dict:
    """Create a custom world configuration - works with ANY topic."""
    world_id = name.lower().replace(" ", "_").replace("-", "_")[:30]
    world_id = "".join(c for c in world_id if c.isalnum() or c == "_")

    return {
        "id": world_id,
        "name": name,
        "subject": subject,
        "description": description or f"Explore {name} in immersive 3D",
        "prompt": prompt,
        "reference_image": reference_image or f"{world_id}.jpg",
        "positions": {
            "start": f"Beginning your journey in {name}",
            "forward": f"Moving forward through {name}",
            "left": f"Exploring the left side of {name}",
            "right": f"Exploring the right side of {name}",
        },
    }


def load_world_from_json(config_path: Path) -> Dict:
    """Load world config from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
    required = ["name", "subject", "prompt"]
    for field in required:
        if field not in config:
            raise ValueError(f"Config missing required field: {field}")
    return create_custom_world(**config)


def main():
    parser = argparse.ArgumentParser(
        description="Generate World Library - Works with ANY topic!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predefined world
  python generate_world_library.py --world solar_system

  # ANY custom world
  python generate_world_library.py --custom \\
      --name "Black Hole Journey" \\
      --subject "astrophysics" \\
      --prompt "Flying toward a supermassive black hole, accretion disk glowing"

  # From JSON config
  python generate_world_library.py --config my_world.json
        """
    )

    # Predefined
    parser.add_argument("--world", type=str, help="Generate predefined world by ID")
    parser.add_argument("--all", action="store_true", help="Generate all predefined worlds")
    parser.add_argument("--list", action="store_true", help="List predefined worlds")

    # Custom world (ANY topic)
    parser.add_argument("--custom", action="store_true", help="Generate custom world")
    parser.add_argument("--name", type=str, help="Custom world name")
    parser.add_argument("--subject", type=str, help="Custom world subject")
    parser.add_argument("--prompt", type=str, help="Custom world visual prompt")
    parser.add_argument("--description", type=str, help="Custom world description")
    parser.add_argument("--reference", type=str, help="Reference image path")
    parser.add_argument("--config", type=str, help="JSON config file path")

    # Generation options
    parser.add_argument("--actions", type=str, default=None,
                        help="Comma-separated actions to generate")
    parser.add_argument("--depth", type=int, default=2,
                        help="Branching depth (default: 2)")
    parser.add_argument("--output", type=str, default=OUTPUT_BASE,
                        help="Output directory")
    parser.add_argument("--pose_dir", type=str, default="./assets/pose",
                        help="Pose files directory")
    parser.add_argument("--reference_dir", type=str, default="./assets/reference",
                        help="Reference images directory")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be generated")

    args = parser.parse_args()

    if args.list:
        print("\n" + "="*60)
        print("PREDEFINED WORLDS")
        print("="*60)
        for world in WORLDS:
            print(f"\n  {world['id']}: {world['name']}")
            print(f"    Subject: {world['subject']}")
        print("\n" + "="*60)
        print("CUSTOM WORLDS - Generate ANY topic:")
        print("="*60)
        print("\n  python generate_world_library.py --custom \\")
        print('      --name "Your Topic" \\')
        print('      --subject "subject" \\')
        print('      --prompt "Visual description..."')
        return

    output_base = Path(args.output)
    pose_dir = Path(args.pose_dir)
    reference_dir = Path(args.reference_dir)

    output_base.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    print("Generating pose files...")
    for action in ACTIONS:
        pose_file = pose_dir / f"{action}_{NUM_LATENTS}_latents.json"
        if not pose_file.exists():
            poses = generate_pose_sequence(NUM_LATENTS, action)
            save_pose_file(poses, pose_file)

    actions = args.actions.split(",") if args.actions else ACTIONS

    # Custom world from JSON
    if args.config:
        world = load_world_from_json(Path(args.config))
        print(f"\nLoaded custom world: {world['name']}")

    # Custom world from CLI
    elif args.custom:
        if not all([args.name, args.subject, args.prompt]):
            print("ERROR: --custom requires --name, --subject, and --prompt")
            sys.exit(1)
        world = create_custom_world(
            name=args.name, subject=args.subject, prompt=args.prompt,
            description=args.description, reference_image=args.reference,
        )
        print(f"\n{'='*60}")
        print(f"CUSTOM WORLD: {world['name']}")
        print(f"{'='*60}")

    # Predefined world
    elif args.world:
        world = get_world_by_id(args.world)
        if not world:
            print(f"ERROR: '{args.world}' not found. Use --list or --custom")
            sys.exit(1)

    # All predefined
    elif args.all:
        if args.dry_run:
            for w in WORLDS:
                segs = 1 + len(actions) + (len(actions)**2 if args.depth >= 2 else 0)
                print(f"  {w['id']}: {segs} segments")
            return

        for world in WORLDS:
            generate_world(world, output_base, pose_dir, reference_dir,
                          actions=actions, depth=args.depth)
        print("\nAll worlds generated!")
        return

    else:
        parser.print_help()
        return

    # Generate single world (custom or predefined)
    if args.dry_run:
        segs = 1 + len(actions) + (len(actions)**2 if args.depth >= 2 else 0)
        print(f"\nWould generate: {world['name']} ({segs} segments)")
        return

    results = generate_world(world, output_base, pose_dir, reference_dir,
                            actions=actions, depth=args.depth)
    success = sum(1 for v in results['segments'].values() if v)
    print(f"\nComplete: {success}/{len(results['segments'])} segments")


if __name__ == "__main__":
    main()
