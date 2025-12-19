#!/usr/bin/env python3
"""
Create mock world data for local development and testing.

This creates metadata.json files and placeholder info for each world,
allowing the frontend to be tested without actual video generation.

Usage:
    python create_mock_worlds.py
    python create_mock_worlds.py --output ./world_library
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime

# World definitions (same as generate_world_library.py)
WORLDS = [
    {
        "id": "solar_system",
        "name": "Solar System Explorer",
        "subject": "astronomy",
        "description": "Explore our solar system from Mercury to Neptune",
        "prompt": "Flying through outer space with stars and nebulae in the background, approaching planets in our solar system",
        "positions": {
            "start": "Floating in space, Sun visible in the distance",
            "forward": "Approaching Mercury, the closest planet to the Sun",
            "forward_forward": "Flying past Mercury, Venus coming into view",
            "forward_left": "Turning toward the asteroid belt",
            "forward_right": "Viewing the outer planets in the distance",
            "left": "Looking at the constellation Orion",
            "right": "Mars visible in the distance with its red surface",
            "turn_left": "Rotating to view different star clusters",
            "turn_right": "Turning to see Jupiter and its moons",
            "backward": "Moving back toward the outer solar system",
        },
        "educationalPoints": [
            "Mercury is the smallest planet and closest to the Sun",
            "Venus is the hottest planet due to its thick atmosphere",
            "Earth is the only planet known to have liquid water on its surface",
            "Mars has the largest volcano in the solar system - Olympus Mons",
        ],
        "vocabulary": ["orbit", "gravity", "asteroid", "comet", "nebula"],
        "suggestedQuestions": [
            "Why is Mercury not the hottest planet?",
            "What makes Earth special?",
            "How big is Jupiter compared to Earth?",
        ],
    },
    {
        "id": "human_cell",
        "name": "Inside a Human Cell",
        "subject": "biology",
        "description": "Explore the interior of a living cell",
        "prompt": "Floating inside a human cell with organelles visible",
        "positions": {
            "start": "Near the cell membrane, gateway to the cell",
            "forward": "Moving toward the nucleus, the control center",
            "forward_forward": "Approaching the nuclear envelope",
            "left": "Floating toward mitochondria, the powerhouses",
            "right": "Approaching the endoplasmic reticulum",
            "turn_left": "Viewing ribosomes scattered throughout",
            "turn_right": "Looking at the Golgi apparatus",
        },
        "educationalPoints": [
            "The cell membrane controls what enters and leaves",
            "The nucleus contains DNA, the genetic blueprint",
            "Mitochondria produce ATP, the energy currency",
        ],
        "vocabulary": ["organelle", "cytoplasm", "membrane", "nucleus", "mitochondria"],
        "suggestedQuestions": [
            "Why are mitochondria called powerhouses?",
            "What is DNA and where is it found?",
        ],
    },
    {
        "id": "ancient_rome",
        "name": "Ancient Roman Forum",
        "subject": "history",
        "description": "Walk through the heart of the Roman Empire",
        "prompt": "Walking through the ancient Roman Forum at its height",
        "positions": {
            "start": "At the entrance to the Roman Forum",
            "forward": "Walking toward the Senate building",
            "forward_forward": "Standing before the Curia Julia",
            "left": "Approaching the Temple of Saturn",
            "right": "Heading toward the Colosseum",
            "turn_left": "Viewing the Arch of Titus",
            "turn_right": "Looking at merchant stalls",
        },
        "educationalPoints": [
            "The Roman Forum was the center of public life",
            "The Senate made laws and governed the Republic",
            "Roman architecture influenced buildings for centuries",
        ],
        "vocabulary": ["senate", "republic", "emperor", "gladiator", "forum"],
        "suggestedQuestions": [
            "How was the Roman Republic governed?",
            "What happened in the Colosseum?",
        ],
    },
]


def create_mock_world(world: dict, output_dir: Path) -> None:
    """Create mock data for a single world."""
    world_dir = output_dir / world["id"]
    world_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata.json
    metadata = {
        "world": {
            "id": world["id"],
            "name": world["name"],
            "subject": world["subject"],
            "description": world["description"],
            "prompt": world.get("prompt", ""),
            "positions": world.get("positions", {}),
            "educationalPoints": world.get("educationalPoints", []),
            "vocabulary": world.get("vocabulary", []),
            "suggestedQuestions": world.get("suggestedQuestions", []),
        },
        "generated_at": datetime.now().isoformat(),
        "is_mock": True,
        "settings": {
            "video_length": 125,
            "height": 480,
            "width": 832,
        },
    }

    metadata_path = world_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created: {metadata_path}")

    # Create a placeholder README for the world
    readme_path = world_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# {world['name']}\n\n")
        f.write(f"Subject: {world['subject']}\n\n")
        f.write(f"## Description\n{world['description']}\n\n")
        f.write("## Video Segments Needed\n\n")
        f.write("Generate these video segments using `generate_world_library.py`:\n\n")
        f.write("- `start.mp4` - Entry point\n")
        for action in ["forward", "backward", "left", "right", "turn_left", "turn_right"]:
            f.write(f"- `from_start_{action}.mp4`\n")
        f.write("\n")

    print(f"Created: {readme_path}")


def create_index_file(worlds: list, output_dir: Path) -> None:
    """Create an index.json file listing all available worlds."""
    index = {
        "worlds": [
            {
                "id": w["id"],
                "name": w["name"],
                "subject": w["subject"],
                "description": w["description"],
                "thumbnail": f"{w['id']}/thumbnail.jpg",
            }
            for w in worlds
        ],
        "generated_at": datetime.now().isoformat(),
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nCreated index: {index_path}")


def main():
    parser = argparse.ArgumentParser(description="Create mock world data for development")
    parser.add_argument(
        "--output",
        type=str,
        default="./world_library",
        help="Output directory for world data",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Creating Mock World Data")
    print(f"{'='*60}\n")
    print(f"Output directory: {output_dir.absolute()}\n")

    for world in WORLDS:
        create_mock_world(world, output_dir)
        print()

    create_index_file(WORLDS, output_dir)

    print(f"\n{'='*60}")
    print("Mock Data Created Successfully!")
    print(f"{'='*60}")
    print(f"\nTo serve the world library:")
    print(f"  python scripts/serve_worlds.py --dir {output_dir}")
    print(f"\nThe frontend will load metadata from:")
    print(f"  http://localhost:8010/worlds/solar_system/metadata.json")
    print(f"\nNote: Video segments (.mp4 files) still need to be generated")
    print(f"using generate_world_library.py on a GPU server.\n")


if __name__ == "__main__":
    main()
