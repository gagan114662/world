"""
Pose Generator for HY-WorldPlay

Generates camera pose JSON files for different actions:
- idle: no movement
- forward: move forward (+Z translation)
- backward: move backward (-Z translation)
- left: strafe left (-X translation)
- right: strafe right (+X translation)
- turn_left: rotate left (Y rotation)
- turn_right: rotate right (Y rotation)
- look_up: rotate up (-X rotation)
- look_down: rotate down (+X rotation)
"""

import json
import math
import numpy as np
from pathlib import Path


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    """Create Y-axis rotation matrix"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    """Create X-axis rotation matrix"""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ])


def generate_pose_sequence(
    num_latents: int = 32,
    action: str = "idle",
    translation_speed: float = 0.08,
    rotation_speed: float = 0.02,  # radians per frame
) -> dict:
    """
    Generate pose sequence for a given action.

    Args:
        num_latents: Number of latent frames (video_length = (num_latents-1)*4 + 1)
        action: One of: idle, forward, backward, left, right, turn_left, turn_right, look_up, look_down
        translation_speed: Translation per frame (default 0.08)
        rotation_speed: Rotation per frame in radians (default ~1.15 degrees)

    Returns:
        Dictionary with pose data for each frame
    """
    # Camera intrinsics (standard values from HY-WorldPlay)
    K = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0]
    ]

    poses = {}

    for i in range(num_latents):
        # Start with identity matrix
        c2w = np.eye(4)

        if action == "idle":
            # No movement
            pass

        elif action == "forward":
            # Move forward (+Z)
            c2w[2, 3] = i * translation_speed

        elif action == "backward":
            # Move backward (-Z)
            c2w[2, 3] = -i * translation_speed

        elif action == "left":
            # Strafe left (-X)
            c2w[0, 3] = -i * translation_speed

        elif action == "right":
            # Strafe right (+X)
            c2w[0, 3] = i * translation_speed

        elif action == "turn_left":
            # Rotate left (positive Y rotation)
            angle = i * rotation_speed
            rot = rotation_matrix_y(angle)
            c2w = rot @ c2w

        elif action == "turn_right":
            # Rotate right (negative Y rotation)
            angle = -i * rotation_speed
            rot = rotation_matrix_y(angle)
            c2w = rot @ c2w

        elif action == "look_up":
            # Rotate up (negative X rotation)
            angle = -i * rotation_speed
            rot = rotation_matrix_x(angle)
            c2w = rot @ c2w

        elif action == "look_down":
            # Rotate down (positive X rotation)
            angle = i * rotation_speed
            rot = rotation_matrix_x(angle)
            c2w = rot @ c2w

        elif action == "forward_left":
            # Diagonal forward-left
            c2w[2, 3] = i * translation_speed * 0.707  # cos(45)
            c2w[0, 3] = -i * translation_speed * 0.707

        elif action == "forward_right":
            # Diagonal forward-right
            c2w[2, 3] = i * translation_speed * 0.707
            c2w[0, 3] = i * translation_speed * 0.707

        poses[str(i)] = {
            "extrinsic": c2w.tolist(),
            "K": K
        }

    return poses


def save_pose_file(poses: dict, output_path: Path):
    """Save poses to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(poses, f, indent=4)
    print(f"Saved pose file: {output_path}")


def generate_all_action_poses(output_dir: Path, num_latents: int = 32):
    """Generate pose files for all actions"""
    output_dir.mkdir(parents=True, exist_ok=True)

    actions = [
        "idle",
        "forward",
        "backward",
        "left",
        "right",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
        "forward_left",
        "forward_right",
    ]

    for action in actions:
        poses = generate_pose_sequence(num_latents=num_latents, action=action)
        output_path = output_dir / f"{action}_{num_latents}_latents.json"
        save_pose_file(poses, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pose files for HY-WorldPlay")
    parser.add_argument("--output_dir", type=str, default="./assets/pose",
                        help="Output directory for pose files")
    parser.add_argument("--num_latents", type=int, default=32,
                        help="Number of latent frames (video_length = (num_latents-1)*4 + 1)")
    parser.add_argument("--action", type=str, default=None,
                        help="Generate for specific action only")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.action:
        poses = generate_pose_sequence(num_latents=args.num_latents, action=args.action)
        output_path = output_dir / f"{args.action}_{args.num_latents}_latents.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pose_file(poses, output_path)
    else:
        generate_all_action_poses(output_dir, args.num_latents)
