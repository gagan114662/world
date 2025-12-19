"""
Inference Engine - wraps HY-WorldPlay for video generation.

This module interfaces with the HY-WorldPlay pipeline to generate
video frames based on prompts and user actions.
"""

import asyncio
import os
import sys
from typing import AsyncGenerator, Dict, Any, Optional, List
import numpy as np

# Add HY-WorldPlay to path
HY_WORLDPLAY_PATH = os.getenv("HY_WORLDPLAY_PATH", "../HY-WorldPlay")
sys.path.insert(0, HY_WORLDPLAY_PATH)


class WorldPlayInferenceEngine:
    """
    Wraps HY-WorldPlay pipeline for real-time video generation.

    Handles:
    - Prompt processing
    - Camera trajectory generation
    - Action encoding for dual representation
    - Frame generation with streaming output
    """

    # Action encoding mapping (keyboard state to action index)
    ACTION_MAPPING = {
        "still": 0,
        "forward": 1,
        "backward": 2,
        "left": 3,
        "right": 4,
        "forward_left": 5,
        "forward_right": 6,
        "backward_left": 7,
        "backward_right": 8
    }

    def __init__(self, model_manager):
        """
        Initialize the inference engine.

        Args:
            model_manager: ModelManager instance with loaded models
        """
        self.model_manager = model_manager
        self.pipeline = None
        self.chunk_size = 16  # Frames per generation chunk
        self.fps = 24

    async def initialize(self):
        """Initialize the inference pipeline."""
        if not self.model_manager.models_loaded:
            raise RuntimeError("Models not loaded. Call model_manager.load_models() first.")

        # Get pipeline from model manager
        self.pipeline = self.model_manager.get_pipeline()

    async def generate(
        self,
        prompt: str,
        camera_trajectory: Dict[str, Any],
        action_queue: asyncio.Queue,
        num_frames: int = 0,  # 0 = infinite streaming
        reference_image: Optional[np.ndarray] = None
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate video frames based on prompt and user actions.

        Args:
            prompt: Text description of the world to generate
            camera_trajectory: Camera movement configuration
            action_queue: Queue of user actions to process
            num_frames: Number of frames to generate (0 for infinite)
            reference_image: Optional reference image for I2V mode

        Yields:
            numpy arrays of video frames (H, W, 3) in RGB format
        """
        if self.pipeline is None:
            await self.initialize()

        # Build initial camera trajectory
        trajectory = self._build_trajectory(camera_trajectory, self.chunk_size)

        # Context frames for continuation
        context_frames: List[np.ndarray] = []

        frame_count = 0
        generating = True

        while generating:
            # Check for user actions
            current_action = await self._get_latest_action(action_queue)

            # Encode action for HY-WorldPlay
            action_encoding = self._encode_action(current_action)

            # Update trajectory based on action
            if current_action:
                trajectory = self._update_trajectory(trajectory, current_action)

            try:
                # Generate chunk of frames
                # In production, this calls the actual HY-WorldPlay pipeline:
                #
                # frames = self.pipeline(
                #     prompt=prompt,
                #     context_frames=context_frames[-4:] if context_frames else None,
                #     pose_json=trajectory,
                #     actions=action_encoding,
                #     num_frames=self.chunk_size,
                #     num_inference_steps=4,  # Use distilled model
                # )

                # Simulation: Generate placeholder frames
                # Replace this with actual pipeline call in production
                frames = await self._simulate_generation(
                    prompt=prompt,
                    action=current_action,
                    frame_start=frame_count
                )

                # Yield each frame
                for frame in frames:
                    yield frame
                    frame_count += 1

                    if num_frames > 0 and frame_count >= num_frames:
                        generating = False
                        break

                # Update context for next chunk
                context_frames.extend(frames)
                # Keep only recent context
                if len(context_frames) > 32:
                    context_frames = context_frames[-32:]

            except Exception as e:
                print(f"Generation error: {e}")
                # On error, yield a blank frame and continue
                yield np.zeros((480, 854, 3), dtype=np.uint8)
                await asyncio.sleep(1.0 / self.fps)

    async def _get_latest_action(self, action_queue: asyncio.Queue) -> Optional[Dict]:
        """Get the latest action from the queue, draining old actions."""
        latest_action = None
        try:
            while True:
                action = action_queue.get_nowait()
                latest_action = action
        except asyncio.QueueEmpty:
            pass
        return latest_action

    def _encode_action(self, action: Optional[Dict]) -> np.ndarray:
        """
        Encode user action to HY-WorldPlay's dual action representation.

        Returns one-hot encoded action vector.
        """
        if action is None:
            # No action = still
            action_idx = self.ACTION_MAPPING["still"]
        else:
            keyboard = action.get("keyboard", {})

            # Determine action from keyboard state
            forward = keyboard.get("w", False)
            backward = keyboard.get("s", False)
            left = keyboard.get("a", False)
            right = keyboard.get("d", False)

            if forward and left:
                action_name = "forward_left"
            elif forward and right:
                action_name = "forward_right"
            elif backward and left:
                action_name = "backward_left"
            elif backward and right:
                action_name = "backward_right"
            elif forward:
                action_name = "forward"
            elif backward:
                action_name = "backward"
            elif left:
                action_name = "left"
            elif right:
                action_name = "right"
            else:
                action_name = "still"

            action_idx = self.ACTION_MAPPING[action_name]

        # One-hot encoding
        encoding = np.zeros(9, dtype=np.float32)
        encoding[action_idx] = 1.0

        return encoding

    def _build_trajectory(
        self,
        camera_config: Dict[str, Any],
        num_frames: int
    ) -> Dict[str, Any]:
        """
        Build camera trajectory JSON for HY-WorldPlay.

        Args:
            camera_config: Camera configuration from request
            num_frames: Number of frames in the trajectory

        Returns:
            Trajectory dict in HY-WorldPlay format
        """
        initial_position = camera_config.get("initial_position", [0, 0, 0])
        initial_rotation = camera_config.get("initial_rotation", [0, 0, 0])

        # Build W2C matrices for each frame
        # In production, use proper matrix math
        frames = []
        for i in range(num_frames):
            frames.append({
                "frame_idx": i,
                "w2c": self._position_to_w2c(initial_position, initial_rotation),
                "intrinsics": {
                    "fx": 500,
                    "fy": 500,
                    "cx": 427,
                    "cy": 240
                }
            })

        return {
            "mode": camera_config.get("mode", "first_person"),
            "fov": camera_config.get("fov", 60),
            "frames": frames
        }

    def _update_trajectory(
        self,
        trajectory: Dict[str, Any],
        action: Dict
    ) -> Dict[str, Any]:
        """
        Update camera trajectory based on user action.

        Applies movement and rotation from action to the trajectory.
        """
        keyboard = action.get("keyboard", {})
        mouse = action.get("mouse", {})

        # Movement speed
        move_speed = 0.1
        rotate_speed = 2.0

        # Calculate movement delta
        dx, dy, dz = 0, 0, 0
        if keyboard.get("w"):
            dz -= move_speed
        if keyboard.get("s"):
            dz += move_speed
        if keyboard.get("a"):
            dx -= move_speed
        if keyboard.get("d"):
            dx += move_speed
        if keyboard.get("space"):
            dy += move_speed
        if keyboard.get("shift"):
            dy -= move_speed

        # Calculate rotation from mouse
        yaw_delta = mouse.get("dx", 0) * rotate_speed
        pitch_delta = mouse.get("dy", 0) * rotate_speed

        # Update trajectory frames
        # This is simplified - in production, use proper matrix operations
        for frame in trajectory.get("frames", []):
            # Update position (simplified)
            pass

        return trajectory

    def _position_to_w2c(
        self,
        position: List[float],
        rotation: List[float]
    ) -> List[List[float]]:
        """
        Convert position and rotation to world-to-camera matrix.

        Returns 4x4 transformation matrix.
        """
        # Simplified identity matrix
        # In production, compute proper rotation + translation matrix
        return [
            [1, 0, 0, -position[0]],
            [0, 1, 0, -position[1]],
            [0, 0, 1, -position[2]],
            [0, 0, 0, 1]
        ]

    async def _simulate_generation(
        self,
        prompt: str,
        action: Optional[Dict],
        frame_start: int
    ) -> List[np.ndarray]:
        """
        Simulate frame generation for testing.

        In production, this is replaced by actual HY-WorldPlay inference.
        """
        frames = []

        # Simulate generation time
        await asyncio.sleep(0.5)  # ~500ms for 16 frames

        for i in range(self.chunk_size):
            # Create a gradient frame based on frame number
            frame_num = frame_start + i
            frame = np.zeros((480, 854, 3), dtype=np.uint8)

            # Color based on time
            r = int(128 + 127 * np.sin(frame_num * 0.1))
            g = int(128 + 127 * np.sin(frame_num * 0.1 + 2))
            b = int(128 + 127 * np.sin(frame_num * 0.1 + 4))

            frame[:, :] = [r, g, b]

            # Add action indicator
            if action:
                keyboard = action.get("keyboard", {})
                if keyboard.get("w"):
                    frame[:50, :] = [0, 255, 0]  # Green top = forward
                if keyboard.get("s"):
                    frame[-50:, :] = [255, 0, 0]  # Red bottom = backward

            frames.append(frame)

        return frames
