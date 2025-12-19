"""
Model Manager - handles loading and management of HY-WorldPlay models.

This module manages:
- Model loading and initialization
- GPU memory management
- Model offloading for memory efficiency
"""

import asyncio
import os
import sys
from typing import Optional, Dict, Any

# Check for CUDA availability
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


class ModelManager:
    """
    Manages HY-WorldPlay model lifecycle.

    Handles model loading, GPU memory management, and provides
    access to the inference pipeline.
    """

    def __init__(
        self,
        model_path: str,
        action_ckpt: str,
        device: str = "cuda",
        model_offload: bool = True,
        use_flash_attn: bool = True
    ):
        """
        Initialize the model manager.

        Args:
            model_path: Path to HunyuanVideo-1.5 base model
            action_ckpt: Path to HY-WorldPlay action checkpoint
            device: Device to use ("cuda" or "cpu")
            model_offload: Enable model offloading for memory efficiency
            use_flash_attn: Use Flash Attention for faster inference
        """
        self.model_path = model_path
        self.action_ckpt = action_ckpt
        self.device = device if CUDA_AVAILABLE else "cpu"
        self.model_offload = model_offload
        self.use_flash_attn = use_flash_attn

        self.pipeline = None
        self.models_loaded = False
        self.gpu_available = CUDA_AVAILABLE

        # Model components
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.scheduler = None

    async def load_models(self):
        """
        Load HY-WorldPlay models.

        This is an async wrapper around the synchronous loading process.
        """
        if self.models_loaded:
            print("Models already loaded")
            return

        print(f"Loading models from {self.model_path}")
        print(f"Action checkpoint: {self.action_ckpt}")
        print(f"Device: {self.device}")
        print(f"Model offload: {self.model_offload}")

        # Run loading in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_models_sync)

        self.models_loaded = True
        print("Models loaded successfully!")

    def _load_models_sync(self):
        """
        Synchronous model loading.

        In production, this loads the actual HY-WorldPlay pipeline.
        """
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Running in simulation mode.")
            self._load_simulation_models()
            return

        try:
            # Add HY-WorldPlay to path
            hy_path = os.getenv("HY_WORLDPLAY_PATH", "../HY-WorldPlay")
            if hy_path not in sys.path:
                sys.path.insert(0, hy_path)

            # In production, load actual models:
            #
            # from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
            #
            # self.pipeline = HunyuanVideo_1_5_Pipeline.create_pipeline(
            #     model_path=self.model_path,
            #     action_ckpt=self.action_ckpt,
            #     model_type="ar",  # Autoregressive
            #     few_step=True,    # Use distilled model
            #     num_inference_steps=4,
            #     model_offload=self.model_offload,
            #     use_flash_attn=self.use_flash_attn,
            #     device=self.device
            # )

            # For now, load simulation
            self._load_simulation_models()

        except Exception as e:
            print(f"Failed to load HY-WorldPlay models: {e}")
            print("Falling back to simulation mode.")
            self._load_simulation_models()

    def _load_simulation_models(self):
        """Load simulation models for testing without GPU."""
        print("Loading simulation models...")
        self.pipeline = SimulationPipeline()

    async def unload_models(self):
        """Unload models and free GPU memory."""
        if not self.models_loaded:
            return

        print("Unloading models...")

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch

            # Delete model components
            if self.pipeline:
                del self.pipeline
            if self.transformer:
                del self.transformer
            if self.vae:
                del self.vae
            if self.text_encoder:
                del self.text_encoder

            # Clear CUDA cache
            torch.cuda.empty_cache()

        self.pipeline = None
        self.models_loaded = False
        print("Models unloaded.")

    def get_pipeline(self):
        """Get the inference pipeline."""
        if not self.models_loaded:
            raise RuntimeError("Models not loaded")
        return self.pipeline

    async def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information and memory usage."""
        if not CUDA_AVAILABLE:
            return {
                "available": False,
                "device": "cpu"
            }

        import torch

        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        memory_total = props.total_memory / 1e9

        return {
            "available": True,
            "device": torch.cuda.get_device_name(device),
            "compute_capability": f"{props.major}.{props.minor}",
            "memory": {
                "allocated_gb": round(memory_allocated, 2),
                "reserved_gb": round(memory_reserved, 2),
                "total_gb": round(memory_total, 2),
                "free_gb": round(memory_total - memory_allocated, 2)
            },
            "utilization": self._get_gpu_utilization()
        }

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0


class SimulationPipeline:
    """
    Simulation pipeline for testing without actual HY-WorldPlay.

    Generates placeholder frames for development and testing.
    """

    def __init__(self):
        self.frame_count = 0

    def __call__(
        self,
        prompt: str,
        context_frames=None,
        pose_json=None,
        actions=None,
        num_frames: int = 16,
        **kwargs
    ):
        """
        Generate simulated frames.

        Returns list of numpy arrays representing video frames.
        """
        import numpy as np

        frames = []
        for i in range(num_frames):
            # Create gradient frame
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
            t = (self.frame_count + i) / 24.0  # Time in seconds

            # Animated gradient based on prompt hash
            prompt_hash = hash(prompt) % 360
            r = int(128 + 127 * np.sin(t + prompt_hash * 0.01))
            g = int(128 + 127 * np.sin(t + 2 + prompt_hash * 0.01))
            b = int(128 + 127 * np.sin(t + 4 + prompt_hash * 0.01))

            frame[:, :] = [r, g, b]

            # Add text overlay (simulation indicator)
            # In production, this would be actual generated content

            frames.append(frame)

        self.frame_count += num_frames
        return frames

    def generate_streaming(self, **kwargs):
        """Generator version for streaming."""
        frames = self(**kwargs)
        for frame in frames:
            yield frame
