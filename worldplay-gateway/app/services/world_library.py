"""
World Library Service

Manages pre-generated worlds and on-demand generation requests.
"""

import os
import asyncio
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum

import redis.asyncio as redis


# Configuration
WORLD_LIBRARY_PATH = os.getenv("WORLD_LIBRARY_PATH", "/workspace/world_library")
WORLD_CDN_URL = os.getenv("WORLD_CDN_URL", "http://localhost:8010/worlds")


class GenerationStatus(str, Enum):
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETE = "complete"
    FAILED = "failed"


class WorldLibraryService:
    """
    Manages the pre-generated world library and on-demand generation.

    Pre-generated worlds are stored in:
        {WORLD_LIBRARY_PATH}/{world_id}/
            - metadata.json
            - start.mp4
            - from_start_forward.mp4
            - from_start_left.mp4
            - ...
    """

    def __init__(self, redis_client: redis.Redis, library_path: str = WORLD_LIBRARY_PATH):
        self.redis = redis_client
        self.library_path = Path(library_path)
        self.generation_queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start_worker(self):
        """Start background worker for on-demand generation."""
        self._worker_task = asyncio.create_task(self._generation_worker())
        print("World Library: Generation worker started")

    async def stop_worker(self):
        """Stop background worker."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _generation_worker(self):
        """Background worker that processes generation requests."""
        while True:
            try:
                job = await self.generation_queue.get()
                await self._process_generation_job(job)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"World Library: Generation error: {e}")

    async def _process_generation_job(self, job: Dict):
        """Process a single generation job."""
        job_id = job["job_id"]
        world_id = job["world_id"]
        topic = job["topic"]
        prompt = job["prompt"]

        try:
            # Update status
            await self._set_job_status(job_id, GenerationStatus.GENERATING, progress=0)

            # Call HY-WorldPlay generation script
            # This would typically SSH to a GPU worker or use a job queue
            # For now, we mark it as queued for manual processing

            # In production, this would:
            # 1. Send job to GPU cluster (RunPod, Lambda Labs, etc.)
            # 2. Monitor progress
            # 3. Download completed videos
            # 4. Update library

            await self._set_job_status(
                job_id,
                GenerationStatus.QUEUED,
                message="Job queued for GPU processing. Check back in 5-10 minutes."
            )

        except Exception as e:
            await self._set_job_status(
                job_id,
                GenerationStatus.FAILED,
                error=str(e)
            )

    async def _set_job_status(
        self,
        job_id: str,
        status: GenerationStatus,
        progress: int = 0,
        message: str = "",
        error: str = ""
    ):
        """Update job status in Redis."""
        job_data = {
            "status": status.value,
            "progress": progress,
            "message": message,
            "error": error,
            "updated_at": datetime.now().isoformat()
        }
        await self.redis.hset(f"world_gen:{job_id}", mapping=job_data)
        await self.redis.expire(f"world_gen:{job_id}", 86400)  # 24h TTL

    def list_available_worlds(self) -> List[Dict]:
        """List all pre-generated worlds in the library."""
        worlds = []

        if not self.library_path.exists():
            return worlds

        for world_dir in self.library_path.iterdir():
            if world_dir.is_dir():
                metadata_file = world_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        worlds.append({
                            "id": world_dir.name,
                            "name": metadata.get("world", {}).get("name", world_dir.name),
                            "subject": metadata.get("world", {}).get("subject", "unknown"),
                            "description": metadata.get("world", {}).get("description", ""),
                            "segments": list(metadata.get("results", {}).get("segments", {}).keys()),
                            "generated_at": metadata.get("generated_at", ""),
                        })
                    except json.JSONDecodeError:
                        pass

        return worlds

    def get_world_metadata(self, world_id: str) -> Optional[Dict]:
        """Get metadata for a specific world."""
        world_dir = self.library_path / world_id
        metadata_file = world_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def get_world_segment_url(self, world_id: str, segment_id: str) -> Optional[str]:
        """Get CDN URL for a world segment video."""
        world_dir = self.library_path / world_id
        segment_file = world_dir / f"{segment_id}.mp4"

        if segment_file.exists():
            return f"{WORLD_CDN_URL}/{world_id}/{segment_id}.mp4"

        return None

    def list_world_segments(self, world_id: str) -> List[str]:
        """List all available segments for a world."""
        world_dir = self.library_path / world_id

        if not world_dir.exists():
            return []

        segments = []
        for f in world_dir.glob("*.mp4"):
            segments.append(f.stem)

        return segments

    async def request_world_generation(
        self,
        topic: str,
        prompt: str,
        user_id: Optional[str] = None,
        reference_image_url: Optional[str] = None
    ) -> Dict:
        """
        Request generation of a new world.

        Returns:
            Dict with job_id and status
        """
        # Check if similar world already exists
        for world in self.list_available_worlds():
            if world["subject"] == topic or topic.lower() in world["name"].lower():
                return {
                    "status": "exists",
                    "world_id": world["id"],
                    "message": f"Similar world '{world['name']}' already exists"
                }

        # Create new generation job
        job_id = str(uuid.uuid4())
        world_id = topic.lower().replace(" ", "_")[:30] + "_" + job_id[:8]

        job = {
            "job_id": job_id,
            "world_id": world_id,
            "topic": topic,
            "prompt": prompt,
            "user_id": user_id,
            "reference_image_url": reference_image_url,
            "created_at": datetime.now().isoformat(),
        }

        # Store job in Redis
        await self.redis.hset(f"world_gen:{job_id}", mapping={
            "status": GenerationStatus.QUEUED.value,
            "world_id": world_id,
            "topic": topic,
            "progress": 0,
            "message": "Job queued",
            "created_at": job["created_at"]
        })
        await self.redis.expire(f"world_gen:{job_id}", 86400)

        # Queue for processing
        await self.generation_queue.put(job)

        return {
            "status": "generating",
            "job_id": job_id,
            "world_id": world_id,
            "message": "World generation queued. This may take 5-10 minutes.",
            "estimated_time_seconds": 600,
        }

    async def get_generation_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a generation job."""
        job_data = await self.redis.hgetall(f"world_gen:{job_id}")

        if not job_data:
            return None

        return {
            "job_id": job_id,
            "status": job_data.get("status", "unknown"),
            "world_id": job_data.get("world_id"),
            "topic": job_data.get("topic"),
            "progress": int(job_data.get("progress", 0)),
            "message": job_data.get("message", ""),
            "error": job_data.get("error", ""),
            "created_at": job_data.get("created_at"),
            "updated_at": job_data.get("updated_at"),
        }

    def is_world_available(self, world_id: str) -> bool:
        """Check if a world is available in the library."""
        world_dir = self.library_path / world_id
        return world_dir.exists() and (world_dir / "start.mp4").exists()


# Singleton instance (created during app startup)
_world_library: Optional[WorldLibraryService] = None


def get_world_library() -> WorldLibraryService:
    """Get the world library service instance."""
    if _world_library is None:
        raise RuntimeError("World library not initialized")
    return _world_library


def init_world_library(redis_client: redis.Redis) -> WorldLibraryService:
    """Initialize the world library service."""
    global _world_library
    _world_library = WorldLibraryService(redis_client)
    return _world_library
