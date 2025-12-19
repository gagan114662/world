"""
World Orchestrator - coordinates world generation requests and session management.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from app.models.schemas import (
    GenerateWorldRequest,
    WorldSession,
    WorldStatus,
    UserAction,
    NarrationMessage,
    PreGeneratedWorld,
    CameraConfig,
)
from app.services.cache_manager import CacheManager
from app.services.prompt_builder import PromptBuilder
from app.workers.gpu_client import GPUClient


class WorldOrchestrator:
    """
    Orchestrates world generation, session management, and GPU worker communication.
    """

    def __init__(
        self,
        gpu_client: GPUClient,
        cache_manager: CacheManager,
        redis_client
    ):
        self.gpu_client = gpu_client
        self.cache_manager = cache_manager
        self.redis = redis_client
        self.prompt_builder = PromptBuilder()

        # In-memory session cache for quick access
        self.sessions: Dict[str, WorldSession] = {}

        # Frame buffers per session (last 48 frames = 2 seconds at 24 FPS)
        self.frame_buffers: Dict[str, List[bytes]] = {}
        self.frame_buffer_size = 48

        # Narration queues per session
        self.narration_queues: Dict[str, asyncio.Queue] = {}

        # Action queues per session
        self.action_queues: Dict[str, asyncio.Queue] = {}

        # Generation tasks
        self.generation_tasks: Dict[str, asyncio.Task] = {}

    async def create_session(self, request: GenerateWorldRequest) -> WorldSession:
        """Create a new world generation session."""
        session_id = request.session_id or str(uuid.uuid4())

        # Build optimized prompt
        optimized_prompt = self.prompt_builder.build_prompt(
            subject=request.subject,
            topic=request.topic,
            base_prompt=request.prompt,
            learning_objective=request.learning_objective
        )

        session = WorldSession(
            session_id=session_id,
            tutoring_session_id=request.tutoring_session_id,
            subject=request.subject,
            topic=request.topic,
            prompt=optimized_prompt,
            learning_objective=request.learning_objective,
            camera=request.camera,
            status=WorldStatus.PENDING,
            created_at=datetime.utcnow(),
            last_interaction=datetime.utcnow()
        )

        # Store in memory and Redis
        self.sessions[session_id] = session
        await self._save_session_to_redis(session)

        # Initialize buffers and queues
        self.frame_buffers[session_id] = []
        self.narration_queues[session_id] = asyncio.Queue()
        self.action_queues[session_id] = asyncio.Queue()

        return session

    async def start_generation(self, session_id: str):
        """Start world generation for a session."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Update status
        session.status = WorldStatus.GENERATING
        await self._save_session_to_redis(session)

        # Check for pre-generated content first
        pregenerated = await self.cache_manager.get_pregenerated(
            subject=session.subject,
            topic=session.topic,
            prompt=session.prompt
        )

        if pregenerated:
            # Use pre-generated frames for instant start
            self.frame_buffers[session_id] = pregenerated.frames[:self.frame_buffer_size]
            session.status = WorldStatus.ACTIVE
            await self._save_session_to_redis(session)

        # Start generation task
        task = asyncio.create_task(
            self._generation_loop(session_id)
        )
        self.generation_tasks[session_id] = task

    async def _generation_loop(self, session_id: str):
        """Main generation loop - gets frames from GPU worker."""
        session = self.sessions.get(session_id)
        if not session:
            return

        try:
            session.status = WorldStatus.ACTIVE
            await self._save_session_to_redis(session)

            # Build camera trajectory for initial generation
            camera_trajectory = self._build_camera_trajectory(session.camera)

            # Start streaming from GPU worker
            async for frame_data in self.gpu_client.generate_stream(
                prompt=session.prompt,
                camera_trajectory=camera_trajectory,
                session_id=session_id
            ):
                if session_id not in self.sessions:
                    break  # Session was deleted

                current_session = self.sessions[session_id]
                if current_session.status == WorldStatus.PAUSED:
                    # Wait while paused
                    while current_session.status == WorldStatus.PAUSED:
                        await asyncio.sleep(0.1)
                        if session_id not in self.sessions:
                            return

                # Check for user actions and update generation
                if session_id in self.action_queues:
                    while not self.action_queues[session_id].empty():
                        action = await self.action_queues[session_id].get()
                        await self.gpu_client.send_action(session_id, action)

                # Add frame to buffer
                if session_id in self.frame_buffers:
                    self.frame_buffers[session_id].append(frame_data)
                    # Keep buffer at max size
                    if len(self.frame_buffers[session_id]) > self.frame_buffer_size:
                        self.frame_buffers[session_id].pop(0)

                # Update session stats
                current_session.frame_count += 1
                current_session.duration_seconds = current_session.frame_count / 24.0
                current_session.last_interaction = datetime.utcnow()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if session_id in self.sessions:
                self.sessions[session_id].status = WorldStatus.ERROR
                self.sessions[session_id].error_message = str(e)
                await self._save_session_to_redis(self.sessions[session_id])

    def _build_camera_trajectory(self, camera: CameraConfig) -> dict:
        """Build camera trajectory JSON for HY-WorldPlay."""
        return {
            "initial_position": [camera.position.x, camera.position.y, camera.position.z],
            "initial_rotation": [camera.rotation.pitch, camera.rotation.yaw, camera.rotation.roll],
            "mode": camera.mode.value,
            "fov": camera.fov,
            # HY-WorldPlay expects specific format
            "frames": []  # Will be populated dynamically
        }

    async def get_session(self, session_id: str) -> Optional[WorldSession]:
        """Get session by ID."""
        if session_id in self.sessions:
            return self.sessions[session_id]

        # Try Redis
        session_data = await self.redis.hgetall(f"world:session:{session_id}")
        if session_data:
            session = WorldSession(**json.loads(session_data.get("data", "{}")))
            self.sessions[session_id] = session
            return session

        return None

    async def get_next_frame(self, session_id: str) -> Optional[bytes]:
        """Get the next frame for streaming."""
        if session_id not in self.frame_buffers:
            return None

        buffer = self.frame_buffers[session_id]
        if buffer:
            return buffer[-1]  # Return most recent frame
        return None

    async def get_next_narration(self, session_id: str) -> Optional[NarrationMessage]:
        """Get the next narration message if available."""
        if session_id not in self.narration_queues:
            return None

        try:
            return self.narration_queues[session_id].get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def add_narration(self, session_id: str, narration: NarrationMessage):
        """Add a narration message to the queue."""
        if session_id in self.narration_queues:
            await self.narration_queues[session_id].put(narration)

    async def process_action(self, session_id: str, action: UserAction):
        """Process user action and update generation."""
        if session_id in self.action_queues:
            await self.action_queues[session_id].put(action)

        # Update last interaction time
        if session_id in self.sessions:
            self.sessions[session_id].last_interaction = datetime.utcnow()

    async def pause_session(self, session_id: str) -> bool:
        """Pause a session."""
        if session_id not in self.sessions:
            return False

        self.sessions[session_id].status = WorldStatus.PAUSED
        await self._save_session_to_redis(self.sessions[session_id])
        return True

    async def resume_session(self, session_id: str) -> bool:
        """Resume a paused session."""
        if session_id not in self.sessions:
            return False

        self.sessions[session_id].status = WorldStatus.ACTIVE
        await self._save_session_to_redis(self.sessions[session_id])
        return True

    async def reset_session(self, session_id: str) -> bool:
        """Reset a session to initial state."""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        session.frame_count = 0
        session.duration_seconds = 0.0
        session.status = WorldStatus.GENERATING

        # Clear frame buffer
        if session_id in self.frame_buffers:
            self.frame_buffers[session_id].clear()

        # Restart generation
        if session_id in self.generation_tasks:
            self.generation_tasks[session_id].cancel()

        await self.start_generation(session_id)
        return True

    async def transition_scene(
        self,
        session_id: str,
        new_prompt: str,
        transition_type: str = "fade",
        duration_ms: int = 1000
    ):
        """Transition to a new scene within the same session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        # Update prompt
        optimized_prompt = self.prompt_builder.build_prompt(
            subject=session.subject,
            topic=session.topic,
            base_prompt=new_prompt,
            learning_objective=session.learning_objective
        )
        session.prompt = optimized_prompt

        # Signal GPU worker to transition
        await self.gpu_client.transition(
            session_id=session_id,
            new_prompt=optimized_prompt,
            transition_type=transition_type,
            duration_ms=duration_ms
        )

    async def end_session(self, session_id: str) -> bool:
        """End a session and clean up resources."""
        if session_id not in self.sessions:
            return False

        # Cancel generation task
        if session_id in self.generation_tasks:
            self.generation_tasks[session_id].cancel()
            del self.generation_tasks[session_id]

        # Clean up buffers and queues
        if session_id in self.frame_buffers:
            del self.frame_buffers[session_id]
        if session_id in self.narration_queues:
            del self.narration_queues[session_id]
        if session_id in self.action_queues:
            del self.action_queues[session_id]

        # Update status
        self.sessions[session_id].status = WorldStatus.COMPLETED
        await self._save_session_to_redis(self.sessions[session_id])

        # Remove from memory (keep in Redis for history)
        del self.sessions[session_id]

        # Notify GPU worker
        await self.gpu_client.end_session(session_id)

        return True

    async def get_latency(self, session_id: str) -> Optional[float]:
        """Get current latency for a session."""
        return await self.gpu_client.get_latency(session_id)

    async def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization."""
        return await self.gpu_client.get_utilization()

    async def get_pregenerated_worlds(
        self,
        subject: str,
        topic: str
    ) -> List[PreGeneratedWorld]:
        """Get list of pre-generated worlds for a topic."""
        return await self.cache_manager.list_pregenerated(subject, topic)

    async def _save_session_to_redis(self, session: WorldSession):
        """Save session state to Redis."""
        await self.redis.hset(
            f"world:session:{session.session_id}",
            mapping={
                "data": session.model_dump_json(),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        # Set TTL of 30 minutes
        await self.redis.expire(f"world:session:{session.session_id}", 1800)
