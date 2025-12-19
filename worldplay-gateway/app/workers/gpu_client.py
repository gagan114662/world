"""
GPU Client - communicates with the WorldPlay Worker running on GPU server.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict, Any
from dataclasses import dataclass

# Note: In production, you would use grpcio for gRPC communication
# For now, we'll use a WebSocket-based approach for simplicity


@dataclass
class GPUWorkerStatus:
    """Status of the GPU worker."""
    connected: bool
    gpu_utilization: float
    memory_used_gb: float
    memory_total_gb: float
    active_sessions: int
    queue_length: int


class GPUClient:
    """
    Client for communicating with the WorldPlay GPU Worker.

    Handles:
    - Connection management
    - Generation requests
    - Action streaming
    - Health monitoring
    """

    def __init__(self, host: str = "localhost", port: int = 50051):
        self.host = host
        self.port = port
        self.connected = False

        # WebSocket connection (placeholder - would be gRPC in production)
        self.ws = None

        # Session tracking
        self.active_sessions: Dict[str, dict] = {}

        # Latency tracking per session
        self.latencies: Dict[str, float] = {}

        # Current GPU status
        self.status = GPUWorkerStatus(
            connected=False,
            gpu_utilization=0.0,
            memory_used_gb=0.0,
            memory_total_gb=14.0,  # Default assumption
            active_sessions=0,
            queue_length=0
        )

    async def connect(self):
        """Establish connection to GPU worker."""
        try:
            # In production, this would be:
            # self.channel = grpc.aio.insecure_channel(f'{self.host}:{self.port}')
            # self.stub = worldplay_pb2_grpc.WorldPlayWorkerStub(self.channel)

            # For now, simulate connection
            self.connected = True
            self.status.connected = True
            print(f"GPU Client connected to {self.host}:{self.port}")

        except Exception as e:
            self.connected = False
            self.status.connected = False
            print(f"Failed to connect to GPU worker: {e}")

    async def disconnect(self):
        """Close connection to GPU worker."""
        self.connected = False
        self.status.connected = False

        # Clean up active sessions
        for session_id in list(self.active_sessions.keys()):
            await self.end_session(session_id)

    async def health_check(self) -> str:
        """Check GPU worker health."""
        if not self.connected:
            return "disconnected"

        try:
            # In production, call health RPC
            # response = await self.stub.HealthCheck(Empty())
            # return response.status

            return "healthy"
        except Exception:
            return "unhealthy"

    async def generate_stream(
        self,
        prompt: str,
        camera_trajectory: dict,
        session_id: str,
        reference_image: Optional[bytes] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream generated video frames from GPU worker.

        Yields H.264 encoded frame data.
        """
        if not self.connected:
            raise ConnectionError("GPU worker not connected")

        # Register session
        self.active_sessions[session_id] = {
            "prompt": prompt,
            "start_time": time.time(),
            "frame_count": 0
        }
        self.status.active_sessions = len(self.active_sessions)

        try:
            # In production, this would be:
            # request = GenerateRequest(
            #     session_id=session_id,
            #     prompt=prompt,
            #     camera=camera_trajectory,
            #     reference_image=reference_image
            # )
            # async for chunk in self.stub.GenerateWorld(request):
            #     for frame in chunk.frames:
            #         yield frame

            # Simulation: Generate placeholder frames at 24 FPS
            # In production, this streams real frames from HY-WorldPlay
            frame_interval = 1.0 / 24

            while session_id in self.active_sessions:
                start_time = time.time()

                # Generate a placeholder frame (in production, real frame data)
                frame_data = self._create_placeholder_frame(
                    session_id,
                    self.active_sessions[session_id]["frame_count"]
                )

                self.active_sessions[session_id]["frame_count"] += 1

                # Track latency
                generation_time = time.time() - start_time
                self.latencies[session_id] = generation_time * 1000  # ms

                yield frame_data

                # Maintain frame rate
                elapsed = time.time() - start_time
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)

        except asyncio.CancelledError:
            pass
        finally:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.status.active_sessions = len(self.active_sessions)

    def _create_placeholder_frame(self, session_id: str, frame_num: int) -> bytes:
        """
        Create a placeholder frame for testing.

        In production, this would not exist - real frames come from HY-WorldPlay.
        """
        # Return minimal valid data structure
        # In production, this would be actual H.264 encoded frame data
        placeholder = {
            "type": "frame",
            "session_id": session_id,
            "frame_num": frame_num,
            "timestamp": int(time.time() * 1000),
            "placeholder": True
        }
        return json.dumps(placeholder).encode()

    async def send_action(self, session_id: str, action) -> bool:
        """
        Send user action to GPU worker to influence generation.

        Returns True if action was accepted.
        """
        if not self.connected or session_id not in self.active_sessions:
            return False

        try:
            # In production:
            # request = ActionRequest(
            #     session_id=session_id,
            #     keyboard=action.keyboard.dict(),
            #     mouse=action.mouse.dict(),
            #     client_timestamp=action.timestamp
            # )
            # response = await self.stub.UpdateAction(request)
            # return response.accepted

            # Simulation: Always accept
            return True
        except Exception:
            return False

    async def transition(
        self,
        session_id: str,
        new_prompt: str,
        transition_type: str = "fade",
        duration_ms: int = 1000
    ) -> bool:
        """
        Request scene transition on GPU worker.

        Returns True if transition was initiated.
        """
        if not self.connected or session_id not in self.active_sessions:
            return False

        try:
            # In production:
            # request = TransitionRequest(
            #     session_id=session_id,
            #     new_prompt=new_prompt,
            #     transition_type=transition_type,
            #     duration_ms=duration_ms
            # )
            # response = await self.stub.Transition(request)
            # return response.success

            # Update local session info
            self.active_sessions[session_id]["prompt"] = new_prompt
            return True
        except Exception:
            return False

    async def end_session(self, session_id: str) -> bool:
        """End a generation session on GPU worker."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            self.status.active_sessions = len(self.active_sessions)

        if session_id in self.latencies:
            del self.latencies[session_id]

        # In production:
        # request = EndSessionRequest(session_id=session_id)
        # await self.stub.EndSession(request)

        return True

    async def get_latency(self, session_id: str) -> Optional[float]:
        """Get current generation latency for a session in milliseconds."""
        return self.latencies.get(session_id)

    async def get_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not self.connected:
            return None

        # In production, query GPU worker
        # response = await self.stub.GetStatus(Empty())
        # return response.gpu_utilization

        # Simulation based on active sessions
        return min(100.0, len(self.active_sessions) * 15.0)

    async def get_status(self) -> GPUWorkerStatus:
        """Get full GPU worker status."""
        if self.connected:
            # Update status from worker
            # In production:
            # response = await self.stub.GetStatus(Empty())
            # self.status.gpu_utilization = response.gpu_utilization
            # self.status.memory_used_gb = response.memory_used_gb

            self.status.gpu_utilization = await self.get_utilization() or 0.0
            self.status.active_sessions = len(self.active_sessions)

        return self.status

    async def queue_generation(
        self,
        prompt: str,
        camera_trajectory: dict,
        priority: str = "normal"
    ) -> str:
        """
        Queue a generation request (for pre-generation).

        Returns a job ID for tracking.
        """
        # In production:
        # request = QueueRequest(
        #     prompt=prompt,
        #     camera=camera_trajectory,
        #     priority=priority
        # )
        # response = await self.stub.QueueGeneration(request)
        # return response.job_id

        import uuid
        return str(uuid.uuid4())
