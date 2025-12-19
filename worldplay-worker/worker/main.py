"""
WorldPlay Worker - GPU-based inference service for HY-WorldPlay.

This service runs on the GPU server and handles:
- Model loading and management
- Video generation from HY-WorldPlay
- Action processing for interactive generation
- Frame encoding and streaming

Run with: python -m worker.main
"""

import asyncio
import json
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from worker.inference_engine import WorldPlayInferenceEngine
from worker.model_manager import ModelManager
from worker.frame_encoder import FrameEncoder


# Configuration from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MODEL_PATH = os.getenv("HY_WORLDPLAY_MODEL_PATH", "../HY-WorldPlay/ckpts/hunyuan-video-1.5")
ACTION_CKPT = os.getenv("HY_WORLDPLAY_ACTION_CKPT", "../HY-WorldPlay/ckpts/worldplay/HY-World1.5-Autoregressive-480P-I2V")
HOST = os.getenv("WORKER_HOST", "0.0.0.0")
PORT = int(os.getenv("WORKER_PORT", "50051"))


# FastAPI app for HTTP/WebSocket endpoints
app = FastAPI(
    title="WorldPlay Worker",
    description="GPU-based inference service for HY-WorldPlay",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_manager: Optional[ModelManager] = None
inference_engine: Optional[WorldPlayInferenceEngine] = None
frame_encoder: Optional[FrameEncoder] = None
redis_client: Optional[redis.Redis] = None
executor: Optional[ThreadPoolExecutor] = None

# Active generation sessions
active_sessions: Dict[str, dict] = {}


@app.on_event("startup")
async def startup():
    """Initialize models and connections on startup."""
    global model_manager, inference_engine, frame_encoder, redis_client, executor

    print("Starting WorldPlay Worker...")

    # Initialize thread pool for CPU-bound tasks
    executor = ThreadPoolExecutor(max_workers=4)

    # Connect to Redis
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    print(f"Connected to Redis: {REDIS_URL}")

    # Initialize model manager
    model_manager = ModelManager(
        model_path=MODEL_PATH,
        action_ckpt=ACTION_CKPT
    )

    # Load models (this takes time)
    print("Loading HY-WorldPlay models...")
    await model_manager.load_models()
    print("Models loaded successfully!")

    # Initialize inference engine
    inference_engine = WorldPlayInferenceEngine(model_manager)

    # Initialize frame encoder
    frame_encoder = FrameEncoder()

    print(f"WorldPlay Worker ready on {HOST}:{PORT}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global model_manager, redis_client, executor

    print("Shutting down WorldPlay Worker...")

    # Cancel active sessions
    for session_id in list(active_sessions.keys()):
        await end_session(session_id)

    # Unload models
    if model_manager:
        await model_manager.unload_models()

    # Close Redis
    if redis_client:
        await redis_client.close()

    # Shutdown executor
    if executor:
        executor.shutdown(wait=True)

    print("Shutdown complete.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_manager.models_loaded if model_manager else False,
        "gpu_available": model_manager.gpu_available if model_manager else False,
        "active_sessions": len(active_sessions)
    }


@app.get("/status")
async def get_status():
    """Get detailed worker status."""
    gpu_info = await model_manager.get_gpu_info() if model_manager else {}

    return {
        "models_loaded": model_manager.models_loaded if model_manager else False,
        "gpu": gpu_info,
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions.keys())
    }


@app.post("/generate")
async def start_generation(request: dict):
    """
    Start a new generation session.

    Expected request:
    {
        "session_id": "uuid",
        "prompt": "description",
        "camera_trajectory": {...},
        "reference_image": "base64_optional"
    }
    """
    session_id = request.get("session_id")
    prompt = request.get("prompt")
    camera_trajectory = request.get("camera_trajectory", {})

    if not session_id or not prompt:
        return {"error": "session_id and prompt are required"}

    if session_id in active_sessions:
        return {"error": "Session already exists"}

    # Create session
    active_sessions[session_id] = {
        "prompt": prompt,
        "camera_trajectory": camera_trajectory,
        "status": "initializing",
        "frame_count": 0,
        "actions": asyncio.Queue()
    }

    # Start generation in background
    asyncio.create_task(run_generation(session_id))

    return {
        "status": "started",
        "session_id": session_id,
        "websocket_url": f"/ws/{session_id}"
    }


async def run_generation(session_id: str):
    """Run the generation loop for a session."""
    if session_id not in active_sessions:
        return

    session = active_sessions[session_id]
    session["status"] = "generating"

    try:
        # Generate frames
        async for frame in inference_engine.generate(
            prompt=session["prompt"],
            camera_trajectory=session["camera_trajectory"],
            action_queue=session["actions"]
        ):
            if session_id not in active_sessions:
                break

            # Encode frame
            encoded_frame = await frame_encoder.encode(frame)

            # Publish to Redis for gateway to consume
            await redis_client.publish(
                f"world:frames:{session_id}",
                encoded_frame
            )

            session["frame_count"] += 1

    except Exception as e:
        print(f"Generation error for {session_id}: {e}")
        if session_id in active_sessions:
            active_sessions[session_id]["status"] = "error"
            active_sessions[session_id]["error"] = str(e)


@app.post("/action/{session_id}")
async def send_action(session_id: str, action: dict):
    """Send user action to influence generation."""
    if session_id not in active_sessions:
        return {"error": "Session not found"}

    await active_sessions[session_id]["actions"].put(action)
    return {"status": "accepted"}


@app.post("/transition/{session_id}")
async def transition_scene(session_id: str, request: dict):
    """Transition to a new scene."""
    if session_id not in active_sessions:
        return {"error": "Session not found"}

    new_prompt = request.get("new_prompt")
    if not new_prompt:
        return {"error": "new_prompt is required"}

    # Update session prompt (engine will pick up the change)
    active_sessions[session_id]["prompt"] = new_prompt
    active_sessions[session_id]["transitioning"] = True

    return {"status": "transitioning"}


@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a generation session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"status": "ended"}
    return {"error": "Session not found"}


@app.websocket("/ws/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for bidirectional streaming.

    Sends: Video frames (binary)
    Receives: User actions (JSON)
    """
    if session_id not in active_sessions:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    # Subscribe to frame channel
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"world:frames:{session_id}")

    try:
        # Task to send frames
        async def send_frames():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await websocket.send_bytes(message["data"])

        # Task to receive actions
        async def receive_actions():
            while True:
                data = await websocket.receive()
                if "text" in data:
                    action = json.loads(data["text"])
                    if session_id in active_sessions:
                        await active_sessions[session_id]["actions"].put(action)

        # Run both tasks concurrently
        await asyncio.gather(
            send_frames(),
            receive_actions()
        )

    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"world:frames:{session_id}")


def main():
    """Entry point for the worker."""
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run the server
    uvicorn.run(
        "worker.main:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
