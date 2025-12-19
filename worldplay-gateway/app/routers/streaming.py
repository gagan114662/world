"""
Streaming Router - handles WebSocket connections for real-time video streaming.
"""

import asyncio
import json
import time
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request

from app.models.schemas import UserAction, KeyboardState, MouseState, StreamMessage


router = APIRouter()


# Active WebSocket connections per session
active_connections: Dict[str, Set[WebSocket]] = {}


class ConnectionManager:
    """Manages WebSocket connections for world streaming."""

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        self.active_connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            self.active_connections[session_id].discard(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]

    async def broadcast_frame(self, session_id: str, frame_data: bytes):
        """Broadcast a video frame to all connections for a session."""
        if session_id in self.active_connections:
            dead_connections = set()
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_bytes(frame_data)
                except Exception:
                    dead_connections.add(websocket)

            # Clean up dead connections
            for ws in dead_connections:
                self.active_connections[session_id].discard(ws)

    async def broadcast_message(self, session_id: str, message: dict):
        """Broadcast a JSON message to all connections for a session."""
        if session_id in self.active_connections:
            dead_connections = set()
            for websocket in self.active_connections[session_id]:
                try:
                    await websocket.send_json(message)
                except Exception:
                    dead_connections.add(websocket)

            for ws in dead_connections:
                self.active_connections[session_id].discard(ws)

    def get_connection_count(self, session_id: str) -> int:
        """Get number of active connections for a session."""
        return len(self.active_connections.get(session_id, set()))


manager = ConnectionManager()


def get_orchestrator(request: Request):
    """Get orchestrator from app state."""
    return request.app.state.orchestrator


@router.websocket("/{session_id}/stream")
async def stream_world(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for bidirectional world streaming.

    Sends: Video frames (binary), metadata (JSON), narration (JSON)
    Receives: User actions (JSON) - keyboard/mouse inputs
    """
    # Get orchestrator from app state
    app = websocket.app
    orchestrator = app.state.orchestrator

    # Verify session exists
    session = await orchestrator.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await manager.connect(session_id, websocket)

    # Send initial connection metadata
    await websocket.send_json({
        "type": "connected",
        "payload": {
            "session_id": session_id,
            "fps": 24,
            "resolution": "480p",
            "subject": session.subject,
            "topic": session.topic
        },
        "timestamp": int(time.time() * 1000)
    })

    try:
        # Start frame streaming task
        frame_task = asyncio.create_task(
            stream_frames(websocket, session_id, orchestrator)
        )

        # Handle incoming user actions
        while True:
            try:
                data = await websocket.receive()

                if "text" in data:
                    # JSON message (user action or control)
                    message = json.loads(data["text"])
                    await handle_client_message(message, session_id, orchestrator)

                elif "bytes" in data:
                    # Binary data (shouldn't happen from client, but handle it)
                    pass

            except WebSocketDisconnect:
                break

    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")

    finally:
        frame_task.cancel()
        manager.disconnect(session_id, websocket)

        # Check if this was the last connection
        if manager.get_connection_count(session_id) == 0:
            # Optionally pause generation when no viewers
            await orchestrator.pause_session(session_id)


async def stream_frames(websocket: WebSocket, session_id: str, orchestrator):
    """Stream video frames to the WebSocket client."""
    frame_interval = 1.0 / 24  # 24 FPS
    last_frame_time = time.time()

    while True:
        try:
            # Get next frame from orchestrator
            frame_data = await orchestrator.get_next_frame(session_id)

            if frame_data:
                # Calculate timing
                current_time = time.time()
                elapsed = current_time - last_frame_time

                # Wait if we're ahead of schedule
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)

                # Send frame as binary
                await websocket.send_bytes(frame_data)
                last_frame_time = time.time()

                # Periodically send metadata
                session = await orchestrator.get_session(session_id)
                if session and session.frame_count % 24 == 0:  # Every second
                    await websocket.send_json({
                        "type": "metadata",
                        "payload": {
                            "frame_count": session.frame_count,
                            "duration_seconds": session.duration_seconds,
                            "latency_ms": await orchestrator.get_latency(session_id)
                        },
                        "timestamp": int(time.time() * 1000)
                    })
            else:
                # No frame available, wait a bit
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Frame streaming error: {e}")
            await asyncio.sleep(0.1)


async def handle_client_message(message: dict, session_id: str, orchestrator):
    """Handle incoming client messages (actions, controls)."""
    msg_type = message.get("type", "")
    payload = message.get("payload", {})

    if msg_type == "action":
        # User input action
        action = UserAction(
            keyboard=KeyboardState(**payload.get("keyboard", {})),
            mouse=MouseState(**payload.get("mouse", {})),
            timestamp=payload.get("timestamp", 0)
        )
        await orchestrator.process_action(session_id, action)

    elif msg_type == "control":
        # Control command
        command = payload.get("command", "")
        if command == "pause":
            await orchestrator.pause_session(session_id)
        elif command == "resume":
            await orchestrator.resume_session(session_id)
        elif command == "reset":
            await orchestrator.reset_session(session_id)

    elif msg_type == "ping":
        # Keep-alive ping - respond with pong
        # This is handled by the WebSocket connection itself
        pass


@router.websocket("/{session_id}/narration")
async def stream_narration(websocket: WebSocket, session_id: str):
    """
    Separate WebSocket for AI tutor narration stream.

    This allows the narration to be synced with the video stream
    while keeping the channels separate for flexibility.
    """
    app = websocket.app
    orchestrator = app.state.orchestrator

    session = await orchestrator.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()

    try:
        while True:
            # Get next narration from orchestrator
            narration = await orchestrator.get_next_narration(session_id)

            if narration:
                await websocket.send_json({
                    "type": "narration",
                    "payload": {
                        "text": narration.text,
                        "audio_url": narration.audio_url,
                        "highlights": narration.highlights
                    },
                    "timestamp": int(time.time() * 1000)
                })
            else:
                await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Narration stream error: {e}")
