"""
Pydantic models for WorldPlay Gateway API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class WorldStatus(str, Enum):
    """Status of a world generation session."""
    PENDING = "pending"
    GENERATING = "generating"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


class CameraMode(str, Enum):
    """Camera perspective modes."""
    FIRST_PERSON = "first_person"
    THIRD_PERSON = "third_person"
    ORBIT = "orbit"


class CameraPosition(BaseModel):
    """Camera position in 3D space."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class CameraRotation(BaseModel):
    """Camera rotation (Euler angles in degrees)."""
    pitch: float = 0.0  # X-axis rotation
    yaw: float = 0.0    # Y-axis rotation
    roll: float = 0.0   # Z-axis rotation


class CameraConfig(BaseModel):
    """Camera configuration for world generation."""
    position: CameraPosition = Field(default_factory=CameraPosition)
    rotation: CameraRotation = Field(default_factory=CameraRotation)
    mode: CameraMode = CameraMode.FIRST_PERSON
    fov: float = 60.0  # Field of view in degrees


class KeyboardState(BaseModel):
    """Keyboard input state."""
    w: bool = False  # Forward
    a: bool = False  # Left
    s: bool = False  # Backward
    d: bool = False  # Right
    space: bool = False  # Jump/Up
    shift: bool = False  # Sprint/Down


class MouseState(BaseModel):
    """Mouse input state."""
    dx: float = 0.0  # Delta X (horizontal movement)
    dy: float = 0.0  # Delta Y (vertical movement)
    scroll: float = 0.0  # Scroll wheel delta
    left_button: bool = False
    right_button: bool = False


class UserAction(BaseModel):
    """User action input combining keyboard and mouse."""
    keyboard: KeyboardState = Field(default_factory=KeyboardState)
    mouse: MouseState = Field(default_factory=MouseState)
    timestamp: int = 0  # Client timestamp in milliseconds


class GenerateWorldRequest(BaseModel):
    """Request to generate a new world."""
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(..., description="Subject area (e.g., physics, biology)")
    topic: str = Field(..., description="Specific topic within subject")
    prompt: str = Field(..., description="Text description of the world to generate")
    learning_objective: Optional[str] = Field(None, description="What the student should learn")
    reference_image_url: Optional[str] = Field(None, description="Optional reference image URL")
    camera: CameraConfig = Field(default_factory=CameraConfig)
    tutoring_session_id: Optional[str] = Field(None, description="Link to tutoring session")


class WorldSession(BaseModel):
    """World generation session state."""
    session_id: str
    user_id: Optional[str] = None
    tutoring_session_id: Optional[str] = None
    subject: str
    topic: str
    prompt: str
    learning_objective: Optional[str] = None
    status: WorldStatus = WorldStatus.PENDING
    camera: CameraConfig = Field(default_factory=CameraConfig)
    frame_count: int = 0
    duration_seconds: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    world_state: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


class GenerateWorldResponse(BaseModel):
    """Response after initiating world generation."""
    session_id: str
    status: WorldStatus
    websocket_url: str
    estimated_start_time_ms: int = 0
    message: str = "World generation initiated"


class WorldSessionStatus(BaseModel):
    """Current status of a world session."""
    session_id: str
    status: WorldStatus
    frame_count: int
    duration_seconds: float
    latency_ms: Optional[float] = None
    gpu_utilization: Optional[float] = None


class TransitionRequest(BaseModel):
    """Request to transition to a new scene within the same session."""
    new_prompt: str = Field(..., description="Description of the new scene")
    transition_type: str = Field("fade", description="Type of transition: fade, cut, morph")
    duration_ms: int = Field(1000, description="Transition duration in milliseconds")


class NarrationMessage(BaseModel):
    """AI tutor narration message."""
    text: str
    audio_url: Optional[str] = None
    highlights: List[str] = Field(default_factory=list)
    timestamp: int = 0


class StreamMessage(BaseModel):
    """WebSocket message for streaming."""
    type: str  # frame, metadata, narration, error, control
    payload: Dict[str, Any]
    timestamp: int = 0


class PreGeneratedWorld(BaseModel):
    """Pre-generated world metadata."""
    cache_key: str
    subject: str
    topic: str
    prompt: str
    duration_seconds: float
    frame_count: int
    thumbnail_url: Optional[str] = None
    created_at: datetime
