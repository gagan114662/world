"""
World Generation Router - handles world creation and management endpoints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends

from app.models.schemas import (
    GenerateWorldRequest,
    GenerateWorldResponse,
    WorldSession,
    WorldSessionStatus,
    WorldStatus,
    TransitionRequest,
)


router = APIRouter()


def get_orchestrator(request: Request):
    """Dependency to get the world orchestrator from app state."""
    return request.app.state.orchestrator


@router.post("/generate", response_model=GenerateWorldResponse)
async def generate_world(
    request: GenerateWorldRequest,
    orchestrator=Depends(get_orchestrator)
):
    """
    Initiate world generation based on learning context.

    This endpoint starts the world generation process and returns
    a WebSocket URL for streaming the video frames.
    """
    try:
        session = await orchestrator.create_session(request)

        # Start generation in background
        await orchestrator.start_generation(session.session_id)

        return GenerateWorldResponse(
            session_id=session.session_id,
            status=session.status,
            websocket_url=f"/ws/worlds/{session.session_id}/stream",
            estimated_start_time_ms=500,
            message="World generation initiated. Connect to WebSocket for video stream."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate world: {str(e)}")


@router.get("/{session_id}", response_model=WorldSession)
async def get_world_session(
    session_id: str,
    orchestrator=Depends(get_orchestrator)
):
    """Get the current state of a world session."""
    session = await orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


@router.get("/{session_id}/status", response_model=WorldSessionStatus)
async def get_world_status(
    session_id: str,
    orchestrator=Depends(get_orchestrator)
):
    """Get the current status of a world session."""
    session = await orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return WorldSessionStatus(
        session_id=session.session_id,
        status=session.status,
        frame_count=session.frame_count,
        duration_seconds=session.duration_seconds,
        latency_ms=await orchestrator.get_latency(session_id),
        gpu_utilization=await orchestrator.get_gpu_utilization()
    )


@router.post("/{session_id}/pause")
async def pause_world(
    session_id: str,
    orchestrator=Depends(get_orchestrator)
):
    """Pause world generation for a session."""
    success = await orchestrator.pause_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "paused", "session_id": session_id}


@router.post("/{session_id}/resume")
async def resume_world(
    session_id: str,
    orchestrator=Depends(get_orchestrator)
):
    """Resume world generation for a paused session."""
    success = await orchestrator.resume_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "active", "session_id": session_id}


@router.post("/{session_id}/transition")
async def transition_scene(
    session_id: str,
    request: TransitionRequest,
    orchestrator=Depends(get_orchestrator)
):
    """
    Transition to a new scene within the same session.

    This allows smooth scene changes without creating a new session,
    useful for exploring related concepts.
    """
    session = await orchestrator.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    try:
        await orchestrator.transition_scene(
            session_id=session_id,
            new_prompt=request.new_prompt,
            transition_type=request.transition_type,
            duration_ms=request.duration_ms
        )
        return {
            "status": "transitioning",
            "session_id": session_id,
            "new_prompt": request.new_prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transition failed: {str(e)}")


@router.delete("/{session_id}")
async def end_world_session(
    session_id: str,
    orchestrator=Depends(get_orchestrator)
):
    """End a world session and clean up resources."""
    success = await orchestrator.end_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return {"status": "completed", "session_id": session_id}


@router.get("/pregenerated/{subject}/{topic}")
async def get_pregenerated_worlds(
    subject: str,
    topic: str,
    orchestrator=Depends(get_orchestrator)
):
    """
    Get list of pre-generated worlds for a topic.

    Pre-generated worlds allow instant start for popular topics.
    """
    worlds = await orchestrator.get_pregenerated_worlds(subject, topic)
    return {
        "subject": subject,
        "topic": topic,
        "pregenerated_worlds": worlds
    }
