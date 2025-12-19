"""
WorldPlay Gateway Service
Orchestrates world generation requests between frontend/tutor and GPU worker.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.routers import generation, streaming
from app.services.world_orchestrator import WorldOrchestrator
from app.services.cache_manager import CacheManager
from app.services.world_library import WorldLibraryService, init_world_library
from app.workers.gpu_client import GPUClient


# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
GPU_HOST = os.getenv("WORLDPLAY_GPU_HOST", "localhost")
GPU_PORT = int(os.getenv("WORLDPLAY_GPU_PORT", "50051"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    # Startup
    app.state.redis = redis.from_url(REDIS_URL, decode_responses=True)
    app.state.gpu_client = GPUClient(host=GPU_HOST, port=GPU_PORT)
    app.state.cache_manager = CacheManager(app.state.redis)
    app.state.orchestrator = WorldOrchestrator(
        gpu_client=app.state.gpu_client,
        cache_manager=app.state.cache_manager,
        redis_client=app.state.redis
    )

    # Initialize world library service
    app.state.world_library = init_world_library(app.state.redis)
    await app.state.world_library.start_worker()

    # Connect to GPU worker
    await app.state.gpu_client.connect()
    print(f"WorldPlay Gateway started. GPU Worker: {GPU_HOST}:{GPU_PORT}")

    yield

    # Shutdown
    await app.state.world_library.stop_worker()
    await app.state.gpu_client.disconnect()
    await app.state.redis.close()
    print("WorldPlay Gateway shutdown complete.")


app = FastAPI(
    title="WorldPlay Gateway",
    description="Orchestrates real-time 3D world generation for immersive learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(generation.router, prefix="/api/v1/worlds", tags=["World Generation"])
app.include_router(streaming.router, prefix="/ws/worlds", tags=["Streaming"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gpu_status = await app.state.gpu_client.health_check()
    return {
        "status": "healthy",
        "gpu_worker": gpu_status,
        "redis": "connected" if app.state.redis else "disconnected"
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "WorldPlay Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Subject configurations
SUBJECT_CONFIGS = {
    "physics": {
        "topics": {
            "gravity": {
                "world_templates": [
                    "A space station interior with floating objects demonstrating zero gravity",
                    "Moon surface with an astronaut jumping to show reduced gravity",
                    "Galileo's leaning tower experiment with falling objects"
                ],
                "default_camera": "first_person",
                "interactive_elements": ["floating_objects", "jump_mechanics"]
            },
            "electricity": {
                "world_templates": [
                    "Inside an electrical circuit with visible electron flow",
                    "A power plant control room with real-time energy visualization"
                ],
                "default_camera": "third_person"
            },
            "waves": {
                "world_templates": [
                    "Ocean with visible wave patterns and interference",
                    "Sound waves visualized in a concert hall"
                ],
                "default_camera": "first_person"
            }
        }
    },
    "biology": {
        "topics": {
            "cell_structure": {
                "world_templates": [
                    "Journey inside a human cell with labeled organelles",
                    "Comparison of plant and animal cells side by side"
                ],
                "default_camera": "first_person",
                "scale": "microscopic"
            },
            "ecosystems": {
                "world_templates": [
                    "Amazon rainforest ecosystem with visible food web",
                    "Coral reef with marine life interactions"
                ],
                "default_camera": "first_person"
            }
        }
    },
    "history": {
        "topics": {
            "ancient_egypt": {
                "world_templates": [
                    "Construction of the Great Pyramid of Giza",
                    "Walking through ancient Alexandria's library"
                ],
                "default_camera": "first_person",
                "time_period": "2500 BCE"
            },
            "roman_empire": {
                "world_templates": [
                    "The Roman Colosseum during gladiator games",
                    "Walking the streets of ancient Rome"
                ],
                "default_camera": "first_person"
            }
        }
    },
    "chemistry": {
        "topics": {
            "molecular_structure": {
                "world_templates": [
                    "Inside a water molecule showing atomic bonds",
                    "Comparison of molecular structures: diamond vs graphite"
                ],
                "default_camera": "first_person",
                "scale": "atomic"
            },
            "reactions": {
                "world_templates": [
                    "Combustion reaction with visible molecular transformations",
                    "Acid-base neutralization at molecular level"
                ],
                "default_camera": "third_person"
            }
        }
    },
    "geography": {
        "topics": {
            "plate_tectonics": {
                "world_templates": [
                    "Earth's crust showing moving tectonic plates",
                    "Volcanic eruption with visible magma chamber"
                ],
                "default_camera": "third_person"
            },
            "climate_zones": {
                "world_templates": [
                    "Journey from equator to poles showing climate changes",
                    "Rainforest vs desert ecosystem comparison"
                ],
                "default_camera": "first_person"
            }
        }
    },
    "math": {
        "topics": {
            "3d_geometry": {
                "world_templates": [
                    "Floating 3D shapes that can be rotated and examined",
                    "Architectural space demonstrating geometric principles"
                ],
                "default_camera": "first_person"
            },
            "trigonometry": {
                "world_templates": [
                    "Unit circle in 3D space with interactive angles",
                    "Wave patterns showing sine and cosine relationships"
                ],
                "default_camera": "third_person"
            },
            "coordinate_systems": {
                "world_templates": [
                    "3D coordinate space with movable points",
                    "Graph visualization in immersive environment"
                ],
                "default_camera": "first_person"
            }
        }
    }
}


@app.get("/api/v1/subjects")
async def list_subjects():
    """List all available subjects."""
    return {
        "subjects": list(SUBJECT_CONFIGS.keys())
    }


@app.get("/api/v1/subjects/{subject}/topics")
async def list_topics(subject: str):
    """List topics for a specific subject."""
    if subject not in SUBJECT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' not found")

    topics = SUBJECT_CONFIGS[subject]["topics"]
    return {
        "subject": subject,
        "topics": [
            {
                "name": name,
                "templates": config.get("world_templates", []),
                "camera": config.get("default_camera", "first_person")
            }
            for name, config in topics.items()
        ]
    }


@app.get("/api/v1/subjects/{subject}/topics/{topic}/templates")
async def get_topic_templates(subject: str, topic: str):
    """Get world templates for a specific topic."""
    if subject not in SUBJECT_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Subject '{subject}' not found")

    topics = SUBJECT_CONFIGS[subject]["topics"]
    if topic not in topics:
        raise HTTPException(status_code=404, detail=f"Topic '{topic}' not found in {subject}")

    return {
        "subject": subject,
        "topic": topic,
        "templates": topics[topic].get("world_templates", []),
        "default_camera": topics[topic].get("default_camera", "first_person"),
        "interactive_elements": topics[topic].get("interactive_elements", [])
    }


# ============================================================================
# Pre-Generated World Library Endpoints
# ============================================================================

@app.get("/api/v1/worlds/library")
async def list_library_worlds():
    """List all pre-generated worlds in the library."""
    worlds = app.state.world_library.list_available_worlds()
    return {
        "worlds": worlds,
        "count": len(worlds)
    }


@app.get("/api/v1/worlds/library/{world_id}")
async def get_library_world(world_id: str):
    """Get metadata for a pre-generated world."""
    metadata = app.state.world_library.get_world_metadata(world_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")

    return metadata


@app.get("/api/v1/worlds/library/{world_id}/segments")
async def list_world_segments(world_id: str):
    """List available segments for a world."""
    segments = app.state.world_library.list_world_segments(world_id)
    if not segments:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found or has no segments")

    return {
        "world_id": world_id,
        "segments": segments,
        "count": len(segments)
    }


@app.get("/api/v1/worlds/library/{world_id}/segments/{segment_id}")
async def get_segment_url(world_id: str, segment_id: str):
    """Get CDN URL for a specific video segment."""
    url = app.state.world_library.get_world_segment_url(world_id, segment_id)
    if not url:
        raise HTTPException(status_code=404, detail=f"Segment '{segment_id}' not found in world '{world_id}'")

    return {
        "world_id": world_id,
        "segment_id": segment_id,
        "url": url
    }


class WorldGenerationRequest(BaseModel):
    topic: str
    prompt: str
    user_id: Optional[str] = None
    reference_image_url: Optional[str] = None


@app.post("/api/v1/worlds/generate-custom")
async def request_custom_world_generation(request: WorldGenerationRequest):
    """
    Request generation of a custom world (on-demand).

    This queues a background generation job. Check status with GET /api/v1/worlds/generation/{job_id}
    """
    result = await app.state.world_library.request_world_generation(
        topic=request.topic,
        prompt=request.prompt,
        user_id=request.user_id,
        reference_image_url=request.reference_image_url
    )

    return result


@app.get("/api/v1/worlds/generation/{job_id}")
async def get_generation_status(job_id: str):
    """Get status of a world generation job."""
    status = await app.state.world_library.get_generation_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Generation job '{job_id}' not found")

    return status


@app.get("/api/v1/worlds/pregenerated/{subject}/{topic}")
async def get_pregenerated_worlds_for_topic(subject: str, topic: str):
    """Get pre-generated worlds matching a subject/topic."""
    all_worlds = app.state.world_library.list_available_worlds()

    # Filter by subject
    matching_worlds = [
        w for w in all_worlds
        if w["subject"] == subject or subject.lower() in w["name"].lower()
    ]

    return {
        "subject": subject,
        "topic": topic,
        "pregenerated_worlds": matching_worlds
    }
