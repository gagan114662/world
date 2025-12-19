"""
Cache Manager - handles multi-tier caching for world generation.
"""

import hashlib
import json
from datetime import datetime
from typing import List, Optional, Any
from dataclasses import dataclass

from app.models.schemas import PreGeneratedWorld


@dataclass
class CachedFrames:
    """Container for cached frame data."""
    frames: List[bytes]
    prompt: str
    subject: str
    topic: str
    created_at: datetime


class CacheManager:
    """
    Multi-tier cache manager for world generation.

    Tiers:
    - L1: In-memory frame buffer (handled by orchestrator)
    - L2: Redis for session state and recent frames
    - L3: S3/MinIO for pre-generated worlds (configured externally)
    """

    def __init__(self, redis_client):
        self.redis = redis_client

        # S3 client would be initialized here if using cloud storage
        # self.s3 = boto3.client('s3')
        self.storage_enabled = False  # Set True when S3 is configured

    def _compute_cache_key(self, subject: str, topic: str, prompt: str) -> str:
        """Compute a cache key from subject, topic, and prompt."""
        content = f"{subject}:{topic}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    async def get_pregenerated(
        self,
        subject: str,
        topic: str,
        prompt: str
    ) -> Optional[CachedFrames]:
        """
        Get pre-generated frames for a prompt if available.

        Returns None if no cached content exists.
        """
        cache_key = self._compute_cache_key(subject, topic, prompt)

        # Check Redis first (L2)
        cached = await self.redis.get(f"world:pregenerated:{cache_key}")
        if cached:
            data = json.loads(cached)
            # Frames would be stored as base64 or fetched from S3
            # For now, return metadata only
            return CachedFrames(
                frames=[],  # Would be populated from storage
                prompt=data.get("prompt", prompt),
                subject=subject,
                topic=topic,
                created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
            )

        # Check S3 (L3) if storage is enabled
        if self.storage_enabled:
            # s3_key = f"pregenerated/{subject}/{topic}/{cache_key}"
            # frames = await self._fetch_from_s3(s3_key)
            # if frames:
            #     return CachedFrames(...)
            pass

        return None

    async def list_pregenerated(
        self,
        subject: str,
        topic: str
    ) -> List[PreGeneratedWorld]:
        """List all pre-generated worlds for a subject/topic."""
        # Get from Redis set
        cache_keys = await self.redis.smembers(f"world:pregenerated:{subject}:{topic}")

        worlds = []
        for cache_key in cache_keys:
            metadata = await self.redis.hgetall(f"world:pregenerated:meta:{cache_key}")
            if metadata:
                worlds.append(PreGeneratedWorld(
                    cache_key=cache_key,
                    subject=subject,
                    topic=topic,
                    prompt=metadata.get("prompt", ""),
                    duration_seconds=float(metadata.get("duration", 30)),
                    frame_count=int(metadata.get("frame_count", 720)),
                    thumbnail_url=metadata.get("thumbnail_url"),
                    created_at=datetime.fromisoformat(
                        metadata.get("created_at", datetime.utcnow().isoformat())
                    )
                ))

        return worlds

    async def save_pregenerated(
        self,
        subject: str,
        topic: str,
        prompt: str,
        frames: List[bytes],
        thumbnail: Optional[bytes] = None
    ) -> str:
        """
        Save pre-generated world to cache.

        Returns the cache key.
        """
        cache_key = self._compute_cache_key(subject, topic, prompt)

        # Save metadata to Redis
        metadata = {
            "prompt": prompt,
            "duration": len(frames) / 24.0,
            "frame_count": len(frames),
            "created_at": datetime.utcnow().isoformat()
        }

        await self.redis.hset(
            f"world:pregenerated:meta:{cache_key}",
            mapping=metadata
        )

        # Add to topic's set of pregenerated worlds
        await self.redis.sadd(f"world:pregenerated:{subject}:{topic}", cache_key)

        # Store frames in S3 if enabled
        if self.storage_enabled:
            # await self._upload_to_s3(
            #     key=f"pregenerated/{subject}/{topic}/{cache_key}",
            #     frames=frames
            # )
            pass

        return cache_key

    async def invalidate(self, cache_key: str):
        """Invalidate a cached world."""
        # Get metadata to find subject/topic
        metadata = await self.redis.hgetall(f"world:pregenerated:meta:{cache_key}")
        if metadata:
            subject = metadata.get("subject", "")
            topic = metadata.get("topic", "")

            # Remove from set
            await self.redis.srem(f"world:pregenerated:{subject}:{topic}", cache_key)

        # Delete metadata
        await self.redis.delete(f"world:pregenerated:meta:{cache_key}")

        # Delete from S3 if enabled
        if self.storage_enabled:
            # await self._delete_from_s3(cache_key)
            pass

    async def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        # Count total pre-generated worlds
        total_worlds = 0
        subjects = ["physics", "biology", "history", "chemistry", "geography", "math"]

        for subject in subjects:
            # This is a simplified count - would need proper iteration in production
            pattern = f"world:pregenerated:{subject}:*"
            # Note: SCAN would be better for large datasets
            keys = await self.redis.keys(pattern)
            for key in keys:
                count = await self.redis.scard(key)
                total_worlds += count

        return {
            "total_pregenerated_worlds": total_worlds,
            "storage_enabled": self.storage_enabled
        }
