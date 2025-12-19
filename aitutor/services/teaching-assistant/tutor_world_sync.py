"""
Tutor-World Sync Service

Coordinates between Gemini AI tutor and WorldPlay world generation.
Decides when to trigger world generation and handles narration.
"""

import os
import asyncio
import httpx
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


WORLDPLAY_API_URL = os.getenv("WORLDPLAY_API_URL", "http://localhost:8010")


class LearningMode(Enum):
    """Current learning mode."""
    TEXT = "text"           # Standard text/voice tutoring
    WORLD = "world"         # Immersive 3D world exploration
    TRANSITION = "transition"  # Transitioning between modes


@dataclass
class TutoringResponse:
    """Response from tutoring analysis."""
    mode: LearningMode
    text_response: Optional[str] = None
    world_session_id: Optional[str] = None
    world_prompt: Optional[str] = None
    narration: Optional[str] = None
    audio_url: Optional[str] = None
    highlights: List[str] = None

    def __post_init__(self):
        if self.highlights is None:
            self.highlights = []


@dataclass
class StudentContext:
    """Context about the current student."""
    user_id: str
    grade_level: int
    subject: str
    topic: str
    session_id: str
    preferences: Dict[str, Any] = None
    performance_history: Dict[str, float] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.performance_history is None:
            self.performance_history = {}


# Enhanced system prompt for Gemini with world generation capabilities
GEMINI_WORLD_SYSTEM_PROMPT = """
You are an AI tutor with access to a real-time 3D world generation system called WorldPlay.
When explaining concepts that benefit from visual or spatial understanding, you can generate
immersive 3D worlds for students to explore and interact with.

## Available Functions

You have access to the following world generation functions:

1. **generate_world(subject, topic, prompt, learning_objective)**
   - Creates a new 3D world for the student to explore
   - Use descriptive prompts for better generation
   - Example: generate_world("physics", "gravity", "A space station with floating objects demonstrating weightlessness", "Understand how objects behave in microgravity")

2. **narrate_world(narration_text)**
   - Speak to the student while they explore the world
   - Guide their attention to important elements
   - Ask questions about what they observe

3. **highlight_element(element_name)**
   - Draw the student's attention to a specific element in the world
   - Example: highlight_element("floating_water_droplet")

4. **transition_scene(new_prompt)**
   - Smoothly transition to a related scene
   - Use for showing different aspects of the same concept
   - Example: transition_scene("The same space station, but now with artificial gravity spinning")

## When to Generate Worlds

Generate immersive worlds when the topic benefits from visual/spatial understanding:

**Physics:**
- Forces and motion (objects moving, colliding)
- Gravity (planets, orbits, free fall)
- Waves (sound, light, water)
- Electricity (circuits, electron flow)
- Magnetism (field visualization)

**Biology:**
- Cell structure (journey inside a cell)
- Ecosystems (rainforest, coral reef)
- Anatomy (organs, systems)
- Evolution (prehistoric scenes)

**Chemistry:**
- Molecular structure (3D molecules)
- Chemical reactions (bonds forming/breaking)
- States of matter (phase transitions)

**History:**
- Historical sites (pyramids, Colosseum)
- Historical events (reconstructions)
- Daily life in different eras

**Geography:**
- Landforms (mountains, rivers)
- Plate tectonics (earthquake zones)
- Climate zones (compare biomes)

**Math:**
- 3D geometry (shapes in space)
- Coordinate systems (3D graphs)
- Trigonometry (unit circle, waves)

## Teaching Approach in Worlds

When a student is exploring a world:

1. **Narrate actively**: Describe what they're seeing and why it matters
2. **Ask guiding questions**: "What do you notice about how the objects move?"
3. **Encourage exploration**: "Try moving forward to see the effect closer"
4. **Connect to concepts**: "This is exactly what Newton's first law describes"
5. **Check understanding**: "Why do you think the ball floats?"

## Response Format

When you decide to generate a world, respond with:
- A brief introduction explaining what they'll explore
- The world generation command
- Initial narration to guide them

When in text mode:
- Provide clear explanations
- Use analogies and examples
- Ask questions to check understanding

Always be encouraging and adapt to the student's pace and level.
"""


# Topics that benefit from world generation
WORLD_BENEFICIAL_TOPICS = {
    "physics": [
        "gravity", "motion", "forces", "electricity", "magnetism",
        "waves", "sound", "light", "energy", "momentum", "orbits",
        "projectile", "friction", "acceleration", "velocity"
    ],
    "biology": [
        "cell", "ecosystem", "anatomy", "evolution", "photosynthesis",
        "respiration", "circulation", "digestion", "nervous_system",
        "plants", "animals", "bacteria", "virus", "dna", "genetics"
    ],
    "chemistry": [
        "atoms", "molecules", "bonds", "reactions", "states_of_matter",
        "acids", "bases", "periodic_table", "electron", "compound",
        "solution", "mixture", "oxidation", "combustion"
    ],
    "history": [
        "ancient", "egypt", "rome", "greece", "medieval", "renaissance",
        "industrial", "world_war", "civilization", "empire", "revolution"
    ],
    "geography": [
        "plate_tectonics", "volcanoes", "earthquakes", "mountains",
        "rivers", "climate", "biomes", "erosion", "weather", "ocean"
    ],
    "math": [
        "3d_geometry", "coordinate", "trigonometry", "vectors",
        "graphing", "shapes", "angles", "transformations", "symmetry"
    ]
}


class TutorWorldSyncService:
    """
    Service that coordinates between Gemini AI tutor and WorldPlay.

    Analyzes student interactions to decide when visual/spatial learning
    would be beneficial and triggers world generation accordingly.
    """

    def __init__(self, gemini_client=None):
        """
        Initialize the sync service.

        Args:
            gemini_client: Client for Gemini API (optional, for testing)
        """
        self.gemini_client = gemini_client
        self.http_client = httpx.AsyncClient()
        self.active_world_sessions: Dict[str, str] = {}  # user_id -> world_session_id

    async def process_student_message(
        self,
        message: str,
        student_context: StudentContext
    ) -> TutoringResponse:
        """
        Process a student message and determine the appropriate response.

        Analyzes whether the topic would benefit from world generation
        and either triggers generation or returns a text response.

        Args:
            message: The student's message/question
            student_context: Context about the student

        Returns:
            TutoringResponse with mode and content
        """
        # Check if student is asking about a visual/spatial topic
        should_generate_world = self._should_generate_world(
            message=message,
            subject=student_context.subject,
            topic=student_context.topic
        )

        if should_generate_world:
            return await self._generate_world_response(message, student_context)
        else:
            return await self._generate_text_response(message, student_context)

    def _should_generate_world(
        self,
        message: str,
        subject: str,
        topic: str
    ) -> bool:
        """
        Determine if world generation would benefit the learning experience.

        Checks topic against known visual/spatial beneficial topics and
        analyzes the message for visual learning indicators.
        """
        message_lower = message.lower()

        # Check for explicit visual learning requests
        visual_keywords = [
            "show me", "visualize", "see", "look", "explore",
            "what does", "how does", "demonstrate", "example",
            "imagine", "picture", "3d", "world"
        ]

        has_visual_request = any(kw in message_lower for kw in visual_keywords)

        # Check if topic benefits from visualization
        subject_topics = WORLD_BENEFICIAL_TOPICS.get(subject.lower(), [])
        topic_benefits = any(t in topic.lower() for t in subject_topics)

        # Check if message mentions beneficial keywords
        all_beneficial_terms = []
        for terms in WORLD_BENEFICIAL_TOPICS.values():
            all_beneficial_terms.extend(terms)

        mentions_beneficial_topic = any(
            term.replace("_", " ") in message_lower or term in message_lower
            for term in all_beneficial_terms
        )

        # Generate world if:
        # 1. Explicit visual request, OR
        # 2. Topic naturally benefits from visualization AND message is asking about it
        return has_visual_request or (topic_benefits and mentions_beneficial_topic)

    async def _generate_world_response(
        self,
        message: str,
        student_context: StudentContext
    ) -> TutoringResponse:
        """Generate a response that includes world generation."""

        # Build world prompt based on context
        world_prompt = self._build_world_prompt(
            message=message,
            subject=student_context.subject,
            topic=student_context.topic,
            grade_level=student_context.grade_level
        )

        # Learning objective from the message
        learning_objective = self._extract_learning_objective(message)

        try:
            # Call WorldPlay API to generate world
            response = await self.http_client.post(
                f"{WORLDPLAY_API_URL}/api/v1/worlds/generate",
                json={
                    "session_id": None,  # Let server generate
                    "subject": student_context.subject,
                    "topic": student_context.topic,
                    "prompt": world_prompt,
                    "learning_objective": learning_objective,
                    "tutoring_session_id": student_context.session_id
                }
            )
            response.raise_for_status()
            world_data = response.json()

            # Store active world session
            self.active_world_sessions[student_context.user_id] = world_data["session_id"]

            # Generate initial narration
            narration = self._generate_initial_narration(
                subject=student_context.subject,
                topic=student_context.topic,
                world_prompt=world_prompt
            )

            return TutoringResponse(
                mode=LearningMode.WORLD,
                text_response=f"Let me show you this! I'm creating an immersive environment where you can explore {student_context.topic}.",
                world_session_id=world_data["session_id"],
                world_prompt=world_prompt,
                narration=narration
            )

        except Exception as e:
            print(f"World generation failed: {e}")
            # Fall back to text response
            return await self._generate_text_response(message, student_context)

    async def _generate_text_response(
        self,
        message: str,
        student_context: StudentContext
    ) -> TutoringResponse:
        """Generate a standard text tutoring response."""

        # In production, this would call Gemini API
        # For now, return a placeholder
        response_text = f"Let me explain {student_context.topic} in {student_context.subject}..."

        return TutoringResponse(
            mode=LearningMode.TEXT,
            text_response=response_text
        )

    def _build_world_prompt(
        self,
        message: str,
        subject: str,
        topic: str,
        grade_level: int
    ) -> str:
        """Build an optimized prompt for world generation."""

        # Subject-specific templates
        templates = {
            "physics": "A realistic environment demonstrating {topic} with clear visual indicators of physical forces and motion. Educational labels appear on key elements.",
            "biology": "An immersive biological environment showing {topic} at appropriate scale with color-coded structures and animated processes.",
            "chemistry": "A molecular-level visualization of {topic} with atoms and bonds clearly visible, showing chemical processes in action.",
            "history": "A historically accurate recreation of {topic} with period-appropriate architecture, clothing, and daily life.",
            "geography": "A realistic geographical landscape demonstrating {topic} with geological features and natural processes visible.",
            "math": "A clean 3D mathematical space visualizing {topic} with coordinate grids, labeled dimensions, and interactive elements."
        }

        base_template = templates.get(subject.lower(), "An educational environment demonstrating {topic}.")
        base_prompt = base_template.format(topic=topic.replace("_", " "))

        # Adjust complexity based on grade level
        if grade_level <= 6:
            complexity = "Simple and colorful, with clear labels and friendly presentation."
        elif grade_level <= 9:
            complexity = "Moderately detailed with educational annotations."
        else:
            complexity = "Detailed and scientifically accurate with advanced terminology."

        return f"{base_prompt} {complexity}"

    def _extract_learning_objective(self, message: str) -> str:
        """Extract or infer a learning objective from the message."""

        # Simple extraction - in production, use NLP
        if "how" in message.lower():
            return f"Understand {message.lower().replace('how', '').strip()}"
        elif "what" in message.lower():
            return f"Learn about {message.lower().replace('what', '').strip()}"
        elif "why" in message.lower():
            return f"Understand why {message.lower().replace('why', '').strip()}"
        else:
            return f"Explore and understand: {message[:100]}"

    def _generate_initial_narration(
        self,
        subject: str,
        topic: str,
        world_prompt: str
    ) -> str:
        """Generate initial narration for when the world loads."""

        narrations = {
            "physics": f"Welcome to our physics exploration! You're now in an environment where you can see {topic.replace('_', ' ')} in action. Use WASD to move around and explore. What do you notice first?",
            "biology": f"Welcome to our biology world! You're now exploring {topic.replace('_', ' ')}. Take a look around - can you identify the different structures?",
            "chemistry": f"Welcome to the molecular world! Here you can see {topic.replace('_', ' ')} at the atomic level. Notice how the atoms are arranged.",
            "history": f"Welcome to the past! You've traveled back in time to explore {topic.replace('_', ' ')}. Look around and observe how people lived.",
            "geography": f"Welcome to our geography exploration! You can now see {topic.replace('_', ' ')} up close. Explore the landscape and notice the formations.",
            "math": f"Welcome to mathematical space! Here you can interact with {topic.replace('_', ' ')} in three dimensions. Try moving around to see different perspectives."
        }

        return narrations.get(subject.lower(), f"Welcome! You're now exploring {topic}. Move around with WASD and look around with your mouse.")

    async def send_narration(
        self,
        session_id: str,
        narration: str,
        highlights: List[str] = None
    ) -> bool:
        """Send narration to an active world session."""

        if highlights is None:
            highlights = []

        try:
            # This would integrate with the world's narration WebSocket
            # For now, just log it
            print(f"Narration for {session_id}: {narration}")
            return True
        except Exception as e:
            print(f"Failed to send narration: {e}")
            return False

    async def transition_world(
        self,
        session_id: str,
        new_prompt: str,
        reason: str = "Showing related concept"
    ) -> bool:
        """Transition to a new scene in an active world."""

        try:
            response = await self.http_client.post(
                f"{WORLDPLAY_API_URL}/api/v1/worlds/{session_id}/transition",
                json={
                    "new_prompt": new_prompt,
                    "transition_type": "fade",
                    "duration_ms": 1000
                }
            )
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"World transition failed: {e}")
            return False

    async def end_world_session(self, user_id: str) -> bool:
        """End the active world session for a user."""

        session_id = self.active_world_sessions.get(user_id)
        if not session_id:
            return False

        try:
            response = await self.http_client.delete(
                f"{WORLDPLAY_API_URL}/api/v1/worlds/{session_id}"
            )
            response.raise_for_status()
            del self.active_world_sessions[user_id]
            return True
        except Exception as e:
            print(f"Failed to end world session: {e}")
            return False

    async def close(self):
        """Clean up resources."""
        await self.http_client.aclose()


# Singleton instance
tutor_world_sync = TutorWorldSyncService()
