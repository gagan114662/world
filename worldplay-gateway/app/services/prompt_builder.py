"""
Prompt Builder - constructs optimized prompts for HY-WorldPlay based on educational context.
"""

from typing import Optional, Dict, Any


# Subject-specific prompt templates
PROMPT_TEMPLATES = {
    "physics": {
        "base": """
A scientifically accurate {scene_type} demonstrating {concept}.
The environment is {realism_level} with clear visual indicators of physical properties.
Interactive elements respond to physics realistically.
Educational labels appear when focusing on key elements.
{perspective} perspective view.
""",
        "style": "scientific, realistic, educational labels visible",
        "negative_prompt": "cartoon, unrealistic physics, fantasy elements, violence"
    },
    "biology": {
        "base": """
An accurate biological representation of {concept}.
The environment shows {biological_processes} in action at {scale} scale.
Color-coded labels identify key structures.
Dynamic processes are animated and clearly visible.
{perspective} perspective view.
""",
        "style": "biological accuracy, vibrant colors, educational, detailed",
        "negative_prompt": "cartoon, inaccurate anatomy, fantasy creatures"
    },
    "history": {
        "base": """
A historically accurate recreation of {concept} in {time_period}.
Architecture, clothing, and technology are period-appropriate.
The scene depicts {scene_type} with authentic details.
Ambient atmosphere reflects the historical era.
{perspective} perspective view.
""",
        "style": "historical accuracy, period-appropriate, cinematic, detailed architecture",
        "negative_prompt": "modern elements, anachronisms, fantasy"
    },
    "chemistry": {
        "base": """
A visualization of {concept} at {scale} scale.
Molecular structures and atomic bonds are clearly visible.
Chemical processes are shown with color-coded elements.
Reactions are animated showing transformations.
{perspective} perspective view.
""",
        "style": "scientific visualization, atomic detail, color-coded elements",
        "negative_prompt": "inaccurate chemistry, fantasy elements"
    },
    "geography": {
        "base": """
A realistic representation of {concept}.
Geological or geographical features are accurately depicted.
Scale and proportions are educational and accurate.
Natural processes are visible and labeled.
{perspective} perspective view.
""",
        "style": "realistic geography, aerial view capable, natural lighting",
        "negative_prompt": "fantasy landscapes, incorrect scale"
    },
    "math": {
        "base": """
A 3D visualization of {concept}.
Mathematical objects are clearly defined with labeled dimensions.
Coordinate systems and grids are visible when relevant.
Interactive manipulation of mathematical elements is possible.
{perspective} perspective view.
""",
        "style": "clean geometric visualization, grid lines, labeled axes",
        "negative_prompt": "cluttered, unclear geometry, distracting elements"
    }
}

# Default template for subjects not specifically defined
DEFAULT_TEMPLATE = {
    "base": """
An educational visualization of {concept}.
Clear, well-lit environment optimized for learning.
Key elements are highlighted and can be examined closely.
{perspective} perspective view.
""",
    "style": "educational, clear, well-organized",
    "negative_prompt": "confusing, dark, cluttered"
}


class PromptBuilder:
    """
    Builds optimized prompts for HY-WorldPlay based on educational context.

    Handles subject-specific prompt templates, learning objective integration,
    and prompt optimization for better generation quality.
    """

    def __init__(self):
        self.templates = PROMPT_TEMPLATES
        self.default_template = DEFAULT_TEMPLATE

    def build_prompt(
        self,
        subject: str,
        topic: str,
        base_prompt: str,
        learning_objective: Optional[str] = None,
        perspective: str = "first_person",
        scale: str = "human",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build an optimized prompt for world generation.

        Args:
            subject: The subject area (e.g., "physics", "biology")
            topic: Specific topic within the subject
            base_prompt: User-provided prompt description
            learning_objective: What the student should learn
            perspective: Camera perspective ("first_person" or "third_person")
            scale: Scale of the environment ("atomic", "microscopic", "human", "planetary")
            additional_context: Extra context parameters

        Returns:
            Optimized prompt string for HY-WorldPlay
        """
        # Get subject-specific template
        template_config = self.templates.get(subject.lower(), self.default_template)
        template = template_config["base"]
        style = template_config["style"]

        # Build context variables
        context = {
            "concept": topic,
            "scene_type": self._extract_scene_type(base_prompt),
            "realism_level": "photorealistic and scientifically accurate",
            "perspective": perspective.replace("_", " "),
            "scale": scale,
            "biological_processes": self._extract_processes(base_prompt, subject),
            "time_period": self._extract_time_period(base_prompt)
        }

        # Apply additional context
        if additional_context:
            context.update(additional_context)

        # Build the prompt
        structured_prompt = template.format(**context)

        # Add the user's specific description
        full_prompt = f"{base_prompt}\n\n{structured_prompt}"

        # Add learning objective if provided
        if learning_objective:
            full_prompt += f"\n\nEducational focus: {learning_objective}"

        # Add style guidance
        full_prompt += f"\n\nStyle: {style}"

        # Clean up whitespace
        full_prompt = " ".join(full_prompt.split())

        return full_prompt

    def get_negative_prompt(self, subject: str) -> str:
        """Get the negative prompt for a subject to improve generation quality."""
        template_config = self.templates.get(subject.lower(), self.default_template)
        base_negative = template_config.get("negative_prompt", "")

        # Add common negative prompts for educational content
        common_negative = "blurry, low quality, distorted, watermark, text overlay, inappropriate content"

        return f"{base_negative}, {common_negative}"

    def _extract_scene_type(self, prompt: str) -> str:
        """Extract the scene type from the prompt."""
        scene_keywords = {
            "interior": ["inside", "interior", "room", "building", "indoors"],
            "exterior": ["outside", "outdoor", "landscape", "nature", "sky"],
            "underwater": ["underwater", "ocean", "sea", "marine"],
            "space": ["space", "cosmic", "planet", "star", "galaxy"],
            "microscopic": ["cell", "molecular", "atom", "microscopic"],
            "historical_site": ["ancient", "ruins", "historical", "temple", "monument"]
        }

        prompt_lower = prompt.lower()
        for scene_type, keywords in scene_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return scene_type

        return "environment"

    def _extract_processes(self, prompt: str, subject: str) -> str:
        """Extract biological/chemical processes from the prompt."""
        if subject.lower() == "biology":
            process_keywords = {
                "photosynthesis": ["photosynthesis", "chloroplast", "plant cell"],
                "cell division": ["mitosis", "meiosis", "cell division"],
                "respiration": ["respiration", "mitochondria", "ATP"],
                "digestion": ["digestion", "stomach", "intestine"],
                "circulation": ["blood", "heart", "circulation", "arteries"]
            }
        elif subject.lower() == "chemistry":
            process_keywords = {
                "combustion": ["combustion", "burning", "fire"],
                "oxidation": ["oxidation", "rust", "corrosion"],
                "bonding": ["bond", "ionic", "covalent", "electron sharing"]
            }
        else:
            return "natural processes"

        prompt_lower = prompt.lower()
        for process, keywords in process_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return process

        return "biological processes" if subject.lower() == "biology" else "chemical processes"

    def _extract_time_period(self, prompt: str) -> str:
        """Extract historical time period from the prompt."""
        time_periods = {
            "ancient Egypt (3000-30 BCE)": ["egypt", "pyramid", "pharaoh", "nile"],
            "ancient Rome (500 BCE - 500 CE)": ["rome", "roman", "colosseum", "gladiator"],
            "ancient Greece (800-31 BCE)": ["greece", "greek", "athens", "parthenon"],
            "medieval Europe (500-1500 CE)": ["medieval", "castle", "knight", "feudal"],
            "Renaissance (1400-1600 CE)": ["renaissance", "leonardo", "florence"],
            "Industrial Revolution (1760-1840)": ["industrial", "factory", "steam"],
            "World War era (1914-1945)": ["world war", "wwi", "wwii", "trench"]
        }

        prompt_lower = prompt.lower()
        for period, keywords in time_periods.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return period

        return "the historical period"

    def enhance_for_interactivity(self, prompt: str, interactive_elements: list) -> str:
        """Enhance prompt with interactive element descriptions."""
        if not interactive_elements:
            return prompt

        element_descriptions = []
        for element in interactive_elements:
            if element == "floating_objects":
                element_descriptions.append("objects that float and can be pushed")
            elif element == "jump_mechanics":
                element_descriptions.append("surfaces where jumping demonstrates gravity")
            elif element == "switches":
                element_descriptions.append("interactive switches and controls")
            elif element == "zoom_capable":
                element_descriptions.append("details visible at multiple zoom levels")

        if element_descriptions:
            prompt += f"\n\nInteractive elements: {', '.join(element_descriptions)}"

        return prompt
