"""
Analysis Service - Handles complex AI analysis tasks like "Explain to Learn".
Uses Gemini to identify knowledge gaps and generate quizzes from audio explanations.
"""

import os
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from shared.logging_config import get_logger

logger = get_logger(__name__)

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str

class AnalysisResult(BaseModel):
    knowledge_gaps: List[str]
    quiz: List[QuizQuestion]

class AnalysisService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_id = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Default to flash for speed/cost
        
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"AnalysisService initialized with model: {self.model_id}")

    async def analyze_explanation(self, audio_content: bytes, mime_type: str = "audio/mp3") -> Optional[AnalysisResult]:
        """
        Analyze an audio explanation and generate knowledge gaps and a quiz.
        """
        if not self.client:
            logger.error("AnalysisService not properly initialized (missing API key)")
            return None

        prompt = """
        You are an expert AI Tutor. A student has provided an audio recording of themselves explaining a difficult concept. 
        Analyze the content of this audio. 
        1. Identify specific knowledge gaps or misconceptions in their explanation.
        2. Create a 3-question multiple choice quiz to test their understanding of the areas where they showed gaps or where the explanation was weak.
        3. For each question, provide a detailed explanation of why the correct answer is right and why the student might have missed it based on their specific explanation.

        Format your response as a valid JSON object with the following structure:
        {
          "knowledge_gaps": ["gap 1", "gap 2"],
          "quiz": [
            {
              "question": "...",
              "options": ["Option A", "Option B", "Option C", "Option D"],
              "correct_answer_index": 0,
              "explanation": "..."
            }
          ]
        }
        
        IMPORTANT: Return ONLY the raw JSON object. No markdown formatting, no preamble.
        """

        try:
            # The new google-genai SDK uses client.models.generate_content
            # We use wait_for_completion=True for simple blocking call in this async wrapper
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    types.Part.from_bytes(data=audio_content, mime_type=mime_type),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            if not response or not response.text:
                logger.error("Gemini returned an empty response")
                return None

            # Parse the JSON response
            data = json.loads(response.text)
            
            # Validate with Pydantic
            return AnalysisResult(**data)

        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}", exc_info=True)
            return None

analysis_service = AnalysisService()
