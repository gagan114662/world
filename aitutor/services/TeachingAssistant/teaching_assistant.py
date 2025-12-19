"""
Teaching Assistant - Refactored to be stateless
All state is stored in MongoDB via SessionManager.
Each method takes session_id or user_id as parameter.
"""

from typing import Optional, Dict, Any

from .greeting_handler import GreetingHandler
from .session_manager import SessionManager
from managers.mongodb_manager import MongoDBManager

from shared.logging_config import get_logger

logger = get_logger(__name__)


class TeachingAssistant:
    """
    Refactored to be stateless - all state stored in MongoDB.
    Each method takes session_id or user_id as parameter.
    """

    def __init__(self):
        mongo = MongoDBManager()
        self.session_manager = SessionManager(mongo)
        self.greeting_handler = GreetingHandler()
        logger.info("[TEACHING_ASSISTANT] Initialized with MongoDB-backed session manager")

    def start_session(self, user_id: str) -> dict:
        """Start a new session, returns greeting prompt and session info"""
        session = self.session_manager.create_session(user_id)
        greeting = self.greeting_handler.get_greeting(user_id)
        return {
            "session_id": session["session_id"],
            "prompt": greeting,
            "session_info": self.session_manager.get_session_info(session["session_id"])
        }

    def end_session(self, session_id: str) -> dict:
        """End session, returns closing prompt with stats"""
        session_summary = self.session_manager.end_session(session_id)
        if not session_summary:
            return {
                "prompt": "",
                "session_info": {"session_active": False}
            }

        closing = self.greeting_handler.get_closing(
            duration_minutes=session_summary.get("duration_minutes", 0),
            questions_answered=session_summary.get("questions_answered", 0)
        )
        return {
            "prompt": closing,
            "session_info": session_summary
        }

    def record_question_answered(
        self,
        session_id: str,
        question_id: str,
        is_correct: bool
    ) -> None:
        """Record a question answer"""
        self.session_manager.record_question_answered(session_id, is_correct)

    def record_conversation_turn(self, session_id: str) -> None:
        """Record a conversation turn"""
        self.session_manager.record_conversation_turn(session_id)

    def check_inactivity(self, session_id: str) -> Optional[str]:
        """Check inactivity and return prompt if needed"""
        if self.session_manager.check_inactivity(session_id):
            prompt = self.greeting_handler.get_inactivity_prompt()
            self.session_manager.push_instruction(session_id, prompt)
            return prompt
        return None

    def get_session_info(self, session_id: str) -> dict:
        """Get current session info"""
        return self.session_manager.get_session_info(session_id)

    def get_active_session(self, user_id: str) -> Optional[dict]:
        """Get active session for user"""
        return self.session_manager.get_active_session(user_id)

    def push_instruction(self, session_id: str, instruction: str) -> str:
        """Push an instruction to be delivered via SSE"""
        return self.session_manager.push_instruction(session_id, instruction)
