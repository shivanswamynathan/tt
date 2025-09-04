from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
import logging
from backend.core.llm import GeminiLLMWrapper
from backend.core.mongodb_client import MongoDBClient
from backend.models.schemas import SessionState
from backend.config import Config
from backend.prompts.revision_prompts import RevisionPrompts
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicRevisionAgent:
    """
    AI tutor agent for managing adaptive learning sessions.
    - Orchestrates revision, quiz, and feedback phases based on student progress
    - Detects student intent and routes to appropriate response handlers
    - Persists conversation history and tracks learning metrics in MongoDB
    """

    def __init__(self, llm_wrapper: GeminiLLMWrapper, mongodb_client: MongoDBClient):
        self.llm = llm_wrapper
        self.mongodb = mongodb_client
        self.prompts = RevisionPrompts()

    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> dict:
        """Start with brief topic intro + proceed question"""
        
        # Get subtopics for this topic
        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        subtopics_count = len(subtopics)
        
        # Calculate topic config
        topic_config = Config.calculate_topic_limits(subtopics_count)
        
        # Initialize session with subtopic tracking
        session_data = {
            "session_id": session_id,
            "student_id": student_id,
            "topic": topic,
            "started_at": datetime.now(),
            "conversation_count": 0,
            "is_complete": False,
            "stage": "topic_intro",
            "quiz_frequency": topic_config["quiz_frequency"],
            "concepts_learned": [],
            "quiz_scores": [],
            "needs_remedial": False,
            "max_conversations": topic_config["max_conversations"],
            "completion_threshold": topic_config["completion_threshold"],
            "conversation_history": [],
            # New subtopic tracking
            "current_subtopic_index": 0,
            "subtopic_completion_status": [False] * subtopics_count,
            "subtopic_quiz_scores": []
        }
        
        # Generate brief intro using new prompt
        intro_response = await self._generate_response_from_prompt(
            self.prompts.get_topic_introduction_prompt(topic, subtopics_count),
            "You are a mobile-friendly tutor. Follow the prompt instructions for length and tone (usually 4-5 lines). Be engaging and simple."

        )
        
        # Save initial turn
        session_data["conversation_history"].append({
            "turn": 0,
            "type": "topic_intro",
            "assistant_message": intro_response,
            "timestamp": datetime.now()
        })
        self.mongodb.save_revision_session(session_data)
        
        return {
            "response": intro_response,
            "topic": topic,
            "session_id": session_id,
            "conversation_count": 0,
            "is_session_complete": False,
            "current_stage": "topic_intro",
            "sources": [],
            "max_conversations": topic_config["max_conversations"],
            "completion_threshold": topic_config["completion_threshold"]
        }

    async def handle_user_input(self, session_id: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """Process user input with subtopic-based flow control"""
        
        session_data = self.mongodb.get_revision_session(session_id)
        if not session_data:
            return {"response": "Session not found.", "is_session_complete": False}

        # Update conversation count
        session_data["conversation_count"] += 1
        last_bot_message = session_data.get("conversation_history", [])[-1].get("assistant_message", "") if session_data.get("conversation_history") else ""

        # Detect intent
        intent = await self._detect_intent(user_query, last_bot_message)

        # Route based on intent
        if intent == "END":
            response_data = await self._complete_session(session_data)
        elif intent == "PROCEED":
            response_data = await self._start_first_subtopic(session_data, last_bot_message)
        elif intent == "LEARN":
            response_data = await self._teach_current_subtopic(session_data, last_bot_message)
        elif intent == "QUIZ":
            response_data = await self._generate_subtopic_quiz(session_data, last_bot_message)
        elif intent == "FEEDBACK":
            response_data = await self._evaluate_subtopic_quiz(session_data, user_query, last_bot_message)
        elif intent == "RETRY":
            response_data = await self._retry_subtopic_explanation(session_data, last_bot_message)
        elif intent == "NEXT":
            response_data = await self._move_to_next_subtopic(session_data, last_bot_message)
        else:
            response_data = await self._handle_general_question(session_data, user_query, last_bot_message)

        # Save conversation turn
        await self._save_conversation_turn(session_data, user_query, response_data)
        return response_data

    async def _detect_intent(self, user_query: str, last_bot_message: str) -> str:
        """Enhanced intent detection for subtopic flow"""
        if not user_query:
            return "CONTINUE"

        prompt = f"""
        Decide what the user wants based on conversation context.

        Last assistant message: "{last_bot_message}"
        User message: "{user_query}"

        Classify into one of these:
        - PROCEED: User wants to start learning concepts/subtopics
        - LEARN: User wants to learn current subtopic
        - QUIZ: User wants quiz for current subtopic  
        - FEEDBACK: User submitted quiz answers
        - RETRY: User wants to retry after failing
        - NEXT: User wants to move to next subtopic
        - END: User wants to finish session

        Reply with ONLY one word: PROCEED, LEARN, QUIZ, FEEDBACK, RETRY, NEXT, END
        """

        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are a precise classifier. Reply with PROCEED, LEARN, QUIZ, FEEDBACK, RETRY, NEXT, or END."),
                HumanMessage(content=prompt)
            ])
            return response.strip().upper()
        except Exception:
            # Enhanced fallback logic
            text = user_query.lower()
            if any(x in text for x in ["yes", "ready", "start", "proceed", "begin"]):
                return "PROCEED"
            if any(x in text for x in ["quiz", "test", "check"]):
                return "QUIZ"
            if any(x in text for x in ["next", "move on", "continue"]):
                return "NEXT"
            if any(x in text for x in ["retry", "again", "explain"]):
                return "RETRY"
            if any(x in text for x in ["end", "finish", "done", "stop"]):
                return "END"
            if "=" in text or any(x in text for x in ["answer", "a)", "b)", "c)", "true", "false"]):
                return "FEEDBACK"
            return "LEARN"

    async def _evaluate_quiz_performance(self, user_answers: str, topic: str) -> float:
        """
        Score student quiz answers using AI evaluation.
        - Uses LLM to evaluate answer quality and correctness
        - Extracts numerical score from AI response
        - Returns normalized score between 0.0 and 1.0
        
        Input: user_answers (str), topic (str)
        Output: float (performance score 0.0-1.0)
        """
        prompt = f"""
        Evaluate the student's quiz answers for "{topic}".
        Answers: {user_answers}
        Score from 0.0 to 1.0 (only number).
        """

        try:
            response = await self.llm.generate_response([
                SystemMessage(content="Reply only with a number between 0.0 and 1.0"),
                HumanMessage(content=prompt)
            ])
            import re
            match = re.search(r'0\.\d+|1\.0|0|1', response)
            return float(match.group()) if match else 0.5
        except:
            return 0.5

    async def _generate_response_from_prompt(self, prompt: str, system_message: str) -> str:

        """
        Generate AI response using structured prompts.
        - Combines system instructions with specific prompt content
        - Sends formatted messages to LLM for response generation
        - Returns clean AI-generated educational content
        
        Input: prompt (str), system_message (str)
        Output: str (AI-generated response)
        """
        return await self.llm.generate_response([
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ])

    def _extract_concept_name(self, text: str) -> str:
        """
        Extract key concept from content text for tracking.
        - Identifies main concept from first sentence or text chunk
        - Limits concept name to first 3 words or 50 characters
        - Provides concept labels for learning progress tracking
        
        Input: text (str)
        Output: str (extracted concept name)
        """
        first_sentence = text.split('.')[0] if '.' in text else text
        words = first_sentence.split()
        return " ".join(words[:3]) if len(words) >= 3 else first_sentence[:50]

    async def _save_conversation_turn(self, session_data: Dict[str, Any], user_query: Optional[str], response_data: Dict[str, Any]):

        """
        Persist conversation interaction and update progress.
        - Saves individual conversation turn with timestamps
        - Updates session progress metrics and learning state
        - Maintains complete conversation history for context
        
        Input: session_data (Dict), user_query (Optional[str]), response_data (Dict)
        Output: None (saves to database)
        """
        turn_data = {
            "turn": session_data["conversation_count"],
            "user_message": user_query,
            "assistant_message": response_data["response"],
            "stage": response_data["current_stage"],
            "timestamp": datetime.now()
        }
        self.mongodb.save_conversation_turn(session_data["session_id"], turn_data)

        progress_data = {
            "conversation_count": session_data["conversation_count"],
            "current_stage": response_data["current_stage"],
            "concepts_learned": session_data.get("concepts_learned", []),
            "quiz_scores": session_data.get("quiz_scores", []),
            "needs_remedial": session_data.get("needs_remedial", False)
        }
        self.mongodb.update_session_progress(session_data["session_id"], progress_data)

    async def _complete_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:

        """
        Finalize session with performance summary.
        - Calculates final statistics including average scores and concepts learned
        - Generates session completion summary for student
        - Marks session as complete in database with final metrics
        
        Input: session_data (Dict)
        Output: Dict with completion response, is_session_complete (True), session_summary
        """
        quiz_scores = session_data.get('quiz_scores', [])
        avg_score = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0
        concepts_learned = len(session_data.get('concepts_learned', []))

        summary = f"Session complete! You learned {concepts_learned} concepts and achieved {avg_score:.1%} average score."
        final_data = {
            "is_complete": True,
            "completed_at": datetime.now(),
            "final_stats": {
                "total_interactions": session_data["conversation_count"],
                "concepts_learned": concepts_learned,
                "quizzes_taken": len(quiz_scores),
                "average_performance": avg_score
            },
            "session_summary": summary
        }
        self.mongodb.update_session_progress(session_data["session_id"], final_data)
        return {
            "response": summary,
            "is_session_complete": True,
            "session_summary": summary
        }
    async def _start_first_subtopic(self, session_data: Dict[str, Any], last_bot_message: str) -> Dict[str, Any]:
        """Start learning the first subtopic"""
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        
        if not subtopics:
            return {"response": "No subtopics found.", "current_stage": "error", "is_session_complete": False}
        
        first_subtopic = subtopics[0]
        session_data["current_subtopic_index"] = 0
        session_data["stage"] = "learning"
        
        response = await self._generate_response_from_prompt(
            self.prompts.get_subtopic_learning_prompt(
                first_subtopic["subtopic_title"],
                first_subtopic["content"],
                first_subtopic["subtopic_number"],
                last_bot_message
            ),
            "You are a mobile-friendly tutor. Keep responses to maximum 2 lines."
        )
        
        return {
            "response": response,
            "current_stage": "learning",
            "is_session_complete": False,
            "sources": [first_subtopic["subtopic_number"]]
        }

    async def _teach_current_subtopic(self, session_data: Dict[str, Any], last_bot_message: str) -> Dict[str, Any]:
        """Teach the current subtopic"""
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        current_index = session_data.get("current_subtopic_index", 0)
        
        if current_index >= len(subtopics):
            return await self._complete_session(session_data)
        
        current_subtopic = subtopics[current_index]
        
        response = await self._generate_response_from_prompt(
            self.prompts.get_subtopic_learning_prompt(
                current_subtopic["subtopic_title"],
                current_subtopic["content"],
                current_subtopic["subtopic_number"],
                last_bot_message
            ),
            "You are a mobile-friendly tutor. Keep responses to maximum 2 lines."
        )
        
        return {
            "response": response,
            "current_stage": "learning", 
            "is_session_complete": False,
            "sources": [current_subtopic["subtopic_number"]]
        }

    async def _generate_subtopic_quiz(self, session_data: Dict[str, Any], last_bot_message: str) -> Dict[str, Any]:
        """Generate 2-question quiz for current subtopic"""
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        current_index = session_data.get("current_subtopic_index", 0)
        
        current_subtopic = subtopics[current_index]
        session_data["stage"] = "quiz"
        
        response = await self._generate_response_from_prompt(
            self.prompts.get_subtopic_quiz_prompt(
                current_subtopic["subtopic_title"],
                current_subtopic["content"],
                current_subtopic["subtopic_number"],
                last_bot_message
            ),
            "Create exactly 2 questions. Keep it short."
        )
        
        return {
            "response": response,
            "current_stage": "quiz",
            "is_session_complete": False
        }

    async def _evaluate_subtopic_quiz(self, session_data: Dict[str, Any], user_answers: str, last_bot_message: str) -> Dict[str, Any]:
        """Check answers and give pass/fail feedback"""
        score = await self._evaluate_quiz_performance(user_answers, session_data["topic"])
        passed = score >= 0.6
        
        current_index = session_data.get("current_subtopic_index", 0)
        
        # Initialize tracking lists if needed
        if "subtopic_quiz_scores" not in session_data:
            session_data["subtopic_quiz_scores"] = []
        if "subtopic_completion_status" not in session_data:
            session_data["subtopic_completion_status"] = []
        
        # Extend lists using range/for loop
        needed_completion_length = current_index + 1
        for i in range(len(session_data["subtopic_completion_status"]), needed_completion_length):
            session_data["subtopic_completion_status"].append(False)
            
        needed_scores_length = current_index + 1  
        for i in range(len(session_data["subtopic_quiz_scores"]), needed_scores_length):
            session_data["subtopic_quiz_scores"].append(0.0)
        
        session_data["subtopic_quiz_scores"][current_index] = score
        session_data["subtopic_completion_status"][current_index] = passed

        self.mongodb.update_session_progress(session_data["session_id"], {
            "subtopic_quiz_scores": session_data["subtopic_quiz_scores"],
            "subtopic_completion_status": session_data["subtopic_completion_status"]
        })
        
        feedback_response = await self._generate_response_from_prompt(
            self.prompts.get_subtopic_feedback_prompt(
                user_answers,
                "",  # Don't need correct answers for feedback
                passed,
                f"3.{current_index + 1}",
                last_bot_message
            ),
            "Give brief feedback. Maximum 2 lines."
        )
        
        if passed:
            session_data["stage"] = "ready_next"
            next_stage = "passed"
        else:
            session_data["stage"] = "retry"
            next_stage = "failed"
        
        return {
            "response": feedback_response,
            "current_stage": next_stage,
            "is_session_complete": False,
            "performance_score": score
        }

    async def _retry_subtopic_explanation(self, session_data: Dict[str, Any], last_bot_message: str) -> Dict[str, Any]:
        """Re-explain current subtopic simply"""
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        current_index = session_data.get("current_subtopic_index", 0)
        
        current_subtopic = subtopics[current_index]
        
        response = await self._generate_response_from_prompt(
            self.prompts.get_retry_explanation_prompt(
                current_subtopic["subtopic_title"],
                current_subtopic["content"],
                current_subtopic["subtopic_number"],
                last_bot_message
            ),
            "You are re-explaining simply. Maximum 2 lines."
        )
        
        return {
            "response": response,
            "current_stage": "retry_learning",
            "is_session_complete": False,
            "sources": [current_subtopic["subtopic_number"]]
        }

    async def _move_to_next_subtopic(self, session_data: Dict[str, Any], last_bot_message: str) -> Dict[str, Any]:
        current_index = session_data.get("current_subtopic_index", 0)
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)

        completion_status = session_data.get("subtopic_completion_status", [])
        for i in range(len(completion_status), len(subtopics)):
            completion_status.append(False)
        session_data["subtopic_completion_status"] = completion_status

        next_index = current_index + 1

        if next_index >= len(subtopics):
            quiz_scores = session_data.get("subtopic_quiz_scores", [])
            overall_score = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0
            passed_count = sum(1 for status in completion_status if status)

            final_response = await self._generate_response_from_prompt(
                self.prompts.get_final_assessment_prompt(overall_score, len(subtopics), passed_count, last_bot_message),
                "Give final celebration. Follow prompt style (4-5 lines)."
            )

            session_data["is_complete"] = True
            return {
                "response": final_response,
                "current_stage": "complete",
                "is_session_complete": True,
                "session_summary": final_response
            }

        warning_msg = ""
        if not completion_status[current_index]:
            warning_msg = "You didn’t pass the last quiz, but let’s keep going and improve! ✅\n\n"

        session_data["current_subtopic_index"] = next_index
        next_subtopic = subtopics[next_index]

        response = await self._generate_response_from_prompt(
            self.prompts.get_subtopic_learning_prompt(
                next_subtopic["subtopic_title"],
                next_subtopic["content"],
                next_subtopic["subtopic_number"],
                last_bot_message
            ),
            "You are a mobile-friendly tutor. Follow the prompt style (4-5 lines, engaging)."
        )

        return {
            "response": warning_msg + response,
            "current_stage": "learning",
            "is_session_complete": False,
            "sources": [next_subtopic["subtopic_number"]]
        }


    async def _handle_general_question(self, session_data: Dict[str, Any], user_query: str, last_bot_message: str) -> Dict[str, Any]:
        """Handle general questions about current subtopic"""
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        current_index = session_data.get("current_subtopic_index", 0)
        
        if current_index < len(subtopics):
            current_subtopic = subtopics[current_index]
            content_text = current_subtopic["content"][:300]
        else:
            content_text = "General topic content"
        
        response = await self._generate_response_from_prompt(
            f"""
            Previous assistant message: "{last_bot_message}"
            Student asked: "{user_query}"
            
            Answer briefly in 1-2 lines using this content:
            {content_text}
            
            Be helpful and ask if they want to continue learning.
            """,
            "You are answering a question briefly. Maximum 2 lines."
        )
        
        return {
            "response": response,
            "current_stage": "question_answered",
            "is_session_complete": False,
            "sources": [f"3.{current_index + 1}"] if current_index < len(subtopics) else []
        }
