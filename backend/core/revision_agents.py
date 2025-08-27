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
        """
        Create new learning session with topic-specific configuration.
        - Fetches topic content and calculates difficulty-based parameters
        - Generates opening AI response to begin revision
        - Saves initial session state with conversation tracking
        
        Input: topic (str), student_id (str), session_id (str)
        Output: dict with response, session_id, conversation_count, sources, limits
        """

        topic_content = self.mongodb.get_topic_content(topic, limit=10)
        content_chunks = len(self.mongodb.get_topic_content(topic, limit=100))
        topic_config = Config.calculate_topic_limits(content_chunks)

        # Initial session data
        session_data = {
            "session_id": session_id,
            "student_id": student_id,
            "topic": topic,
            "started_at": datetime.now(),
            "conversation_count": 0,
            "is_complete": False,
            "stage": "revision",
            "quiz_frequency": topic_config["quiz_frequency"],
            "concepts_learned": [],
            "quiz_scores": [],
            "needs_remedial": False,
            "max_conversations": topic_config["max_conversations"],
            "completion_threshold": topic_config["completion_threshold"],
            "conversation_history": []
        }

        # Initial content
        content_text = "\n".join([chunk["text"][:400] for chunk in topic_content])
        revision_response = await self._generate_response_from_prompt(
            self.prompts.get_basic_revision_prompt(topic, content_text, is_start=True),
            "You are an expert educational tutor starting a revision session."
        )

        # Save initial turn
        session_data["conversation_history"].append({
            "turn": 0,
            "type": "revision_start",
            "assistant_message": revision_response,
            "timestamp": datetime.now()
        })
        self.mongodb.save_revision_session(session_data)

        return {
            "response": revision_response,
            "topic": topic,
            "session_id": session_id,
            "conversation_count": 0,
            "is_session_complete": False,
            "current_stage": "revision",
            "sources": [chunk.get("id", "Unknown") for chunk in topic_content],
            "max_conversations": topic_config["max_conversations"],
            "completion_threshold": topic_config["completion_threshold"]
        }

    async def handle_user_input(self, session_id: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """
        Process student input and determine next learning action.
        - Detects intent (quiz, question, continue, feedback, end) from user message
        - Routes to appropriate handler based on detected intent
        - Updates conversation history and progress metrics
        
        Input: session_id (str), user_query (Optional[str])
        Output: Dict with response, current_stage, is_session_complete, sources

        """

        session_data = self.mongodb.get_revision_session(session_id)
        if not session_data:
            return {"response": "Session not found.", "is_session_complete": False}

        # Update counters
        session_data["conversation_count"] += 1
        last_bot_message = session_data.get("conversation_history", [])[-1].get("assistant_message", "") if session_data.get("conversation_history") else ""

        # Detect intent
        intent = await self._detect_intent(user_query, last_bot_message)

        if intent == "END":
            response_data = await self._complete_session(session_data)
        elif intent == "QUIZ":
            response_data = await self._generate_auto_quiz(session_data)
        elif intent == "FEEDBACK":
            response_data = await self._generate_feedback_response(session_data, user_query)
        else:  # QUESTION or CONTINUE
            response_data = await self._handle_revision_or_question(session_data, user_query)

        # Save conversation turn
        await self._save_conversation_turn(session_data, user_query, response_data)

        return response_data

    async def _detect_intent(self, user_query: str, last_bot_message: str) -> str:
        """
        Classify user intent from conversation context.
        - Analyzes user message and previous AI response for context
        - Uses LLM to classify intent into predefined categories
        - Falls back to keyword matching if LLM classification fails
        
        Input: user_query (str), last_bot_message (str)
        Output: str (QUIZ, QUESTION, CONTINUE, FEEDBACK, or END)
        """
        if not user_query:
            return "CONTINUE"

        prompt = f"""
        Decide what the user wants to do based on the last assistant message and the user's message.

        Last assistant message: "{last_bot_message}"
        User message: "{user_query}"

        Classify into one of these:
        - QUIZ: If the user wants to start a quiz or confirms a quiz
        - QUESTION: If the user asks a question about the topic
        - CONTINUE: If the user wants to continue the lesson without quiz
        - FEEDBACK: If the user is submitting quiz answers or expects feedback
        - END: If the user wants to finish or stop the session

        Reply with ONLY one word: QUIZ, QUESTION, CONTINUE, FEEDBACK, END
        """

        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are a precise classifier. Reply with QUIZ, QUESTION, CONTINUE, FEEDBACK, or END."),
                HumanMessage(content=prompt)
            ])
            return response.strip().upper()
        except Exception:
            # Fallback
            text = user_query.lower()
            if any(x in text for x in ["quiz", "test"]):
                return "QUIZ"
            if any(x in text for x in ["end", "finish", "done", "quit"]):
                return "END"
            if len(text.split()) > 4 and text.endswith("."):  # likely feedback
                return "FEEDBACK"
            if any(x in text for x in ["what", "how", "why", "explain", "?"]):
                return "QUESTION"
            return "CONTINUE"

    async def _handle_revision_or_question(self, session_data: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Generate educational content for revision or answer questions.
        - Fetches relevant topic content chunks from database
        - Tracks new concepts learned during the interaction
        - Generates contextual AI response using revision prompts
        
        Input: session_data (Dict), user_query (str)
        Output: Dict with response, current_stage, is_session_complete, sources

        """
        topic_content = self.mongodb.get_topic_content(session_data["topic"], limit=3)
        content_text = "\n".join([chunk["text"][:300] for chunk in topic_content])

        # Add new concept
        if topic_content:
            concept = self._extract_concept_name(topic_content[0]["text"])
            if concept not in session_data.get("concepts_learned", []):
                session_data["concepts_learned"].append(concept)

        prompt = self.prompts.get_basic_revision_prompt(
            session_data["topic"],
            content_text,
            user_query,
            is_start=False,
            last_bot_message=session_data.get("conversation_history", [])[-1].get("assistant_message", "") if session_data.get("conversation_history") else ""
        )
        response = await self._generate_response_from_prompt(prompt, "You are an expert educational tutor continuing the lesson.")

        return {
            "response": response,
            "current_stage": "revision",
            "is_session_complete": False,
            "sources": [chunk.get("id", "Unknown") for chunk in topic_content]
        }

    async def _generate_feedback_response(self, session_data: Dict[str, Any], user_answers: str) -> Dict[str, Any]:
        """
        Evaluate quiz answers and provide performance feedback.
        - Scores student answers using LLM evaluation
        - Updates quiz scores and remedial learning flags
        - Generates personalized feedback based on performance level
        
        Input: session_data (Dict), user_answers (str)
        Output: Dict with feedback response, current_stage, performance_score

        """
        performance_score = await self._evaluate_quiz_performance(user_answers, session_data["topic"])
        level = "poor" if performance_score <= 0.5 else "good"

        session_data["quiz_scores"].append(performance_score)
        session_data["needs_remedial"] = (level == "poor")

        feedback = await self._generate_response_from_prompt(
            self.prompts.get_feedback_progress_prompt(
                user_answers,
                session_data["topic"],
                "",
                level,
                last_bot_message=session_data.get("conversation_history", [])[-1].get("assistant_message", "") if session_data.get("conversation_history") else ""
            ),
            "You are an expert tutor providing feedback."
        )

        return {
            "response": feedback,
            "current_stage": "feedback" if level == "poor" else "revision",
            "is_session_complete": False,
            "performance_score": performance_score
        }

   
    async def _generate_auto_quiz(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create adaptive quiz based on learned concepts.
        - Selects recent concepts for quiz generation
        - Adjusts difficulty based on previous quiz performance
        - Generates quiz questions using topic-specific prompts
        
        Input: session_data (Dict)
        Output: Dict with quiz questions, current_stage (quiz), is_session_complete
        """
        concepts = session_data.get('concepts_learned', [])[-3:] or [session_data["topic"]]
        quiz_scores = session_data.get('quiz_scores', [])
        difficulty = "medium"
        if quiz_scores:
            avg = sum(quiz_scores) / len(quiz_scores)
            difficulty = "easy" if avg < 0.5 else "medium" if avg < 0.8 else "hard"

        quiz_response = await self._generate_response_from_prompt(
            self.prompts.get_auto_quiz_prompt(session_data["topic"], concepts, difficulty),
            "You are an expert educational tutor creating automatic quizzes."
        )

        session_data["current_stage"] = "quiz"
        return {
            "response": quiz_response,
            "current_stage": "quiz",
            "is_session_complete": False
        }

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
