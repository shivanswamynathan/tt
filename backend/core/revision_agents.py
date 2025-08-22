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
    """Dynamic Revision Agent with LLM-based question detection and automatic quizzes"""
    
    def __init__(self, llm_wrapper: GeminiLLMWrapper, mongodb_client: MongoDBClient):
        self.llm = llm_wrapper
        self.mongodb = mongodb_client
        self.session_states: Dict[str, SessionState] = {}
        self.prompts = RevisionPrompts()
        
    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> dict:
        """Start session with dynamic topic configuration"""
        
        # Get topic content and calculate dynamic limits
        topic_content = self.mongodb.get_topic_content(topic, limit=10)
        content_chunks = len(self.mongodb.get_topic_content_chunks(topic))
        
        # Calculate dynamic configuration
        topic_config = Config.calculate_topic_limits(content_chunks)
        
        # Create session state
        session_state = SessionState(
            session_id=session_id,
            topic=topic,
            student_id=student_id,
            conversation_count=0,
            started_at=datetime.now(),
            last_interaction=datetime.now(),
            is_complete=False,
            key_concepts_covered=[],
            user_understanding_level="normal",
            max_conversations=topic_config["max_conversations"],
            completion_threshold=topic_config["completion_threshold"]
        )
        
        # Initialize dynamic session tracking
        session_state.current_stage = "revision"
        session_state.quiz_frequency = topic_config["quiz_frequency"]
        session_state.concepts_learned = []
        session_state.quiz_scores = []
        session_state.needs_remedial = False
        
        self.session_states[session_id] = session_state
        
        # Generate initial revision content
        content_text = "\n".join([chunk["text"][:400] for chunk in topic_content])
        
        revision_response = await self._generate_response_from_prompt(
            self.prompts.get_basic_revision_prompt(topic, content_text, is_start=True),
            "You are an expert educational tutor starting a revision session."
        )
        
        # Save session
        self._save_initial_session(session_id, student_id, topic, revision_response, topic_config)
        
        return {
            "response": revision_response,
            "topic": topic,
            "session_id": session_id,
            "conversation_count": 0,
            "is_session_complete": False,
            "current_stage": "revision",
            "sources": [chunk.get("chunk_id", "Unknown") for chunk in topic_content],
            "max_conversations": topic_config["max_conversations"],
            "completion_threshold": topic_config["completion_threshold"]
        }
    
    async def continue_revision(self, session_id: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """Continue revision with dynamic flow and LLM-based question detection"""
        
        # Get session
        session_state = await self._get_or_restore_session(session_id)
        if not session_state:
            return {"response": "Session not found. Please start a new revision session.", "is_session_complete": False}
        
        session_state.conversation_count += 1
        session_state.last_interaction = datetime.now()
        
        # Check for session end
        if user_query and await self._wants_to_end_session(user_query):
            return await self._complete_session(session_state)
        
        # Dynamic question detection using LLM
        if user_query and await self._is_question(user_query):
            response_data = await self._handle_user_question(session_state, user_query)
        
        # Check if it's time for automatic quiz
        elif self._should_auto_quiz(session_state):
            response_data = await self._generate_auto_quiz(session_state)
        
        # Process quiz answers if quiz is in progress
        elif session_state.current_stage == "quiz":
            response_data = await self._process_quiz_answers(session_state, user_query)
        
        # Continue with revision content
        else:
            response_data = await self._continue_revision_content(session_state, user_query)
        
        # Save conversation
        await self._save_conversation_turn(session_state, user_query, response_data)
        
        # Add metadata
        response_data.update({
            "topic": session_state.topic,
            "session_id": session_id,
            "conversation_count": session_state.conversation_count,
            "max_conversations": session_state.max_conversations,
            "completion_threshold": session_state.completion_threshold
        })
        
        return response_data
    
    async def _is_question(self, user_input: str) -> bool:
        """LLM-based dynamic question detection"""
        if not user_input or len(user_input.strip()) < 3:
            return False
        
        prompt = f"""
        Is this user input a question that needs an educational answer? Reply only "YES" or "NO"
        
        User input: "{user_input}"
        
        Consider it a question if the user is:
        - Asking for explanation or clarification
        - Seeking help understanding something
        - Requesting more information
        - Expressing confusion
        
        Reply only: YES or NO
        """
        
        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are a precise classifier. Reply only YES or NO."),
                HumanMessage(content=prompt)
            ])
            return "YES" in response.upper()
        except Exception as e:
            logger.error(f"Question detection error: {e}")
            # Fallback to simple detection
            return "?" in user_input or any(word in user_input.lower() for word in ["what", "how", "why", "explain"])
    
    async def _wants_to_end_session(self, user_input: str) -> bool:
        """LLM-based session end detection"""
        prompt = f"""
        Does the user want to end or stop the learning session? Reply only "YES" or "NO"
        
        User input: "{user_input}"
        
        Reply only: YES or NO
        """
        
        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are a precise classifier. Reply only YES or NO."),
                HumanMessage(content=prompt)
            ])
            return "YES" in response.upper()
        except Exception as e:
            logger.error(f"End session detection error: {e}")
            return any(phrase in user_input.lower() for phrase in ["end", "finish", "stop", "done", "exit"])
    
    def _should_auto_quiz(self, session_state: SessionState) -> bool:
        """Determine if it's time for automatic quiz"""
        quiz_freq = getattr(session_state, 'quiz_frequency', 5)
        
        # Auto quiz conditions:
        # 1. Every N interactions (based on topic size)
        # 2. Not already in quiz
        # 3. Has learned some concepts
        
        return (
            session_state.conversation_count % quiz_freq == 0 and
            session_state.conversation_count > 2 and
            session_state.current_stage != "quiz" and
            len(getattr(session_state, 'concepts_learned', [])) > 0
        )
    
    async def _handle_user_question(self, session_state: SessionState, user_query: str) -> Dict[str, Any]:
        """Handle user question using Type 1 prompt"""
        
        # Search for relevant content
        relevant_content = self.mongodb.search_topic_content(session_state.topic, user_query, limit=3)
        context = "\n".join([chunk["text"][:300] for chunk in relevant_content]) if relevant_content else ""
        
        # Generate answer using basic revision prompt
        response = await self._generate_response_from_prompt(
            self.prompts.get_basic_revision_prompt(session_state.topic, context, user_query),
            "You are an expert educational tutor answering student questions."
        )
        
        return {
            "response": response,
            "current_stage": "question_answered",
            "is_session_complete": False,
            "sources": [chunk.get("chunk_id", "Unknown") for chunk in relevant_content]
        }
    
    async def _generate_auto_quiz(self, session_state: SessionState) -> Dict[str, Any]:
        """Generate automatic quiz using Type 2 prompt"""
        
        # Get recent concepts learned
        concepts = getattr(session_state, 'concepts_learned', [])[-3:] or [session_state.topic]
        
        # Determine difficulty based on performance history
        quiz_scores = getattr(session_state, 'quiz_scores', [])
        if quiz_scores:
            avg_score = sum(quiz_scores) / len(quiz_scores)
            difficulty = "easy" if avg_score < 0.5 else "medium" if avg_score < 0.8 else "hard"
        else:
            difficulty = "medium"
        
        # Generate quiz
        quiz_response = await self._generate_response_from_prompt(
            self.prompts.get_auto_quiz_prompt(session_state.topic, concepts, difficulty),
            "You are an expert educational tutor creating automatic quizzes."
        )
        
        # Update session state
        session_state.current_stage = "quiz"
        session_state.current_quiz_concepts = concepts
        
        return {
            "response": quiz_response,
            "current_stage": "quiz",
            "is_session_complete": False,
            "sources": []
        }
    
    async def _process_quiz_answers(self, session_state: SessionState, user_answers: str) -> Dict[str, Any]:
        """Process quiz answers and provide feedback using Type 3 prompt"""
        
        # Evaluate performance using LLM
        performance_score = await self._evaluate_quiz_performance(user_answers, session_state.topic)
        performance_level = "poor" if performance_score <= 0.5 else "good"
        
        # Store performance
        quiz_scores = getattr(session_state, 'quiz_scores', [])
        quiz_scores.append(performance_score)
        session_state.quiz_scores = quiz_scores
        session_state.needs_remedial = (performance_level == "poor")
        
        # Generate feedback
        feedback_response = await self._generate_response_from_prompt(
            self.prompts.get_feedback_progress_prompt(user_answers, session_state.topic, "", performance_level),
            "You are an expert educational tutor providing quiz feedback."
        )
        
        # Update stage
        session_state.current_stage = "feedback" if performance_level == "poor" else "revision"
        
        return {
            "response": feedback_response,
            "current_stage": session_state.current_stage,
            "is_session_complete": False,
            "sources": [],
            "performance_score": performance_score,
            "needs_remedial": session_state.needs_remedial
        }
    
    async def _continue_revision_content(self, session_state: SessionState, user_query: str) -> Dict[str, Any]:
        """Continue with revision content using Type 1 prompt"""
        
        # Get next content chunk
        topic_content = self.mongodb.get_topic_content(session_state.topic, limit=3)
        content_text = "\n".join([chunk["text"][:300] for chunk in topic_content])
        
        # Track concepts learned
        if topic_content:
            new_concept = self._extract_concept_name(topic_content[0]["text"])
            concepts_learned = getattr(session_state, 'concepts_learned', [])
            if new_concept not in concepts_learned:
                concepts_learned.append(new_concept)
                session_state.concepts_learned = concepts_learned
        
        # Generate response
        response = await self._generate_response_from_prompt(
            self.prompts.get_basic_revision_prompt(session_state.topic, content_text),
            "You are an expert educational tutor continuing the revision lesson."
        )
        
        return {
            "response": response,
            "current_stage": "revision",
            "is_session_complete": False,
            "sources": [chunk.get("chunk_id", "Unknown") for chunk in topic_content]
        }
    
    async def _evaluate_quiz_performance(self, user_answers: str, topic: str) -> float:
        """LLM-based quiz performance evaluation"""
        prompt = f"""
        Evaluate how well the student answered the quiz about "{topic}".
        
        Student's answers: {user_answers}
        
        Give a score from 0.0 to 1.0 where:
        - 0.0 = All wrong or no attempt
        - 0.3 = Poor understanding, mostly wrong
        - 0.5 = Some understanding, half correct
        - 0.7 = Good understanding, mostly correct  
        - 1.0 = Excellent, all correct
        
        Reply with just the number (e.g., 0.7)
        """
        
        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are a precise evaluator. Reply only with a number."),
                HumanMessage(content=prompt)
            ])
            
            # Extract number from response
            import re
            match = re.search(r'0\.\d+|1\.0|0|1', response)
            if match:
                return float(match.group())
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            logger.error(f"Performance evaluation error: {e}")
            return 0.5  # Default neutral score
    
    async def _generate_response_from_prompt(self, prompt: str, system_message: str) -> str:
        """Helper to generate response using LLM"""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        return await self.llm.generate_response(messages)
    
    def _extract_concept_name(self, text: str) -> str:
        """Extract concept name from text"""
        sentences = text.split('.')
        if sentences:
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    words = sentence.strip().split()[:4]
                    return " ".join(words)
        return text[:50] + "..." if len(text) > 50 else text
    
    # Session management methods (same as before)
    async def _get_or_restore_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session or restore from MongoDB"""
        if session_id in self.session_states:
            return self.session_states[session_id]
        
        session_data = self.mongodb.get_revision_session(session_id)
        if session_data:
            session_state = self._restore_session_state(session_data)
            self.session_states[session_id] = session_state
            return session_state
        
        return None
    
    def _restore_session_state(self, session_data: Dict[str, Any]) -> SessionState:
        """Restore session state from MongoDB data"""
        session_state = SessionState(
            session_id=session_data["session_id"],
            topic=session_data["topic"],
            student_id=session_data["student_id"],
            conversation_count=session_data.get("conversation_count", 0),
            started_at=session_data["started_at"],
            last_interaction=session_data.get("updated_at", datetime.now()),
            is_complete=session_data.get("is_complete", False),
            key_concepts_covered=session_data.get("concepts_covered", []),
            user_understanding_level="normal",
            max_conversations=session_data.get("max_conversations"),
            completion_threshold=session_data.get("completion_threshold")
        )
        
        # Restore dynamic tracking
        session_state.current_stage = session_data.get("stage", "revision")
        session_state.quiz_frequency = session_data.get("quiz_frequency", 5)
        session_state.concepts_learned = session_data.get("concepts_learned", [])
        session_state.quiz_scores = session_data.get("quiz_scores", [])
        session_state.needs_remedial = session_data.get("needs_remedial", False)
        
        return session_state
    
    def _save_initial_session(self, session_id: str, student_id: str, topic: str, 
                             response: str, topic_config: dict):
        """Save initial session data to MongoDB"""
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
            "conversation_history": [{
                "turn": 0,
                "type": "revision_start",
                "assistant_message": response,
                "timestamp": datetime.now()
            }]
        }
        self.mongodb.save_revision_session(session_data)
    
    async def _save_conversation_turn(self, session_state: SessionState, user_query: Optional[str], response_data: Dict[str, Any]):
        """Save conversation turn and update progress"""
        turn_data = {
            "turn": session_state.conversation_count,
            "user_message": user_query,
            "assistant_message": response_data["response"],
            "stage": response_data["current_stage"],
            "timestamp": datetime.now()
        }
        self.mongodb.save_conversation_turn(session_state.session_id, turn_data)
        
        # Update session progress
        progress_data = {
            "conversation_count": session_state.conversation_count,
            "current_stage": response_data["current_stage"],
            "concepts_learned": getattr(session_state, 'concepts_learned', []),
            "quiz_scores": getattr(session_state, 'quiz_scores', []),
            "needs_remedial": getattr(session_state, 'needs_remedial', False)
        }
        self.mongodb.update_session_progress(session_state.session_id, progress_data)
    
    async def _complete_session(self, session_state: SessionState) -> Dict[str, Any]:
        """Complete the revision session"""
        session_state.is_complete = True
        
        # Calculate dynamic statistics
        quiz_scores = getattr(session_state, 'quiz_scores', [])
        avg_performance = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0
        concepts_learned = len(getattr(session_state, 'concepts_learned', []))
        
        session_stats = {
            "total_interactions": session_state.conversation_count,
            "concepts_learned": concepts_learned,
            "quizzes_taken": len(quiz_scores),
            "average_performance": avg_performance,
            "duration_minutes": (datetime.now() - session_state.started_at).total_seconds() / 60
        }
        
        summary = f"Excellent work! 🎉 You completed your revision of {session_state.topic} with {session_state.conversation_count} interactions. You learned {concepts_learned} concepts and achieved {avg_performance:.1%} average quiz performance. Keep up the great learning! 🌟"
        
        # Update in MongoDB
        final_data = {
            "is_complete": True,
            "completed_at": datetime.now(),
            "final_stats": session_stats,
            "session_summary": summary
        }
        self.mongodb.update_session_progress(session_state.session_id, final_data)
        
        return {
            "response": summary,
            "topic": session_state.topic,
            "session_id": session_state.session_id,
            "conversation_count": session_state.conversation_count,
            "is_session_complete": True,
            "session_summary": summary,
            "session_stats": session_stats
        }