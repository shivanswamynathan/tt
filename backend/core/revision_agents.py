from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from typing import List, Dict, Any, Optional, Annotated, TypedDict
import logging
from backend.core.llm import GeminiLLMWrapper
from backend.core.mongodb_client import MongoDBClient
from backend.prompts.revision_prompts import RevisionPrompts
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# State Definition for LangGraph
class RevisionState(TypedDict):
    """State for the revision learning flow"""
    # Session info
    session_id: str
    student_id: str
    topic: str
    
    # Current position
    current_subtopic_index: int
    current_message_step: int
    
    # Content data
    subtopics: List[Dict[str, Any]]
    
    # Learning progress
    concepts_learned: List[str]
    concept_scores: List[float]
    total_interactions: int
    
    # Current interaction
    user_input: Optional[str]
    last_ai_response: Optional[str]
    waiting_for_answer: bool
    current_question: Optional[str]
    
    # Session state
    session_complete: bool
    stage: str  # 'intro', 'learning', 'question', 'feedback', 'complete'
    
    # Messages for context
    messages: Annotated[List, add_messages]

class LangGraphRevisionAgent:
    """
    Enhanced revision agent using LangGraph for structured learning flow.
    
    Flow:
    1. Topic introduction
    2. For each concept:
       - Explain step by step (multiple short messages)
       - Ask check question
       - Wait for answer
       - Give feedback
       - Move to next concept
    3. Final summary
    """

    def __init__(self, llm_wrapper: GeminiLLMWrapper, mongodb_client: MongoDBClient):
        self.llm = llm_wrapper
        self.mongodb = mongodb_client
        self.prompts = RevisionPrompts()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        workflow = StateGraph(RevisionState)
        
        # Add nodes
        workflow.add_node("initialize_session", self._initialize_session)
        workflow.add_node("topic_introduction", self._topic_introduction)
        workflow.add_node("start_concept_explanation", self._start_concept_explanation)
        workflow.add_node("continue_explanation", self._continue_explanation)
        workflow.add_node("ask_check_question", self._ask_check_question)
        workflow.add_node("evaluate_answer", self._evaluate_answer)
        workflow.add_node("give_feedback", self._give_feedback)
        workflow.add_node("move_to_next_concept", self._move_to_next_concept)
        workflow.add_node("handle_user_question", self._handle_user_question)
        workflow.add_node("session_summary", self._session_summary)
        
        # Define edges
        workflow.add_edge(START, "initialize_session")
        workflow.add_edge("initialize_session", "topic_introduction")
        
        # Conditional routing from topic introduction
        workflow.add_conditional_edges(
            "topic_introduction",
            self._route_after_intro,
            {
                "start_concept": "start_concept_explanation",
                "complete": "session_summary"
            }
        )
        
        # Concept explanation flow
        workflow.add_conditional_edges(
            "start_concept_explanation",
            self._route_explanation,
            {
                "continue": "continue_explanation",
                "question": "ask_check_question"
            }
        )
        
        workflow.add_conditional_edges(
            "continue_explanation",
            self._route_explanation,
            {
                "continue": "continue_explanation",
                "question": "ask_check_question"
            }
        )
        
        # Question and feedback flow
        workflow.add_edge("ask_check_question", "evaluate_answer")
        workflow.add_edge("evaluate_answer", "give_feedback")
        
        workflow.add_conditional_edges(
            "give_feedback",
            self._route_after_feedback,
            {
                "next_concept": "move_to_next_concept",
                "complete": "session_summary"
            }
        )
        
        workflow.add_conditional_edges(
            "move_to_next_concept",
            self._route_next_concept,
            {
                "start_concept": "start_concept_explanation",
                "complete": "session_summary"
            }
        )
        
        # User question handling
        workflow.add_conditional_edges(
            "handle_user_question",
            self._route_after_user_question,
            {
                "continue_explanation": "continue_explanation",
                "ask_question": "ask_check_question",
                "next_concept": "move_to_next_concept"
            }
        )
        
        workflow.add_edge("session_summary", END)
        
        return workflow.compile()

    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> dict:
        """Start a new revision session"""
        
        # Get subtopics
        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        
        if not subtopics:
            return {
                "response": "Sorry, no content found for this topic.",
                "is_session_complete": True,
                "current_stage": "error"
            }
        
        # Initialize state
        initial_state = RevisionState(
            session_id=session_id,
            student_id=student_id,
            topic=topic,
            current_subtopic_index=0,
            current_message_step=0,
            subtopics=subtopics,
            concepts_learned=[],
            concept_scores=[],
            total_interactions=0,
            user_input=None,
            last_ai_response=None,
            waiting_for_answer=False,
            current_question=None,
            session_complete=False,
            stage="intro",
            messages=[]
        )
        
        # Run the graph
        result = await self.graph.ainvoke(initial_state)
        
        # Save to database
        self._save_session_state(result)
        
        return {
            "response": result["last_ai_response"],
            "topic": topic,
            "session_id": session_id,
            "conversation_count": result["total_interactions"],
            "is_session_complete": result["session_complete"],
            "current_stage": result["stage"],
            "sources": [f"3.{result['current_subtopic_index'] + 1}"] if result["subtopics"] else []
        }

    async def handle_user_input(self, session_id: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """Handle user input and continue the session"""
        
        # Load session state
        session_data = self.mongodb.get_revision_session(session_id)
        if not session_data:
            return {"response": "Session not found.", "is_session_complete": False}
        
        # Convert to state format
        state = self._load_session_state(session_data, user_query)
        
        # Determine entry point based on current state
        if self._is_user_asking_question(user_query, state):
            # User asked a question - handle it
            state["stage"] = "user_question"
            result = await self._handle_user_question(state)
        elif state["waiting_for_answer"] and state["current_question"]:
            # User is answering a check question
            state["user_input"] = user_query
            result = await self._evaluate_answer(state)
            result = await self._give_feedback(result)
            
            # Continue flow
            if result["current_subtopic_index"] < len(result["subtopics"]) - 1:
                result = await self._move_to_next_concept(result)
                if not result["session_complete"]:
                    result = await self._start_concept_explanation(result)
            else:
                result["session_complete"] = True
                result = await self._session_summary(result)
        else:
            # Continue current explanation or start next step
            if state["stage"] == "learning":
                result = await self._continue_explanation(state)
                if result["stage"] == "question":
                    result = await self._ask_check_question(result)
            else:
                # Default to continuing explanation
                result = await self._continue_explanation(state)
        
        # Save updated state
        self._save_session_state(result)
        
        return {
            "response": result["last_ai_response"],
            "topic": result["topic"],
            "session_id": session_id,
            "conversation_count": result["total_interactions"],
            "is_session_complete": result["session_complete"],
            "current_stage": result["stage"],
            "sources": [f"3.{result['current_subtopic_index'] + 1}"] if result["subtopics"] else []
        }

    # Graph Node Functions
    async def _initialize_session(self, state: RevisionState) -> RevisionState:
        """Initialize the session"""
        state["total_interactions"] = 0
        state["stage"] = "intro"
        return state

    async def _topic_introduction(self, state: RevisionState) -> RevisionState:
        """Give topic introduction"""
        subtopics_count = len(state["subtopics"])
        
        intro_prompt = self.prompts.get_topic_introduction_prompt(
            state["topic"], 
            subtopics_count
        )
        
        response = await self._generate_response(intro_prompt, "Keep it brief and engaging (2-3 lines).")
        
        state["last_ai_response"] = response
        state["total_interactions"] += 1
        state["messages"].append(AIMessage(content=response))
        
        return state

    async def _start_concept_explanation(self, state: RevisionState) -> RevisionState:
        """Start explaining current concept"""
        if state["current_subtopic_index"] >= len(state["subtopics"]):
            state["session_complete"] = True
            return state
        
        current_subtopic = state["subtopics"][state["current_subtopic_index"]]
        state["current_message_step"] = 1
        state["stage"] = "learning"
        
        # Generate first explanation message
        explanation_prompt = self.prompts.get_step_by_step_explanation_prompt(
            current_subtopic["subtopic_title"],
            current_subtopic["content"],
            step=1,
            total_steps=3  # We'll do 3 explanation steps
        )
        
        response = await self._generate_response(
            explanation_prompt, 
            "Explain this step in 1-2 simple sentences. Be clear and engaging."
        )
        
        state["last_ai_response"] = response
        state["total_interactions"] += 1
        state["messages"].append(AIMessage(content=response))
        
        return state

    async def _continue_explanation(self, state: RevisionState) -> RevisionState:
        """Continue step-by-step explanation"""
        if state["current_subtopic_index"] >= len(state["subtopics"]):
            state["session_complete"] = True
            return state
        
        current_subtopic = state["subtopics"][state["current_subtopic_index"]]
        state["current_message_step"] += 1
        
        if state["current_message_step"] <= 3:  # 3 explanation steps
            explanation_prompt = self.prompts.get_step_by_step_explanation_prompt(
                current_subtopic["subtopic_title"],
                current_subtopic["content"],
                step=state["current_message_step"],
                total_steps=3
            )
            
            response = await self._generate_response(
                explanation_prompt,
                "Continue the explanation in 1-2 simple sentences."
            )
            
            state["last_ai_response"] = response
            state["total_interactions"] += 1
            state["messages"].append(AIMessage(content=response))
        else:
            # Move to question phase
            state["stage"] = "question"
        
        return state

    async def _ask_check_question(self, state: RevisionState) -> RevisionState:
        """Ask a simple check question"""
        if state["current_subtopic_index"] >= len(state["subtopics"]):
            state["session_complete"] = True
            return state
        
        current_subtopic = state["subtopics"][state["current_subtopic_index"]]
        
        question_prompt = self.prompts.get_simple_check_question_prompt(
            current_subtopic["subtopic_title"],
            current_subtopic["content"]
        )
        
        response = await self._generate_response(
            question_prompt,
            "Ask one simple question to check understanding. Keep it short."
        )
        
        state["last_ai_response"] = response
        state["current_question"] = response
        state["waiting_for_answer"] = True
        state["stage"] = "question"
        state["total_interactions"] += 1
        state["messages"].append(AIMessage(content=response))
        
        return state

    async def _evaluate_answer(self, state: RevisionState) -> RevisionState:
        """Evaluate user's answer"""
        if not state["user_input"] or not state["current_question"]:
            return state
        
        current_subtopic = state["subtopics"][state["current_subtopic_index"]]
        
        # Evaluate the answer
        score = await self._score_answer(
            state["user_input"],
            state["current_question"],
            current_subtopic["content"]
        )
        
        state["concept_scores"].append(score)
        state["waiting_for_answer"] = False
        state["stage"] = "feedback"
        
        # Add user message to context
        state["messages"].append(HumanMessage(content=state["user_input"]))
        
        return state

    async def _give_feedback(self, state: RevisionState) -> RevisionState:
        """Give feedback on the answer"""
        if not state["concept_scores"]:
            return state
        
        current_score = state["concept_scores"][-1]
        current_subtopic = state["subtopics"][state["current_subtopic_index"]]
        
        feedback_prompt = self.prompts.get_answer_feedback_prompt(
            state["user_input"],
            current_score >= 0.6,  # Pass threshold
            current_subtopic["subtopic_title"]
        )
        
        response = await self._generate_response(
            feedback_prompt,
            "Give brief, encouraging feedback (1-2 lines)."
        )
        
        state["last_ai_response"] = response
        state["total_interactions"] += 1
        state["messages"].append(AIMessage(content=response))
        
        # Mark concept as learned if passed
        if current_score >= 0.6:
            concept_name = current_subtopic["subtopic_title"]
            if concept_name not in state["concepts_learned"]:
                state["concepts_learned"].append(concept_name)
        
        return state

    async def _move_to_next_concept(self, state: RevisionState) -> RevisionState:
        """Move to the next concept"""
        state["current_subtopic_index"] += 1
        state["current_message_step"] = 0
        state["current_question"] = None
        state["user_input"] = None
        
        if state["current_subtopic_index"] >= len(state["subtopics"]):
            state["session_complete"] = True
            state["stage"] = "complete"
        else:
            state["stage"] = "learning"
        
        return state

    async def _handle_user_question(self, state: RevisionState) -> RevisionState:
        """Handle user's question"""
        if state["current_subtopic_index"] < len(state["subtopics"]):
            current_subtopic = state["subtopics"][state["current_subtopic_index"]]
            content_context = current_subtopic["content"][:300]
        else:
            content_context = "General topic content"
        
        question_response_prompt = self.prompts.get_user_question_response_prompt(
            state["user_input"],
            content_context
        )
        
        response = await self._generate_response(
            question_response_prompt,
            "Answer the question briefly and ask if they want to continue."
        )
        
        state["last_ai_response"] = response
        state["total_interactions"] += 1
        state["messages"].append(HumanMessage(content=state["user_input"]))
        state["messages"].append(AIMessage(content=response))
        
        # Reset to learning mode
        state["stage"] = "learning"
        
        return state

    async def _session_summary(self, state: RevisionState) -> RevisionState:
        """Provide session summary"""
        total_concepts = len(state["subtopics"])
        concepts_learned = len(state["concepts_learned"])
        avg_score = sum(state["concept_scores"]) / len(state["concept_scores"]) if state["concept_scores"] else 0
        
        summary_prompt = self.prompts.get_session_summary_prompt(
            total_concepts,
            concepts_learned,
            avg_score,
            state["concepts_learned"]
        )
        
        response = await self._generate_response(
            summary_prompt,
            "Give encouraging final summary (3-4 lines)."
        )
        
        state["last_ai_response"] = response
        state["session_complete"] = True
        state["stage"] = "complete"
        state["total_interactions"] += 1
        state["messages"].append(AIMessage(content=response))
        
        return state

    # Routing Functions
    def _route_after_intro(self, state: RevisionState) -> str:
        """Route after topic introduction"""
        if len(state["subtopics"]) == 0:
            return "complete"
        return "start_concept"

    def _route_explanation(self, state: RevisionState) -> str:
        """Route during explanation phase"""
        if state["current_message_step"] < 3:
            return "continue"
        return "question"

    def _route_after_feedback(self, state: RevisionState) -> str:
        """Route after giving feedback"""
        if state["current_subtopic_index"] >= len(state["subtopics"]) - 1:
            return "complete"
        return "next_concept"

    def _route_next_concept(self, state: RevisionState) -> str:
        """Route when moving to next concept"""
        if state["session_complete"]:
            return "complete"
        return "start_concept"

    def _route_after_user_question(self, state: RevisionState) -> str:
        """Route after handling user question"""
        if state["waiting_for_answer"]:
            return "ask_question"
        elif state["current_message_step"] < 3:
            return "continue_explanation"
        else:
            return "next_concept"

    # Helper Functions
    async def _generate_response(self, prompt: str, system_message: str) -> str:
        """Generate AI response"""
        try:
            response = await self.llm.generate_response([
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ])
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm having trouble responding right now. Let's continue."

    async def _score_answer(self, user_answer: str, question: str, content: str) -> float:
        """Score the user's answer"""
        scoring_prompt = f"""
        Question: {question}
        Student Answer: {user_answer}
        Content Context: {content[:200]}
        
        Score from 0.0 to 1.0 based on correctness and understanding.
        Reply with only the number.
        """
        
        try:
            response = await self.llm.generate_response([
                SystemMessage(content="You are scoring an answer. Reply only with a number between 0.0 and 1.0."),
                HumanMessage(content=scoring_prompt)
            ])
            
            import re
            match = re.search(r'0\.\d+|1\.0|0|1', response)
            return float(match.group()) if match else 0.5
        except:
            return 0.5

    def _is_user_asking_question(self, user_input: str, state: RevisionState) -> bool:
        """Check if user is asking a question rather than answering"""
        if not user_input:
            return False
        
        # If we're waiting for an answer and user gives a short response, it's probably an answer
        if state["waiting_for_answer"] and len(user_input.split()) <= 5:
            return False
        
        # Check for question indicators
        question_indicators = ["what", "how", "why", "when", "where", "can you", "could you", "explain", "?"]
        user_lower = user_input.lower()
        
        return any(indicator in user_lower for indicator in question_indicators)

    def _save_session_state(self, state: RevisionState):
        """Save session state to database"""
        session_data = {
            "session_id": state["session_id"],
            "student_id": state["student_id"],
            "topic": state["topic"],
            "current_subtopic_index": state["current_subtopic_index"],
            "current_message_step": state["current_message_step"],
            "concepts_learned": state["concepts_learned"],
            "concept_scores": state["concept_scores"],
            "total_interactions": state["total_interactions"],
            "waiting_for_answer": state["waiting_for_answer"],
            "current_question": state["current_question"],
            "session_complete": state["session_complete"],
            "stage": state["stage"],
            "updated_at": datetime.now()
        }
        
        self.mongodb.save_revision_session(session_data)

    def _load_session_state(self, session_data: Dict[str, Any], user_input: str) -> RevisionState:
        """Load session state from database"""
        # Get subtopics
        topic_title = session_data["topic"].split(": ")[-1] if ": " in session_data["topic"] else session_data["topic"]
        subtopics = self.mongodb.get_topic_subtopics(topic_title)
        
        return RevisionState(
            session_id=session_data["session_id"],
            student_id=session_data["student_id"],
            topic=session_data["topic"],
            current_subtopic_index=session_data.get("current_subtopic_index", 0),
            current_message_step=session_data.get("current_message_step", 0),
            subtopics=subtopics,
            concepts_learned=session_data.get("concepts_learned", []),
            concept_scores=session_data.get("concept_scores", []),
            total_interactions=session_data.get("total_interactions", 0),
            user_input=user_input,
            last_ai_response=session_data.get("last_ai_response", ""),
            waiting_for_answer=session_data.get("waiting_for_answer", False),
            current_question=session_data.get("current_question"),
            session_complete=session_data.get("session_complete", False),
            stage=session_data.get("stage", "learning"),
            messages=[]  # We could load this from conversation_history if needed
        )
