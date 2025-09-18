import asyncio
from typing import TypedDict,List, Dict, Any, Optional
from .revision_agent import RevisionAgent
from backend.core.quiz_agent import QuizAgent
from backend.core.feedback_agent import FeedbackAgent
from backend.core.qa_agent import QAAgent
from backend.core.conclusion_agent import ConclusionAgent
from .mongodb_client import MongoDBClient
from backend.models.schemas import RevisionSessionData, SessionState
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# LangGraph import - we use it to express the orchestrator flow
from langgraph.graph import StateGraph

class OrchestratorState(TypedDict, total=False):
    user_message: Optional[str]
    assistant_message: Optional[str]
    stage: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    current_chunk_index: Optional[int]
    current_question_concept: Optional[str]
    current_expected_keywords: Optional[List[str]]
    expecting_answer: Optional[bool]

class OrchestratorAgent:
    """
    Main orchestrator that implements the flow using langgraph.
    It stores conversation turns and session progress in MongoDB (via MongoDBClient).
    """
    def __init__(self, mongodb: Optional[MongoDBClient] = None):
        self.rev_agent = RevisionAgent()
        self.quiz_agent = QuizAgent()
        self.feedback_agent = FeedbackAgent()
        self.qa_agent = QAAgent()
        self.conclusion_agent = ConclusionAgent()
        self.mongo = mongodb or MongoDBClient()

        # Build LangGraph flow
        self.graph = StateGraph(OrchestratorState)
        self._build_graph()

    def _build_graph(self):
        g = self.graph

        # Add nodes (no logic yet, just placeholders)
        g.add_node("EXPLAIN", lambda state: state)
        g.add_node("ASK_CHECK", lambda state: state)
        g.add_node("WAIT_ANSWER", lambda state: state)
        g.add_node("GIVE_FEEDBACK", lambda state: state)
        g.add_node("NEXT_CONCEPT", lambda state: state)

        # Add edges between them
        g.add_edge("EXPLAIN", "ASK_CHECK")
        g.add_edge("ASK_CHECK", "WAIT_ANSWER")
        g.add_edge("WAIT_ANSWER", "GIVE_FEEDBACK")
        g.add_edge("GIVE_FEEDBACK", "NEXT_CONCEPT")
        g.add_edge("NEXT_CONCEPT", "EXPLAIN")
        g.add_edge("NEXT_CONCEPT", "END")


    # utility to stringify conversation history for prompts (latest first)
    def _format_conversation_history(self, session_doc: Dict[str, Any], limit: int = 10) -> str:
        if not session_doc:
            return ""
        conv = session_doc.get("conversation_history", [])[-limit:]
        # reverse for latest-first display
        conv = list(reversed(conv))
        lines = []
        for i, turn in enumerate(conv):
            user = turn.get("user_message", "")
            assistant = turn.get("assistant_message", "")
            ts = turn.get("timestamp", "")
            lines.append(f"[{i}] user: {user} | assistant: {assistant}")
        return "\n".join(lines)

    async def start_revision_session(self, topic: str, student_id: str, session_id: str) -> Dict[str, Any]:
        """
        Create/initialize session, fetch topic content from MongoDB (subtopics), and return first explanation + check question.
        """
        # prepare or fetch session doc
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            # create a new revision session doc structure
            session_doc = {
                "session_id": session_id,
                "student_id": student_id,
                "topic": topic,
                "started_at": datetime.utcnow(),
                "conversation_count": 0,
                "is_complete": False,
                "max_conversations": 999,
                "completion_threshold": 0,
                "conversation_history": []
            }
            self.mongo.save_revision_session(session_doc)

        # fetch subtopics for the topic
        # topic in your db may be stored as 'topic_title' - adapt as your schema
        # Accept both "Chapter - Topic #: title" or plain title
        topic_title = topic.split(": ")[-1] if ": " in topic else topic
        subtopics = self.mongo.get_topic_subtopics(topic_title)
        # if not found, try get_topic_content fallback
        if not subtopics:
            subtopic_chunks = self.mongo.get_topic_content(topic)
            subtopics = [{"subtopic_number": c["id"], "subtopic_title": c.get("subtopic_title", ""), "content": c["text"]} for c in subtopic_chunks]

        # store concept chunks into session_doc
        session_doc["concept_chunks"] = subtopics
        session_doc["current_chunk_index"] = 0
        self.mongo.save_revision_session(session_doc)

        # start first concept
        return await self._present_current_concept(session_doc)

    async def _present_current_concept(self, session_doc: Dict[str, Any]) -> Dict[str, Any]:
        idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        if idx >= len(chunks):
            # Finished all concepts
            session_doc["is_complete"] = True
            self.mongo.save_revision_session(session_doc)
            summary = await self.conclusion_agent.summary(
                correct=len([p for p in session_doc.get("concepts_learned", []) if p]),
                total=len(chunks),
                conversation_history=self._format_conversation_history(session_doc)
            )
            return {"response": summary, "is_session_complete": True, "conversation_count": session_doc.get("conversation_count", 0)}
        current = chunks[idx]
        title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
        content = current.get("content", "")
        # generate explanation steps and check question
        conv_hist = self._format_conversation_history(session_doc)
        steps = await self.rev_agent.generate_explanation_steps(title, content, conversation_history=conv_hist, steps=3)
        check_q = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
        # extract expected keywords for this specific check question
        try:
            expected_keywords = await self.rev_agent.extract_expected_keywords(title, content, check_q)
        except Exception:
            expected_keywords = [w for w in (title.split()[:3])]
        # Append assistant message with explanation
        concept_message = "\n".join(steps)
        question_message = "Question: " + check_q

        messages = [
            {
                "assistant_message": concept_message,
                "message_type": "concept"
            },
            {
                "assistant_message": question_message, 
                "message_type": "question"
            }
        ]
        # Create separate turns for concept and question
        for i, message in enumerate(messages):
            turn = {
                "turn": session_doc.get("conversation_count", 0) + 1 + i,
                "user_message": None,
                "assistant_message": message["assistant_message"],
                "stage": "explain",
                "timestamp": datetime.utcnow(),
                "concept_covered": title,
                "question_asked": (message["message_type"] == "question"),
                "message_type": message["message_type"]
            }
            session_doc.setdefault("conversation_history", []).append(turn)

        # Update conversation count (add 2 since we have 2 messages)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + len(messages)
        # set expecting answer
        session_doc["expecting_answer"] = True
        session_doc["current_question_concept"] = title
        session_doc["current_expected_keywords"] = expected_keywords
        self.mongo.save_revision_session(session_doc)

        logger.info(f"Messages being returned: {messages}")

        result = {
            "response": messages,
            "message_format": "multiple_bubbles",
            "is_session_complete": False, 
            "conversation_count": session_doc["conversation_count"], 
            "current_stage": "explain", 
            "current_concept": title
        }

        logger.info(f"DEBUGGING - Full result: {result}")
        return result
        

    async def handle_user_input(self, session_id: str, user_query: str) -> Dict[str, Any]:
        session_doc = self.mongo.get_revision_session(session_id) or {}
        if not session_doc:
            return {"response":"Session not found. Start a new revision session.", "is_session_complete": True, "conversation_count": 0}

        # Get conversation history first
        conv_hist = self._format_conversation_history(session_doc)
        
        # Dynamic question detection using LLM
        current_concept = session_doc.get("current_question_concept", "")
        question_intent = await self.rev_agent.detect_question_intent(
            user_input=user_query,
            current_concept=current_concept,
            conversation_history=conv_hist
        )
        is_question = (question_intent == "ASKING_QUESTION")
        
        # Save user turn
        user_turn = {
            "turn": session_doc.get("conversation_count", 0) + 1,
            "user_message": user_query,
            "assistant_message": None,
            "stage": "user_input",
            "timestamp": datetime.utcnow(),
            "concept_covered": session_doc.get("current_question_concept")
        }
        session_doc.setdefault("conversation_history", []).append(user_turn)
        session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
        self.mongo.save_revision_session(session_doc)

        # Handle simple acknowledgements so they are not graded
        if question_intent == "ACKNOWLEDGEMENT":
            nudge = "Great! When you're ready, please answer the check question above."
            assistant_turn = {
                "turn": session_doc["conversation_count"],
                "user_message": user_query,
                "assistant_message": nudge,
                "stage": "ack",
                "timestamp": datetime.utcnow(),
            }
            session_doc.setdefault("conversation_history", []).append(assistant_turn)
            self.mongo.save_revision_session(session_doc)
            return {"response": nudge, "conversation_count": session_doc["conversation_count"], "is_session_complete": False, "current_stage": "ack"}

        if is_question:
            # Handle Q&A using revision agent with current concept context
            current_concept = session_doc.get("current_question_concept", "")
            current_chunk_idx = session_doc.get("current_chunk_index", 0)
            concept_chunks = session_doc.get("concept_chunks", [])
            current_content = ""
            
            if current_chunk_idx < len(concept_chunks):
                current_content = concept_chunks[current_chunk_idx].get("content", "")
            
            # 1) Answer the user's question
            answer = await self.rev_agent.handle_qa_request(
                user_question=user_query,
                current_concept=current_concept,
                content=current_content,
                conversation_history=conv_hist
            )

            # 2) Generate a formal check question and expected keywords for this concept
            check_q = await self.rev_agent.make_check_question(
                title=current_concept or (concept_chunks[current_chunk_idx].get("subtopic_title") if current_chunk_idx < len(concept_chunks) else ""),
                content=current_content,
                conversation_history=conv_hist
            )
            try:
                expected_keywords = await self.rev_agent.extract_expected_keywords(
                    current_concept or (concept_chunks[current_chunk_idx].get("subtopic_title") if current_chunk_idx < len(concept_chunks) else ""),
                    current_content,
                    check_q
                )
            except Exception:
                expected_keywords = [w for w in (current_concept.split()[:3])]

            # 3) Persist QA turn and updated state to expect an answer next
            combined = answer.strip() + "\n\nCheck question: " + check_q.strip()
            assistant_turn = {
                "turn": session_doc["conversation_count"],
                "user_message": user_query,
                "assistant_message": combined,
                "stage": "qa",
                "timestamp": datetime.utcnow(),
            }
            session_doc.setdefault("conversation_history", []).append(assistant_turn)
            session_doc["expecting_answer"] = True
            session_doc["current_question_concept"] = current_concept or (concept_chunks[current_chunk_idx].get("subtopic_title") if current_chunk_idx < len(concept_chunks) else "")
            session_doc["current_expected_keywords"] = expected_keywords
            self.mongo.save_revision_session(session_doc)

            return {
                "response": combined,
                "conversation_count": session_doc["conversation_count"],
                "is_session_complete": False,
                "current_stage": "qa"
            }

        # otherwise, treat it as answer to the check question
        expected_keywords = session_doc.get("current_expected_keywords", [])
        # If expected keywords appear to be default or empty, regenerate using LLM based on last check question
        if not expected_keywords or expected_keywords == [w for w in (session_doc.get("current_question_concept","" ).split()[:3])]:
            current_chunk_idx = session_doc.get("current_chunk_index", 0)
            chunks = session_doc.get("concept_chunks", [])
            title = session_doc.get("current_question_concept", "")
            content = ""
            question_text = ""
            if current_chunk_idx < len(chunks):
                content = chunks[current_chunk_idx].get("content", "")
            for h in reversed(session_doc.get("conversation_history", [])):
                if h.get("stage") == "explain" and h.get("assistant_message"):
                    msg = h["assistant_message"]
                    if "Check question:" in msg:
                        question_text = msg.split("Check question:",1)[1].strip()
                        break
            try:
                new_keywords = await self.rev_agent.extract_expected_keywords(title, content, question_text)
                if new_keywords:
                    expected_keywords = new_keywords
                    session_doc["current_expected_keywords"] = expected_keywords
                    self.mongo.save_revision_session(session_doc)
            except Exception:
                pass
        # Build full context for evaluation
        current_chunk_idx = session_doc.get("current_chunk_index", 0)
        chunks = session_doc.get("concept_chunks", [])
        title = session_doc.get("current_question_concept", "")
        content = ""
        if current_chunk_idx < len(chunks):
            content = chunks[current_chunk_idx].get("content", "")
        assistant_message = ""
        check_question = ""
        for h in reversed(session_doc.get("conversation_history", [])):
            if h.get("stage") == "explain" and h.get("assistant_message"):
                assistant_message = h["assistant_message"]
                if "Check question:" in assistant_message:
                    check_question = assistant_message.split("Check question:",1)[1].strip()
                break
        eval_result = await self.rev_agent.evaluate_answer(
            user_query,
            expected_keywords,
            conversation_history=conv_hist,
            title=title,
            content=content,
            assistant_message=assistant_message,
            check_question=check_question
        )
        verdict = eval_result.get("verdict", "WRONG")
        assistant_feedback = self.feedback_agent.feedback_for(verdict, {"correction": eval_result.get("correction", eval_result.get("justification", ""))})
        
        # save feedback as assistant message
        assistant_turn = {
            "turn": session_doc["conversation_count"],
            "user_message": user_query,
            "assistant_message": assistant_feedback,
            "stage": "feedback",
            "timestamp": datetime.utcnow(),
            "correct_answer": (verdict == "CORRECT")
        }
        session_doc.setdefault("conversation_history", []).append(assistant_turn)

        # store progress flags
        session_doc.setdefault("concepts_covered", []).append(session_doc.get("current_question_concept"))
        if verdict == "CORRECT":
            session_doc.setdefault("concepts_learned", []).append(session_doc.get("current_question_concept"))
            # Only move to next concept if answer is correct
            session_doc["current_chunk_index"] = session_doc.get("current_chunk_index", 0) + 1
            session_doc["expecting_answer"] = False
            session_doc["current_question_concept"] = None
            
            # prepare next concept or conclusion
            next_payload = await self._present_current_concept(session_doc)
            # combine feedback + next concept explanation in response to frontend
            messages = [
                {
                    "assistant_message": assistant_feedback,
                    "message_type": "feedback"
                }
            ]
            # Add next concept messages if available
            next_response = next_payload.get("response", [])
            if isinstance(next_response, list):
                messages.extend(next_response)
            elif next_response:
                messages.append({
                    "assistant_message": next_response,
                    "message_type": "concept"
                })
            combined_response = messages
            self.mongo.save_revision_session(session_doc)
            return {
                "response": combined_response,
                "message_format": "multiple_bubbles",
                "conversation_count": session_doc["conversation_count"],
                "is_session_complete": next_payload.get("is_session_complete", False),
                "current_stage": "feedback_and_next"
            }
        else:
            # for partial/wrong answers, stay on the same concept and re-present it
            session_doc.setdefault("needs_remedial", True)
            session_doc["expecting_answer"] = True  # Keep expecting an answer
            
            # Re-present the same concept with explanation
            current_chunk_idx = session_doc.get("current_chunk_index", 0)
            concept_chunks = session_doc.get("concept_chunks", [])
            if current_chunk_idx < len(concept_chunks):
                current = concept_chunks[current_chunk_idx]
                title = current.get("subtopic_title") or f"Concept {current.get('subtopic_number')}"
                content = current.get("content", "")
                
                # Generate new explanation and question for the same concept
                steps = await self.rev_agent.generate_explanation_steps(title, content, conversation_history=conv_hist, steps=3)
                check_q = await self.rev_agent.make_check_question(title, content, conversation_history=conv_hist)
                
                # Create new turn for re-explanation
                re_explanation = "\n".join(steps) + "\n\nQuestion: " + check_q
                re_explain_turn = {
                    "turn": session_doc.get("conversation_count", 0) + 1,
                    "user_message": None,
                    "assistant_message": re_explanation,
                    "stage": "explain",
                    "timestamp": datetime.utcnow(),
                    "concept_covered": title,
                    "question_asked": True,
                }
                session_doc.setdefault("conversation_history", []).append(re_explain_turn)
                session_doc["conversation_count"] = session_doc.get("conversation_count", 0) + 1
                session_doc["current_question_concept"] = title
                session_doc["current_expected_keywords"] = [w for w in (title.split()[:3])]
                
                # Combine feedback + re-explanation
                feedback_message = assistant_feedback
                re_explanation_message = "\n".join(steps) + "\n\nQuestion: " + check_q

                messages = [
                    {
                        "assistant_message": feedback_message,
                        "message_type": "feedback"
                    },
                    {
                        "assistant_message": re_explanation_message,
                        "message_type": "concept"
                    }
                ]
                self.mongo.save_revision_session(session_doc)
                return {
                    "response": messages,
                    "message_format": "multiple_bubbles",
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "feedback_and_retry"
                }
            else:
                # Fallback if no chunks available
                self.mongo.save_revision_session(session_doc)
                return {
                    "response": assistant_feedback,
                    "conversation_count": session_doc["conversation_count"],
                    "is_session_complete": False,
                    "current_stage": "feedback"
                }
