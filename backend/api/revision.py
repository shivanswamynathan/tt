from fastapi import APIRouter, HTTPException
from backend.models.schemas import RevisionRequest, RevisionResponse, TopicResponse
from backend.core.revision_agents import DynamicRevisionAgent
from backend.core.mongodb_client import MongoDBClient
from datetime import datetime
import uuid
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependencies - would normally be injected
revision_agent: DynamicRevisionAgent = None
mongodb_client: MongoDBClient = None

def set_dependencies(ra: DynamicRevisionAgent, mc: MongoDBClient):
    global revision_agent, mongodb_client
    revision_agent = ra
    mongodb_client = mc

@router.get("/topics", response_model=TopicResponse)
async def get_available_topics():
    """
    Get all available topics for revision.
    
    Retrieves a list of all topics available for student revision sessions
    from the MongoDB database. Topics are used to start new revision sessions.
    
    Returns:
        TopicResponse: Object containing list of available topics
        
    Raises:
        HTTPException: 500 status code if database fetch fails
    """
    try:
        topics = mongodb_client.get_available_topics()
        return TopicResponse(topics=topics)
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch topics")

@router.post("/revision/start", response_model=RevisionResponse)
async def start_revision_session(request: RevisionRequest):
    """
    Start a new revision session for a specific topic.
    
    Initializes a new learning session with the revision agent for the given topic.
    Generates a unique session ID if not provided and begins the interactive revision
    process tailored to the student's learning needs.
    
    Args:
        request (RevisionRequest): Contains topic, student_id, and optional session_id
        
    Returns:
        RevisionResponse: Initial response with session details, conversation count,
                         completion status, and any relevant sources
                         
    Raises:
        HTTPException: 500 status code if session initialization fails
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        result = await revision_agent.start_revision_session(
            topic=request.topic,
            student_id=request.student_id,
            session_id=session_id
        )
        
        return RevisionResponse(
            response=result["response"],
            topic=request.topic,
            session_id=session_id,
            conversation_count=0,
            is_session_complete=result["is_session_complete"],
            session_summary=result.get("session_summary"),
            sources=result.get("sources", []),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error starting revision session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start revision session")

@router.post("/revision/continue", response_model=RevisionResponse)
async def continue_revision_session(request: RevisionRequest):
    """
    Continue an existing revision session with user input.
    
    Processes user queries within an ongoing revision session, maintaining
    conversation context and tracking learning progress. Updates conversation
    count and determines if the session should be marked as complete.
    
    Args:
        request (RevisionRequest): Contains session_id, user query, and topic
        
    Returns:
        RevisionResponse: AI response with updated session state, progress tracking,
                         completion status, and suggested next actions
                         
    Raises:
        HTTPException: 500 status code if query processing fails
    """
    try:
        result = await revision_agent.handle_user_input(
            session_id=request.session_id,
            user_query=request.query
        )
        
        return RevisionResponse(
            response=result["response"],
            topic=result.get("topic", request.topic),
            session_id=request.session_id,
            conversation_count=result["conversation_count"],
            is_session_complete=result["is_session_complete"],
            session_summary=result.get("session_summary"),
            next_suggested_action=result.get("next_suggested_action"),
            sources=result.get("sources", []),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error continuing revision session: {e}")
        raise HTTPException(status_code=500, detail="Failed to continue revision session")

@router.websocket("/ws/revision/{session_id}")
async def revision_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time revision session communication.
    
    Establishes a persistent WebSocket connection for interactive revision sessions.
    Enables real-time back-and-forth conversation between student and AI tutor,
    with live progress updates and session completion notifications.
    
    Args:
        websocket (WebSocket): WebSocket connection object
        session_id (str): Unique identifier for the revision session
        
    Connection Flow:
        - Accepts WebSocket connection
        - Receives user messages and processes with revision agent
        - Sends AI responses with session metadata
        - Notifies when session is complete with summary
        - Handles disconnection and errors gracefully
        
    Message Types Sent:
        - "message": Regular AI response with conversation data
        - "session_complete": Final summary when revision is finished
    """
    await websocket.accept()
    
    try:
        while True:
            
            user_message = await websocket.receive_text()
            logger.info(f"Received message: '{user_message}' for session: {session_id}")  
            
            
            logger.info(f"Processing with revision_agent...")  
            result = await revision_agent.handle_user_input(
                session_id=session_id,
                user_query=user_message
            )
            logger.info(f"Got result: {type(result)} - {result.get('response', 'No response')[:50]}...")  
            
            
            response_data = {
                "type": "message",
                "content": result["response"],
                "conversation_count": result.get("conversation_count", 0),    
                "is_session_complete": result.get("is_session_complete", False), 
                "current_stage": result.get("current_stage", "revision"),    
                "sources": result.get("sources", [])                         
            }
            
            await websocket.send_text(json.dumps(response_data))
            
            
            if result["is_session_complete"]:
                complete_data = {
                    "type": "session_complete",
                    "summary": result.get("session_summary")
                }
                await websocket.send_text(json.dumps(complete_data))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()