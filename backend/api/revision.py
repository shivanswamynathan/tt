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
    """Get all available topics for revision"""
    try:
        topics = mongodb_client.get_available_topics()
        return TopicResponse(topics=topics)
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch topics")

@router.post("/revision/start", response_model=RevisionResponse)
async def start_revision_session(request: RevisionRequest):
    """Start a new revision session for a topic"""
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
    """Continue an existing revision session"""
    try:
        result = await revision_agent.continue_revision(
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
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            user_message = await websocket.receive_text()
            logger.info(f"Received message: '{user_message}' for session: {session_id}")  # ADD THIS
            
            # Process with full context using existing revision agent
            logger.info(f"Processing with revision_agent...")  # ADD THIS
            result = await revision_agent.continue_revision(
                session_id=session_id,
                user_query=user_message
            )
            logger.info(f"Got result: {type(result)} - {result.get('response', 'No response')[:50]}...")  # ADD THIS
            
            # Send response back
            response_data = {
                "type": "message",
                "content": result["response"],
                "conversation_count": result.get("conversation_count", 0),    
                "is_session_complete": result.get("is_session_complete", False), 
                "current_stage": result.get("current_stage", "revision"),    
                "sources": result.get("sources", [])                         
            }
            
            await websocket.send_text(json.dumps(response_data))
            
            # Send completion notification if session is done
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