from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class TopicResponse(BaseModel):
    topics: List[Dict[str, Any]]

class RevisionRequest(BaseModel):
    topic: str
    query: Optional[str] = None
    session_id: str
    student_id: str
    conversation_count: int = 0

class RevisionResponse(BaseModel):
    response: str
    topic: str
    session_id: str
    conversation_count: int
    is_session_complete: bool
    session_summary: Optional[str] = None
    next_suggested_action: Optional[str] = None
    sources: List[str] = []
    current_stage: Optional[str] = None
    timestamp: datetime
    progress_percentage: Optional[float] = None
    max_conversations: Optional[int] = None
    completion_threshold: Optional[int] = None
    session_stats: Optional[Dict[str, Any]] = None

class SessionState(BaseModel):
    session_id: str
    topic: str
    student_id: str
    conversation_count: int
    started_at: datetime
    last_interaction: datetime
    is_complete: bool = False
    key_concepts_covered: List[str] = []
    user_understanding_level: str = "beginner"
    max_conversations: Optional[int] = None
    completion_threshold: Optional[int] = None
    

    current_chunk_index: int = 0
    revision_mode: Optional[str] = None  
    concept_chunks: List[Dict[str, Any]] = []
    expecting_answer: bool = False
    current_question_concept: Optional[str] = None
    quiz_in_progress: bool = False
    quiz_concepts: List[str] = []
    

    current_stage: str = "revision"
    quiz_frequency: int = 5
    concepts_learned: List[str] = []
    quiz_scores: List[float] = []
    needs_remedial: bool = False
    current_quiz_concepts: List[str] = []
    performance_history: List[float] = []
    
    class Config:
        arbitrary_types_allowed = True

class ConversationTurn(BaseModel):
    turn: int
    user_message: Optional[str] = None
    assistant_message: str
    stage: str
    timestamp: datetime
    concept_covered: Optional[str] = None
    question_asked: bool = False
    correct_answer: Optional[bool] = None

class RevisionSessionData(BaseModel):
    """Complete revision session data for MongoDB storage"""
    session_id: str
    student_id: str
    topic: str
    started_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    conversation_count: int = 0
    is_complete: bool = False
    

    max_conversations: int
    completion_threshold: int
    

    stage: str = "kickoff"
    concept_chunks_total: int = 0
    current_chunk_index: int = 0
    concepts_covered: List[str] = []
    revision_mode: Optional[str] = None
    

    quiz_frequency: int = 5
    concepts_learned: List[str] = []
    quiz_scores: List[float] = []
    needs_remedial: bool = False
    

    conversation_history: List[ConversationTurn] = []
    

    session_summary: Optional[str] = None
    final_stats: Optional[Dict[str, Any]] = None

class TopicStats(BaseModel):
    """Statistics for a specific topic"""
    topic: str
    total_sessions: int
    completed_sessions: int
    completion_rate: float
    average_interactions: float
    average_duration_minutes: Optional[float] = None

class StudentProgress(BaseModel):
    """Student's overall progress tracking"""
    student_id: str
    total_sessions: int
    completed_sessions: int
    topics_studied: List[str]
    total_interactions: int
    average_session_length: float
    last_session_date: Optional[datetime] = None
    current_streak: int = 0