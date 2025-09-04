import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # Model Settings
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    # MongoDB Settings
    MONGODB_URI: str = "mongodb+srv://haswath1810:haswath18@cluster0.tkjt0ke.mongodb.net/?retryWrites=true&w=majority"
    DATABASE_NAME: str = "ncert_class8"
    COLLECTION_NAME: str = "test"
    REVISION_COLLECTION: str = "revision_sessions" 
    
    # Dynamic Defaults (calculated per topic)
    MIN_CONVERSATIONS: int = 8
    MAX_CONVERSATIONS: int = 50
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @classmethod
    def calculate_topic_limits(cls, content_chunks: int) -> dict:
        """Calculate dynamic limits based on topic content size"""
        
        # Base calculation on content chunks available
        base_conversations = max(cls.MIN_CONVERSATIONS, content_chunks * 2)
        max_conversations = min(cls.MAX_CONVERSATIONS, base_conversations)
        completion_threshold = max(6, int(max_conversations * 0.6))
        
        return {
            "max_conversations": max_conversations,
            "completion_threshold": completion_threshold,
            "quiz_frequency": max(3, int(max_conversations / 5))  # Quiz every N interactions
        }
    
    @classmethod
    def get_topic_config(cls, topic: str) -> Dict[str, int]:
        """Get configuration for a specific topic (backward compatibility)"""
        
        return {
            "max_conversations": cls.MAX_CONVERSATIONS,
            "completion_threshold": int(cls.MAX_CONVERSATIONS * 0.6)
        }
    

    @classmethod
    def validate_config(cls):
        if not cls.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        if not cls.MONGODB_URI:
            raise ValueError("MONGODB_URI is required")