from pymongo import MongoClient
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from backend.config import Config

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    MongoDB client for handling revision system data operations.
    
    Manages connections and operations for both content data and revision sessions,
    including topic management, session persistence, and student progress tracking.
    
    Example:
        client = MongoDBClient()
        topics = client.get_available_topics()
        session = client.get_revision_session("session_123")
    """
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        self.revision_collection = self.db[Config.REVISION_COLLECTION]  
        self._ensure_text_index()
    
    def _ensure_text_index(self):
        """Ensure text index exists for search functionality"""
        try:
            # Check if text index exists
            indexes = list(self.collection.list_indexes())
            has_text_index = any("text" in str(index.get("key", {})) for index in indexes)
            
            if not has_text_index:
                self.collection.create_index([("text", "text")])
                logger.info("Created text index on 'text' field")
        except Exception as e:
            logger.warning(f"Could not create text index: {e}")
    
    def get_available_topics(self) -> List[Dict[str, Any]]:
        """
        Fetch all available topics with metadata from MongoDB.
        
        Retrieves distinct topics from the content collection and enriches them
        with chunk counts, display names, and configuration details like max
        conversations and completion thresholds for the revision system.
        
        Returns:
            List[Dict[str, Any]]: List of topic dictionaries containing:
                - topic: Full topic name for searches
                - display_name: Shortened name for UI display
                - chunk_count: Number of content sections available
                - description: Human-readable topic description
                - max_conversations: Maximum allowed interactions
                - completion_threshold: Score needed to complete topic
                
        Returns:
            List: Empty list if database error occurs
        """
        try:
            # Get distinct topics
            topics = self.collection.distinct("topic")
            
            # Get topic details with counts
            topic_details = []
            for topic in topics:
                count = self.collection.count_documents({"topic": topic})
                
                # Create shorter display name
                short_name = topic.split(':')[0] if ':' in topic else topic
                short_name = short_name.replace("Exploring the Role of", "").strip()
                
                topic_config = Config.get_topic_config(topic)
                
                topic_details.append({
                    "topic": topic,  # Keep full topic for searches
                    "display_name": short_name,  # Shorter name for display
                    "chunk_count": count,
                    "description": f"Study material with {count} content sections",
                    "max_conversations": topic_config["max_conversations"],
                    "completion_threshold": topic_config["completion_threshold"]
                })
            
            return topic_details
        except Exception as e:
            logger.error(f"Error fetching topics: {e}")
            return []
    
    def get_topic_content(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get content chunks for a specific topic.
        
        Retrieves text content chunks that match the given topic using case-insensitive
        regex matching. Returns limited number of chunks for performance optimization
        during revision sessions and content delivery.
        
        Args:
            topic (str): Topic name to search for (supports partial matching)
            limit (int): Maximum number of content chunks to return (default: 10)
            
        Returns:
            List[Dict[str, Any]]: List of content chunks containing:
                - text: The actual content text
                - id: Unique chunk identifier
                - topic: Full topic name
                
        Returns:
            List: Empty list if topic not found or database error occurs
        """
        try:
            cursor = self.collection.find(
                {"topic": {"$regex": topic, "$options": "i"}},
                {"text": 1, "id": 1, "topic": 1, "_id": 0}
            ).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching topic content: {e}")
            return []
    
    def save_revision_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save or update revision session in MongoDB.
        
        Persists complete session data including conversation history, progress,
        and metadata. Uses upsert operation to either create new session or
        update existing one based on session_id. Automatically adds timestamp.
        
        Args:
            session_data (Dict[str, Any]): Complete session data including:
                - session_id: Unique session identifier
                - student_id: Student identifier
                - topic: Topic being studied
                - conversation_history: List of interactions
                - is_complete: Session completion status
                
        Returns:
            bool: True if save successful, False if database error occurs
        """
        try:
            session_data["updated_at"] = datetime.now()
            
            result = self.revision_collection.update_one(
                {"session_id": session_data["session_id"]},
                {"$set": session_data},
                upsert=True
            )
            
            logger.info(f"Saved revision session: {session_data['session_id']}")
            return True
        except Exception as e:
            logger.error(f"Error saving revision session: {e}")
            return False
    
    def get_revision_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get revision session by session_id.
        
        Retrieves complete session data for continuing existing revision sessions.
        Excludes MongoDB's internal _id field from results for cleaner data handling.
        Essential for maintaining conversation context across interactions.
        
        Args:
            session_id (str): Unique session identifier
            
        Returns:
            Optional[Dict[str, Any]]: Session data if found, None if not found or error
                Contains all session fields like conversation_history, progress, etc.
        """
        try:
            session = self.revision_collection.find_one(
                {"session_id": session_id},
                {"_id": 0}
            )
            return session
        except Exception as e:
            logger.error(f"Error fetching revision session: {e}")
            return None
    
    def get_student_revision_history(self, student_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get revision history for a student.
        
        Retrieves student's past revision sessions sorted by most recent first.
        Useful for tracking learning progress, identifying patterns, and providing
        personalized recommendations based on revision history.
        
        Args:
            student_id (str): Student identifier
            limit (int): Maximum number of sessions to return (default: 20)
            
        Returns:
            List[Dict[str, Any]]: List of session data sorted by started_at (newest first)
                Each session contains topic, completion status, interaction count, etc.
                
        Returns:
            List: Empty list if no history found or database error occurs
        """
        try:
            cursor = self.revision_collection.find(
                {"student_id": student_id},
                {"_id": 0}
            ).sort("started_at", -1).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching student revision history: {e}")
            return []
    
    def get_topic_revision_stats(self, topic: str) -> Dict[str, Any]:
        """ Get statistics for topic revisions.
        
        Calculates comprehensive analytics for a topic including total sessions,
        completion rates, and average interaction counts. Used for topic difficulty
        assessment and system optimization insights.
        """
        try:
            total_sessions = self.revision_collection.count_documents({"topic": topic})
            completed_sessions = self.revision_collection.count_documents({
                "topic": topic, 
                "is_complete": True
            })
            
            pipeline = [
                {"$match": {"topic": topic, "is_complete": True}},
                {"$group": {"_id": None, "avg_interactions": {"$avg": "$conversation_count"}}}
            ]
            
            avg_result = list(self.revision_collection.aggregate(pipeline))
            avg_interactions = avg_result[0]["avg_interactions"] if avg_result else 0
            
            return {
                "topic": topic,
                "total_sessions": total_sessions,
                "completed_sessions": completed_sessions,
                "completion_rate": (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0,
                "average_interactions": round(avg_interactions, 1)
            }
        except Exception as e:
            logger.error(f"Error fetching topic revision stats: {e}")
            return {}
    
    def save_conversation_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """Save a conversation turn to the session.
        
        Appends a single interaction (user query + AI response) to the session's
        conversation history. Maintains chronological order of all interactions
        and updates session timestamp for accurate tracking.
        """
        try:
            result = self.revision_collection.update_one(
                {"session_id": session_id},
                {
                    "$push": {"conversation_history": turn_data},
                    "$set": {"updated_at": datetime.now()}
                }
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error saving conversation turn: {e}")
            return False
    
    def update_session_progress(self, session_id: str, progress_data: Dict[str, Any]) -> bool:
        """
        Update session progress.
        
        Updates session-level progress indicators like conversation count,
        completion status, current learning stage, or performance scores.
        Automatically updates timestamp to track when progress was last modified.
        
        Args:
            session_id (str): Session to update
            progress_data (Dict[str, Any]): Progress updates such as:
                - conversation_count: Number of interactions
                - is_complete: Whether session finished
                - current_stage: Learning phase (revision, assessment, etc.)
                - performance_score: Student's current score
                
        Returns:
            bool: True if update successful, False if session not found or error
        """
        try:
            progress_data["updated_at"] = datetime.now()
            
            result = self.revision_collection.update_one(
                {"session_id": session_id},
                {"$set": progress_data}
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating session progress: {e}")
            return False