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
    """
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]  # Updated collection name
        self.revision_collection = self.db[Config.REVISION_COLLECTION]  
        self._ensure_text_index()
    
    def _ensure_text_index(self):
        """Ensure text index exists for search functionality"""
        try:
            # Check if text index exists
            indexes = list(self.collection.list_indexes())
            has_text_index = any("text" in str(index.get("key", {})) for index in indexes)
            
            if not has_text_index:
                self.collection.create_index([("content", "text")])
                logger.info("Created text index on 'content' field")
        except Exception as e:
            logger.warning(f"Could not create text index: {e}")
    
    def get_available_topics(self) -> List[Dict[str, Any]]:
        """
        Fetch all available topics with metadata from MongoDB.
        
        Returns:
            List[Dict[str, Any]]: List of topic dictionaries containing:
                - topic: Full topic name for searches
                - display_name: Shortened name for UI display
                - chunk_count: Number of subtopics available
                - description: Human-readable topic description
                - max_conversations: Maximum allowed interactions
                - completion_threshold: Score needed to complete topic
        """
        try:
            # Get distinct chapters and topics
            pipeline = [
                {"$group": {
                    "_id": {
                        "chapter": "$chapter",
                        "topic_title": "$topic_title"
                    },
                    "subtopic_count": {"$sum": {"$size": "$subtopics"}},
                    "topic_number": {"$first": "$topic_number"}
                }}
            ]
            
            topics_data = list(self.collection.aggregate(pipeline))
            
            topic_details = []
            for topic_data in topics_data:
                chapter = topic_data["_id"]["chapter"]
                topic_title = topic_data["_id"]["topic_title"]
                subtopic_count = topic_data["subtopic_count"]
                topic_number = topic_data["topic_number"]
                
                # Create full topic identifier
                full_topic = f"{chapter} - Topic {topic_number}: {topic_title}"
                
                topic_config = Config.get_topic_config(full_topic)
                
                topic_details.append({
                    "topic": full_topic,  
                    "display_name": topic_title,  
                    "chunk_count": subtopic_count,
                    "description": f"Chapter: {chapter} | {subtopic_count} subtopics",
                    "max_conversations": topic_config["max_conversations"],
                    "completion_threshold": topic_config["completion_threshold"],
                    "chapter": chapter,
                    "topic_number": topic_number,
                    "topic_title": topic_title
                })
            
            return topic_details
        except Exception as e:
            logger.error(f"Error fetching topics: {e}")
            return []
    
    def get_topic_subtopics(self, topic_title: str) -> List[Dict[str, Any]]:
        """
        Get all subtopics for a specific topic.
        
        Args:
            topic_title (str): The topic title to search for
            
        Returns:
            List[Dict[str, Any]]: List of subtopics with content
        """
        try:
            # Find the document with matching topic_title
            document = self.collection.find_one({"topic_title": topic_title})
            
            if document and "subtopics" in document:
                return document["subtopics"]
            
            return []
        except Exception as e:
            logger.error(f"Error fetching subtopics for {topic_title}: {e}")
            return []
    
    def get_subtopic_content(self, topic_title: str, subtopic_number: str) -> Dict[str, Any]:
        """
        Get specific subtopic content.
        
        Args:
            topic_title (str): The topic title
            subtopic_number (str): Subtopic number (e.g., "3.1")
            
        Returns:
            Dict[str, Any]: Subtopic data or empty dict if not found
        """
        try:
            document = self.collection.find_one({"topic_title": topic_title})
            
            if document and "subtopics" in document:
                for subtopic in document["subtopics"]:
                    if subtopic["subtopic_number"] == subtopic_number:
                        return subtopic
            
            return {}
        except Exception as e:
            logger.error(f"Error fetching subtopic {subtopic_number} for {topic_title}: {e}")
            return {}

    def get_topic_content(self, topic: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get content chunks for a specific topic (backward compatibility).
        
        Args:
            topic (str): Topic name to search for
            limit (int): Maximum number of content chunks to return
            
        Returns:
            List[Dict[str, Any]]: List of content chunks
        """
        try:
            # Extract topic_title from full topic string
            topic_title = topic.split(": ")[-1] if ": " in topic else topic
            
            subtopics = self.get_topic_subtopics(topic_title)
            
            # Convert subtopics to old format for compatibility
            content_chunks = []
            for subtopic in subtopics[:limit]:
                content_chunks.append({
                    "text": subtopic["content"],
                    "id": subtopic["subtopic_number"],
                    "topic": topic,
                    "subtopic_title": subtopic["subtopic_title"]
                })
            
            return content_chunks
        except Exception as e:
            logger.error(f"Error fetching topic content: {e}")
            return []
    
    def save_revision_session(self, session_data: Dict[str, Any]) -> bool:
        """Save or update revision session in MongoDB."""
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
        """Get revision session by session_id."""
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
        """Get revision history for a student."""
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
        """Get statistics for topic revisions."""
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
        """Save a conversation turn to the session."""
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
        """Update session progress."""
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