from pymongo import MongoClient
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from backend.config import Config

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client[Config.DATABASE_NAME]
        self.collection = self.db[Config.COLLECTION_NAME]
        self.revision_collection = self.db[Config.REVISION_COLLECTION]  # New collection
    
    def get_available_topics(self) -> List[Dict[str, Any]]:
        """Fetch all available topics from MongoDB"""
        try:
            # Get distinct topics
            topics = self.collection.distinct("topic")
            
            # Get topic details with counts
            topic_details = []
            for topic in topics:
                count = self.collection.count_documents({"topic": topic})
                
                # Get dynamic config for this topic
                topic_config = Config.get_topic_config(topic)
                
                topic_details.append({
                    "topic": topic,
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
        """Get content chunks for a specific topic"""
        try:
            cursor = self.collection.find(
                {"topic": topic},
                {"text": 1, "chunk_id": 1, "topic": 1, "_id": 0}
            ).limit(limit)
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error fetching topic content: {e}")
            return []
    
    def get_topic_content_chunks(self, topic: str) -> List[Dict[str, Any]]:
        """Get all content chunks for progressive learning"""
        try:
            cursor = self.collection.find(
                {"topic": topic},
                {"text": 1, "chunk_id": 1, "topic": 1, "_id": 0}
            )
            
            chunks = list(cursor)
            # Split large texts into smaller concept chunks if needed
            concept_chunks = []
            
            for chunk in chunks:
                text = chunk["text"]
                # Simple splitting by paragraphs or sentences for concept chunks
                if len(text) > 500:  # If chunk is too large
                    sentences = text.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 400:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                concept_chunks.append({
                                    "text": current_chunk.strip(),
                                    "chunk_id": f"{chunk['chunk_id']}_part_{len(concept_chunks) + 1}",
                                    "topic": chunk["topic"]
                                })
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        concept_chunks.append({
                            "text": current_chunk.strip(),
                            "chunk_id": f"{chunk['chunk_id']}_part_{len(concept_chunks) + 1}",
                            "topic": chunk["topic"]
                        })
                else:
                    concept_chunks.append(chunk)
            
            return concept_chunks
        except Exception as e:
            logger.error(f"Error fetching topic content chunks: {e}")
            return []
    
    def search_topic_content(self, topic: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search within topic content using text search"""
        try:
            # Simple text search within topic
            cursor = self.collection.find(
                {
                    "topic": topic,
                    "$text": {"$search": query}
                },
                {"text": 1, "chunk_id": 1, "topic": 1, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            results = list(cursor)
            
            # If no text search results, fall back to regex search
            if not results:
                cursor = self.collection.find(
                    {
                        "topic": topic,
                        "text": {"$regex": query, "$options": "i"}
                    },
                    {"text": 1, "chunk_id": 1, "topic": 1}
                ).limit(limit)
                results = list(cursor)
            
            return results
        except Exception as e:
            logger.error(f"Error searching topic content: {e}")
            return []
    
    # =============== REVISION SESSION METHODS ===============
    
    def save_revision_session(self, session_data: Dict[str, Any]) -> bool:
        """Save or update revision session in MongoDB"""
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
        """Get revision session by session_id"""
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
        """Get revision history for a student"""
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
        """Get statistics for topic revisions"""
        try:
            total_sessions = self.revision_collection.count_documents({"topic": topic})
            completed_sessions = self.revision_collection.count_documents({
                "topic": topic, 
                "is_complete": True
            })
            
            # Get average interactions for completed sessions
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
        """Save a conversation turn to the session"""
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
        """Update session progress"""
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