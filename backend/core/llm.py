from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage
from typing import List, Optional
import logging
from backend.config import Config  # Import config directly

logger = logging.getLogger(__name__)

class GeminiLLMWrapper:
    def __init__(self):
        """Initialize with config values directly"""
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=Config.GEMINI_API_KEY,
            model=Config.GEMINI_MODEL,
            temperature=0.3,
            max_output_tokens=2048,
        )
    
    async def generate_response(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        try:
            response = await self.llm.ainvoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def generate_response_sync(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now."