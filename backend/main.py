from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from contextlib import asynccontextmanager

from backend.config import Config
from backend.core.llm import GeminiLLMWrapper
from backend.core.mongodb_client import MongoDBClient
from backend.core.revision_agents import DynamicRevisionAgent  
from backend.api import revision


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


llm_wrapper = None
mongodb_client = None
revision_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup"""
    global llm_wrapper, mongodb_client, revision_agent
    
    try:
        # Validate configuration
        Config.validate_config()
        
        # Initialize components
        llm_wrapper = GeminiLLMWrapper()
        mongodb_client = MongoDBClient()
        revision_agent = DynamicRevisionAgent(llm_wrapper, mongodb_client)  
        
        # Set dependencies for routers
        revision.set_dependencies(revision_agent, mongodb_client)
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise e
    
    yield
    
    # Cleanup on shutdown
    if mongodb_client:
        mongodb_client.client.close()
    logger.info("Application shutting down")

# Create FastAPI app
app = FastAPI(
    title="Topic-Based Revision Chatbot API",
    description="Educational chatbot for progressive topic revision",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(revision.router, prefix="/api", tags=["revision"])

@app.get("/")
async def root():
    return {"message": "Topic-Based Revision Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )