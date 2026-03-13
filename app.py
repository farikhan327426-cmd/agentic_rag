from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache, get_llm_cache
#from langchain_community.cache import RedisSemanticCache
from langchain_redis import RedisSemanticCache
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
import time as time_module
from src.agentic_self_rag.agentic_rag.graph import get_graph
from src.agentic_self_rag.core.logger import logger
from src.agentic_self_rag.ingestion.processor import DocumentProcessor
from src.agentic_self_rag.ingestion.embedder import DataIngestor
from src.agentic_self_rag.database.vector_store import VectorStore  # <-- NEW IMPORT ADDED HERE
import os
import json
import hashlib
import shutil
from src.agentic_self_rag.utils.llm_factory import ModelFactory
import tempfile

# Load environment variables
load_dotenv()

# Initialize API
app = FastAPI(
    title="Agentic Self-RAG API",
    description="Professional enterprise API for querying documents via Self-Reflective Agentic RAG.",
    version="1.0.0"
)
# Create a standard Redis client specifically for API-level string caching
api_redis_client = Redis.from_url(os.getenv("REDIS_URL", "redis://redis-service:6379"), decode_responses=True)

def get_api_cache_key(query: str) -> str:
    """Normalizes the question and creates a unique Redis hash key."""
    # Lowercase and remove extra spaces so "What is AI?" and "what is ai?" match perfectly
    clean_query = " ".join(query.lower().split())
    query_hash = hashlib.md5(clean_query.encode()).hexdigest()
    return f"api_cache:query:{query_hash}"

# Configure CORS for full-stack frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Redis client for health checks
redis_client = None
redis_cache_enabled = False

# --- NEW STARTUP EVENT ADDED HERE ---
@app.on_event("startup")
async def startup_event():
    """
    Ensure the Qdrant database collection exists before accepting requests.
    """
    logger.info("Checking and initializing Qdrant Vector Database...")
    try:
        vs = VectorStore()
        vs.create_collection()
        logger.info("Qdrant collection is ready and available.")
        
        logger.info("Initializing Redis Semantic LLM Cache for Fast Responses...")
        redis_url = os.getenv("REDIS_URL", "redis://redis-service:6379")
        embeddings = ModelFactory.get_embeddings()
        
        # Updated for langchain_redis compatibility
        set_llm_cache(RedisSemanticCache(
            redis_url=redis_url,
            embeddings=embeddings,       # PLURAL
            distance_threshold=0.10      # REPLACED score_threshold
        ))
        logger.info("Redis Semantic Cache initialized successfully.")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
# ------------------------------------

# Request Data Model
class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        example="Describe the culture and pricing model.",
        description="The natural language question to ask the RAG system."
    )
    session_id: str = Field(default="user_123", description="Unique ID to remember conversation")

# Response Data Model
class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    rewrite_tries: int
    revisions: int
    is_supported: str
    evidence: List[str]
    is_useful: str
    use_reason: str

@app.post("/api/v1/ask", response_model=QueryResponse, tags=["RAG Application"])
async def ask_question(request: QueryRequest):
    """
    Submit a query to the Agentic Self-RAG architecture with API-Level Redis Caching.
    """
    logger.info(f"--- API REQUEST RECEIVED FOR: {request.query} ---")

    # =================================================================
    # ⚡ 1. THE REDIS GATEWAY CACHE (The 50ms shortcut)
    # =================================================================
    cache_key = get_api_cache_key(request.query)
    
    try:
        cached_result = api_redis_client.get(cache_key)
        if cached_result:
            logger.info("--- ⚡ REDIS CACHE HIT! Bypassing LangGraph entirely. ---")
            parsed_result = json.loads(cached_result)
            return QueryResponse(**parsed_result)
    except Exception as e:
        logger.warning(f"Redis API cache check failed, proceeding to LangGraph: {e}")

    logger.info("--- 🐢 CACHE MISS! Routing query to LangGraph Pipeline ---")
    # =================================================================

    # 2. Initialize Graph and History
    rag_graph = get_graph()
    graph_config = {
        "recursion_limit": 50,
        "configurable": {"thread_id": request.session_id}
    }

    current_state = rag_graph.get_state(graph_config)
    history = []
    if current_state and current_state.values:
        history = current_state.values.get("chat_history", [])
    
    if len(history) > 4:
        history = history[-4:]
    
    initial_state = {
        "question": request.query,
        "retrieval_query": "",
        "chat_history": history,
        "rewrite_tries": 0,
        "route": "not_evaluated",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "not_evaluated",
        "evidence": [],
        "retries": 0,
        "isuse": "not_evaluated",
        "use_reason": ""
    }

    try:
        # 3. Execute Graph
        result = rag_graph.invoke(initial_state, config=graph_config)
        
        final_response = {
            "question": request.query,
            "answer": result.get("answer", "No answer found."),
            "route": result.get("route", "N/A"),
            "rewrite_tries": result.get("rewrite_tries", 0),
            "revisions": result.get("retries", 0),
            "is_supported": result.get("issup", "N/A"),
            "evidence": result.get("evidence", []),
            "is_useful": result.get("isuse", "N/A"),
            "use_reason": result.get("use_reason", "")
        }

        # =================================================================
        # 💾 4. SAVE TO REDIS CACHE FOR FUTURE REQUESTS
        # =================================================================
        try:
            # Save the successful response in Redis for 24 hours (86400 seconds)
            api_redis_client.setex(
                name=cache_key,
                time=86400,
                value=json.dumps(final_response)
            )
            logger.info(f"--- 💾 SAVED TO REDIS CACHE: {cache_key} ---")
        except Exception as e:
            logger.warning(f"Failed to save to Redis cache: {e}")

        return QueryResponse(**final_response)
    
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query through Self-RAG pipeline: {str(e)}"
        )

@app.post("/api/v1/ingest", tags=["Document Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be chunked, embedded, and indexed in the Vector Database.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    logger.info(f"--- API REQUEST RECEIVED TO INGEST FILE: {file.filename} ---")
    
    # Create a temporary directory to store the uploaded file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Save uploaded file safely
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process and chunk the directory containing the PDF
        processor = DocumentProcessor()
        chunks = processor.process_pdfs(temp_dir)
        
        if not chunks:
            raise ValueError("No text could be extracted or chunked from the uploaded PDF.")
            
        # Embed chunks and store in Qdrant Vector DB
        ingestor = DataIngestor()
        ingestor.ingest_chunks(chunks)
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {file.filename}", 
            "chunks_processed": len(chunks)
        }
        #exceptipn handling for file processing and ingestion errors
    except Exception as e:
        logger.error(f"Ingestion API failed for {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest document: {str(e)}"
        )
    finally:
        # Clean up the temporary directory to avoid storage leaks
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple status check endpoint."""
    return {"status": "ok", "service": "Agentic Self-RAG API"}

# Enable running via python directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)