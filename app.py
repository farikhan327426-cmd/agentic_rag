from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from dotenv import load_dotenv

from src.agentic_self_rag.agentic_rag.graph import get_graph
from src.agentic_self_rag.core.logger import logger
from src.agentic_self_rag.ingestion.processor import DocumentProcessor
from src.agentic_self_rag.ingestion.embedder import DataIngestor
from src.agentic_self_rag.database.vector_store import VectorStore  # <-- NEW IMPORT ADDED HERE
import os
import shutil
import tempfile

# Load environment variables
load_dotenv()

# Initialize API
app = FastAPI(
    title="Agentic Self-RAG API",
    description="Professional enterprise API for querying documents via Self-Reflective Agentic RAG.",
    version="1.0.0"
)

# Configure CORS for full-stack frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")
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
    Submit a query to the Agentic Self-RAG architecture.
    The query will be dynamically routed, retrieved, graded, generated, and iteratively improved.
    """
    logger.info(f"--- API REQUEST RECEIVED FOR: {request.query} ---")
    
    # Establish uniform initial state for the StateGraph representation
    initial_state = {
        "question": request.query,
        "retrieval_query": "",
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
        # We use invoke sequentially for REST APIs
        rag_graph = get_graph()  # Lazy initialization on first use
        graph_config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": request.session_id}
        }
        result = rag_graph.invoke(initial_state, config=graph_config)
        
        # Structure the successful execution payload
        return QueryResponse(
            question=request.query,
            answer=result.get("answer", "No answer found."),
            route=result.get("route", "N/A"),
            rewrite_tries=result.get("rewrite_tries", 0),
            revisions=result.get("retries", 0),
            is_supported=result.get("issup", "N/A"),
            evidence=result.get("evidence", []),
            is_useful=result.get("isuse", "N/A"),
            use_reason=result.get("use_reason", "")
        )
    
    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        # Raise standard API exception instead of raw traceback
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