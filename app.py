from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from dotenv import load_dotenv

from src.agentic_self_rag.agentic_rag.graph import app as rag_app
from src.agentic_self_rag.core.logger import logger

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

# Request Data Model
class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        example="Describe the culture and pricing model.",
        description="The natural language question to ask the RAG system."
    )

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
        result = rag_app.invoke(initial_state, config={"recursion_limit": 50})
        
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

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple status check endpoint."""
    return {"status": "ok", "service": "Agentic Self-RAG API"}

# Enable running via python directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
