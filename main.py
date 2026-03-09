import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agentic_self_rag.agentic_rag.graph import app
from src.agentic_self_rag.core.logger import logger

# Initialize FastAPI
fastapi_app = FastAPI(
    title="Agentic Self-RAG API",
    description="Industrial Grade Self-Correcting RAG System"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    steps: list

@fastapi_app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        inputs = {"question": request.question, "retry_count": 0}
        steps = []
        final_state = None

        # Stream the graph execution
        for output in app.stream(inputs):
            for node, state in output.items():
                steps.append(node)
                final_state = state
        
        return QueryResponse(
            answer=final_state["generation"],
            steps=steps
        )
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    # On your Linux server, you would run this via: uv run python -m uvicorn ...
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)