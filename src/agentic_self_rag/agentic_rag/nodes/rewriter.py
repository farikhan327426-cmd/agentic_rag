from pydantic import BaseModel, Field
from agentic_self_rag.utils.llm_factory import ModelFactory
from agentic_self_rag.core.logger import logger
from ..state import AgentState

class RewriteDecision(BaseModel):
    retrieval_query: str = Field(description="Optimized query for vector retrieval.")

def rewrite_question(state: AgentState):
    """Optimizes query for vector retrieval."""
    logger.info("--- REWRITING QUESTION ---")
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_llm = llm.with_structured_output(RewriteDecision)
    
    res = structured_llm.invoke(f"Rewrite this for vector search: {state['question']}")
    logger.info(f"--- REWRITTEN QUERY: {res.retrieval_query} ---")
    return {
        "retrieval_query": res.retrieval_query, 
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [] # Reset docs for new search
    }