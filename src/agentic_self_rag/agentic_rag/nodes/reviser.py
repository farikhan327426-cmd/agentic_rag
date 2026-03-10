from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger
from ..state import AgentState

def revise_answer(state: AgentState):
    """Strictly refines the answer using only context quotes."""
    logger.warning("--- REVISING ANSWER (STRICT MODE) ---")
    llm = ModelFactory.get_llm(model_type="main")
    
    prompt = (
        "You are a STRICT reviser. Use ONLY the CONTEXT quotes to answer. "
        f"Context: {state['context']}\nQuestion: {state['question']}"
    )
    
    out = llm.invoke(prompt)
    logger.info(f"--- REVISED ANSWER: {out.content} ---")
    return {"answer": out.content, "retries": state.get("retries", 0) + 1}