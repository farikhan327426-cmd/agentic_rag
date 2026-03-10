from agentic_self_rag.core.logger import logger

def decide_to_generate(state):
    """Determines whether to generate an answer or re-route/rewrite."""
    if state["route"] == "direct":
        return "generate_direct"
    
    if state["relevance"] == "yes":
        return "generate"
    
    return "rewrite_question"

def grade_generation_v_documents_and_question(state):
    """Determines whether the generation is grounded and useful."""
    if state["hallucination"] == "supported":
        if state["utility"] == "useful":
            return "useful"
        return "not_useful"
    return "not_supported"