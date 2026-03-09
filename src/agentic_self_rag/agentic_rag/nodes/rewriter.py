import yaml
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

def rewrite_question(state: dict):
    """
    Transform the query to produce a better search term.
    """
    question = state["question"]
    # Initialize or increment retry count
    retries = state.get("retry_count", 0) + 1
    
    logger.info(f"--- REWRITING QUESTION (Attempt {retries}) ---")

    # 1. Load Prompt
    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["rewriter_prompts"]["rewrite_instructions"]

    # 2. Setup LLM
    llm = ModelFactory.get_llm(model_type="cheap")
    
    # 3. Generate better query
    response = llm.invoke(system_prompt.format(question=question))
    better_question = response.content
    
    logger.info(f"--- NEW QUERY: {better_question} ---")
    
    return {"question": better_question, "retry_count": retries}