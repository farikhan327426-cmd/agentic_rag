import yaml
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger
from ..state import AgentState

def generate(state: AgentState):
    """
    Generate an answer using the retrieved context.
    """
    question = state["question"]
    relevant_docs = state["relevant_docs"]
    logger.info("--- GENERATING ANSWER ---")

    # 1. Load the prompt from YAML
    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["rag_prompts"]["generation_instructions"]

    # 2. Prepare the LLM (Using our Factory)
    llm = ModelFactory.get_llm(model_type="main")

    # 3. Format the prompt
    context = "\n\n".join([doc["text"] for doc in relevant_docs])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    # 4. Invoke LLM
    response = llm.invoke(messages)
    
    logger.info(f"--- GENERATED ANSWER: {response.content} ---")
    return {"answer": response.content, "context": context, "question": question}


def generate_direct(state: AgentState):
    """
    Generate an answer without using RAG context.
    """
    question = state["question"]
    logger.info("--- GENERATING DIRECT ANSWER ---")

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["rag_prompts"]["direct_instructions"]

    llm = ModelFactory.get_llm(model_type="main")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"}
    ]
    response = llm.invoke(messages)
    
    logger.info(f"--- GENERATED DIRECT ANSWER: {response.content} ---")
    return {"answer": response.content}