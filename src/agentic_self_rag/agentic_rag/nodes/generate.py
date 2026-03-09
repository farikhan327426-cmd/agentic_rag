import yaml
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

def generate(state: dict):
    """
    Generate an answer using the retrieved context.
    """
    question = state["question"]
    documents = state["documents"]
    logger.info("--- GENERATING ANSWER ---")

    # 1. Load the prompt from YAML
    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    prompt_template = prompts["rag_prompts"]["traditional_rag"]

    # 2. Prepare the LLM (Using our Factory)
    llm = ModelFactory.get_llm(model_type="main")

    # 3. Format the prompt
    context = "\n\n".join(documents)
    formatted_prompt = prompt_template.format(context=context, question=question)

    # 4. Invoke LLM
    response = llm.invoke(formatted_prompt)
    
    return {"generation": response.content, "question": question}


def generate_direct(state: dict):
    """
    Generate an answer without using RAG context.
    """
    question = state["question"]
    logger.info("--- GENERATING DIRECT ANSWER ---")

    llm = ModelFactory.get_llm(model_type="main")
    response = llm.invoke(f"Answer the following question directly and concisely: {question}")
    
    return {"generation": response.content}