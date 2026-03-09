import yaml
from pydantic import BaseModel, Field
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

class GradeRetrieval(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Document is relevant to the question, 'yes' or 'no'")

def grade_documents(state: dict):
    """
    Determines whether the retrieved documents are relevant to the question.
    """
    question = state["question"]
    documents = state["documents"]
    
    logger.info("--- CHECKING DOCUMENT RELEVANCE ---")

    # 1. Setup Grader
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_grader = llm.with_structured_output(GradeRetrieval)

    # 2. Load Prompt
    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    system_prompt = prompts["grader_prompts"]["retrieval_grader_instructions"]

    # 3. Grade the documents
    # In a professional setup, we check if the collective context is useful
    combined_context = "\n\n".join(documents)
    
    scoring = structured_grader.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Retrieved context: \n\n {combined_context} \n\n User question: {question}"}
    ])

    relevance = scoring.binary_score.lower()
    logger.info(f"--- RELEVANCE SCORE: {relevance} ---")
    
    return {"relevance": relevance, "documents": documents, "question": question}


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'supported' or 'not supported'")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'useful' or 'not useful'")

def grade_generation_v_documents(state: dict):
    """Determines whether the generation is grounded in the documents."""
    logger.info("--- CHECKING FOR HALLUCINATIONS ---")
    
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_grader = llm.with_structured_output(GradeHallucination)

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    score = structured_grader.invoke([
        {"role": "system", "content": prompts["hallucination_grader_prompts"]["instructions"]},
        {"role": "user", "content": f"Documents: {state['documents']} \n\n Answer: {state['generation']}"}
    ])

    return {"hallucination": score.binary_score.lower()}

def grade_generation_v_question(state: dict):
    """Determines whether the generation actually answers the question."""
    logger.info("--- CHECKING ANSWER UTILITY ---")
    
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_grader = llm.with_structured_output(GradeAnswer)

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    score = structured_grader.invoke([
        {"role": "system", "content": prompts["answer_grader_prompts"]["instructions"]},
        {"role": "user", "content": f"Question: {state['question']} \n\n Answer: {state['generation']}"}
    ])

    return {"utility": score.binary_score.lower()}