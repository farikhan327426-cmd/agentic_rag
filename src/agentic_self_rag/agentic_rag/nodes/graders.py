import yaml
from pydantic import BaseModel, Field
from typing import Literal, List
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

# Pydantic Schemas for Structured Output
class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(description="True if doc discusses the same topic area.")

class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str

def is_relevant(state: dict):
    """Strictly filters documents based on topic area."""
    logger.info("--- FILTERING DOCUMENTS FOR TOPIC RELEVANCE ---")
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_llm = llm.with_structured_output(RelevanceDecision)
    
    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    
    relevant_docs = []
    for doc in state.get("docs", []):
        res = structured_llm.invoke([
            {"role": "system", "content": prompts["grader_prompts"]["relevance_instructions"]},
            {"role": "user", "content": f"Question: {state['question']}\nDoc: {doc['text']}"}
        ])
        if res.is_relevant:
            relevant_docs.append(doc)
    
    return {"relevant_docs": relevant_docs}

def is_sup(state: dict):
    """Checks if answer is grounded in context (Hallucination check)."""
    logger.info("--- VERIFYING GROUNDEDNESS (IsSUP) ---")
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_llm = llm.with_structured_output(IsSUPDecision)

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    res = structured_llm.invoke([
        {"role": "system", "content": prompts["grader_prompts"]["issup_instructions"]},
        {"role": "user", "content": f"Context: {state['context']}\nAnswer: {state['answer']}"}
    ])
    logger.info(f"--- IsSUP Result: {res.issup} | Evidence: {res.evidence} ---")
    return {"issup": res.issup, "evidence": res.evidence}

def is_use(state: dict):
    """Checks if the answer actually answers the user question."""
    logger.info("--- CHECKING ANSWER UTILITY ---")
    llm = ModelFactory.get_llm(model_type="cheap")
    structured_llm = llm.with_structured_output(IsUSEDecision)

    with open("config/prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)

    res = structured_llm.invoke([
        {"role": "system", "content": prompts["grader_prompts"]["isuse_instructions"]},
        {"role": "user", "content": f"Question: {state['question']}\nAnswer: {state['answer']}"}
    ])
    logger.info(f"--- IsUSE Result: {res.isuse} | Reason: {res.reason} ---")
    return {"isuse": res.isuse, "use_reason": res.reason}