def revise_answer(state: dict):
    """
    If the answer is hallucinated or not useful, try to regenerate it.
    """
    logger.info("--- REVISING ANSWER ---")
    question = state["question"]
    documents = state["documents"]
    
    llm = ModelFactory.get_llm(model_type="main")
    
    # Simple re-generation logic for now
    response = llm.invoke(f"Refine the following answer for better accuracy and groundedness. \nQuestion: {question} \nContext: {documents}")
    
    return {"generation": response.content}