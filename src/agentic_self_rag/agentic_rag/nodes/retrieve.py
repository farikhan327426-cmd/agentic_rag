from src.agentic_self_rag.database.vector_store import VectorStore
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

def retrieve(state: dict):
    """
    Retrieve documents from Qdrant based on the question.
    """
    question = state["question"]
    query_to_use = state.get("retrieval_query") or question
    logger.info(f"--- RETRIEVING DOCUMENTS FOR: {query_to_use} ---")
    
    # 1. Get embeddings and Vector Store
    embedder = ModelFactory.get_embeddings()
    vs = VectorStore()
    
    # 2. Embed the query
    query_vector = embedder.embed_query(query_to_use)
    
    # 3. Search Qdrant
    results = vs.search(query_vector, limit=3)
    
    # 4. Extract text from results as dicts
    docs = [{"text": res.payload["text"]} for res in results]
    
    return {"docs": docs, "question": question}