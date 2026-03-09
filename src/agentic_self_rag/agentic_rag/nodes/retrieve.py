from src.agentic_self_rag.database.vector_store import VectorStore
from src.agentic_self_rag.utils.llm_factory import ModelFactory
from src.agentic_self_rag.core.logger import logger

def retrieve(state: dict):
    """
    Retrieve documents from Qdrant based on the question.
    """
    question = state["question"]
    logger.info(f"--- RETRIEVING DOCUMENTS FOR: {question} ---")
    
    # 1. Get embeddings and Vector Store
    embedder = ModelFactory.get_embeddings()
    vs = VectorStore()
    
    # 2. Embed the query
    query_vector = embedder.embed_query(question)
    
    # 3. Search Qdrant
    results = vs.search(query_vector, limit=3)
    
    # 4. Extract text from results
    documents = [res.payload["text"] for res in results]
    
    return {"documents": documents, "question": question}