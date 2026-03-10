from typing import TypedDict, List, Literal

class AgentState(TypedDict):
    question: str
    retrieval_query: str      # Optimized for vector search
    rewrite_tries: int        # Counter for query rewriting
    need_retrieval: bool      # Result from router
    docs: List[dict]          # Raw chunks from Qdrant
    relevant_docs: List[dict] # Filtered chunks
    context: str              # Merged text for LLM
    answer: str               # Current generation
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]       # Quotes supporting the answer
    retries: int              # Counter for answer revision
    isuse: Literal["useful", "not_useful"]
    use_reason: str