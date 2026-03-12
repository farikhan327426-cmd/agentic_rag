from typing import TypedDict, List, Literal

class AgentState(TypedDict):
    question: str
    retrieval_query: str      # Optimized for vector search
    rewrite_tries: int        # Counter for query rewriting
    need_retrieval: bool      # Result from router
    route: str                # Routing decision
    chat_history: List[dict]

    docs: List[dict]          # Raw chunks from Qdrant
    relevant_docs: List[dict] # Filtered chunks
    context: str              # Merged text for LLM
    answer: str               # Current generation
    issup: Literal["fully_supported", "partially_supported", "no_support", "not_evaluated"]
    evidence: List[str]       # Quotes supporting the answer
    retries: int              # Counter for answer revision
    isuse: Literal["useful", "not_useful", "not_evaluated"]
    use_reason: str