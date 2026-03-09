from typing import TypedDict, List

class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    question: str
    route: str
    documents: List[str]
    generation: str
    relevance: str
    hallucination: str
    utility: str
    retry_count: int