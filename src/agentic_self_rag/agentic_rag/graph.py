from langgraph.graph import StateGraph, END, START
from agentic_self_rag.agentic_rag.state import AgentState
from agentic_self_rag.agentic_rag.nodes import (
    router, retrieve, graders, generate, rewriter, reviser
)

def create_graph():
    workflow = StateGraph(AgentState)

    # Define Nodes
    workflow.add_node("decide_retrieval", router.route_question)
    workflow.add_node("retrieve", retrieve.retrieve)
    workflow.add_node("is_relevant", graders.is_relevant)
    workflow.add_node("generate_from_context", generate.generate)
    workflow.add_node("generate_direct", generate.generate_direct)
    workflow.add_node("is_sup", graders.is_sup)
    workflow.add_node("revise_answer", reviser.revise_answer)
    workflow.add_node("is_use", graders.is_use)
    workflow.add_node("rewrite_question", rewriter.rewrite_question)
    workflow.add_node("no_answer_found", lambda x: {"answer": "No answer found."})

    # Define Edges & Routing
    workflow.set_entry_point("decide_retrieval")

    workflow.add_conditional_edges(
        "decide_retrieval",
        lambda x: "retrieve" if x["route"] == "vectorstore" else "generate_direct",
        {
            "retrieve": "retrieve",
            "generate_direct": "generate_direct"
        }
    )

    workflow.add_edge("retrieve", "is_relevant")

    workflow.add_conditional_edges(
        "is_relevant",
        lambda x: "generate_from_context" if x["relevant_docs"] else "no_answer_found",
        {
            "generate_from_context": "generate_from_context",
            "no_answer_found": "no_answer_found"
        }
    )

    workflow.add_edge("generate_from_context", "is_sup")

    workflow.add_conditional_edges(
        "is_sup",
        lambda x: "is_use" if x["issup"] == "fully_supported" or x["retries"] > 5 else "revise_answer",
        {
            "is_use": "is_use",
            "revise_answer": "revise_answer"
        }
    )
    workflow.add_edge("revise_answer", "is_sup")

    workflow.add_conditional_edges(
        "is_use",
        lambda x: END if x["isuse"] == "useful" else "rewrite_question" if x["rewrite_tries"] < 3 else "no_answer_found",
        {
            END: END,
            "rewrite_question": "rewrite_question",
            "no_answer_found": "no_answer_found"
        }
    )
    workflow.add_edge("rewrite_question", "retrieve")

    workflow.add_edge("generate_direct", END)
    workflow.add_edge("no_answer_found", END)

    return workflow.compile()


# Lazy initialization - graph is only compiled when explicitly requested
# This prevents compilation overhead on module import
_graph_instance = None

def get_graph():
    """Get or create the compiled graph instance (lazy initialization)."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_graph()
    return _graph_instance