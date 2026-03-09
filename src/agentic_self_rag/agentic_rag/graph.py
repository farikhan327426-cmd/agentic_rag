from langgraph.graph import END, StateGraph
from src.agentic_self_rag.agentic_rag.state import AgentState
from src.agentic_self_rag.agentic_rag.nodes.router import route_question
from src.agentic_self_rag.agentic_rag.nodes.retrieve import retrieve
from src.agentic_self_rag.agentic_rag.nodes.generate import generate, generate_direct
from src.agentic_self_rag.agentic_rag.nodes.graders import (
    grade_documents, 
    grade_generation_v_documents, 
    grade_generation_v_question
)
from src.agentic_self_rag.agentic_rag.nodes.rewriter import rewrite_question
from src.agentic_self_rag.agentic_rag.nodes.reviser import revise_answer
from src.agentic_self_rag.agentic_rag.edges import (
    decide_to_generate, 
    grade_generation_v_documents_and_question
)

def create_graph():
    workflow = StateGraph(AgentState)

    # 1. Define the Nodes (The Boxes in your diagram)
    workflow.add_node("router", route_question)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_direct", generate_direct)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("grade_hallucination", grade_generation_v_documents)
    workflow.add_node("grade_utility", grade_generation_v_question)
    workflow.add_node("revise_answer", revise_answer)

    # 2. Build the Edges (The Arrows in your diagram)
    workflow.set_entry_point("router")

    # Entry Logic
    workflow.add_conditional_edges(
        "router",
        lambda x: "retrieve" if x["route"] == "vectorstore" else "generate_direct"
    )

    # RAG Path
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        lambda x: "generate" if x["relevance"] == "yes" else "rewrite_question",
        {
            "generate": "generate",
            "rewrite_question": "rewrite_question"
        }
    )

    # Rewrite Loop
    workflow.add_edge("rewrite_question", "retrieve")

    # Generation & Self-Correction Path
    workflow.add_edge("generate", "grade_hallucination")

    workflow.add_conditional_edges(
        "grade_hallucination",
        lambda x: "grade_utility" if x["hallucination"] == "supported" else "revise_answer",
        {
            "grade_utility": "grade_utility",
            "revise_answer": "revise_answer"
        }
    )

    workflow.add_conditional_edges(
        "grade_utility",
        lambda x: END if x["utility"] == "useful" else "revise_answer",
        {
            END: END,
            "revise_answer": "revise_answer"
        }
    )

    # Revise back to checking
    workflow.add_edge("revise_answer", "grade_hallucination")
    
    # Direct Path
    workflow.add_edge("generate_direct", END)

    return workflow.compile()

# Compile the graph
app = create_graph()