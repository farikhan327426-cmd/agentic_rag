from src.agentic_self_rag.agentic_rag.graph import get_graph
from IPython.display import Image, display

def save_graph_image(output_path: str = "graph.png"):
    try:
        # Get the compiled graph (lazy initialization on first use)
        compiled_graph = get_graph()
        # Generate the mermaid diagram as a PNG
        graph_png = compiled_graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(graph_png)
        print(f"Graph image saved to {output_path}")
    except Exception as e:
        print(f"Could not save graph image: {e}")

if __name__ == "__main__":
    save_graph_image()