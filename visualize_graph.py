import os
from src.agentic_self_rag.agentic_rag.graph import create_graph

def save_graph_image(output_path="graph_visualization.png"):
    """
    Compiles the LangGraph workflow and saves its visualization as a PNG image.
    """
    try:
        # Build and compile the graph
        graph = create_graph()
        
        # Generate the PNG image data
        image_data = graph.get_graph().draw_mermaid_png()
        
        # Write the image data to a file
        with open(output_path, "wb") as f:
            f.write(image_data)
            
        print(f"Success! Graph image saved to: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Failed to generate graph image: {e}")
        print("\nNote: Plotting might require additional dependencies.")
        print("Try running: uv pip install langgraph-cli[inmem] grandalf")

if __name__ == "__main__":
    save_graph_image()
