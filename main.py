from src.agentic_self_rag.agentic_rag.graph import app
from src.agentic_self_rag.core.logger import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_agentic_rag(query: str):
    """
    Orchestrates the Self-RAG execution and prints a professional debug report.
    """
    logger.info(f"--- STARTING EXECUTION FOR: {query} ---")
    
    # Initial State matching your specific requirements
    initial_state = {
        "question": query,
        "retrieval_query": "",
        "rewrite_tries": 0,
        "route": "not_evaluated",
        "docs": [],
        "relevant_docs": [],
        "context": "",
        "answer": "",
        "issup": "not_evaluated",
        "evidence": [],
        "retries": 0,
        "isuse": "not_evaluated",
        "use_reason": ""
    }

    # Run the graph (using recursion_limit for the revise/rewrite loops)
    try:
        # We use invoke to get the final result, or stream for real-time logs
        result = app.invoke(initial_state, config={"recursion_limit": 50})
        
        # Professional Output Report
        print("\n" + "="*30)
        print("     RAG EXECUTION REPORT")
        print("="*30)
        print(f"QUESTION:  {query}")
        print(f"ROUTE:     {result.get('route', 'N/A')}")
        print(f"REWRITES:  {result.get('rewrite_tries', 0)}")
        print(f"REVISIONS: {result.get('retries', 0)}")
        
        print("\nVERIFICATION (IsSUP):")
        print(f"  Score:    {result.get('issup')}")
        if result.get('evidence'):
            for e in result['evidence']:
                print(f"  - {e}")

        print("\nUSEFULNESS (IsUSE):")
        print(f"  Score:    {result.get('isuse')}")
        print(f"  Reason:   {result.get('use_reason')}")

        print("\nFINAL ANSWER:")
        print("-" * 20)
        print(result.get('answer'))
        print("-" * 20)
        print("="*30 + "\n")

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")

if __name__ == "__main__":
    # Test with a question that exists in your PDFs
    # run_agentic_rag("What are the features of NexaInsight?")
    
    # Test with a question that might need rewriting/revision
    run_agentic_rag("who is the ceo of nexaai?")