from src.agentic_self_rag.agentic_rag.graph import get_graph
from src.agentic_self_rag.core.logger import logger
from dotenv import load_dotenv
from redis import Redis
import os

# Load environment variables
load_dotenv()

def run_agentic_rag(query: str, session_id: str = None):
    """
    Orchestrates the Self-RAG execution and prints a professional debug report.

    If `session_id` is supplied the graph will load and persist state under
    that key, demonstrating the Redis saver behaviour.  After running we also
    inspect Redis for any cache/state keys associated with the session.
    """
    logger.info(f"--- STARTING EXECUTION FOR: {query} (session={session_id}) ---")
    
    # Get the compiled graph instance (lazy initialization)
    graph = get_graph()

    graph_config = {"recursion_limit": 50}
    if session_id:
        graph_config["configurable"] = {"thread_id": session_id}

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
        result = graph.invoke(initial_state, config=graph_config)
        
        # after run, if session provided, peek at state stored in Redis
        if session_id:
            redis_url = os.getenv("REDIS_URL", "redis://redis-service:6379")
            r = Redis.from_url(redis_url)
            key_pattern = f"langgraph:state:*{session_id}*"
            stored = r.keys(key_pattern)
            print(f"\nsession keys matching {key_pattern}: {stored}")
            # show any LLM cache entries created during this invocation
            cache_keys = r.keys("langchain:cache:*")
            print(f"llm cache keys now in Redis: {cache_keys}\n")

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
    # include a session id to exercise the Redis saver and history
    run_agentic_rag("who is the ceo of nexaai?", session_id="demo_user")
    # feel free to run again with the same query+session to see cache hits
    run_agentic_rag("who is the ceo of nexaai?", session_id="demo_user")