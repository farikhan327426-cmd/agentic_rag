import json
import sys
import warnings
from pathlib import Path
from datasets import Dataset
from ragas import evaluate

# 1. THE FIX: Hide annoying Ragas deprecation warnings cleanly
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2. Revert back to the stable metrics that worked perfectly with LangChain
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import your actual production graph and factory
from agentic_self_rag.agentic_rag.graph import get_graph
from agentic_self_rag.utils.llm_factory import ModelFactory
from agentic_self_rag.core.logger import logger

# Absolute pathing for safety
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "raw" / "dataset.json"

def load_dataset():
    """Loads the golden dataset safely."""
    try:
        with open(DATASET_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {DATASET_PATH}. Please create it.")
        raise

def run_evaluation():
    logger.info("Starting Ragas Evaluation Pipeline...")
    
    test_data = load_dataset()
    app = get_graph() # Lazy load your compiled LangGraph
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 1. GENERATION PHASE: Let your LangGraph answer the test questions
    for item in test_data:
        question = item.get("question", "").strip()
        
        # Skip items with empty questions
        if not question:
            logger.warning("Skipping dataset item with empty question")
            continue
            
        logger.info(f"Processing query: {question}")
        
        # Initialize state with default values
        initial_state = {
            "question": question,
            "retrieval_query": "",
            "rewrite_tries": 0,
            "need_retrieval": False,
            "route": "",
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
        
        # Invoke your graph
        result_state = app.invoke(initial_state)
        
        # Extract data needed by Ragas
        questions.append(question)
        answers.append(result_state.get("answer", "No answer found."))
        ground_truths.append(item["ground_truth"])
        
        # Ragas needs contexts as a list of strings
        raw_docs = result_state.get("relevant_docs", [])
        doc_texts = [doc["text"] for doc in raw_docs] if raw_docs else [""]
        contexts.append(doc_texts)

    # 2. FORMATTING: Convert to HuggingFace Dataset format
    if not questions:
        logger.error("No valid questions found in dataset.")
        return
    
    dataset_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(dataset_dict)

    # 3. JUDGE INITIALIZATION: Use your factory as it was perfectly working before
    judge_llm = ModelFactory.get_llm(model_type="main")
    judge_embeddings = ModelFactory.get_embeddings()

    # 4. EVALUATION PHASE: Ragas calculates the scores using global llm/embeddings
    logger.info("Graph execution complete. Handing over to Ragas for grading...")
    result = evaluate(
        dataset=hf_dataset,
        metrics=[
            ContextPrecision(), 
            ContextRecall(),    
            Faithfulness(),      
            AnswerRelevancy()   
        ],
        llm=judge_llm,
        embeddings=judge_embeddings
    )
    
    # 5. EXPORT RESULTS
    logger.info("--- EVALUATION SCORES ---")
    print(result)
    
    csv_path = BASE_DIR / "evals" / "evaluation_results.csv"
    result.to_pandas().to_csv(csv_path, index=False)
    logger.info(f"Detailed metrics saved to {csv_path}")

if __name__ == "__main__":
    run_evaluation()