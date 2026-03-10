import json
from pathlib import Path
from pydantic import BaseModel, Field
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Aapke apne factory aur logger imports
from agentic_self_rag.utils.llm_factory import ModelFactory
from agentic_self_rag.core.logger import logger
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATASET_PATH = BASE_DIR / "data" / "raw" / "raw_dataset.json"
DATASET_PATH = BASE_DIR / "data" / "raw" / "dataset.json"

# 1. Pydantic Schema: LLM ko strictly is format mein output dene ke liye force karna
class QAPair(BaseModel):
    question: str = Field(description="A highly specific question based on the context.")
    ground_truth: str = Field(description="The exact, accurate answer based ONLY on the context.")

def generate_synthetic_dataset():
    # File load karein
    try:
        with open(RAW_DATASET_PATH, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {RAW_DATASET_PATH}")
        return

    # Apne factory se LLM load karein aur usko Pydantic model ke sath bind karein
    # Use "main" taake fast aur smart model (Llama-3 70B) use ho
    llm = ModelFactory.get_llm(model_type="main")
    structured_llm = llm.with_structured_output(QAPair)

    logger.info("Starting Synthetic Data Generation...")

    # Har khali chunk par loop lagayein
    for idx, item in enumerate(data):
        # Agar pehle se question bhara hua hai to skip kar dein
        if item.get("question") and item.get("ground_truth"):
            continue

        context = item["contexts"][0]
        logger.info(f"Generating QA for chunk {idx + 1}/{len(data)}...")

        # Prompt for the LLM
        prompt = f"""
        You are an expert AI data annotator.
        Read the following text (context) carefully.
        1. Generate a single, highly specific question that can be answered using ONLY this text.
        2. Provide the exact, accurate answer (ground truth) to that question.
        DO NOT use any outside knowledge.

        Context:
        {context}
        """

        try:
            # LLM ko invoke karein
            result = structured_llm.invoke(prompt)
            
            # Khali fields ko bhar dein
            item["question"] = result.question
            item["ground_truth"] = result.ground_truth
            logger.info(f"Success! Q: {result.question}")
            
        except Exception as e:
            logger.error(f"Failed to generate QA for chunk {idx}: {e}")

    # Pura bhara hua data nayi file mein save kar dein
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Synthetic Dataset successfully saved to {DATASET_PATH}")

if __name__ == "__main__":
    generate_synthetic_dataset()