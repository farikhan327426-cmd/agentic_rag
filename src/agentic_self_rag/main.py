# src/agentic_self_rag/main.py

from agentic_self_rag.core.logger import setup_logger

logger = setup_logger(__name__)

def main():
    logger.info("Agentic Self RAG started")

if __name__ == "__main__":
    main()