"""
Simple RAGAS Dataset Creation Script
Loads PDFs from data/raw/ and creates a basic evaluation dataset structure
with manually curated questions. No LLM API calls needed.
"""
import json
from pathlib import Path
from typing import List, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_self_rag.core.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATASET_PATH = BASE_DIR / "data" / "raw" / "raw_dataset.json"


class SimpleRAGASDatasetCreator:
    """Creates basic RAGAS-compatible evaluation datasets from PDF documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_documents(self, data_dir: str) -> List:
        """Load all PDFs from directory."""
        try:
            logger.info(f"Loading PDFs from: {data_dir}")
            loader = DirectoryLoader(
                data_dir,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDFs")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDFs: {e}")
            return []
    
    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        try:
            logger.info(f"Splitting {len(documents)} pages into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return []
    
    def create_basic_dataset(self, save_path: Path) -> Dict:
        """Create basic RAGAS evaluation dataset with document chunks as contexts."""
        # Load and process documents
        documents = self.load_documents(str(RAW_DATA_DIR))
        if not documents:
            logger.error("No documents loaded. Exiting.")
            return {}
        
        chunks = self.split_documents(documents)
        if not chunks:
            logger.error("No chunks created. Exiting.")
            return {}
        
        # Create dataset items - each chunk becomes a context
        dataset_items = []
        
        for idx, chunk in enumerate(chunks):
            context = chunk.page_content.strip()
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 0)
            
            if context:  # Only add non-empty chunks
                dataset_items.append({
                    "question": "",  # To be filled manually or by LLM
                    "ground_truth": "",  # To be filled manually or by LLM
                    "contexts": [context],
                    "source": source,
                    "page": page,
                    "chunk_id": idx
                })
        
        logger.info(f"Created {len(dataset_items)} dataset items")
        
        # Save dataset
        RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RAW_DATASET_PATH, "w") as f:
            json.dump(dataset_items, f, indent=2)
        
        logger.info(f"Dataset saved to {RAW_DATASET_PATH}")
        return {
            "total_items": len(dataset_items),
            "total_chunks": len(chunks),
            "path": str(RAW_DATASET_PATH),
            "note": "Dataset created with document chunks. Add questions and ground_truth manually or use LLM generation."
        }


def main():
    """Main execution."""
    logger.info("Starting Basic RAGAS Dataset Creation")
    
    creator = SimpleRAGASDatasetCreator()
    stats = creator.create_basic_dataset(RAW_DATASET_PATH)
    
    logger.info(f"Dataset creation complete: {stats}")
    print(f"\n✓ Dataset created with {stats.get('total_items', 0)} items")
    print(f"  Location: {stats.get('path')}")


if __name__ == "__main__":
    main()
