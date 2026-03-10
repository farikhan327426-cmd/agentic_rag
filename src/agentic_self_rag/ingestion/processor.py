from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agentic_self_rag.core.logger import logger

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdfs(self, data_path: str):
        """
        Loads all PDFs from a directory and splits them into chunks.
        """
        try:
            logger.info(f"Loading PDFs from: {data_path}")
            loader = DirectoryLoader(
                data_path, 
                glob="*.pdf", 
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            
            logger.info(f"Splitting {len(documents)} pages into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            
            logger.info(f"Created {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.error(f"Error processing PDFs: {e}")
            return []