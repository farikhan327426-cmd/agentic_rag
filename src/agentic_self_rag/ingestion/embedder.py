from uuid import uuid4
from qdrant_client.http import models
from agentic_self_rag.database.vector_store import VectorStore
from agentic_self_rag.utils.llm_factory import ModelFactory
from agentic_self_rag.core.logger import logger

class DataIngestor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embeddings = ModelFactory.get_embeddings()

    def ingest_chunks(self, chunks):
        """
        Embeds chunks and stores them in Qdrant with metadata.
        """
        if not chunks:
            logger.warning("No chunks to ingest.")
            return

        try:
            logger.info(f"Starting ingestion of {len(chunks)} chunks...")
            
            points = []
            for chunk in chunks:
                # Generate Embedding
                vector = self.embeddings.embed_query(chunk.page_content)
                
                # Create a professional payload (metadata)
                payload = {
                    "text": chunk.page_content,
                    "metadata": chunk.metadata  # Contains source file and page number
                }
                
                # Create Qdrant Point
                points.append(
                    models.PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload=payload
                    )
                )

            # Upsert in batches to Qdrant
            self.vector_store.client.upsert(
                collection_name=self.vector_store.collection_name,
                points=points
            )
            logger.info("Ingestion completed successfully.")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise e