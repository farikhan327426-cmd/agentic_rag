from qdrant_client.http import models
from agentic_self_rag.database.connection import qdrant_manager
from agentic_self_rag.core.logger import logger
from agentic_self_rag.core.config_loader import settings

class VectorStore:
    def __init__(self):
        self.client = qdrant_manager.get_client()
        self.collection_name = settings.get("vector_db", {}).get("collection_name", "agentic_rag_docs")
        self.vector_size = 3072 

    def create_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size, 
                        distance=models.Distance.COSINE
                    ),
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise e

    def search(self, query_vector: list, limit: int = 5):
        """
        Updated to use query_points (Modern Qdrant API)
        """
        try:
            # query_points is more efficient and handles the newer Qdrant features
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
            # The modern API returns a 'points' attribute
            return response.points
        except Exception as e:
            # If query_points fails, try the older search method as a fallback
            try:
                logger.warning(f"query_points failed, trying legacy search: {e}")
                return self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True
                )
            except Exception as e2:
                logger.error(f"Both search methods failed: {e2}")
                return []