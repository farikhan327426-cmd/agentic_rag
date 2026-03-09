from qdrant_client import QdrantClient
from src.agentic_self_rag.core.config_loader import settings
from src.agentic_self_rag.core.logger import logger

class QdrantManager:
    """
    Manages the connection to the Qdrant Vector Database.
    """
    def __init__(self):
        self.url = settings.get("vector_db", {}).get("url", "http://localhost:6333")
        self._client = None

    def get_client(self) -> QdrantClient:
        if self._client is None:
            try:
                logger.info(f"Connecting to Qdrant at {self.url}")
                self._client = QdrantClient(url=self.url)
                # Check connection
                self._client.get_collections()
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise ConnectionError(f"Could not reach Qdrant: {e}")
        return self._client

# Global instance for the app
qdrant_manager = QdrantManager()