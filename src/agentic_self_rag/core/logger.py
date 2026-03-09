import sys
from loguru import logger

# Configure loguru to output to both console and a file for debugging on your server
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/app.log", rotation="10 MB", retention="10 days", compression="zip")

# Export it so other files can use it
__all__ = ["logger"]