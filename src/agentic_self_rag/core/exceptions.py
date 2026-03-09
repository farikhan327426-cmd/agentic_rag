class AgenticRAGError(Exception):
    """Base class for all exceptions in this project."""
    pass

class ModelProviderError(AgenticRAGError):
    """Raised when an LLM provider (Groq/Google/OpenRouter) fails."""
    pass

class ConfigurationError(AgenticRAGError):
    """Raised when settings or env vars are invalid."""
    pass