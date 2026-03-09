import os
from typing import Union
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from src.agentic_self_rag.core.config_loader import settings
from src.agentic_self_rag.core.logger import logger
from src.agentic_self_rag.core.exceptions import ModelProviderError
from dotenv import load_dotenv

load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")


class ModelFactory:
    """
    Factory class to instantiate LLMs and Embedding models 
    based on the centralized configuration.
    """

    @staticmethod
    def get_llm(model_type: str = "main"):
        """
        Returns an LLM instance based on settings.yaml.
        model_type can be 'main' (fast/expensive) or 'cheap' (fast/efficient).
        """
        provider = settings.get("llm", {}).get("main_provider")
        
        # Determine which model name to use from YAML
        if model_type == "main":
            model_name = settings.get("llm", {}).get("fast_model")
        else:
            model_name = settings.get("llm", {}).get("cheap_model")
            
        temp = settings.get("llm", {}).get("temperature", 0)

        try:
            if provider == "groq":
                logger.info(f"Initializing Groq LLM: {model_name}")
                return ChatGroq(
                    #api_key=settings.env.GROQ_API_KEY,
                    api_key=groq_api_key,
                    model=model_name,
                    temperature=temp
                )
                logger.info(f"Groq LLM initialized: {model_name}")

            elif provider == "google":
                logger.info(f"Initializing Google LLM: {model_name}")
                return ChatGoogleGenerativeAI(
                    api_key=settings.env.GOOGLE_API_KEY,
                    model=model_name,
                    temperature=temp
                )

            elif provider == "openrouter":
                logger.info(f"Initializing OpenRouter LLM: {model_name}")
                # OpenRouter uses OpenAI SDK with a different base URL
                return ChatOpenAI(
                    api_key=settings.env.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    model=model_name,
                    temperature=temp
                )

            else:
                raise ModelProviderError(f"Unsupported provider: {provider}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise ModelProviderError(f"Error connecting to {provider}: {e}")

    @staticmethod
    def get_embeddings():
        """
        Returns the embedding model specified in settings.yaml.
        """
        provider = settings.get("embeddings", {}).get("provider")
        model_name = settings.get("embeddings", {}).get("model")

        try:
            if provider == "google":
                logger.info(f"Initializing Google Embeddings: {model_name}")
                return GoogleGenerativeAIEmbeddings(
                    api_key=settings.env.GOOGLE_API_KEY,
                    model=model_name
                )
            else:
                raise ModelProviderError(f"Unsupported embedding provider: {provider}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Embeddings: {str(e)}")
            raise ModelProviderError(f"Error initializing {provider} embeddings: {e}")