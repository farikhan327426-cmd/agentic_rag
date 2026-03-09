# src/agentic_self_rag/core/config_loader.py

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class EnvConfig(BaseSettings):
    """
    Validates that necessary environment variables are present.
    """
    GROQ_API_KEY: str
    GOOGLE_API_KEY: str
    OPENROUTER_API_KEY: str
    
    # Load from .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class Settings:
    """
    Loads and provides access to YAML configurations.
    """
    def __init__(self, config_path: str = "config/settings.yaml"):
        with open(config_path, "r") as f:
            self._settings = yaml.safe_load(f)
        
        # Validate Env Vars
        self.env = EnvConfig()

    def get(self, key: str, default: Any = None) -> Any:
        return self._settings.get(key, default)

# Global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Configuration Error: {e}")
    raise e