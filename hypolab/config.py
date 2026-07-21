"""Configuration and environment handling for HypoLab."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HypoLabConfig:
    """Centralized configuration with graceful degradation."""
    
    # LLM Configuration
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    
    # Ollama Local LLM
    use_ollama: bool = False
    ollama_model: str = "llama3.1:8b"
    ollama_url: str = "http://localhost:11434/api/generate"
    
    # Local HF Model (Kaggle/free GPU fallback)
    use_local_model: bool = False
    local_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    local_model_device: str = "auto"
    local_max_tokens: int = 2048
    
    # Statistical Testing
    significance_level: float = 0.05
    max_correlations: int = 50
    
    # Literature Search
    arxiv_max_results: int = 5
    
    # Pipeline
    verbose: bool = True
    
    def __post_init__(self):
        """Load from environment variables if available."""
        self.groq_api_key = os.getenv("GROQ_API_KEY", self.groq_api_key)
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        self.use_local_model = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
        self.significance_level = float(os.getenv("SIGNIFICANCE_LEVEL", self.significance_level))
        
    def has_groq(self) -> bool:
        """Check if Groq API key is available."""
        return self.groq_api_key is not None and len(self.groq_api_key) > 0
    
    def has_ollama(self) -> bool:
        """Check if Ollama is configured."""
        return self.use_ollama