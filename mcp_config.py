"""
MCP Configuration
This module provides configuration options for the Model Context Protocol (MCP) server.
It allows users to configure different LLM providers and other server settings.
"""

import os
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration for LLM providers"""
    provider: str = Field(..., description="LLM provider (e.g., 'openai', 'anthropic', 'cohere')")
    model: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3-opus-20240229')")
    api_key: Optional[str] = Field(None, description="API key for the LLM provider")
    temperature: float = Field(0.7, description="Temperature for text generation")
    max_tokens: int = Field(4000, description="Maximum tokens for text generation")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters for the LLM provider")

class ServerConfig(BaseModel):
    """Configuration for the MCP server"""
    host: str = Field("0.0.0.0", description="Host to run the server on")
    port: int = Field(8000, description="Port to run the server on")
    llm: LLMConfig = Field(..., description="LLM configuration")
    enable_rag: bool = Field(True, description="Enable RAG features")
    enable_spacy: bool = Field(True, description="Enable spaCy analysis")
    enable_lepor: bool = Field(True, description="Enable LEPOR evaluation")
    log_level: str = Field("INFO", description="Logging level")

def load_config(config_path: Optional[str] = None) -> ServerConfig:
    """
    Load configuration from a file or environment variables
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ServerConfig object
    """
    # Default configuration
    default_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "llm": {
            "provider": "openai",
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.7,
            "max_tokens": 4000
        },
        "enable_rag": True,
        "enable_spacy": True,
        "enable_lepor": True,
        "log_level": "INFO"
    }
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Merge with default config
                for key, value in file_config.items():
                    if key == "llm" and isinstance(value, dict):
                        for llm_key, llm_value in value.items():
                            default_config["llm"][llm_key] = llm_value
                    else:
                        default_config[key] = value
        except Exception as e:
            print(f"Error loading config file: {str(e)}")
    
    # Override with environment variables if present
    if os.getenv("MCP_HOST"):
        default_config["host"] = os.getenv("MCP_HOST")
    if os.getenv("MCP_PORT"):
        default_config["port"] = int(os.getenv("MCP_PORT"))
    if os.getenv("MCP_LLM_PROVIDER"):
        default_config["llm"]["provider"] = os.getenv("MCP_LLM_PROVIDER")
    if os.getenv("MCP_LLM_MODEL"):
        default_config["llm"]["model"] = os.getenv("MCP_LLM_MODEL")
    if os.getenv("MCP_LLM_API_KEY"):
        default_config["llm"]["api_key"] = os.getenv("MCP_LLM_API_KEY")
    if os.getenv("MCP_LLM_TEMPERATURE"):
        default_config["llm"]["temperature"] = float(os.getenv("MCP_LLM_TEMPERATURE"))
    if os.getenv("MCP_LLM_MAX_TOKENS"):
        default_config["llm"]["max_tokens"] = int(os.getenv("MCP_LLM_MAX_TOKENS"))
    if os.getenv("MCP_ENABLE_RAG"):
        default_config["enable_rag"] = os.getenv("MCP_ENABLE_RAG").lower() == "true"
    if os.getenv("MCP_ENABLE_SPACY"):
        default_config["enable_spacy"] = os.getenv("MCP_ENABLE_SPACY").lower() == "true"
    if os.getenv("MCP_ENABLE_LEPOR"):
        default_config["enable_lepor"] = os.getenv("MCP_ENABLE_LEPOR").lower() == "true"
    if os.getenv("MCP_LOG_LEVEL"):
        default_config["log_level"] = os.getenv("MCP_LOG_LEVEL")
    
    # Create LLMConfig object
    llm_config = LLMConfig(**default_config["llm"])
    
    # Create and return ServerConfig object
    return ServerConfig(
        host=default_config["host"],
        port=default_config["port"],
        llm=llm_config,
        enable_rag=default_config["enable_rag"],
        enable_spacy=default_config["enable_spacy"],
        enable_lepor=default_config["enable_lepor"],
        log_level=default_config["log_level"]
    )

def save_config(config: ServerConfig, config_path: str) -> bool:
    """
    Save configuration to a file
    
    Args:
        config: ServerConfig object
        config_path: Path to save the configuration to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config file: {str(e)}")
        return False

# Example configuration for Claude
CLAUDE_CONFIG = {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 4000
}

# Example configuration for OpenAI
OPENAI_CONFIG = {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 4000
}

# Example configuration for Cohere
COHERE_CONFIG = {
    "provider": "cohere",
    "model": "command",
    "api_key": os.getenv("COHERE_API_KEY"),
    "temperature": 0.7,
    "max_tokens": 4000
} 