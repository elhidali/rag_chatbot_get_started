"""Module for interfacing with OpenRouter LLMs and HuggingFace Embeddings."""

import logging
from typing import Dict, Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter

from transformers import AutoModel, AutoTokenizer

import config

logger = logging.getLogger(__name__)

def create_embedding() -> HuggingFaceEmbedding:
    """Creates a HuggingFace Embedding model for vector representation.
    
    Returns:
        HuggingFaceEmbedding model.
    """
    model_name = config.EMBEDDING_MODEL_ID
    
    # Load model and tokenizer manually to bypass llama-index's safe_serialization bug
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    embedding = HuggingFaceEmbedding(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer
    )
    logger.info(f"Created HuggingFace Embedding model: {model_name}")
    return embedding

def create_llm(
    temperature: float = config.TEMPERATURE,
    max_new_tokens: int = config.MAX_NEW_TOKENS,
    decoding_method: str = "sample"
) -> OpenRouter:
    """Creates an OpenRouter LLM for generating responses.
    
    Args:
        temperature: Temperature for controlling randomness in generation (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.
        decoding_method: Decoding method to use (sample, greedy).
        
    Returns:
        OpenRouter model.
    """
    import httpx
    llm = OpenRouter(
        model=config.LLM_MODEL_ID,
        api_key=config.OPENROUTER_API_KEY,
        api_base=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_new_tokens,
        http_client=httpx.Client(verify=False)
    )
    
    logger.info(f"Created OpenRouter LLM model: {config.LLM_MODEL_ID}")
    return llm

def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model to use.
    
    Args:
        new_model_id: New LLM model ID to use.
    """
    global config
    config.LLM_MODEL_ID = new_model_id
    logger.info(f"Changed LLM model to: {new_model_id}")