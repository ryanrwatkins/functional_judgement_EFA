"""LLM client utilities for interfacing with different language models."""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import requests
from anthropic import Anthropic
from openai import OpenAI

from config import MODELS
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Custom exception for LLM client operations."""
    pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = MODELS.get(model_name, {})
        self.config.update(kwargs)
        self.request_count = 0
        self.total_tokens = 0
    
    @abstractmethod
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        pass
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get usage statistics for this client."""
        return {
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.request_count = 0
        self.total_tokens = 0


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI GPT models."""
    
    def __init__(self, model_name: str = "gpt-4", **kwargs):
        """Initialize OpenAI client."""
        super().__init__(model_name, **kwargs)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client for model: {model_name}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response using OpenAI API."""
        try:
            max_tokens = max_tokens or self.config.get("max_tokens", 1000)
            temperature = temperature or self.config.get("temperature", 0.7)
            
            response = self.client.chat.completions.create(
                model=self.config.get("model_name", self.model_name),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds to psychological assessment items as instructed."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.request_count += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMClientError(f"OpenAI API error: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, model_name: str = "claude", **kwargs):
        """Initialize Anthropic client."""
        super().__init__(model_name, **kwargs)
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMClientError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client for model: {model_name}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response using Anthropic API."""
        try:
            max_tokens = max_tokens or self.config.get("max_tokens", 1000)
            temperature = temperature or self.config.get("temperature", 0.7)
            
            response = self.client.messages.create(
                model=self.config.get("model_name", "claude-3-sonnet-20240229"),
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            self.request_count += 1
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise LLMClientError(f"Anthropic API error: {str(e)}")


class LlamaClient(BaseLLMClient):
    """Client for local Llama models."""
    
    def __init__(self, model_name: str = "llama", base_url: str = "http://localhost:11434", **kwargs):
        """Initialize Llama client."""
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        logger.info(f"Initialized Llama client for model: {model_name} at {base_url}")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response using local Llama model via Ollama API."""
        try:
            max_tokens = max_tokens or self.config.get("max_tokens", 1000)
            temperature = temperature or self.config.get("temperature", 0.7)
            
            payload = {
                "model": self.config.get("model_name", "llama3"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()
            
            result = response.json()
            self.request_count += 1
            
            # Estimate tokens (rough approximation)
            estimated_tokens = len(prompt.split()) + len(result.get("response", "").split())
            self.total_tokens += estimated_tokens
            
            return result.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Llama API error: {str(e)}")
            raise LLMClientError(f"Llama API error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error with Llama client: {str(e)}")
            raise LLMClientError(f"Llama client error: {str(e)}")


def create_llm_client(model_name: str, **kwargs) -> BaseLLMClient:
    """
    Factory function to create appropriate LLM client.
    
    Args:
        model_name: Name of the model ("gpt-4", "claude", "llama")
        **kwargs: Additional configuration parameters
        
    Returns:
        Appropriate LLM client instance
    """
    if model_name == "gpt-4":
        return OpenAIClient(model_name, **kwargs)
    elif model_name == "claude":
        return AnthropicClient(model_name, **kwargs)
    elif model_name == "llama":
        return LlamaClient(model_name, **kwargs)
    else:
        raise LLMClientError(f"Unsupported model: {model_name}")


class RateLimitedClient:
    """Wrapper to add rate limiting to LLM clients."""
    
    def __init__(self, client: BaseLLMClient, requests_per_minute: int = 60):
        """
        Initialize rate limited client.
        
        Args:
            client: The LLM client to wrap
            requests_per_minute: Maximum requests per minute
        """
        self.client = client
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        
        logger.info(f"Added rate limiting: {requests_per_minute} requests/minute")
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response with rate limiting."""
        # Implement rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        # Make the request
        response = self.client.generate_response(prompt, **kwargs)
        self.last_request_time = time.time()
        
        return response
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get stats from underlying client."""
        return self.client.get_stats()
    
    def reset_stats(self) -> None:
        """Reset stats on underlying client."""
        self.client.reset_stats()