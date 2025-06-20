from google import genai
import os
import logging
import json
import requests
import time
import random

from ibm_granite_community.notebook_utils import get_env_var
from datetime import datetime
from langchain_ollama.llms import OllamaLLM

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log")

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"

def call_llm_gemini(prompt: str, use_cache: bool = True) -> str:
    """Call Google Gemini API with retry logic for rate limits"""
    # Check cache if enabled
    if use_cache:
        cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                if prompt in cache:
                    logger.info(f"CACHE HIT: {prompt[:100]}...")
                    return cache[prompt]
            except:
                logger.warning(f"Failed to load cache, starting with empty cache")

    # Initialize Gemini client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY", "AIzaSyDFkVWREIijOQKbKhA-7B1_Ons9MhLzQ2s"),
    )
    
    # Use Flash model for better rate limits, fallback to your preferred model
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Changed from gemini-2.5-pro-exp-03-25
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt]
            )
            response_text = response.text or "No response received"
            logger.info(f"GEMINI RESPONSE: {response_text[:100]}...")
            
            # Update cache if enabled
            if use_cache:
                cache = {}
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cache = json.load(f)
                    except:
                        pass
                cache[prompt] = response_text
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                except Exception as e:
                    logger.error(f"Failed to save cache: {e}")
            
            return response_text
            
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit hit, waiting {wait_time:.2f}s before retry {attempt + 1}")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"Gemini API error: {e}")
                raise e

def call_llm_ollama(prompt: str, use_cache: bool = True) -> str:
    """Call local Ollama model"""
    model = OllamaLLM(
        model="granite3.3:8b",
        num_ctx=65536,  # 64K context window
    )
    response = model.invoke(prompt)
    logger.info(f"OLLAMA RESPONSE: {response[:100]}...")
    return response

def call_llm_openai(prompt: str, use_cache: bool = True) -> str:
    """Call OpenAI API"""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model="gpt-4o-mini",  # Cost-effective option
        messages=[{"role": "user", "content": prompt}],
    )
    response = r.choices[0].message.content
    logger.info(f"OPENAI RESPONSE: {response[:100]}...")
    return response

def call_llm_anthropic(prompt: str, use_cache: bool = True) -> str:
    """Call Anthropic Claude API"""
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=21000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_text = response.content[0].text
    logger.info(f"ANTHROPIC RESPONSE: {response_text[:100]}...")
    return response_text

def call_llm(prompt: str, use_cache: bool = True) -> str:
    """
    Main LLM calling function. 
    Set LLM_PROVIDER environment variable to choose provider:
    - 'gemini' (default): Google Gemini Flash
    - 'ollama': Local Ollama with Granite
    - 'openai': OpenAI GPT
    - 'anthropic': Claude
    """
    logger.info(f"PROMPT: {prompt[:200]}...")
    
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()  # Default to Ollama to avoid rate limits
    
    try:
        if provider == "gemini":
            return call_llm_gemini(prompt, use_cache)
        elif provider == "ollama":
            return call_llm_ollama(prompt, use_cache)
        elif provider == "openai":
            return call_llm_openai(prompt, use_cache)
        elif provider == "anthropic":
            return call_llm_anthropic(prompt, use_cache)
        else:
            logger.warning(f"Unknown provider '{provider}', falling back to Ollama")
            return call_llm_ollama(prompt, use_cache)
    except Exception as e:
        logger.error(f"Primary provider '{provider}' failed: {e}")
        # Fallback to Ollama if primary fails
        if provider != "ollama":
            logger.info("Falling back to Ollama...")
            return call_llm_ollama(prompt, use_cache)
        raise e

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    
    # Test current provider
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")