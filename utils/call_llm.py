from google import genai
import os
import logging
import json
import requests
from datetime import datetime

# Configure logging
log_directory = os.getenv("LOG_DIR", "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(
    log_directory, f"llm_calls_{datetime.now().strftime('%Y%m%d')}.log"
)

# Set up logger
logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent propagation to root logger
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# Simple cache configuration
cache_file = "llm_cache.json"


def load_cache():
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except:
        logger.warning(f"Failed to load cache.")
    return {}


def save_cache(cache):
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except:
        logger.warning(f"Failed to save cache")


def get_llm_provider():
    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        if os.getenv("OPENROUTER_API_KEY"):
            provider = "OPENROUTER"
        elif os.getenv("GEMINI_PROJECT_ID") or os.getenv("GEMINI_API_KEY"):
            provider = "GEMINI"
    return provider


def _call_llm_provider(prompt: str) -> str:
    """
    Call an LLM provider based on environment variables.
    Environment variables:
    - LLM_PROVIDER: "OLLAMA", "XAI", "OPENROUTER", etc.
    - <provider>_MODEL: Model name (e.g., OLLAMA_MODEL, OPENROUTER_MODEL)
    - <provider>_BASE_URL: Base URL without endpoint (e.g., OLLAMA_BASE_URL, OPENROUTER_BASE_URL)
    - <provider>_API_KEY: API key (e.g., OLLAMA_API_KEY, OPENROUTER_API_KEY)
    """
    import time
    import random

    provider = get_llm_provider()
    if not provider:
        raise ValueError("No LLM provider resolved")

    model_var = f"{provider}_MODEL"
    base_url_var = f"{provider}_BASE_URL"
    api_key_var = f"{provider}_API_KEY"

    model = os.environ.get(model_var)
    if not model and provider == "OPENROUTER":
        model = "google/gemini-2.5-flash"  # sensible default for OpenRouter
        
    base_url = os.environ.get(base_url_var)
    if not base_url and provider == "OPENROUTER":
        base_url = "https://openrouter.ai/api/v1"

    api_key = os.environ.get(api_key_var, "")

    if not model:
        raise ValueError(f"{model_var} environment variable is required")
    if not base_url:
        raise ValueError(f"{base_url_var} environment variable is required")

    base_url_stripped = base_url.rstrip('/')
    if base_url_stripped.endswith("/v1"):
        url = f"{base_url_stripped}/chat/completions"
    else:
        url = f"{base_url_stripped}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if provider == "OPENROUTER":
        headers["HTTP-Referer"] = "https://github.com/google/pocketflow"
        headers["X-Title"] = "PocketFlow Tutorial Builder"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 3000,
    }

    max_retries = 5
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response_json = response.json()
            logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
            response.raise_for_status()
            return response_json["choices"][0]["message"]["content"]
        except Exception as e:
            err_msg = str(e).lower()
            is_retryable = any(
                code in err_msg for code in ["429", "502", "503", "504", "rate limit", "timeout", "exhausted"]
            )
            if is_retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Error calling {provider} ({e}). Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to generate content from {provider} after {attempt+1} attempts: {e}")
                raise e

# By default, we Google Gemini 2.5 pro, as it shows great performance for code understanding
def call_llm(prompt: str, use_cache: bool = True) -> str:
    # Log the prompt
    logger.info(f"PROMPT: {prompt}")

    # Check cache if enabled
    if use_cache:
        # Load cache from disk
        cache = load_cache()
        # Return from cache if exists
        if prompt in cache:
            logger.info(f"RESPONSE: {cache[prompt]}")
            return cache[prompt]

    provider = get_llm_provider()
    if provider == "GEMINI":
        response_text = _call_llm_gemini(prompt)
    else:  # generic method using a URL that is OpenAI compatible API (Ollama, ...)
        response_text = _call_llm_provider(prompt)

    # Log the response
    logger.info(f"RESPONSE: {response_text}")

    # Update cache if enabled
    if use_cache:
        # Load cache again to avoid overwrites
        cache = load_cache()
        # Add to cache and save
        cache[prompt] = response_text
        save_cache(cache)

    return response_text


def _call_llm_gemini(prompt: str) -> str:
    import time
    import random

    if os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    elif os.getenv("GEMINI_PROJECT_ID"):
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1")
        )
    else:
        raise ValueError("Either GEMINI_PROJECT_ID or GEMINI_API_KEY must be set in the environment")
    
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    max_retries = 10
    base_delay = 5.0
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg or "rate limit" in err_msg:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate content after {max_retries} attempts: {e}")
                    raise e
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Rate limit exceeded (429/Resource Exhausted). Retrying in {delay:.2f} seconds... Error: {e}")
                time.sleep(delay)
            else:
                raise e

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
