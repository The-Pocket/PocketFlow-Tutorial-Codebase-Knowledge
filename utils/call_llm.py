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
    if not provider and (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GEMINI_API_KEY")):
        provider = "GEMINI"
    # if necessary, add ANTHROPIC/OPENAI
    return provider


def _call_llm_provider(prompt: str) -> str:
    """
    Call an LLM provider based on environment variables.
    Environment variables:
    - LLM_PROVIDER: "OLLAMA" or "XAI"
    - <provider>_MODEL: Model name (e.g., OLLAMA_MODEL, XAI_MODEL)
    - <provider>_BASE_URL: Base URL of the API. May include or omit a trailing /v1
      (e.g., OLLAMA_BASE_URL=http://localhost:11434 or http://localhost:11434/v1,
             XAI_BASE_URL=https://api.x.ai/v1 or https://openrouter.ai/api/v1)
    - <provider>_API_KEY: API key (e.g., OLLAMA_API_KEY, XAI_API_KEY; optional for providers that don't require it)
    The /chat/completions endpoint is appended; /v1 is added only if not already present in the URL.
    """
    logger.info(f"PROMPT: {prompt}") # log the prompt

    # Read the provider from environment variable
    provider = os.environ.get("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    # Construct the names of the other environment variables
    model_var = f"{provider}_MODEL"
    base_url_var = f"{provider}_BASE_URL"
    api_key_var = f"{provider}_API_KEY"

    # Read the provider-specific variables
    model = os.environ.get(model_var)
    base_url = os.environ.get(base_url_var)
    api_key = os.environ.get(api_key_var, "")  # API key is optional, default to empty string

    # Validate required variables
    if not model:
        raise ValueError(f"{model_var} environment variable is required")
    if not base_url:
        raise ValueError(f"{base_url_var} environment variable is required")

    # Build the chat completions URL. Avoid duplicating /v1 when the caller
    # already includes it in BASE_URL (e.g. https://openrouter.ai/api/v1).
    clean_base = base_url.rstrip('/')
    if clean_base.endswith('/v1'):
        url = f"{clean_base}/chat/completions"
    else:
        url = f"{clean_base}/v1/chat/completions"

    # Configure headers and payload based on provider
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:  # Only add Authorization header if API key is provided
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    response_json = None
    try:
        response = requests.post(url, headers=headers, json=payload)
        # Parse JSON first so we can log it and include error details on failure
        try:
            response_json = response.json()
            logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
        except (ValueError, requests.exceptions.JSONDecodeError):
            logger.warning(
                "Non-JSON response from %s (HTTP %s): %s",
                provider, response.status_code, response.text[:500]
            )
        response.raise_for_status()
        if response_json is None:
            raise Exception(
                f"Empty or non-JSON response from {provider} (HTTP {response.status_code}). "
                f"Verify that {base_url_var} points to a valid OpenAI-compatible endpoint."
            )
        return response_json["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP error occurred: {e}"
        if response_json is not None:
            error_details = response_json.get("error", "No additional details")
            error_message += f" (Details: {error_details})"
        elif response.text:
            error_message += f" (Response: {response.text[:200]})"
        raise Exception(error_message)
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to {provider} API. Check your network connection.")
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {provider} API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while making the request to {provider}: {e}")

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
    if os.getenv("GEMINI_PROJECT_ID"):
        client = genai.Client(
            vertexai=True,
            project=os.getenv("GEMINI_PROJECT_ID"),
            location=os.getenv("GEMINI_LOCATION", "us-central1")
        )
    elif os.getenv("GEMINI_API_KEY"):
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    else:
        raise ValueError("Either GEMINI_PROJECT_ID or GEMINI_API_KEY must be set in the environment")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro-exp-03-25")
    response = client.models.generate_content(
        model=model,
        contents=[prompt]
    )
    return response.text

if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
