import os
import json
import logging

logger = logging.getLogger(__name__)

CACHE_PATH = os.environ.get("CACHE_PATH", os.path.expanduser("~/.code_map_cache.json"))


def load_cache():
    if os.path.exists(CACHE_PATH):
        logger.debug("Loading cache from %s", CACHE_PATH)
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    logger.debug("Saving cache to %s", CACHE_PATH)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)


def get_llm_provider():
    return os.environ.get("LLM_PROVIDER", "gemini").lower()


def _call_llm_provider(prompt: str) -> str:
    provider = get_llm_provider()
    if provider == "gemini":
        return _call_llm_gemini(prompt)
    elif provider == "deepseek":
        return _call_llm_deepseek(prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def call_llm(prompt: str, use_cache: bool = True) -> str:
    if use_cache:
        cache = load_cache()
        if prompt in cache:
            logger.debug("Cache hit for prompt")
            return cache[prompt]

    result = _call_llm_provider(prompt)

    if use_cache:
        cache = load_cache()
        cache[prompt] = result
        save_cache(cache)

    return result


def _call_llm_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-pro")
    logger.debug("Using Gemini model: %s", model_name)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


def _call_llm_deepseek(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY environment variable not set.")
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    logger.debug("Using DeepSeek model: %s at %s", model, base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
