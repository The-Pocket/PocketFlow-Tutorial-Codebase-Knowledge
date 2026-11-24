import os
import logging
import json
import shutil
import signal
import subprocess
import tempfile
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
    - <provider>_BASE_URL: Base URL without endpoint (e.g., OLLAMA_BASE_URL, XAI_BASE_URL)
    - <provider>_API_KEY: API key (e.g., OLLAMA_API_KEY, XAI_API_KEY; optional for providers that don't require it)
    The endpoint /v1/chat/completions will be appended to the base URL.
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

    # Append the endpoint to the base URL
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

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

    try:
        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json() # Log the response
        logger.info("RESPONSE:\n%s", json.dumps(response_json, indent=2))
        #logger.info(f"RESPONSE: {response.json()}")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_message = f"HTTP error occurred: {e}"
        try:
            error_details = response.json().get("error", "No additional details")
            error_message += f" (Details: {error_details})"
        except:
            pass
        raise Exception(error_message)
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to {provider} API. Check your network connection.")
    except requests.exceptions.Timeout:
        raise Exception(f"Request to {provider} API timed out.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"An error occurred while making the request to {provider}: {e}")
    except ValueError:
        raise Exception(f"Failed to parse response as JSON from {provider}. The server might have returned an invalid response.")

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
    elif provider == "COPILOT_CLI":
        response_text = _call_llm_copilot_cli(prompt)
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
    from google import genai  # Only needed for this provider

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


# Used if COPILOT_CLI_MODEL is not set. The first prompts contain the whole code base,
# so the default model is combined with its large context window.
COPILOT_DEFAULT_MODEL = "claude-sonnet-5.5"
COPILOT_DEFAULT_CONTEXT = "long_context"

# The Copilot CLI is an agent whose own system prompt asks for short chat answers
# ("100 words or less"). It cannot be replaced, so every prompt states explicitly
# that the answer is a document. It is repeated at the end because the code context
# in between can be several hundred KB long.
COPILOT_PROMPT_PREFIX = """The answer to this message is written to a file by a script; it is not a chat reply.
Answer in full from the text below alone, as long as the content requires. Guidance to keep chat replies short does not apply to this document.
Do not use tools, do not announce what you are going to do, and return only the requested output.

"""

COPILOT_PROMPT_SUFFIX = """

Reminder: return the complete document in exactly the requested format, without any introduction or closing remark. There is no word limit.
"""


def _find_copilot() -> str:
    configured = os.getenv("COPILOT_CLI_PATH")
    executable = shutil.which(configured or "copilot")
    if not executable:
        raise FileNotFoundError(
            f"GitHub Copilot CLI not found ({configured or 'copilot'}). "
            "Install it or set COPILOT_CLI_PATH to the executable."
        )
    return os.path.abspath(executable)


def _kill_process_tree(process):
    # Killing only the direct child is not enough: copilot may be a launcher script, and
    # a surviving grandchild keeps the output pipes open and the working directory locked.
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_copilot(command, prompt: str, timeout: float):
    env = os.environ.copy()
    # GITHUB_TOKEN is used by this tool to crawl repositories. The Copilot CLI would prefer it
    # over the stored login. Use COPILOT_GITHUB_TOKEN to pass a token to Copilot on purpose.
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "COPILOT_ALLOW_ALL"):
        env.pop(name, None)

    # An empty working directory: no AGENTS.md, no repository, and "@name" in source code
    # (attributes, decorators, npm scopes) cannot be mistaken for a reference to a file.
    with tempfile.TemporaryDirectory(
        prefix="copilot_", ignore_cleanup_errors=True
    ) as working_directory:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_directory,
            env=env,
            start_new_session=os.name != "nt",
        )
        try:
            # The prompt goes through stdin. As an argument it would exceed the limit
            # of the operating system (about 32,000 characters on Windows).
            stdout, stderr = process.communicate(prompt.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            raise TimeoutError(
                f"GitHub Copilot CLI did not answer within {timeout:g} seconds. "
                "Increase COPILOT_CLI_TIMEOUT if the model needs more time."
            )
        except BaseException:
            _kill_process_tree(process)
            raise

    return (
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace").strip(),
    )


def _call_llm_copilot_cli(prompt: str) -> str:
    """
    Call the locally installed GitHub Copilot CLI instead of an HTTP endpoint.
    Uses the account that is logged in to the CLI (run `copilot login` once).
    Environment variables (all optional):
    - COPILOT_CLI_PATH: Executable, if `copilot` is not in PATH
    - COPILOT_CLI_MODEL: Model name, e.g. gpt-6-luna (default: claude-sonnet-5.5)
    - COPILOT_CLI_CONTEXT: Context window tier, "default" or "long_context"
      (default: long_context for the default model, otherwise decided by the CLI)
    - COPILOT_CLI_REASONING_EFFORT: Reasoning effort, e.g. "medium"
    - COPILOT_CLI_TIMEOUT: Timeout per call in seconds (default: 1200)
    """
    command = [
        _find_copilot(),
        # Text output is rendered for the terminal and loses headings, code fences and
        # mermaid blocks. Only the JSON events contain the unchanged Markdown.
        "--output-format", "json",
        "--stream", "off",
        # An empty allowlist is ignored by the CLI. A name that does not exist disables all tools.
        "--available-tools=none",
        "--disable-builtin-mcps",
        "--no-custom-instructions",
        "--no-ask-user",
    ]
    model = os.getenv("COPILOT_CLI_MODEL")
    context = os.getenv("COPILOT_CLI_CONTEXT")
    if not model:
        model = COPILOT_DEFAULT_MODEL
        context = context or COPILOT_DEFAULT_CONTEXT

    for option, value in (
        ("--model", model),
        ("--context", context),
        ("--reasoning-effort", os.getenv("COPILOT_CLI_REASONING_EFFORT")),
    ):
        if value:
            command += [option, value]

    timeout = float(os.getenv("COPILOT_CLI_TIMEOUT") or 1200)
    exit_code, stdout, stderr = _run_copilot(
        command, COPILOT_PROMPT_PREFIX + prompt + COPILOT_PROMPT_SUFFIX, timeout
    )

    messages = []
    errors = []
    # Not splitlines(): JSON strings may contain unescaped U+2028 and similar separators.
    for line in stdout.split("\n"):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        data = event.get("data") or {}
        if event.get("type") == "assistant.message" and data.get("content"):
            messages.append(data["content"])
        elif event.get("type") == "session.error":
            errors.append(data.get("message", "Unknown error"))
        elif event.get("type") == "result":
            logger.info(f"COPILOT USAGE: {json.dumps(event.get('usage'))}")

    if exit_code != 0 or not messages:
        details = "; ".join(errors) or stderr or "no answer received"
        if "authenticat" in details.lower():
            # The advice of the CLI to set one of these variables does not work here.
            details += (
                " Note: GITHUB_TOKEN and GH_TOKEN are not passed on to Copilot. "
                "Run `copilot login` or set COPILOT_GITHUB_TOKEN."
            )
        raise Exception(f"GitHub Copilot CLI failed (exit code {exit_code}): {details}")

    if len(messages) > 1:
        logger.warning(f"Copilot returned {len(messages)} messages, using the last one.")
    return messages[-1]


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"

    # First call - should hit the API
    print("Making call...")
    response1 = call_llm(test_prompt, use_cache=False)
    print(f"Response: {response1}")
