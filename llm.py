import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import openai
from typing import Any, Dict, List, Optional
import config

_client = None
_model_name = None
_full_log_file = None
_full_log_lock = threading.Lock()
_call_counter = 0


def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env()


def set_full_log(path: Path) -> None:
    global _full_log_file
    _full_log_file = open(path, "w", buffering=1)


def _log_call(label: str, messages, raw_response: str, stripped_response: str) -> None:
    global _call_counter
    if _full_log_file is None:
        return
    with _full_log_lock:
        _call_counter += 1
        sep = "=" * 70
        _full_log_file.write(f"\n{sep}\n")
        _full_log_file.write(f"CALL #{_call_counter}  [{label}]\n")
        _full_log_file.write(f"{sep}\n\n")
        for msg in messages:
            role = msg["role"].upper()
            _full_log_file.write(f"--- {role} ---\n{msg['content']}\n\n")
        if raw_response != stripped_response:
            _full_log_file.write(f"--- RAW RESPONSE (includes think tags) ---\n{raw_response}\n\n")
        _full_log_file.write(f"--- RESPONSE ---\n{stripped_response}\n\n")


def _use_openrouter() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        if _use_openrouter():
            _client = openai.OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        else:
            _client = openai.OpenAI(
                base_url=config.LOCAL_BASE_URL,
                api_key=config.LOCAL_API_KEY,
            )
    return _client


def get_model_name() -> str:
    global _model_name
    if _model_name is None:
        if _use_openrouter():
            _model_name = config.OPENROUTER_MODEL
        elif config.LOCAL_MODEL_NAME:
            _model_name = config.LOCAL_MODEL_NAME
        else:
            models = get_client().models.list()
            _model_name = models.data[0].id
    return _model_name


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks emitted by native reasoning models."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def complete(messages, max_tokens: Optional[int] = None, label: str = "") -> str:
    response = get_client().chat.completions.create(
        model=get_model_name(),
        messages=messages,
        max_tokens=max_tokens,
        extra_body={"reasoning": {"enabled": True}},
    )
    if response and response.choices and response.choices[0] and response.choices[0].message:
        raw = response.choices[0].message.content or ""
    else:
        raw = ""
    stripped = _strip_think_tags(raw)
    _log_call(label, messages, raw, stripped)
    return stripped


def complete_many(tasks, max_tokens: Optional[int] = None) -> list[str]:
    """
    Run multiple complete() calls in parallel.
    tasks: list of (messages, label) tuples — one per call.
    Returns results in the same order as tasks.
    """
    results = [""] * len(tasks)
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = {
            executor.submit(complete, messages, max_tokens, label): i
            for i, (messages, label) in enumerate(tasks)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results
