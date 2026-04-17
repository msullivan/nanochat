"""Chat completion client with swappable backends.

Backends:
- "gateway": Vercel AI Gateway (OpenAI-compatible). Requires AI_GATEWAY_API_KEY.
- "claude":  invokes the local `claude -p` CLI (Claude Code in headless mode).
             Uses whatever auth `claude` is already configured with.

Both backends return a dict shaped like {"choices": [{"message": {"content": str}}]}
so callers can treat them uniformly.

Structured output:
- Gateway backend accepts `response_format` (OpenAI json_schema style).
- Claude backend ignores `response_format`; pass `example_output` (a small JSON
  string) instead — it gets appended to the user prompt as a shape example,
  which Claude follows more reliably than schema descriptions.
"""
import json
import os
import subprocess

import requests

# Optional: load .env if python-dotenv is available. Callers can also just set
# AI_GATEWAY_API_KEY in their shell environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

AI_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"


def chat_completion(
    messages,
    *,
    model,
    response_format=None,
    example_output=None,
    temperature=1.0,
    backend="gateway",
):
    """Dispatch a chat completion to the chosen backend.

    Args:
        messages: list of {"role": ..., "content": ...} dicts.
        model: model id (gateway: "google/gemini-2.5-flash"; claude: "sonnet"/"opus"/"haiku"
            or a full model id). May be None for claude to use its default.
        response_format: gateway-only. OpenAI-style response_format (e.g. json_schema).
        example_output: claude-only. A JSON string shown to the model as a shape example.
        temperature: sampling temperature (gateway only; claude uses its own default).
        backend: "gateway" or "claude".

    Returns:
        {"choices": [{"message": {"content": <string>}}]}
    """
    if backend == "gateway":
        return _gateway_completion(
            messages,
            model=model,
            response_format=response_format,
            temperature=temperature,
        )
    elif backend == "claude":
        return _claude_completion(
            messages,
            model=model,
            example_output=example_output,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}")


def _gateway_completion(messages, *, model, response_format=None, temperature=1.0):
    api_key = os.environ.get("AI_GATEWAY_API_KEY")
    if not api_key:
        raise RuntimeError("AI_GATEWAY_API_KEY not set (check .env or environment)")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    response = requests.post(AI_GATEWAY_URL, headers=headers, json=payload)
    result = response.json()
    if "error" in result:
        raise RuntimeError(f"AI Gateway error: {result['error']}")
    return result


def _claude_completion(messages, *, model=None, example_output=None, timeout=600):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
    if not user:
        raise ValueError("claude backend requires at least one user message")

    if example_output is not None:
        user = (
            f"{user}\n\n"
            "Respond with a JSON object matching this example shape exactly "
            "(same keys, same nesting). No prose, no code fences, just the JSON:\n\n"
            f"{example_output}"
        )

    args = ["claude", "-p", "--output-format", "json"]
    if model:
        args.extend(["--model", model])
    if system:
        args.extend(["--append-system-prompt", system])
    args.append(user)

    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise RuntimeError(f"claude -p reported error: {data.get('result', data)}")

    content = _strip_code_fences(data["result"])
    return {"choices": [{"message": {"content": content}}]}


def _strip_code_fences(text):
    """Remove surrounding ```json ... ``` fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop opening ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
