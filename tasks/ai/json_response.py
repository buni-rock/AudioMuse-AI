"""Helpers for extracting JSON from model responses and provider wrappers."""

import json
import re


def ollama_response_shape(value):
    """Describe Ollama response fields without exposing content or reasoning text."""
    envelope = value if isinstance(value, dict) else {}
    message = envelope.get("message") if isinstance(envelope.get("message"), dict) else {}
    content = message.get("content")
    response_content = message.get("response")
    thinking = (
        message.get("thinking") or message.get("reasoning")
        or envelope.get("thinking") or envelope.get("reasoning") or ""
    )
    tool_calls = message.get("tool_calls") or envelope.get("tool_calls")
    return {
        "top_level_keys": sorted(str(key) for key in envelope),
        "message_keys": sorted(str(key) for key in message),
        "content_type": type(content).__name__ if content is not None else "NoneType",
        "content_length": len(content) if isinstance(content, str) else None,
        "response_type": type(response_content).__name__ if response_content is not None else "NoneType",
        "response_length": len(response_content) if isinstance(response_content, str) else None,
        "thinking_type": type(thinking).__name__ if thinking is not None else "NoneType",
        "thinking_length": len(thinking) if isinstance(thinking, str) else None,
        "tool_calls": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "done": envelope.get("done"),
        "done_reason": envelope.get("done_reason"),
        "model": envelope.get("model"),
        "prompt_eval_count": envelope.get("prompt_eval_count"),
        "eval_count": envelope.get("eval_count"),
        "error": envelope.get("error"),
        "additional_top_level_keys": sorted(
            str(key) for key in envelope
            if key not in {
                "created_at", "done", "done_reason", "eval_count", "eval_duration",
                "load_duration", "message", "model", "prompt_eval_count",
                "prompt_eval_duration", "total_duration", "error",
            }
        ),
    }


def response_text_and_thinking(value):
    """Unwrap common Ollama/OpenAI response shapes into content and thinking."""
    thinking = ""
    current = value
    for _ in range(4):
        if isinstance(current, str):
            return current, thinking
        if not isinstance(current, dict):
            return "", thinking
        for key in ("thinking", "reasoning"):
            part = current.get(key)
            if isinstance(part, str) and part.strip():
                thinking = "\n".join(p for p in (thinking, part.strip()) if p)
        message = current.get("message")
        if isinstance(message, dict):
            current = message.get("content") or message.get("response") or ""
            continue
        choices = current.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or choice.get("message") or choice
            if isinstance(delta, dict):
                current = delta.get("content") or delta.get("text") or ""
                continue
        for key in ("response", "content", "text", "output"):
            part = current.get(key)
            if isinstance(part, (str, dict)):
                current = part
                break
        else:
            return "", thinking
    return current if isinstance(current, str) else "", thinking


def extract_json_text(text):
    """Extract one JSON object/array, tolerating fences and safe surrounding prose."""
    if not isinstance(text, str):
        return "", "response content is not text"
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    cleaned = cleaned.replace("</think>", "").strip()

    fences = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    candidates = [block.strip() for block in fences if block.strip()]
    candidates.append(cleaned)
    decoder = json.JSONDecoder()
    last_error = "no JSON object or array found"
    for candidate in candidates:
        try:
            decoder.raw_decode(candidate)
            return candidate, None
        except json.JSONDecodeError as exc:
            last_error = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        for index, char in enumerate(candidate):
            if char not in "{[":
                continue
            try:
                _, end = decoder.raw_decode(candidate[index:])
                return candidate[index:index + end], None
            except json.JSONDecodeError as exc:
                last_error = f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
    return "", last_error


def parse_json_response(value):
    """Return (parsed, extracted_text, thinking, error) for provider output."""
    text, thinking = response_text_and_thinking(value)
    if not text.strip():
        return None, text, thinking, "response content is empty"
    extracted, error = extract_json_text(text)
    if error:
        return None, extracted, thinking, error
    try:
        return json.loads(extracted), extracted, thinking, None
    except (ValueError, TypeError) as exc:
        return None, extracted, thinking, f"JSON decoding failed: {exc}"
