# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""OpenAI-compatible client (OpenAI, OpenRouter, Ollama) for the playlist AI.

The HTTP backend dispatched from ``tasks.ai.api`` for every non-SDK provider.
generate_text streams SSE completions; call_with_tools does single-turn
function-calling; call_with_tools_ollama tries native /api/chat tool-calling
first (Hermes template), falling back to structured JSON output on
/api/generate when native calls fail.

Main Features:
* Detects Ollama vs OpenAI shape from the URL, adds OpenRouter referer headers, and strips <think>/[/INST] reasoning tags from streamed output; generate_text maps an Ollama /api/chat URL to its /api/generate sibling.
* Robust 400 fallbacks: retries without reasoning_effort (caching rejecting models), swaps max_tokens->max_completion_tokens, and cycles DeepSeek thinking-off forms; tool-call count is capped to 4 and all failures return a generic error, never a traceback.
* Ollama dual-path: native /api/chat tool-calling (enable_thinking=false for Qwen) with structured-output format=schema fallback; tool names are validated against the registry and invalid names trigger a feedback retry.
"""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
import requests

import config
from tasks.ai.json_response import parse_json_response, response_text_and_thinking

logger = logging.getLogger(__name__)

THINK_END_TAG = "</think>"

_OLLAMA_GENERATE_PATH = "/api/generate"
_OLLAMA_CHAT_PATH = "/api/chat"
_ZEROABLE_ARGS = ("tempo_min", "tempo_max", "energy_min", "min_rating")

_MODELS_REJECTING_REASONING = set()


def _tool_function_specs(tools: List[Dict]) -> List[Dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in tools
    ]


def _is_ollama_format_url(server_url: str) -> bool:
    s = server_url.lower()
    return _OLLAMA_GENERATE_PATH in s or _OLLAMA_CHAT_PATH in s


def _ollama_endpoints(ollama_url: str):
    lowered = ollama_url.lower()
    if _OLLAMA_CHAT_PATH in lowered:
        return ollama_url, re.sub(
            re.escape(_OLLAMA_CHAT_PATH), _OLLAMA_GENERATE_PATH, ollama_url, flags=re.IGNORECASE
        )
    if _OLLAMA_GENERATE_PATH in lowered:
        return re.sub(
            re.escape(_OLLAMA_GENERATE_PATH), _OLLAMA_CHAT_PATH, ollama_url, flags=re.IGNORECASE
        ), ollama_url
    base = ollama_url.rstrip("/")
    return base + _OLLAMA_CHAT_PATH, base + _OLLAMA_GENERATE_PATH


def _build_openai_headers(api_key: str, server_url: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "no-key-needed":
        headers["Authorization"] = f"Bearer {api_key}"
    if "openrouter" in server_url.lower():
        headers["HTTP-Referer"] = "https://github.com/NeptuneHub/AudioMuse-AI"
        headers["X-Title"] = "AudioMuse-AI"
    return headers


def _safe_endpoint(url: str) -> str:
    """Remove credentials and query values before an endpoint enters logs."""
    try:
        parts = urlsplit(str(url))
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except Exception:
        return "<invalid endpoint>"


def generate_text(
    server_url: str,
    model_name: str,
    full_prompt: str,
    api_key: str = "no-key-needed",
    *,
    skip_delay: bool = False,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    structured_format: Optional[Dict | str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    is_ollama_format = _is_ollama_format_url(server_url)
    if is_ollama_format:
        return generate_text_ollama_chat(
            server_url, model_name, full_prompt, temperature=temperature,
            max_tokens=max_tokens, structured_format=structured_format,
            system_prompt=system_prompt,
        )
    is_openai_format = not is_ollama_format
    provider_label = "Ollama" if is_ollama_format else "OpenAI/OpenRouter"
    if is_ollama_format:
        server_url = _ollama_endpoints(server_url)[1]

    headers = _build_openai_headers(api_key, server_url)

    temp = 0.7 if temperature is None else float(temperature)
    out_tokens = 8000 if max_tokens is None else int(max_tokens)

    is_deepseek = "deepseek" in (model_name or "").lower()

    if is_openai_format:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": full_prompt}],
            "stream": True,
            "temperature": temp,
            "max_tokens": out_tokens,
        }
        if model_name not in _MODELS_REJECTING_REASONING:
            payload["reasoning_effort"] = "low" if is_deepseek else "none"
    else:
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": True,
            "options": {"num_predict": out_tokens, "temperature": temp},
            "think": False,
        }
        if structured_format is not None:
            payload["format"] = structured_format

    max_retries = 3
    base_delay = 5
    tried_aggressive_fallback = False
    tried_ultra_minimal_fallback = False

    for attempt in range(max_retries + 1):
        try:
            if is_openai_format and attempt == 0 and not skip_delay:
                openai_call_delay = int(os.environ.get("OPENAI_API_CALL_DELAY_SECONDS", "7"))
                if openai_call_delay > 0:
                    logger.debug(
                        "Waiting for %ss before OpenAI/OpenRouter API call to respect rate limits.",
                        openai_call_delay,
                    )
                    time.sleep(openai_call_delay)

            logger.debug(
                "Starting API call for model '%s' at '%s' (format: %s). Attempt %d/%d",
                model_name,
                _safe_endpoint(server_url) if is_ollama_format else "<configured endpoint>",
                "OpenAI" if is_openai_format else "Ollama",
                attempt + 1,
                max_retries + 1,
            )

            response = requests.post(
                server_url, headers=headers, data=json.dumps(payload), stream=True, timeout=960
            )
            if is_ollama_format:
                logger.debug(
                    "Ollama text request endpoint=%s model=%s HTTP status=%s structured=%s",
                    _safe_endpoint(server_url), model_name, response.status_code,
                    structured_format is not None,
                )
            response.raise_for_status()
            full_raw_response_content = ""
            full_thinking_content = ""
            raw_sse_lines = []

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="ignore").strip()
                raw_sse_lines.append(line_str)
                if line_str.startswith(":"):
                    continue
                if line_str.startswith("data: "):
                    line_str = line_str[6:]
                    if line_str == "[DONE]":
                        break
                try:
                    chunk = json.loads(line_str)
                    if is_openai_format:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            choice = chunk["choices"][0]
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                content = delta.get("content")
                                if content is not None:
                                    full_raw_response_content += content
                            elif "text" in choice:
                                text = choice.get("text")
                                if text is not None:
                                    full_raw_response_content += text
                            finish_reason = choice.get("finish_reason")
                            if finish_reason == "length":
                                logger.warning("Response truncated due to max_tokens limit")
                                break
                            elif finish_reason in ("stop", "tool_calls", "content_filter", "error"):
                                break
                    else:
                        if isinstance(chunk.get("response"), str):
                            full_raw_response_content += chunk["response"]
                        if isinstance(chunk.get("thinking"), str):
                            full_thinking_content += chunk["thinking"]
                        if chunk.get("done"):
                            break
                except json.JSONDecodeError:
                    logger.debug("Could not decode JSON line from stream: %s", line_str)
                    continue

            thought_enders = [THINK_END_TAG, "[/INST]", "[/THOUGHT]"]
            extracted_text = full_raw_response_content.strip()
            for end_tag in thought_enders:
                if end_tag in extracted_text:
                    extracted_text = extracted_text.split(end_tag, 1)[-1].strip()

            if is_ollama_format:
                logger.debug("Ollama raw response chunks=%r", raw_sse_lines)
                logger.debug("Ollama extracted response text=%r", extracted_text)
                logger.debug("Ollama separate thinking content=%r", full_thinking_content)
            if extracted_text:
                logger.info(
                    "%s API returned non-empty content (length=%d chars).",
                    provider_label,
                    len(extracted_text),
                )
                return extracted_text
            logger.warning(
                "%s returned empty content (raw response length: %d chars).",
                provider_label,
                len(full_raw_response_content),
            )
            logger.debug(
                "Raw SSE stream metadata: %d lines received; preview suppressed to avoid sensitive data logging.",
                len(raw_sse_lines),
            )
            if attempt < max_retries:
                sleep_time = base_delay * (2**attempt)
                logger.info("Retrying in %s seconds due to empty content...", sleep_time)
                time.sleep(sleep_time)
                continue
            return "Error: AI returned empty content after retries."

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(
                    "Rate limit exceeded (429). Attempt %d/%d", attempt + 1, max_retries + 1
                )
                if attempt < max_retries:
                    sleep_time = base_delay * (2**attempt)
                    logger.info("Retrying in %s seconds...", sleep_time)
                    time.sleep(sleep_time)
                    continue

            if e.response.status_code == 400 and is_openai_format:
                try:
                    error_body = e.response.json()
                    error_obj = error_body.get("error", {})
                    if not isinstance(error_obj, dict):
                        error_obj = {}
                    error_code = error_obj.get("code", "") or ""
                    error_param = error_obj.get("param", "") or ""
                    error_message = (error_obj.get("message", "") or "").lower()
                    if "reasoning_effort" in payload and (
                        error_param == "reasoning_effort" or "reasoning_effort" in error_message
                    ):
                        logger.info("reasoning_effort rejected (400); retrying without it")
                        payload.pop("reasoning_effort", None)
                        _MODELS_REJECTING_REASONING.add(model_name)
                        continue
                    if error_code in ("unsupported_parameter", "unsupported_value"):
                        if not tried_aggressive_fallback:
                            logger.info(
                                "Unsupported parameter detected (code: %s), switching to max_completion_tokens and removing temperature",
                                error_code,
                            )
                            payload.pop("temperature", None)
                            payload.pop("max_tokens", None)
                            payload.pop("reasoning_effort", None)
                            payload["max_completion_tokens"] = out_tokens
                            tried_aggressive_fallback = True
                            continue
                        elif not tried_ultra_minimal_fallback:
                            logger.info(
                                "Still failing with max_completion_tokens (code: %s), removing it (ultra-minimal mode)",
                                error_code,
                            )
                            payload.pop("max_completion_tokens", None)
                            tried_ultra_minimal_fallback = True
                            continue
                except (json.JSONDecodeError, KeyError, AttributeError):
                    pass

            try:
                error_detail = e.response.text
                logger.exception(
                    "Error calling OpenAI-compatible API. Response body: %s",
                    error_detail,
                )
            except Exception:
                logger.exception("Error calling OpenAI-compatible API")
            return "Error: AI service is currently unavailable."

        except requests.exceptions.RequestException:
            logger.exception("Error calling OpenAI-compatible API")
            return "Error: AI service is currently unavailable."
        except Exception:
            logger.exception(
                "An unexpected error occurred in ai_api_openai.generate_text"
            )
            return "Error: AI service is currently unavailable."

    return "Error: Max retries exceeded."


def generate_text_ollama_chat(
    server_url: str, model_name: str, full_prompt: str, *,
    temperature: Optional[float] = None, max_tokens: Optional[int] = None,
    structured_format: Optional[Dict | str] = None, system_prompt: Optional[str] = None,
) -> str:
    """Generate Ollama text through the shared native /api/chat transport."""
    chat_url, _ = _ollama_endpoints(server_url)
    timeout = config.AI_REQUEST_TIMEOUT_SECONDS
    payload = {
        "model": model_name,
        "messages": ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        + [{"role": "user", "content": full_prompt}],
        "stream": False,
        "think": False if "qwen" in (model_name or "").lower() else None,
        "options": {
            "temperature": 0.7 if temperature is None else float(temperature),
            "num_predict": 8000 if max_tokens is None else int(max_tokens),
        },
    }
    payload = {key: value for key, value in payload.items() if value is not None}
    if structured_format is not None:
        payload["format"] = structured_format
    endpoint = _safe_endpoint(chat_url)
    logger.info("Ollama text call started: provider=OLLAMA model=%s endpoint=%s timeout=%ss", model_name, endpoint, timeout)
    try:
        envelope = _ollama_chat_request(
            chat_url, model_name, payload, timeout=timeout, operation="text",
        )
        message = envelope.get("message") if isinstance(envelope, dict) else None
        content = message.get("content", "") if isinstance(message, dict) else ""
        logger.info("Ollama text response envelope: type=%s keys=%s message_type=%s", type(envelope).__name__, sorted(envelope) if isinstance(envelope, dict) else [], type(message).__name__)
        logger.info("Ollama text extracted content=%r", content)
        if isinstance(content, str) and content.strip():
            return content.strip()
        logger.error("Ollama text response had no message.content (provider=OLLAMA model=%s endpoint=%s)", model_name, endpoint)
        return "Error: AI returned empty content."
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        exception_message = str(exc)
        exception_message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", exception_message)
        exception_message = re.sub(r"(?i)(api[\s_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", exception_message)
        exception_message = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", exception_message)
        logger.error(
            "Ollama text call failed: provider=OLLAMA model=%s endpoint=%s HTTP status=%s timeout=%ss exception=%s: %s",
            model_name, endpoint, status, timeout, type(exc).__name__, exception_message,
            exc_info=True,
        )
        return f"Error: {type(exc).__name__}" + (f" HTTP {status}" if status else "") + f": {exception_message}"


def _ollama_chat_request(
    server_url: str, model_name: str, payload: Dict, *, timeout: int,
    operation: str,
) -> Dict:
    """Shared non-streaming Ollama /api/chat request used by planner and curator."""
    chat_url, _ = _ollama_endpoints(server_url)
    endpoint = _safe_endpoint(chat_url)
    message_roles = [m.get("role") for m in payload.get("messages", []) if isinstance(m, dict)]
    logger.info(
        "Ollama %s request: provider=OLLAMA model=%s endpoint=%s method=POST timeout=%ss roles=%s tools=%d format=%s",
        operation, model_name, endpoint, timeout, message_roles,
        len(payload.get("tools", [])) if isinstance(payload.get("tools"), list) else 0,
        "json" if payload.get("format") == "json" else ("schema" if isinstance(payload.get("format"), dict) else "none"),
    )
    with httpx.Client(timeout=timeout) as client:
        response = client.post(chat_url, json=payload)
        logger.info("Ollama %s HTTP status=%s endpoint=%s model=%s", operation, response.status_code, endpoint, model_name)
        try:
            response.raise_for_status()
        except Exception:
            try:
                body = response.json()
                shape = (type(body).__name__, sorted(body) if isinstance(body, dict) else None)
            except Exception:
                shape = ("non-json", None)
            logger.error("Ollama %s error response shape=%s", operation, shape)
            raise
        envelope = response.json()
    logger.info(
        "Ollama %s response envelope: type=%s keys=%s",
        operation, type(envelope).__name__, sorted(envelope) if isinstance(envelope, dict) else [],
    )
    return envelope


def call_with_tools(
    server_url: str,
    model_name: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
) -> Dict:
    try:
        functions = _tool_function_specs(tools)

        headers = _build_openai_headers(api_key, server_url)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "tools": functions,
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 1024,
        }

        is_deepseek = "deepseek" in (model_name or "").lower()
        deepseek_thinking_off_forms = [
            {"thinking": {"type": "disabled"}},
            {"thinking": "none"},
            {"thinking_mode": "non_think"},
        ]
        if is_deepseek:
            payload.update(deepseek_thinking_off_forms[0])
        elif model_name not in _MODELS_REJECTING_REASONING:
            payload["reasoning_effort"] = "none"

        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        log_messages.append(f"Using timeout: {timeout} seconds for OpenAI request")

        def _post(p):
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(server_url, headers=headers, json=p)
                resp.raise_for_status()
                return resp.json()

        try:
            result = _post(payload)
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code != 400:
                raise
            if is_deepseek:
                result = None
                for shape in deepseek_thinking_off_forms[1:]:
                    payload.pop("thinking", None)
                    payload.pop("thinking_mode", None)
                    payload.update(shape)
                    log_messages.append(
                        "DeepSeek rejected the thinking-disable parameter; retrying with an alternate form"
                    )
                    try:
                        result = _post(payload)
                        break
                    except httpx.HTTPStatusError as retry_err:
                        if retry_err.response.status_code != 400:
                            raise
                if result is None:
                    payload.pop("thinking", None)
                    payload.pop("thinking_mode", None)
                    log_messages.append(
                        "DeepSeek rejected all thinking-disable forms; retrying without them"
                    )
                    result = _post(payload)
            elif "reasoning_effort" in payload:
                log_messages.append(
                    "reasoning_effort unsupported by this model; retrying without it"
                )
                payload.pop("reasoning_effort", None)
                _MODELS_REJECTING_REASONING.add(model_name)
                result = _post(payload)
            else:
                raise

        tool_calls = []
        if "choices" in result and result["choices"]:
            message = result["choices"][0].get("message", {})
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    if tc.get("type") == "function":
                        tool_calls.append(
                            {
                                "name": tc["function"]["name"],
                                "arguments": json.loads(tc["function"]["arguments"]),
                            }
                        )

        if len(tool_calls) > 4:
            log_messages.append(f"OpenAI returned {len(tool_calls)} tool calls; capping to first 4")
            tool_calls = tool_calls[:4]

        if not tool_calls:
            text_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            log_messages.append(f"OpenAI did not call tools. Response: {text_response[:200]}")
            return {"error": "AI did not call any tools", "ai_response": text_response}

        log_messages.append(f"OpenAI called {len(tool_calls)} tools")
        return {"tool_calls": tool_calls}

    except httpx.ReadTimeout:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning(f"OpenAI request timed out after {timeout} seconds")
        log_messages.append(
            f"Request timed out after {timeout} seconds. Consider increasing AI_REQUEST_TIMEOUT_SECONDS environment variable."
        )
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        timeout = config.AI_REQUEST_TIMEOUT_SECONDS
        logger.warning("OpenAI request timed out", exc_info=True)
        log_messages.append(f"Request timed out after {timeout} seconds.")
        return {
            "error": f"Request timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception:
        logger.exception("Error calling OpenAI with tools")
        return {"error": "OpenAI service is currently unavailable."}


def _is_droppable_arg(key: str, value) -> bool:
    if value is None or value == "" or value == [] or value == {}:
        return True
    return key in _ZEROABLE_ARGS and value == 0


def _clean_call_arguments(tc: Dict, name: str, log_messages: List[str]) -> None:
    if "arguments" not in tc:
        tc["arguments"] = {}
    elif not isinstance(tc["arguments"], dict):
        log_messages.append(f"Coerced non-dict arguments for tool '{name}' to empty dict")
        tc["arguments"] = {}
    args = tc["arguments"]
    for k in [k for k, v in args.items() if _is_droppable_arg(k, v)]:
        log_messages.append(f"   Stripped empty/default arg '{k}={args[k]}' from {name}")
        del args[k]


def _validate_tool_calls(
    tool_calls: List[Dict],
    known_names: set,
    log_messages: List[str],
) -> Dict:
    """Validate and clean a list of raw tool-call dicts against the tool registry.

    Returns ``{"tool_calls": [...], "reasoning": "..."}`` on success or
    ``{"error": "..."}`` with a specific reason.
    """
    valid_calls: List[Dict] = []
    unknown_names: List[str] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict) or "name" not in tc:
            log_messages.append(f"WARN: Skipping invalid tool call: {tc}")
            continue
        name = tc.get("name", "")
        if name not in known_names:
            unknown_names.append(name)
            log_messages.append(
                f"WARN: Unknown tool '{name}' (known: {sorted(known_names)}); dropped"
            )
            continue
        _clean_call_arguments(tc, name, log_messages)
        valid_calls.append(tc)

    if unknown_names:
        msg = (
            f"Unknown tool(s) requested: {', '.join(unknown_names)}. "
            f"Use only: {', '.join(sorted(known_names))}."
        )
        log_messages.append(f"ERROR: {msg}")
        return {"error": msg}

    if not valid_calls:
        return {"error": "No valid tool calls found in Ollama response"}

    log_messages.append(f"OK: Ollama returned {len(valid_calls)} valid tool calls")
    return {"tool_calls": valid_calls}


def _strip_thinking(cleaned: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if "<think>" in cleaned:
        cleaned = (
            cleaned.split(THINK_END_TAG)[-1].strip()
            if THINK_END_TAG in cleaned
            else re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
        )
    return cleaned


def _extract_json_fence(cleaned: str) -> str:
    if "```json" in cleaned:
        cleaned = cleaned.split("```json")[1].split("```")[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```")[1].split("```")[0]
    return cleaned.strip()


def _tool_calls_from_parsed(parsed, log_messages: List[str]):
    """Return (tool_calls, error): exactly one is non-None."""
    if isinstance(parsed, dict) and "tool_calls" in parsed:
        return parsed["tool_calls"], None
    if isinstance(parsed, list):
        log_messages.append("WARN: Got array directly (expected object with tool_calls field)")
        return parsed, None
    if isinstance(parsed, dict) and "name" in parsed:
        log_messages.append("WARN: Got single tool call object (expected tool_calls array)")
        return [parsed], None
    if isinstance(parsed, dict) and "tool" in parsed and "arguments" in parsed:
        log_messages.append("WARN: Remapped {'tool','arguments'} -> {'name','arguments'} format")
        return [{"name": parsed["tool"], "arguments": parsed["arguments"]}], None
    keys = list(parsed.keys()) if isinstance(parsed, dict) else "N/A"
    log_messages.append(f"WARN: Unexpected JSON structure: {type(parsed)}, keys: {keys}")
    return None, {"error": "Ollama response missing 'tool_calls' field"}


def _parse_ollama_tool_response(response_text, log_messages: List[str], tools: List[Dict]) -> Dict:
    """Parse wrappers/prose/fenced JSON and validate Ollama planner tool calls."""
    known_names = {t.get("name") for t in tools if t.get("name")}
    logger.debug("Ollama planner raw response=%r", response_text)
    parsed, extracted, thinking, parse_error = parse_json_response(response_text)
    logger.debug(
        "Ollama planner extracted content=%r thinking_present=%s thinking=%r",
        extracted, bool(thinking.strip()), thinking,
    )
    log_messages.append(
        f"Ollama planner response parsed: thinking={'present' if thinking.strip() else 'absent'}"
    )
    if parse_error:
        reason = f"invalid JSON: {parse_error}"
        logger.warning("Ollama planner response rejected: %s", reason)
        log_messages.append(f"Ollama planner response rejected: {reason}")
        return {"error": reason}

    if isinstance(parsed, dict) and "response" in parsed and isinstance(parsed["response"], str):
        parsed, extracted, nested_thinking, parse_error = parse_json_response(parsed["response"])
        thinking = "\n".join(x for x in (thinking, nested_thinking) if x)
        if parse_error:
            reason = f"invalid nested response JSON: {parse_error}"
            log_messages.append(f"Ollama planner response rejected: {reason}")
            return {"error": reason}

    logger.debug("Ollama planner parsed JSON=%r", parsed)
    if isinstance(parsed, dict):
        reasoning = parsed.get("reasoning") or parsed.get("thinking")
    else:
        reasoning = None
    tool_calls, error = _tool_calls_from_parsed(parsed, log_messages)
    if error:
        reason = error.get("error", "unexpected planner JSON shape")
        logger.warning("Ollama planner response rejected: %s", reason)
        log_messages.append(f"Ollama planner response rejected: {reason}")
        return {"error": reason}
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    result = _validate_tool_calls(tool_calls, known_names, log_messages)
    if "tool_calls" not in result:
        reason = result.get("error", "tool-call validation failed")
        logger.warning("Ollama planner validation failed: %s", reason)
        log_messages.append(f"Ollama planner validation failed: {reason}")
        return {"error": reason}
    if isinstance(reasoning, str) and reasoning.strip():
        result["reasoning"] = reasoning.strip()
    return result


def _coerce_ollama_tool_args(raw_args, name: str, log_messages: List[str]) -> Dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            log_messages.append(
                f"WARN: Could not parse arguments for tool '{name}', using empty dict"
            )
            return {}
        return args if isinstance(args, dict) else {}
    return {}


def _try_native_ollama_tool_call(
    chat_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict],
    timeout: int,
) -> Optional[Dict]:
    """Attempt native Ollama /api/chat tool-calling (Hermes template path).

    Returns ``{"tool_calls": [...]}`` on success, or ``None`` when the model
    emitted no tool calls (caller should fall back to structured output).
    """
    from tasks.ai.prompts import build_mcp_system_prompt  # noqa: E402

    system_prompt = build_mcp_system_prompt(tools, library_context)
    ollama_tools = _tool_function_specs(tools)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "tools": ollama_tools,
        "stream": False,
        "options": {
            "temperature": config.AI_TOOLCALL_TEMPERATURE,
            "top_p": config.AI_TOOLCALL_TOP_P,
            "top_k": config.AI_TOOLCALL_TOP_K,
            "min_p": config.AI_TOOLCALL_MIN_P,
            "num_predict": config.AI_TOOLCALL_NUM_PREDICT,
        },
    }
    if "qwen" in (model_name or "").lower():
        payload["think"] = False

    expected = [{"name": t.get("name"), "parameters": t.get("inputSchema")} for t in tools]
    log_messages.append("Attempting native Ollama /api/chat tool-calling...")
    logger.debug("Ollama planner expected tool schema=%s", expected)
    result = _ollama_chat_request(
        chat_url, model_name, payload, timeout=timeout, operation="planner",
    )

    logger.debug("Ollama planner native raw response wrapper=%r", result)
    message = result.get("message", {})
    if not isinstance(message, dict):
        message = {}
    thinking = message.get("thinking") or result.get("thinking") or result.get("reasoning") or ""
    if thinking:
        logger.debug("Ollama planner native separate thinking/reasoning=%r", thinking)
        log_messages.append("Ollama native response included separate thinking/reasoning content")
    raw_tool_calls = message.get("tool_calls") or []

    if not raw_tool_calls:
        content = message.get("content", "")
        logger.debug("Ollama planner native extracted response/content=%r", content)
        log_messages.append(
            f"Native tool-calling returned 0 tool calls; response content length={len(content)}"
        )
        if content:
            parsed_content = _parse_ollama_tool_response(content, log_messages, tools)
            if "tool_calls" in parsed_content:
                if thinking and not parsed_content.get("reasoning"):
                    parsed_content["reasoning"] = thinking
                return parsed_content
        return None

    tool_calls: List[Dict] = []
    for tc in raw_tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name", "")
        tool_calls.append(
            {"name": name, "arguments": _coerce_ollama_tool_args(fn.get("arguments"), name, log_messages)}
        )

    known_names = {t.get("name") for t in tools if t.get("name")}
    validated = _validate_tool_calls(tool_calls, known_names, log_messages)
    if "tool_calls" in validated:
        log_messages.append(
            f"OK: Native tool-calling returned {len(validated['tool_calls'])} tool(s)"
        )
        return validated
    log_messages.append(
        f"Native tool-calling validation failed: {validated.get('error', 'unknown')}"
    )
    return None


def _try_structured_ollama_call(
    generate_url: str,
    model_name: str,
    prompt: str,
    tools: List[Dict],
    log_messages: List[str],
    timeout: int,
) -> Dict:
    """Fallback: prompt-based JSON emission constrained by format=<schema>."""
    from tasks.ai.prompts import build_tool_calls_schema  # noqa: E402

    schema = build_tool_calls_schema(tools)
    logger.debug(
        "Ollama planner expected structured schema=%s",
        json.dumps(schema, ensure_ascii=False, sort_keys=True),
    )
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "think": False,
        "options": {
            "temperature": config.AI_TOOLCALL_TEMPERATURE,
            "top_p": config.AI_TOOLCALL_TOP_P,
            "top_k": config.AI_TOOLCALL_TOP_K,
            "min_p": config.AI_TOOLCALL_MIN_P,
            "num_predict": config.AI_TOOLCALL_NUM_PREDICT,
        },
    }
    logger.debug("Ollama planner endpoint=%s model=%s path=structured", _safe_endpoint(generate_url), model_name)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(generate_url, json=payload)
        logger.debug("Ollama planner structured HTTP status=%s endpoint=%s model=%s", response.status_code, _safe_endpoint(generate_url), model_name)
        response.raise_for_status()
        result = response.json()

    logger.debug("Ollama planner structured raw response wrapper=%r", result)
    response_text, thinking = response_text_and_thinking(result)
    if not response_text:
        reason = f"Ollama response had no response/content field; wrapper keys={sorted(result) if isinstance(result, dict) else type(result).__name__}"
        log_messages.append(f"Ollama planner response rejected: {reason}")
        return {"error": reason}
    logger.debug("Ollama planner extracted response/content=%r separate_thinking=%r", response_text, thinking)
    parsed_result = _parse_ollama_tool_response(response_text, log_messages, tools)
    if thinking and "tool_calls" in parsed_result and not parsed_result.get("reasoning"):
        parsed_result["reasoning"] = thinking
    return parsed_result


def call_with_tools_ollama(
    ollama_url: str,
    model_name: str,
    user_message: str,
    tools: List[Dict],
    log_messages: List[str],
    library_context: Optional[Dict] = None,
) -> Dict:
    """Single-turn tool-calling for Ollama with native API first, structured fallback.

    Strategy (two paths, one feedback retry each):
    1. PRIMARY: Native /api/chat with ``tools`` parameter (Hermes template Qwen
       was trained on). Disables thinking via ``enable_thinking=false``.
    2. FALLBACK: Prompt-based JSON emission on /api/generate constrained by
       ``format=<schema>``, used when native path returns no tool calls or when
       the user's URL is already a /api/generate endpoint.
    """
    from tasks.ai.prompts import build_ollama_tool_calling_prompt  # noqa: E402

    is_generate_url = _OLLAMA_GENERATE_PATH in ollama_url.lower()
    chat_url, generate_url = _ollama_endpoints(ollama_url)

    timeout = config.AI_REQUEST_TIMEOUT_SECONDS
    logger.debug(
        "Ollama planner configured endpoint=%s model=%s expected_tool_schema=%s",
        _safe_endpoint(ollama_url), model_name,
        json.dumps([{"name": t.get("name"), "parameters": t.get("inputSchema")} for t in tools], ensure_ascii=False),
    )
    log_messages.append(f"Ollama planner model: {model_name}; endpoint: {_safe_endpoint(ollama_url)}")
    log_messages.append(f"Using timeout: {timeout} seconds for Ollama request")

    if not is_generate_url:
        try:
            result = _try_native_ollama_tool_call(
                chat_url, model_name, user_message, tools,
                log_messages, library_context, timeout,
            )
            if result is not None and "tool_calls" in result:
                return result
            log_messages.append("Native path returned no tool calls; falling back to format=schema")
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            reason = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
            logger.warning("Native Ollama tool-calling failed (%s); trying structured output", reason)
            log_messages.append(f"Native /api/chat failed: {reason}; trying format=schema")

    log_messages.append("Using Ollama structured-output (format=schema) path")
    try:
        base_prompt = build_ollama_tool_calling_prompt(user_message, tools, library_context)
        prompt = base_prompt
        last_result: Dict = {"error": "Ollama returned no usable tool calls"}
        for attempt in range(2):
            last_result = _try_structured_ollama_call(
                generate_url, model_name, prompt, tools,
                log_messages, timeout,
            )
            if "tool_calls" in last_result:
                if attempt > 0:
                    log_messages.append("OK: structured retry with feedback produced a valid plan")
                return last_result
            if attempt == 0:
                err = last_result.get("error", "invalid output")
                log_messages.append(f"Retrying once with feedback: {err}")
                prompt = (
                    f"{base_prompt}\n\nYour previous reply was invalid ({err}). "
                    "Return ONLY the JSON object in the required shape."
                )
        return last_result

    except httpx.ReadTimeout:
        log_messages.append(
            f"Ollama request timed out after {timeout} seconds. Your model or hardware may be too slow."
        )
        log_messages.append(
            "TIP: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value (e.g., 600 for 10 minutes)"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except httpx.TimeoutException:
        log_messages.append(f"Ollama request timed out after {timeout} seconds.")
        log_messages.append(
            "TIP: Set AI_REQUEST_TIMEOUT_SECONDS environment variable to a higher value"
        )
        return {
            "error": f"Ollama timed out after {timeout} seconds. Increase AI_REQUEST_TIMEOUT_SECONDS for slower hardware or larger models."
        }
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        reason = f"{type(exc).__name__}" + (f" (HTTP {status})" if status else "")
        logger.error("Error calling Ollama with tools (%s)", reason)
        log_messages.append(f"Ollama planner request failed: {reason}")
        return {"error": reason}
