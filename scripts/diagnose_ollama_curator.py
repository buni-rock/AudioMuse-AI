#!/usr/bin/env python3
"""Compare Ollama curator JSON responses across thinking and schema settings.

Run from the repository root, for example:
    python3 scripts/diagnose_ollama_curator.py --url http://192.168.144.8:11434

Only response shape and JSON validity are printed; prompt, content, and
reasoning text are deliberately omitted.
"""

import argparse
import json
import time
import urllib.error
import urllib.request


IDS = ["A1", "B2", "C3", "D4", "E5"]
PROMPT = (
    "Rank these five real candidate IDs from most appropriate to least for a playlist. "
    "Return JSON only with ranked_ids, using each supplied ID no more than once: "
    + ", ".join(IDS)
)
SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ranked_ids"],
}
MATRIX = [
    ("think_false_schema", False, True),
    ("think_true_schema", True, True),
    ("think_omitted_schema", None, True),
    ("think_false_no_schema", False, False),
    ("think_omitted_no_schema", None, False),
]


def _post_json(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.load(response)


def _model_info(base_url, model, timeout):
    status, body = _post_json(
        f"{base_url}/api/show", {"model": model}, timeout,
    )
    details = body.get("details") if isinstance(body.get("details"), dict) else {}
    return {
        "model": model,
        "http": status,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "capabilities": body.get("capabilities"),
    }


def _run_case(base_url, model, case, think, use_schema, max_predict, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only the requested JSON object."},
            {"role": "user", "content": PROMPT},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_predict},
    }
    if think is not None:
        payload["think"] = think
    if use_schema:
        payload["format"] = SCHEMA

    started = time.monotonic()
    try:
        status, body = _post_json(f"{base_url}/api/chat", payload, timeout)
        message = body.get("message") if isinstance(body, dict) else {}
        message = message if isinstance(message, dict) else {}
        content = message.get("content")
        thinking = message.get("thinking")
        parsed = None
        parse_error = None
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError) as exc:
                parse_error = type(exc).__name__
        ranked = parsed.get("ranked_ids") if isinstance(parsed, dict) else None
        valid = (
            isinstance(ranked, list)
            and len(ranked) == len(set(ranked))
            and all(isinstance(item, str) and item in IDS for item in ranked)
            and len(ranked) == len(IDS)
        )
        result = {
            "model": model,
            "case": case,
            "http": status,
            "latency_seconds": round(time.monotonic() - started, 2),
            "top_level_keys": sorted(body) if isinstance(body, dict) else type(body).__name__,
            "message_keys": sorted(message),
            "content_type": type(content).__name__ if content is not None else "NoneType",
            "content_length": len(content) if isinstance(content, str) else None,
            "thinking_type": type(thinking).__name__ if thinking is not None else "NoneType",
            "thinking_length": len(thinking) if isinstance(thinking, str) else None,
            "tool_calls": len(message.get("tool_calls") or []) if isinstance(message.get("tool_calls"), list) else 0,
            "done": body.get("done"),
            "done_reason": body.get("done_reason"),
            "model_returned": body.get("model"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"),
            "valid_ranked_ids": valid,
            "parse_error": parse_error,
        }
    except urllib.error.HTTPError as exc:
        result = {
            "model": model,
            "case": case,
            "http": exc.code,
            "latency_seconds": round(time.monotonic() - started, 2),
            "error_type": "HTTPError",
        }
    except Exception as exc:
        result = {
            "model": model,
            "case": case,
            "latency_seconds": round(time.monotonic() - started, 2),
            "error_type": type(exc).__name__,
        }
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://192.168.144.8:11434")
    parser.add_argument(
        "--models", nargs="+",
        default=["qwen3.6:27b", "gemma4:26b", "gpt-oss:120b-64k"],
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--num-predict", type=int, default=1800)
    parser.add_argument("--reasoning-levels", action="store_true", help="also test think=low and think=medium with schema")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    with urllib.request.urlopen(f"{base_url}/api/version", timeout=args.timeout) as response:
        version = json.load(response)
    print(json.dumps({"server_version": version.get("version")}, ensure_ascii=False), flush=True)
    for model in args.models:
        try:
            print(json.dumps({"model_info": _model_info(base_url, model, args.timeout)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(json.dumps({"model_info_error": model, "error_type": type(exc).__name__}), flush=True)
        for case, think, schema in MATRIX:
            _run_case(base_url, model, case, think, schema, args.num_predict, args.timeout)
        if args.reasoning_levels:
            for level in ("low", "medium"):
                _run_case(base_url, model, f"think_{level}_schema", level, True, args.num_predict, args.timeout)


if __name__ == "__main__":
    main()
