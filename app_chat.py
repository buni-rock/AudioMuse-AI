# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Flask blueprint for the AI chat playlist generator.

Serves the chat UI (mounted at `/chat`) and turns a natural-language request
into a playlist by calling `tasks.ai.planner.plan_and_execute_once` with the
MCP tools from `tasks.ai.tools`, then materializes the result via
`app_server_context.create_instant_playlist_for_server`.

Main Features:
* Routes: `/` chat page, `/api/config_defaults`, `/api/chatPlaylist`,
  `/api/chatPlaylistStream` (Server-Sent Events), `/api/create_playlist`.
* Per-request AI provider/model override (Ollama/OpenAI/Gemini/Mistral) and
  optional `tasks.playlist_ordering.order_playlist` post-processing.
* The stored OpenAI key is sent only to the configured OPENAI_SERVER_URL (or
  for an admin), so a request-supplied URL never receives it.
* A song count or time budget written in the request sizes the playlist (a
  budget is trimmed by real track lengths); a requested per-artist cap replaces
  the default and is not relaxed; a journey keeps its path order.
"""

from flask import Blueprint, copy_current_request_context, render_template, request, jsonify, Response, stream_with_context, g
from flasgger import swag_from  # Import swag_from
import json  # For JSON serialization of tool arguments
import logging
import queue
import re
import threading
import time
from urllib.parse import urlsplit, urlunsplit

import app_server_context
from error import error_manager
from error.error_dictionary import (
    ERR_CONFIG_INVALID,
    ERR_INVALID_REQUEST,
    ERR_PLAYLIST_REJECTED,
    UNKNOWN_ERROR_CODE,
)


logger = logging.getLogger(__name__)
# Import config module - read attributes at call time so runtime updates take effect
import config
from error.responses import json_error, json_exception

_SSE_DATA_PREFIX = "data: "
_SSE_HEARTBEAT_SECONDS = 15
_BUDGET_SECONDS_PER_SONG = 180.0
_BUDGET_TOLERANCE = 1.05

# Create a Blueprint for chat-related routes
chat_bp = Blueprint(
    'chat_bp',
    __name__,
    template_folder='templates',  # Specifies where to look for templates like chat.html
    static_folder='static',
)


@chat_bp.route('/')
@swag_from(
    {
        'tags': ['Chat UI'],
        'summary': 'Serves the main chat interface HTML page.',
        'responses': {
            '200': {
                'description': 'HTML content of the chat page.',
                'content': {'text/html': {'schema': {'type': 'string'}}},
            }
        },
    }
)
def chat_home():
    """
    Serves the main chat page.
    """
    return render_template(
        'chat.html',
        title='AudioMuse-AI - Instant Playlist',
        active='chat',
        instant_playlist_n_results_default=config.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS,
        instant_playlist_max_n_results=config.INSTANT_PLAYLIST_MAX_N_RESULTS,
        instant_playlist_selection_mode=config.INSTANT_PLAYLIST_SELECTION_MODE,
    )


@chat_bp.route('/api/config_defaults', methods=['GET'])
@swag_from(
    {
        'tags': ['Chat Configuration'],
        'summary': 'Get default AI configuration for the chat interface.',
        'responses': {
            '200': {
                'description': 'Default AI configuration.',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'default_ai_provider': {'type': 'string', 'example': 'OLLAMA'},
                                'default_ollama_model_name': {
                                    'type': 'string',
                                    'example': 'mistral:7b',
                                },
                                'ollama_server_url': {
                                    'type': 'string',
                                    'example': 'http://127.0.0.1:11434/api/generate',
                                },
                                'default_openai_model_name': {'type': 'string', 'example': 'gpt-4'},
                                'openai_server_url': {
                                    'type': 'string',
                                    'example': 'https://openrouter.ai/api/v1/chat/completions',
                                },
                                'default_gemini_model_name': {
                                    'type': 'string',
                                    'example': 'gemini-2.5-pro',
                                },
                                'default_mistral_model_name': {
                                    'type': 'string',
                                    'example': 'ministral-3b-latest',
                                },
                                'instant_playlist_default_n_results': {
                                    'type': 'integer',
                                    'example': 50,
                                },
                                'instant_playlist_max_n_results': {
                                    'type': 'integer',
                                    'example': 200,
                                },
                            },
                        }
                    }
                },
            }
        },
    }
)
def chat_config_defaults_api():
    """
    API endpoint to provide default configuration values for the chat interface.
    """
    # Read from config module attributes (may be overridden by DB settings via apply_settings_to_config)
    import config as cfg

    return jsonify(
        {
            "default_ai_provider": cfg.AI_MODEL_PROVIDER,
            "default_ollama_model_name": cfg.OLLAMA_MODEL_NAME,
            "ollama_server_url": cfg.OLLAMA_SERVER_URL,
            "default_openai_model_name": cfg.OPENAI_MODEL_NAME,
            "openai_server_url": cfg.OPENAI_SERVER_URL,
            "default_gemini_model_name": cfg.GEMINI_MODEL_NAME,
            "default_mistral_model_name": cfg.MISTRAL_MODEL_NAME,
            "instant_playlist_default_n_results": cfg.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS,
            "instant_playlist_max_n_results": cfg.INSTANT_PLAYLIST_MAX_N_RESULTS,
            "instant_playlist_selection_mode": cfg.INSTANT_PLAYLIST_SELECTION_MODE,
        }
    ), 200


def _ollama_tags_url(server_url):
    """Derive the Ollama model-list endpoint from a configured generate/chat URL."""
    raw = str(server_url or "").strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a valid HTTP or HTTPS Ollama server URL.")
    path = re.sub(r"/api/(?:generate|chat|tags)/?$", "", parsed.path, flags=re.IGNORECASE)
    path = re.sub(r"/api/?$", "", path, flags=re.IGNORECASE).rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/tags", parsed.query, ""))


@chat_bp.route('/api/ollama_models', methods=['POST'])
@swag_from(
    {
        'tags': ['Chat Configuration'],
        'summary': 'List models available on the selected Ollama server.',
        'responses': {
            '200': {'description': 'Available Ollama model names.'},
            '400': {'description': 'Invalid Ollama server URL.'},
            '502': {'description': 'Ollama server could not provide its model list.'},
        },
    }
)
def ollama_models_api():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'models': [], 'error': 'A JSON object with server_url is required.'}), 400
    server_url = data.get('server_url') or config.OLLAMA_SERVER_URL
    try:
        tags_url = _ollama_tags_url(server_url)
    except (TypeError, ValueError) as exc:
        return jsonify({'models': [], 'error': str(exc)}), 400

    from ssrf_guard import validate_outbound_url
    is_safe, reason = validate_outbound_url(tags_url)
    if not is_safe:
        return jsonify({'models': [], 'error': reason or 'The Ollama server URL is not allowed.'}), 400

    try:
        import requests
        response = requests.get(
            tags_url,
            headers={'Accept': 'application/json'},
            timeout=min(8, max(1, int(config.AI_REQUEST_TIMEOUT_SECONDS))),
            allow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        models = payload.get('models') if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return jsonify({'models': [], 'error': 'Ollama returned an invalid model list.'}), 502
        names = set()
        for model in models:
            if isinstance(model, dict):
                name = str(model.get('name') or model.get('model') or '').strip()
                if name:
                    names.add(name)
        names = sorted(names, key=str.casefold)
        return jsonify({'models': names}), 200
    except Exception as exc:
        logger.info("Ollama model listing failed (%s)", type(exc).__name__)
        return jsonify({'models': [], 'error': 'Could not load models from this Ollama server.'}), 502


def _reject_missing_user_input(data):
    # Shared guard for both chat endpoints: 400 on non-dict body or blank userInput.
    if (
        not isinstance(data, dict)
        or not isinstance(data.get('userInput'), str)
        or not data['userInput'].strip()
    ):
        return json_error(ERR_INVALID_REQUEST, "Missing userInput in request")
    return None


def _resolve_target_song_count(data):
    raw = (data or {}).get('n', config.INSTANT_PLAYLIST_DEFAULT_N_RESULTS)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return config.INSTANT_PLAYLIST_DEFAULT_N_RESULTS
    return max(1, n)


def _resolve_llm_song_target(data, requested_count=None):
    """Return a request count without applying the retired 30-track LLM cap."""
    raw_ui_count = (data or {}).get('n')
    if raw_ui_count is None:
        ui_cap = int(config.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS)
    else:
        try:
            ui_cap = max(1, int(raw_ui_count))
        except (TypeError, ValueError):
            ui_cap = int(config.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS)
    maximum = max(1, int(config.INSTANT_PLAYLIST_MAX_N_RESULTS))
    effective_target = min(maximum, max(1, int(requested_count))) if requested_count else min(ui_cap, maximum)
    return effective_target, ui_cap, maximum


def _resolve_llm_candidate_limit(effective_target):
    del effective_target
    configured = max(0, int(config.INSTANT_PLAYLIST_COMPOSER_MAX_CANDIDATES))
    return configured or None


_CLOUD_KEY_CHECKS = {
    "OPENAI": ("OpenAI", "openai_key", None),
    "GEMINI": ("Gemini", "gemini_key", "YOUR-GEMINI-API-KEY-HERE"),
    "MISTRAL": ("Mistral", "mistral_key", "YOUR-MISTRAL-API-KEY-HERE"),
}


def _openai_key_for_url(openai_url):
    if str(openai_url or '').strip() == str(config.OPENAI_SERVER_URL or '').strip():
        return config.OPENAI_API_KEY
    if getattr(g, 'auth_role', None) == 'admin':
        return config.OPENAI_API_KEY
    logger.warning(
        "chat_playlist_api: non-admin request targets an OpenAI URL other than the "
        "configured one; the stored OpenAI key is not sent"
    )
    return "no-key-needed"


def _missing_cloud_api_key(ai_provider, ai_secrets):
    display_name, secret_key, placeholder = _CLOUD_KEY_CHECKS.get(
        ai_provider, (None, None, None)
    )
    if display_name is None:
        return None
    value = (ai_secrets or {}).get(secret_key)
    if not value or (placeholder is not None and value == placeholder):
        return display_name
    return None


@chat_bp.route('/api/chatPlaylist', methods=['POST'])
@swag_from(
    {
        'tags': ['Chat Interaction'],
        'summary': 'Process user chat input to generate a playlist idea using AI.',
        'requestBody': {
            'description': 'User input and AI configuration for generating a playlist.',
            'required': True,
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'required': ['userInput'],
                        'properties': {
                            'userInput': {
                                'type': 'string',
                                'description': "The user's natural language request for a playlist.",
                                'example': "Songs for a rainy afternoon",
                            },
                            'ai_provider': {
                                'type': 'string',
                                'description': 'The AI provider to use (OLLAMA, OPENAI, GEMINI, MISTRAL, NONE). Defaults to server config.',
                                'example': 'GEMINI',
                                'enum': ['OLLAMA', 'OPENAI', 'GEMINI', "MISTRAL", 'NONE'],
                            },
                            'ai_model': {
                                'type': 'string',
                                'description': 'The specific AI model name to use. Defaults to server config for the provider.',
                                'example': 'gemini-2.5-pro',
                            },
                            'ollama_server_url': {
                                'type': 'string',
                                'description': 'Custom Ollama server URL (if ai_provider is OLLAMA).',
                                'example': 'http://localhost:11434/api/generate',
                            },
                            'openai_server_url': {
                                'type': 'string',
                                'description': 'Custom OpenAI/OpenRouter server URL (if ai_provider is OPENAI).',
                                'example': 'https://openrouter.ai/api/v1/chat/completions',
                            },
                            'openai_api_key': {
                                'type': 'string',
                                'description': 'OpenAI/OpenRouter API key (required if ai_provider is OPENAI).',
                            },
                            'gemini_api_key': {
                                'type': 'string',
                                'description': 'Custom Gemini API key (optional, defaults to server configuration).',
                            },
                            'mistral_api_key': {
                                'type': 'string',
                                'description': 'Custom Mistral API key (optional, defaults to server configuration).',
                            },
                            'n': {
                                'type': 'integer',
                                'description': 'UI default song count for LLM Compose when the request does not specify a count; explicit natural-language counts are interpreted by LLM2. Final size is bounded by INSTANT_PLAYLIST_MAX_N_RESULTS.',
                                'example': 50,
                                'minimum': 1,
                            },
                            'selection_mode': {
                                'type': 'string',
                                'enum': ['NATIVE', 'LLM_COMPOSE'],
                                'description': 'Final selection strategy. Legacy LLM_RERANK and LLM_CURATE values map to LLM_COMPOSE.',
                            },
                        },
                    }
                }
            },
        },
        'responses': {
            '200': {
                'description': 'AI response containing the playlist idea, SQL query, and processing log.',
                'content': {
                    'application/json': {
                        'schema': {
                            'type': 'object',
                            'properties': {
                                'response': {
                                    'type': 'object',
                                    'properties': {
                                        'message': {
                                            'type': 'string',
                                            'description': 'Log of AI interaction and processing.',
                                        },
                                        'original_request': {
                                            'type': 'string',
                                            'description': "The user's original input.",
                                        },
                                        'ai_provider_used': {
                                            'type': 'string',
                                            'description': 'The AI provider that was used for the request.',
                                        },
                                        'ai_model_selected': {
                                            'type': 'string',
                                            'description': 'The specific AI model that was selected/used.',
                                        },
                                        'executed_query': {
                                            'type': 'string',
                                            'nullable': True,
                                            'description': 'The SQL query that was executed (or last attempted).',
                                        },
                                        'query_results': {
                                            'type': 'array',
                                            'nullable': True,
                                            'description': 'List of songs returned by the query.',
                                            'items': {
                                                'type': 'object',
                                                'properties': {
                                                    'item_id': {'type': 'string'},
                                                    'title': {'type': 'string'},
                                                    'artist': {'type': 'string'},
                                                },
                                            },
                                        },
                                    },
                                }
                            },
                        }
                    }
                },
            },
            '400': {
                'description': 'Bad Request - Missing input or invalid parameters.',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
                    }
                },
            },
        },
    }
)
def chat_playlist_api():
    """
    Process user chat input to generate a playlist using AI with MCP tools.

    MCP TOOLS (4 CORE):
    1. seed_search - Songs similar to named seed songs/artists (union/alchemy/subtract)
    2. text_match - Semantic match on sound (CLAP) or lyric topics
    3. search_database - Filter by artist, album, genre, voice, mood, year, tempo, energy, key (ALL filters in ONE call)
    4. knowledge_lookup - Popularity/cultural requests turned into a grounded library recipe

    AI analyzes request -> calls tools -> combines results -> returns the
    requested number of songs (`n`, default INSTANT_PLAYLIST_DEFAULT_N_RESULTS)

    Non-streaming variant: runs the whole pipeline then returns the full JSON.
    """
    data = request.get_json()
    err = _reject_missing_user_input(data)
    if err:
        return err
    try:
        app_server_context.resolve_request_server_id(data)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')
    log_messages = []
    resp_obj, status = _drain_pipeline(_run_chat_pipeline(data, log_messages))
    if status >= 400:
        reason = log_messages[-1] if log_messages else None
        return json_error(ERR_CONFIG_INVALID, reason, http_status=status, response=resp_obj)
    return jsonify({"response": resp_obj}), status


@chat_bp.route('/api/chatPlaylistStream', methods=['POST'])
def chat_playlist_stream_api():
    """
    Streaming variant of the playlist generator.

    Emits a Server-Sent-Event for every progress line as the pipeline produces it,
    then a final ``done`` event with the full response payload, so the frontend can
    show real, live progress with real per-step timing.

    A background driver advances ``_run_chat_pipeline`` while this response
    emits SSE heartbeats. This keeps reverse proxies from timing out during long
    synchronous model calls; pipeline progress is still flushed as log events.
    """
    data = request.get_json()
    err = _reject_missing_user_input(data)
    if err:
        return err
    try:
        app_server_context.resolve_request_server_id(data)
    except ValueError:
        logger.warning("Invalid server selection.", exc_info=True)
        return json_error(ERR_INVALID_REQUEST, 'Invalid server selection.')

    @stream_with_context
    def generate():
        log_messages: list = []
        sent = 0

        def _flush():
            nonlocal sent
            out = ""
            while sent < len(log_messages):
                out += (
                    _SSE_DATA_PREFIX
                    + json.dumps({"type": "log", "line": log_messages[sent], "t": time.time()})
                    + "\n\n"
                )
                sent += 1
            return out

        # Emit a byte immediately so proxies/the browser open the pipe and don't
        # buffer while the first (slow) stage runs.
        yield ": stream-open\n\n"

        pipeline = _run_chat_pipeline(data, log_messages)
        events = queue.Queue()

        @copy_current_request_context
        def drive_pipeline():
            try:
                while True:
                    try:
                        next(pipeline)
                    except StopIteration as stop:
                        events.put(("done", stop.value))
                        return
                    events.put(("tick", None))
            except Exception as exc:  # propagate into the response thread for normal error handling
                events.put(("error", exc))

        threading.Thread(
            target=drive_pipeline,
            name="chat-playlist-pipeline",
            daemon=True,
        ).start()

        resp_obj = None
        try:
            while True:
                try:
                    event_type, value = events.get(timeout=_SSE_HEARTBEAT_SECONDS)
                except queue.Empty:
                    chunk = _flush()
                    if chunk:
                        yield chunk
                    # SSE comments keep the connection alive without changing the
                    # event stream consumed by the browser.
                    yield ": keep-alive\n\n"
                    continue

                if event_type == "tick":
                    chunk = _flush()
                    if chunk:
                        yield chunk
                    continue
                if event_type == "error":
                    raise value
                if event_type == "done":
                    resp_obj = (value or ({}, 200))[0]
                    break
        except Exception as exc:  # noqa: BLE001 - keep broad catch to protect streaming endpoint
            logger.exception("Streaming chat pipeline failed")
            failure = error_manager.build(error_manager.classify(exc, UNKNOWN_ERROR_CODE))
            if failure["error_code"] == UNKNOWN_ERROR_CODE:
                failure["error"] = "An internal error has occurred."
            else:
                failure["error"] = failure["error_message"]
            yield (
                _SSE_DATA_PREFIX
                + json.dumps({"type": "error", **failure, "t": time.time()})
                + "\n\n"
            )
            return

        trailing = _flush()
        if trailing:
            yield trailing
        yield (
            _SSE_DATA_PREFIX
            + json.dumps({"type": "done", "response": resp_obj, "t": time.time()})
            + "\n\n"
        )

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def _drain_pipeline(pipeline):
    """Run a ``_run_chat_pipeline`` generator to completion, discarding the
    progress ticks, and return its final ``(response_obj, status)`` value."""
    try:
        while True:
            next(pipeline)
    except StopIteration as stop:
        return stop.value or ({}, 200)


def _trim_to_duration(songs, total_seconds, log_messages):
    from tasks.ai.tool_impl import _fetch_pool_features

    try:
        feats = _fetch_pool_features([s['item_id'] for s in songs])
    except Exception:
        logger.exception("Reading track lengths for the playlist time budget failed")
        return songs
    kept, used = [], 0.0
    limit = total_seconds * _BUDGET_TOLERANCE
    for song in songs:
        length = (feats.get(song['item_id']) or {}).get('duration') or _BUDGET_SECONDS_PER_SONG
        if kept and used + length > limit:
            break
        kept.append(song)
        used += length
    if len(kept) < len(songs):
        log_messages.append(
            f"   Time budget: kept {len(kept)} songs, about {int(used // 60)} of the "
            f"{int(total_seconds // 60)} requested minutes"
        )
    return kept


def _run_chat_pipeline(data, log_messages):
    """Core chat-to-playlist pipeline, a GENERATOR. Appends progress to
    ``log_messages`` and ``yield``s a bare tick after each blocking step so the
    streaming endpoint can flush new lines live. Its final ``return`` value is
    ``(response_obj_dict, http_status)`` (read via ``StopIteration.value`` /
    ``_drain_pipeline``). Early ``return``s before the first ``yield`` still work --
    the function is a generator by virtue of the ``yield from`` below.
    """
    # Mask API key if present in the debug log
    data_for_log = dict(data) if data else {}
    if 'gemini_api_key' in data_for_log and data_for_log['gemini_api_key']:
        data_for_log['gemini_api_key'] = 'API-KEY'
    if 'mistral_api_key' in data_for_log and data_for_log['mistral_api_key']:
        data_for_log['mistral_api_key'] = 'API-KEY'
    if 'openai_api_key' in data_for_log and data_for_log['openai_api_key']:
        data_for_log['openai_api_key'] = 'API-KEY'
    logger.debug("chat_playlist_api called. Raw request data: %s", data_for_log)

    from tasks.ai.tools import get_mcp_tools
    from tasks.ai.planner import plan_and_execute_once

    original_user_input = data.get('userInput')
    # Rating intent is supplied by the structured planner interpretation.
    _user_wants_rating = False
    ai_provider = data.get('ai_provider', config.AI_MODEL_PROVIDER).upper()
    ai_model_from_request = data.get('ai_model')

    log_messages.append("NEW MCP-BASED PLAYLIST GENERATION")
    log_messages.append(f"Request: '{original_user_input}'")
    log_messages.append(f"AI Provider: {ai_provider}")

    # Check if AI provider is NONE
    if ai_provider == "NONE":
        return (
            {
                "message": "No AI provider selected. Please configure an AI provider to use this feature.",
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": None,
                "executed_query": None,
                "query_results": None,
            },
            200,
        )

    # Build AI configuration object.
    # SECURITY: API keys come ONLY from server-side config (DB-overlaid).
    # Any *_api_key field in the client payload is ignored to prevent token
    # exfiltration via the chat endpoint -- the user explicitly may select a
    # provider/model/url from the client, but the secret token must already be
    # saved on the server.
    #
    # Secrets are kept in a SEPARATE dict (`ai_secrets`) so they never coexist
    # with loggable fields. This breaks CodeQL's clear-text-logging taint flow:
    # nothing logged below ever indexes into a dict that holds keys.
    ai_config = {
        'provider': ai_provider,
        'ollama_url': data.get('ollama_server_url', config.OLLAMA_SERVER_URL),
        'ollama_model': ai_model_from_request or config.OLLAMA_MODEL_NAME,
        'openai_url': data.get('openai_server_url', config.OPENAI_SERVER_URL),
        'openai_model': ai_model_from_request or config.OPENAI_MODEL_NAME,
        'gemini_model': ai_model_from_request or config.GEMINI_MODEL_NAME,
        'mistral_model': ai_model_from_request or config.MISTRAL_MODEL_NAME,
    }
    ai_secrets = {
        'openai_key': _openai_key_for_url(ai_config['openai_url']),
        'gemini_key': config.GEMINI_API_KEY,
        'mistral_key': config.MISTRAL_API_KEY,
    }
    # The downstream AI layer expects a single merged dict.
    ai_config_with_secrets = {**ai_config, **ai_secrets}

    # Log the resolved AI target so it shows up in the flask log (without keys).
    _resolved_url = {
        "OLLAMA": ai_config['ollama_url'],
        "OPENAI": ai_config['openai_url'],
        "GEMINI": "(gemini-api)",
        "MISTRAL": "(mistral-api)",
    }.get(ai_provider, "(none)")
    _resolved_model = {
        "OLLAMA": ai_config['ollama_model'],
        "OPENAI": ai_config['openai_model'],
        "GEMINI": ai_config['gemini_model'],
        "MISTRAL": ai_config['mistral_model'],
    }.get(ai_provider, "(none)")
    logger.info(
        "chat_playlist_api -> provider=%s url=%s model=%s (default_provider=%s, client_override=%s)",
        ai_provider,
        _resolved_url,
        _resolved_model,
        config.AI_MODEL_PROVIDER,
        bool(data.get('ai_provider')),
    )

    # Validate API keys for cloud providers
    missing_key_provider = _missing_cloud_api_key(ai_provider, ai_secrets)
    if missing_key_provider is not None:
        error_msg = (
            f"Error: {missing_key_provider} API key is missing. "
            "Please provide a valid API key."
        )
        log_messages.append(error_msg)
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get(f'{ai_provider.lower()}_model'),
                "executed_query": None,
                "query_results": None,
            },
            400,
        )

    # ====================
    # MCP AGENTIC WORKFLOW
    # ====================

    log_messages.append("\nUsing MCP Agentic Workflow for playlist generation")
    selection_mode = str(data.get('selection_mode') or config.INSTANT_PLAYLIST_SELECTION_MODE).upper()
    if selection_mode in {'LLM_RERANK', 'LLM_CURATE'}:
        selection_mode = 'LLM_COMPOSE'
    if selection_mode not in {'NATIVE', 'LLM_COMPOSE'}:
        selection_mode = 'NATIVE'
    is_llm_compose = selection_mode == 'LLM_COMPOSE'
    target_song_count = _resolve_target_song_count(data)
    # Native mode keeps its planner-owned request shape. LLM Compose only uses
    # this count as a retrieval/default hint; LLM2 owns final playlist semantics.
    shape = {}
    duration_only = False
    ui_song_cap = int(config.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS)
    if is_llm_compose:
        target_song_count, ui_song_cap, _ = _resolve_llm_song_target(data, None)
        log_messages.append(f"UI default song count: {ui_song_cap}")
        log_messages.append(f"Retrieval sizing default: {target_song_count}")
        llm_candidate_limit = _resolve_llm_candidate_limit(target_song_count)
        log_messages.append(f"Configured composer capacity: {llm_candidate_limit or 'unlimited'}")
        max_final_count = None
    else:
        llm_candidate_limit = None
        max_final_count = int(config.INSTANT_PLAYLIST_MAX_N_RESULTS)

    mandatory_seed = None
    mandatory_tracks = []

    # Get MCP tools and library context
    mcp_tools = get_mcp_tools()
    log_messages.append(f"Available tools: {', '.join([t['name'] for t in mcp_tools])}")

    # Fetch library context for smarter AI prompting
    from tasks.mcp_helper import get_library_context

    library_context = get_library_context()
    if library_context.get('total_songs', 0) > 0:
        log_messages.append(
            f"Library: {library_context['total_songs']} songs, {library_context['unique_artists']} artists"
        )

    yield

    from config import MAX_SONGS_PER_ARTIST_PLAYLIST

    collection_cap = config.INSTANT_PLAYLIST_RETRIEVAL_MAX_CANDIDATES

    planner_request = original_user_input if is_llm_compose else (
        f"{original_user_input}\n\n"
        f"UI Number of songs setting: {target_song_count}."
    )
    plan_result = yield from plan_and_execute_once(
        user_message=planner_request,
        tools=mcp_tools,
        ai_config=ai_config_with_secrets,
        log_messages=log_messages,
        library_context=library_context,
        user_wants_rating=_user_wants_rating,
        collection_cap=collection_cap,
        target_song_count=target_song_count,
        raw_user_request=original_user_input,
        max_final_count=max_final_count,
        retrieval_only=is_llm_compose,
    )

    if 'error' in plan_result:
        # No fallback: do NOT invent an unrelated genre playlist. Return no
        # results so the user sees that the AI couldn't build a plan for this
        # request, rather than a made-up playlist.
        log_messages.append(f"AI planning failed: {plan_result['error']}")
        return (
            {
                "message": "\n".join(log_messages),
                "original_request": original_user_input,
                "ai_provider_used": ai_provider,
                "ai_model_selected": ai_config.get(f'{ai_provider.lower()}_model'),
                "executed_query": None,
                "query_results": None,
                "shortfall_reason": str(plan_result['error']),
            },
            200,
        )

    all_songs = plan_result['songs']
    song_sources = plan_result['song_sources']
    seed_provenance = plan_result.get('seed_provenance') or {}
    tools_used_history = plan_result['tools_used_history']
    plan_notes = plan_result.get('plan_notes', [])
    executed_query_str = plan_result['executed_query_str']
    filter_applied = plan_result.get('filter_applied', False)
    semantic_intent = plan_result.get('intent') or {}
    mandatory_tracks = list(plan_result.get('mandatory_tracks') or [])
    resolved_anchors = list(plan_result.get('canonical_retrieval_anchors') or
                            semantic_intent.get('canonical_retrieval_anchors') or [])
    if is_llm_compose:
        log_messages.append(f"Canonical anchors handed to Composer: {len(resolved_anchors)}")
    excluded_tracks = list(plan_result.get('excluded_tracks') or [])
    excluded_song_ids = {str(track['item_id']) for track in excluded_tracks if track.get('item_id') is not None}
    if excluded_song_ids:
        before_exclusions = len(all_songs)
        if not is_llm_compose:
            all_songs = [song for song in all_songs if str(song.get('item_id')) not in excluded_song_ids]
        log_messages.append(f"Planner exclusions applied: {before_exclusions - len(all_songs)} excluded song(s) removed")
    if is_llm_compose:
        # LLM1 resolves retrieval references only. Give LLM2 the resolved track
        # even when the planner described it as excluded/reference-only; LLM2
        # decides whether it belongs in the final playlist.
        mandatory_tracks = []
        existing_ids = {str(song.get('item_id')) for song in all_songs}
        for anchor in resolved_anchors:
            track = anchor.get('resolved_track') if isinstance(anchor, dict) else None
            if isinstance(track, dict) and track.get('item_id') is not None:
                if str(track['item_id']) not in existing_ids:
                    all_songs.append(track)
                    existing_ids.add(str(track['item_id']))
    for neighborhood, ids in seed_provenance.items():
        log_messages.append(f"Per-seed candidate count: {neighborhood}: {len(ids)}")
    if seed_provenance:
        log_messages.append(f"Seed neighborhoods represented at retrieval: {len(seed_provenance)}")
    if mandatory_tracks and not is_llm_compose:
        scoped_mandatory = app_server_context.scope_results(
            mandatory_tracks, None, id_key='item_id', translate=False
        )
        scoped_ids = {str(track['item_id']) for track in scoped_mandatory}
        mandatory_tracks = [
            track for track in mandatory_tracks if str(track['item_id']) in scoped_ids
        ]
        log_messages.append(f"Mandatory tracks available on selected server: {len(mandatory_tracks)}")
    if not is_llm_compose:
        log_messages.append(f"Mandatory tracks: {len(mandatory_tracks)}")
    mandatory_seed = next(iter(mandatory_tracks), None)
    shape = {
        'total_seconds': plan_result.get('target_duration_seconds'),
        'song_count': plan_result.get('requested_final_count'),
        'max_per_artist': (semantic_intent.get('constraints') or {}).get('max_per_artist', plan_result.get('max_per_artist')),
    }
    semantic_target = None if is_llm_compose else plan_result.get('requested_final_count')
    if semantic_target is not None:
        log_messages.append(f"Requested final total: {semantic_target}")
        if is_llm_compose:
            target_song_count, ui_song_cap, _ = _resolve_llm_song_target(
                data, semantic_target
            )
            target_song_count = max(len(mandatory_tracks), target_song_count)
            llm_candidate_limit = _resolve_llm_candidate_limit(target_song_count)
            log_messages.append(f"Effective final target: {target_song_count}")
            log_messages.append(
                f"Request was capped: {'yes' if target_song_count < semantic_target else 'no'}"
            )
        else:
            target_song_count = max(len(mandatory_tracks), semantic_target)
    elif shape.get('total_seconds') is None and not is_llm_compose:
        log_messages.append(f"Target: {target_song_count} songs")
    if len(mandatory_tracks) > target_song_count:
        log_messages.append(
            f"Effective target increased from {target_song_count} to {len(mandatory_tracks)} "
            "to preserve every mandatory planner anchor"
        )
        target_song_count = len(mandatory_tracks)
    if shape.get('total_seconds'):
        log_messages.append("Duration constraint detected from planner intent")
        log_messages.append(f"Target duration: {int(shape['total_seconds'])} s")
        log_messages.append("Tolerance: 15 s")

    # Keep canonical ids here: this pool is filtered for availability but stays
    # internal - it feeds playlist selection and create_instant_playlist_for_server,
    # which re-translates to the server's ids itself. Translating now would double it.
    composer_server_ids = None
    if is_llm_compose:
        # Resolve availability and response IDs once. A second registry lookup
        # after composition can block or fail, discarding an otherwise valid
        # playlist after we have already reported success.
        composer_server_ids = app_server_context.translate_ids_for_request(
            [song['item_id'] for song in all_songs]
        )
        scoped_pool = [
            song for song in all_songs
            if str(song.get('item_id')) in composer_server_ids
        ]
    else:
        scoped_pool = app_server_context.scope_results(
            all_songs, None, id_key='item_id', translate=False
        )
    if len(scoped_pool) != len(all_songs):
        log_messages.append(
            f"\nServer availability: removed {len(all_songs) - len(scoped_pool)} "
            "unavailable songs before playlist selection"
        )
    all_songs = scoped_pool
    mandatory_seed_ids = [t['item_id'] for t in mandatory_tracks]
    if mandatory_tracks:
        existing = {str(song.get('item_id')) for song in all_songs}
        mandatory_ordered = []
        for track in mandatory_tracks:
            if str(track['item_id']) not in existing:
                mandatory_ordered.append(track)
            else:
                mandatory_ordered.append(track)
                all_songs = [song for song in all_songs if str(song.get('item_id')) != str(track['item_id'])]
        all_songs = mandatory_ordered + all_songs
        log_messages.append(f"Mandatory tracks added to candidate pool: {len(mandatory_tracks)}")

    log_messages.append(f"\nSelection mode: {selection_mode}")
    log_messages.append(f"Candidates retrieved: {len(all_songs)}")
    from tasks.playlist_curation import suppress_duplicate_title_artist
    all_songs, title_artist_duplicates = suppress_duplicate_title_artist(all_songs)
    if title_artist_duplicates:
        log_messages.append(
            f"Playlist duplicate-content suppression: removed {title_artist_duplicates} "
            "same-title/same-artist duplicate(s)"
        )
    native_candidate_pool = list(all_songs)
    if llm_candidate_limit is not None:
        log_messages.append(f"Candidates shortlisted for LLM: {min(len(all_songs), llm_candidate_limit)}")
    selection_source = 'native'
    candidates_sent = 0
    candidates_retrieved_before_composer = len(all_songs)
    llm_selection_started = None
    playlist_rules_started = None
    composition_result = None
    composer_requested_target = None
    composer_requested_output = None
    composer_target_mode = None
    composer_duration_diagnostics = None
    composer_shortfall_reason = None
    composer_failure_category = None
    composer_failure_reason = None
    candidate_duration_rows = None
    if is_llm_compose and all_songs:
        log_messages.append("Composing final playlist...")
        yield
        try:
            from tasks.playlist_curation import compose_playlist_with_llm
            from tasks.ai.tool_impl import _fetch_pool_features
            llm_selection_started = time.monotonic()
            candidate_duration_rows = _fetch_pool_features(
                [song['item_id'] for song in all_songs]
            )
            composition_result, candidates_sent = compose_playlist_with_llm(
                original_user_input, all_songs, ai_config_with_secrets,
                seed_provenance=seed_provenance,
                resolved_anchors=resolved_anchors,
                ui_default_count=ui_song_cap,
                log_messages=log_messages,
                duration_rows=candidate_duration_rows,
                musical_review=True,
            )
            if composition_result.get('error'):
                failure = composition_result['error']
                category = failure.get('category', 'SCHEMA_MISMATCH') if isinstance(failure, dict) else 'SCHEMA_MISMATCH'
                reason = failure.get('reason', 'Composer response could not be validated.') if isinstance(failure, dict) else str(failure)
                composer_failure_category = category
                composer_failure_reason = reason
                log_messages.append(f"Playlist composition failed: {category}")
                if category == 'TIMEOUT':
                    log_messages.append(
                        f"{candidates_sent} candidate songs were retrieved, but the LLM composer timed out before it could build the final playlist."
                    )
                elif category == 'OUTPUT_LIMIT':
                    log_messages.append(
                        f"{candidates_sent} candidate songs were retrieved, but the LLM composer exhausted its output-token budget before returning a complete playlist."
                    )
                else:
                    # Provider errors may contain the model's entire malformed
                    # response. Keep technical logs useful without echoing it.
                    log_messages.append(f"Composer failure detail: {str(reason)[:500]}")
                selection_source = 'LLM compose failed'
                all_songs = []
                composer_shortfall_reason = f"{category}: {reason}"
                raise ValueError(category)
            all_songs = list(composition_result.get('playlist') or [])
            output_shape = composition_result.get('requested_output') or {}
            composer_requested_output = output_shape
            composer_shortfall_reason = composition_result.get('shortfall_reason')
            target_song_count = output_shape.get('target_count')
            duration_target = output_shape.get('target_duration_seconds')
            composer_target_mode = (
                'COUNT_AND_DURATION' if target_song_count is not None and duration_target is not None else
                'DURATION' if duration_target is not None else
                'COUNT' if target_song_count is not None else 'OPEN'
            )
            log_messages.append(f"Composer semantic target_count: {target_song_count if target_song_count is not None else 'none'}")
            log_messages.append(f"Composer semantic target_duration: {str(duration_target) + ' s' if duration_target is not None else 'none'}")
            log_messages.append(f"Composer preferred tracks returned: {len(all_songs)}")
            include_decisions = [
                d for d in composition_result.get('anchor_decisions', [])
                if isinstance(d, dict) and d.get('include') is True
            ]
            log_messages.append(
                f"Composer anchors preserved: {len(include_decisions)}/{len(composition_result.get('anchor_decisions', []))}"
            )
            infra_max = max(1, int(config.INSTANT_PLAYLIST_MAX_N_RESULTS))
            if (target_song_count is not None and target_song_count > infra_max) or len(all_songs) > infra_max:
                raise ValueError(
                    f"Composer target or result exceeds the configured {infra_max}-track infrastructure limit."
                )
            composer_requested_target = target_song_count
            if duration_target is not None:
                from tasks.playlist_curation import finalize_composer_duration
                song_anchors = [a for a in resolved_anchors if isinstance(a, dict) and a.get('type') == 'song']
                anchor_by_ref = {
                    f'A{index:03d}': anchor for index, anchor in enumerate(song_anchors, 1)
                }
                required_ids = [
                    anchor_by_ref[d['id']]['resolved_track']['item_id']
                    for d in include_decisions if d.get('id') in anchor_by_ref
                ]
                preferred_count = len(all_songs)
                duration_rows = candidate_duration_rows
                tolerance = config.INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS
                all_songs, actual_duration_seconds, composer_duration_diagnostics = finalize_composer_duration(
                    all_songs, duration_rows, duration_target,
                    target_count=target_song_count, required_ids=required_ids,
                    tolerance_seconds=tolerance,
                    preferred_count=composition_result.get('preferred_final_count'),
                )
                details = composer_duration_diagnostics
                log_messages.extend([
                    'Duration optimizer:',
                    f"  candidates considered: {preferred_count}",
                    f"  required anchors: {len(required_ids)}",
                    f"  target: {int(duration_target)} s",
                    f"  selected tracks: {len(all_songs)}",
                    f"  actual: {actual_duration_seconds if actual_duration_seconds is not None else 'unknown'} s",
                    f"  error: {details.get('error_seconds', 'unknown')} s",
                    f"  within tolerance: {'yes' if details.get('within_tolerance') else 'no'}",
                ])
                if actual_duration_seconds is not None:
                    actual_text = f"{actual_duration_seconds // 60}:{actual_duration_seconds % 60:02d}"
                    target_text = f"{int(duration_target) // 60}:{int(duration_target) % 60:02d}"
                    label = 'Validated playlist' if details.get('within_tolerance') and not details.get('reason') else 'Closest playlist'
                    log_messages.append(f"{label}: {len(all_songs)} tracks · {actual_text} / {target_text}")
            # Composer membership is final for count-only requests and the
            # preference set for duration requests. Neither changes the
            # semantic target_count returned by LLM2.
            shape['total_seconds'] = output_shape.get('target_duration_seconds')
            shape['song_count'] = output_shape.get('target_count')
            semantic_intent = {}  # LLM1 semantics are not authoritative in compose mode.
            selection_source = 'LLM compose only'
            song_sources = {song['item_id']: 0 for song in all_songs}
            if composer_requested_target is not None:
                log_messages.append(f"Final count target interpreted by composer: {composer_requested_target}")
            log_messages.append(f"Final playlist IDs returned: {len(all_songs)}")
            if composition_result.get('shortfall_reason'):
                log_messages.append(f"Composer reported shortfall: {composition_result['shortfall_reason']}")
        except Exception as exc:
            logger.exception("Playlist composition failed")
            if selection_source != 'LLM compose failed':
                composer_failure_category = 'SCHEMA_MISMATCH'
                composer_failure_reason = str(exc)
                log_messages.append("Playlist composition failed: SCHEMA_MISMATCH")
            composition_result = None
            all_songs = []
            selection_source = 'LLM compose failed'
        finally:
            if llm_selection_started is not None:
                log_messages.append(f"Playlist composition wall-clock: {time.monotonic() - llm_selection_started:.1f}s")
        playlist_rules_started = time.monotonic()
    elif is_llm_compose:
        log_messages.append("Playlist composition skipped: no candidates are available")
        selection_source = 'LLM compose failed'
    else:
        selection_source = 'native'
    if playlist_rules_started is None:
        playlist_rules_started = time.monotonic()
    from tasks.playlist_curation import song_family_key, suppress_song_families
    allow_multiple_versions = bool((semantic_intent.get('constraints') or {}).get('allow_multiple_versions'))
    before_family_filter = list(all_songs)
    family_groups = {}
    for song in before_family_filter:
        family_key = song_family_key(song)
        if family_key:
            family_groups.setdefault(family_key, []).append(song)
    if is_llm_compose:
        # LLM2 has already decided whether song-family variants belong.
        family_suppressed = 0
    else:
        all_songs, family_suppressed = suppress_song_families(
            all_songs, mandatory_ids=mandatory_seed_ids,
            allow_multiple=allow_multiple_versions, min_spacing=5,
        )
    if family_suppressed:
        retained_ids = {str(song.get('item_id')) for song in all_songs}
        for family in family_groups.values():
            if len(family) < 2:
                continue
            retained = [s for s in family if str(s.get('item_id')) in retained_ids]
            if len(retained) < len(family):
                log_messages.append(
                    f"Song-family suppression: {family[0].get('title', 'Unknown')}; "
                    f"candidates: {len(family)}; kept: {', '.join(s.get('title', 'Unknown') for s in retained)}; "
                    f"suppressed variants: {len(family) - len(retained)}"
                )
        log_messages.append(f"Song-family duplicates suppressed: {family_suppressed}")
    log_messages.append(f"Final unique song families: {len({song_family_key(s) for s in all_songs if song_family_key(s)})}")

    log_messages.append(f"Selection source: {selection_source}")
    if selection_source == 'native fallback':
        log_messages.append("Selection strategy: native fallback")
    if is_llm_compose and selection_source == 'LLM compose failed':
        log_messages.append(
            f"\nFinal playlist unavailable; UI default target: {ui_song_cap}; "
            f"retrieved candidates before Composer: {candidates_retrieved_before_composer}"
        )
    else:
        if is_llm_compose and composer_target_mode == 'DURATION':
            log_messages.append(f"\nValidated playlist: {len(all_songs)} tracks for {int(shape['total_seconds'])} s target")
        else:
            log_messages.append(
                f"\nCollected {len(all_songs)} songs (target "
                f"{(composer_requested_target if composer_requested_target is not None else 'unspecified') if is_llm_compose else target_song_count}, "
                f"cap {collection_cap})"
            )

    yield

    # Prepare final results
    actual_duration_seconds = (
        composer_duration_diagnostics.get('actual_seconds')
        if composer_duration_diagnostics else None
    )
    if all_songs or (is_llm_compose and composer_duration_diagnostics):
        # NOTE: rating is NOT hard-filtered here. Like every other filter dim it
        # is applied as a SOFT re-rank inside tasks.ai.rerank (rating/5
        # gradient), so high-rated songs float up but nothing is removed.

        # --- Phase 1: Artist Diversity Cap on full collected pool ---
        requested_cap = plan_result.get('max_per_artist')
        max_per_artist = len(all_songs) if is_llm_compose else (requested_cap or MAX_SONGS_PER_ARTIST_PLAYLIST)
        artist_song_counts = {}
        diversified_pool = []
        diversity_overflow = []
        mandatory_id_set = {str(value) for value in mandatory_seed_ids}
        for song in all_songs:
            artist = song.get('artist', 'Unknown')
            artist_song_counts[artist] = artist_song_counts.get(artist, 0) + 1
            if str(song.get('item_id')) in mandatory_id_set or artist_song_counts[artist] <= max_per_artist:
                diversified_pool.append(song)
            else:
                diversity_overflow.append(song)

        diversity_removed = len(all_songs) - len(diversified_pool)
        if diversity_removed > 0:
            log_messages.append(
                f"\nArtist diversity: removed {diversity_removed} excess songs from pool (max {max_per_artist}/artist)"
            )

        if (
            selection_source in {'LLM rerank only', 'LLM curate only'}
            and semantic_target is not None
            and len(diversified_pool) < target_song_count
        ):
            log_messages.append(
                f"LLM selection unusable after artist diversity: {len(diversified_pool)} candidates cannot satisfy "
                f"the requested final count of {target_song_count}"
            )
            log_messages.append("LLM candidate pool discarded; full native fallback selected")
            selection_mode = 'NATIVE'
            selection_source = 'native fallback'
            all_songs, _ = suppress_song_families(
                native_candidate_pool, mandatory_ids=mandatory_seed_ids,
                allow_multiple=allow_multiple_versions, min_spacing=5,
            )
            artist_song_counts = {}
            diversified_pool = []
            diversity_overflow = []
            for song in all_songs:
                artist = song.get('artist', 'Unknown')
                artist_song_counts[artist] = artist_song_counts.get(artist, 0) + 1
                if str(song.get('item_id')) in mandatory_id_set or artist_song_counts[artist] <= max_per_artist:
                    diversified_pool.append(song)
                else:
                    diversity_overflow.append(song)

        optimized_duration = False
        duration_target_seconds = shape.get('total_seconds')
        if duration_target_seconds is not None and duration_target_seconds > 0 and not is_llm_compose:
            log_messages.append("Duration optimizer:")
            try:
                from tasks.ai.tool_impl import _fetch_pool_features
                from tasks.playlist_curation import optimize_playlist_duration
                optimizer_limit = min(
                    len(diversified_pool),
                    config.INSTANT_PLAYLIST_DURATION_OPTIMIZER_CANDIDATES,
                )
                if optimizer_limit < len(diversified_pool):
                    log_messages.append(
                        f"Configured duration optimizer limit applied: {len(diversified_pool)} -> {optimizer_limit}"
                    )
                if selection_source in {'LLM rerank only', 'LLM curate only'}:
                    optimizer_limit = min(target_song_count, len(diversified_pool))
                optimizer_pool = diversified_pool[:optimizer_limit]
                # Keep mandatory seeds inside the bounded optimization pool.
                if mandatory_seed_ids:
                    mandatory_set = {str(i) for i in mandatory_seed_ids}
                    mandatory_pool = [s for s in diversified_pool if str(s['item_id']) in mandatory_set]
                    optimizer_pool = mandatory_pool + [
                        s for s in optimizer_pool if str(s['item_id']) not in mandatory_set
                    ][:max(0, optimizer_limit - len(mandatory_pool))]
                duration_features = _fetch_pool_features([s['item_id'] for s in optimizer_pool])
                durations = {key: row.get('duration') for key, row in duration_features.items()}
                duration_diagnostics = {}
                duration_tolerance_seconds = 15

                # A successful rerank stays isolated unless it cannot possibly
                # reach the requested duration, even using every ranked track.
                # In that case restart selection from the native pool as a unit.
                known_durations = [durations.get(song['item_id']) for song in optimizer_pool]
                if (
                    selection_source == 'LLM rerank only'
                    and known_durations
                    and all(value is not None and float(value) > 0 for value in known_durations)
                    and sum(int(float(value)) for value in known_durations)
                    < int(duration_target_seconds) - duration_tolerance_seconds
                ):
                    available_seconds = sum(int(float(value)) for value in known_durations)
                    log_messages.append("LLM rerank candidate pool insufficient for requested duration")
                    log_messages.append(f"Required target: {int(duration_target_seconds)} s")
                    log_messages.append(f"Available reranked duration: {available_seconds} s")
                    log_messages.append("Fallback reason: insufficient reranked pool")
                    log_messages.append("LLM rerank unusable: insufficient duration for hard constraint")
                    selection_mode = 'NATIVE'
                    selection_source = 'native fallback'
                    all_songs = list(native_candidate_pool)
                    all_songs, _ = suppress_song_families(
                        all_songs, mandatory_ids=mandatory_seed_ids,
                        allow_multiple=allow_multiple_versions, min_spacing=5,
                    )
                    artist_song_counts = {}
                    diversified_pool = []
                    diversity_overflow = []
                    for song in all_songs:
                        artist = song.get('artist', 'Unknown')
                        artist_song_counts[artist] = artist_song_counts.get(artist, 0) + 1
                        if str(song.get('item_id')) in mandatory_id_set or artist_song_counts[artist] <= max_per_artist:
                            diversified_pool.append(song)
                        else:
                            diversity_overflow.append(song)
                    optimizer_limit = min(
                        config.INSTANT_PLAYLIST_DURATION_OPTIMIZER_CANDIDATES,
                        len(diversified_pool), target_song_count,
                    )
                    optimizer_pool = diversified_pool[:optimizer_limit]
                    if mandatory_seed_ids:
                        mandatory_set = {str(i) for i in mandatory_seed_ids}
                        optimizer_pool = [s for s in diversified_pool if str(s['item_id']) in mandatory_set] + [
                            s for s in optimizer_pool if str(s['item_id']) not in mandatory_set
                        ]
                    duration_features = _fetch_pool_features([s['item_id'] for s in optimizer_pool])
                    durations = {key: row.get('duration') for key, row in duration_features.items()}
                    log_messages.append("LLM candidate pool discarded; full native fallback selected")
                    log_messages.append(f"Selection strategy: {selection_source}")
                    log_messages.append(f"Selection source: {selection_source}")
                    log_messages.append(f"Duration optimizer input after native fallback: {len(optimizer_pool)} candidates")
                if mandatory_seed:
                    seed_duration = durations.get(mandatory_seed['item_id'])
                    log_messages.append(f"Mandatory seed: {mandatory_seed['title']} ({mandatory_seed['item_id']})")
                    log_messages.append(f"Seed duration: {int(float(seed_duration)) if seed_duration else 'unknown'} s")
                    if seed_duration:
                        log_messages.append(f"Remaining budget: {int(shape['total_seconds']) - int(float(seed_duration))} s")
                log_messages.append(f"Duration optimizer input: {len(optimizer_pool)} candidates")
                optimizer_ranked_ids = [s.get('item_id') for s in optimizer_pool]
                final_query_results_list = optimize_playlist_duration(
                    optimizer_pool, durations, duration_target_seconds, target_song_count,
                    max_per_artist, exact_count=bool(shape.get('song_count')),
                    mandatory_ids=mandatory_seed_ids, ranked_ids=optimizer_ranked_ids,
                    tolerance_seconds=duration_tolerance_seconds,
                    diagnostics=duration_diagnostics,
                )
                missing_duration = [s for s in final_query_results_list if not durations.get(s['item_id'])]
                actual_seconds = sum(int(float(durations.get(s['item_id']) or 0)) for s in final_query_results_list)
                actual_duration_seconds = None if missing_duration else actual_seconds
                log_messages.append(f"Final songs: {len(final_query_results_list)}")
                log_messages.append(f"Selected for duration target: {len(final_query_results_list)} songs")
                log_messages.append(
                    f"Solutions inside tolerance: {duration_diagnostics.get('solutions_in_tolerance', 0)}"
                )
                log_messages.append(
                    f"Ranking cost: {duration_diagnostics.get('ranking_cost', 0)}"
                )
                average_rank = duration_diagnostics.get('average_rank')
                if average_rank is not None:
                    log_messages.append(f"Average candidate rank: {average_rank:.1f}")
                    log_messages.append(f"Worst candidate rank: {duration_diagnostics.get('worst_rank')}")
                if mandatory_seed_ids:
                    retained = any(
                        str(song.get('item_id')) == str(mandatory_seed_ids[0])
                        for song in final_query_results_list
                    )
                    log_messages.append(f"Mandatory seed retained: {str(retained).lower()}")
                if missing_duration:
                    log_messages.append("Final duration unavailable: one or more selected tracks lack duration metadata")
                else:
                    log_messages.append(f"Final duration: {actual_seconds} s")
                    log_messages.append(f"Duration error: {actual_seconds - int(duration_target_seconds):+d} s")
                best_exact_cost = duration_diagnostics.get('best_exact_ranking_cost')
                if best_exact_cost is not None:
                    log_messages.append(
                        f"Best exact-duration solution: {int(duration_target_seconds)} s; ranking cost {best_exact_cost}"
                    )
                if duration_diagnostics.get('solutions_in_tolerance', 0):
                    if (
                        best_exact_cost is not None
                        and duration_diagnostics.get('ranking_cost', 0) < best_exact_cost
                        and actual_seconds != int(duration_target_seconds)
                    ):
                        log_messages.append(
                            "Reason: selected the stronger-ranked playlist within tolerance instead of a weaker exact-duration combination"
                        )
                    else:
                        log_messages.append(
                            "Reason: selected the strongest-ranked playlist inside the duration tolerance"
                        )
                else:
                    log_messages.append(
                        "Reason: no playlist fit the duration tolerance; minimized duration error first"
                    )
                rank_by_id = {
                    str(item_id): rank
                    for rank, item_id in enumerate(
                        [i for i in optimizer_ranked_ids if str(i) not in {str(sid) for sid in mandatory_seed_ids}],
                        start=1,
                    )
                }
                for song in final_query_results_list:
                    if str(song.get('item_id')) in {str(sid) for sid in mandatory_seed_ids}:
                        log_messages.append(
                            f"Duration optimizer selected: {song.get('title', 'Unknown')} - mandatory seed"
                        )
                    else:
                        log_messages.append(
                            f"Duration optimizer selected: {song.get('title', 'Unknown')} - "
                            f"{selection_source} rank {rank_by_id.get(str(song.get('item_id')), '?')}"
                        )
                optimized_duration = True
            except Exception:
                logger.exception("Duration optimization failed")
                final_query_results_list = [s for s in diversified_pool if str(s.get('item_id')) in {str(i) for i in mandatory_seed_ids}]
        # --- Phase 2: Proportional sampling from diversified pool ---
        if is_llm_compose:
            final_query_results_list = list(diversified_pool)
        elif optimized_duration:
            pass
        elif len(diversified_pool) <= target_song_count:
            # Not enough songs after diversity cap - use all, then backfill from overflow
            final_query_results_list = list(diversified_pool)
            if requested_cap and diversity_overflow:
                log_messages.append(
                    f"   The request allows at most {requested_cap} song(s) per artist; the cap was kept"
                )
            elif len(final_query_results_list) < target_song_count and diversity_overflow and selection_mode == 'NATIVE':
                # Progressive cap relaxation: raise per-artist cap until we hit target or exhaust overflow
                current_cap = max_per_artist
                while len(final_query_results_list) < target_song_count and diversity_overflow:
                    current_cap += 1
                    # Recount artists in current final list
                    diverse_artist_counts = {}
                    for s in final_query_results_list:
                        a = s.get('artist', 'Unknown')
                        diverse_artist_counts[a] = diverse_artist_counts.get(a, 0) + 1
                    # Try to add overflow songs that fit the raised cap
                    still_overflow = []
                    backfill_added = 0
                    for song in diversity_overflow:
                        if len(final_query_results_list) >= target_song_count:
                            still_overflow.append(song)
                            continue
                        artist = song.get('artist', 'Unknown')
                        if diverse_artist_counts.get(artist, 0) < current_cap:
                            final_query_results_list.append(song)
                            diverse_artist_counts[artist] = diverse_artist_counts.get(artist, 0) + 1
                            backfill_added += 1
                        else:
                            still_overflow.append(song)
                    diversity_overflow = still_overflow
                    if backfill_added == 0:
                        break  # No progress at this cap level, stop
                if current_cap > max_per_artist:
                    log_messages.append(
                        f"   Progressive cap relaxation: {max_per_artist} -> {current_cap}/artist to reach {len(final_query_results_list)} songs"
                    )
            elif selection_mode != 'NATIVE' and diversity_overflow:
                log_messages.append(
                    f"   Artist cap retained at {max_per_artist}/artist; returned {len(final_query_results_list)} songs"
                )
        else:
            # More diversified songs than target - sample proportionally by tool call
            songs_by_call = {}
            for song in diversified_pool:
                call_index = song_sources.get(song['item_id'], -1)
                if call_index not in songs_by_call:
                    songs_by_call[call_index] = []
                songs_by_call[call_index].append(song)

            total_in_pool = len(diversified_pool)
            final_query_results_list = []
            for call_index, tool_songs in songs_by_call.items():
                proportion = len(tool_songs) / total_in_pool
                allocated = int(proportion * target_song_count)
                if allocated == 0 and len(tool_songs) > 0:
                    allocated = 1
                final_query_results_list.extend(tool_songs[:allocated])

            # Round-up correction: fill remaining slots from diversified songs not yet selected
            if len(final_query_results_list) < target_song_count:
                selected_ids = {s['item_id'] for s in final_query_results_list}
                remaining = [s for s in diversified_pool if s['item_id'] not in selected_ids]
                needed = target_song_count - len(final_query_results_list)
                final_query_results_list.extend(remaining[:needed])

            final_query_results_list = final_query_results_list[:target_song_count]

        if shape.get('total_seconds') is not None and not shape.get('song_count') and not optimized_duration and not is_llm_compose:
            final_query_results_list = _trim_to_duration(
                final_query_results_list, shape['total_seconds'], log_messages
            )

        log_messages.append(
            f"\nPool: {len(all_songs)} collected -> {len(diversified_pool)} after diversity cap -> {len(final_query_results_list)} in final playlist"
        )

        # --- Song Ordering for Smooth Transitions (Phase 3A) ---
        # Only when NO filter drove the result. When a filter/score was applied
        # (e.g. "female vocalist", a genre, year, etc.), the songs are already in
        # the order the score produced -- matched songs on top, then the rest by
        # similarity. Re-sorting by tempo/energy/key here would scramble that and
        # bury the matched songs, so the scored order is preserved instead.
        if is_llm_compose:
            log_messages.append("Playlist kept in LLM composer order")
        elif filter_applied:
            log_messages.append(
                "\nPlaylist kept in filter-ranked order (matched songs first); smooth-transition reorder skipped"
            )
        elif plan_result.get('keep_order'):
            log_messages.append(
                "\nPlaylist kept in journey order (from the first seed to the second)"
            )
        elif selection_source in {'LLM rerank only', 'LLM curate only', 'LLM compose only'}:
            log_messages.append("Playlist kept in LLM composer order")
        elif shape.get('total_seconds') is None:
            log_messages.append(
                "\nPlaylist kept in native/curator rank order (no explicit duration constraint)"
            )
        else:
            try:
                from tasks.playlist_ordering import order_playlist
                from config import PLAYLIST_ENERGY_ARC

                original_playlist = list(final_query_results_list)
                song_id_list = [s['item_id'] for s in original_playlist]
                ordered_ids = order_playlist(song_id_list, energy_arc=PLAYLIST_ENERGY_ARC)
                from tasks.playlist_curation import reorder_preserving_membership
                before_ids = {str(s['item_id']) for s in original_playlist}
                final_query_results_list = reorder_preserving_membership(original_playlist, ordered_ids)
                after_ids = {str(s['item_id']) for s in final_query_results_list}
                if before_ids != after_ids:
                    logger.error("Playlist ordering changed membership; restoring original member set")
                    log_messages.append("Ordering membership mismatch detected; omitted tracks were restored")
                    final_query_results_list = original_playlist
                log_messages.append("\nPlaylist ordered for smooth transitions (membership preserved)")
            except Exception:
                logger.warning("Playlist ordering failed (non-fatal)", exc_info=True)
                log_messages.append(
                    "\nPlaylist ordering skipped due to an internal processing issue"
                )

        # Final integrity guard: explicit, resolved, non-excluded tracks are
        # authoritative even if a selector, artist cap, or ordering step omitted one.
        mandatory_by_id = {str(t['item_id']): t for t in mandatory_tracks}
        final_by_id = {str(s.get('item_id')): s for s in final_query_results_list}
        for track_id, track in mandatory_by_id.items():
            if track_id not in final_by_id:
                final_query_results_list.append(track)
                final_by_id[track_id] = track
        if mandatory_by_id:
            mandatory_prefix = [final_by_id[str(t['item_id'])] for t in mandatory_tracks if str(t['item_id']) in final_by_id]
            mandatory_set = set(mandatory_by_id)
            optionals = [s for s in final_query_results_list if str(s.get('item_id')) not in mandatory_set]
            final_query_results_list = mandatory_prefix + optionals
            if len(final_query_results_list) > target_song_count:
                final_query_results_list = mandatory_prefix + optionals[:max(0, target_song_count - len(mandatory_prefix))]
        present_mandatory = sum(
            1 for track_id in mandatory_by_id
            if any(str(song.get('item_id')) == track_id for song in final_query_results_list)
        )
        log_messages.append(f"Mandatory tracks present in final playlist: {present_mandatory}/{len(mandatory_by_id)}")
        if family_suppressed:
            log_messages.append(
                f"Final unique song families: {len({song_family_key(s) for s in final_query_results_list if song_family_key(s)})}"
            )

        final_executed_query_str = executed_query_str

        if plan_notes:
            log_messages.append("\nPlan notes:")
            for n in plan_notes:
                log_messages.append(f"   {n}")

        if is_llm_compose and composer_duration_diagnostics and (
            not composer_duration_diagnostics.get('within_tolerance') or
            composer_duration_diagnostics.get('reason')
        ):
            reason = composer_duration_diagnostics.get('reason') or 'Duration target was not met.'
            error = composer_duration_diagnostics.get('error_seconds')
            log_messages.append(
                f"\nPlaylist duration validation failed: {reason} "
                f"(difference {error:+d} s)" if isinstance(error, int) else
                f"\nPlaylist duration validation failed: {reason}"
            )
            composer_shortfall_reason = (
                f"{reason} Closest difference: {error:+d} s."
                if isinstance(error, int) else reason
            )
            composer_failure_category = 'DURATION_MISMATCH'
            selection_source = 'LLM compose failed'
            final_query_results_list = []
        log_messages.append(f"   Total songs collected: {len(all_songs)}")
        log_messages.append(f"   Tools called: {len(tools_used_history)}")

        # Show tool contribution breakdown (collected vs final)
        log_messages.append("\nTool Contribution (Collected -> Final Playlist):")

        # Count songs in final playlist by tool call
        final_by_call = {}
        for song in final_query_results_list:
            call_index = song_sources.get(song['item_id'], -1)
            final_by_call[call_index] = final_by_call.get(call_index, 0) + 1

        for tool_info in tools_used_history:
            tool_name = tool_info['name']
            song_count = tool_info.get('songs', 0)
            args = tool_info.get('args', {})
            args_preview = []
            if 'artist' in args:
                args_preview.append(f"artist='{args['artist']}'")
            elif 'artist_name' in args:
                args_preview.append(f"artist='{args['artist_name']}'")
            if 'song_title' in args:
                args_preview.append(f"title='{args['song_title']}'")
            if 'genres' in args and args['genres']:
                args_preview.append(f"genres={args['genres'][:2]}")
            if 'moods' in args and args['moods']:
                args_preview.append(f"moods={args['moods'][:2]}")
            if 'exclude_artists' in args and args['exclude_artists']:
                args_preview.append(f"exclude_artists={args['exclude_artists'][:2]}")
            if 'exclude_genres' in args and args['exclude_genres']:
                args_preview.append(f"exclude_genres={args['exclude_genres'][:2]}")
            if args.get('voices'):
                args_preview.append(f"voices={args['voices'][:1]}")
            if args.get('instrumental') is not None:
                args_preview.append(f"instrumental={args['instrumental']}")
            for label, lo_key, hi_key in (
                ('year', 'year_min', 'year_max'),
                ('tempo', 'tempo_min', 'tempo_max'),
                ('energy', 'energy_min', 'energy_max'),
                ('duration', 'duration_min', 'duration_max'),
            ):
                if args.get(lo_key) is not None or args.get(hi_key) is not None:
                    args_preview.append(f"{label}={args.get(lo_key, '')}..{args.get(hi_key, '')}")
            if args.get('album'):
                args_preview.append(f"album='{args['album']}'")
            if args.get('query'):
                args_preview.append(f"query='{str(args['query'])[:30]}'")
            if args.get('seeds'):
                args_preview.append(f"seeds={len(args['seeds'])}")
            if args.get('instruments'):
                args_preview.append(f"instruments={args['instruments'][:2]}")
            if 'user_request' in args:
                args_preview.append(f"request='{args['user_request'][:30]}...'")

            args_str = ", ".join(args_preview) if args_preview else "no filters"
            call_index = tool_info.get('call_index', -1)
            final_count = final_by_call.get(call_index, 0)
            if tool_info.get('role') == 'pool':
                log_messages.append(
                    f"   - {tool_name}({args_str}): found {song_count} candidates, re-ranked by the filter below"
                )
            elif tool_info.get('role') == 'rerank':
                log_messages.append(
                    f"   - {tool_name}({args_str}): re-ranked those candidates -> {final_count} in final playlist"
                )
            elif song_count != final_count:
                log_messages.append(
                    f"   - {tool_name}({args_str}): {song_count} collected -> {final_count} in final playlist"
                )
            else:
                log_messages.append(f"   - {tool_name}({args_str}): {song_count} songs")
    elif (is_llm_compose and selection_source == 'LLM compose failed'
          and (candidates_retrieved_before_composer > 0 or composer_failure_category)):
        log_messages.append("\nNo Composer result was available; playlist validation was skipped")
        final_query_results_list = []
        final_executed_query_str = "LLM Compose failed before playlist validation"
    else:
        log_messages.append("\nNo songs collected")
        if plan_notes:
            log_messages.append("\nPlan notes:")
            for n in plan_notes:
                log_messages.append(f"   {n}")
        log_messages.append(
            "\nNo matching songs were found in your library for this request "
            "(a corrective retry was already attempted). Try naming an artist, album or "
            "genre that exists in your library, or loosen the constraints."
        )
        final_query_results_list = None
        final_executed_query_str = executed_query_str or "MCP single-pass: No results"

    actual_model_used = ai_config.get(f'{ai_provider.lower()}_model')

    if selection_source in {'LLM rerank only', 'LLM curate only', 'native fallback', 'LLM compose only'}:
        if composer_target_mode == 'DURATION':
            log_messages.append(f"Final requested duration: {int(shape['total_seconds'])} s")
        else:
            log_messages.append(f"Final requested target: {composer_requested_target or target_song_count}")
        final_count = len(final_query_results_list or [])
        log_messages.append(f"Final playlist: {final_count}")
        if selection_source == 'LLM compose only' and composer_requested_target is not None and final_count < composer_requested_target:
            log_messages.append("Reason: composer reported fewer suitable tracks than the requested target")
        elif selection_source == 'LLM curate only' and final_count < target_song_count:
            log_messages.append(
                "Reason: curator returned fewer high-confidence candidates than target"
            )
        elif selection_source == 'LLM rerank only' and final_count < target_song_count:
            log_messages.append(
                "Reason: reranker returned fewer usable candidates than target"
            )

    # The pool stayed canonical for internal selection/ordering; translate the
    # FINAL list to the selected server's provider ids so the response never emits
    # an internal fp_ id. /api/create_playlist resolves them back to canonical.
    if is_llm_compose and final_query_results_list:
        try:
            duration_rows = candidate_duration_rows or {}
            duration_values = [duration_rows.get(song['item_id'], {}).get('duration') for song in final_query_results_list]
            if all(value is not None and float(value) > 0 for value in duration_values):
                actual_duration_seconds = sum(int(float(value)) for value in duration_values)
            if shape.get('total_seconds'):
                log_messages.append(
                    f"Composer duration validation: target={int(shape['total_seconds'])} s; "
                    f"actual={actual_duration_seconds if actual_duration_seconds is not None else 'unknown'} s"
                )
        except Exception:
            logger.warning("Could not compute composed playlist duration", exc_info=True)
    if is_llm_compose and final_query_results_list:
        missing_ids = [
            str(song['item_id']) for song in final_query_results_list
            if str(song['item_id']) not in composer_server_ids
        ]
        if missing_ids:
            log_messages.append(
                f"Playlist delivery failed: {len(missing_ids)} selected tracks have no provider ID"
            )
            composer_shortfall_reason = 'Selected tracks are unavailable on the media server.'
            final_query_results_list = []
        else:
            final_query_results_list = [
                {**song, 'item_id': composer_server_ids[str(song['item_id'])]}
                for song in final_query_results_list
            ]
    elif final_query_results_list:
        final_query_results_list = app_server_context.scope_results(
            final_query_results_list, None, id_key='item_id'
        )
    log_messages.append(f"Playlist response ready: {len(final_query_results_list or [])} tracks")
    if final_query_results_list:
        log_messages.append(
            f"OK SUCCESS! Generated playlist with {len(final_query_results_list)} songs"
        )

    log_messages.append(f"Playlist rules wall-clock: {time.monotonic() - playlist_rules_started:.1f}s")

    # Return final response object (caller wraps it for HTTP).
    return (
        {
            "message": "\n".join(log_messages),
            "original_request": original_user_input,
            "ai_provider_used": ai_provider,
            "ai_model_selected": actual_model_used,
            "executed_query": final_executed_query_str,
            "query_results": final_query_results_list,
            "target_duration_seconds": shape.get('total_seconds'),
            "actual_duration_seconds": actual_duration_seconds,
            "requested_output": composer_requested_output,
            "shortfall_reason": composer_shortfall_reason,
        },
        200,
    )


@chat_bp.route('/api/create_playlist', methods=['POST'])
@swag_from(
    {
        'tags': ['Chat Interaction'],
        'summary': 'Create a playlist on the media server from a list of song item IDs.',
        'requestBody': {
            'description': 'Playlist name and song item IDs.',
            'required': True,
            'content': {
                'application/json': {
                    'schema': {
                        'type': 'object',
                        'required': ['playlist_name', 'item_ids'],
                        'properties': {
                            'playlist_name': {
                                'type': 'string',
                                'description': 'The desired name for the playlist.',
                                'example': 'My Awesome Mix',
                            },
                            'item_ids': {
                                'type': 'array',
                                'description': 'A list of item IDs for the songs to include.',
                                'items': {'type': 'string'},
                                'example': [
                                    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                                    "yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy",
                                ],
                            },
                        },
                    }
                }
            },
        },
        'responses': {
            '200': {
                'description': 'Playlist successfully created.',
                'content': {
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}}
                    }
                },
            },
            '400': {'description': 'Bad Request - Missing parameters or invalid input.'},
            '500': {
                'description': 'Server Error - Failed to create playlist.',
                'content': {  # Added content for 400 and 500 for consistency
                    'application/json': {
                        'schema': {'type': 'object', 'properties': {'message': {'type': 'string'}}}
                    }
                },
            },
        },
    }
)
def create_media_server_playlist_api():
    """
    API endpoint to create a playlist on the configured media server.
    """
    data = request.get_json()
    if not data or 'playlist_name' not in data or 'item_ids' not in data:
        reason = "Error: Missing playlist_name or item_ids in request"
        return json_error(ERR_INVALID_REQUEST, reason, message=reason)

    user_playlist_name = data.get('playlist_name')
    item_ids = data.get('item_ids')  # This will be a list of strings

    if not user_playlist_name or not str(user_playlist_name).strip():
        reason = "Error: Playlist name cannot be empty."
        return json_error(ERR_INVALID_REQUEST, reason, message=reason)
    if not item_ids:
        reason = "Error: No songs provided to create the playlist."
        return json_error(ERR_INVALID_REQUEST, reason, message=reason)

    try:
        server_id = app_server_context.resolve_request_server_id(data)
    except ValueError as exc:
        return json_exception(exc, ERR_INVALID_REQUEST, message=None)

    # The client posts back the provider ids it got from /api/chatPlaylist;
    # canonicalize them so the dispatcher translates to the target server exactly
    # once. A canonical id passes through unchanged (older clients keep working).
    resolved = app_server_context.resolve_input_item_ids(item_ids, data)
    item_ids = [resolved.get(str(i), i) for i in item_ids]

    try:
        try:
            info = app_server_context.create_instant_playlist_for_server(
                user_playlist_name, item_ids, server_id
            )
        except ValueError as exc:
            return json_exception(exc, ERR_PLAYLIST_REJECTED, message=None)
        created_playlist_info = info['result']

        if not created_playlist_info:
            raise Exception("Media server did not return playlist information after creation.")

        return jsonify(
            {
                "message": f"Successfully created playlist '{user_playlist_name}' on the media server with ID: {created_playlist_info.get('Id')}"
            }
        ), 200

    except Exception as e:
        # Log detailed error on the server
        error_details_for_server = f"Media Server API Request Exception: {str(e)}\n"
        if hasattr(e, 'response') and e.response is not None:  # type: ignore[attr-defined]
            try:
                error_details_for_server += f" - Media Server Response: {e.response.text}\n"
            except Exception:
                pass  # nosec
        logger.exception(
            "Error in create_media_server_playlist_api: %s", error_details_for_server
        )
        # Return generic, structured error to client (traceback stays in the log only).
        failed = "An internal error occurred while creating the playlist."
        return json_exception(e, UNKNOWN_ERROR_CODE, failed, message=failed)
