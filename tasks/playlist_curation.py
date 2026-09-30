"""Grounded optional curation helpers for Instant Playlist."""

import base64
import hashlib
import json
import logging
import math
import os
import re
import unicodedata

import config
from tasks.ai.json_response import parse_json_response

logger = logging.getLogger(__name__)
_RERANK_SAFETY_MARGIN = 5
_LLM_CANDIDATE_MAX_OUTPUT_TOKENS = 1200


def effective_llm_artist_cap(final_target, absolute_cap, fraction=None):
    """Scale artist diversity to short LLM playlists while retaining a hard cap."""
    fraction = config.INSTANT_PLAYLIST_MAX_ARTIST_FRACTION if fraction is None else float(fraction)
    return min(max(1, int(absolute_cap)), max(1, math.ceil(max(1, int(final_target)) * fraction)))


def probe_curator_provider(ai_config):
    """Exercise only the curator provider adapter with a tiny JSON request."""
    from tasks.ai.api import generate_text

    content = generate_text(
        'Return exactly: {"ranked_ids":["C002","C001"]}', ai_config,
        skip_delay=True, temperature=0.0, max_tokens=100,
        structured_format={
            "type": "object",
            "properties": {"ranked_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["ranked_ids"],
            "additionalProperties": False,
        } if str(ai_config.get("provider", "")).upper() == "OLLAMA" else None,
        system_prompt="Return JSON only.",
    )
    if str(content).startswith("Error:"):
        return {"status": "PROVIDER_ERROR", "content": content, "parsed": None}
    parsed, extracted, _thinking, parse_error = parse_json_response(content)
    if parse_error:
        return {"status": "PARSE_ERROR", "content": content, "parsed": None, "error": parse_error}
    success = isinstance(parsed, dict) and parsed.get("ranked_ids") == ["C002", "C001"]
    return {
        "status": "SUCCESS" if success else "VALIDATION_ERROR",
        "content": extracted,
        "parsed": parsed,
    }


def probe_curator_candidate_window(ai_config, count=40):
    """Ask the configured model to report both ends and the size of a test shortlist."""
    from tasks.ai.api import generate_text

    count = max(2, min(100, int(count)))
    records = [
        {"id": f"C{i:03d}", "title": f"Probe track {i:03d}", "artist": "Probe artist"}
        for i in range(1, count + 1)
    ]
    schema = {
        "type": "object",
        "properties": {
            "first_id": {"type": "string"},
            "last_id": {"type": "string"},
            "candidate_count": {"type": "integer"},
        },
        "required": ["first_id", "last_id", "candidate_count"],
        "additionalProperties": False,
    }
    content = generate_text(
        "Read the complete candidate array. Return its first id, last id, and number of records. "
        "Do not infer these from the first or last visible item if the array continues.\n"
        + json.dumps(records, ensure_ascii=False, separators=(",", ":")),
        ai_config,
        skip_delay=True,
        temperature=0.0,
        max_tokens=100,
        structured_format=schema if str(ai_config.get("provider", "")).upper() == "OLLAMA" else None,
        system_prompt="Inspect the full supplied array and return only the requested JSON object.",
    )
    parsed, extracted, _thinking, parse_error = parse_json_response(content)
    expected = {
        "first_id": "C001",
        "last_id": f"C{count:03d}",
        "candidate_count": count,
    }
    valid = (
        not parse_error
        and isinstance(parsed, dict)
        and set(parsed) == set(expected)
        and parsed == expected
    )
    return {
        "status": "SUCCESS" if valid else "VALIDATION_ERROR" if not parse_error else "PARSE_ERROR",
        "candidate_count_sent": len(records),
        "payload_chars": len(json.dumps(records, ensure_ascii=False, separators=(",", ":"))),
        "content": extracted,
        "parsed": parsed,
        "expected": expected,
        "error": parse_error,
    }


def inspect_llm_candidate_selection(raw, candidate_ids, mode, candidate_titles=None):
    """Parse a response and explain every rejection while accepting exact IDs only."""
    mode = str(mode or "").upper()
    key = "ranked_ids" if mode == "LLM_RERANK" else "selected_ids"
    allowed = {str(item_id) for item_id in candidate_ids}
    obj, extracted, thinking, parse_error = parse_json_response(raw)
    diagnostics = {
        "mode": mode,
        "candidate_count": len(candidate_ids),
        "candidate_id_format": type(candidate_ids[0]).__name__ if candidate_ids else "empty",
        "first_candidate_ids": [str(item_id) for item_id in candidate_ids[:3]],
        "thinking_present": bool(thinking.strip()),
        "http_extracted_content_type": type(raw).__name__,
        "extracted_content": extracted,
        "parsed_json": obj,
        "json_top_level_type": type(obj).__name__ if obj is not None else None,
        "normalization_input_type": type(obj).__name__ if obj is not None else None,
        "normalization_output_aliases": [],
        "expected_key": key,
        "raw_response_shape": None,
        "returned_keys": [],
        "normalization_note": None,
        "normalized_alias_count": 0,
        "returned_id_count": 0,
        "first_returned_ids": [],
        "valid_ids": [],
        "invalid_ids": [],
        "duplicates_removed": [],
        "rejection_reason": None,
    }
    if parse_error:
        diagnostics["rejection_reason"] = f"invalid JSON: {parse_error}"
        return [], diagnostics
    diagnostics["raw_response_shape"] = "list" if isinstance(obj, list) else type(obj).__name__
    diagnostics["returned_keys"] = sorted(str(k) for k in obj) if isinstance(obj, dict) else []
    values, normalization_note = _normalize_curator_aliases(obj, mode)
    diagnostics["normalization_note"] = normalization_note
    if values is None:
        diagnostics["rejection_reason"] = (
            f"missing or ambiguous alias list for {key!r}; returned keys: {diagnostics['returned_keys']}"
        )
        return [], diagnostics
    diagnostics["returned_id_count"] = len(values)
    diagnostics["normalized_alias_count"] = len(values)
    diagnostics["normalization_output_aliases"] = list(values)
    diagnostics["first_returned_ids"] = values[:10]
    seen, valid, invalid, duplicates = set(), [], [], []
    titles = {str(title).casefold() for title in (candidate_titles or {}).values() if title}
    for value in values:
        if not isinstance(value, str):
            invalid.append({"value": value, "reason": "ID must be a string"})
        elif value in seen:
            duplicates.append(value)
        elif value not in allowed:
            reason = "title returned instead of ID" if value.casefold() in titles else "ID was not present in the candidate map"
            invalid.append({"value": value, "reason": reason})
            seen.add(value)
        else:
            valid.append(value)
            seen.add(value)
    diagnostics["valid_ids"] = valid
    diagnostics["invalid_ids"] = invalid
    diagnostics["duplicates_removed"] = duplicates
    if not valid:
        diagnostics["rejection_reason"] = "response contained no exact candidate IDs"
    return valid, diagnostics


def _normalize_curator_aliases(obj, mode):
    """Accept only the exact response object required for the selected mode."""
    preferred = "ranked_ids" if str(mode or "").upper() == "LLM_RERANK" else "selected_ids"
    if not isinstance(obj, dict):
        return None, "response must be an object"
    if set(obj) != {preferred}:
        return None, f"expected exactly the {preferred!r} property"
    if not isinstance(obj[preferred], list):
        return None, f"{preferred} is not a list"
    return obj[preferred], f"exact {preferred} list detected"


def validate_llm_candidate_selection(raw, candidate_ids, mode):
    """Return unique known IDs from a provider response, preserving its order."""
    ids, _ = inspect_llm_candidate_selection(raw, candidate_ids, mode)
    return ids


def build_llm_candidate_payload(songs, features=None, aliases=None):
    """Build metadata-only records shared by ranking and curation calls."""
    del features  # AudioMuse analysis data is intentionally never sent to the LLM.
    records = []
    for song in songs:
        item_id = song.get("item_id")
        if item_id is None:
            continue
        alias = aliases.get(str(item_id)) if aliases is not None else str(item_id)
        if alias is None:
            continue
        row = {"id": alias}
        for key in ("title", "artist", "album"):
            value = song.get(key)
            if value is None and key == "artist":
                value = song.get("author")
            if value is not None and value != "":
                row[key] = str(value)
        records.append(row)
    return records


def _rerank_context_key(user_request, resolved_seed):
    seed = resolved_seed or {}
    stable_seed = {
        "title": str(seed.get("title") or ""),
        "artist": str(seed.get("artist") or seed.get("author") or ""),
        "item_id": str(seed.get("item_id") or ""),
    }
    return json.dumps([str(user_request or ""), stable_seed], ensure_ascii=False, sort_keys=True)


def _opaque_rerank_aliases(songs, context):
    """Create stable aliases that reveal neither item identity nor candidate order."""
    aliases, used = {}, set()
    for song in songs:
        item_id = str(song.get("item_id"))
        digest = hashlib.sha256(f"alias\0{context}\0{item_id}".encode("utf-8")).digest()
        # A base32 alphabet makes short, opaque aliases that are easy to return.
        token = base64.b32encode(digest).decode("ascii").rstrip("=")
        length = 5
        alias = token[:length]
        while alias in used:
            length += 1
            alias = token[:length]
        aliases[item_id] = alias
        used.add(alias)
    return aliases


def _rerank_candidate_order(songs, context):
    """Deterministically shuffle independent of AudioMuse's incoming order."""
    unique = {}
    for song in songs:
        item_id = song.get("item_id")
        if item_id is not None:
            unique.setdefault(str(item_id), song)
    return sorted(
        unique.values(),
        key=lambda song: hashlib.sha256(
            f"shuffle\0{context}\0{song.get('item_id')}".encode("utf-8")
        ).digest(),
    )


def build_llm_candidate_alias_map(songs, context):
    """Build request-local opaque aliases and the authoritative reverse map."""
    item_id_to_alias = _opaque_rerank_aliases(songs, context)
    alias_to_song = {}
    for song in songs:
        item_id = song.get("item_id")
        alias = item_id_to_alias.get(str(item_id)) if item_id is not None else None
        if alias is not None:
            alias_to_song.setdefault(alias, song)
    return item_id_to_alias, alias_to_song


def prepare_llm_candidate_shortlist(songs, context, limit=50):
    """Bound, de-duplicate, shuffle, alias, and serialize one shared LLM shortlist."""
    bounded = list(songs[:max(0, int(limit))])
    shuffled = _rerank_candidate_order(bounded, context)
    item_id_to_alias, alias_to_song = build_llm_candidate_alias_map(shuffled, context)
    records = build_llm_candidate_payload(shuffled, aliases=item_id_to_alias)
    return shuffled, item_id_to_alias, alias_to_song, records


def _normalized_content_text(value):
    value = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _title_artist_identity(song):
    artist = str(song.get("artist") or song.get("author") or "").strip()
    title = str(song.get("title") or "").strip()
    title = re.sub(r"^(?:(?:\(\s*\d{1,3}\s*\)|\d{1,3})[\s._-]*)+", "", title)
    bracketed_artist = re.match(r"^\[([^\]]+)\]\s*", title)
    if bracketed_artist and _normalized_content_text(bracketed_artist.group(1)) == _normalized_content_text(artist):
        title = title[bracketed_artist.end():]
    return _normalized_content_text(artist), _normalized_content_text(title)


def suppress_duplicate_title_artist(songs):
    """Keep the first playlist occurrence for each normalized artist/title pair."""
    kept, seen = [], set()
    for song in songs:
        identity = _title_artist_identity(song)
        if not all(identity):
            kept.append(song)
            continue
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(song)
    return kept, len(songs) - len(kept)


_VERSION_MARKER_RE = re.compile(
    r"(?:remix|\b[a-z0-9 &'._-]*\bmix\b|radio\s+edit|single\s+edit|extended\s+edit|"
    r"\bedit\b|instrumental|acoustic|demo|live(?:\s+version)?|remaster(?:ed)?|"
    r"single\s+version|album\s+version|\bversion\b|mono\s+version|stereo\s+version)",
    re.I,
)
_TRAILING_VERSION_RE = re.compile(
    r"\s*(?:\(([^()]*)\)|\[([^\[\]]*)\]|\s[-–—]\s([^()\[\]]+))\s*$"
)


def song_family_key(song):
    """Return artist plus conservative base-title identity for version grouping."""
    artist = song.get("artist") or song.get("author") or ""
    title = str(song.get("title") or "").strip()
    while title:
        match = _TRAILING_VERSION_RE.search(title)
        if not match:
            break
        suffix = next((value for value in match.groups() if value is not None), "")
        if not _VERSION_MARKER_RE.search(suffix):
            break
        title = title[:match.start()].strip()
    # Recognize an unbracketed trailing edit/remix label too, without stripping
    # collaboration annotations such as “(Feat. Eyelar)”.
    title = re.sub(
        r"\s+[-–—]\s+(?=[^ ]*(?:remix|mix|edit|live|instrumental|acoustic|remaster))[^()]+$",
        "", title, flags=re.I,
    ).strip()
    artist_key = _normalized_content_text(artist)
    title_key = _normalized_content_text(title)
    return (artist_key, title_key) if artist_key and title_key else None


def suppress_song_families(songs, mandatory_ids=(), allow_multiple=False, min_spacing=5):
    """Suppress alternate versions before count selection, preserving ranked order.

    Mandatory items take precedence over ranked optional variants. When a user
    explicitly requests multiple versions, retain them and schedule each family
    member with at least ``min_spacing`` intervening tracks when possible.
    """
    mandatory = {str(value) for value in mandatory_ids}
    groups = {}
    for song in songs:
        key = song_family_key(song)
        if key is not None:
            groups.setdefault(key, []).append(song)
    kept, suppressed = [], 0
    for key, family in groups.items():
        mandatory_family = [s for s in family if str(s.get("item_id")) in mandatory]
        if mandatory_family:
            chosen = mandatory_family if allow_multiple else mandatory_family[:1]
        else:
            chosen = family if allow_multiple else family[:1]
        chosen_ids = {str(s.get("item_id")) for s in chosen}
        suppressed += len(family) - len(chosen)
    if not allow_multiple:
        for song in songs:
            key = song_family_key(song)
            family = groups.get(key, []) if key is not None else [song]
            mandatory_family = [s for s in family if str(s.get("item_id")) in mandatory]
            winner_songs = mandatory_family or family[:1] or [song]
            winners = {str(s.get("item_id")) for s in winner_songs}
            if str(song.get("item_id")) in winners:
                kept.append(song)
        return kept, suppressed

    # Stable greedy spacing: defer a repeated family member until enough other
    # tracks have been placed, then append any unavoidable leftovers at the end.
    pending = list(songs)
    last_position = {}
    while pending:
        progress = False
        deferred = []
        for song in pending:
            key = song_family_key(song)
            if key is None or key not in last_position or len(kept) - last_position[key] > min_spacing:
                kept.append(song)
                if key is not None:
                    last_position[key] = len(kept) - 1
                progress = True
            else:
                deferred.append(song)
        if not progress:
            kept.extend(deferred)
            break
        pending = deferred
    return kept, suppressed


def curate_candidates_with_llm(
    user_request, songs, mode, ai_config, limit=100, include_audio=False,
    log_messages=None, target_count=None, resolved_seed=None,
):
    """Call the configured provider and return strictly validated candidate IDs."""
    from tasks.ai.api import generate_text

    mode = str(mode or "").upper()
    pool = list(songs[:limit])
    rerank_context = _rerank_context_key(user_request, resolved_seed)
    pool, item_id_to_alias, alias_to_song, records = prepare_llm_candidate_shortlist(
        pool, rerank_context, limit=len(pool)
    )
    if len(records) < 1:
        logger.warning("Curator skipped: no usable candidates (mode=%s)", mode)
        return [], len(records)
    candidate_ids = list(alias_to_song)
    candidate_titles = {alias: song.get("title") for alias, song in alias_to_song.items()}
    serialized_records = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    requested_curated = 20 if target_count is None else max(1, int(target_count))
    max_curated = min(requested_curated, len(records))
    effective_target = requested_curated
    rerank_required_count = min(len(records), effective_target + _RERANK_SAFETY_MARGIN)
    if mode == "LLM_RERANK":
        logger.info(
            "Selection mode: %s; candidates available: %d; candidates sent to curator: %d",
            mode, len(songs), len(records),
        )
    else:
        logger.info(
            "Selection mode: %s; candidates available: %d; candidates sent to curator: %d",
            mode, len(songs), len(records),
        )
    context_setting = (
        ai_config.get("ollama_num_ctx")
        or getattr(config, "OLLAMA_NUM_CTX", None)
        or os.environ.get("OLLAMA_NUM_CTX")
        or os.environ.get("OLLAMA_CONTEXT_SIZE")
        or "not configured (Ollama server/model default)"
    )
    provider = str(ai_config.get("provider") or "unknown").upper()
    model = ai_config.get(f"{provider.lower()}_model") or "unknown"
    # Provider helpers log the sanitized endpoint; never write configured URLs
    # here because they may contain credentials or query parameters.
    logger.info("Curator provider call started: provider=%s model=%s", provider, model)
    if log_messages is not None:
        log_messages.append(f"Curator provider call started: {provider}/{model}")
    if provider == "OLLAMA":
        try:
            from tasks.ai.providers.openai import _ollama_endpoints, _safe_endpoint
            endpoint = _safe_endpoint(_ollama_endpoints(ai_config.get("ollama_url") or "")[0])
            logger.info("Curator endpoint: %s", endpoint)
            logger.info("Curator timeout: %ss", config.AI_REQUEST_TIMEOUT_SECONDS)
            if log_messages is not None:
                log_messages.append(f"Curator endpoint: {endpoint}")
        except Exception as exc:
            logger.warning("Could not resolve sanitized curator endpoint (%s)", type(exc).__name__)
    if mode == "LLM_RERANK":
        task_contract = (
            "Rank every candidate you can evaluate from most to least appropriate for this playlist. "
            "Use your musical knowledge of artists, songs, albums, genres, style, and scene/subgenre relationships. "
            "Judge suitability for this request; broad genre overlap alone is not enough. "
            "Every candidate is real and belongs to the user's library. Do not invent songs. "
            "Return only supplied candidate IDs, each at most once, in ranked_ids. "
            "Candidate presentation order has no significance. "
            f"Return at least {rerank_required_count} unique IDs when possible. "
            "Return one JSON object with only ranked_ids.\n"
        )
    else:
        task_contract = (
            f"Select at most {max_curated} candidates that genuinely belong in the requested playlist. "
            "Use your knowledge of artists, songs, albums, musical style, genre/subgenre, "
            "scene relationships, and the user's stated intent. Do not select a song merely "
            "because it is broadly in the same genre. You may return fewer than this limit "
            "when other candidates are weak. Candidate order has no significance. "
            "Do not invent songs. Return only candidate IDs in selected_ids as one JSON object.\n"
        )
    seed_record = None
    if resolved_seed and resolved_seed.get("item_id") is not None:
        seed_key = str(resolved_seed["item_id"])
        seed_alias = item_id_to_alias.get(seed_key)
        if seed_alias:
            seed_record = next((record for record in records if record["id"] == seed_alias), None)
        else:
            seed_payload = build_llm_candidate_payload(
                [resolved_seed], aliases={seed_key: "SEED"}
            )
            seed_record = seed_payload[0] if seed_payload else None
        if seed_record:
            seed_record = {key: value for key, value in seed_record.items() if key != "id"}
    if mode == "LLM_RERANK":
        seed_text = "Unknown"
        if seed_record:
            seed_text = f"{seed_record.get('title', 'Unknown')} by {seed_record.get('artist', 'Unknown')}"
            if seed_record.get("album"):
                seed_text += f"\nAlbum: {seed_record['album']}"
        prompt = (
            "You are choosing music for a playlist from the user's own music library.\n\n"
            f"User request:\n{user_request}\n\n"
            f"Resolved seed:\n{seed_text}\n\n"
            "Below are real candidate songs from the user's library. Rank them from most appropriate to least appropriate "
            "for the user's request. Use your knowledge of artists, songs, albums, musical genres, musical style, and "
            "scene/subgenre relationships. Do not treat two songs as close merely because both are broadly classified "
            "as rock or metal. Judge how suitable each candidate is for THIS playlist. Every candidate is real and already "
            "exists in the user's library. Do not invent songs. Return only candidate IDs in ranked_ids. Prefer candidates "
            "that best fit the user's actual musical intent. Candidate presentation order has no significance.\n\n"
            f"{task_contract}Candidates: {serialized_records}"
        )
    else:
        prompt = (
            "You are curating a playlist from songs that already exist in the user's music library.\n\n"
            f"User request:\n{user_request}\n\n"
            "Resolved explicit/seed songs:\n"
            f"{json.dumps([seed_record] if seed_record else [], ensure_ascii=False, separators=(',', ':'))}\n\n"
            "Below are candidate songs retrieved from the user's library.\n"
            f"{task_contract}"
            "Return JSON with this shape: {\"selected_ids\":[\"alias\"]}.\n\n"
            f"Candidates: {serialized_records}"
        )
    serialized_chars = len(serialized_records)
    approximate_prompt_tokens = (len(prompt) + 3) // 4
    if log_messages is not None:
        log_messages.append("LLM candidate payload: metadata only")
        log_messages.append(f"LLM shortlist size: {len(records)}")
        log_messages.append("AudioMuse scores sent to LLM: no")
        log_messages.append("AudioMuse native ranks sent to LLM: no")
        log_messages.append("Candidate presentation shuffled: yes")
        if mode == "LLM_RERANK":
            for alias, song in list(alias_to_song.items())[:5]:
                log_messages.append(
                    f"Rerank alias mapping: {alias} -> "
                    f"{song.get('artist') or song.get('author') or 'Unknown'} - {song.get('title') or 'Unknown'}"
                )
        log_messages.append(f"Serialized candidate payload chars: {serialized_chars}")
        log_messages.append(f"Approximate prompt tokens: {approximate_prompt_tokens}")
        log_messages.append(f"Effective LLM output token budget: {_LLM_CANDIDATE_MAX_OUTPUT_TOKENS}")
        if provider == "OLLAMA":
            log_messages.append(f"Configured Ollama context size: {context_setting}")
    logger.info(
        "Curator prompt size: candidates_sent=%d payload_chars=%d approximate_prompt_tokens=%d ollama_context=%s",
        len(records), serialized_chars, approximate_prompt_tokens,
        context_setting if provider == "OLLAMA" else "not applicable",
    )
    logger.info("Effective LLM output token budget: %d", _LLM_CANDIDATE_MAX_OUTPUT_TOKENS)
    response_key = "ranked_ids" if mode == "LLM_RERANK" else "selected_ids"
    structured_schema = {
        "type": "object",
        "properties": {
            response_key: {
                "type": "array",
                "items": {"type": "string"},
                **({"minItems": rerank_required_count, "uniqueItems": True} if mode == "LLM_RERANK" else {}),
                **({"maxItems": len(records)} if mode == "LLM_RERANK" else {}),
            }
        },
        "required": [response_key],
        **({"additionalProperties": False} if mode == "LLM_RERANK" else {}),
    }
    def _request_and_inspect(current_prompt):
        provider_options = {}
        if provider == "OLLAMA":
            provider_options["think"] = False
        raw_response = generate_text(
            current_prompt, ai_config, skip_delay=True, temperature=0.1,
            max_tokens=_LLM_CANDIDATE_MAX_OUTPUT_TOKENS,
            structured_format=structured_schema if provider == "OLLAMA" else None,
            system_prompt="You curate playlist tracks. Return only the required JSON object.",
            **provider_options,
        )
        safe_response = str(raw_response or "")
        safe_response = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", safe_response)
        safe_response = re.sub(r"(?i)(api[\s_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", safe_response)
        logger.info(
            "Curator raw response metadata: type=%s length=%d",
            type(raw_response).__name__, len(safe_response),
        )
        if log_messages is not None:
            log_messages.append(f"Curator raw response length: {len(safe_response)} chars")
        if safe_response.strip().startswith("Error:"):
            return raw_response, [], None, safe_response
        ids, diagnostics = inspect_llm_candidate_selection(
            raw_response, candidate_ids, mode, candidate_titles
        )
        if mode == "LLM_CURATE" and len(ids) > max_curated:
            if log_messages is not None:
                log_messages.append(
                    f"Curator selection capped to effective target: {max_curated}"
                )
            ids = ids[:max_curated]
        return raw_response, ids, diagnostics, None

    def _log_rerank_coverage(ids, attempt):
        coverage = len(ids) / len(records) if records else 0.0
        percent = round(coverage * 100)
        status = "SUCCESS" if len(ids) >= rerank_required_count else "INCOMPLETE"
        logger.info(
            "Rerank attempt %d target=%d candidates=%d required=%d returned_valid=%d coverage=%.1f%% status=%s",
            attempt, effective_target, len(records), rerank_required_count, len(ids), coverage * 100, status,
        )
        if log_messages is not None:
            log_messages.append(f"Effective target: {effective_target}")
            log_messages.append(f"Rerank candidates sent: {len(records)}")
            log_messages.append(f"Rerank required usable aliases: {rerank_required_count}")
            log_messages.append(f"Valid reranked aliases: {len(ids)}")
            log_messages.append(f"LLM returned valid ranked candidates: {len(ids)}")
            log_messages.append(f"Rerank returned aliases: {len(ids)}")
            log_messages.append(f"Rerank coverage: {percent}%")
            log_messages.append(f"Rerank status: {status}")
        return len(ids) >= rerank_required_count

    try:
        raw, ids, diag, provider_error = _request_and_inspect(prompt)
        if provider_error:
            logger.error("Curator status: PROVIDER_ERROR")
            logger.error("Curator provider call failed: provider=%s model=%s transport/provider error=%s", provider, model, provider_error)
            if log_messages is not None:
                log_messages.append(f"Curator status: PROVIDER_ERROR ({provider_error})")
            return [], len(records)

        if mode == "LLM_RERANK":
            rerank_sufficient = _log_rerank_coverage(ids, 1)
            if not rerank_sufficient:
                if log_messages is not None:
                    log_messages.append("Rerank response incomplete; retrying once with a stricter compact prompt")
                retry_prompt = (
                    "Your previous ranked_ids response did not contain enough unique valid aliases. "
                    f"Return at least {rerank_required_count} unique supplied candidate aliases, "
                    "ordered from most to least relevant. Return only the required JSON object.\n"
                    + prompt
                )
                raw, ids, diag, provider_error = _request_and_inspect(retry_prompt)
                if provider_error:
                    if log_messages is not None:
                        log_messages.append(f"Curator retry status: PROVIDER_ERROR ({provider_error})")
                    return [], len(records)
                rerank_sufficient = _log_rerank_coverage(ids, 2)
                if not rerank_sufficient:
                    if log_messages is not None:
                        log_messages.append(
                            f"Curator JSON parsing: {'success' if diag['parsed_json'] is not None else 'failure'}"
                        )
                        log_messages.append(f"Valid aliases: {len(ids)}")
                        if diag["rejection_reason"]:
                            log_messages.append(f"Curator validation: {diag['rejection_reason']}")
                        log_messages.append("Curator status: INCOMPLETE_RERANK")
                        log_messages.append("LLM rerank unusable: incomplete after retry; falling back to Native")
                    return [], len(records)

        status = "PARSE_ERROR" if diag["parsed_json"] is None else "SUCCESS" if ids else "VALIDATION_ERROR"
        logger.info("Curator status: %s", status)
        logger.info(
            "Curator extracted response metadata: content_length=%s parsed_type=%s thinking_present=%s",
            len(diag["extracted_content"] or ""), diag["json_top_level_type"],
            diag["thinking_present"],
        )
        logger.info("Curator JSON parsing: %s", "success" if diag["parsed_json"] is not None else "failure")
        logger.info("Curator HTTP extracted content type: %s", diag["http_extracted_content_type"])
        logger.info("Curator JSON top-level type: %s", diag["json_top_level_type"])
        logger.info("Curator normalization input type: %s", diag["normalization_input_type"])
        logger.info("Curator normalization output aliases: %s", diag["normalization_output_aliases"])
        logger.info("Curator raw response shape: %s", diag["raw_response_shape"])
        logger.info("Curator returned keys: %s", diag["returned_keys"])
        logger.info("Curator response normalization: %s", diag["normalization_note"])
        logger.info("Normalized alias count: %d", diag["normalized_alias_count"])
        if log_messages is not None:
            log_messages.append("Curator provider call succeeded")
            log_messages.append(
                f"Curator JSON parsing: {'success' if diag['parsed_json'] is not None else 'failure'}"
            )
            log_messages.append(f"Curator HTTP extracted content type: {diag['http_extracted_content_type']}")
            log_messages.append(f"Curator JSON top-level type: {diag['json_top_level_type']}")
            log_messages.append(f"Curator normalization input type: {diag['normalization_input_type']}")
            log_messages.append(f"Curator normalization output aliases: {diag['normalization_output_aliases']}")
            log_messages.append(f"Curator status: {status}")
            log_messages.append(f"Curator raw response shape: {diag['raw_response_shape']}")
            log_messages.append(f"Returned keys: {diag['returned_keys']}")
            log_messages.append(f"Curator response normalization: {diag['normalization_note']}")
            log_messages.append(f"Normalized alias count: {diag['normalized_alias_count']}")
            log_messages.append(f"Aliases returned: {diag['returned_id_count']}")
            log_messages.append(f"Valid aliases: {len(ids)}")
            log_messages.append(f"Invalid aliases: {len(diag['invalid_ids'])}")
            log_messages.append(f"Duplicates removed: {len(diag['duplicates_removed'])}")
            if diag["rejection_reason"]:
                log_messages.append(f"Curator validation: {diag['rejection_reason']}")
        logger.info(
            "Curator diagnostics mode=%s returned_aliases=%d parsed_ids=%s valid_aliases=%d "
            "invalid_aliases=%d duplicates_removed=%d rejection_reason=%s",
            mode, diag["returned_id_count"], diag["first_returned_ids"], len(ids),
            len(diag["invalid_ids"]), len(diag["duplicates_removed"]), diag["rejection_reason"],
        )
        logger.info("Parsed aliases: %s", diag["first_returned_ids"])
        logger.info("Valid aliases after validation: %d", len(ids))
        if mode == "LLM_CURATE":
            if log_messages is not None:
                log_messages.append(f"Curator selected aliases: {len(ids)}")
        authoritative = [str(alias_to_song[alias]["item_id"]) for alias in ids if alias in alias_to_song]
        logger.info("After mapping aliases: authoritative tracks recovered=%d", len(authoritative))
        if log_messages is not None:
            log_messages.append(f"Authoritative tracks recovered: {len(authoritative)}")
        return authoritative, len(records)
    except Exception:
        logger.exception("Curator provider call failed (mode=%s)", mode)
        if log_messages is not None:
            log_messages.append("Curator provider/API exception; see server log for exception details")
        return [], len(records)


def rank_candidates_by_ids(songs, ranked_ids, mandatory_ids=(), include_unselected=True):
    """Apply exact-ID ranking while preserving mandatory tracks and optionally the pool."""
    by_id = {str(song.get("item_id")): song for song in songs if song.get("item_id") is not None}
    result, seen = [], set()
    for item_id in list(mandatory_ids) + list(ranked_ids):
        key = str(item_id)
        if key in by_id and key not in seen:
            result.append(by_id[key])
            seen.add(key)
    if include_unselected:
        for song in songs:
            key = str(song.get("item_id"))
            if key in by_id and key not in seen:
                result.append(song)
                seen.add(key)
    return result


def reorder_preserving_membership(songs, ordered_ids):
    """Apply an order result without allowing it to add or drop playlist members."""
    return rank_candidates_by_ids(songs, ordered_ids, include_unselected=True)


def optimize_playlist_duration(
    songs, durations, target_seconds, count, max_per_artist, exact_count=False,
    mandatory_ids=(), ranked_ids=None, tolerance_seconds=15, diagnostics=None,
):
    """Choose a duration-valid subset, preferring candidates with better ranks."""
    if not songs or target_seconds <= 0 or count <= 0:
        return []
    by_id, ordered = {}, []
    for song in songs:
        item_id = song.get("item_id")
        if item_id is None or str(item_id) in by_id:
            continue
        by_id[str(item_id)] = song
        ordered.append(song)
    mandatory = [by_id[k] for k in dict.fromkeys(map(str, mandatory_ids)) if k in by_id]
    mandatory_set = {str(song["item_id"]) for song in mandatory}

    def duration_for(song):
        value = durations.get(song.get("item_id"), durations.get(str(song.get("item_id"))))
        try:
            seconds = int(float(value))
            return seconds if seconds > 0 else None
        except (TypeError, ValueError):
            return None

    mandatory_durations = [duration_for(song) for song in mandatory]
    if any(seconds is None for seconds in mandatory_durations):
        logger.warning("Duration optimizer cannot size a mandatory track with missing duration")
        if diagnostics is not None:
            diagnostics.update({"missing_mandatory_duration": True, "solutions_in_tolerance": 0})
        return mandatory
    seed_seconds = sum(mandatory_durations)
    cap = max(1, int(max_per_artist or 1))
    artist_counts = {}
    for song in mandatory:
        artist = song.get("artist") or "Unknown"
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
    if any(n > cap for n in artist_counts.values()):
        logger.warning("Mandatory tracks exceed artist cap; preserving mandatory tracks")

    rank_order = []
    seen_rank_ids = set()
    for item_id in ranked_ids or [song.get("item_id") for song in ordered]:
        key = str(item_id)
        if key not in seen_rank_ids and key in by_id and key not in mandatory_set:
            rank_order.append(key)
            seen_rank_ids.add(key)
    for song in ordered:
        key = str(song["item_id"])
        if key not in mandatory_set and key not in seen_rank_ids:
            rank_order.append(key)
            seen_rank_ids.add(key)
    rank_by_id = {item_id: rank for rank, item_id in enumerate(rank_order, start=1)}

    candidates = []
    for item_id in rank_order:
        song = by_id[item_id]
        if str(song["item_id"]) in mandatory_set:
            continue
        artist = song.get("artist") or "Unknown"
        if artist_counts.get(artist, 0) >= cap:
            continue
        seconds = duration_for(song)
        if seconds is None:
            continue
        artist_counts[artist] = artist_counts.get(artist, 0) + 1
        candidates.append((rank_by_id[str(song["item_id"])], song, seconds))

    slots = max(0, count - len(mandatory))
    # Each state stores rank cost, repeat-artist penalty, chosen candidate
    # indices, and an artist bitset. Candidate ranks are supplied by the
    # curator order, reranker order, or native pool order.
    artist_indexes = {}
    for _, song, _ in candidates:
        artist = str(song.get("artist") or "Unknown")
        artist_indexes.setdefault(artist, len(artist_indexes))
    mandatory_mask = 0
    for song in mandatory:
        artist = str(song.get("artist") or "Unknown")
        bit = 1 << artist_indexes.setdefault(artist, len(artist_indexes))
        mandatory_mask |= bit
    states = {(0, 0): (0, 0, (), mandatory_mask)}
    for idx, (_, _, seconds) in enumerate(candidates):
        additions = {}
        rank, song, _ = candidates[idx]
        artist = str(song.get("artist") or "Unknown")
        artist_bit = 1 << artist_indexes[artist]
        for (n, total), (rank_cost, diversity_cost, chosen, artist_mask) in list(states.items()):
            if n < slots:
                key = (n + 1, total + seconds)
                new_chosen = chosen + (idx,)
                candidate_state = (
                    rank_cost + rank,
                    diversity_cost + int(bool(artist_mask & artist_bit)),
                    new_chosen,
                    artist_mask | artist_bit,
                )
                old = additions.get(key, states.get(key))
                if old is None or (
                    candidate_state[0], candidate_state[1], candidate_state[2]
                ) < (old[0], old[1], old[2]):
                    additions[key] = candidate_state
        states.update(additions)
    eligible = list(states.items())
    if exact_count and slots and any(n == slots for (n, _), _ in eligible):
        eligible = [(key, chosen) for key, chosen in eligible if key[0] == slots]
    target = int(target_seconds)
    tolerance = max(0, int(tolerance_seconds))
    valid = [
        pair for pair in eligible
        if abs(seed_seconds + pair[0][1] - target) <= tolerance
    ]

    def score(pair, *, in_band):
        (n, subtotal), (rank_cost, diversity_cost, chosen, _mask) = pair
        duration_error = abs(seed_seconds + subtotal - target)
        rank_tiebreak = tuple(candidates[i][0] for i in chosen)
        if in_band:
            return rank_cost, diversity_cost, duration_error, rank_tiebreak, -n
        return duration_error, rank_cost, diversity_cost, rank_tiebreak, -n

    selected = min(valid, key=lambda pair: score(pair, in_band=True)) if valid else min(
        eligible, key=lambda pair: score(pair, in_band=False)
    )
    (selected_count, selected_subtotal), (rank_cost, diversity_cost, chosen, _mask) = selected
    selected_duration = seed_seconds + selected_subtotal
    ranks = [candidates[i][0] for i in chosen]
    exact = [pair for pair in eligible if seed_seconds + pair[0][1] == target]
    best_exact = min(exact, key=lambda pair: score(pair, in_band=True)) if exact else None
    if diagnostics is not None:
        diagnostics.update({
            "input_candidates": len(candidates) + len(mandatory),
            "solutions_in_tolerance": len(valid),
            "tolerance_seconds": tolerance,
            "selected_duration": selected_duration,
            "duration_error": selected_duration - target,
            "ranking_cost": rank_cost,
            "diversity_penalty": diversity_cost,
            "selected_ranks": ranks,
            "average_rank": (sum(ranks) / len(ranks)) if ranks else None,
            "worst_rank": max(ranks) if ranks else None,
            "best_exact_duration": target if best_exact is not None else None,
            "best_exact_ranking_cost": best_exact[1][0] if best_exact is not None else None,
            "selected_count": selected_count + len(mandatory),
        })
    return mandatory + [candidates[i][1] for i in chosen]
