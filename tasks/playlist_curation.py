"""Grounded optional curation helpers for Instant Playlist."""

import json
import logging
import os
import re
import unicodedata
from decimal import Decimal

import config
from tasks.ai.json_response import parse_json_response

logger = logging.getLogger(__name__)


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
    """Build compact curator records; IDs are request-local aliases when supplied."""
    features = features or {}
    records = []
    for song in songs:
        item_id = song.get("item_id")
        if item_id is None:
            continue
        alias = aliases.get(str(item_id)) if aliases is not None else str(item_id)
        if alias is None:
            continue
        row = {"id": alias}
        for key in ("title", "artist", "album", "duration_seconds", "duration"):
            value = song.get(key)
            if value is not None and value != "":
                row["duration_seconds" if key == "duration" else key] = (
                    float(value) if isinstance(value, Decimal) else value
                )
        feat = features.get(item_id, {})
        if feat.get("duration") is not None:
            value = feat["duration"]
            row["duration_seconds"] = float(value) if isinstance(value, Decimal) else value
        for key in ("tempo", "energy", "year", "key", "scale"):
            value = feat.get(key)
            if value is not None and value != "":
                row[key] = float(value) if isinstance(value, Decimal) else value
        records.append(row)
    return records


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


def curate_candidates_with_llm(
    user_request, songs, mode, ai_config, limit=100, include_audio=False,
    log_messages=None,
):
    """Call the configured provider and return strictly validated candidate IDs."""
    from tasks.ai.api import generate_text

    mode = str(mode or "").upper()
    pool = list(songs[:limit])
    features = {}
    if include_audio:
        try:
            from tasks.ai.tool_impl import _fetch_pool_features
            features = _fetch_pool_features([s["item_id"] for s in pool])
        except Exception:
            logger.warning("Could not fetch curator audio features", exc_info=True)
    # The model sees only compact, request-local aliases. The authoritative
    # AudioMuse objects and IDs remain server-side and are recovered below.
    alias_to_song = {}
    seen_item_ids = set()
    for song in pool:
        item_id = song.get("item_id")
        if item_id is not None and str(item_id) not in seen_item_ids:
            alias_to_song[f"C{len(alias_to_song) + 1:03d}"] = song
            seen_item_ids.add(str(item_id))
    item_id_to_alias = {str(song["item_id"]): alias for alias, song in alias_to_song.items()}
    records = build_llm_candidate_payload(pool, features, item_id_to_alias)
    if len(records) < 1:
        logger.warning("Curator skipped: no usable candidates (mode=%s)", mode)
        return [], len(records)
    candidate_ids = list(alias_to_song)
    candidate_titles = {alias: song.get("title") for alias, song in alias_to_song.items()}
    alias_range = f"{candidate_ids[0]}-{candidate_ids[-1]}" if candidate_ids else "empty"
    serialized_records = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    max_curated = min(20, len(records))
    logger.info(
        "Selection mode: %s; candidates available: %d; candidates sent to curator: %d; alias range: %s",
        mode, len(songs), len(records), alias_range,
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
            "Rank the supplied candidates from best to worst for the request. "
            f"Return every alias in ranked order in one JSON object with only ranked_ids; include up to {len(records)} aliases.\n"
        )
    else:
        task_contract = (
            f"Select about {max_curated} of the best candidates, in preference order. "
            "Return one JSON object with only selected_ids. Do not return analysis, summaries, metadata, or explanations.\n"
        )
    prompt = (
        "Use only candidate IDs from this list. Output one JSON object and nothing else.\n"
        f"Request: {user_request}\n{task_contract}"
        f"Candidates: {serialized_records}"
    )
    serialized_chars = len(serialized_records)
    approximate_prompt_tokens = (len(prompt) + 3) // 4
    if log_messages is not None:
        log_messages.append(f"Native shortlist for curator: {len(records)}")
        log_messages.append(f"Serialized candidate payload chars: {serialized_chars}")
        log_messages.append(f"Approximate prompt tokens: {approximate_prompt_tokens}")
        if provider == "OLLAMA":
            log_messages.append(f"Configured Ollama context size: {context_setting}")
    logger.info(
        "Curator prompt size: candidates_sent=%d payload_chars=%d approximate_prompt_tokens=%d ollama_context=%s",
        len(records), serialized_chars, approximate_prompt_tokens,
        context_setting if provider == "OLLAMA" else "not applicable",
    )
    response_key = "ranked_ids" if mode == "LLM_RERANK" else "selected_ids"
    structured_schema = {
        "type": "object",
        "properties": {
            response_key: {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": len(records) if mode == "LLM_RERANK" else max_curated,
            }
        },
        "required": [response_key],
        "additionalProperties": False,
    }
    try:
        raw = generate_text(
            prompt, ai_config, skip_delay=True, temperature=0.1,
            max_tokens=1200 if mode == "LLM_CURATE" else 1800,
            structured_format=structured_schema if provider == "OLLAMA" else None,
            system_prompt="You curate playlist tracks. Return only the required JSON object.",
        )
        safe_raw = str(raw or "")
        safe_raw = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", safe_raw)
        safe_raw = re.sub(r"(?i)(api[\s_-]?key\s*[:=]\s*)\S+", r"\1[REDACTED]", safe_raw)
        logger.info("Raw curator content (log preview): %s", safe_raw[:4000])
        if log_messages is not None:
            log_messages.append(f"Raw curator content: {safe_raw[:4000]}")
        if safe_raw.strip().startswith("Error:"):
            logger.error("Curator status: PROVIDER_ERROR")
            logger.error("Curator provider call failed: provider=%s model=%s transport/provider error=%s", provider, model, safe_raw)
            if log_messages is not None:
                log_messages.append(f"Curator status: PROVIDER_ERROR ({safe_raw})")
            return [], len(records)
        logger.info("Curator provider call succeeded")
        if log_messages is not None:
            log_messages.append("Curator provider call succeeded")
        ids, diag = inspect_llm_candidate_selection(raw, candidate_ids, mode, candidate_titles)
        if diag["parsed_json"] is None:
            status = "PARSE_ERROR"
        elif not ids:
            status = "VALIDATION_ERROR"
        else:
            status = "SUCCESS"
        logger.info("Curator status: %s", status)
        logger.info(
            "Curator extracted content=%r parsed_json=%r thinking_present=%s",
            diag["extracted_content"], diag["parsed_json"], diag["thinking_present"],
        )
        logger.info("Curator JSON parsing: %s", "success" if diag["parsed_json"] is not None else "failure")
        logger.info("Curator HTTP extracted content type: %s", diag["http_extracted_content_type"])
        logger.info("Curator JSON top-level type: %s", diag["json_top_level_type"])
        logger.info("Curator normalization input type: %s", diag["normalization_input_type"])
        logger.info("Curator normalization output aliases: %s", diag["normalization_output_aliases"])
        parsed_preview = json.dumps(diag["parsed_json"], ensure_ascii=False, separators=(",", ":")) if diag["parsed_json"] is not None else "null"
        logger.info("Curator parsed JSON (log preview): %s", parsed_preview[:4000])
        logger.info("Curator raw response shape: %s", diag["raw_response_shape"])
        logger.info("Curator returned keys: %s", diag["returned_keys"])
        logger.info("Curator response normalization: %s", diag["normalization_note"])
        logger.info("Normalized alias count: %d", diag["normalized_alias_count"])
        if log_messages is not None:
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
            log_messages.append(
                f"Invalid aliases: {len(diag['invalid_ids'])}; duplicates removed: {len(diag['duplicates_removed'])}"
            )
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
