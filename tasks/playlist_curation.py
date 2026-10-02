"""Grounded optional curation helpers for Instant Playlist."""

import base64
import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter, deque

import config
from tasks.ai.json_response import parse_json_response

logger = logging.getLogger(__name__)
_RERANK_SAFETY_MARGIN = 5
_LLM_CANDIDATE_MAX_OUTPUT_TOKENS = config.INSTANT_PLAYLIST_LLM_OUTPUT_TOKENS


def _composer_output_budget():
    """Return the configured technical generation ceiling."""
    return config.COMPOSER_MAX_OUTPUT_TOKENS


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


def _balanced_candidate_shortlist(songs, seed_provenance, limit, mandatory_ids=()):
    """Allocate bounded slots across seeds and artists without choosing the playlist."""
    limit = max(0, int(limit))
    by_id = {str(song.get('item_id')): song for song in songs if song.get('item_id') is not None}
    selected, selected_ids = [], set()
    for item_id in mandatory_ids or ():
        key = str(item_id)
        if key in by_id and key not in selected_ids and len(selected) < limit:
            selected.append(by_id[key])
            selected_ids.add(key)

    neighborhoods = []
    if isinstance(seed_provenance, dict):
        for label, ids in seed_provenance.items():
            group = [by_id[str(item_id)] for item_id in ids if str(item_id) in by_id]
            if group:
                artist_queues = {}
                for song in group:
                    artist = str(song.get('artist') or song.get('author') or 'Unknown').casefold()
                    artist_queues.setdefault(artist, deque()).append(song)
                varied_group = []
                while any(artist_queues.values()):
                    for queue in artist_queues.values():
                        if queue:
                            varied_group.append(queue.popleft())
                neighborhoods.append((str(label), varied_group))
    # Round-robin through ranked members of each group; overlap is de-duplicated
    # and unused capacity is filled from the stable global candidate order.
    rank = 0
    while len(selected) < limit:
        added_this_round = False
        for _label, group in neighborhoods:
            if rank >= len(group):
                continue
            song = group[rank]
            key = str(song.get('item_id'))
            if key not in selected_ids:
                selected.append(song)
                selected_ids.add(key)
                added_this_round = True
                if len(selected) >= limit:
                    break
        if not neighborhoods or rank >= max((len(group) for _, group in neighborhoods), default=0):
            break
        rank += 1
        if not added_this_round and rank >= max((len(group) for _, group in neighborhoods), default=0):
            break
    for song in songs:
        key = str(song.get('item_id'))
        if key not in selected_ids:
            selected.append(song)
            selected_ids.add(key)
            if len(selected) >= limit:
                break
    return selected[:limit], [label for label, group in neighborhoods if any(
        str(song.get('item_id')) in selected_ids for song in group
    )]


def prepare_llm_candidate_shortlist(songs, context, limit=None, seed_provenance=None, mandatory_ids=()):
    """Balance, de-duplicate, shuffle, alias, and serialize one LLM shortlist."""
    if limit is None:
        limit = config.INSTANT_PLAYLIST_COMPOSER_MAX_CANDIDATES or len(songs)
    bounded, represented = _balanced_candidate_shortlist(
        songs, seed_provenance, limit, mandatory_ids=mandatory_ids
    )
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
    r"single\s+version|album\s+version|\bversion\b|mono\s+version|stereo\s+version|"
    r"evolution\s+track|bonus\s+track)",
    re.I,
)
_TRAILING_VERSION_RE = re.compile(
    r"\s*(?:\(([^()]*)\)|\[([^\[\]]*)\]|\s[-–—]\s([^()\[\]]+))\s*$"
)


def is_alternate_recording(song):
    """Identify explicit version labels in library titles, not request language."""
    title = str(song.get("title") or "").strip()
    if re.search(r"\bversion$", title, re.I):
        return True
    match = _TRAILING_VERSION_RE.search(title)
    if not match:
        return False
    suffix = next((value for value in match.groups() if value is not None), "")
    return bool(_VERSION_MARKER_RE.search(suffix))


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
            chosen = mandatory_family
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
    log_messages=None, target_count=None, resolved_seed=None, semantic_intent=None,
    seed_provenance=None, mandatory_ids=(), ui_default_count=None,
):
    """Call the configured provider and return strictly validated candidate IDs."""
    from tasks.ai.api import generate_text

    mode = str(mode or "").upper()
    pool = list(songs)
    rerank_context = _rerank_context_key(user_request, resolved_seed)
    pool, item_id_to_alias, alias_to_song, records = prepare_llm_candidate_shortlist(
        pool, rerank_context, limit=limit, seed_provenance=seed_provenance,
        mandatory_ids=mandatory_ids,
    )
    selected_ids = {str(song.get("item_id")) for song in pool}
    represented_seeds = [
        label for label, ids in (seed_provenance or {}).items()
        if any(str(item_id) in selected_ids for item_id in ids)
    ]
    if len(records) < 1:
        logger.warning("Curator skipped: no usable candidates (mode=%s)", mode)
        return [], len(records)
    candidate_ids = list(alias_to_song)
    candidate_titles = {alias: song.get("title") for alias, song in alias_to_song.items()}
    serialized_records = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    curator_intent = {
        key: semantic_intent.get(key)
        for key in (
            'anchors', 'count', 'duration_seconds', 'constraints', 'playlist_intent',
            'activity', 'lyrical_theme', 'transition_intent', 'ordering_intent',
            'diversity_intent', 'similarity_intent',
        )
    } if isinstance(semantic_intent, dict) else {}
    intent_block = (
        "Authoritative planner interpretation (use this to judge candidate relevance; "
        "do not change its anchors, count, duration, or constraints):\n"
        + json.dumps(curator_intent, ensure_ascii=False, separators=(",", ":"))
        + "\n\n"
    )
    requested_curated = 20 if target_count is None else max(1, int(target_count))
    max_curated = len(records)
    effective_target = requested_curated
    rerank_required_count = min(len(records), max(1, int(effective_target * 1.2 + 0.999)))
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
            "Rank supplied candidates from most to least appropriate for the authoritative planner intent. "
            "Do not reinterpret or alter the requested count, duration, anchors, or constraints. "
            "Every candidate is real and belongs to the user's library. Do not invent songs. "
            f"Return at least {rerank_required_count} unique IDs when possible. "
            'Return only a JSON object shaped like {"ranked_ids":["C..."]}.\n'
        )
    else:
        task_contract = (
            f"Select candidates that best fit the authoritative planner intent, up to {max_curated}. "
            "Do not reinterpret or alter the requested count, duration, anchors, or constraints. "
            "Use musical knowledge to judge actual suitability, not broad genre overlap alone. "
            "Do not invent songs. Candidate order has no significance. "
            'Return only a JSON object shaped like {"selected_ids":["C..."]}.\n'
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
            f"{intent_block}"
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
            f"{intent_block}"
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
        if seed_provenance:
            log_messages.append(
                f"Seed neighborhoods represented: {len(represented_seeds)}/{len(seed_provenance)}"
            )
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
        "type": "object", "additionalProperties": False,
        "properties": {response_key: {
            "type": "array", "items": {"type": "string"},
            **({"minItems": rerank_required_count} if mode == "LLM_RERANK" else {}),
        }},
        "required": [response_key],
    }
    def _request_and_inspect(current_prompt):
        provider_options = {"think": False} if provider == "OLLAMA" else {}
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
        logger.info("Curator raw response metadata: type=%s length=%d", type(raw_response).__name__, len(safe_response))
        if log_messages is not None:
            log_messages.append(f"Curator raw response length: {len(safe_response)} chars")
        if safe_response.strip().startswith("Error:"):
            return raw_response, [], None, safe_response
        parsed_response, _, _, response_parse_error = parse_json_response(raw_response)
        if response_parse_error or not isinstance(parsed_response, dict):
            return raw_response, [], None, "invalid structured ID response"
        ids, diagnostics = inspect_llm_candidate_selection(
            json.dumps(parsed_response, ensure_ascii=False), candidate_ids, mode, candidate_titles
        )
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
        if mode == "LLM_CURATE":
            if len(ids) > effective_target and log_messages is not None:
                log_messages.append(f"Curator selection capped to effective target: {effective_target}")
            ids = ids[:effective_target]
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


def compose_playlist_with_llm(
    user_request, songs, ai_config, *, seed_provenance=None,
    resolved_anchors=None, ui_default_count=None, log_messages=None,
    duration_rows=None, musical_review=False, expansion_callback=None,
    preinterpreted_intent=None,
):
    """Compose one ordered final playlist from the original request and library candidates."""
    from tasks.ai.api import generate_text

    pool = list(songs)
    capacity = max(0, int(config.INSTANT_PLAYLIST_COMPOSER_MAX_CANDIDATES))
    limit = capacity or len(pool)
    context = _rerank_context_key(user_request, {"compose": True})
    selected, item_id_to_alias, alias_to_song, records = prepare_llm_candidate_shortlist(
        pool, context, limit=limit, seed_provenance=seed_provenance,
        mandatory_ids=[
            anchor.get("resolved_track", {}).get("item_id")
            for anchor in (resolved_anchors or [])
            if isinstance(anchor, dict) and anchor.get("resolved_track", {}).get("item_id") is not None
        ],
    )
    if not records:
        return {"playlist": [], "requested_output": {}, "shortfall_reason": "No candidates were available."}, 0

    candidate_ids = set(alias_to_song)
    anchor_records = []
    for anchor in resolved_anchors or []:
        if not isinstance(anchor, dict):
            continue
        if anchor.get("type") != "song":
            continue
        track = anchor.get("resolved_track") or {}
        alias = item_id_to_alias.get(str(track.get("item_id")))
        user_reference = anchor.get("user_reference") or {
            key: str(anchor.get(key) or "").strip()
            for key in ("title", "artist", "name", "album") if anchor.get(key)
        }
        resolved_identity = anchor.get("resolved") or {
            "title": str(track.get("title") or ""),
            "artist": str(track.get("artist") or track.get("author") or ""),
            "track_id": str(track.get("item_id") or ""),
        }
        anchor_records.append({
            "id": alias,
            "type": anchor.get("type"),
            "user_reference": user_reference,
            "resolved_library_track": resolved_identity,
            "available_on_server": bool(alias),
        })

    # Request-local references keep Phase B output compact. The shuffled,
    # balanced candidate universe and authoritative song objects are unchanged.
    numeric_to_song = {index: alias_to_song[record["id"]] for index, record in enumerate(records, 1)}
    item_id_to_numeric = {
        str(song["item_id"]): index for index, song in numeric_to_song.items()
    }
    numeric_records = [
        {**record, "id": index} for index, record in enumerate(records, 1)
    ]
    neighborhood_labels = []
    neighborhoods_by_track = {}
    for label, ids in (seed_provenance or {}).items():
        neighborhood_labels.append(str(label))
        neighborhood_ref = len(neighborhood_labels)
        for item_id in ids:
            neighborhoods_by_track.setdefault(str(item_id), []).append(neighborhood_ref)
    for index, record in enumerate(numeric_records, 1):
        memberships = neighborhoods_by_track.get(str(numeric_to_song[index]["item_id"]))
        if memberships:
            record["seed_neighborhoods"] = memberships
    duration_rows = duration_rows or {}
    candidate_durations = []
    for index, record in enumerate(numeric_records, 1):
        song = numeric_to_song[index]
        raw_duration = (duration_rows.get(song["item_id"]) or {}).get("duration")
        try:
            seconds = int(float(raw_duration))
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            record["duration_seconds"] = seconds
            candidate_durations.append(seconds)
    phase_a_anchors = [
        {**anchor, "id": f"A{index:03d}"}
        for index, anchor in enumerate(anchor_records, 1)
    ]
    anchor_ref = {
        anchor["id"]: item_id_to_numeric[str(anchor["resolved_library_track"]["track_id"])]
        for anchor in phase_a_anchors
        if str(anchor["resolved_library_track"]["track_id"]) in item_id_to_numeric
    }
    provider = str(ai_config.get("provider") or "unknown").upper()
    context_setting = int(config.INSTANT_PLAYLIST_COMPOSER_CONTEXT_SIZE)
    output_ceiling = _composer_output_budget()
    timeout = int(config.INSTANT_PLAYLIST_COMPOSER_TIMEOUT_SECONDS)
    default_count = int(ui_default_count or config.INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS)
    serialized_anchors = json.dumps(phase_a_anchors, ensure_ascii=False, separators=(",", ":"))
    original_request = json.dumps(str(user_request or ""), ensure_ascii=False)

    if log_messages is not None:
        selected_ids = {str(song.get("item_id")) for song in selected}
        represented_seeds = sum(
            any(str(item_id) in selected_ids for item_id in ids)
            for ids in (seed_provenance or {}).values()
        )
        log_messages.extend([
            f"Merged candidates: {len(pool)}",
            f"Configured composer capacity: {'unlimited' if not capacity else capacity}; "
            f"reduction applied: {'yes' if len(selected) < len(pool) else 'no'}",
            f"Seed neighborhoods retained: {represented_seeds}/{len(seed_provenance or {})}",
            f"Candidates sent to LLM2: {len(records)}",
            f"Composer context present: resolved anchors={len(phase_a_anchors)}",
            f"Configured Composer max output tokens: {output_ceiling}",
            f"Composer context size: {context_setting if provider == 'OLLAMA' else 'not applicable'}",
            f"Composer timeout: {timeout}s" if timeout else "Composer timeout: unlimited",
        ])
        for index, anchor in enumerate(phase_a_anchors, 1):
            identity = anchor["resolved_library_track"]
            log_messages.append(
                f"Composer anchor A{index:03d}: {identity.get('title', '')} / {identity.get('artist', '')}"
            )

    def call_phase(phase, prompt, system_prompt, selection_count=None):
        metadata = {}
        is_selection = (phase.startswith("Phase B") and not phase.startswith("Phase B balance")) or phase in {"Review", "Coverage repair", "Artist mix repair"}
        budget = (
            min(output_ceiling, max(2048 if phase in {"Review", "Coverage repair"} else 512,
                                    128 + 8 * (selection_count or len(records))))
            if is_selection else min(output_ceiling, 512)
        )
        options = (
            {"think": False, "allow_think_fallbacks": False, "num_ctx": context_setting,
             "call_metadata": metadata, "timeout": timeout, "selection_stream": is_selection}
            if provider == "OLLAMA" else {}
        )
        if provider == "OLLAMA":
            logger.info(
                "Compose %s provider configuration: format=json think=false num_ctx=%d num_predict=%d",
                phase, context_setting, budget,
            )
        logger.info("Compose %s instructions: %s", phase, prompt.split("Original user request (verbatim):", 1)[0])
        started = time.monotonic()
        raw = generate_text(
            prompt, ai_config, skip_delay=True, temperature=0.1,
            max_tokens=budget,
            structured_format="json" if provider == "OLLAMA" else None,
            system_prompt=system_prompt, **options,
        )
        content = raw if isinstance(raw, str) else ""
        # Content is the assistant's final channel; provider thinking is never logged.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Compose %s raw assistant content before parsing (%d chars):\n%s", phase, len(content), content)
        reason = str(metadata.get("done_reason") or "unknown")
        tokens = metadata.get("eval_count", "unknown")
        logger.info(
            "Compose %s response: duration=%.1fs content_chars=%d output_tokens=%s done_reason=%s",
            phase, time.monotonic() - started, len(content), tokens, reason,
        )
        if log_messages is not None:
            log_messages.append(f"Compose {phase}: duration={time.monotonic() - started:.1f}s; output tokens={tokens}; budget={budget}; done_reason={reason}")
        if reason == "COMPOSER_REPETITION":
            return content, "COMPOSER_REPETITION"
        if reason.casefold() in {"length", "max_tokens", "max_output_tokens"}:
            return content, "OUTPUT_LIMIT"
        if content.startswith("Error:"):
            lower = content.casefold()
            if "timeout" in lower or "timed out" in lower:
                return content, "TIMEOUT"
            if "empty assistant content" in lower or "returned no content" in lower:
                return content, "EMPTY_CONTENT"
            if "context" in lower and any(word in lower for word in ("limit", "length", "exceed")):
                return content, "CONTEXT_LIMIT"
            if any(term in lower for term in ("token", "num_predict")) and any(word in lower for word in ("limit", "length", "exceed")):
                return content, "OUTPUT_LIMIT"
            return content, "HTTP_ERROR"
        if not content.strip():
            return content, "EMPTY_CONTENT"
        return content, None

    def failed(category, reason, repair_attempted=False):
        return {
            "playlist": [], "requested_output": {}, "shortfall_reason": None,
            "error": {"category": category, "reason": reason, "repair_attempted": repair_attempted},
        }, len(records)

    phase_a_prompt = (
        "Interpret the ORIGINAL user request into the final executable playlist intent. "
        "Return only JSON: {\"anchor_decisions\":{\"A001\":true},\"target_count\":64,"
        "\"target_duration_seconds\":null,\"allow_nonstandard_versions\":false,"
        "\"discovery_priority\":\"balanced\"}. "
        "Do not return playlist IDs, candidate rankings, "
        "explanations, analysis, or markdown. Include one boolean decision for every resolved anchor ID. "
        "Set target_count to the final TOTAL track count, including every anchor marked true. "
        "For 'add N' requests, add N to the count of included anchors. "
        "If the request states a duration but no track count, target_count MUST be null. "
        "Use the UI default only when the request states neither a count nor a duration. "
        "target_duration_seconds is the requested duration in seconds. "
        "Set allow_nonstandard_versions true when the user wants live, remixed, acoustic, "
        "instrumental, edited or other alternate recordings; otherwise set it false. "
        "Set discovery_priority to focused for a narrow artist/song request, balanced for "
        "a general mix, or exploratory when the user wants a broad discovery mix. "
        "For several contrasting positive song seeds, prefer balanced or exploratory "
        "unless the request clearly asks to stay close to those artists. "
        "Stop after the JSON object.\n"
        f"Original user request (verbatim): {original_request}\n"
        f"UI default song count: {default_count}\n"
        f"Resolved anchors: {serialized_anchors}\n"
        f"Available candidate count: {len(records)}"
    )
    if preinterpreted_intent is None:
        raw_a, failure = call_phase("Phase A", phase_a_prompt, "Interpret playlist intent. Return only JSON.")
        if failure:
            return failed(failure, f"Compose interpretation provider failed: {failure}.")
        intent, _extracted, _thinking, parse_error = parse_json_response(raw_a)
    else:
        intent, parse_error = preinterpreted_intent, None
    anchor_ids = {anchor["id"] for anchor in phase_a_anchors}
    def valid_phase_a(parsed):
        required_keys = {"anchor_decisions", "target_count", "target_duration_seconds"}
        if not isinstance(parsed, dict) or not required_keys.issubset(parsed):
            return False
        allowed_keys = required_keys | {"allow_nonstandard_versions", "discovery_priority"}
        if not set(parsed).issubset(allowed_keys):
            return False
        decisions = parsed["anchor_decisions"]
        count = parsed["target_count"]
        duration = parsed["target_duration_seconds"]
        return (
            isinstance(decisions, dict) and set(decisions) == anchor_ids
            and all(type(value) is bool for value in decisions.values())
            and (count is None or type(count) is int and count > 0)
            and (duration is None or type(duration) in (int, float)
                 and math.isfinite(duration) and duration > 0)
            and (count is not None or duration is not None)
            and type(parsed.get("allow_nonstandard_versions", False)) is bool
            and parsed.get("discovery_priority", "balanced") in {"focused", "balanced", "exploratory"}
        )
    if parse_error or not valid_phase_a(intent):
        return failed("SCHEMA_MISMATCH", "Compose interpretation values failed validation.")
    missing_required = [
        anchor for anchor in phase_a_anchors
        if intent["anchor_decisions"][anchor["id"]] and anchor["id"] not in anchor_ref
    ]
    if missing_required:
        names = ", ".join(
            f"{anchor['resolved_library_track'].get('title', 'Unknown')} / "
            f"{anchor['resolved_library_track'].get('artist', 'Unknown')}"
            for anchor in missing_required
        )
        return failed(
            "COMPOSER_ANCHOR_UNAVAILABLE",
            f"Required library track(s) are unavailable on the selected server: {names}.",
        )
    target_count = intent["target_count"]
    target_duration = intent["target_duration_seconds"]
    if (target_count is not None and target_duration is not None
            and len(candidate_durations) >= target_count):
        shortest_possible = sum(sorted(candidate_durations)[:target_count])
        if shortest_possible > target_duration + config.INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS:
            repair_prompt = (
                f"{phase_a_prompt}\n"
                f"Your previous target_count={target_count} and target_duration_seconds={target_duration} "
                f"cannot both fit the library: even the {target_count} shortest eligible tracks "
                f"take {shortest_possible} seconds. Re-read the original request. If it does not "
                "explicitly request a track count, set target_count=null; otherwise retain the "
                "explicit count so the application can report that it is impossible. "
                "Return the same Phase A JSON fields only."
            )
            repaired_raw, repair_failure = call_phase(
                "Phase A feasibility repair", repair_prompt,
                "Reconsider only the count interpretation. Return only JSON.",
            )
            if repair_failure:
                return failed(repair_failure, f"Compose interpretation repair failed: {repair_failure}.")
            repaired, _extracted, _thinking, repair_parse_error = parse_json_response(repaired_raw)
            if (repair_parse_error or not valid_phase_a(repaired)
                    or repaired["anchor_decisions"] != intent["anchor_decisions"]
                    or repaired["target_duration_seconds"] != target_duration):
                return failed("SCHEMA_MISMATCH", "Compose interpretation repair was invalid.")
            intent = repaired
            target_count = intent["target_count"]
            target_duration = intent["target_duration_seconds"]
            if (target_count is not None and len(candidate_durations) >= target_count
                    and sum(sorted(candidate_durations)[:target_count])
                    > target_duration + config.INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS):
                return failed(
                    "COMPOSER_SELECTION_SIZE_MISMATCH",
                    "The requested count and duration cannot both fit the library.",
                )
    if expansion_callback is not None and preinterpreted_intent is None:
        try:
            expanded_pool = expansion_callback(intent, pool)
        except Exception:
            logger.exception("Adaptive candidate retrieval failed; continuing with the initial pool")
            expanded_pool = None
        if expanded_pool and len(expanded_pool) > len(pool):
            if log_messages is not None:
                log_messages.append(
                    f"Adaptive candidate universe: {len(pool)} -> {len(expanded_pool)} tracks"
                )
            return compose_playlist_with_llm(
                user_request, expanded_pool, ai_config,
                seed_provenance=seed_provenance, resolved_anchors=resolved_anchors,
                ui_default_count=ui_default_count, log_messages=log_messages,
                duration_rows=duration_rows, musical_review=musical_review,
                preinterpreted_intent=intent,
            )
    decisions = intent["anchor_decisions"]
    required_refs = {anchor_ref[key] for key, include in decisions.items() if include}
    excluded_refs = {anchor_ref[key] for key, include in decisions.items()
                     if not include and key in anchor_ref}
    positive_neighborhoods = set()
    for anchor in phase_a_anchors:
        if not decisions.get(anchor["id"]):
            continue
        identity = anchor["resolved_library_track"]
        artist = _normalized_content_text(identity.get("artist"))
        title = _normalized_content_text(identity.get("title"))
        for index, label in enumerate(neighborhood_labels, 1):
            label_artist, _, label_title = label.partition(" / ")
            if (_normalized_content_text(label_artist) == artist
                    and _normalized_content_text(label_title) == title):
                positive_neighborhoods.add(index)
    if not intent.get("allow_nonstandard_versions", False):
        numeric_records = [
            row for row in numeric_records
            if row["id"] in required_refs or not is_alternate_recording(numeric_to_song[row["id"]])
        ]
        candidate_durations = [
            row["duration_seconds"] for row in numeric_records if "duration_seconds" in row
        ]
    eligible_refs = {row["id"] for row in numeric_records}
    available_count = len(eligible_refs - excluded_refs)
    if target_duration is not None:
        if candidate_durations:
            median_duration = sorted(candidate_durations)[len(candidate_durations) // 2]
            duration_options = math.ceil(target_duration / median_duration) * 3
        else:
            duration_options = max(12, math.ceil(available_count / 2))
        selection_count = min(available_count, max(
            len(required_refs), (target_count or 0) * 2, duration_options,
        ))
    else:
        selection_count = target_count
    if target_count is not None and available_count < target_count:
        return failed(
            "COMPOSER_SELECTION_SIZE_MISMATCH",
            f"Only {available_count} eligible candidates are available for {target_count} tracks.",
        )
    neighborhood_minima = {}
    artist_policy = None
    if target_count is not None and len(positive_neighborhoods) > 1:
        available_artist_counts = Counter(
            row.get("artist") or "Unknown" for row in numeric_records
            if row["id"] not in excluded_refs
        )
        required_artist_counts = Counter(
            next((row.get("artist") or "Unknown" for row in numeric_records
                  if row["id"] == ref), "Unknown")
            for ref in required_refs
        )
        summary = []
        for index in sorted(positive_neighborhoods):
            members = [
                row for row in numeric_records
                if index in row.get("seed_neighborhoods", [])
                and row["id"] not in excluded_refs
            ]
            summary.append({
                "id": str(index), "seed": neighborhood_labels[index - 1],
                "available_tracks": len(members),
                "available_artists": dict(Counter(
                    row.get("artist") or "Unknown" for row in members
                ).most_common(16)),
            })
        balance_prompt = (
            "Plan musical representation across the user's positive song seeds using the "
            "REAL candidate availability. This is a musical decision for LLM Compose, not "
            "a fixed quota. Decide the minimum number of candidate tracks the final "
            "playlist should draw from EACH seed neighborhood to represent all references "
            "meaningfully. Counts may differ substantially when the music or availability "
            "warrants it. A track in two neighborhoods can count toward both. Keep the "
            "minima feasible within the final playlist size, allowing room for other "
            "related tracks. Also decide an artist mix from the actual availability: "
            "maximum tracks per artist and minimum distinct artists. A focused request "
            "may allow concentration; a balanced or exploratory mix should prefer "
            "meaningful discovery without choosing unrelated music. These are musical "
            "targets, not a fixed source-code quota. Return only JSON with "
            "minimum_neighborhood_tracks and artist_mix objects; artist_mix must contain "
            "positive integer max_tracks_per_artist and minimum_distinct_artists.\n"
            f"Original request: {original_request}\n"
            f"Discovery priority interpreted by LLM2A: {intent.get('discovery_priority', 'balanced')}\n"
            f"Final track count: {target_count}\n"
            f"Included library anchors: {len(required_refs)}\n"
            f"Additional track slots after included anchors: {target_count - len(required_refs)}\n"
            f"Available distinct artists: {len(available_artist_counts)}\n"
            f"Available artist counts: {json.dumps(available_artist_counts.most_common(30), ensure_ascii=False)}\n"
            f"Seed candidate availability: {json.dumps(summary, ensure_ascii=False)}"
        )
        availability = {
            int(row["id"]): row["available_tracks"] for row in summary
        }
        for attempt in range(2):
            balance_raw, balance_failure = call_phase(
                "Phase B balance" if attempt == 0 else "Phase B balance repair",
                balance_prompt, "Decide musical seed representation. Return only JSON.",
            )
            if balance_failure:
                return failed(balance_failure, "Composer musical balance plan was unavailable.")
            balance, _, _, balance_parse_error = parse_json_response(balance_raw)
            requested_minima = balance.get("minimum_neighborhood_tracks") if isinstance(balance, dict) else None
            raw_policy = balance.get("artist_mix") if isinstance(balance, dict) else None
            if (balance_parse_error or not isinstance(requested_minima, dict)
                    or not set(balance).issubset({"minimum_neighborhood_tracks", "artist_mix"})
                    or set(requested_minima) != {str(index) for index in positive_neighborhoods}
                    or any(type(value) is not int or value < 1 for value in requested_minima.values())):
                return failed("SCHEMA_MISMATCH", "Composer musical balance plan was invalid.")
            neighborhood_minima = {int(key): value for key, value in requested_minima.items()}
            artist_policy = None
            if (isinstance(raw_policy, dict)
                    and set(raw_policy) == {"max_tracks_per_artist", "minimum_distinct_artists"}
                    and all(type(value) is int and value > 0 for value in raw_policy.values())):
                artist_policy = raw_policy
            excess = sum(neighborhood_minima.values()) - (target_count - len(required_refs))
            over_available = {
                index: value for index, value in neighborhood_minima.items()
                if value > availability[index]
            }
            cap = artist_policy.get("max_tracks_per_artist") if artist_policy else None
            minimum_artists = artist_policy.get("minimum_distinct_artists") if artist_policy else None
            artist_infeasible = bool(artist_policy and (
                cap > target_count or minimum_artists > min(target_count, len(available_artist_counts))
                or any(count > cap for count in required_artist_counts.values())
                or sum(min(cap, count) for count in available_artist_counts.values()) < target_count
            ))
            if excess <= 0 and not over_available and not artist_infeasible:
                break
            if attempt:
                if excess > 0 or over_available:
                    return failed("COMPOSER_SELECTION_SIZE_MISMATCH", "Composer musical balance plan exceeds available candidates.")
                artist_policy = None  # Keep the valid playlist plan if its soft artist goal is infeasible.
                break
            balance_prompt += (
                "\n\nREVISE YOUR MUSICAL BALANCE PLAN: Your previous minima were "
                f"{json.dumps(neighborhood_minima, sort_keys=True)}. Their sum exceeds "
                f"the {target_count - len(required_refs)} available additional slots by "
                f"{max(0, excess)}. Counts above neighborhood availability: "
                f"{json.dumps(over_available, sort_keys=True)}. Decide revised unequal "
                "minima that fit the real slot and candidate counts. Revise artist_mix too "
                "if its cap or distinct-artist target exceeds the available music. "
                "Return the same JSON fields."
            )
        if log_messages is not None:
            log_messages.append(f"Compose Phase B musical seed minima: {json.dumps(neighborhood_minima, sort_keys=True)}")
            log_messages.append(f"Compose Phase B artist mix: {json.dumps(artist_policy, sort_keys=True)}")
    if log_messages is not None:
        log_messages.append(
            f"Compose Phase A: target_count={target_count}; target_duration={target_duration}; "
            f"anchor decisions={json.dumps(decisions, sort_keys=True)}; "
            f"allow_nonstandard_versions={intent.get('allow_nonstandard_versions', False)}"
        )
    logger.info(
        "Compose Phase A: target_count=%s target_duration=%s anchors_included=%d",
        target_count, target_duration, len(required_refs),
    )

    phase_b_anchors = [
        {**anchor, "candidate_ref": anchor_ref[anchor["id"]], "include": decisions[anchor["id"]]}
        for anchor in phase_a_anchors if anchor["id"] in anchor_ref
    ]
    duration_instruction = (
        f"Rank exactly {selection_count} musically suitable, UNIQUE candidate references in preference order. "
        "These are Composer-selected options for the duration optimizer, which will choose the "
        "final subset using real library durations. Include enough varied options for an accurate duration. "
        if target_duration is not None else
        "playlist_ids is the final selection, not a ranking of all candidates. "
        "Return exactly the required count of UNIQUE references. Never rank the complete candidate pool. "
    )
    selection_prompt = (
        "Compose an ordered playlist selection from these candidates. Return only JSON with playlist_ids, "
        "an array of request-local INTEGER references. Do not restate target_count, anchor decisions, "
        "constraints, reasoning, scores, rejected IDs, or prose. "
        f"{duration_instruction}"
        "Phase A intent is authoritative: do not reinterpret count or duration. "
        "Order the most suitable tracks first. Normally choose only one version of each song; "
        "include remixes, live versions or edits when the request calls for them. "
        "Use the seed neighborhoods to consider all of the user's musical references. "
        "Make a musical compromise across different styles rather than equal quotas or "
        "an avoidable concentration in one or two artists. "
        "Include every anchor marked include=true exactly once; exclude anchors marked include=false "
        "when the original request uses them only as inspiration or explicitly excludes them. "
        "Return only {\"playlist_ids\":[1,2,3]}. "
        "Stop immediately after the JSON object.\n"
        f"Original user request (verbatim): {original_request}\n"
        f"Normalized intent: {json.dumps(intent, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Required final count: {target_count if target_count is not None else 'duration-driven'}\n"
        f"Composer option count: {selection_count}\n"
        f"Seed neighborhoods: {json.dumps(neighborhood_labels, ensure_ascii=False, separators=(',', ':'))}\n"
        f"LLM2 musical seed minima: {json.dumps(neighborhood_minima, sort_keys=True)}\n"
        f"Resolved anchors: {json.dumps(phase_b_anchors, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Available candidates: {json.dumps(numeric_records, ensure_ascii=False, separators=(',', ':'))}"
    )
    if log_messages is not None:
        log_messages.append(f"Compose Phase B: candidates supplied={len(numeric_records)}; requested selections={selection_count}")

    if target_count is not None and target_count > int(config.INSTANT_PLAYLIST_MAX_N_RESULTS):
        return failed(
            "COMPOSER_SELECTION_SIZE_MISMATCH",
            f"Phase A target_count={target_count} exceeds the configured final playlist capacity.",
        )
    if target_count is not None and len(required_refs) > target_count:
        return failed(
            "SCHEMA_MISMATCH",
            "Phase A included more required anchors than its target_count.",
        )

    # The selection call supplies a near-final ordered preference list. Only a
    # malformed response is repaired with another full selection; count variance
    # is handled below without regenerating the entire playlist.
    repair_attempted = False
    if neighborhood_minima:
        by_ref = {row["id"]: row for row in numeric_records}
        artist_cap = artist_policy.get("max_tracks_per_artist") if artist_policy else None
        refs = [
            anchor_ref[anchor["id"]] for anchor in phase_a_anchors
            if decisions[anchor["id"]]
        ]
        chosen = set(refs)

        def select_exact_group(phase, label, rows, requested):
            selected_ids = []
            for attempt in range(4 if artist_cap is not None else 2):
                missing = requested - len(selected_ids)
                if missing <= 0:
                    break
                active_cap = artist_cap if attempt < 3 else None
                if attempt == 3 and log_messages is not None:
                    log_messages.append(f"{phase}: relaxing soft artist target to preserve playlist completion")
                artist_counts = Counter(
                    by_ref[ref].get("artist") or "Unknown" for ref in refs + selected_ids
                )
                remaining = [
                    row for row in rows
                    if row["id"] not in chosen and row["id"] not in selected_ids
                    and (active_cap is None
                         or artist_counts[row.get("artist") or "Unknown"] < active_cap)
                ]
                if len(remaining) < missing and active_cap is not None:
                    if log_messages is not None:
                        log_messages.append(f"{phase}: artist target leaves only {len(remaining)} choices for {missing} slots")
                    continue
                retry_guidance = (
                    f"Your earlier reply did not fill all slots. Return exactly {missing} "
                    "eligible numeric IDs in one JSON object with playlist_ids and no prose.\n"
                    if attempt else ""
                )
                prompt = (
                    "Choose musically suitable real-library tracks for this portion of the "
                    "playlist. This neighborhood's representation was decided by LLM2's "
                    "musical balance plan. Return exactly the missing number of UNIQUE "
                    "numeric candidate references. Return one JSON object containing only "
                    "playlist_ids, with the chosen references as its array; no prose or "
                    "example values.\n"
                    f"Original request: {original_request}\n"
                    f"Normalized intent: {json.dumps(intent, ensure_ascii=False)}\n"
                    f"LLM2 artist mix: {json.dumps(artist_policy, ensure_ascii=False)}\n"
                    f"Seed neighborhood: {label}\n"
                    f"Missing track choices: {missing}\n"
                    f"Already chosen tracks: {json.dumps([by_ref[ref] for ref in refs + selected_ids], ensure_ascii=False)}\n"
                    f"{retry_guidance}"
                    f"Eligible candidates: {json.dumps(remaining, ensure_ascii=False)}"
                )
                raw, failure = call_phase(
                    phase, prompt, "Choose only supplied library candidates. Return only JSON.", missing,
                )
                if failure:
                    if log_messages is not None:
                        log_messages.append(f"{phase}: provider guard {failure}")
                    return None
                parsed, _, _, parse_error = parse_json_response(raw)
                allowed = {row["id"] for row in remaining}
                proposed = None
                if isinstance(parsed, dict) and set(parsed).issubset(
                    {"playlist_ids", "ids", "selected_ids", "track_ids"}
                ):
                    options = []
                    for priority, key in enumerate(("playlist_ids", "ids", "selected_ids", "track_ids")):
                        values = parsed.get(key)
                        if not isinstance(values, list):
                            continue
                        normalized = [int(value) if type(value) is str and value.isdecimal() else value
                                      for value in values]
                        valid_count = len({value for value in normalized
                                           if type(value) is int and value in allowed})
                        options.append((valid_count, -priority, values))
                    if options:
                        proposed = max(options)[2]
                if parse_error or not isinstance(proposed, list):
                    if log_messages is not None:
                        log_messages.append(f"{phase}: invalid JSON shape; retrying remaining choices")
                    continue
                for ref in proposed:
                    if type(ref) is str and ref.isdecimal():
                        ref = int(ref)
                    artist = by_ref[ref].get("artist") or "Unknown" if type(ref) is int and ref in by_ref else None
                    if (type(ref) is int and ref in allowed and ref not in selected_ids
                            and (active_cap is None or artist_counts[artist] < active_cap)):
                        selected_ids.append(ref)
                        artist_counts[artist] += 1
                        if len(selected_ids) == requested:
                            break
                if log_messages is not None:
                    log_messages.append(
                        f"{phase}: model returned {len(proposed)} references; "
                        f"validated {len(selected_ids)}/{requested}; "
                        f"sample={json.dumps(proposed[:5])}"
                    )
            return selected_ids if len(selected_ids) == requested else None

        for index, quota in sorted(neighborhood_minima.items()):
            neighborhood_rows = [
                row for row in numeric_records
                if index in row.get("seed_neighborhoods", [])
                and row["id"] not in excluded_refs
            ]
            group_ids = select_exact_group(
                f"Phase B neighborhood {index}", neighborhood_labels[index - 1],
                neighborhood_rows, quota,
            )
            if group_ids is None:
                return failed(
                    "COMPOSER_SELECTION_SIZE_MISMATCH",
                    f"LLM2 could not select {quota} valid tracks for {neighborhood_labels[index - 1]}.",
                )
            refs.extend(group_ids)
            chosen.update(group_ids)
            if log_messages is not None:
                log_messages.append(
                    f"Compose Phase B neighborhood {index}: {len(group_ids)} model-selected tracks"
                )
        remaining_slots = target_count - len(refs)
        if remaining_slots < 0:
            return failed("COMPOSER_SELECTION_SIZE_MISMATCH", "Musical balance plan exceeded the final track count.")
        if remaining_slots:
            global_rows = [
                row for row in numeric_records
                if row["id"] not in chosen and row["id"] not in excluded_refs
            ]
            extra_ids = select_exact_group(
                "Phase B global fill", "All related music", global_rows, remaining_slots,
            )
            if extra_ids is None:
                return failed("COMPOSER_SELECTION_SIZE_MISMATCH", "LLM2 could not fill the remaining playlist slots.")
            refs.extend(extra_ids)
            chosen.update(extra_ids)
        order_presentation = []
        for song in _rerank_candidate_order(
            [numeric_to_song[ref] for ref in refs], context + "-final-order"
        ):
            row = by_ref[item_id_to_numeric[str(song["item_id"])]]
            order_presentation.append({
                key: row[key] for key in ("id", "title", "artist", "seed_neighborhoods")
                if key in row
            })
        order_prompt = (
            "Order exactly these LLM2-selected library tracks as a coherent playlist. "
            "Return every supplied numeric ID exactly once, with no additions, omissions, "
            "or prose. Return one JSON object with only playlist_ids, containing the ordered "
            "numeric references. Respect any explicit ordering in the "
            "original request. Otherwise weave the different seed directions into a "
            "musically plausible journey rather than placing each seed in one long block. "
            "The selected tracks below are shuffled; their presentation order is not a "
            "suggested playlist order.\n"
            f"Original request: {original_request}\n"
            f"Selected tracks: {json.dumps(order_presentation, ensure_ascii=False)}"
        )
        ordered_ids = None
        for attempt in range(2):
            prompt = order_prompt + (
                "\nYour prior ordering reply was unusable. Return the complete numeric ID list "
                "once, in a single playlist_ids array, and stop." if attempt else ""
            )
            order_raw, order_failure = call_phase(
                "Phase B order" if not attempt else "Phase B order retry",
                prompt, "Order selected music. Return only JSON.", len(refs),
            )
            if order_failure:
                continue
            order_result, _, _, order_parse_error = parse_json_response(order_raw)
            order_key = next(iter(order_result)) if isinstance(order_result, dict) and len(order_result) == 1 else None
            candidate_order = order_result.get(order_key) if order_key in {"playlist_ids", "ids", "selected_ids"} else None
            if not order_parse_error and isinstance(candidate_order, list):
                ordered_ids = candidate_order
                break
        if ordered_ids is None:
            ordered_ids = list(refs)
            if log_messages is not None:
                log_messages.append("Compose Phase B order: model ordering unavailable; preserving LLM2 selections")
        ordered = []
        ordered_set = set()
        for ref in ordered_ids:
            if type(ref) is int and ref in chosen and ref not in ordered_set:
                ordered.append(ref)
                ordered_set.add(ref)
        ordered.extend(ref for ref in refs if ref not in ordered_set)
        refs = ordered
        shortfall = None
        if log_messages is not None:
            log_messages.append(f"Compose Phase B order: {len(ordered_set)}/{len(refs)} model-ordered references")
    else:
        for attempt in range(2):
            raw_b, failure = call_phase(
                "Phase B" if attempt == 0 else "Phase B repair",
                selection_prompt, "Select only ordered playlist candidates. Return only JSON.", selection_count,
            )
            if failure and failure not in {"COMPOSER_REPETITION", "OUTPUT_LIMIT"}:
                return failed(failure, f"Compose selection provider failed: {failure}.", repair_attempted)
            if failure and log_messages is not None:
                log_messages.append(f"Compose Phase B guard: {failure}; preserving valid unique prefix")
            selection, _extracted, _thinking, parse_error = parse_json_response(raw_b)
            refs = selection.get("playlist_ids") if isinstance(selection, dict) else None
            shortfall = selection.get("shortfall_reason") if isinstance(selection, dict) else None
            valid_shape = (
                not parse_error and isinstance(selection, dict)
                and set(selection) in ({"playlist_ids"}, {"playlist_ids", "shortfall_reason"})
                and isinstance(refs, list)
                and (shortfall is None or isinstance(shortfall, str) and bool(shortfall.strip()))
            )
            if valid_shape:
                break
            if failure:
                refs, shortfall = [], None
                break
            if attempt == 1:
                return failed("SCHEMA_MISMATCH", "Compose selection JSON shape is invalid.", repair_attempted)
            repair_attempted = True
            selection_prompt += (
                "\n\nREPAIR SELECTION JSON ONLY: Return {\"playlist_ids\":[1,2,3]} "
                "with numeric request-local references. Do not reinterpret the request."
            )

    raw_count = len(refs)
    valid_refs, seen = [], set()
    malformed_removed = unknown_removed = duplicate_removed = excluded_removed = 0
    for ref in refs:
        if type(ref) is not int:
            malformed_removed += 1
        elif ref not in eligible_refs:
            unknown_removed += 1
        elif ref in seen:
            duplicate_removed += 1
        elif ref in excluded_refs:
            excluded_removed += 1
        else:
            valid_refs.append(ref)
            seen.add(ref)
    anchors_present = len(required_refs & seen)
    summary = (
        f"Composer raw IDs: {raw_count}; Valid IDs: {len(valid_refs)}; "
        f"Duplicate IDs removed: {duplicate_removed}; Unknown IDs removed: {unknown_removed}; "
        f"Malformed IDs removed: {malformed_removed}; Excluded anchors removed: {excluded_removed}; "
        f"Required anchors present: {anchors_present}/{len(required_refs)}"
    )
    logger.info(summary)
    if log_messages is not None:
        log_messages.append(summary)
        log_messages.append(f"Compose Phase B: IDs returned={raw_count}; unique IDs={len(valid_refs)}")

    # Phase A's inclusion decisions are execution invariants. Add a missing
    # authoritative anchor at the tail, then protect it during any tail trim.
    missing_required = list(dict.fromkeys(
        ref for ref in anchor_ref.values() if ref in required_refs and ref not in seen
    ))
    if missing_required:
        valid_refs.extend(missing_required)
        seen.update(missing_required)
        logger.info("Composer required anchors inserted: %s", missing_required)
        if log_messages is not None:
            log_messages.append(f"Composer required anchors inserted: {missing_required}")

    if log_messages is not None:
        log_messages.append(f"Validated ordered selection: {len(valid_refs)}")
        log_messages.append(f"Target count: {target_count}")
    if selection_count is not None and len(valid_refs) > selection_count:
        if raw_count == len(numeric_records):
            logger.warning("Composer substantially over-selected candidates: returned entire candidate universe")
            if log_messages is not None:
                log_messages.append("Composer substantially over-selected candidates: entire candidate universe returned")
        trimmed = len(valid_refs) - selection_count
        while len(valid_refs) > selection_count:
            tail_nonrequired = next(
                (index for index in range(len(valid_refs) - 1, -1, -1)
                 if valid_refs[index] not in required_refs), None,
            )
            if tail_nonrequired is None:
                return failed("SCHEMA_MISMATCH", "No non-required tracks can be trimmed.", repair_attempted)
            del valid_refs[tail_nonrequired]
        if log_messages is not None:
            log_messages.append(f"Tail trimmed: {trimmed}")
        logger.info("Composer tail trimmed: %d", trimmed)

    if selection_count is not None and len(valid_refs) < selection_count:
        if repair_attempted:
            return failed("COMPOSER_SELECTION_SIZE_MISMATCH", "Selection is still short after one repair.", True)
        repair_attempted = True
        missing_count = selection_count - len(valid_refs)
        remaining = [
            row for row in numeric_records
            if row["id"] not in seen and row["id"] not in excluded_refs
        ]
        if log_messages is not None:
            log_messages.extend([
                f"Shortfall: {missing_count}",
                f"Targeted fill requested: {missing_count}",
                f"Compose Phase B targeted fill: candidates supplied={len(remaining)}; requested selections={missing_count}",
            ])
        fill_prompt = (
            "Select only the missing tracks that best complete this EXISTING ordered playlist. "
            f"Return JSON with playlist_ids containing at most {missing_count} UNIQUE numeric references. "
            "Do not return already-selected references, reinterpret the request, restate the target, "
            "or regenerate the playlist. Preserve artist variety and avoid repeated song versions. "
            "Stop after the JSON object.\n"
            f"Original user request (verbatim): {original_request}\n"
            f"Normalized intent: {json.dumps(intent, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Missing count: {missing_count}\n"
            f"Already selected references in order: {json.dumps(valid_refs)}\n"
            f"Remaining available candidates: {json.dumps(remaining, ensure_ascii=False, separators=(',', ':'))}"
        )
        fill_raw, fill_failure = call_phase(
            "Phase B targeted fill", fill_prompt, "Fill only missing playlist positions. Return only JSON.", missing_count,
        )
        fill_refs = []
        if not fill_failure:
            fill_result, _extracted, _thinking, fill_parse_error = parse_json_response(fill_raw)
            if (
                not fill_parse_error and isinstance(fill_result, dict)
                and set(fill_result) == {"playlist_ids"}
                and isinstance(fill_result["playlist_ids"], list)
            ):
                fill_refs = fill_result["playlist_ids"]
            else:
                fill_failure = "SCHEMA_MISMATCH"
        fill_added = 0
        for ref in fill_refs:
            if (
                fill_added >= missing_count or type(ref) is not int
                or ref not in eligible_refs or ref in seen or ref in excluded_refs
            ):
                continue
            valid_refs.append(ref)
            seen.add(ref)
            fill_added += 1
        if log_messages is not None:
            log_messages.append(f"Targeted fill returned: {fill_added}")
            log_messages.append(f"Compose Phase B targeted fill: IDs returned={len(fill_refs)}; unique usable additions={fill_added}")
        logger.info("Composer targeted fill: requested=%d added=%d failure=%s", missing_count, fill_added, fill_failure)
        if fill_failure and not shortfall:
            return failed(fill_failure, f"Compose targeted fill failed: {fill_failure}.", repair_attempted)

    if selection_count is not None and len(valid_refs) < selection_count:
        if (target_count is not None and len(valid_refs) < target_count) or not shortfall:
            return failed(
                "COMPOSER_SELECTION_SIZE_MISMATCH",
                f"Composer option count={selection_count}; final valid IDs={len(valid_refs)} after targeted fill.",
                repair_attempted,
            )
    else:
        shortfall = None
    if len(valid_refs) != len(set(valid_refs)) or not required_refs.issubset(valid_refs):
        return failed("SCHEMA_MISMATCH", "Final Composer validation failed.", repair_attempted)

    preferred_final_count = None
    if musical_review and not neighborhood_minima and len(valid_refs) > 1 and len(valid_refs) < available_count:
        by_ref = {row["id"]: row for row in numeric_records}
        current_rows = [by_ref[ref] for ref in valid_refs]
        selected_artists = Counter(row.get("artist") or "Unknown" for row in current_rows)
        available_artists = Counter(
            row.get("artist") or "Unknown" for row in numeric_records
        )
        selected_neighborhoods = Counter(
            neighborhood for row in current_rows
            for neighborhood in row.get("seed_neighborhoods", [])
        )
        available_neighborhoods = Counter(
            neighborhood for row in numeric_records
            for neighborhood in row.get("seed_neighborhoods", [])
        )
        uncovered_neighborhoods = [
            index for index in range(1, len(neighborhood_labels) + 1)
            if available_neighborhoods[index] and not selected_neighborhoods[index]
        ]
        uncovered_options = {
            neighborhood_labels[index - 1]: [
                {key: row[key] for key in ("id", "title", "artist") if key in row}
                for row in numeric_records
                if index in row.get("seed_neighborhoods", [])
                and row["id"] not in excluded_refs
            ]
            for index in uncovered_neighborhoods
        }
        remaining_rows = [
            row for row in numeric_records
            if row["id"] not in valid_refs and row["id"] not in excluded_refs
        ]
        duration_facts = ""
        if target_duration is not None and target_count is None:
            required_seconds = sum(
                by_ref[ref].get("duration_seconds", 0) for ref in required_refs
            )
            optional_lengths = sorted(
                by_ref[ref].get("duration_seconds", 0)
                for ref in valid_refs if ref not in required_refs
            )
            optional_lengths = [seconds for seconds in optional_lengths if seconds > 0]
            feasible_count_ceiling = len(required_refs)
            running_seconds = required_seconds
            for seconds in optional_lengths:
                if running_seconds + seconds > target_duration + config.INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS:
                    break
                running_seconds += seconds
                feasible_count_ceiling += 1
            preview, preview_seconds, _ = finalize_composer_duration(
                [numeric_to_song[ref] for ref in valid_refs], duration_rows,
                target_duration,
                required_ids=[numeric_to_song[ref]["item_id"] for ref in required_refs],
                tolerance_seconds=config.INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS,
            )
            preview_refs = [item_id_to_numeric[str(song["item_id"])] for song in preview]
            duration_facts = (
                f"Duration target: {target_duration} seconds. Current duration-validated preview: "
                f"{json.dumps([by_ref[ref] for ref in preview_refs], ensure_ascii=False)} "
                f"totaling {preview_seconds} seconds. "
                f"With the lengths of the currently selected options, no more than "
                f"{feasible_count_ceiling} tracks can fit the target even if the shortest options "
                f"are used. Choose a preferred_final_count within that feasible range; "
                f"propose swaps at positions in the preview when its musical mix should change.\n"
            )
        review_output = (
            '{"swaps":[{"remove_id":1,"add_id":2}],"ordered_ids":[2,3,4],"preferred_final_count":6}'
            if target_duration is not None and target_count is None else
            '{"swaps":[{"remove_id":1,"add_id":2}],"ordered_ids":[2,3,4]}'
        )
        duration_review_instruction = (
            "For this duration-only request, decide how many tracks the final musical playlist "
            "should contain based on the candidate lengths and the user's intent. Set "
            "preferred_final_count to that positive number. The application will verify the "
            "actual total duration. "
            if target_duration is not None and target_count is None else ""
        )
        review_prompt = (
            "Review your ordered music selection against the user's ORIGINAL request. "
            "Decide whether its artist mix, seed coverage, versions, mood and musical journey "
            "fit the request and the real candidate availability. You own this musical judgment; "
            "do not apply equal seed quotas or a fixed artist cap. When a better fit exists, "
            "propose one-for-one swaps from remaining candidates. Preserve required anchors, "
            "playlist size, and the order of unchanged tracks. Otherwise return no swaps. "
            "When the request names several contrasting songs, examine how the current selection "
            "represents each musical direction and compare its artist mix with what is available. "
            "If viable alternatives are present, use as many swaps as the musical result needs; "
            "a small cosmetic adjustment is insufficient when a broad request is dominated by "
            "one or two artists. "
            "A seed neighborhood with available candidates but no selected tracks is a specific "
            "coverage gap: choose musically suitable replacements from it when the user treats "
            "that seed as a positive reference. Do not assign fixed equal quotas. "
            "Return ordered_ids containing EVERY selected reference exactly once AFTER swaps, "
            "ordered as a coherent playlist; avoid long blocks from one artist unless the "
            "user asks for them. "
            "Prefer standard studio versions over live, demo, remixed or edited tracks unless "
            "the request calls for those versions. "
            f"{duration_review_instruction}"
            f"Return only JSON: {review_output}.\n"
            f"Original user request (verbatim): {original_request}\n"
            f"Normalized intent: {json.dumps(intent, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Required anchor references: {json.dumps(sorted(required_refs))}\n"
            f"Seed neighborhood labels: {json.dumps(neighborhood_labels, ensure_ascii=False)}\n"
            f"Selected artists: {json.dumps(selected_artists, ensure_ascii=False)}\n"
            f"Available artists: {json.dumps(available_artists, ensure_ascii=False)}\n"
            f"Selected seed neighborhood counts: {json.dumps(selected_neighborhoods)}\n"
            f"Available seed neighborhood counts: {json.dumps(available_neighborhoods)}\n"
            f"Uncovered seed neighborhoods and eligible options: {json.dumps(uncovered_options, ensure_ascii=False)}\n"
            f"{duration_facts}"
            f"Current ordered selection: {json.dumps(current_rows, ensure_ascii=False, separators=(',', ':'))}\n"
            f"Remaining candidates: {json.dumps(remaining_rows, ensure_ascii=False, separators=(',', ':'))}"
        )
        review_raw, review_failure = call_phase(
            "Review", review_prompt, "Review musical fit. Return only JSON.", len(valid_refs),
        )
        if review_failure:
            if log_messages is not None:
                log_messages.append(f"Compose musical review unavailable: {review_failure}")
        else:
            review, _extracted, _thinking, review_parse_error = parse_json_response(review_raw)
            swaps = review.get("swaps") if isinstance(review, dict) else None
            expected_review_keys = (
                {"swaps", "preferred_final_count"}
                if target_duration is not None and target_count is None else {"swaps"}
            )
            ordered_ids = review.get("ordered_ids") if isinstance(review, dict) else None
            if (review_parse_error or not isinstance(swaps, list)
                    or set(review) not in (expected_review_keys,
                                           expected_review_keys | {"ordered_ids"})
                    or (ordered_ids is not None and not isinstance(ordered_ids, list))):
                if log_messages is not None:
                    log_messages.append("Compose musical review returned invalid JSON; original selection retained")
            else:
                proposed_count = review.get("preferred_final_count")
                if (type(proposed_count) is int and 1 <= proposed_count <= len(valid_refs)):
                    preferred_final_count = proposed_count
                    if log_messages is not None:
                        log_messages.append(f"Compose preferred final duration count: {proposed_count}")
                applied = 0
                replacements = {}
                selected_set = set(valid_refs)
                for swap in swaps:
                    if not isinstance(swap, dict) or set(swap) != {"remove_id", "add_id"}:
                        continue
                    old_ref, new_ref = swap["remove_id"], swap["add_id"]
                    if (type(old_ref) is not int or type(new_ref) is not int
                            or old_ref not in selected_set or old_ref in required_refs
                            or new_ref not in by_ref or new_ref in selected_set
                            or new_ref in excluded_refs):
                        continue
                    position = valid_refs.index(old_ref)
                    valid_refs[position] = new_ref
                    selected_set.remove(old_ref)
                    selected_set.add(new_ref)
                    replacements[old_ref] = new_ref
                    applied += 1
                if ordered_ids is not None:
                    reordered = []
                    reordered_set = set()
                    for proposed_ref in ordered_ids:
                        if type(proposed_ref) is not int:
                            continue
                        ref = replacements.get(proposed_ref, proposed_ref)
                        if ref in selected_set and ref not in reordered_set:
                            reordered.append(ref)
                            reordered_set.add(ref)
                    for ref in valid_refs:
                        if ref not in reordered_set:
                            reordered.append(ref)
                    valid_refs = reordered
                    if log_messages is not None:
                        log_messages.append(f"Compose musical review: {len(reordered_set)} model-ordered references applied")
                if log_messages is not None:
                    log_messages.append(f"Compose musical review: {applied} model-selected swaps applied")

        selected_set = set(valid_refs)
        covered_counts = Counter(
            neighborhood for ref in valid_refs
            for neighborhood in by_ref[ref].get("seed_neighborhoods", [])
        )
        required_coverage = (
            neighborhood_minima or {
                index: 1 for index in positive_neighborhoods
                if available_neighborhoods[index]
            }
        )
        missing_positive = sorted(
            index for index, minimum in required_coverage.items()
            if covered_counts[index] < minimum
        )
        if missing_positive:
            focus = {
                neighborhood_labels[index - 1]: [
                    {key: row[key] for key in ("id", "title", "artist") if key in row}
                    for row in numeric_records
                    if index in row.get("seed_neighborhoods", [])
                    and row["id"] not in selected_set
                    and row["id"] not in excluded_refs
                ]
                for index in missing_positive
            }
            coverage_prompt = (
                "Repair this specific musical coverage gap in your playlist. The original "
                "request included these positive song seeds, and your selection falls below "
                "the musical coverage planned for the real candidate availability. "
                "Choose musically fitting candidate replacements from EACH deficient neighborhood. "
                "Preserve required anchors and the playlist size. "
                "Return only JSON with swaps: "
                '{"swaps":[{"remove_id":1,"add_id":2}]}.\n'
                f"Original request: {original_request}\n"
                f"Required references: {json.dumps(sorted(required_refs))}\n"
                f"LLM2 neighborhood minima: {json.dumps(required_coverage, sort_keys=True)}\n"
                f"Current neighborhood counts: {json.dumps(covered_counts, sort_keys=True)}\n"
                f"Current selection: {json.dumps([by_ref[ref] for ref in valid_refs], ensure_ascii=False)}\n"
                f"Missing positive seed options: {json.dumps(focus, ensure_ascii=False)}"
            )
            coverage_raw, coverage_failure = call_phase(
                "Coverage repair", coverage_prompt,
                "Repair missing positive seed coverage. Return only JSON.", len(valid_refs),
            )
            if not coverage_failure:
                coverage_result, _, _, coverage_parse_error = parse_json_response(coverage_raw)
                proposed_swaps = coverage_result.get("swaps") if isinstance(coverage_result, dict) else None
                if (not coverage_parse_error and isinstance(proposed_swaps, list)
                        and set(coverage_result) == {"swaps"}):
                    for swap in proposed_swaps:
                        if not isinstance(swap, dict) or set(swap) != {"remove_id", "add_id"}:
                            continue
                        old_ref, new_ref = swap["remove_id"], swap["add_id"]
                        if (type(old_ref) is not int or type(new_ref) is not int
                                or old_ref not in selected_set or old_ref in required_refs
                                or new_ref not in by_ref or new_ref in selected_set
                                or new_ref in excluded_refs):
                            continue
                        valid_refs[valid_refs.index(old_ref)] = new_ref
                        selected_set.remove(old_ref)
                        selected_set.add(new_ref)
            covered_counts = Counter(
                neighborhood for ref in valid_refs
                for neighborhood in by_ref[ref].get("seed_neighborhoods", [])
            )
            still_missing = [
                neighborhood_labels[index - 1] for index in missing_positive
                if covered_counts[index] < required_coverage[index]
            ]
            if log_messages is not None:
                log_messages.append(
                    f"Compose positive seed coverage repair: "
                    f"{len(missing_positive) - len(still_missing)}/{len(missing_positive)} neighborhoods added"
                )
            if still_missing:
                return failed(
                    "COMPOSER_SEED_COVERAGE_GAP",
                    f"Composer left positive seed neighborhoods uncovered: {', '.join(still_missing)}.",
                )

    if neighborhood_minima:
        final_coverage = Counter(
            neighborhood for ref in valid_refs
            for neighborhood in by_ref[ref].get("seed_neighborhoods", [])
        )
        deficient = [
            neighborhood_labels[index - 1]
            for index, minimum in neighborhood_minima.items()
            if final_coverage[index] < minimum
        ]
        if deficient:
            return failed(
                "COMPOSER_SEED_COVERAGE_GAP",
                f"LLM2 seed selection missed its own musical plan: {', '.join(deficient)}.",
            )
        if log_messages is not None:
            log_messages.append(f"Compose Phase B final seed coverage: {json.dumps(final_coverage, sort_keys=True)}")

    if artist_policy and valid_refs:
        by_ref = {row["id"]: row for row in numeric_records}
        artist_cap = artist_policy["max_tracks_per_artist"]
        artist_floor = artist_policy["minimum_distinct_artists"]

        def artist_mix_gap(refs):
            counts = Counter(by_ref[ref].get("artist") or "Unknown" for ref in refs)
            excess = sum(max(0, count - artist_cap) for count in counts.values())
            return excess + max(0, artist_floor - len(counts)), counts

        gap, artist_counts = artist_mix_gap(valid_refs)
        if gap:
            remaining = [row for row in numeric_records
                         if row["id"] not in set(valid_refs) and row["id"] not in excluded_refs]
            repair_prompt = (
                "Improve the artist variety of this selected playlist using only the supplied "
                "real-library alternatives. Return JSON with swaps only: "
                '{"swaps":[{"remove_id":1,"add_id":2}]}. Preserve every required anchor, '
                "the playlist size, and each seed neighborhood's musical minimum. "
                "Choose musically fitting replacements; do not add unrelated music solely "
                "to meet a number. Return an empty swaps list if no suitable improvement exists.\n"
                f"Original request: {original_request}\n"
                f"Discovery priority: {intent.get('discovery_priority', 'balanced')}\n"
                f"Artist mix chosen by LLM2: {json.dumps(artist_policy)}\n"
                f"Current distinct artists: {len(artist_counts)}\n"
                f"Current artist counts: {json.dumps(artist_counts.most_common(), ensure_ascii=False)}\n"
                f"Required references: {json.dumps(sorted(required_refs))}\n"
                f"Seed minima: {json.dumps(neighborhood_minima, sort_keys=True)}\n"
                f"Current selection: {json.dumps([by_ref[ref] for ref in valid_refs], ensure_ascii=False)}\n"
                f"Alternatives: {json.dumps(remaining, ensure_ascii=False)}"
            )
            repair_raw, repair_failure = call_phase(
                "Artist mix repair", repair_prompt,
                "Improve musical artist variety using only supplied candidates. Return JSON.",
                len(valid_refs),
            )
            if not repair_failure:
                proposal, _, _, parse_error = parse_json_response(repair_raw)
                swaps = proposal.get("swaps") if isinstance(proposal, dict) else None
                if not parse_error and isinstance(swaps, list) and set(proposal) == {"swaps"}:
                    for swap in swaps:
                        if not isinstance(swap, dict) or set(swap) != {"remove_id", "add_id"}:
                            continue
                        old, new = swap["remove_id"], swap["add_id"]
                        if (type(old) is not int or type(new) is not int or old in required_refs
                                or old not in valid_refs or new not in eligible_refs
                                or new in valid_refs or new in excluded_refs):
                            continue
                        candidate = list(valid_refs)
                        candidate[candidate.index(old)] = new
                        if any(
                            sum(index in by_ref[ref].get("seed_neighborhoods", [])
                                for ref in candidate) < minimum
                            for index, minimum in neighborhood_minima.items()
                        ):
                            continue
                        new_gap, _ = artist_mix_gap(candidate)
                        if new_gap < gap:
                            valid_refs = candidate
                            gap = new_gap
            if log_messages is not None:
                log_messages.append(f"Compose artist mix repair: remaining gap={gap}")
        if log_messages is not None:
            final_gap, final_counts = artist_mix_gap(valid_refs)
            log_messages.append(
                f"Compose artist variety: {len(final_counts)} distinct artists; "
                f"largest artist share={max(final_counts.values())}/{len(valid_refs)}; "
                f"LLM2 target gap={final_gap}"
            )

    authoritative = [numeric_to_song[ref] for ref in valid_refs]
    if log_messages is not None:
        log_messages.append(
            f"Compose selection validation: required anchors={len(required_refs)}/{len(required_refs)}; "
            f"final tracks={len(authoritative)}"
        )
        log_messages.append(
            f"Composer preferred tracks: {len(authoritative)}" if target_duration is not None
            else f"Final playlist: {len(authoritative)}"
        )
        log_messages.append(f"Composer result: playlist IDs returned={raw_count}")
    return {
        "playlist": authoritative,
        "requested_output": {"target_count": target_count, "target_duration_seconds": target_duration},
        "anchor_decisions": [{"id": key, "include": decisions[key]} for key in sorted(decisions)],
        "shortfall_reason": shortfall,
        "candidate_count": len(records),
        "preferred_final_count": preferred_final_count,
    }, len(records)


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


def finalize_composer_duration(songs, duration_rows, target_seconds, *, target_count=None,
                               required_ids=(), tolerance_seconds=15,
                               preferred_count=None):
    """Select from Composer preference order using library durations, then validate."""
    durations = {
        str(item_id): row.get('duration')
        for item_id, row in duration_rows.items()
    }
    required = list(dict.fromkeys(map(str, required_ids)))
    available = {str(song.get('item_id')) for song in songs}
    if any(item_id not in available for item_id in required):
        return [], None, {'reason': 'A required Composer anchor is absent from the candidate pool.'}
    diagnostics = {}
    if target_count is None and type(preferred_count) is int and 0 < preferred_count <= len(songs):
        preferred_diagnostics = {}
        preferred = optimize_playlist_duration(
            songs, durations, int(target_seconds), preferred_count,
            max(1, len(songs)), exact_count=True,
            mandatory_ids=required,
            ranked_ids=[song.get('item_id') for song in songs],
            tolerance_seconds=tolerance_seconds, diagnostics=preferred_diagnostics,
        )
        preferred_seconds = preferred_diagnostics.get('selected_duration')
        if (len(preferred) == preferred_count and preferred_seconds is not None
                and abs(preferred_seconds - int(target_seconds)) <= tolerance_seconds):
            diagnostics.update(preferred_diagnostics)
            diagnostics['composer_preferred_count_used'] = True
            diagnostics['actual_seconds'] = preferred_seconds
            diagnostics['error_seconds'] = preferred_seconds - int(target_seconds)
            diagnostics['within_tolerance'] = True
            return preferred, preferred_seconds, diagnostics
        diagnostics['composer_preferred_count_used'] = False
    selected = optimize_playlist_duration(
        songs, durations, int(target_seconds),
        int(target_count) if target_count is not None else len(songs),
        max(1, len(songs)), exact_count=target_count is not None,
        mandatory_ids=required,
        ranked_ids=[song.get('item_id') for song in songs],
        tolerance_seconds=tolerance_seconds, diagnostics=diagnostics,
    )
    if not selected and songs:
        # A zero-track playlist cannot satisfy a positive duration request.
        # The optimizer's empty state can otherwise beat an overlong single
        # track numerically, hiding the nearest playable result.
        valid_singles = []
        for song in songs:
            try:
                seconds = int(float(durations.get(str(song.get('item_id')))))
            except (TypeError, ValueError):
                continue
            if seconds > 0:
                valid_singles.append((abs(seconds - int(target_seconds)), song))
        if valid_singles:
            selected = [min(valid_singles, key=lambda pair: pair[0])[1]]
    if target_count is not None and len(selected) != int(target_count):
        diagnostics['reason'] = 'The requested track count could not be satisfied.'
    values = []
    for song in selected:
        try:
            seconds = int(float(durations.get(str(song.get('item_id')))))
            if seconds <= 0:
                raise ValueError('nonpositive duration')
            values.append(seconds)
        except (TypeError, ValueError):
            diagnostics['reason'] = 'Authoritative duration is missing for a selected track.'
            return selected, None, diagnostics
    actual = sum(values)
    diagnostics['actual_seconds'] = actual
    diagnostics['error_seconds'] = actual - int(target_seconds)
    diagnostics['within_tolerance'] = abs(actual - int(target_seconds)) <= int(tolerance_seconds)
    if not diagnostics['within_tolerance']:
        diagnostics['reason'] = 'No available subset met the configured duration tolerance.'
    return selected, actual, diagnostics


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
