# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Prompt and JSON-schema templates for the playlist AI.

Central store of every prompt string the AI layer sends: the playlist-naming
template, the single-call tool-router system prompt, the Ollama tool-calling
prompt, and the grounded brainstorm recipe prompt. Consumed by ``planner``,
``api``, ``tool_impl``, and providers.

Main Features:
* Tool prose and the Ollama structured-output grammar are both DERIVED from the get_mcp_tools schemas (names, descriptions, per-argument types and enums), so the routing knowledge lives in one place and stays in sync across providers.
* build_title_naming_prompt renders the classic full-title naming prompt: the user-editable instructions followed by a fixed sample of the playlist songs, so editing the instructions can never change what data is sent.
* EXAMPLE_TEXT_QUERIES holds the example text_match queries, shared with the planner so a plan that copies one is caught.
* build_tool_calls_schema emits a typed per-tool grammar (reasoning field first with a hard maxLength, name+arguments branches with enum-locked labels, array caps from the shared maxItems) used to constrain Ollama structured output; prompts stay short with a few diverse worked examples per intent class, including a three-tool plan, exclusion ('no rap') and language/scene routing rules.
"""

import copy
import json
from typing import Dict, List, Optional

import config


NAMING_MODES = ('concept', 'title')

TITLE_PROMPT_PLAYLIST_HEADER = 'This is the playlist:\n'


def normalize_naming_mode(mode) -> str:
    mode = str(mode or '').strip().lower()
    return mode if mode in NAMING_MODES else NAMING_MODES[0]


def title_prompt_song_block(songs, max_songs: int) -> str:
    sample = list(songs or [])[: max(1, int(max_songs))]
    lines = '\n'.join(
        f"- {title or 'Unknown Title'} by {author or 'Unknown Artist'}"
        for _item_id, title, author in sample
    )
    return f'{TITLE_PROMPT_PLAYLIST_HEADER}{lines}\n\n'


TITLE_PROMPT_RECENT_TITLES = 8


def title_prompt_used_titles_block(used_titles) -> str:
    recent = [title for title in (used_titles or []) if title][-TITLE_PROMPT_RECENT_TITLES:]
    if not recent:
        return ''
    return 'Titles already used, do not reuse them: ' + ' | '.join(recent) + '\n\n'


def build_title_naming_prompt(instructions: str, songs, max_songs: int, used_titles=None) -> str:
    return (
        (instructions or '').rstrip()
        + '\n\n'
        + title_prompt_song_block(songs, max_songs)
        + title_prompt_used_titles_block(used_titles)
    )


playlist_concept_prompt_template = (
    "Concept extraction only. Genre: {genre}. Verified evidence: {evidence}. "
    "{dimension_rule} Use one ordinary word. Concept only: no genre, title, "
    "explanation, marketing/container word, or invented context. {avoid_rule}"
    "Return {candidate_count} different candidate words on one line, comma "
    "separated, best first. No numbering, no explanation."
)


def _get_dynamic_genres(library_context: Optional[Dict]) -> str:
    if library_context and library_context.get('top_genres'):
        return ', '.join(library_context['top_genres'][:10])
    return config.AI_FALLBACK_GENRES


def _render_tool_line(tool: Dict) -> str:
    props = (tool.get('inputSchema') or {}).get('properties') or {}
    args = ", ".join(props.keys())
    return f"- {tool['name']}({args}): {tool['description']}"


def build_mcp_system_prompt(
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    tool_names = {t.get('name') for t in tools}
    tools_block = "\n".join(_render_tool_line(t) for t in tools)

    genres_line = _get_dynamic_genres(library_context)
    voices_line = ", ".join(v for v in config.VOICE_VOCAB if v.endswith('vocalists'))
    moods_line = ", ".join(config.OTHER_FEATURE_LABELS)

    rules: List[str] = []
    finder_options: List[str] = []
    if 'seed_search' in tool_names:
        finder_options.append(
            "the user names a song/artist to imitate ('like X', 'similar to X') -> seed_search"
        )
    if 'text_match' in tool_names:
        finder_options.append(
            "the user describes a sound or a lyric topic -> text_match"
        )
    if 'knowledge_lookup' in tool_names:
        finder_options.append(
            "the user asks for popular/famous/'best of' songs without naming a "
            "specific artist, or for a language/nationality/scene ('French rap', "
            "'Italian pop', 'K-pop') -> knowledge_lookup"
        )
    finder_options.append(
        "the request is plain metadata only -> search_database by itself"
    )
    rules.append("Pick how to FIND songs: " + "; ".join(finder_options) + ".")
    rules.append(
        "Put EVERY stated metadata constraint (genre, voice, mood, year/decade, tempo, "
        "energy, key, scale, rating, artist, album, instrumental) into ONE search_database "
        "call, next to the finder tool when there is one."
    )
    rules.append(
        "Exclusions ('no X', 'without X', 'except X', 'anything but X') go in "
        "search_database exclude_artists/exclude_genres. NEVER put an excluded name in "
        "seeds, in a text_match query, or in the positive artist/genres fields."
    )
    if 'seed_search' in tool_names:
        rules.append(
            "An artist's own songs ('songs by X', 'play X', 'best of X', where X is an "
            "artist's name) -> search_database with artist='X'. Similar to X ('like X', "
            "'sounds like X', 'in the style of X') -> seed_search."
        )
    rules.append(
        "Fill only fields the user asked for, using the closest listed value; when no "
        "field or listed value fits a word, leave it out."
    )
    rules.append(
        "Emit each tool at most once. Use one finder tool per distinct part of the request "
        "(a named artist to resemble, a sound or topic to match, a popularity ask), plus the "
        "one search_database that carries every metadata constraint."
    )
    rules_block = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, start=1))

    return f"""You are a music playlist planner. Turn the user's request into tool calls; the app runs them against the user's own music library.

TOOLS:
{tools_block}

VALUES for search_database:
- genres in this library: {genres_line}
- voices: {voices_line}
- moods: {moods_line}
- scale: major or minor. key: tonic note like C or F# (major/minor goes in scale).
- Decade words map to years: '90s' -> year_min 1990, year_max 1999. energy 0.0-1.0 (calm <= 0.35, intense >= 0.7). tempo 40-200 BPM (slow <= 90, fast >= 130). min_rating 1-5.

HOW TO PLAN:
{rules_block}"""


EXAMPLE_TEXT_QUERIES = (
    "warm rhodes keys, mellow downtempo groove",
    "soft acoustic guitar for studying",
    "growing old",
)


def _example(reasoning: str, calls: List[Dict]) -> str:
    return json.dumps({"reasoning": reasoning, "tool_calls": calls}, ensure_ascii=True)


def _text_match_modes(tools: List[Dict]) -> set:
    for t in tools:
        if t.get('name') == 'text_match':
            props = (t.get('inputSchema') or {}).get('properties') or {}
            return set((props.get('mode') or {}).get('enum') or [])
    return set()


def _build_examples(tools: List[Dict]) -> List[str]:
    tool_names = {t.get('name') for t in tools}
    modes = _text_match_modes(tools)

    examples: List[str] = []
    if 'search_database' in tool_names:
        examples.append(
            '"energetic songs by Artist A"\n'
            + _example(
                "Artist A's own songs, filtered to high energy.",
                [
                    {
                        "name": "search_database",
                        "arguments": {"artist": "Artist A", "energy_min": 0.65},
                    }
                ],
            )
        )
        examples.append(
            '"aggressive metal from the 80s"\n'
            + _example(
                "Pure metadata: genre metal, mood aggressive, decade 1980s.",
                [
                    {
                        "name": "search_database",
                        "arguments": {
                            "genres": ["metal"],
                            "moods": ["aggressive"],
                            "year_min": 1980,
                            "year_max": 1989,
                        },
                    }
                ],
            )
        )
        examples.append(
            '"party songs but absolutely no rap and nothing by Artist B"\n'
            + _example(
                "Party mood with a genre and an artist exclusion.",
                [
                    {
                        "name": "search_database",
                        "arguments": {
                            "moods": ["party"],
                            "exclude_genres": ["Hip-Hop"],
                            "exclude_artists": ["Artist B"],
                        },
                    }
                ],
            )
        )
    if 'seed_search' in tool_names and 'search_database' in tool_names:
        examples.append(
            '"like Song 1 by Artist C but with a female voice"\n'
            + _example(
                "Songs similar to a named track, constrained to female vocals.",
                [
                    {
                        "name": "seed_search",
                        "arguments": {
                            "seeds": [
                                {"type": "song", "title": "Song 1", "artist": "Artist C"}
                            ]
                        },
                    },
                    {
                        "name": "search_database",
                        "arguments": {"voices": ["female vocalists"]},
                    },
                ],
            )
        )
    if (
        {'seed_search', 'text_match', 'search_database'} <= tool_names
        and 'audio' in modes
    ):
        examples.append(
            '"chill 2000s songs like Artist D with a warm rhodes sound, nothing by Artist E"\n'
            + _example(
                "Three parts: an artist to resemble, a sound to match, and the era plus "
                "the exclusion.",
                [
                    {
                        "name": "seed_search",
                        "arguments": {"seeds": [{"type": "artist", "name": "Artist D"}]},
                    },
                    {
                        "name": "text_match",
                        "arguments": {
                            "query": EXAMPLE_TEXT_QUERIES[0],
                            "mode": "audio",
                        },
                    },
                    {
                        "name": "search_database",
                        "arguments": {
                            "year_min": 2000,
                            "year_max": 2009,
                            "exclude_artists": ["Artist E"],
                        },
                    },
                ],
            )
        )
    if 'seed_search' in tool_names:
        examples.append(
            '"in the style of Band F but not Band G"\n'
            + _example(
                "Similar to one named artist while removing another's flavor.",
                [
                    {
                        "name": "seed_search",
                        "arguments": {
                            "seeds": [{"type": "artist", "name": "Band F"}],
                            "blend_mode": "subtract",
                            "subtract": [{"type": "artist", "name": "Band G"}],
                        },
                    }
                ],
            )
        )
    if 'text_match' in tool_names and 'audio' in modes:
        examples.append(
            '"soft acoustic guitar for studying"\n'
            + _example(
                "A sound description, matched by how the music sounds.",
                [
                    {
                        "name": "text_match",
                        "arguments": {"query": EXAMPLE_TEXT_QUERIES[1], "mode": "audio"},
                    }
                ],
            )
        )
    if 'text_match' in tool_names and 'lyrics' in modes:
        examples.append(
            '"songs about growing old"\n'
            + _example(
                "A lyric topic, matched by what the words are about.",
                [
                    {
                        "name": "text_match",
                        "arguments": {"query": EXAMPLE_TEXT_QUERIES[2], "mode": "lyrics"},
                    }
                ],
            )
        )
    if 'knowledge_lookup' in tool_names:
        examples.append(
            '"greatest disco hits of the 70s"\n'
            + _example(
                "A popularity request that needs world knowledge.",
                [
                    {
                        "name": "knowledge_lookup",
                        "arguments": {"user_request": "greatest disco hits of the 70s"},
                    }
                ],
            )
        )
    return examples


def build_ollama_tool_calling_prompt(
    user_message: str,
    tools: List[Dict],
    library_context: Optional[Dict] = None,
) -> str:
    system_prompt = build_mcp_system_prompt(tools, library_context)
    examples_text = "\n\n".join(_build_examples(tools))

    return f"""{system_prompt}

OUTPUT: return ONLY one JSON object, no other text:
{{"reasoning": "one short sentence: what to find and which tools", "tool_calls": [{{"name": "tool_name", "arguments": {{...}}}}]}}

EXAMPLES:
{examples_text}

Request: "{user_message}"
Fill only fields the user asked for. Return ONLY the JSON object."""


def build_tool_calls_schema(tools: List[Dict]) -> Dict:
    branches: List[Dict] = []
    for t in tools:
        name = t.get('name')
        if not name:
            continue
        arg_schema = copy.deepcopy(t.get('inputSchema') or {"type": "object"})
        arg_schema['additionalProperties'] = False
        branches.append(
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "enum": [name]},
                    "arguments": arg_schema,
                },
                "required": ["name", "arguments"],
            }
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reasoning": {
                "type": "string",
                "maxLength": 300,
                "description": "One short sentence: what to find and which tools.",
            },
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": config.AI_MAX_TOOL_CALLS,
                "items": {"oneOf": branches} if branches else {"type": "object"},
            },
        },
        "required": ["reasoning", "tool_calls"],
    }


def build_playlist_plan_tool(tools: List[Dict], retrieval_only: bool = False) -> Dict:
    """Require structured user intent alongside the complete retrieval plan."""
    call_branches = []
    for tool in tools:
        name = tool.get('name')
        if not name:
            continue
        call_branches.append({
            "type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "enum": [name]},
                "arguments": copy.deepcopy(tool.get('inputSchema') or {"type": "object"}),
            },
            "required": ["name", "arguments"],
        })
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    anchor_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["song", "artist", "album"]},
            "title": nullable_string, "artist": nullable_string,
            "name": nullable_string, "album": nullable_string,
            "role": {"type": "string", "enum": ["anchor", "mandatory", "include", "reference_only", "exclusion", "preferred_artist", "style_reference", "start", "destination"]},
            "include_in_final": {"type": "boolean"},
        },
        "required": ["type", "title", "artist", "name", "album", "role", "include_in_final"],
    }
    count_schema = {"anyOf": [
        {"type": "null"},
        {"type": "object", "additionalProperties": False,
         "properties": {"mode": {"type": "string", "enum": ["total", "additional"]},
                        "value": {"type": "integer", "minimum": 1}},
         "required": ["mode", "value"]},
    ]}
    nullable_integer = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    constraints_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "genres": {"type": "array", "items": {"type": "string"}},
            "moods": {"type": "array", "items": {"type": "string"}},
            "voices": {"type": "array", "items": {"type": "string"}},
            "year_min": nullable_integer, "year_max": nullable_integer,
            "energy_min": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "energy_max": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "tempo_min": nullable_integer, "tempo_max": nullable_integer,
            "exclude_artists": {"type": "array", "items": {"type": "string"}},
            "exclude_genres": {"type": "array", "items": {"type": "string"}},
            "max_per_artist": nullable_integer,
            "artist": nullable_string,
            "allow_multiple_versions": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        },
        "required": ["genres", "moods", "voices", "year_min", "year_max", "energy_min", "energy_max", "tempo_min", "tempo_max", "exclude_artists", "exclude_genres", "max_per_artist", "artist", "allow_multiple_versions"],
    }
    intent_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "anchors": {"type": "array", "items": anchor_schema},
            "count": count_schema,
            "duration_seconds": nullable_integer,
            "constraints": constraints_schema,
            "playlist_intent": {"type": "string"},
            "activity": nullable_string,
            "lyrical_theme": nullable_string,
            "transition_intent": nullable_string,
            "ordering_intent": nullable_string,
            "diversity_intent": nullable_string,
            "similarity_intent": nullable_string,
            "retrieval_size_hint": nullable_integer,
        },
        "required": ["anchors", "count", "duration_seconds", "constraints", "playlist_intent", "activity", "lyrical_theme", "transition_intent", "ordering_intent", "diversity_intent", "similarity_intent", "retrieval_size_hint"],
    }
    if retrieval_only:
        # In compose mode LLM1 is deliberately unable to return final-playlist
        # decisions such as inclusion, count semantics, duration, or ordering.
        retrieval_anchor_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "enum": ["song", "artist", "album"]},
                "title": nullable_string, "artist": nullable_string,
                "name": nullable_string, "album": nullable_string,
            },
            "required": ["type", "title", "artist", "name", "album"],
        }
        intent_schema = {
            "type": "object", "additionalProperties": False,
            "properties": {
                "anchors": {"type": "array", "items": retrieval_anchor_schema},
                "retrieval_size_hint": nullable_integer,
            },
            "required": ["anchors", "retrieval_size_hint"],
        }
    wrapper_schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "intent": intent_schema,
            "tool_calls": {
                "type": "array", "minItems": 1,
                "maxItems": max(12, int(config.AI_MAX_TOOL_CALLS) * 3),
                "items": {"oneOf": call_branches} if call_branches else {"type": "object"},
            },
        }, "required": ["intent", "tool_calls"],
    }
    tool_lines = "\n".join(f"- {tool['name']}: {tool.get('description', '')}" for tool in tools)
    return {
        "name": "submit_playlist_plan",
        "description": (
            (
                "Plan retrieval only: identify human-readable song, artist, and album references; "
                "choose AudioMuse retrieval tools and a retrieval-size hint. Never decide final playlist "
                "membership, anchor inclusion, count semantics, duration, exclusions, versions, artist "
                "limits, or order. Preserve any explicitly supplied song title and artist exactly as written "
                "in the anchor fields; do not substitute an artist based on outside knowledge. Use the UI "
                "song count as the retrieval-size hint unless the request states a different numeric size. "
                "The hint is a target for EACH seed neighborhood, not a global result count. Include each song "
                "reference in seed_search. Retrieval filters may describe requested sound/style.\n"
            ) if retrieval_only else (
            "Interpret the complete user request into authoritative structured semantic intent, "
            "then produce all retrieval/tool calls needed to carry it out. The semantic intent "
            "must express every named song/artist/album and whether each is mandatory, reference-only, "
            "excluded, a preferred artist, style reference, start, or destination. Resolve total versus "
            "additional song counts, explicit duration, filters/exclusions, playlist purpose, and ordering "
            "intent from the user's meaning. Do not infer semantics in the application from raw wording. "
            "A seed artist is not an artist filter unless the request independently constrains the artist. "
            "Set retrieval_size_hint for enough candidates to satisfy the final target after filtering and "
            "selection; it is independent from the final playlist size. Include every song anchor in "
            "seed_search and use all applicable retrieval tools. Return one complete plan.\n"
            )
            + "Available retrieval tools:\n" + tool_lines
        ),
        "inputSchema": wrapper_schema,
    }


def build_ai_brainstorm_prompt(user_request: str) -> str:
    from tasks.ai.vocab import GENRE_VOCAB

    genres_line = ", ".join(GENRE_VOCAB)
    moods_line = ", ".join(config.OTHER_FEATURE_LABELS)
    voices_line = ", ".join(config.VOICE_VOCAB)
    return f"""You are a music expert. Turn the request into a RECIPE used to search a music library.
You do NOT know which songs are in the library, so you MUST NOT name any songs. Describe and categorise only; the library does the finding.

User request: "{user_request}"

Return ONE JSON object with EXACTLY this shape:
{{"filters": {{"genres": [], "moods": [], "voices": [], "year_min": null, "year_max": null, "energy_min": null, "energy_max": null, "tempo_min": null, "tempo_max": null}}, "sound_descriptions": [], "seed_artists": [], "lyric_themes": []}}

FIELD GUIDE (leave a field empty/null when the request does not imply it -- never invent constraints):
- filters.genres: 0+ values, chosen ONLY from: {genres_line}
- filters.moods: 0+ values, chosen ONLY from: {moods_line}
- filters.voices: 0+ values, chosen ONLY from: {voices_line}
- filters.year_min / year_max: 4-digit years. A decade like "90s" -> 1990 and 1999. "90s and 2000s" -> 1990 and 2009.
- filters.energy_min / energy_max: numbers 0.0 (calm) to 1.0 (intense).
- filters.tempo_min / tempo_max: BPM, 40 to 200.
- sound_descriptions: 2 to {config.AI_BRAINSTORM_SOUND_DESCRIPTIONS_MAX} vivid phrases describing HOW the ideal songs SOUND (instruments, production, era, energy, vibe). This is the most important field. NOT song names.
- seed_artists: up to {config.AI_BRAINSTORM_SEED_ARTISTS_MAX} well-known ARTISTS that exemplify the request. Artists ONLY, never songs. Omit if none are obvious.
- lyric_themes: 0 to {config.AI_BRAINSTORM_LYRIC_THEMES_MAX} short phrases ONLY when the request is about a TOPIC the lyrics should cover (e.g. "heartbreak", "summer roadtrip").

RULES:
- NEVER output a song title anywhere.
- genres / moods / voices MUST come from the lists above, or be left empty.
- Output ONLY the JSON object. No markdown fences, no comments, no extra text.

EXAMPLE -- request "100 of the best rap songs from the 90s and 2000s" (Artist A to Artist D stand for real, well-known artists that fit the request):
{{"filters": {{"genres": ["Hip-Hop"], "moods": [], "voices": [], "year_min": 1990, "year_max": 2009, "energy_min": 0.5, "energy_max": 1.0, "tempo_min": null, "tempo_max": null}}, "sound_descriptions": ["gritty 90s east coast boom bap hip hop with hard-hitting drums and jazzy samples", "glossy early 2000s mainstream rap with heavy bass and crossover hooks"], "seed_artists": ["Artist A", "Artist B", "Artist C", "Artist D"], "lyric_themes": []}}

Now produce the JSON recipe for "{user_request}":"""
