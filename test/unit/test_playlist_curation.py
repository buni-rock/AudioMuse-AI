import json

import pytest

from tasks.playlist_curation import (
    build_llm_candidate_payload,
    prepare_llm_candidate_shortlist,
    curate_candidates_with_llm,
    effective_llm_artist_cap,
    optimize_playlist_duration,
    rank_candidates_by_ids,
    reorder_preserving_membership,
    inspect_llm_candidate_selection,
    _rerank_context_key,
    validate_llm_candidate_selection,
)


def _install_compact_composer_response(monkeypatch, *, target_count, selected_count=None, shortfall=None):
    import tasks.ai.api as api

    captured = []

    def generate(prompt, _config, **kwargs):
        captured.append((prompt, kwargs))
        if prompt.startswith('Interpret the ORIGINAL'):
            anchors = json.loads(prompt.split('Resolved anchors: ', 1)[1].split('\nAvailable candidate count:', 1)[0])
            payload = {'anchor_decisions': {anchor['id']: True for anchor in anchors},
                       'target_count': target_count, 'target_duration_seconds': None}
        elif prompt.startswith('Select only the missing tracks'):
            remaining = json.loads(prompt.split('Remaining available candidates: ', 1)[1])
            missing = int(prompt.split('Missing count: ', 1)[1].split('\n', 1)[0])
            payload = {'playlist_ids': [row['id'] for row in remaining[:missing]]}
        else:
            records = json.loads(prompt.split('Available candidates: ', 1)[1].split('\n\nREPAIR', 1)[0])
            anchors = json.loads(prompt.split('Resolved anchors: ', 1)[1].split('\nAvailable candidates:', 1)[0])
            required = [anchor['candidate_ref'] for anchor in anchors if anchor['include']]
            ids = required + [row['id'] for row in records if row['id'] not in required]
            payload = {'playlist_ids': ids[:selected_count or target_count]}
            if shortfall is not None:
                payload['shortfall_reason'] = shortfall
        raw = json.dumps(payload, separators=(',', ':'))
        metadata = kwargs.get('call_metadata')
        if metadata is not None:
            metadata.update(done_reason='stop', eval_count=80, assistant_content_length=len(raw))
        return raw

    monkeypatch.setattr(api, 'generate_text', generate)
    return captured


def test_compose_debug_fixture_ten_candidates_two_anchors_five_tracks(monkeypatch):
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Fixture Artist'}
             for index in range(10)]
    anchors = [{'type': 'song', 'resolved_track': song} for song in songs[:2]]
    logs = []
    calls = _install_compact_composer_response(monkeypatch, target_count=5)
    result, sent = compose_playlist_with_llm(
        'Include the two listed songs in a five-track playlist', songs,
        {'provider': 'OLLAMA'}, resolved_anchors=anchors, ui_default_count=50, log_messages=logs,
    )

    assert sent == 10
    assert 'Composer context present: resolved anchors=2' in logs
    assert len(result['playlist']) == 5
    assert len(calls) == 2
    assert all(kwargs['structured_format'] == 'json' and kwargs['max_tokens'] == 512 for _, kwargs in calls)
    assert 'playlist_ids' not in calls[0][0]
    assert 'request-local INTEGER references' in calls[1][0]
    assert 'Original user request (verbatim):' in calls[1][0]
    assert any('Compose Phase A: target_count=5' in line for line in logs)
    assert any('Compose Phase B: IDs returned=5; unique IDs=5' in line for line in logs)


def test_compose_keeps_all_242_candidates_and_four_canonical_anchors(monkeypatch):
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': f'Artist {index % 17}'}
             for index in range(242)]
    anchor_specs = [
        ('track-0', 'Temple Of The King', 'Rainbow'),
        ('track-1', 'Loud And Clear', 'The Cranberries'),
        ('track-2', 'Every Breaking Wave', 'U2'),
        ('track-3', 'Paradise (What About Us?) (Feat. Tarja)', 'Within Temptation'),
    ]
    anchors = []
    for item_id, title, artist in anchor_specs:
        track = next(song for song in songs if song['item_id'] == item_id)
        track.update(title=title, artist=artist)
        anchors.append({'type': 'song', 'title': title, 'artist': artist,
                        'resolved_track': track})
    logs = []
    calls = _install_compact_composer_response(monkeypatch, target_count=64)
    result, sent = compose_playlist_with_llm(
        'Here is a list of songs I love:\n'
        '- Temple of the King from Rainbow,\n'
        '- Lound and clear from The Cranberries,\n'
        '- Every breaking wave from U2,\n'
        '- Paradise from Within Temptation.\n\n'
        'Could you assemble a playlist to contain these songs and add another\n'
        '60 similar songs?', songs,
        {'provider': 'OLLAMA'}, resolved_anchors=anchors, ui_default_count=50, log_messages=logs,
    )

    assert sent == 242
    assert len(calls) == 2
    assert len(result['playlist']) == 64
    assert all(any(song['item_id'] == item_id for song in result['playlist'])
               for item_id, _, _ in anchor_specs)
    assert result['requested_output']['target_count'] == 64
    assert any('Composer anchor A004: Paradise (What About Us?) (Feat. Tarja) / Within Temptation' in line for line in logs)
    assert any('Compose Phase A: target_count=64; target_duration=None; anchor decisions=' in line for line in logs)
    assert any('Compose Phase B: candidates supplied=242; requested selections=64' in line for line in logs)
    assert any('Compose Phase B: IDs returned=64; unique IDs=64' in line for line in logs)


def test_compose_truncated_selection_attempts_only_one_targeted_fill(monkeypatch, caplog):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []
    raw = '{"playlist_ids":[1,2,3'

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL'):
            kwargs['call_metadata'].update(done_reason='stop', eval_count=30)
            return '{"anchor_decisions":{},"target_count":5,"target_duration_seconds":null}'
        kwargs['call_metadata'].update(done_reason='length', eval_count=4096)
        return raw

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'Choose five songs', [{'item_id': 'track-1', 'title': 'Track 1', 'artist': 'Artist'}],
        {'provider': 'OLLAMA'},
    )
    assert result['error']['category'] == 'OUTPUT_LIMIT'
    assert len(calls) == 3
    assert calls[-1].startswith("Select only the missing tracks")


@pytest.mark.parametrize('shortfall_reason', [None, 'Only some tracks are suitable.'])
def test_compose_trims_overselection_after_validation(monkeypatch, shortfall_reason):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return '{"anchor_decisions":{},"target_count":64,"target_duration_seconds":null}'
        records = json.loads(prompt.split('Available candidates: ', 1)[1].split('\n\nREPAIR', 1)[0])
        payload = {'playlist_ids': [record['id'] for record in records]}
        if shortfall_reason is not None:
            payload['shortfall_reason'] = shortfall_reason
        return json.dumps(payload)

    monkeypatch.setattr(api, 'generate_text', generate)
    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(242)]
    result, sent = compose_playlist_with_llm('Choose 64 tracks', songs, {'provider': 'OLLAMA'})
    assert sent == 242
    assert len(calls) == 2
    assert len(result['playlist']) == 64
    assert result['shortfall_reason'] is None
    records = json.loads(calls[1].split('Available candidates: ', 1)[1])
    assert [song['title'] for song in result['playlist']] == [row['title'] for row in records[:64]]


def test_compose_fills_explained_shortfall_without_regenerating_selection(monkeypatch):
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(10)]
    anchors = [{'type': 'song', 'resolved_track': song} for song in songs[:2]]
    calls = _install_compact_composer_response(
        monkeypatch, target_count=5, selected_count=3,
        shortfall='Only three candidates suit the request.',
    )
    result, _ = compose_playlist_with_llm(
        'Choose five tracks', songs, {'provider': 'OLLAMA'}, resolved_anchors=anchors,
    )
    assert len(result['playlist']) == 5
    assert len(calls) == 3
    assert calls[2][0].startswith('Select only the missing tracks')
    assert result['requested_output']['target_count'] == 5
    assert result['shortfall_reason'] is None


@pytest.mark.parametrize('raw_refs,valid_count,fill_count,trimmed', [
    (list(range(1, 64)) + [1, 2, 3, 1000, 'bad', True], 63, 1, 0),
    (list(range(1, 75)) + [1], 74, 0, 10),
    (list(range(1, 62)), 61, 3, 0),
    (list(range(1, 65)), 64, 0, 0),
    (list(range(1, 68)), 67, 0, 3),
])
def test_compose_finalizes_count_variance(monkeypatch, raw_refs, valid_count, fill_count, trimmed):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return '{"anchor_decisions":{},"target_count":64,"target_duration_seconds":null}'
        if prompt.startswith('Select only the missing tracks'):
            return json.dumps({'playlist_ids': list(range(valid_count + 1, valid_count + fill_count + 1))})
        return json.dumps({'playlist_ids': raw_refs})

    monkeypatch.setattr(api, 'generate_text', generate)
    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(242)]
    logs = []
    result, sent = compose_playlist_with_llm(
        'Choose 64 songs', songs, {'provider': 'OLLAMA'}, log_messages=logs,
    )
    assert sent == 242
    assert len(result['playlist']) == 64
    assert len(calls) == (3 if fill_count else 2)
    assert not any('selection repair' in line for line in logs)
    records = json.loads(calls[1].split('Available candidates: ', 1)[1])
    assert [song['title'] for song in result['playlist']] == [row['title'] for row in records[:64]]
    assert any(f'Composer raw IDs: {len(raw_refs)}; Valid IDs: {valid_count};' in line for line in logs)
    if fill_count:
        assert f'Targeted fill requested: {fill_count}' in logs
        assert f'Targeted fill returned: {fill_count}' in logs
    if trimmed:
        assert f'Tail trimmed: {trimmed}' in logs


def test_compose_repairs_duplicate_selection_without_reinterpreting(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return '{"anchor_decisions":{},"target_count":3,"target_duration_seconds":null}'
        return '{"playlist_ids":[1,1,2]}' if len(calls) == 2 else '{"playlist_ids":[3]}'

    monkeypatch.setattr(api, 'generate_text', generate)
    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(10)]
    result, _ = compose_playlist_with_llm('Choose three tracks', songs, {'provider': 'OLLAMA'})
    assert len(result['playlist']) == 3
    assert len(calls) == 3
    assert sum(prompt.startswith('Interpret the ORIGINAL') for prompt in calls) == 1
    assert calls[2].startswith('Select only the missing tracks')


def test_compose_repairs_missing_required_anchor(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return '{"anchor_decisions":{"A001":true},"target_count":3,"target_duration_seconds":null}'
        anchors = json.loads(prompt.split('Resolved anchors: ', 1)[1].split('\nAvailable candidates:', 1)[0])
        anchor_ref = anchors[0]['candidate_ref']
        choices = [ref for ref in range(1, 5) if ref != anchor_ref][:3]
        if len(calls) == 3:
            choices[0] = anchor_ref
        return json.dumps({'playlist_ids': choices})

    monkeypatch.setattr(api, 'generate_text', generate)
    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(10)]
    result, _ = compose_playlist_with_llm(
        'Include the named song in three tracks', songs, {'provider': 'OLLAMA'},
        resolved_anchors=[{'type': 'song', 'resolved_track': songs[0]}],
    )
    assert len(calls) == 2
    assert len(result['playlist']) == 3
    assert songs[0]['item_id'] in {song['item_id'] for song in result['playlist']}


def test_compose_does_not_repair_provider_timeout(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    calls = []

    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return '{"anchor_decisions":{},"target_count":3,"target_duration_seconds":null}'
        return 'Error: Request timed out'

    monkeypatch.setattr(api, 'generate_text', generate)
    songs = [{'item_id': f'track-{index}', 'title': f'Track {index}', 'artist': 'Artist'}
             for index in range(10)]
    result, _ = compose_playlist_with_llm('Choose three tracks', songs, {'provider': 'OLLAMA'})
    assert result['error']['category'] == 'TIMEOUT'
    assert len(calls) == 2


def _rerank_aliases(prompt):
    records = json.loads(prompt.rsplit("Candidates: ", 1)[1])
    return [record["id"] for record in records]


def test_llm_selection_rejects_unknown_and_duplicate_ids():
    raw = json.dumps({"selected_ids": ["a", "invented", "a", "b"]})
    assert validate_llm_candidate_selection(raw, ["a", "b"], "LLM_CURATE") == ["a", "b"]


def test_multi_seed_shortlist_balances_all_neighborhoods_before_global_fill():
    songs = []
    provenance = {}
    for seed_index in range(4):
        label = f"seed-{seed_index}"
        ids = []
        for rank in range(200):
            item_id = f"{label}-track-{rank}"
            ids.append(item_id)
            songs.append({"item_id": item_id, "title": item_id, "artist": label})
        provenance[label] = ids

    shortlist, _aliases, _reverse, _payload = prepare_llm_candidate_shortlist(
        songs, "multi-seed request", limit=50, seed_provenance=provenance,
    )
    selected = {song["item_id"] for song in shortlist}
    assert len(shortlist) == 50
    assert all(any(item_id in selected for item_id in ids) for ids in provenance.values())
    assert len(selected) == 50


def test_bad_response_is_empty_and_falls_back_to_native(monkeypatch):
    import tasks.ai.api as api
    monkeypatch.setattr(api, "generate_text", lambda *a, **k: "not json")
    ids, sent = curate_candidates_with_llm(
        "request", [{"item_id": "a"}, {"item_id": "b"}], "LLM_CURATE", {}, include_audio=False
    )
    assert ids == []
    assert sent == 2


def test_curator_distinguishes_provider_error_from_valid_empty_alias_response(monkeypatch):
    import tasks.ai.api as api
    songs = [{"item_id": "a"}]
    logs = []
    monkeypatch.setattr(api, "generate_text", lambda *a, **k: "Error: AI service is currently unavailable.")
    ids, _ = curate_candidates_with_llm(
        "request", songs, "LLM_RERANK", {"provider": "OLLAMA", "ollama_model": "qwen3.5:9b"},
        include_audio=False, log_messages=logs,
    )
    assert ids == []
    assert any(line.startswith("Curator status: PROVIDER_ERROR") for line in logs)
    assert not any("JSON parsing: failure" in line for line in logs)

    logs.clear()
    monkeypatch.setattr(api, "generate_text", lambda *a, **k: '{"ranked_ids":[]}')
    ids, _ = curate_candidates_with_llm(
        "request", songs, "LLM_RERANK", {"provider": "OLLAMA", "ollama_model": "qwen3.5:9b"},
        include_audio=False, log_messages=logs,
    )
    assert ids == []
    assert "Curator JSON parsing: success" in logs
    assert "Curator validation: response contained no exact candidate IDs" in logs


def test_curator_provider_failure_is_nonfatal(monkeypatch):
    import tasks.ai.api as api
    def fail(*args, **kwargs):
        raise TimeoutError("timeout")
    monkeypatch.setattr(api, "generate_text", fail)
    ids, _ = curate_candidates_with_llm(
        "request", [{"item_id": "a"}, {"item_id": "b"}], "LLM_RERANK", {}, include_audio=False
    )
    assert ids == []


def test_candidate_payload_contains_only_compact_catalog_metadata():
    payload = build_llm_candidate_payload(
        [{"item_id": "a", "title": "Song", "artist": "Artist", "album": "Record",
          "duration": 180, "genre": "Disco", "year": 2024,
          "musicnn_similarity": 0.91, "dclap_similarity": 0.87}],
        {"a": {"tempo": 120, "energy": None, "mood_vector": "symphonic metal:0.8",
               "other_features": "uplifting:0.9,calm:0.7"}},
    )
    assert payload == [{
        "id": "a", "title": "Song", "artist": "Artist", "album": "Record",
    }]


@pytest.mark.parametrize("target, expected", [(5, 2), (10, 4), (25, 5)])
def test_llm_artist_cap_scales_with_target_and_respects_absolute_ceiling(target, expected):
    assert effective_llm_artist_cap(target, 5, 0.40) == expected


def test_duration_optimizer_exact_and_closest_results_respect_count_and_artist_cap():
    songs = [
        {"item_id": "a", "artist": "X"},
        {"item_id": "b", "artist": "X"},
        {"item_id": "c", "artist": "Y"},
    ]
    exact = optimize_playlist_duration(songs, {"a": 180, "b": 200, "c": 120}, 300, 2, 1)
    assert {s["item_id"] for s in exact} == {"a", "c"}
    closest = optimize_playlist_duration(songs, {"a": 180, "b": 200, "c": 120}, 299, 2, 1)
    assert sum({"a": 180, "b": 200, "c": 120}[s["item_id"]] for s in closest) == 300


@pytest.mark.parametrize(
    "raw",
    [
        '{"ranked_ids":["opaque-a","opaque-b"]}',
        '```json\n { "ranked_ids" : [ "opaque-a", "opaque-b" ] } \n```',
        'The result is below:\n```json\n{"ranked_ids":["opaque-a","opaque-b"]}\n```\nDone.',
    ],
)
def test_curator_accepts_plain_fenced_and_surrounded_json(raw):
    assert validate_llm_candidate_selection(raw, ["opaque-a", "opaque-b"], "LLM_RERANK") == [
        "opaque-a", "opaque-b"
    ]


def test_curator_diagnostics_identify_titles_unknown_ids_wrong_key_and_duplicates():
    ids, diag = inspect_llm_candidate_selection(
        '{"selected_ids":["Nightwish Song","invented","opaque-a","opaque-a"]}',
        ["opaque-a", "opaque-b"], "LLM_CURATE",
        candidate_titles={"opaque-a": "Nightwish Song"},
    )
    assert ids == ["opaque-a"]
    assert diag["invalid_ids"] == [
        {"value": "Nightwish Song", "reason": "title returned instead of ID"},
        {"value": "invented", "reason": "ID was not present in the candidate map"},
    ]
    assert diag["duplicates_removed"] == ["opaque-a"]
    _, wrong_key = inspect_llm_candidate_selection(
        '{"songs":["opaque-a"]}', ["opaque-a"], "LLM_RERANK"
    )
    assert "ranked_ids" in wrong_key["rejection_reason"]


def test_malformed_json_is_rejected_with_reason():
    ids, diag = inspect_llm_candidate_selection(
        '{"ranked_ids":[', ["opaque-a"], "LLM_RERANK"
    )
    assert ids == []
    assert diag["rejection_reason"].startswith("invalid JSON:")


def test_valid_curator_ranking_changes_native_order(monkeypatch):
    import tasks.ai.api as api

    songs = [
        {"item_id": "opaque-z", "title": "Z song", "artist": "Z"},
        {"item_id": "opaque-b", "title": "B song", "artist": "B"},
        {"item_id": "opaque-a", "title": "A song", "artist": "A"},
    ]
    captured = {}

    def fake_generate(prompt, config, **kwargs):
        captured["prompt"] = prompt
        captured["format"] = kwargs.get("structured_format")
        captured["think"] = kwargs.get("think")
        records = json.loads(prompt.rsplit("Candidates: ", 1)[1])
        by_title = {row.get("title"): row["id"] for row in records}
        return json.dumps({"ranked_ids": [by_title["A song"], by_title["B song"], by_title["Z song"]]})

    monkeypatch.setattr(api, "generate_text", fake_generate)
    ids, sent = curate_candidates_with_llm(
        "similar to a seed", songs, "LLM_RERANK", {"provider": "OLLAMA"},
        include_audio=False,
    )
    ranked = rank_candidates_by_ids(songs, ids)
    assert sent == 3
    assert [s["item_id"] for s in ranked] == ["opaque-a", "opaque-b", "opaque-z"]
    assert [s["item_id"] for s in ranked] != [s["item_id"] for s in songs]
    assert "Return only candidate IDs" in captured["prompt"]
    assert captured["format"]["required"] == ["ranked_ids"]
    assert captured["format"]["additionalProperties"] is False
    assert captured["think"] is False
    assert captured["format"]["properties"]["ranked_ids"]["minItems"] == 3
    assert "rank them from most appropriate to least appropriate" in captured["prompt"].lower()
    assert "native rank" not in captured["prompt"].lower()


def test_target_aware_rerank_retries_then_accepts_target_plus_margin(monkeypatch):
    import tasks.ai.api as api

    songs = [{'item_id': f'track-{i:03d}', 'title': f'Song {i:03d}'} for i in range(50)]
    response_counts = [10, 30]
    prompts, schemas = [], []

    def fake_generate(prompt, _config, **kwargs):
        prompts.append(prompt)
        schemas.append(kwargs['structured_format'])
        aliases = _rerank_aliases(prompt)
        count = response_counts.pop(0)
        return json.dumps({'ranked_ids': aliases[-count:][::-1]})

    monkeypatch.setattr(api, 'generate_text', fake_generate)
    logs = []
    ids, sent = curate_candidates_with_llm(
        'request', songs, 'LLM_RERANK', {'provider': 'OLLAMA'},
        limit=50, include_audio=False, log_messages=logs, target_count=25,
    )

    assert sent == 50
    assert len(ids) == 30
    assert prompts[1].startswith('Your previous ranked_ids response did not contain enough')
    assert schemas[0]['properties']['ranked_ids']['minItems'] == 30
    assert 'Rerank coverage: 20%' in logs
    assert 'Rerank status: INCOMPLETE' in logs
    assert 'Rerank coverage: 60%' in logs
    assert 'Effective target: 25' in logs
    assert 'Rerank required usable aliases: 30' in logs
    assert 'Valid reranked aliases: 30' in logs
    assert 'Rerank status: SUCCESS' in logs


def test_target_aware_rerank_accepts_30_of_50_without_retry(monkeypatch):
    import tasks.ai.api as api
    calls = []
    monkeypatch.setattr(
        api, 'generate_text',
        lambda prompt, *args, **kwargs: calls.append(1)
        or json.dumps({'ranked_ids': _rerank_aliases(prompt)[-30:][::-1]}),
    )
    logs = []
    ids, _ = curate_candidates_with_llm(
        'request', [{'item_id': f'track-{i:03d}'} for i in range(50)],
        'LLM_RERANK', {}, limit=50, include_audio=False,
        log_messages=logs, target_count=25,
    )
    assert len(ids) == 30
    assert len(calls) == 1
    assert 'Rerank coverage: 60%' in logs
    assert 'Rerank status: SUCCESS' in logs


def test_curator_prompt_includes_compact_resolved_seed_metadata(monkeypatch):
    import tasks.ai.api as api
    captured = {}
    def fake_generate(prompt, *_args, **_kwargs):
        captured['prompt'] = prompt
        return json.dumps({'selected_ids': _rerank_aliases(prompt)[:1]})
    monkeypatch.setattr(api, 'generate_text', fake_generate)
    ids, _ = curate_candidates_with_llm(
        'similar to this seed track', [{'item_id': 'seed-id', 'title': 'Harvest', 'artist': 'Nightwish'}],
        'LLM_CURATE', {'provider': 'OLLAMA'}, include_audio=True, target_count=5,
        resolved_seed={'item_id': 'seed-id', 'title': 'Harvest', 'artist': 'Nightwish'},
    )
    assert ids == ['seed-id']
    assert 'Authoritative planner interpretation' in captured['prompt']
    assert '"title":"Harvest"' in captured['prompt']
    assert '"artist":"Nightwish"' in captured['prompt']
    assert 'audio profile' not in captured['prompt']


def test_incomplete_rerank_after_retry_is_rejected_for_native_fallback(monkeypatch):
    import tasks.ai.api as api

    songs = [{'item_id': f'track-{i:03d}'} for i in range(50)]
    calls = []
    def fake_generate(*_args, **_kwargs):
        calls.append(1)
        aliases = _rerank_aliases(_args[0])
        count = 10 if len(calls) == 1 else 20
        return json.dumps({'ranked_ids': aliases[:count]})
    monkeypatch.setattr(api, 'generate_text', fake_generate)
    logs = []
    ids, _ = curate_candidates_with_llm(
        'request', songs, 'LLM_RERANK', {}, limit=50,
        include_audio=False, log_messages=logs, target_count=25,
    )
    assert ids == []
    assert len(calls) == 2
    assert 'Rerank status: INCOMPLETE' in logs
    assert 'Valid reranked aliases: 20' in logs
    assert 'Rerank required usable aliases: 30' in logs
    assert 'Curator status: INCOMPLETE_RERANK' in logs
    assert 'LLM rerank unusable: incomplete after retry; falling back to Native' in logs


def test_curate_uses_opaque_aliases_and_maps_back_to_authoritative_tracks(monkeypatch):
    import tasks.ai.api as api

    songs = [
        {"item_id": "fp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "title": "A", "artist": "AA"},
        {"item_id": "fp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", "title": "B", "artist": "BB"},
    ]
    captured = {}

    def fake_generate(prompt, config, **kwargs):
        captured["prompt"] = prompt
        return json.dumps({"selected_ids": _rerank_aliases(prompt)[::-1]})

    monkeypatch.setattr(api, "generate_text", fake_generate)
    ids, sent = curate_candidates_with_llm("request", songs, "LLM_CURATE", {}, include_audio=False)
    expected_pool = prepare_llm_candidate_shortlist(songs, _rerank_context_key("request", None), limit=2)[0]
    assert sent == 2
    assert ids == [song["item_id"] for song in expected_pool[::-1]]
    records = json.loads(captured["prompt"].rsplit("Candidates: ", 1)[1])
    aliases = [record["id"] for record in records]
    assert len(aliases) == 2 and all(len(alias) >= 5 for alias in aliases)
    assert not any(alias.startswith("C00") for alias in aliases)
    assert "fp_aaaaaaaa" not in captured["prompt"]
    assert "fp_bbbbbbbb" not in captured["prompt"]


def test_rerank_payload_is_order_independent_opaque_and_excludes_audiomuse_signals(monkeypatch):
    import tasks.ai.api as api

    songs = [
        {
            "item_id": f"track-{i}", "title": title, "artist": artist,
            "album": f"Album {i}", "duration_seconds": 200 + i,
            "native_rank": i, "musicnn_similarity": 0.99 - i / 100,
            "dclap_distance": 0.01 + i / 100, "similarity": 0.8,
        }
        for i, (title, artist) in enumerate([
            ("Harvest", "Nightwish"), ("Angels", "Within Temptation"),
            ("Inis Mona", "Eluveitie"), ("Path", "Apocalyptica"),
            ("Justin Song", "Justin Bieber"), ("Black", "Metal Artist"),
        ])
    ]
    prompts = []

    def fake_generate(prompt, *_args, **_kwargs):
        prompts.append(prompt)
        aliases = _rerank_aliases(prompt)
        return json.dumps({"ranked_ids": aliases})

    monkeypatch.setattr(api, "generate_text", fake_generate)
    logs = []
    seed = {"item_id": "seed", "title": "Harvest", "artist": "Nightwish", "album": "Human Nature"}
    first_ids, _ = curate_candidates_with_llm(
        "Use Harvest by Nightwish as a seed", songs, "LLM_RERANK", {},
        limit=6, include_audio=True, log_messages=logs, target_count=1, resolved_seed=seed,
    )
    second_ids, _ = curate_candidates_with_llm(
        "Use Harvest by Nightwish as a seed", list(reversed(songs)), "LLM_RERANK", {},
        limit=6, include_audio=False, target_count=1, resolved_seed=seed,
    )

    first_records = json.loads(prompts[0].rsplit("Candidates: ", 1)[1])
    second_records = json.loads(prompts[1].rsplit("Candidates: ", 1)[1])
    assert first_records == second_records
    assert first_ids == second_ids
    assert [row["title"] for row in first_records] != [song["title"] for song in songs]
    assert {key for row in first_records for key in row} <= {"id", "title", "artist", "album"}
    assert not any(
        token in prompts[0].casefold()
        for token in ("native_rank", "musicnn", "dclap", "similarity", "duration_seconds", "score")
    )
    assert "Candidate presentation order has no significance" in prompts[0]
    assert "Candidate presentation shuffled: yes" in logs
    assert "AudioMuse scores sent to LLM: no" in logs
    assert "AudioMuse native ranks sent to LLM: no" in logs
    assert all(len(row["id"]) >= 5 and not row["id"].startswith("C00") for row in first_records)


def test_curator_alias_validation_rejects_unknown_and_deduplicates(monkeypatch):
    import tasks.ai.api as api
    songs = [{"item_id": "fp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
             {"item_id": "fp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
    monkeypatch.setattr(
        api, "generate_text",
        lambda prompt, *a, **k: json.dumps({
            "ranked_ids": [_rerank_aliases(prompt)[1], "ZZZZZ", _rerank_aliases(prompt)[1], _rerank_aliases(prompt)[0]],
        }),
    )
    ids, _ = curate_candidates_with_llm("request", songs, "LLM_RERANK", {}, include_audio=False)
    assert ids == [songs[1]["item_id"], songs[0]["item_id"]]


@pytest.mark.parametrize(
    "raw",
    [
        '{"selected_ids":["C003","C001"]}',
        '["C003","C001"]',
        '{"id":"C003"}',
        '{"ids":["C003","C001"]}',
        '{"ranked_ids":["C003"],"summary":"analysis"}',
    ],
)
def test_curator_rejects_responses_outside_the_exact_ranked_schema(raw):
    aliases, diagnostics = inspect_llm_candidate_selection(
        raw, ["C001", "C002", "C003"], "LLM_RERANK",
    )
    assert aliases == []
    assert diagnostics["normalized_alias_count"] == 0


def test_curator_normalization_rejects_unknown_alias_but_keeps_valid_alias():
    aliases, diagnostics = inspect_llm_candidate_selection(
        '{"ranked_ids":["C999","C001"]}', ["C001", "C002"], "LLM_RERANK",
    )
    assert aliases == ["C001"]
    assert diagnostics["invalid_ids"] == [
        {"value": "C999", "reason": "ID was not present in the candidate map"}
    ]


def test_curator_rejects_analysis_object_without_exact_ranked_ids(monkeypatch):
    import tasks.ai.api as api
    songs = [
        {"item_id": "track-a", "title": "Song A"},
        {"item_id": "track-b", "title": "Song B"},
        {"item_id": "track-c", "title": "Song C"},
    ]
    logs = []
    monkeypatch.setattr(api, "generate_text", lambda *a, **k: '{"summary":"analysis","high_energy_tracks":["C003"]}')
    ids, _ = curate_candidates_with_llm(
        "request", songs, "LLM_RERANK", {}, include_audio=False, log_messages=logs,
    )
    assert ids == []
    assert any("Curator validation:" in line and "ranked_ids" in line for line in logs)
    assert "Valid aliases: 0" in logs


def test_partial_rerank_supplements_remaining_candidates_in_native_order():
    songs = [{"item_id": f"track-{alias}", "alias": alias} for alias in ("C001", "C002", "C003", "C004")]
    ranked = rank_candidates_by_ids(
        songs, ["track-C003"], include_unselected=True,
    )
    assert [song["item_id"] for song in ranked] == [
        "track-C003", "track-C001", "track-C002", "track-C004",
    ]


def test_llm_curate_uses_aliases_and_maps_to_real_ids(monkeypatch):
    import tasks.ai.api as api
    songs = [{"item_id": "fp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
             {"item_id": "fp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}]
    monkeypatch.setattr(
        api, "generate_text",
        lambda prompt, *a, **k: json.dumps({"selected_ids": [_rerank_aliases(prompt)[1]]}),
    )
    ids, _ = curate_candidates_with_llm("request", songs, "LLM_CURATE", {}, include_audio=False)
    expected_pool = prepare_llm_candidate_shortlist(songs, _rerank_context_key("request", None), limit=2)[0]
    assert ids == [expected_pool[1]["item_id"]]


def test_duration_optimizer_uses_real_durations_and_keeps_mandatory_seed():
    from tasks.ai.planner import requested_playlist_shape

    request = "30 minute playlist"
    shape = requested_playlist_shape(request)
    assert shape["total_seconds"] == 1800
    songs = [
        {"item_id": "seed-opaque", "artist": "Nightwish", "title": "Dark Chest Of Wonders"},
        {"item_id": "id-240", "artist": "A"},
        {"item_id": "id-220", "artist": "B"},
        {"item_id": "id-300", "artist": "C"},
        {"item_id": "id-190", "artist": "D"},
        {"item_id": "id-360", "artist": "E"},
        {"item_id": "id-218", "artist": "F"},
    ]
    durations = {
        "seed-opaque": 269, "id-240": 240, "id-220": 220, "id-300": 300,
        "id-190": 190, "id-360": 360, "id-218": 218,
    }
    result = optimize_playlist_duration(
        songs, durations, shape["total_seconds"], 100, 1,
        mandatory_ids=["seed-opaque"],
    )
    assert result[0]["item_id"] == "seed-opaque"
    assert sum(durations[s["item_id"]] for s in result) == 1797


def test_rerank_and_curate_share_opaque_shortlist_payload_and_provider_settings(monkeypatch):
    import tasks.ai.api as api

    songs = [
        {
            "item_id": f"track-{i:03d}", "title": f"Song {i:03d}",
            "artist": f"Artist {i % 7}", "album": f"Album {i % 4}",
            "genre": "Indie pop", "year": 2000 + (i % 20), "duration": 180,
            "native_rank": i, "musicnn_similarity": 0.98, "dclap_distance": 0.04,
        }
        for i in range(1, 51)
    ]
    captured, logs = [], []

    def fake_generate(prompt, config, **kwargs):
        aliases = _rerank_aliases(prompt)
        captured.append({
            "prompt": prompt, "schema": kwargs.get("structured_format"),
            "max_tokens": kwargs.get("max_tokens"), "think": kwargs.get("think"),
        })
        if 'ranked_ids' in prompt:
            return json.dumps({"ranked_ids": aliases})
        return json.dumps({"selected_ids": aliases[:17]})

    monkeypatch.setattr(api, "generate_text", fake_generate)
    rerank_ids, rerank_sent = curate_candidates_with_llm(
        "request", songs, "LLM_RERANK", {"provider": "OLLAMA"}, limit=50,
        include_audio=True, target_count=25, log_messages=[],
    )
    curated_ids, curate_sent = curate_candidates_with_llm(
        "request", songs, "LLM_CURATE", {"provider": "OLLAMA"}, limit=50,
        include_audio=True, target_count=25, log_messages=logs,
    )
    rerank_records, curate_records = [
        json.loads(call["prompt"].rsplit("Candidates: ", 1)[1]) for call in captured
    ]

    assert rerank_sent == curate_sent == 50
    assert len(rerank_ids) == 50 and len(curated_ids) == 17
    assert rerank_records == curate_records
    assert all(len(row["id"]) >= 5 and not row["id"].startswith("C00") for row in curate_records)
    assert {key for row in curate_records for key in row} <= {"id", "title", "artist", "album"}
    assert "native_rank" not in captured[1]["prompt"]
    assert "musicnn" not in captured[1]["prompt"] and "dclap" not in captured[1]["prompt"]
    assert len(captured[1]["prompt"].rsplit("Candidates: ", 1)[1]) < 7000
    assert captured[1]["schema"] == {
        "type": "object", "additionalProperties": False,
        "properties": {"selected_ids": {"type": "array", "items": {"type": "string"}}},
        "required": ["selected_ids"],
    }
    assert captured[0]["max_tokens"] == captured[1]["max_tokens"] == 4096
    assert captured[0]["think"] == captured[1]["think"] is False
    assert "Candidate presentation shuffled: yes" in logs
    assert "AudioMuse scores sent to LLM: no" in logs
    assert "AudioMuse native ranks sent to LLM: no" in logs
    assert "Effective LLM output token budget: 4096" in logs
    assert any(line.startswith("Serialized candidate payload chars:") for line in logs)


def test_shared_shortlist_builds_same_aliases_and_authoritative_mapping():
    from tasks.playlist_curation import prepare_llm_candidate_shortlist

    songs = [
        {"item_id": f"track-{i:03d}", "title": f"Song {i:03d}", "artist": "Artist", "album": "Record"}
        for i in range(50)
    ]
    rerank = prepare_llm_candidate_shortlist(songs, "same-request", limit=50)
    curate = prepare_llm_candidate_shortlist(songs, "same-request", limit=50)

    assert rerank[0] == curate[0]
    assert rerank[1] == curate[1]
    assert rerank[2] == curate[2]
    assert rerank[3] == curate[3]
    assert len(rerank[3]) == 50
    assert all(set(row) <= {"id", "title", "artist", "album"} for row in rerank[3])


def test_curate_caps_valid_aliases_to_target_without_rejecting_partial_results(monkeypatch):
    import tasks.ai.api as api

    songs = [{"item_id": f"track-{i}", "title": f"Song {i}", "artist": "Artist"} for i in range(12)]
    monkeypatch.setattr(
        api, "generate_text",
        lambda prompt, *args, **kwargs: json.dumps({"selected_ids": _rerank_aliases(prompt)}),
    )
    logs = []
    ids, sent = curate_candidates_with_llm(
        "request", songs, "LLM_CURATE", {"provider": "OLLAMA"}, limit=12,
        target_count=5, log_messages=logs,
    )

    assert sent == 12
    assert len(ids) == 5
    assert "Curator status: SUCCESS" in logs
    assert "Curator selection capped to effective target: 5" in logs


def test_curator_rejects_analysis_report_even_when_it_contains_candidate_ids():
    aliases, diag = inspect_llm_candidate_selection(
        '{"summary":"tracks C001-C040 are energetic","high_energy_tracks":["C001"]}',
        [f"C{i:03d}" for i in range(1, 41)], "LLM_CURATE",
    )
    assert aliases == []
    assert diag["json_top_level_type"] == "dict"
    assert diag["normalization_input_type"] == "dict"
    assert diag["normalization_output_aliases"] == []


def test_playlist_suppresses_normalized_title_artist_duplicates():
    from tasks.playlist_curation import suppress_duplicate_title_artist

    songs = [
        {"item_id": "a", "title": "(01) [Disturbed] Remnants", "artist": "Disturbed"},
        {"item_id": "b", "title": "remnants", "artist": "DISTURBED"},
        {"item_id": "c", "title": "Remnants", "artist": "Another artist"},
        {"item_id": "d", "title": "Untitled", "artist": ""},
    ]
    kept, removed = suppress_duplicate_title_artist(songs)
    assert [song["item_id"] for song in kept] == ["a", "c", "d"]
    assert removed == 1


def test_song_family_suppression_collapses_remixes_and_promotes_next_ranked_track():
    from tasks.playlist_curation import song_family_key, suppress_song_families

    songs = [
        {'item_id': 'original', 'title': 'Take Me Home (A Girl Like Me)', 'artist': 'Sophie Ellis-Bextor'},
        {'item_id': 'mix', 'title': 'Take Me Home (A Girl Like Me) (Jewels & Stone Mix)', 'artist': 'Sophie Ellis-Bextor'},
        {'item_id': 'radio', 'title': 'Take Me Home (A Girl Like Me) (Radio Edit Remix By DJ Flex)', 'artist': 'Sophie Ellis-Bextor'},
        {'item_id': 'next', 'title': 'Murder On The Dancefloor', 'artist': 'Sophie Ellis-Bextor'},
    ]
    assert song_family_key(songs[0]) == song_family_key(songs[1]) == song_family_key(songs[2])
    kept, removed = suppress_song_families(songs)
    assert [song['item_id'] for song in kept] == ['original', 'next']
    assert removed == 2


def test_mandatory_song_family_variant_wins_and_feature_title_is_preserved():
    from tasks.playlist_curation import song_family_key, suppress_song_families

    songs = [
        {'item_id': 'original', 'title': 'Take Me Home (A Girl Like Me)', 'artist': 'Sophie Ellis-Bextor'},
        {'item_id': 'mandatory-remix', 'title': 'Take Me Home (A Girl Like Me) (Jewels & Stone Mix)', 'artist': 'Sophie Ellis-Bextor'},
        {'item_id': 'dopamine-a', 'title': 'Dopamine (Feat. Eyelar)', 'artist': 'Purple Disco Machine'},
        {'item_id': 'dopamine-b', 'title': 'Dopamine', 'artist': 'Purple Disco Machine'},
    ]
    kept, removed = suppress_song_families(songs, mandatory_ids=['mandatory-remix'])
    assert [song['item_id'] for song in kept] == ['mandatory-remix', 'dopamine-a', 'dopamine-b']
    assert song_family_key(songs[2]) != song_family_key(songs[3])
    assert removed == 1


def test_every_mandatory_song_survives_shared_version_family_suppression():
    from tasks.playlist_curation import suppress_song_families

    songs = [
        {'item_id': 'mandatory-original', 'title': 'Take Me Home', 'artist': 'Artist'},
        {'item_id': 'mandatory-live', 'title': 'Take Me Home (Live)', 'artist': 'Artist'},
        {'item_id': 'optional-remix', 'title': 'Take Me Home (Radio Remix)', 'artist': 'Artist'},
    ]
    kept, removed = suppress_song_families(
        songs, mandatory_ids=['mandatory-original', 'mandatory-live'],
    )
    assert {song['item_id'] for song in kept} == {'mandatory-original', 'mandatory-live'}
    assert removed == 1


def test_intentionally_requested_versions_are_spaced_when_enough_tracks_exist():
    from tasks.playlist_curation import suppress_song_families

    songs = [
        {'item_id': 'take-1', 'title': 'Take Me Home', 'artist': 'Artist A'},
        *[{'item_id': f'other-{i}', 'title': f'Other {i}', 'artist': f'Artist {i}'} for i in range(6)],
        {'item_id': 'take-2', 'title': 'Take Me Home (Radio Edit)', 'artist': 'Artist A'},
    ]
    kept, removed = suppress_song_families(songs, allow_multiple=True, min_spacing=5)
    assert removed == 0
    positions = [i for i, song in enumerate(kept) if song['item_id'].startswith('take-')]
    assert positions[1] - positions[0] >= 6


def test_curator_candidate_window_probe_checks_first_last_and_count(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import probe_curator_candidate_window

    captured = {}

    def fake_generate(prompt, config, **kwargs):
        captured.update(prompt=prompt, schema=kwargs.get("structured_format"))
        return '{"first_id":"C001","last_id":"C040","candidate_count":40}'

    monkeypatch.setattr(api, "generate_text", fake_generate)
    result = probe_curator_candidate_window({"provider": "OLLAMA"}, count=40)
    assert result["status"] == "SUCCESS"
    assert result["candidate_count_sent"] == 40
    assert '"id":"C001"' in captured["prompt"]
    assert '"id":"C040"' in captured["prompt"]
    assert captured["schema"]["additionalProperties"] is False


def test_duration_optimizer_prefers_rank_over_exact_duration_inside_tolerance():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [
        {"item_id": f"high-{i}", "artist": f"High{i}"}
        for i in range(5)
    ] + [
        {"item_id": f"low-{i}", "artist": f"Low{i}"}
        for i in range(5)
    ]
    durations = dict(zip(
        [s["item_id"] for s in songs],
        [350, 360, 360, 360, 363, 350, 360, 360, 360, 370],
    ))
    diagnostics = {}
    result = optimize_playlist_duration(
        songs, durations, 1800, count=5, max_per_artist=1,
        exact_count=True, tolerance_seconds=15, diagnostics=diagnostics,
    )
    assert sum(durations[s["item_id"]] for s in result) == 1793
    assert diagnostics["ranking_cost"] == 15
    assert diagnostics["best_exact_ranking_cost"] > diagnostics["ranking_cost"]


def test_duration_optimizer_uses_exact_duration_when_rank_cost_ties():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [
        {"item_id": key, "artist": key}
        for key in ("rank-1", "rank-2", "rank-3", "rank-4")
    ]
    durations = {"rank-1": 1000, "rank-2": 900, "rank-3": 900, "rank-4": 795}
    result = optimize_playlist_duration(
        songs, durations, 1800, count=2, max_per_artist=1,
        exact_count=True, tolerance_seconds=15,
    )
    assert sum(durations[s["item_id"]] for s in result) == 1800
    assert {s["item_id"] for s in result} == {"rank-2", "rank-3"}


def test_duration_optimizer_uses_closest_duration_when_none_fits_tolerance():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [
        {"item_id": key, "artist": key}
        for key in ("high-a", "high-b", "lower-a", "lower-b")
    ]
    durations = {"high-a": 1000, "high-b": 750, "lower-a": 900, "lower-b": 870}
    diagnostics = {}
    result = optimize_playlist_duration(
        songs, durations, 1800, count=2, max_per_artist=1,
        exact_count=True, tolerance_seconds=15, diagnostics=diagnostics,
    )
    assert sum(durations[s["item_id"]] for s in result) == 1770
    assert diagnostics["solutions_in_tolerance"] == 0


def test_duration_optimizer_always_keeps_mandatory_seed():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [
        {"item_id": "seed", "artist": "Seed Artist"},
        {"item_id": "a", "artist": "Artist A"},
        {"item_id": "b", "artist": "Artist B"},
    ]
    result = optimize_playlist_duration(
        songs, {"seed": 300, "a": 750, "b": 750}, 1800,
        count=3, max_per_artist=1, mandatory_ids=["seed"],
    )
    assert [song["item_id"] for song in result] == ["seed", "a", "b"]


def test_duration_optimizer_uses_llm_curate_order_for_ranks_and_output():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [{"item_id": key, "artist": key} for key in ("C001", "C002", "C005", "C010")]
    diagnostics = {}
    result = optimize_playlist_duration(
        songs, {song["item_id"]: 600 for song in songs}, 1200,
        count=2, max_per_artist=1, exact_count=True,
        ranked_ids=["C005", "C002", "C010", "C001"],
        tolerance_seconds=0, diagnostics=diagnostics,
    )
    assert [song["item_id"] for song in result] == ["C005", "C002"]
    assert diagnostics["selected_ranks"] == [1, 2]


def test_duration_optimizer_uses_native_order_when_no_curator_order_is_supplied():
    from tasks.playlist_curation import optimize_playlist_duration

    songs = [{"item_id": key, "artist": key} for key in ("native-a", "native-b", "native-c")]
    result = optimize_playlist_duration(
        songs, {song["item_id"]: 600 for song in songs}, 1200,
        count=2, max_per_artist=1, exact_count=True, tolerance_seconds=0,
    )
    assert [song["item_id"] for song in result] == ["native-a", "native-b"]


def test_mandatory_seed_survives_curation_and_ordering_membership_is_unchanged():
    songs = [
        {"item_id": "seed-opaque", "title": "Dark Chest Of Wonders"},
        {"item_id": "id-a"},
        {"item_id": "id-b"},
    ]
    curated = rank_candidates_by_ids(
        songs, ["id-b"], mandatory_ids=["seed-opaque"], include_unselected=False
    )
    assert [song["item_id"] for song in curated] == ["seed-opaque", "id-b"]
    before = {song["item_id"] for song in curated}
    after = {song["item_id"] for song in reorder_preserving_membership(curated, ["id-b"])}
    assert before == after



def test_title_only_seed_dispatch_resolves_track_before_similarity_search(monkeypatch):
    import tasks.ai.tools as tools

    monkeypatch.setattr(
        tools, "resolve_song_by_title",
        lambda title: {"item_id": "seed-opaque", "title": "Dark Chest Of Wonders", "author": "Nightwish"},
    )
    called = {}

    def similarity(title, artist, count):
        called.update(title=title, artist=artist, count=count)
        return {"songs": [{"item_id": "neighbor-opaque"}], "message": "similarity results"}

    monkeypatch.setattr(tools, "_song_similarity_api_sync", similarity)
    result = tools._dispatch_seed_search(
        {"seeds": [{"type": "song", "title": "Dark chest of wonders"}], "get_songs": 20},
        {},
    )
    assert called["title"] == "Dark Chest Of Wonders"
    assert called["artist"] == "Nightwish"
    assert result["songs"][0]["item_id"] == "neighbor-opaque"
    assert "Seed ID: seed-opaque" in result["message"]


def test_seed_search_to_curated_duration_pipeline_keeps_named_seed(monkeypatch):
    from tasks.ai import api, planner, tool_impl, tools

    request = (
        'I would like you to build me a 30 minutes playlist starting from Dark chest of wonders songs. '
        'The playlist should contain similar songs to this one.'
    )
    duration = planner.requested_playlist_shape(request)['total_seconds']
    assert duration == 1800

    resolved = {
        'item_id': 'seed-opaque', 'title': 'Dark Chest Of Wonders',
        'author': 'Nightwish', 'album': 'Once',
    }
    monkeypatch.setattr(tool_impl, 'resolve_song_by_title', lambda title, artist_hint='': resolved)
    planner_calls = [{
        'name': 'seed_search',
        'arguments': {'seeds': [{'type': 'song', 'title': 'Dark Chest of Wonders'}]},
    }]
    normalized_calls = planner.validate_plan_args(
        planner_calls, user_wants_rating=False, request_text=request,
    )
    assert [call['name'] for call in normalized_calls] == ['seed_search']
    seed = normalized_calls[0]['arguments']['seeds'][0]
    assert seed == {'type': 'song', 'title': 'Dark Chest Of Wonders', 'artist': 'Nightwish'}

    similarity_called = {}
    candidates = [
        {'item_id': 'candidate-a', 'title': 'A', 'artist': 'Artist A'},
        {'item_id': 'candidate-b', 'title': 'B', 'artist': 'Artist B'},
        {'item_id': 'candidate-c', 'title': 'C', 'artist': 'Artist C'},
        {'item_id': 'candidate-d', 'title': 'D', 'artist': 'Artist D'},
        {'item_id': 'candidate-e', 'title': 'E', 'artist': 'Artist E'},
        {'item_id': 'candidate-f', 'title': 'F', 'artist': 'Artist F'},
    ]

    def fake_similarity(title, artist, count):
        similarity_called.update(title=title, artist=artist)
        return {'songs': candidates, 'message': 'mock similarity results'}

    monkeypatch.setattr(tools, '_song_similarity_api_sync', fake_similarity)
    retrieved = tools._dispatch_seed_search(normalized_calls[0]['arguments'], {})
    assert similarity_called == {'title': 'Dark Chest Of Wonders', 'artist': 'Nightwish'}
    assert retrieved['songs']
    candidate_pool = [
        {'item_id': resolved['item_id'], 'title': resolved['title'], 'artist': 'Nightwish'},
        *retrieved['songs'],
    ]

    prompt_seen = {}

    def fake_generate_text(prompt, *args, **kwargs):
        prompt_seen['text'] = prompt
        return json.dumps({'ranked_ids': _rerank_aliases(prompt)})

    monkeypatch.setattr(api, 'generate_text', fake_generate_text)
    ranked_ids, sent = curate_candidates_with_llm(
        request, candidate_pool, 'LLM_RERANK', {}, limit=100,
    )
    assert sent > 0
    records = json.loads(prompt_seen['text'].rsplit('Candidates: ', 1)[1])
    assert len(records) == 7
    assert all(len(record['id']) >= 5 for record in records)
    assert all(song['item_id'] not in prompt_seen['text'] for song in candidate_pool)
    curated = rank_candidates_by_ids(
        candidate_pool, ranked_ids, mandatory_ids=['seed-opaque'], include_unselected=True,
    )
    assert curated[0]['item_id'] == 'seed-opaque'

    durations = {
        'seed-opaque': 269, 'candidate-a': 240, 'candidate-b': 220,
        'candidate-c': 300, 'candidate-d': 190, 'candidate-e': 360,
        'candidate-f': 250,
    }
    final = optimize_playlist_duration(
        curated, durations, duration, count=100, max_per_artist=10,
        mandatory_ids=['seed-opaque'],
    )
    assert 'seed-opaque' in {song['item_id'] for song in final}
    assert len(final) < 10
    assert not any(call['name'] == 'text_match' for call in normalized_calls)


def test_planner_and_curator_share_ollama_chat_adapter_and_curator_validates_aliases(monkeypatch):
    from tasks.ai import api
    from tasks.ai.providers import openai

    calls = []

    def fake_chat(url, model, payload, *, timeout, operation):
        calls.append((operation, url, model, payload, timeout))
        if operation == "planner":
            return {"message": {"tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]}}
        user_prompt = payload["messages"][-1]["content"]
        return {"message": {"content": json.dumps({"ranked_ids": _rerank_aliases(user_prompt)[::-1]})}}

    monkeypatch.setattr(openai, "_ollama_chat_request", fake_chat)
    config = {"provider": "OLLAMA", "ollama_url": "http://ollama:11434", "ollama_model": "qwen3.5:9b"}
    plan = api.call_with_tools(
        "find tracks", [{"name": "lookup", "description": "lookup", "inputSchema": {"type": "object", "properties": {}}}],
        config, log_messages=[],
    )
    assert "tool_calls" in plan

    logs = []
    ids, sent = curate_candidates_with_llm(
        "request", [{"item_id": "track-a"}, {"item_id": "track-b"}],
        "LLM_RERANK", config, include_audio=False, log_messages=logs,
    )
    assert sent == 2
    assert set(ids) == {"track-a", "track-b"}
    assert [call[0] for call in calls] == ["planner", "text"]
    assert calls[0][1] == calls[1][1] == "http://ollama:11434/api/chat"
    assert calls[0][2] == calls[1][2] == "qwen3.5:9b"
    assert "Curator status: SUCCESS" in logs
    assert "Valid aliases: 2" in logs


def test_direct_curator_provider_probe_uses_curator_adapter(monkeypatch):
    from tasks.ai.providers import openai
    from tasks.playlist_curation import probe_curator_provider

    seen = {}

    def fake_chat(url, model, payload, *, timeout, operation):
        seen.update(url=url, model=model, payload=payload, operation=operation)
        return {"message": {"content": '{"ranked_ids":["C002","C001"]}'}}

    monkeypatch.setattr(openai, "_ollama_chat_request", fake_chat)
    result = probe_curator_provider({
        "provider": "OLLAMA", "ollama_url": "http://ollama:11434", "ollama_model": "qwen3.5:9b",
    })
    assert result["status"] == "SUCCESS"
    assert result["parsed"]["ranked_ids"] == ["C002", "C001"]
    assert seen["operation"] == "text"
    assert [m["role"] for m in seen["payload"]["messages"]] == ["system", "user"]
    assert seen["payload"]["format"]["required"] == ["ranked_ids"]
    assert seen["payload"]["format"]["additionalProperties"] is False


def test_ollama_text_response_retries_thinking_modes_without_exposing_reasoning(monkeypatch, caplog):
    from tasks.ai.providers import openai

    requests = []
    envelopes = [
        {
            "model": "gpt-oss:120b-64k", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": "", "thinking": "private reasoning text"},
        },
        {
            "model": "gpt-oss:120b-64k", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": '{"ranked_ids":["A1"]}'},
        },
        {
            "model": "gpt-oss:120b-64k", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": '{"ranked_ids":["A1"]}'},
        },
    ]

    def fake_request(_url, model, payload, **kwargs):
        requests.append((model, dict(payload), kwargs))
        return envelopes.pop(0)

    monkeypatch.setattr(openai, "_ollama_chat_request", fake_request)
    monkeypatch.setattr(openai, "_OLLAMA_TEXT_THINK_SETTINGS", {})
    schema = {
        "type": "object", "properties": {"ranked_ids": {
            "type": "array", "items": {"type": "string"},
        }}, "required": ["ranked_ids"],
    }
    result = openai.generate_text_ollama_chat(
        "http://ollama.local:11434", "gpt-oss:120b-64k", "Rank A1",
        think=False, structured_format=schema,
    )

    assert result == '{"ranked_ids":["A1"]}'
    assert requests[0][1]["think"] is False
    assert requests[1][1]["think"] == "low"
    assert requests[0][1]["stream"] is False
    assert requests[0][1]["format"] == schema
    assert "private reasoning text" not in caplog.text

    second_result = openai.generate_text_ollama_chat(
        "http://ollama.local:11434", "gpt-oss:120b-64k", "Rank A1",
        think=False, structured_format=schema,
    )
    assert second_result == result
    assert len(requests) == 3
    assert requests[2][1]["think"] == "low"


def test_ollama_response_shape_reports_metadata_without_content_or_thinking():
    from tasks.ai.json_response import ollama_response_shape

    shape = ollama_response_shape({
        "model": "gemma4:26b", "done": True, "done_reason": "stop",
        "prompt_eval_count": 22, "eval_count": 41,
        "message": {
            "role": "assistant", "content": "{\"ranked_ids\":[]}",
            "thinking": "private reasoning text", "tool_calls": [],
        },
    })
    assert shape["top_level_keys"] == [
        "done", "done_reason", "eval_count", "message", "model", "prompt_eval_count",
    ]
    assert shape["message_keys"] == ["content", "role", "thinking", "tool_calls"]
    assert shape["content_length"] == len('{"ranked_ids":[]}')
    assert shape["thinking_length"] == len("private reasoning text")
    assert shape["tool_calls"] == 0
    assert "private reasoning text" not in repr(shape)


def test_ollama_request_logging_reports_shape_without_logging_playlist_prompt(monkeypatch, caplog):
    import logging
    from tasks.ai.providers import openai

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "model": "gemma4:26b", "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": '{"ranked_ids":["A1"]}'},
            }

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def post(self, url, json):
            captured.update(url=url, payload=json)
            return FakeResponse()

    monkeypatch.setattr(openai.httpx, "Client", FakeClient)
    prompt = "PRIVATE PLAYLIST PROMPT TEXT"
    with caplog.at_level(logging.INFO):
        result = openai.generate_text_ollama_chat(
            "http://ollama.local:11434", "gemma4:26b", prompt,
            think=False, structured_format={
                "type": "object", "properties": {"ranked_ids": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "uniqueItems": True,
                }}, "required": ["ranked_ids"],
            },
        )

    assert result == '{"ranked_ids":["A1"]}'
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["num_predict"] == 8000
    assert "fields=['format', 'messages', 'model', 'options', 'stream', 'think']" in caplog.text
    assert "think=False" in caplog.text
    assert "minItems" in caplog.text
    assert "PRIVATE PLAYLIST PROMPT TEXT" not in caplog.text


def test_curator_provider_failure_logs_reason_and_returns_empty_for_native_fallback(monkeypatch, caplog):
    import httpx
    from tasks.ai.providers import openai

    def fail(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(openai, "_ollama_chat_request", fail)
    logs = []
    ids, _ = curate_candidates_with_llm(
        "request", [{"item_id": "track-a"}], "LLM_RERANK",
        {"provider": "OLLAMA", "ollama_url": "http://ollama:11434", "ollama_model": "qwen3.5:9b"},
        include_audio=False, log_messages=logs,
    )
    assert ids == []
    assert "ConnectError" in caplog.text
    assert any(line.startswith("Curator status: PROVIDER_ERROR") for line in logs)
    assert "Curator JSON parsing: failure" not in logs
def test_finalize_composer_duration_uses_library_lengths_and_ignores_model_claim():
    from tasks.playlist_curation import finalize_composer_duration

    songs = [
        {'item_id': str(i), 'title': f'Track {i}', 'artist': f'Artist {i}'}
        for i in range(23)
    ]
    durations = {str(i): {'duration': 280 if i < 22 else 412} for i in range(23)}
    # The complete preference list is 6572 seconds. It is a pool, not a
    # duration-compliant final playlist, regardless of Composer prose.
    assert sum(row['duration'] for row in durations.values()) == 6572
    selected, actual, details = finalize_composer_duration(
        songs, durations, 1800, tolerance_seconds=15,
    )
    assert len(selected) < len(songs)
    assert actual == 1812
    assert details['within_tolerance']


def test_finalize_composer_duration_preserves_required_anchor_and_count():
    from tasks.playlist_curation import finalize_composer_duration

    songs = [
        {'item_id': str(i), 'title': f'Track {i}', 'artist': f'Artist {i}'}
        for i in range(4)
    ]
    durations = {'0': {'duration': 600}, '1': {'duration': 300},
                 '2': {'duration': 300}, '3': {'duration': 1000}}
    selected, actual, details = finalize_composer_duration(
        songs, durations, 900, target_count=2, required_ids=['0'],
    )
    assert len(selected) == 2
    assert selected[0]['item_id'] == '0'
    assert actual == 900
    assert details['within_tolerance']


def test_finalize_composer_duration_rejects_6572_second_violation():
    from tasks.playlist_curation import finalize_composer_duration

    songs = [{'item_id': 'a', 'title': 'Long track', 'artist': 'Artist'}]
    selected, actual, details = finalize_composer_duration(
        songs, {'a': {'duration': 6572}}, 1800,
    )
    assert selected == songs
    assert actual == 6572
    assert details['error_seconds'] == 4772
    assert not details['within_tolerance']


def test_compose_repetition_keeps_61_and_fills_only_three(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm
    calls = []
    prefix = []
    expected_track_ids = []
    def generate(prompt, _config, **kwargs):
        calls.append(prompt)
        kwargs['call_metadata']['done_reason'] = 'stop'
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({'anchor_decisions': {}, 'target_count': 64, 'target_duration_seconds': None})
        if prompt.startswith('Select only the missing tracks'):
            assert 'Missing count: 3' in prompt
            remaining = json.loads(prompt.split('Remaining available candidates: ', 1)[1])
            assert not set(prefix) & {r['id'] for r in remaining}
            return json.dumps({'playlist_ids': [r['id'] for r in remaining[:3]]})
        records = json.loads(prompt.split('Available candidates: ', 1)[1])
        prefix.extend(r['id'] for r in records[:61])
        expected_track_ids.extend(r['title'] for r in records[:61])
        kwargs['call_metadata']['done_reason'] = 'COMPOSER_REPETITION'
        return json.dumps({'playlist_ids': prefix + [prefix[-1]] * 4})
    monkeypatch.setattr(api, 'generate_text', generate)
    logs = []
    result, sent = compose_playlist_with_llm('Choose 64 songs',
        [{'item_id': str(i), 'title': str(i), 'artist': 'Artist'} for i in range(242)],
        {'provider': 'OLLAMA'}, log_messages=logs)
    assert len(calls) == 3
    assert sent == 242
    assert len(result['playlist']) == 64
    assert len({s['item_id'] for s in result['playlist']}) == 64
    assert [s['item_id'] for s in result['playlist'][:61]] == expected_track_ids
    assert any('COMPOSER_REPETITION' in line for line in logs)
    assert 'Targeted fill requested: 3' in logs
