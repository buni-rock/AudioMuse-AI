import json

import pytest

from tasks.playlist_curation import (
    build_llm_candidate_payload,
    prepare_llm_candidate_shortlist,
    optimize_playlist_duration,
    rank_candidates_by_ids,
    reorder_preserving_membership,
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


def test_duration_composer_requests_ranked_options_from_library_lengths(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': str(i), 'title': f'Track {i}', 'artist': f'Artist {i % 5}'}
        for i in range(24)
    ]
    prompts = []

    def generate(prompt, _config, **kwargs):
        prompts.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {'A001': True},
                'target_count': None, 'target_duration_seconds': 1800,
            })
        records = json.loads(prompt.split('Available candidates: ', 1)[1])
        return json.dumps({'playlist_ids': [row['id'] for row in records[:18]]})

    monkeypatch.setattr(api, 'generate_text', generate)
    result, sent = compose_playlist_with_llm(
        'Use Track 0 as a seed for exactly 30 minutes', songs,
        {'provider': 'OLLAMA'},
        resolved_anchors=[{'type': 'song', 'resolved_track': songs[0]}],
        duration_rows={song['item_id']: {'duration': 300} for song in songs},
    )
    assert sent == 24
    assert result['requested_output'] == {
        'target_count': None, 'target_duration_seconds': 1800,
    }
    assert len(result['playlist']) == 18
    assert 'Rank exactly 18 musically suitable' in prompts[1]
    assert '"duration_seconds":300' in prompts[1]


def test_duration_composer_repairs_infeasible_ui_default_count(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': str(i), 'title': f'Track {i}', 'artist': f'Artist {i % 8}'}
        for i in range(51)
    ]
    prompts = []

    def generate(prompt, _config, **kwargs):
        prompts.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL') and 'Your previous target_count' not in prompt:
            return json.dumps({
                'anchor_decisions': {}, 'target_count': 50,
                'target_duration_seconds': 1800,
            })
        if 'Your previous target_count' in prompt:
            return json.dumps({
                'anchor_decisions': {}, 'target_count': None,
                'target_duration_seconds': 1800,
            })
        records = json.loads(prompt.split('Available candidates: ', 1)[1])
        return json.dumps({'playlist_ids': [row['id'] for row in records[:18]]})

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'Make a playlist of exactly 30 minutes', songs, {'provider': 'OLLAMA'},
        ui_default_count=50,
        duration_rows={song['item_id']: {'duration': 300} for song in songs},
    )
    assert len(prompts) == 3
    assert 'target_count MUST be null' in prompts[0]
    assert 'Your previous target_count=50' in prompts[1]
    assert result['requested_output']['target_count'] is None
    assert len(result['playlist']) == 18


def test_composer_musical_review_applies_model_chosen_swaps(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': f'a-{i}', 'title': f'A {i}', 'artist': 'Artist A'}
        for i in range(3)
    ] + [
        {'item_id': 'b', 'title': 'B', 'artist': 'Artist B'},
        {'item_id': 'c', 'title': 'C', 'artist': 'Artist C'},
    ]
    prompts = []
    expected_order = []

    def generate(prompt, _config, **kwargs):
        prompts.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {}, 'target_count': 3,
                'target_duration_seconds': None,
            })
        if prompt.startswith('Review your ordered music selection'):
            chosen = json.loads(prompt.split('Current ordered selection: ', 1)[1].split('\nRemaining candidates:', 1)[0])
            remaining = json.loads(prompt.split('Remaining candidates: ', 1)[1])
            expected_order[:] = [remaining[1]['artist'], chosen[0]['artist'], remaining[0]['artist']]
            return json.dumps({'swaps': [
                {'remove_id': chosen[1]['id'], 'add_id': remaining[0]['id']},
                {'remove_id': chosen[2]['id'], 'add_id': remaining[1]['id']},
            ], 'ordered_ids': [remaining[1]['id'], chosen[0]['id'], remaining[0]['id']]})
        records = json.loads(prompt.split('Available candidates: ', 1)[1])
        return json.dumps({'playlist_ids': [row['id'] for row in records if row['artist'] == 'Artist A']})

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'A varied three-track playlist', songs, {'provider': 'OLLAMA'},
        musical_review=True,
    )
    assert len(prompts) == 3
    assert len(result['playlist']) == 3
    assert {song['artist'] for song in result['playlist']} == {'Artist A', 'Artist B', 'Artist C'}
    assert [song['artist'] for song in result['playlist']] == expected_order


def test_composer_obeys_model_version_policy_for_candidate_selection(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': 'studio', 'title': 'The Song', 'artist': 'Artist'},
        {'item_id': 'live', 'title': 'The Song (Live)', 'artist': 'Artist'},
        {'item_id': 'instrumental', 'title': 'Other Song (Instrumental version)', 'artist': 'Artist'},
        {'item_id': 'soundtrack', 'title': 'A Song Soundtrack Version', 'artist': 'Artist'},
        {'item_id': 'evolution', 'title': 'A Song (Evolution track)', 'artist': 'Artist'},
        {'item_id': 'other', 'title': 'Other Song', 'artist': 'Artist'},
    ]
    available = []

    def generate(prompt, _config, **_kwargs):
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {}, 'target_count': 2,
                'target_duration_seconds': None,
                'allow_nonstandard_versions': False,
            })
        records = json.loads(prompt.split('Available candidates: ', 1)[1])
        available.extend(records)
        return json.dumps({'playlist_ids': [row['id'] for row in records]})

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'Two standard studio tracks', songs, {'provider': 'OLLAMA'},
    )
    assert {row['title'] for row in available} == {'The Song', 'Other Song'}
    assert {song['item_id'] for song in result['playlist']} == {'studio', 'other'}


def test_composer_selects_each_seed_neighborhood_to_model_balance_plan(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': 'a0', 'title': 'Seed A', 'artist': 'A'},
        {'item_id': 'b0', 'title': 'Seed B', 'artist': 'B'},
        {'item_id': 'a1', 'title': 'A One', 'artist': 'A'},
        {'item_id': 'a2', 'title': 'A Two', 'artist': 'A'},
        {'item_id': 'b1', 'title': 'B One', 'artist': 'B'},
        {'item_id': 'b2', 'title': 'B Two', 'artist': 'B'},
    ]
    prompts = []

    def generate(prompt, _config, **_kwargs):
        prompts.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {'A001': True, 'A002': True},
                'target_count': 4, 'target_duration_seconds': None,
            })
        if prompt.startswith('Plan musical representation'):
            if 'REVISE YOUR MUSICAL BALANCE PLAN' not in prompt:
                return json.dumps({'minimum_neighborhood_tracks': {'1': 2, '2': 1}})
            return json.dumps({'minimum_neighborhood_tracks': {'1': 1, '2': 1}})
        if prompt.startswith('Choose musically suitable real-library tracks'):
            rows = json.loads(prompt.split('Eligible candidates: ', 1)[1])
            key = 'ids' if 'Seed neighborhood: B / Seed B' in prompt else 'playlist_ids'
            return json.dumps({key: [rows[0]['id']]})
        if prompt.startswith('Order exactly these LLM2-selected library tracks'):
            rows = json.loads(prompt.split('Selected tracks: ', 1)[1])
            return json.dumps({'playlist_ids': [row['id'] for row in reversed(rows)]})
        raise AssertionError(f'Unexpected Composer phase: {prompt[:80]}')

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'A playlist inspired by Seed A and Seed B', songs, {'provider': 'OLLAMA'},
        seed_provenance={'A / Seed A': {'a1', 'a2'}, 'B / Seed B': {'b1', 'b2'}},
        resolved_anchors=[{'type': 'song', 'resolved_track': songs[0]},
                          {'type': 'song', 'resolved_track': songs[1]}],
        musical_review=True,
    )
    assert sum('Choose musically suitable real-library tracks' in prompt for prompt in prompts) == 2
    assert sum('REVISE YOUR MUSICAL BALANCE PLAN' in prompt for prompt in prompts) == 1
    assert 'Order exactly these LLM2-selected library tracks' in prompts[-1]
    assert {'a0', 'b0'}.issubset({song['item_id'] for song in result['playlist']})
    assert len(result['playlist']) == 4
    assert {'a1', 'a2'} & {song['item_id'] for song in result['playlist']}
    assert {'b1', 'b2'} & {song['item_id'] for song in result['playlist']}


def test_composer_expands_candidates_after_phase_a_without_reinterpreting(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': str(index), 'title': f'Song {index}', 'artist': f'Artist {index}'}
        for index in range(4)
    ]
    calls = []

    def generate(prompt, _config, **_kwargs):
        calls.append(prompt)
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {}, 'target_count': 3,
                'target_duration_seconds': None, 'discovery_priority': 'exploratory',
            })
        rows = json.loads(prompt.split('Available candidates: ', 1)[1])
        return json.dumps({'playlist_ids': [row['id'] for row in rows[:3]]})

    def expand(intent, pool):
        assert intent['target_count'] == 3
        assert intent['discovery_priority'] == 'exploratory'
        return pool + songs[2:]

    monkeypatch.setattr(api, 'generate_text', generate)
    result, supplied = compose_playlist_with_llm(
        'Explore related artists', songs[:2], {'provider': 'OLLAMA'},
        expansion_callback=expand,
    )
    assert not result.get('error')
    assert supplied == 4
    assert len(result['playlist']) == 3
    assert sum(prompt.startswith('Interpret the ORIGINAL') for prompt in calls) == 1


def test_composer_rejects_required_anchor_unavailable_on_server(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    available = {'item_id': 'available', 'title': 'Available', 'artist': 'Artist'}
    missing = {'item_id': 'missing', 'title': 'Missing', 'artist': 'Artist'}
    monkeypatch.setattr(api, 'generate_text', lambda *_args, **_kwargs: json.dumps({
        'anchor_decisions': {'A001': True, 'A002': True},
        'target_count': 2, 'target_duration_seconds': None,
    }))
    result, _ = compose_playlist_with_llm(
        'Include both songs', [available], {'provider': 'OLLAMA'},
        resolved_anchors=[
            {'type': 'song', 'resolved_track': available},
            {'type': 'song', 'resolved_track': missing},
        ],
    )
    assert result['error']['category'] == 'COMPOSER_ANCHOR_UNAVAILABLE'
    assert 'Missing' in result['error']['reason']


def test_composer_continues_when_soft_balance_plan_is_malformed_twice(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': key, 'title': key, 'artist': artist}
        for key, artist in [('seed-a', 'A'), ('seed-b', 'B'),
                            ('related-a', 'C'), ('related-b', 'D')]
    ]
    balance_calls = []

    def generate(prompt, _config, **_kwargs):
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({'anchor_decisions': {'A001': True, 'A002': True},
                               'target_count': 4, 'target_duration_seconds': None})
        if prompt.startswith('Plan musical representation'):
            balance_calls.append(prompt)
            return json.dumps({'minimum_neighborhood_tracks': {'1': 'many'},
                               'artist_mix': {'max_tracks_per_artist': 2}})
        if prompt.startswith('Compose an ordered playlist selection'):
            rows = json.loads(prompt.split('Available candidates: ', 1)[1])
            return json.dumps({'playlist_ids': [row['id'] for row in rows]})
        raise AssertionError(prompt[:90])

    monkeypatch.setattr(api, 'generate_text', generate)
    logs = []
    result, sent = compose_playlist_with_llm(
        'Make a four-track discovery playlist around these two songs',
        songs, {'provider': 'OLLAMA'},
        seed_provenance={'A / seed-a': ['related-a'],
                         'B / seed-b': ['related-b']},
        resolved_anchors=[{'type': 'song', 'resolved_track': songs[0]},
                          {'type': 'song', 'resolved_track': songs[1]}],
        log_messages=logs,
    )
    assert sent == 4
    assert len(balance_calls) == 2
    assert len(result['playlist']) == 4
    assert not result.get('error')
    assert any('continuing without seed minima' in line for line in logs)


def test_composer_artist_mix_limits_concentration_without_losing_anchors(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': key, 'title': key, 'artist': artist}
        for key, artist in [
            ('a0', 'A'), ('b0', 'B'), ('a1', 'A'), ('a2', 'A'),
            ('c1', 'C'), ('d1', 'D'), ('b1', 'B'), ('b2', 'B'),
            ('e1', 'E'), ('f1', 'F'),
        ]
    ]

    def generate(prompt, _config, **_kwargs):
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {'A001': True, 'A002': True},
                'target_count': 6, 'target_duration_seconds': None,
                'discovery_priority': 'exploratory',
            })
        if prompt.startswith('Plan musical representation'):
            return json.dumps({
                'minimum_neighborhood_tracks': {'1': 2, '2': 2},
                'artist_mix': {'max_tracks_per_artist': 2, 'minimum_distinct_artists': 4},
            })
        if prompt.startswith('Choose musically suitable real-library tracks'):
            rows = json.loads(prompt.split('Eligible candidates: ', 1)[1])
            distinct = []
            seen_artists = set()
            for row in rows:
                if row['artist'] not in seen_artists:
                    distinct.append(row['id'])
                    seen_artists.add(row['artist'])
                if len(distinct) == 2:
                    break
            return json.dumps({'playlist_ids': distinct})
        if prompt.startswith('Order exactly these LLM2-selected library tracks'):
            rows = json.loads(prompt.split('Selected tracks: ', 1)[1])
            return json.dumps({'playlist_ids': [row['id'] for row in rows]})
        raise AssertionError(f'Unexpected Composer phase: {prompt[:80]}')

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'Discover music around two different songs', songs, {'provider': 'OLLAMA'},
        seed_provenance={
            'A / a0': ['a1', 'a2', 'c1', 'd1'],
            'B / b0': ['b1', 'b2', 'e1', 'f1'],
        },
        resolved_anchors=[
            {'type': 'song', 'resolved_track': songs[0]},
            {'type': 'song', 'resolved_track': songs[1]},
        ],
    )
    assert not result.get('error')
    assert len(result['playlist']) == 6
    assert {'a0', 'b0'}.issubset({song['item_id'] for song in result['playlist']})
    artists = [song['artist'] for song in result['playlist']]
    assert len(set(artists)) >= 4
    assert max(artists.count(artist) for artist in set(artists)) <= 2


def test_composer_retries_malformed_global_fill_without_losing_playlist(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': key, 'title': key, 'artist': artist}
        for key, artist in [
            ('a0', 'A'), ('b0', 'B'), ('a1', 'C'), ('a2', 'D'),
            ('b1', 'E'), ('b2', 'F'), ('extra', 'G'),
        ]
    ]
    global_calls = []
    order_calls = []

    def generate(prompt, _config, **_kwargs):
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({
                'anchor_decisions': {'A001': True, 'A002': True},
                'target_count': 5, 'target_duration_seconds': None,
            })
        if prompt.startswith('Plan musical representation'):
            return json.dumps({'minimum_neighborhood_tracks': {'1': 1, '2': 1}})
        if prompt.startswith('Choose musically suitable real-library tracks'):
            rows = json.loads(prompt.split('Eligible candidates: ', 1)[1])
            if 'Seed neighborhood: All related music' in prompt:
                global_calls.append(prompt)
                if len(global_calls) == 1:
                    return 'this is not JSON'
                return json.dumps({'playlist_ids': [999, 998, 997], 'track_ids': [rows[0]['id']]})
            return json.dumps({'playlist_ids': [rows[0]['id']]})
        if prompt.startswith('Order exactly these LLM2-selected library tracks'):
            order_calls.append(prompt)
            _kwargs['call_metadata']['done_reason'] = 'COMPOSER_REPETITION'
            return ''
        raise AssertionError(f'Unexpected Composer phase: {prompt[:80]}')

    monkeypatch.setattr(api, 'generate_text', generate)
    logs = []
    result, _ = compose_playlist_with_llm(
        'Mix these two songs', songs, {'provider': 'OLLAMA'},
        seed_provenance={'A / a0': ['a1', 'a2'], 'B / b0': ['b1', 'b2']},
        resolved_anchors=[
            {'type': 'song', 'resolved_track': songs[0]},
            {'type': 'song', 'resolved_track': songs[1]},
        ],
        log_messages=logs,
    )
    assert not result.get('error')
    assert len(result['playlist']) == 5
    assert len(global_calls) == 2
    assert len(order_calls) == 2
    assert any('model ordering unavailable; preserving LLM2 selections' in line for line in logs)


def test_composer_moves_unfilled_soft_seed_slots_to_llm_global_fill(monkeypatch):
    import tasks.ai.api as api
    from tasks.playlist_curation import compose_playlist_with_llm

    songs = [
        {'item_id': key, 'title': key, 'artist': artist}
        for key, artist in [
            ('a0', 'A'), ('b0', 'B'), ('a1', 'C'), ('a2', 'D'),
            ('b1', 'E'), ('b2', 'F'), ('b3', 'G'), ('global', 'H'),
        ]
    ]
    neighborhood_two_calls = []
    logs = []

    def generate(prompt, _config, **_kwargs):
        if prompt.startswith('Interpret the ORIGINAL'):
            return json.dumps({'anchor_decisions': {'A001': True, 'A002': True},
                               'target_count': 6, 'target_duration_seconds': None})
        if prompt.startswith('Plan musical representation'):
            return json.dumps({'minimum_neighborhood_tracks': {'1': 2, '2': 2}})
        if prompt.startswith('Choose musically suitable real-library tracks'):
            rows = json.loads(prompt.split('Eligible candidates: ', 1)[1])
            if 'Seed neighborhood: B / b0' in prompt:
                neighborhood_two_calls.append(prompt)
                return json.dumps({'playlist_ids': [rows[0]['id']]}) if len(neighborhood_two_calls) == 1 else json.dumps({'playlist_ids': [999]})
            count = int(prompt.split('Missing track choices: ', 1)[1].split('\n', 1)[0])
            return json.dumps({'playlist_ids': [row['id'] for row in rows[:count]]})
        if prompt.startswith('Order exactly these LLM2-selected library tracks'):
            rows = json.loads(prompt.split('Selected tracks: ', 1)[1])
            return json.dumps({'playlist_ids': [row['id'] for row in rows]})
        raise AssertionError(prompt[:90])

    monkeypatch.setattr(api, 'generate_text', generate)
    result, _ = compose_playlist_with_llm(
        'Mix these two songs into six tracks', songs, {'provider': 'OLLAMA'},
        seed_provenance={'A / a0': ['a1', 'a2'],
                         'B / b0': ['b1', 'b2', 'b3']},
        resolved_anchors=[{'type': 'song', 'resolved_track': songs[0]},
                          {'type': 'song', 'resolved_track': songs[1]}],
        log_messages=logs,
    )
    assert not result.get('error')
    assert len(result['playlist']) == 6
    assert {'a0', 'b0'}.issubset({song['item_id'] for song in result['playlist']})
    assert any('unfilled slots transferred to the global musical selection' in line for line in logs)


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
        'Choose five songs', [
            {'item_id': f'track-{i}', 'title': f'Track {i}', 'artist': 'Artist'}
            for i in range(5)
        ],
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


def test_partial_rank_supplements_remaining_candidates_in_native_order():
    songs = [{"item_id": f"track-{alias}", "alias": alias} for alias in ("C001", "C002", "C003", "C004")]
    ranked = rank_candidates_by_ids(
        songs, ["track-C003"], include_unselected=True,
    )
    assert [song["item_id"] for song in ranked] == [
        "track-C003", "track-C001", "track-C002", "track-C004",
    ]


def test_duration_optimizer_uses_real_durations_and_keeps_mandatory_seed():
    target_seconds = 1800
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
        songs, durations, target_seconds, 100, 1,
        mandatory_ids=["seed-opaque"],
    )
    assert result[0]["item_id"] == "seed-opaque"
    assert sum(durations[s["item_id"]] for s in result) == 1797


def test_shared_shortlist_builds_stable_aliases_and_authoritative_mapping():
    from tasks.playlist_curation import prepare_llm_candidate_shortlist

    songs = [
        {"item_id": f"track-{i:03d}", "title": f"Song {i:03d}", "artist": "Artist", "album": "Record"}
        for i in range(50)
    ]
    first = prepare_llm_candidate_shortlist(songs, "same-request", limit=50)
    second = prepare_llm_candidate_shortlist(songs, "same-request", limit=50)

    assert first == second
    assert len(first[3]) == 50
    assert all(set(row) <= {"id", "title", "artist", "album"} for row in first[3])


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

    def similarity(title, artist, count, *, seed_id=""):
        called.update(title=title, artist=artist, count=count, seed_id=seed_id)
        return {"songs": [{"item_id": "neighbor-opaque"}], "message": "similarity results"}

    monkeypatch.setattr(tools, "_song_similarity_api_sync", similarity)
    result = tools._dispatch_seed_search(
        {"seeds": [{"type": "song", "title": "Dark chest of wonders"}], "get_songs": 20},
        {},
    )
    assert called["title"] == "Dark Chest Of Wonders"
    assert called["artist"] == "Nightwish"
    assert called["seed_id"] == "seed-opaque"
    assert result["songs"][0]["item_id"] == "neighbor-opaque"
    assert "Seed ID: seed-opaque" in result["message"]


def test_canonical_song_seed_reaches_audiomuse_by_physical_id(monkeypatch):
    from tasks.ai import tools

    calls = []

    def similarity(title, artist, count, *, seed_id=""):
        calls.append((title, artist, count, seed_id))
        return {"songs": [{"item_id": "related"}], "message": "canonical lookup"}

    monkeypatch.setattr(tools, "_song_similarity_api_sync", similarity)
    result = tools._dispatch_seed_search({
        "seeds": [{"type": "song", "title": "Harvest", "artist": "Nightwish",
                   "track_id": "fp_47d67f301b070bf71a772b977c02f506f0abf6041f57179e43f"}],
        "get_songs": 50,
    }, {})
    assert calls == [("Harvest", "Nightwish", 50,
                      "fp_47d67f301b070bf71a772b977c02f506f0abf6041f57179e43f")]
    assert result["songs"][0]["item_id"] == "related"


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


def test_finalize_composer_duration_honors_model_preferred_count_when_feasible():
    from tasks.playlist_curation import finalize_composer_duration

    songs = [
        {'item_id': str(i), 'title': f'Track {i}', 'artist': f'Artist {i}'}
        for i in range(10)
    ]
    durations = {
        str(i): {'duration': 450 if i < 4 else 300}
        for i in range(10)
    }
    selected, actual, details = finalize_composer_duration(
        songs, durations, 1800, preferred_count=6, tolerance_seconds=15,
    )
    assert len(selected) == 6
    assert actual == 1800
    assert details['composer_preferred_count_used']


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
