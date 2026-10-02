# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Unit tests for chat request pre-validation and result shaping.

Covers the input guards and artist-diversity enforcement used by the chat
pipeline.

Main Features:
* Seed-search song-seed validation and search_database filter detection.
* Artist-diversity capping and progressive cap relaxation from the overflow.
* Playlist-length resolution from the request `n`, with no API upper bound.
* Multi-seed search interleaves the seeds rank by rank and drops shared songs.
"""

import json

import pytest

import app_chat
import config
from tasks.ai import tools


TRUTHY_SEARCH_FILTERS = {
    'genres': ['rock'],
    'moods': ['sad'],
    'tempo_min': 90,
    'tempo_max': 140,
    'energy_min': 0.2,
    'energy_max': 0.9,
    'key': 'C',
    'scale': 'minor',
    'year_min': 1990,
    'year_max': 1999,
    'min_rating': 4,
    'album': 'Album X',
    'other_features': ['party'],
    'candidate_item_ids': ['abc123'],
    'voices': ['female vocalists'],
    'instrumental': True,
    'exclude_artists': ['Artist B'],
    'exclude_genres': ['Hip-Hop'],
    'duration_min': 360,
    'duration_max': 180,
    'added_within_days': 30,
}


def _rerank_aliases(prompt):
    return json.loads(prompt.rsplit('Candidates: ', 1)[1])


def _song(item_id, artist):
    return {'item_id': item_id, 'artist': artist, 'title': f'{artist} {item_id}'}


def _run_pipeline_with_pool(monkeypatch, songs, payload_extra=None, filter_applied=True, plan_result_extra=None, translate_ids=None):
    import tasks.ai.planner as planner
    import tasks.mcp_helper as mcp_helper

    def _fake_plan(**kwargs):
        yield from ()
        return {
            'songs': list(songs),
            'song_sources': {s['item_id']: 0 for s in songs},
            'tools_used_history': [],
            'plan_notes': [],
            'executed_query_str': 'stub-query',
            'filter_applied': filter_applied,
            **(plan_result_extra or {}),
        }

    monkeypatch.setattr(planner, 'plan_and_execute_once', _fake_plan)
    monkeypatch.setattr(mcp_helper, 'get_library_context', lambda: {'total_songs': 0})
    monkeypatch.setattr(
        app_chat.app_server_context,
        'scope_results',
        lambda rows, _server, **kwargs: list(rows),
    )
    monkeypatch.setattr(
        app_chat.app_server_context,
        'translate_ids_for_request',
        translate_ids or (lambda ids: {str(item_id): str(item_id) for item_id in ids}),
    )

    payload = {'userInput': 'build me a playlist', 'ai_provider': 'OLLAMA'}
    payload.update(payload_extra or {})

    log_messages = []
    response, status = app_chat._drain_pipeline(
        app_chat._run_chat_pipeline(payload, log_messages)
    )
    assert status == 200
    return response


def test_planner_failure_is_visible_in_playlist_response(monkeypatch):
    response = _run_pipeline_with_pool(
        monkeypatch, [],
        {'selection_mode': 'LLM_COMPOSE'},
        plan_result_extra={'error': 'Ollama model process was killed by the server.'},
    )
    assert response['query_results'] is None
    assert response['shortfall_reason'] == 'Ollama model process was killed by the server.'


def test_compose_duration_only_finalizes_subset_and_receives_canonical_anchor(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    songs = [
        {'item_id': str(i), 'title': 'Harvest' if i == 0 else f'Track {i}',
         'artist': 'Nightwish' if i == 0 else f'Artist {i}'}
        for i in range(23)
    ]
    anchor = {
        'type': 'song', 'title': 'Harvest', 'artist': 'Nightwish',
        'resolved_track': songs[0],
        'resolved': {'title': 'Harvest', 'artist': 'Nightwish', 'track_id': '0'},
        'user_reference': {'title': 'Harvest', 'artist': 'the Nightwish'},
    }
    seen = {}

    def compose(_request, candidate_songs, _config, **kwargs):
        seen['anchors'] = kwargs['resolved_anchors']
        return {
            'playlist': candidate_songs,
            'requested_output': {'target_count': None, 'target_duration_seconds': 1800},
            'anchor_decisions': [{'id': 'A001', 'include': True}],
            'shortfall_reason': '30 minutes reached',
        }, len(candidate_songs)

    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm', compose)
    monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {
        str(item_id): {'duration': 280 if str(item_id) != '22' else 412}
        for item_id in ids
    })
    response = _run_pipeline_with_pool(
        monkeypatch, songs,
        {'selection_mode': 'LLM_COMPOSE', 'userInput': 'Use Harvest by Nightwish as a seed. Make the playlist exactly 30 minutes.'},
        plan_result_extra={
            'intent': {'anchors': [], 'resolved_anchors': []},
            'canonical_retrieval_anchors': [anchor],
        },
    )
    assert seen['anchors'] == [anchor]
    assert response['requested_output']['target_count'] is None
    assert response['actual_duration_seconds'] == 1812
    assert len(response['query_results']) == 6
    assert response['query_results'][0]['item_id'] == '0'
    assert 'Canonical anchors handed to Composer: 1' in response['message']
    assert 'Final requested duration: 1800 s' in response['message']
    assert 'Required anchors in final playlist: 1/1' in response['message']
    assert 'Duration fallback:' not in response['message']
    assert 'OK SUCCESS' in response['message']


def test_harvest_30_minute_request_uses_broader_pool_only_after_preferred_failure(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    request = ("Can you create me a playlist which uses the Nightwish's song Harvest "
               "as a seed? I want to have a playlist of exactly 30 minutes. "
               "The songs should be similar to the seed.")
    songs = [
        {'item_id': 'canonical-harvest', 'title': 'Harvest', 'artist': 'Nightwish'},
        {'item_id': 'preferred', 'title': 'Preferred', 'artist': 'Artist A'},
        {'item_id': 'broader', 'title': 'Broader', 'artist': 'Artist B'},
    ]
    anchor = {'type': 'song', 'resolved_track': songs[0],
              'resolved': {'title': 'Harvest', 'artist': 'Nightwish',
                           'track_id': 'canonical-harvest'}}

    def compose(user_request, pool, _config, **_kwargs):
        assert user_request == request
        assert len(pool) == 3
        return {'playlist': pool[:2],
                'requested_output': {'target_count': None, 'target_duration_seconds': 1800},
                'anchor_decisions': [{'id': 'A001', 'include': True}]}, 3

    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm', compose)
    monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {
        str(item_id): {'duration': {'canonical-harvest': 600, 'preferred': 500,
                                    'broader': 700}[str(item_id)]} for item_id in ids
    })
    response = _run_pipeline_with_pool(
        monkeypatch, songs, {'selection_mode': 'LLM_COMPOSE', 'userInput': request},
        plan_result_extra={'canonical_retrieval_anchors': [anchor]},
    )
    assert response['shortfall_reason'] is None
    assert response['actual_duration_seconds'] == 1800
    assert {song['item_id'] for song in response['query_results']} == {
        'canonical-harvest', 'preferred', 'broader'
    }
    assert 'Duration fallback: tolerance met using broader candidate universe' in response['message']
    assert 'Required anchors in final playlist: 1/1' in response['message']


def test_four_song_request_returns_64_tracks_and_reports_four_required_anchors(monkeypatch):
    from tasks import playlist_curation

    request = ('Here is a list of songs I love:\n'
               '- Temple of the King from Rainbow,\n'
               '- Lound and clear from The Cranberries,\n'
               '- Every breaking wave from U2,\n'
               '- Paradise from Within Temptation.\n'
               'Could you assemble a playlist to contain these songs and add another 60 similar songs?')
    seeds = [('Temple Of The King', 'Rainbow'), ('Loud And Clear', 'The Cranberries'),
             ('Every Breaking Wave', 'U2'),
             ('Paradise (What About Us?) (Feat. Tarja)', 'Within Temptation')]
    songs = [{'item_id': f'canonical-{i}', 'title': title, 'artist': artist}
             for i, (title, artist) in enumerate(seeds)]
    songs += [{'item_id': f'discovery-{i}', 'title': f'Discovery {i}',
               'artist': f'Discovery Artist {i}'} for i in range(60)]
    anchors = [{'type': 'song', 'resolved_track': song,
                'resolved': {'title': song['title'], 'artist': song['artist'],
                             'track_id': song['item_id']}} for song in songs[:4]]

    def compose(user_request, pool, _config, **_kwargs):
        assert user_request == request
        return {'playlist': list(pool),
                'requested_output': {'target_count': 64, 'target_duration_seconds': None},
                'anchor_decisions': [{'id': f'A{i:03d}', 'include': True}
                                     for i in range(1, 5)]}, 64

    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm', compose)
    response = _run_pipeline_with_pool(
        monkeypatch, songs, {'selection_mode': 'LLM_COMPOSE', 'userInput': request},
        plan_result_extra={'canonical_retrieval_anchors': anchors},
    )
    assert len(response['query_results']) == 64
    assert {song['item_id'] for song in songs[:4]}.issubset(
        {song['item_id'] for song in response['query_results']}
    )
    assert 'Required anchors in final playlist: 4/4' in response['message']


def test_compose_duration_violation_cannot_report_success(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    song = {'item_id': 'long', 'title': 'Long track', 'artist': 'Artist'}
    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm',
                        lambda _request, songs, _config, **kwargs: ({
                            'playlist': songs,
                            'requested_output': {'target_count': None, 'target_duration_seconds': 1800},
                            'anchor_decisions': [], 'shortfall_reason': '30 minutes reached',
                        }, len(songs)))
    monkeypatch.setattr(tool_impl, '_fetch_pool_features',
                        lambda ids: {item_id: {'duration': 6572} for item_id in ids})
    response = _run_pipeline_with_pool(
        monkeypatch, [song], {'selection_mode': 'LLM_COMPOSE'},
    )
    assert response['query_results'] == []
    assert response['actual_duration_seconds'] == 6572
    assert 'Playlist duration validation failed' in response['message']
    assert 'OK SUCCESS' not in response['message']


def test_compose_count_only_does_not_run_duration_optimizer(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    songs = [_song(str(i), f'Artist {i}') for i in range(3)]
    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm',
                        lambda _request, rows, _config, **kwargs: ({
                            'playlist': rows[:2],
                            'requested_output': {'target_count': 2, 'target_duration_seconds': None},
                            'anchor_decisions': [], 'shortfall_reason': None,
                        }, len(rows)))
    monkeypatch.setattr(playlist_curation, 'finalize_composer_duration',
                        lambda *args, **kwargs: pytest.fail('duration finalizer ran for count only'))
    monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {})
    response = _run_pipeline_with_pool(
        monkeypatch, songs, {'selection_mode': 'LLM_COMPOSE'},
    )
    assert len(response['query_results']) == 2
    assert response['target_duration_seconds'] is None
    assert 'OK SUCCESS' in response['message']


def test_compose_count_and_duration_respects_both(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    songs = [_song(str(i), f'Artist {i}') for i in range(4)]
    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm',
                        lambda _request, rows, _config, **kwargs: ({
                            'playlist': rows,
                            'requested_output': {'target_count': 2, 'target_duration_seconds': 900},
                            'anchor_decisions': [], 'shortfall_reason': None,
                        }, len(rows)))
    monkeypatch.setattr(tool_impl, '_fetch_pool_features', lambda ids: {
        item_id: {'duration': {'0': 600, '1': 300, '2': 310, '3': 1000}[item_id]}
        for item_id in ids
    })
    response = _run_pipeline_with_pool(
        monkeypatch, songs, {'selection_mode': 'LLM_COMPOSE'},
    )
    assert len(response['query_results']) == 2
    assert response['actual_duration_seconds'] == 900
    assert 'OK SUCCESS' in response['message']


def test_compose_delivers_provider_ids_from_initial_availability_check(monkeypatch):
    from tasks import playlist_curation
    import tasks.ai.tool_impl as tool_impl

    songs = [_song(str(i), f'Artist {i}') for i in range(3)]
    translations = []
    feature_calls = []

    def translate(ids):
        translations.append(list(ids))
        return {str(item_id): f'provider-{item_id}' for item_id in ids}

    def features(ids):
        feature_calls.append(list(ids))
        return {str(item_id): {'duration': 200} for item_id in ids}

    monkeypatch.setattr(playlist_curation, 'compose_playlist_with_llm',
                        lambda _request, rows, _config, **kwargs: ({
                            'playlist': rows,
                            'requested_output': {'target_count': 3, 'target_duration_seconds': None},
                            'anchor_decisions': [], 'shortfall_reason': None,
                        }, len(rows)))
    monkeypatch.setattr(tool_impl, '_fetch_pool_features', features)
    response = _run_pipeline_with_pool(
        monkeypatch, songs, {'selection_mode': 'LLM_COMPOSE'}, translate_ids=translate,
    )
    assert [row['item_id'] for row in response['query_results']] == [
        'provider-0', 'provider-1', 'provider-2',
    ]
    assert len(translations) == 1
    assert len(feature_calls) == 1
    assert 'Playlist response ready: 3 tracks' in response['message']


def test_resolved_explicit_song_is_mandatory_and_first_in_native_playlist(monkeypatch):
    import tasks.ai.tool_impl as tool_impl

    monkeypatch.setattr(tool_impl, 'resolve_song_by_title', lambda title, artist='': {
        'item_id': 'anchor', 'title': 'Harvest', 'author': 'Nightwish', 'album': 'HVMAN. :II: NATURE.'
    })
    songs = [
        {'item_id': f'optional-{i}', 'title': f'Optional {i}', 'artist': f'Artist {i}'}
        for i in range(5)
    ]
    response = _run_pipeline_with_pool(
        monkeypatch, songs,
        {'n': 3, 'userInput': 'Songs similar to Harvest by Nightwish.'},
        filter_applied=True,
        plan_result_extra={
            'intent': {'anchors': [{'type': 'song', 'title': 'Harvest', 'artist': 'Nightwish',
                                   'role': 'mandatory', 'include_in_final': True}],
                       'resolved_anchors': []},
            'mandatory_tracks': [{'item_id': 'anchor', 'title': 'Harvest', 'artist': 'Nightwish'}],
        },
    )
    assert len(response['query_results']) == 3
    assert response['query_results'][0]['item_id'] == 'anchor'
    assert 'Mandatory tracks present in final playlist: 1/1' in response['message']


def test_explicitly_excluded_song_is_not_mandatory(monkeypatch):
    import tasks.ai.tool_impl as tool_impl

    monkeypatch.setattr(tool_impl, 'resolve_song_by_title', lambda title, artist='': {
        'item_id': 'anchor', 'title': 'Harvest', 'author': 'Nightwish', 'album': 'Album'
    })
    songs = [
        {'item_id': f'optional-{i}', 'title': f'Optional {i}', 'artist': f'Artist {i}'}
        for i in range(5)
    ]
    response = _run_pipeline_with_pool(
        monkeypatch, songs,
        {'n': 3, 'userInput': "Songs like Harvest but don't include Harvest."},
        filter_applied=True,
        plan_result_extra={
            'intent': {'anchors': [{'type': 'song', 'title': 'Harvest', 'artist': 'Nightwish',
                                   'role': 'anchor', 'include_in_final': False}],
                       'resolved_anchors': []},
        },
    )
    assert all(song['item_id'] != 'anchor' for song in response['query_results'])
    assert 'Mandatory tracks: 0' in response['message']


def test_multiple_explicit_songs_raise_small_target_and_keep_mention_order(monkeypatch):
    import tasks.ai.tool_impl as tool_impl

    rows = {
        'Harvest': {'item_id': 'harvest', 'title': 'Harvest', 'author': 'Nightwish'},
        'Ghost Love Score': {'item_id': 'gls', 'title': 'Ghost Love Score', 'author': 'Nightwish'},
    }
    monkeypatch.setattr(tool_impl, 'resolve_song_by_title', lambda title, artist='': rows.get(title))
    response = _run_pipeline_with_pool(
        monkeypatch,
        [{'item_id': 'other', 'title': 'Other Song', 'artist': 'Other Artist'}],
        {'n': 1, 'userInput': 'Use Harvest and Ghost Love Score as seeds.'},
        filter_applied=True,
        plan_result_extra={
            'intent': {'anchors': [
                {'type': 'song', 'title': 'Harvest', 'artist': 'Nightwish', 'role': 'mandatory', 'include_in_final': True},
                {'type': 'song', 'title': 'Ghost Love Score', 'artist': 'Nightwish', 'role': 'mandatory', 'include_in_final': True},
            ], 'resolved_anchors': []},
            'mandatory_tracks': [
                {'item_id': 'harvest', 'title': 'Harvest', 'artist': 'Nightwish'},
                {'item_id': 'gls', 'title': 'Ghost Love Score', 'artist': 'Nightwish'},
            ],
        },
    )
    assert [song['item_id'] for song in response['query_results']] == ['harvest', 'gls']
    assert 'Mandatory tracks present in final playlist: 2/2' in response['message']


class TestSeedSearchSongSeedValidation:
    @pytest.mark.parametrize(
        'title,artist',
        [
            ('', 'Artist'),
            ('Song', ''),
            ('   ', 'Artist'),
            ('Song', '  \t  '),
            ('', ''),
        ],
    )
    def test_song_seed_with_blank_title_or_artist_is_skipped_before_the_similarity_call(
        self, monkeypatch, title, artist
    ):
        calls = []
        monkeypatch.setattr(
            tools,
            '_song_similarity_api_sync',
            lambda *args: calls.append(args) or {'songs': [], 'message': ''},
        )

        result = tools._dispatch_seed_search(
            {'seeds': [{'type': 'song', 'title': title, 'artist': artist}]}, {}
        )

        assert calls == []
        assert result['songs'] == []
        assert (
            'seed_search: skipping song seed with no title' in result['message']
            or 'seed_search: title-only seed could not be resolved' in result['message']
        )

    def test_complete_song_seed_reaches_the_similarity_call_stripped_of_whitespace(
        self, monkeypatch
    ):
        calls = []

        def _fake_similarity(seed_title, seed_artist, limit):
            calls.append((seed_title, seed_artist, limit))
            return {'songs': [{'item_id': 's1'}], 'message': 'ok'}

        monkeypatch.setattr(tools, '_song_similarity_api_sync', _fake_similarity)

        result = tools._dispatch_seed_search(
            {
                'seeds': [{'type': 'song', 'title': ' Song ', 'artist': ' Artist '}],
                'get_songs': 60,
            },
            {},
        )

        assert calls == [('Song', 'Artist', 60)]
        assert result['songs'] == [{'item_id': 's1'}]


class TestSeedSearchMultiSeedInterleave:
    def test_two_seeds_alternate_rank_by_rank_and_drop_shared_songs(self, monkeypatch):
        per_seed = {
            'Song 1': ['a1', 'a2', 'shared', 'a4'],
            'Song 2': ['b1', 'shared', 'b3'],
        }

        def _fake_similarity(seed_title, seed_artist, limit):
            return {'songs': [{'item_id': i} for i in per_seed[seed_title]], 'message': ''}

        monkeypatch.setattr(tools, '_song_similarity_api_sync', _fake_similarity)

        result = tools._dispatch_seed_search(
            {
                'seeds': [
                    {'type': 'song', 'title': 'Song 1', 'artist': 'Artist A'},
                    {'type': 'song', 'title': 'Song 2', 'artist': 'Artist B'},
                ],
                'get_songs': 60,
            },
            {},
        )

        assert [s['item_id'] for s in result['songs']] == [
            'a1', 'b1', 'a2', 'shared', 'b3', 'a4'
        ]

    def test_the_second_seed_reaches_the_head_of_a_long_first_list(self, monkeypatch):
        def _fake_similarity(seed_title, seed_artist, limit):
            return {
                'songs': [{'item_id': f'{seed_title}-{i}'} for i in range(limit)],
                'message': '',
            }

        monkeypatch.setattr(tools, '_song_similarity_api_sync', _fake_similarity)

        result = tools._dispatch_seed_search(
            {
                'seeds': [
                    {'type': 'song', 'title': 'Song 1', 'artist': 'Artist A'},
                    {'type': 'song', 'title': 'Song 2', 'artist': 'Artist B'},
                ],
                'get_songs': 50,
            },
            {},
        )

        head = [s['item_id'] for s in result['songs'][:10]]
        assert sum(1 for i in head if i.startswith('Song 2-')) == 5


class TestSearchDatabaseFilterDetection:
    def test_other_filter_key_set_is_exactly_the_one_this_module_pins(self):
        assert set(TRUTHY_SEARCH_FILTERS) == set(tools._SEARCH_OTHER_FILTER_KEYS)

    @pytest.mark.parametrize('key', sorted(TRUTHY_SEARCH_FILTERS))
    def test_each_other_filter_key_alone_counts_as_a_filter(self, key):
        assert tools._has_other_search_filters({key: TRUTHY_SEARCH_FILTERS[key]}) is True

    @pytest.mark.parametrize(
        'tool_args',
        [
            {},
            {'artist': 'Artist A'},
            {'get_songs': 200},
            {'genres': [], 'moods': None, 'album': ''},
        ],
    )
    def test_artist_only_and_empty_values_do_not_count_as_a_filter(self, tool_args):
        assert tools._has_other_search_filters(tool_args) is False

    def test_search_database_with_zero_filters_still_runs_the_query_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            tools,
            '_database_genre_query_sync',
            lambda *args, **kwargs: calls.append(kwargs.get('fuzzy_match'))
            or {'songs': [], 'message': 'no-op'},
        )

        result = tools._dispatch_search_database({})

        assert calls == [False]
        assert result == {'songs': [], 'message': 'no-op'}

    @pytest.mark.parametrize(
        'extra_filter,expected_fuzzy_calls',
        [
            ({}, ['Artist A']),
            ({'genres': ['rock']}, []),
            ({'voices': ['female vocalists']}, []),
            ({'exclude_genres': ['Hip-Hop']}, []),
        ],
    )
    def test_fuzzy_artist_fallback_runs_only_when_no_other_filter_narrows_the_search(
        self, monkeypatch, extra_filter, expected_fuzzy_calls
    ):
        query_calls = []
        fuzzy_calls = []

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(
            tools,
            '_database_genre_query_sync',
            lambda *args, **kwargs: query_calls.append(kwargs.get('fuzzy_match'))
            or {'songs': [], 'message': 'empty'},
        )
        monkeypatch.setattr(tools, '_get_db_connection', _FakeConn)
        monkeypatch.setattr(
            tools,
            '_fuzzy_match_author_title',
            lambda conn, name: fuzzy_calls.append(name) or None,
        )

        tool_args = {'artist': 'Artist A'}
        tool_args.update(extra_filter)
        tools._dispatch_search_database(tool_args)

        assert query_calls == [False, True]
        assert fuzzy_calls == expected_fuzzy_calls


class TestArtistDiversityEnforcement:
    @pytest.mark.parametrize('cap', [3, 5])
    def test_final_playlist_holds_at_most_the_configured_songs_per_artist(
        self, monkeypatch, cap
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', cap)
        songs = [_song(f'b{i}', 'Band C') for i in range(20)]
        songs += [_song(f'u{i}', f'Solo{i}') for i in range(180)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 100})

        results = response['query_results']
        assert len(results) == 100
        assert [s['item_id'] for s in results if s['artist'] == 'Band C'] == [
            f'b{i}' for i in range(cap)
        ]
        assert f'removed {20 - cap} excess songs from pool (max {cap}/artist)' in response['message']

    def test_overflow_songs_are_dropped_when_the_capped_pool_already_fills_the_target(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{a}s{i}', f'Artist{a}') for a in range(20) for i in range(10)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 100})

        results = response['query_results']
        assert [s['item_id'] for s in results] == [
            f'a{a}s{i}' for a in range(20) for i in range(5)
        ]
        assert 'Progressive cap relaxation' not in response['message']

    def test_pool_short_of_target_relaxes_the_cap_until_the_overflow_is_exhausted(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{i}', 'ArtistA') for i in range(30)]
        songs += [_song(f'b{i}', 'ArtistB') for i in range(30)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 100})

        results = response['query_results']
        assert len(results) == 60
        assert len([s for s in results if s['artist'] == 'ArtistA']) == 30
        assert len([s for s in results if s['artist'] == 'ArtistB']) == 30
        assert (
            'Progressive cap relaxation: 5 -> 30/artist to reach 60 songs' in response['message']
        )

    def test_cap_relaxation_admits_one_song_per_artist_per_level_so_small_overflows_go_first(
        self, monkeypatch
    ):
        monkeypatch.setattr(config, 'MAX_SONGS_PER_ARTIST_PLAYLIST', 5)
        songs = [_song(f'a{i}', 'ArtistA') for i in range(20)]
        songs += [_song(f'b{i}', 'ArtistB') for i in range(6)]
        songs += [_song(f'u{i}', f'Solo{i}') for i in range(80)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 100})

        results = response['query_results']
        assert len(results) == 100
        assert [s['item_id'] for s in results if s['artist'] == 'ArtistB'] == [
            f'b{i}' for i in range(6)
        ]
        assert [s['item_id'] for s in results if s['artist'] == 'ArtistA'] == [
            f'a{i}' for i in range(14)
        ]
        assert (
            'Progressive cap relaxation: 5 -> 14/artist to reach 100 songs' in response['message']
        )


class TestPlaylistLength:
    @pytest.mark.parametrize(
        'ui_count,requested_count,expected',
        [(25, 70, 70), (25, 12, 12), (45, 70, 70), (100, 100, 100)],
    )
    def test_llm_target_uses_explicit_request_with_infrastructure_limit(
        self, monkeypatch, ui_count, requested_count, expected
    ):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_MAX_N_RESULTS', 200)
        assert app_chat._resolve_llm_song_target(
            {'n': ui_count}, requested_count
        ) == (expected, ui_count, 200)

    def test_llm_target_without_prompt_count_uses_ui_default(self, monkeypatch):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_UI_DEFAULT_N_RESULTS', 25)
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_MAX_N_RESULTS', 200)
        assert app_chat._resolve_llm_song_target({}, None) == (25, 25, 200)

    def test_composer_candidate_limit_is_context_capacity(self, monkeypatch):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_COMPOSER_MAX_CANDIDATES', 300)
        assert app_chat._resolve_llm_candidate_limit(10) == 300
        assert app_chat._resolve_llm_candidate_limit(64) == 300

    def test_no_duration_constraint_skips_optimizer_and_preserves_native_order(self, monkeypatch):
        from tasks import playlist_curation

        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_SELECTION_MODE', 'NATIVE')
        monkeypatch.setattr(
            playlist_curation,
            'optimize_playlist_duration',
            lambda *args, **kwargs: pytest.fail('duration optimizer ran without a duration constraint'),
        )
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(8)]
        response = _run_pipeline_with_pool(
            monkeypatch, songs, {'n': 4}, filter_applied=False,
        )
        assert [song['item_id'] for song in response['query_results']] == ['u0', 'u1', 'u2', 'u3']
        assert 'Duration optimizer' not in response['message']
        assert 'Playlist kept in native/curator rank order (no explicit duration constraint)' in response['message']

    def test_request_without_n_falls_back_to_the_configured_default(self, monkeypatch):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_DEFAULT_N_RESULTS', 50)
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(300)]

        response = _run_pipeline_with_pool(monkeypatch, songs)

        assert len(response['query_results']) == 50
        assert 'Target: 50 songs' in response['message']

    def test_requested_n_sets_the_playlist_length(self, monkeypatch):
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(300)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 25})

        assert len(response['query_results']) == 25
        assert 'Target: 25 songs' in response['message']

    def test_api_honours_an_n_above_the_frontend_only_maximum(self, monkeypatch):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_MAX_N_RESULTS', 200)
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(600)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 500})

        assert len(response['query_results']) == 500

    def test_collected_pool_grows_with_the_target_so_a_long_playlist_can_fill(self, monkeypatch):
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(4000)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 300})

        assert len(response['query_results']) == 300

    @pytest.mark.parametrize('bad_value', ['many', None, '', {'n': 10}])
    def test_unparsable_n_falls_back_to_the_configured_default(self, monkeypatch, bad_value):
        monkeypatch.setattr(config, 'INSTANT_PLAYLIST_DEFAULT_N_RESULTS', 50)
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(300)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': bad_value})

        assert len(response['query_results']) == 50

    def test_a_fractional_n_is_truncated_to_a_whole_number_of_songs(self, monkeypatch):
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(300)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': 12.7})

        assert len(response['query_results']) == 12

    @pytest.mark.parametrize('bad_value', [0, -5])
    def test_n_below_one_is_floored_to_a_single_song(self, monkeypatch, bad_value):
        songs = [_song(f'u{i}', f'Solo{i}') for i in range(300)]

        response = _run_pipeline_with_pool(monkeypatch, songs, {'n': bad_value})

        assert len(response['query_results']) == 1


class TestStreamingErrorEvent:
    @staticmethod
    def _error_event(monkeypatch, failure):
        from flask import Flask

        def _pipeline(data, log_messages):
            raise failure
            yield

        monkeypatch.setattr(app_chat, '_run_chat_pipeline', _pipeline)
        monkeypatch.setattr(
            app_chat.app_server_context, 'resolve_request_server_id', lambda data: None
        )
        app = Flask(__name__)
        app.register_blueprint(app_chat.chat_bp, url_prefix='/chat')
        response = app.test_client().post(
            '/chat/api/chatPlaylistStream', json={'userInput': 'calm piano'}
        )
        events = [
            json.loads(line[len('data: '):])
            for line in response.get_data(as_text=True).splitlines()
            if line.startswith('data: ')
        ]
        return next(event for event in events if event['type'] == 'error')

    def test_a_classified_failure_shows_its_own_message_not_the_generic_one(self, monkeypatch):
        operational_error = type('OperationalError', (Exception,), {'__module__': 'psycopg2'})

        event = self._error_event(monkeypatch, operational_error('server closed'))

        assert event['error_code'] == 4001
        assert event['error'] == event['error_message'], (
            'chat.html renders the error alias through apiErrorText; a fixed generic alias '
            'hid the database outage the event had already classified'
        )
        assert 'server closed' not in json.dumps(event)

    def test_an_unclassified_failure_keeps_the_generic_text(self, monkeypatch):
        event = self._error_event(monkeypatch, KeyError('secret detail'))

        assert event['error_code'] == 9999
        assert event['error'] == 'An internal error has occurred.'
        assert 'secret detail' not in json.dumps(event)


def test_ollama_model_api_lists_models_from_selected_server(monkeypatch):
    from flask import Flask
    import requests
    import ssrf_guard

    requested = {}
    monkeypatch.setattr(ssrf_guard, 'validate_outbound_url', lambda _url: (True, None))

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {'models': [
                {'name': 'qwen3.5:9b'}, {'model': 'llama3.2:latest'}, {'name': 'qwen3.5:9b'},
            ]}

    def fake_get(url, **kwargs):
        requested.update(url=url, **kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, 'get', fake_get)
    flask_app = Flask(__name__)
    flask_app.register_blueprint(app_chat.chat_bp, url_prefix='/chat')
    response = flask_app.test_client().post(
        '/chat/api/ollama_models',
        json={'server_url': 'http://ollama.local:11434/api/generate'},
    )

    assert response.status_code == 200
    assert response.json['models'] == ['llama3.2:latest', 'qwen3.5:9b']
    assert requested['url'] == 'http://ollama.local:11434/api/tags'
    assert requested['timeout'] <= 8


def test_ollama_model_api_rejects_invalid_url_and_handles_unavailable_server(monkeypatch):
    from flask import Flask
    import requests
    import ssrf_guard

    flask_app = Flask(__name__)
    flask_app.register_blueprint(app_chat.chat_bp, url_prefix='/chat')
    client = flask_app.test_client()
    invalid = client.post('/chat/api/ollama_models', json={'server_url': 'file:///etc/passwd'})
    assert invalid.status_code == 400

    monkeypatch.setattr(ssrf_guard, 'validate_outbound_url', lambda _url: (True, None))
    monkeypatch.setattr(requests, 'get', lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError()))
    unavailable = client.post(
        '/chat/api/ollama_models', json={'server_url': 'http://ollama.local:11434'}
    )
    assert unavailable.status_code == 502
    assert unavailable.json['models'] == []
