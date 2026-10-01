import json
from unittest.mock import MagicMock

from tasks.ai.providers.openai import SelectionStreamGuard, _ollama_selection_stream


def test_chunk_boundaries_and_natural_completion():
    guard = SelectionStreamGuard()
    for chunk in ['{"playlist_', 'ids": [1', '23, 2', '4, 8', ']}']:
        guard.feed(chunk)
    assert guard.refs == [123, 24, 8]
    assert guard.closed and not guard.repetition


def test_repeated_aliases_and_cycles_stop_early():
    for refs in [['XNBM'] * 50, [1, 2] * 50]:
        guard = SelectionStreamGuard()
        guard.feed(json.dumps({'playlist_ids': refs}))
        assert guard.repetition
        assert len(guard.refs) <= 6


def test_stream_closes_and_preserves_prefix(monkeypatch):
    client = MagicMock()
    response = client.__enter__.return_value.stream.return_value.__enter__.return_value
    consumed = []
    def chunks():
        for ref in range(1, 62):
            consumed.append(ref)
            yield json.dumps({'message': {'content': ('{"playlist_ids":[' if ref == 1 else '') + str(ref) + ','}})
        for _ in range(1000):
            consumed.append(61)
            yield json.dumps({'message': {'content': '61,'}})
    response.iter_lines.side_effect = chunks
    monkeypatch.setattr('tasks.ai.providers.openai.httpx.Client', lambda **kwargs: client)
    result = _ollama_selection_stream('http://example/api/chat', 'model', {}, timeout=None, operation='text')
    assert result['done_reason'] == 'COMPOSER_REPETITION'
    refs = json.loads(result['message']['content'])['playlist_ids']
    assert refs[:61] == list(range(1, 62))
    assert len(consumed) == 65
    client.__enter__.return_value.stream.return_value.__exit__.assert_called_once()


def test_stream_retains_real_stop_reason_and_token_count(monkeypatch):
    client = MagicMock()
    response = client.__enter__.return_value.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([
        json.dumps({'message': {'content': '{"playlist_ids":[1,2]}'}}),
        json.dumps({'done': True, 'done_reason': 'stop', 'eval_count': 12}),
    ])
    monkeypatch.setattr('tasks.ai.providers.openai.httpx.Client', lambda **kwargs: client)
    result = _ollama_selection_stream('http://example/api/chat', 'model', {}, timeout=None, operation='text')
    assert result['done_reason'] == 'stop'
    assert result['eval_count'] == 12


def test_length_limit_recovers_completed_numeric_prefix(monkeypatch):
    client = MagicMock()
    response = client.__enter__.return_value.stream.return_value.__enter__.return_value
    response.iter_lines.return_value = iter([
        json.dumps({'message': {'content': '{"playlist_ids":[12,34,5'}}),
        json.dumps({'done': True, 'done_reason': 'length', 'eval_count': 512}),
    ])
    monkeypatch.setattr('tasks.ai.providers.openai.httpx.Client', lambda **kwargs: client)
    result = _ollama_selection_stream('http://example/api/chat', 'model', {}, timeout=None, operation='text')
    assert result['done_reason'] == 'length'
    assert json.loads(result['message']['content']) == {'playlist_ids': [12, 34]}


def test_strict_composer_think_false_overrides_cached_fallback(monkeypatch):
    from tasks.ai.providers import openai
    captured = []
    def transport(url, model, payload, **kwargs):
        captured.append(payload)
        return {'message': {'content': '{"playlist_ids":[1]}'}, 'done_reason': 'stop'}
    monkeypatch.setattr(openai, '_OLLAMA_TEXT_THINK_SETTINGS', {('http://example/api/chat', 'model'): 'low'})
    monkeypatch.setattr(openai, '_ollama_selection_stream', transport)
    raw = openai.generate_text_ollama_chat('http://example', 'model', 'select',
        think=False, allow_think_fallbacks=False, selection_stream=True)
    assert json.loads(raw) == {'playlist_ids': [1]}
    assert len(captured) == 1
    assert captured[0]['think'] is False
