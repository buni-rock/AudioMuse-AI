import httpx


def test_killed_ollama_model_reports_server_failure_without_retry(monkeypatch):
    from tasks.ai.providers import openai

    request = httpx.Request('POST', 'http://localhost:11434/api/chat')
    response = httpx.Response(
        500, request=request,
        json={'error': 'llama-server process has terminated: signal: killed'},
    )

    def fail_native(*args, **kwargs):
        raise httpx.HTTPStatusError('HTTP 500', request=request, response=response)

    def no_structured_retry(*args, **kwargs):
        raise AssertionError('A killed model must not be loaded again')

    monkeypatch.setattr(openai, '_try_native_ollama_tool_call', fail_native)
    monkeypatch.setattr(openai, '_try_structured_ollama_call', no_structured_retry)
    logs = []
    result = openai.call_with_tools_ollama(
        'http://localhost:11434', 'qwen3.6:27b', 'Find Harvest', [], logs,
    )

    assert 'qwen3.6:27b' in result['error']
    assert 'killed by the server' in result['error']
    assert 'smaller model' in result['error']
    assert any('Ollama planner unavailable' in line for line in logs)
    assert not any('structured-output' in line for line in logs)
