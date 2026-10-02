# Validate Instant Playlist from this checkout

The regular Compose file uses the published AudioMuse-AI image. To run this
branch's source without local bind mounts, use the source override:

```sh
docker compose -f deployment/docker-compose.yaml \
  -f deployment/docker-compose-source.override.yaml up --build -d
```

The application also needs an indexed music library and a reachable AI provider.
In Instant Playlist, select **Ollama**, enter a model and an Ollama URL reachable
from the application container, and select **LLM Compose**. An Ollama server on
the host may need a container-reachable hostname or IP address; `localhost` in
the container refers to the container itself. The model must already be
available on that Ollama server.

The branch's focused unit tests do not require the music library or Ollama.
With Python 3.12 and the repository's test requirements installed, run:

```sh
python -m pip install -r test/requirements.txt
python -m pytest -q \
  test/unit/test_ai.py \
  test/unit/test_app_chat.py \
  test/unit/test_app_chat_ssrf.py \
  test/unit/test_chat_refinement.py \
  test/unit/test_composer_stream_guard.py \
  test/unit/test_mcp_server.py \
  test/unit/test_ollama_planner_failure.py \
  test/unit/test_playlist_curation.py \
  test/unit/test_sql_injection_params.py \
  test/unit/test_tool_plan.py
```

For live validation, request **LLM Compose** with these prompts after indexing
the relevant tracks:

1. `Can you create me a playlist which uses the Nightwish's song Harvest as a seed? I want to have a playlist of exactly 30 minutes. The songs should be similar to the seed.`
2. `Here is a list of songs I love: Temple of the King from Rainbow, Lound and clear from The Cranberries, Every breaking wave from U2, Paradise from Within Temptation. Could you assemble a playlist to contain these songs and add another 60 similar songs?`

Check that the first playlist contains the resolved Harvest track and has a
duration within `INSTANT_PLAYLIST_DURATION_TOLERANCE_SECONDS` of 1,800 seconds.
Check that the second contains the four resolved tracks and exactly 64 distinct
track IDs. The songs and artist distribution depend on the indexed library and
model output.
