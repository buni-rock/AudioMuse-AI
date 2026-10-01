# Composer production regression — 2026-10-01

The exact four-song request was submitted twice through the real endpoint:
`POST http://audiomuse.acasa/chat/api/chatPlaylistStream`.
Provider: Ollama; model: `gemma4:26b`; selection mode: `LLM_COMPOSE`; UI count: 50.
The second run used the final rebuilt and restarted application.

```text
Here is a list of songs I love:

- Temple of the King from Rainbow,
- Lound and clear from The Cranberries,
- Every breaking wave from U2,
- Paradise from Within Temptation.

Could you assemble a playlist to contain these songs and add another
60 similar songs?
```

## Verified in both runs

- Phase A returned target_count=64, target_duration_seconds=null, and included all four anchors.
- Retrieval preserved 60 candidates per seed, 240 retrieved neighbors, all four neighborhoods, and 242 Composer candidates after anchor merging and duplicate-content suppression.
- Phase B used request-local numeric IDs and stopped naturally; no pathological repetition or repair was needed.
- The model over-selected; the existing validator removed duplicates and trimmed the tail while protecting the anchors. Raw and validated counts are shown below.
- The final endpoint response contained exactly 64 distinct track IDs and all four canonical requested songs (including Loud And Clear and Paradise (What About Us?) (Feat. Tarja)).

## Automated checks

82 focused tests passed in `test/unit/test_composer_stream_guard.py` and
`test/unit/test_playlist_curation.py`. Coverage includes chunk boundaries,
repeated aliases/cycles, immediate stream closure, token-limit prefix recovery,
strict think=false, and preserving the first 61 tracks in order while requesting
only three additions from remaining candidates.

A broader run also executed the chat/refinement suites: 271 passed, 14 failed.
Failures concern old count/candidate-cap expectations, legacy RERANK/CURATE
fixtures, and a genre-backfill expectation. Those broader failures remain;
this report does not claim the entire suite passes.

## First production run

```text
Compose Phase A: duration=16.4s; output tokens=50; budget=512; done_reason=stop
Compose Phase A: target_count=64; target_duration=None; anchor decisions={"A001": true, "A002": true, "A003": true, "A004": true}
Compose Phase B: candidates supplied=242; requested selections=64
Compose Phase B: duration=30.9s; output tokens=298; budget=512; done_reason=stop
Compose Phase B: IDs returned=80; unique IDs=77
Tail trimmed: 13
Final playlist: 64
Final playlist: 64
```

## Final production run

```text
Compose Phase A: duration=24.7s; output tokens=50; budget=512; done_reason=stop
Compose Phase A: target_count=64; target_duration=None; anchor decisions={"A001": true, "A002": true, "A003": true, "A004": true}
Compose Phase B: candidates supplied=242; requested selections=64
Compose Phase B: duration=31.1s; output tokens=297; budget=512; done_reason=stop
Compose Phase B: IDs returned=83; unique IDs=82
Tail trimmed: 18
Final playlist: 64
Final playlist: 64
```
