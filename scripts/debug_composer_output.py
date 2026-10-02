#!/usr/bin/env python3
"""Run the production Composer prompt against 10 then 242 synthetic candidates.

Example: python3 scripts/debug_composer_output.py --url http://192.168.144.8:11434 --model gemma4:26b
Use --composer-source /tmp/playlist_curation.py when probing a copied source file.
"""

import argparse
import importlib.util
import logging

import config


def _composer(source):
    if not source:
        from tasks.playlist_curation import compose_playlist_with_llm
        return compose_playlist_with_llm
    spec = importlib.util.spec_from_file_location("composer_debug_source", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compose_playlist_with_llm


def _fixture(size, anchor_count):
    songs = [
        {"item_id": f"probe-{index:03d}", "title": f"Probe Track {index:03d}",
         "artist": f"Probe Artist {index % 17}"}
        for index in range(size)
    ]
    if size == 242:
        for song, title, artist in zip(songs, (
            "Temple Of The King", "Loud And Clear", "Every Breaking Wave",
            "Paradise (What About Us?) (Feat. Tarja)",
        ), ("Rainbow", "The Cranberries", "U2", "Within Temptation")):
            song.update(title=title, artist=artist)
    anchors = [
        {"type": "song", "title": song["title"], "artist": song["artist"],
         "resolved_track": song}
        for song in songs[:anchor_count]
    ]
    return songs, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=config.OLLAMA_SERVER_URL)
    parser.add_argument("--model", default=config.OLLAMA_MODEL_NAME)
    parser.add_argument("--composer-source")
    parser.add_argument("--debug-full", action="store_true", help="log full assistant content at DEBUG level")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug_full else logging.INFO)
    config.COMPOSER_MAX_OUTPUT_TOKENS = 4096
    compose = _composer(args.composer_source)
    ai_config = {"provider": "OLLAMA", "ollama_url": args.url, "ollama_model": args.model}

    for size, anchor_count, target in ((10, 2, 5), (242, 4, 64)):
        songs, anchors = _fixture(size, anchor_count)
        request = (
            "Here is a list of songs I love:\n"
            "- Temple of the King from Rainbow,\n"
            "- Lound and clear from The Cranberries,\n"
            "- Every breaking wave from U2,\n"
            "- Paradise from Within Temptation.\n\n"
            "Could you assemble a playlist to contain these songs and add another\n"
            "60 similar songs?"
            if size == 242 else
            f"Include the {anchor_count} listed songs and add another {target - anchor_count} similar songs."
        )
        logs = []
        result, sent = compose(
            request, songs, ai_config, resolved_anchors=anchors,
            ui_default_count=50, log_messages=logs,
        )
        selected_ids = {song["item_id"] for song in result.get("playlist", [])}
        anchors_present = sum(anchor["resolved_track"]["item_id"] in selected_ids for anchor in anchors)
        print(f"case={size} candidates_sent={sent} playlist_count={len(result.get('playlist', []))} "
              f"anchors_present={anchors_present}/{anchor_count} error={result.get('error')}", flush=True)
        for line in logs:
            if any(marker in line for marker in (
                "Configured Composer max output tokens:",
                "Compose interpretation:", "Compose selection:",
                "Compose selection validation:", "Composer result:",
                "Composer raw IDs:", "Validated ordered selection:",
                "Targeted fill requested:", "Targeted fill returned:",
                "Tail trimmed:", "Final playlist:",
            )):
                print(line, flush=True)
        if (
            result.get("error") or len(result.get("playlist", [])) != target
            or anchors_present != anchor_count
            or not any(f"Compose interpretation: target_count={target};" in line for line in logs)
            or f"Final playlist: {target}" not in logs
        ):
            print("Probe stopped after an unsuccessful case.", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
