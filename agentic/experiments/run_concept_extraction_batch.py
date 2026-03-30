#!/usr/bin/env python3
"""Batch concept extraction for Experiment 2.

Extracts concepts from ALL 1,908 conversations using Ollama (llama3.2).
Designed for incremental runs — saves state every 25 episodes, can be
resumed with --resume.

Usage:
    # First run
    python experiments/run_concept_extraction_batch.py --limit 100

    # Resume
    python experiments/run_concept_extraction_batch.py --resume --limit 200

    # Full run (will take ~12-16 hours with llama3.2)
    python experiments/run_concept_extraction_batch.py --resume
"""

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from concept_extraction_v2 import (
    ExtractionState,
    extract_concepts_from_episode,
    get_remaining_episodes,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).parent / "results" / "hierarchy" / "extraction_state.json"


def make_ollama_fn(model: str = "llama3.2"):
    import requests

    def fn(prompt):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt,
                   "stream": False, "options": {"temperature": 0.3}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"]
    return fn


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max episodes (0 = all remaining)")
    parser.add_argument("--model", type=str, default="llama3.2")
    parser.add_argument("--save-every", type=int, default=25)
    args = parser.parse_args()

    if args.resume and STATE_FILE.exists():
        state = ExtractionState.load(STATE_FILE)
        logger.info(f"Resumed: {state.total_extractions} done, "
                    f"{len(state.vocabulary)} concepts")
    else:
        state = ExtractionState()

    remaining = get_remaining_episodes(state)
    if args.limit > 0:
        remaining = remaining[:args.limit]

    logger.info(f"Episodes to process: {len(remaining)}")
    if not remaining:
        logger.info("All done!")
        return

    llm_fn = make_ollama_fn(args.model)
    t_start = time.time()

    for i, episode_id in enumerate(remaining):
        try:
            result = extract_concepts_from_episode(episode_id, state, llm_fn)
            concepts_str = ', '.join(result.concepts[:5])
            if len(result.concepts) > 5:
                concepts_str += f', ... (+{len(result.concepts) - 5})'
            print(f"[{state.total_extractions}/{state.total_extractions + len(remaining) - i - 1}] "
                  f"{result.title}: {concepts_str} ({result.extraction_time:.1f}s)")
        except Exception as e:
            logger.error(f"FAILED {episode_id}: {e}")
            continue

        if (i + 1) % args.save_every == 0:
            state.save(STATE_FILE)
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta_remaining = (len(remaining) - i - 1) / rate
            logger.info(f"  [checkpoint] {state.total_extractions} done, "
                        f"{len(state.vocabulary)} concepts, "
                        f"{rate:.1f} ep/s, "
                        f"ETA: {eta_remaining / 60:.0f} min")

    state.save(STATE_FILE)
    elapsed = time.time() - t_start
    print(f"\nComplete: {state.total_extractions} episodes, "
          f"{len(state.vocabulary)} unique concepts, "
          f"{elapsed / 60:.1f} min total")


if __name__ == "__main__":
    main()
