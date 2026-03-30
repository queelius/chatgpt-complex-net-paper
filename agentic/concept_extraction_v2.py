"""Semantic concept extraction from individual episodes (Experiment 2).

Extracts 3-10 noun-phrase concepts per episode using an LLM, maintaining a
running vocabulary for online deduplication. Concepts are fine-grained;
hierarchy emerges later by clustering concept embeddings.

Design decisions (from brainstorming):
  - Granularity: single-level extraction, hierarchy emerges from clustering
  - Deduplication: embed-then-cluster (geometric), but LLM also reuses known concepts
  - Count per episode: variable, LLM decides (3-10)
  - Concept form: noun phrases
  - Running vocabulary: LLM sees existing concepts and reuses where appropriate
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
CONVERSATIONS_DIR = REPO_ROOT / "dev" / "chatgpt-4-11-2025_json_no_embeddings"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class EpisodeConcepts:
    """Concepts extracted from a single episode."""
    episode_id: str
    title: str
    concepts: list[str]  # noun phrases
    message_count: int
    extraction_time: float  # seconds


@dataclass
class ExtractionState:
    """Running state for incremental concept extraction."""
    vocabulary: list[str] = field(default_factory=list)  # known concepts so far
    episodes: list[EpisodeConcepts] = field(default_factory=list)
    total_extractions: int = 0
    total_concepts_raw: int = 0

    def add(self, result: EpisodeConcepts):
        self.episodes.append(result)
        self.total_extractions += 1
        self.total_concepts_raw += len(result.concepts)
        # Update vocabulary with new concepts
        vocab_set = set(c.lower() for c in self.vocabulary)
        for concept in result.concepts:
            if concept.lower() not in vocab_set:
                self.vocabulary.append(concept)
                vocab_set.add(concept.lower())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'vocabulary': self.vocabulary,
                'vocabulary_size': len(self.vocabulary),
                'total_extractions': self.total_extractions,
                'total_concepts_raw': self.total_concepts_raw,
                'episodes': [asdict(e) for e in self.episodes],
            }, f, indent=2)
        logger.info(f"Saved state: {self.total_extractions} episodes, "
                    f"{len(self.vocabulary)} unique concepts → {path}")

    @classmethod
    def load(cls, path: Path) -> 'ExtractionState':
        with open(path) as f:
            data = json.load(f)
        state = cls(
            vocabulary=data['vocabulary'],
            total_extractions=data['total_extractions'],
            total_concepts_raw=data['total_concepts_raw'],
        )
        for ep in data['episodes']:
            state.episodes.append(EpisodeConcepts(**ep))
        return state


# ---------------------------------------------------------------------------
# Conversation loading
# ---------------------------------------------------------------------------

def load_conversation(episode_id: str, conversations_dir: Optional[Path] = None) -> dict:
    """Load a single conversation JSON."""
    conv_dir = conversations_dir or CONVERSATIONS_DIR
    fpath = conv_dir / f"{episode_id}.json"
    with open(fpath) as f:
        doc = json.load(f)
    return doc


def format_conversation_for_extraction(doc: dict, max_messages: int = 20,
                                        max_chars: int = 4000) -> str:
    """Format a conversation for the extraction prompt.

    Includes title and first N messages, truncated to max_chars.
    """
    title = doc.get('title', 'Untitled')
    messages = doc.get('messages', [])

    parts = [f"Title: {title}\n"]
    char_count = len(parts[0])

    for m in messages[:max_messages]:
        role = m.get('role', 'unknown')
        content = m.get('content', '')
        if isinstance(content, list):
            # Handle content parts (some exports have list format)
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
            content = ' '.join(text_parts)
        if not isinstance(content, str) or not content.strip():
            continue

        line = f"[{role}] {content[:500]}\n"
        if char_count + len(line) > max_chars:
            remaining = max_chars - char_count
            if remaining > 50:
                parts.append(line[:remaining] + "...")
            break
        parts.append(line)
        char_count += len(line)

    return ''.join(parts)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

def build_extraction_prompt(
    conversation_text: str,
    vocabulary: list[str],
    max_vocab_shown: int = 200,
) -> str:
    """Build the prompt for concept extraction from a single episode.

    Includes the running vocabulary so the LLM can reuse existing concepts.
    """
    vocab_section = ""
    if vocabulary:
        shown = vocabulary[:max_vocab_shown]
        vocab_section = f"""
EXISTING CONCEPTS (reuse these where appropriate rather than creating synonyms):
{', '.join(shown)}
{'... and ' + str(len(vocabulary) - max_vocab_shown) + ' more' if len(vocabulary) > max_vocab_shown else ''}

"""

    return f"""Extract the key knowledge concepts from this AI conversation as noun phrases.

RULES:
- Extract 3-10 concepts depending on the conversation's breadth
- Each concept should be a noun phrase (2-5 words), e.g., "Bayesian posterior estimation", "Python package management"
- Be specific: "Metropolis-Hastings algorithm" not "statistics"
- Reuse existing concepts from the vocabulary below when the conversation covers them
- Only create new concepts when the conversation covers genuinely new ground
- Focus on WHAT knowledge is being explored, not HOW the conversation flows
{vocab_section}
CONVERSATION:
{conversation_text}

Return ONLY a JSON array of concept strings, e.g.:
["concept one", "concept two", "concept three"]"""


# ---------------------------------------------------------------------------
# Extraction (designed for pluggable LLM backend)
# ---------------------------------------------------------------------------

def extract_concepts_from_episode(
    episode_id: str,
    state: ExtractionState,
    llm_fn,
    conversations_dir: Optional[Path] = None,
) -> EpisodeConcepts:
    """Extract concepts from a single episode using the provided LLM function.

    Args:
        episode_id: Filename stem of the conversation
        state: Running extraction state (vocabulary)
        llm_fn: Callable that takes a prompt string and returns a response string
        conversations_dir: Override for conversation JSON directory
    """
    doc = load_conversation(episode_id, conversations_dir)
    conv_text = format_conversation_for_extraction(doc)
    prompt = build_extraction_prompt(conv_text, state.vocabulary)

    t0 = time.time()
    response = llm_fn(prompt)
    elapsed = time.time() - t0

    concepts = parse_concept_list(response)

    result = EpisodeConcepts(
        episode_id=episode_id,
        title=doc.get('title', episode_id),
        concepts=concepts,
        message_count=len(doc.get('messages', [])),
        extraction_time=round(elapsed, 2),
    )

    state.add(result)
    return result


def parse_concept_list(response: str) -> list[str]:
    """Parse LLM response into a list of concept strings."""
    import re

    # Try to find a JSON array in the response
    json_match = re.search(r'\[.*?\]', response, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            if isinstance(items, list):
                return [str(item).strip() for item in items
                        if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: extract quoted strings
    quoted = re.findall(r'"([^"]+)"', response)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    # Last resort: split by newlines, strip bullets
    lines = response.strip().split('\n')
    concepts = []
    for line in lines:
        line = re.sub(r'^[\s\-\*\d\.]+', '', line).strip()
        if line and len(line) < 100:
            concepts.append(line)
    return concepts[:10]


# ---------------------------------------------------------------------------
# Batch extraction helpers
# ---------------------------------------------------------------------------

def get_all_episode_ids(conversations_dir: Optional[Path] = None) -> list[str]:
    """Get all episode IDs from the conversations directory."""
    conv_dir = conversations_dir or CONVERSATIONS_DIR
    return sorted([f.stem for f in conv_dir.glob("*.json")])


def get_remaining_episodes(
    state: ExtractionState,
    conversations_dir: Optional[Path] = None,
) -> list[str]:
    """Get episode IDs not yet extracted."""
    all_ids = set(get_all_episode_ids(conversations_dir))
    done_ids = set(e.episode_id for e in state.episodes)
    return sorted(all_ids - done_ids)


# ---------------------------------------------------------------------------
# CLI for manual/Ollama-based extraction
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Extract concepts from individual episodes"
    )
    parser.add_argument("--conversations-dir", type=str, default=None)
    parser.add_argument("--state-file", type=str,
                        default="experiments/results/hierarchy/extraction_state.json")
    parser.add_argument("--model", type=str, default="llama3.2",
                        help="Ollama model for extraction")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max episodes to extract in this run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing state file")
    args = parser.parse_args()

    conv_dir = Path(args.conversations_dir) if args.conversations_dir else None
    state_path = Path(args.state_file)

    if args.resume and state_path.exists():
        state = ExtractionState.load(state_path)
        logger.info(f"Resumed: {state.total_extractions} done, "
                    f"{len(state.vocabulary)} concepts in vocabulary")
    else:
        state = ExtractionState()

    remaining = get_remaining_episodes(state, conv_dir)
    logger.info(f"Remaining: {len(remaining)} episodes")

    if not remaining:
        logger.info("All episodes extracted!")
        return

    # Ollama LLM function
    import requests
    def ollama_fn(prompt):
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": args.model, "prompt": prompt,
                   "stream": False, "options": {"temperature": 0.3}},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["response"]

    batch = remaining[:args.limit]
    for i, episode_id in enumerate(batch):
        try:
            result = extract_concepts_from_episode(
                episode_id, state, ollama_fn, conv_dir
            )
            print(f"[{i+1}/{len(batch)}] {result.title}: "
                  f"{result.concepts} ({result.extraction_time:.1f}s)")
        except Exception as e:
            logger.error(f"Failed {episode_id}: {e}")
            continue

        # Save periodically
        if (i + 1) % 10 == 0:
            state.save(state_path)

    state.save(state_path)
    print(f"\nDone: {state.total_extractions} episodes, "
          f"{len(state.vocabulary)} unique concepts")


if __name__ == "__main__":
    main()
