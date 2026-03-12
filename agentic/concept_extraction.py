"""Concept extraction from community clusters via LLM.

Builds on the core types from comp-net-2025-journal/code/ but integrates
with the agentic analysis pipeline. Extracts abstract concepts from
Louvain communities, builds concept→episode instantiation links,
and concept—concept association links.

Two data sources:
  - ChatGPT: 1908 conversations in dev/chatgpt-4-11-2025_json_no_embeddings/
  - Agentic: 3945 sessions in agentic/data/sessions-text-only/

LLM backend: Ollama (llama3.2 or larger) for local extraction.
"""

import json
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable

import networkx as nx
import numpy as np
import requests

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
AGENTIC_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConceptNode:
    """An abstract concept extracted from a community cluster."""
    id: str
    name: str
    description: str
    source_community: int
    example_episodes: list[str] = field(default_factory=list)

@dataclass
class ExtractionResult:
    """Result of concept extraction for one community."""
    community_id: int
    community_size: int
    concepts: list[ConceptNode]
    raw_response: str
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def load_edges_and_build_graph(
    edges_file: str,
    threshold: float = 0.9,
    parent_only_ids: Optional[set[str]] = None,
) -> nx.Graph:
    """Load edge file and build thresholded graph.

    Args:
        edges_file: Path to JSON edge file (list of [id1, id2, sim] or multi-line)
        threshold: Similarity threshold
        parent_only_ids: If set, only include edges between these node IDs
    """
    path = Path(edges_file)
    G = nx.Graph()

    with open(path, "r") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            # Try loading as JSON array
            try:
                edges = json.load(f)
                for e in edges:
                    if len(e) >= 3 and e[2] >= threshold:
                        s, t = str(e[0]), str(e[1])
                        if parent_only_ids and (s not in parent_only_ids or t not in parent_only_ids):
                            continue
                        G.add_edge(s, t, weight=e[2])
            except json.JSONDecodeError:
                # Multi-line format: parse line-by-line
                f.seek(0)
                _load_multiline_edges(f, G, threshold, parent_only_ids)
        else:
            _load_multiline_edges(f, G, threshold, parent_only_ids)

    return G


def _load_multiline_edges(f, G, threshold, parent_only_ids):
    """Parse multi-line pretty-printed JSON edge file."""
    buf = []
    for line in f:
        line = line.strip().rstrip(",")
        if not line or line in ("[]", "[", "]"):
            continue
        if line.startswith('"'):
            buf.append(line.strip('"'))
        elif line[0].isdigit() or line[0] == "-":
            try:
                buf.append(float(line))
            except ValueError:
                pass
        elif line == "[":
            buf = []
        if len(buf) == 3:
            s, t, w = str(buf[0]), str(buf[1]), buf[2]
            if w >= threshold:
                if parent_only_ids is None or (s in parent_only_ids and t in parent_only_ids):
                    G.add_edge(s, t, weight=w)
            buf = []


def detect_communities(G: nx.Graph, random_state: int = 42) -> dict[str, int]:
    """Run Louvain community detection. Returns node→community_id mapping."""
    import community.community_louvain as louvain
    partition = louvain.best_partition(G, random_state=random_state)
    return partition


def get_community_members(partition: dict[str, int]) -> dict[int, list[str]]:
    """Invert partition map: community_id → list of node IDs."""
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)
    return communities


# ---------------------------------------------------------------------------
# Session content loading
# ---------------------------------------------------------------------------

def load_chatgpt_sessions(
    data_dir: str,
    node_ids: list[str],
) -> dict[str, dict]:
    """Load ChatGPT conversation JSONs. Node IDs are filename slugs (stem)."""
    data_path = Path(data_dir)
    sessions = {}
    id_set = set(node_ids)

    for fpath in data_path.glob("*.json"):
        slug = fpath.stem
        if slug not in id_set:
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            msgs = doc.get("messages", [])
            title = doc.get("title", slug)
            content_parts = []
            for m in msgs[:6]:  # First 6 messages for context
                role = m.get("role", "")
                text = m.get("content", "")
                if text and isinstance(text, str):
                    content_parts.append(f"[{role}] {text[:300]}")
            sessions[slug] = {
                "title": title,
                "content": "\n".join(content_parts),
                "message_count": len(msgs),
            }
        except Exception:
            continue

    return sessions


def load_agentic_sessions(
    data_dir: str,
    node_ids: list[str],
) -> dict[str, dict]:
    """Load agentic session JSONs. Node IDs use ':' but filenames use '_'."""
    data_path = Path(data_dir)
    sessions = {}

    # Map filename-safe IDs back to original IDs
    id_set = set(node_ids)
    safe_to_original = {}
    for nid in node_ids:
        safe = nid.replace(":", "_")
        safe_to_original[safe] = nid

    for fpath in data_path.glob("*.json"):
        stem = fpath.stem
        original_id = safe_to_original.get(stem)
        if original_id is None:
            # Try direct match
            if stem in id_set:
                original_id = stem
            else:
                continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                doc = json.load(f)
            msgs = doc.get("messages", [])
            meta = doc.get("metadata", {})
            title = meta.get("title", fpath.stem)

            content_parts = []
            for m in msgs[:8]:
                text = m.get("content", "")
                role = m.get("role", "")
                if text and isinstance(text, str):
                    content_parts.append(f"[{role}] {text[:300]}")

            sessions[original_id] = {
                "title": title,
                "content": "\n".join(content_parts),
                "message_count": len(msgs),
            }
        except Exception:
            continue

    return sessions


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str, model: str = "llama3.2", temperature: float = 0.3) -> str:
    """Call Ollama API for text generation."""
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["response"]


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------

def build_extraction_prompt(
    sessions: list[dict],
    community_id: int,
    max_concepts: int = 5,
) -> str:
    """Build LLM prompt for concept extraction from a community."""
    episode_lines = []
    for s in sessions:
        title = s.get("title", "Untitled")
        content = s.get("content", "")
        excerpt = content[:400] + "..." if len(content) > 400 else content
        episode_lines.append(f"- **{title}**\n  {excerpt}")

    episodes_text = "\n".join(episode_lines)

    return f"""Here are {len(sessions)} AI conversations/sessions from a single user that cluster together in a semantic similarity network (Community {community_id}):

{episodes_text}

Identify 2-{max_concepts} abstract concepts, principles, or cognitive patterns that these conversations share. Focus on:
- Underlying intellectual themes (not just surface topics)
- Problem-solving approaches or reasoning patterns
- Knowledge structures that connect these conversations
- Be specific: "Bayesian inference as model selection" is better than "statistics"

Return ONLY a JSON array:
[
  {{
    "name": "Short concept name (2-5 words)",
    "description": "One sentence: what the concept is and how it manifests in these conversations",
    "example_episodes": ["title1", "title2"]
  }}
]"""


def parse_concepts(
    response: str,
    community_id: int,
) -> list[ConceptNode]:
    """Parse LLM response into ConceptNode objects."""
    # Extract JSON array from response
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        logger.warning(f"No JSON array in response for community {community_id}")
        return []

    try:
        items = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for community {community_id}: {e}")
        return []

    concepts = []
    for item in items:
        name = item.get("name", "").strip()
        desc = item.get("description", "").strip()
        examples = item.get("example_episodes", [])
        if not name or not desc:
            continue

        cid = f"C{community_id}_{hashlib.md5(name.lower().encode()).hexdigest()[:6]}"
        concepts.append(ConceptNode(
            id=cid,
            name=name,
            description=desc,
            source_community=community_id,
            example_episodes=examples,
        ))

    return concepts


def extract_concepts_for_community(
    community_id: int,
    sessions: list[dict],
    llm_fn: Callable[[str], str],
    max_concepts: int = 5,
    max_sessions: int = 20,
) -> ExtractionResult:
    """Extract concepts from a single community.

    If the community has more than max_sessions members, sample by
    preferring sessions with more messages (deeper = more informative).
    """
    # Sample if community is too large
    if len(sessions) > max_sessions:
        # Sort by message count descending, take top max_sessions
        sessions = sorted(sessions, key=lambda s: s.get("message_count", 0), reverse=True)[:max_sessions]

    prompt = build_extraction_prompt(sessions, community_id, max_concepts)

    t0 = time.time()
    response = llm_fn(prompt)
    elapsed = time.time() - t0

    concepts = parse_concepts(response, community_id)

    return ExtractionResult(
        community_id=community_id,
        community_size=len(sessions),
        concepts=concepts,
        raw_response=response,
        elapsed_seconds=round(elapsed, 1),
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_concept_extraction(
    dataset: str,
    edges_file: str,
    sessions_dir: str,
    threshold: float = 0.9,
    parent_only_ids: Optional[set[str]] = None,
    llm_model: str = "llama3.2",
    min_community_size: int = 3,
    max_concepts: int = 5,
    output_file: Optional[str] = None,
) -> dict:
    """Run the full concept extraction pipeline.

    Returns dict with communities, concepts, and graph metadata.
    """
    logger.info(f"Loading edges from {edges_file} (θ={threshold})")
    G = load_edges_and_build_graph(edges_file, threshold, parent_only_ids)
    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Community detection
    logger.info("Running Louvain community detection...")
    partition = detect_communities(G)
    communities = get_community_members(partition)
    logger.info(f"Found {len(communities)} communities")

    # Filter to communities above minimum size
    large_communities = {
        cid: members for cid, members in communities.items()
        if len(members) >= min_community_size
    }
    logger.info(f"{len(large_communities)} communities with ≥{min_community_size} members")

    # Load session content
    node_ids = list(G.nodes())
    if dataset == "chatgpt":
        chatgpt_dir = str(REPO_ROOT / "dev" / "chatgpt-4-11-2025_json_no_embeddings")
        sessions = load_chatgpt_sessions(chatgpt_dir, node_ids)
    else:
        sessions = load_agentic_sessions(sessions_dir, node_ids)
    logger.info(f"Loaded {len(sessions)} session documents")

    # LLM function
    def llm_fn(prompt):
        return ollama_generate(prompt, model=llm_model)

    # Extract concepts from each community
    all_results = []
    all_concepts = []

    for cid in sorted(large_communities.keys(), key=lambda c: len(large_communities[c]), reverse=True):
        members = large_communities[cid]
        community_sessions = [
            sessions[nid] for nid in members if nid in sessions
        ]
        if len(community_sessions) < min_community_size:
            continue

        logger.info(f"Community {cid}: {len(community_sessions)} sessions")
        try:
            result = extract_concepts_for_community(
                cid, community_sessions, llm_fn, max_concepts=max_concepts
            )
            all_results.append(result)
            all_concepts.extend(result.concepts)

            # Print progress
            concept_names = [c.name for c in result.concepts]
            print(f"  C{cid} ({len(community_sessions)} sessions, {result.elapsed_seconds:.1f}s): "
                  f"{concept_names}")
        except Exception as e:
            logger.error(f"  C{cid}: extraction failed: {e}")
            continue

    # Build concept graph edges
    concept_episodes = {}  # concept_id → set of episode titles
    for c in all_concepts:
        concept_episodes[c.id] = set(c.example_episodes)

    # C—C association links via shared episodes
    associations = []
    concept_ids = list(concept_episodes.keys())
    for i, c1 in enumerate(concept_ids):
        for c2 in concept_ids[i+1:]:
            shared = concept_episodes[c1] & concept_episodes[c2]
            if shared:
                union = concept_episodes[c1] | concept_episodes[c2]
                jaccard = len(shared) / len(union) if union else 0
                associations.append({
                    "concept1": c1,
                    "concept2": c2,
                    "shared_episodes": list(shared),
                    "jaccard": round(jaccard, 3),
                })

    # Assemble output
    output = {
        "dataset": dataset,
        "edges_file": str(edges_file),
        "threshold": threshold,
        "llm_model": llm_model,
        "graph": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": len(communities),
            "communities_extracted": len(all_results),
        },
        "concepts": [asdict(c) for c in all_concepts],
        "concept_count": len(all_concepts),
        "associations": associations,
        "communities": {
            str(r.community_id): {
                "size": r.community_size,
                "concepts": [asdict(c) for c in r.concepts],
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in all_results
        },
        "extraction_results": [
            {
                "community_id": r.community_id,
                "community_size": r.community_size,
                "concept_count": len(r.concepts),
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in all_results
        ],
    }

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Extract concepts from community clusters")
    parser.add_argument("--dataset", choices=["chatgpt", "agentic"], required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--min-community-size", type=int, default=3)
    parser.add_argument("--max-concepts", type=int, default=5)
    parser.add_argument("--parent-only", action="store_true",
                        help="For agentic: only use parent sessions")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Determine edges file and sessions dir
    if args.dataset == "chatgpt":
        edges_file = str(REPO_ROOT / "data" / "network" / "edges_user2.0-ai1.0_t0.9.json")
        sessions_dir = str(REPO_ROOT / "dev" / "chatgpt-4-11-2025_json_no_embeddings")
        parent_ids = None
    else:
        edges_file = str(AGENTIC_DIR / "data" / "edges-all.json")
        sessions_dir = str(AGENTIC_DIR / "data" / "sessions-text-only")
        parent_ids = None
        if args.parent_only:
            # Identify parent sessions (no "_agent-" in filename)
            parent_ids = set()
            for f in Path(sessions_dir).glob("*.json"):
                if "_agent-" not in f.stem:
                    # Convert filename-safe ID back to original
                    parent_ids.add(f.stem.replace("_", ":"))
            logger.info(f"Parent-only mode: {len(parent_ids)} parent sessions")

    output_file = args.output or str(
        AGENTIC_DIR / "experiments" / "results" / f"concepts_{args.dataset}_t{args.threshold}.json"
    )

    result = run_concept_extraction(
        dataset=args.dataset,
        edges_file=edges_file,
        sessions_dir=sessions_dir,
        threshold=args.threshold,
        parent_only_ids=parent_ids,
        llm_model=args.model,
        min_community_size=args.min_community_size,
        max_concepts=args.max_concepts,
        output_file=output_file,
    )

    print(f"\n{'='*60}")
    print(f"Concept extraction complete: {args.dataset}")
    print(f"  Communities extracted: {result['graph']['communities_extracted']}")
    print(f"  Total concepts: {result['concept_count']}")
    print(f"  Cross-community associations: {len(result['associations'])}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
