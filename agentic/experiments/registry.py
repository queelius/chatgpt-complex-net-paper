"""Experiment registry: all experiments in the Cognitive MRI research program.

Organized by batch:
  batch1: Threshold sweeps on existing embeddings (instant, no Ollama needed)
  batch2: ChatGPT weight variation experiments (from ablation study data)
  batch3: Agentic content mode experiments (need new embeddings)
  batch4: Concept extraction experiments (need LLM calls)
"""
from pathlib import Path
from experiments.runner import ExperimentConfig

REPO_ROOT = Path(__file__).parent.parent.parent
AGENTIC_DIR = Path(__file__).parent.parent
ABLATION_EDGES = REPO_ROOT / "dev" / "ablation_study" / "data" / "edges"
ABLATION_FILTERED = REPO_ROOT / "dev" / "ablation_study" / "data" / "edges_filtered"

# Standard thresholds for sweep experiments
THRESHOLD_SWEEP = [0.80, 0.85, 0.875, 0.90, 0.925, 0.95]

# ---------------------------------------------------------------------------
# Batch 1: Threshold sweeps on existing embeddings
# These use pre-computed edges and run in seconds
# ---------------------------------------------------------------------------
batch1 = [
    # Agentic: sweep the full similarity matrix
    ExperimentConfig(
        id="agentic-text-only-sweep",
        name="Agentic text-only threshold sweep",
        description="Sweep θ on existing agentic embeddings (text-only, user:ai=2:1)",
        dataset="agentic",
        edges_file=str(AGENTIC_DIR / "data" / "edges-all.json"),
        total_nodes=3945,
        thresholds=THRESHOLD_SWEEP,
        tags=["batch1", "agentic", "sweep"],
    ),
]

# ---------------------------------------------------------------------------
# Batch 2: ChatGPT weight variation experiments
# The ablation study already computed edges for 9 weight ratios — reuse them!
# ---------------------------------------------------------------------------
CHATGPT_WEIGHT_CONFIGS = [
    ("user1.0-ai1.0",   1.0,   1.0,  "Equal weight"),
    ("user1.0-ai1.618", 1.0,   1.618, "Golden ratio (AI-heavy)"),
    ("user1.0-ai2.0",   1.0,   2.0,  "AI 2x weight"),
    ("user1.0-ai4.0",   1.0,   4.0,  "AI 4x weight"),
    ("user1.0-ai100.0", 1.0,   100.0, "AI-only (effectively)"),
    ("user1.618-ai1.0", 1.618, 1.0,  "Golden ratio (user-heavy)"),
    ("user2.0-ai1.0",   2.0,   1.0,  "User 2x weight (Paper 1-2 optimal)"),
    ("user4.0-ai1.0",   4.0,   1.0,  "User 4x weight"),
    ("user100.0-ai1.0", 100.0, 1.0,  "User-only (effectively)"),
]

batch2 = []
for label, uw, aw, desc in CHATGPT_WEIGHT_CONFIGS:
    # Unfiltered edges file from ablation
    edges_path = ABLATION_EDGES / f"edges_chatgpt-json-llm-{label}.json"
    if edges_path.exists():
        batch2.append(ExperimentConfig(
            id=f"chatgpt-{label}-sweep",
            name=f"ChatGPT {desc} sweep",
            description=f"ChatGPT with {desc} (user={uw}, ai={aw}), threshold sweep",
            dataset="chatgpt",
            edges_file=str(edges_path),
            user_weight=uw,
            assistant_weight=aw,
            total_nodes=1908,
            thresholds=THRESHOLD_SWEEP,
            tags=["batch2", "chatgpt", "sweep", "weights"],
        ))

# Also run single-threshold experiments at θ=0.9 for quick comparison
batch2_single = []
for label, uw, aw, desc in CHATGPT_WEIGHT_CONFIGS:
    edges_path = ABLATION_EDGES / f"edges_chatgpt-json-llm-{label}.json"
    if edges_path.exists():
        batch2_single.append(ExperimentConfig(
            id=f"chatgpt-{label}-t0.9",
            name=f"ChatGPT {desc} at θ=0.9",
            description=f"ChatGPT with {desc} at optimal threshold",
            dataset="chatgpt",
            edges_file=str(edges_path),
            user_weight=uw,
            assistant_weight=aw,
            total_nodes=1908,
            thresholds=[0.9],
            tags=["batch2", "chatgpt", "weights", "quick"],
        ))

# ---------------------------------------------------------------------------
# Batch 3: Agentic content mode experiments
# These NEED new embeddings via Ollama (slow: ~1-2 hours each)
# Defined here as placeholders — edges_file will be set after embedding
# ---------------------------------------------------------------------------
batch3_placeholders = [
    ExperimentConfig(
        id="agentic-text-full",
        name="Agentic text-full",
        description="All content including tool names/results/thinking blocks",
        dataset="agentic",
        content_mode="text-full",
        total_nodes=3945,
        thresholds=[0.85, 0.90, 0.95],
        tags=["batch3", "agentic", "content-mode", "needs-embedding"],
    ),
    ExperimentConfig(
        id="agentic-user-only",
        name="Agentic user-only",
        description="Only human prompts — what did the user actually ask?",
        dataset="agentic",
        content_mode="user-only",
        total_nodes=3945,
        thresholds=[0.85, 0.90, 0.95],
        tags=["batch3", "agentic", "content-mode", "needs-embedding"],
    ),
    ExperimentConfig(
        id="agentic-tool-names-only",
        name="Agentic tool-names-only",
        description="Behavioral fingerprint: what tools were used? Network of work patterns.",
        dataset="agentic",
        content_mode="tool-names-only",
        total_nodes=3945,
        thresholds=[0.85, 0.90, 0.95],
        tags=["batch3", "agentic", "content-mode", "needs-embedding"],
    ),
    ExperimentConfig(
        id="agentic-assistant-only",
        name="Agentic assistant-only",
        description="Only Claude's responses — what did the AI produce?",
        dataset="agentic",
        content_mode="assistant-only",
        total_nodes=3945,
        thresholds=[0.85, 0.90, 0.95],
        tags=["batch3", "agentic", "content-mode", "needs-embedding"],
    ),
]

# ---------------------------------------------------------------------------
# Batch 4: Concept extraction & hierarchical abstraction
# These need LLM reasoning (not just embedding)
# ---------------------------------------------------------------------------
batch4_placeholders = [
    ExperimentConfig(
        id="chatgpt-concept-extraction",
        name="ChatGPT concept extraction",
        description="Extract abstract concepts from the 15 ChatGPT communities, "
                    "build concept→episode instantiation links",
        dataset="chatgpt",
        thresholds=[0.9],
        tags=["batch4", "chatgpt", "concepts", "needs-llm"],
    ),
    ExperimentConfig(
        id="chatgpt-recursive-abstraction",
        name="ChatGPT recursive abstraction",
        description="Recursively extract meta-concepts from concepts, building a "
                    "deep hierarchy: episodes → concepts → meta-concepts → ...",
        dataset="chatgpt",
        thresholds=[0.9],
        tags=["batch4", "chatgpt", "concepts", "hierarchy", "needs-llm"],
    ),
    ExperimentConfig(
        id="chatgpt-decomposition",
        name="ChatGPT conversation decomposition",
        description="Decompose conversations into atomic sub-topics/patterns, "
                    "making each conversation a parent of its constituents",
        dataset="chatgpt",
        thresholds=[0.9],
        tags=["batch4", "chatgpt", "decomposition", "needs-llm"],
    ),
]

# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------
EXPERIMENTS = batch1 + batch2_single + batch2 + batch3_placeholders + batch4_placeholders
