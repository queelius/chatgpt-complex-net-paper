"""Concept-level network experiments (F48-F51).

Runs four analyses on the concept network:
  F48: Guimerà-Amaral role classification
  F49: Recursive abstraction convergence (L2→L3 robustness)
  F50: Hierarchical multiplex E—E + C→E + C—C
  F51: Consensus concept extraction (N=5 semantic clustering)
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_concept_network(threshold=0.8):
    """Build concept network from similarity matrix at given threshold."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    concepts = meta["concepts"]

    G = nx.Graph()
    for c in concepts:
        G.add_node(c["index"], name=c["name"], platform=c["platform"],
                    source_community=c["source_community"], id=c["id"])

    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim[i, j]))

    return G, concepts, sim


def louvain_partition(G):
    """Run Louvain community detection."""
    import community.community_louvain as community_louvain
    return community_louvain.best_partition(G, random_state=42)


# ─── F48: Guimerà-Amaral Roles ───────────────────────────────────────────

def within_module_degree_z(G, partition):
    """Compute within-module degree z-score for each node."""
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)

    z_scores = {}
    for comm, members in communities.items():
        subgraph = G.subgraph(members)
        degrees = {n: subgraph.degree(n) for n in members}
        vals = list(degrees.values())
        mu = np.mean(vals)
        sigma = np.std(vals)
        for n in members:
            z_scores[n] = (degrees[n] - mu) / sigma if sigma > 0 else 0.0
    return z_scores


def participation_coefficient(G, partition):
    """Compute participation coefficient P_i for each node.

    P_i = 1 - Σ_s (k_is / k_i)^2
    where k_is = edges to community s, k_i = total degree.
    """
    p_coeff = {}
    for node in G.nodes():
        ki = G.degree(node)
        if ki == 0:
            p_coeff[node] = 0.0
            continue
        # Count edges to each community
        comm_edges = Counter()
        for neighbor in G.neighbors(node):
            comm_edges[partition[neighbor]] += 1
        p_coeff[node] = 1.0 - sum((kis / ki) ** 2 for kis in comm_edges.values())
    return p_coeff


def classify_role(z, p):
    """Guimerà-Amaral role classification.

    Non-hubs (z < 2.5):
      R1: ultra-peripheral (P ≤ 0.05)
      R2: peripheral (0.05 < P ≤ 0.62)
      R3: non-hub connector (0.62 < P ≤ 0.80)
      R4: non-hub kinless (P > 0.80)

    Hubs (z ≥ 2.5):
      R5: provincial hub (P ≤ 0.30)
      R6: connector hub (0.30 < P ≤ 0.75)
      R7: kinless hub (P > 0.75)
    """
    if z < 2.5:
        if p <= 0.05:
            return "R1-ultra-peripheral"
        elif p <= 0.62:
            return "R2-peripheral"
        elif p <= 0.80:
            return "R3-connector"
        else:
            return "R4-kinless"
    else:
        if p <= 0.30:
            return "R5-provincial-hub"
        elif p <= 0.75:
            return "R6-connector-hub"
        else:
            return "R7-kinless-hub"


def run_guimera_amaral(threshold=0.8):
    """F48: Classify concept network nodes into Guimerà-Amaral roles."""
    print("\n" + "=" * 60)
    print("F48: Guimerà-Amaral Role Classification (Concept Network)")
    print("=" * 60)

    G, concepts, sim = load_concept_network(threshold)
    partition = louvain_partition(G)
    z_scores = within_module_degree_z(G, partition)
    p_coeffs = participation_coefficient(G, partition)

    # Classify each node
    roles = {}
    node_details = []
    for node in G.nodes():
        z = z_scores[node]
        p = p_coeffs[node]
        role = classify_role(z, p)
        roles[node] = role
        c = concepts[node]
        node_details.append({
            "name": c["name"],
            "platform": c["platform"],
            "source_community": c["source_community"],
            "degree": G.degree(node),
            "z_score": round(z, 3),
            "participation": round(p, 3),
            "role": role,
            "louvain_community": partition[node],
        })

    # Summary by role
    role_counts = Counter(roles.values())
    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(set(partition.values()))} communities")
    print(f"\nRole distribution:")
    for role in sorted(role_counts):
        print(f"  {role}: {role_counts[role]}")

    # By platform
    platform_roles = defaultdict(lambda: Counter())
    for node, role in roles.items():
        platform = concepts[node]["platform"]
        platform_roles[platform][role] += 1

    print(f"\nBy platform:")
    for platform in ["chatgpt", "agentic"]:
        print(f"  {platform}:")
        for role in sorted(platform_roles[platform]):
            print(f"    {role}: {platform_roles[platform][role]}")

    # Cross-platform edge fraction by role
    role_cross_platform = defaultdict(lambda: {"total": 0, "cross": 0})
    for u, v in G.edges():
        r = roles[u]  # focus on source node role
        role_cross_platform[r]["total"] += 1
        if concepts[u]["platform"] != concepts[v]["platform"]:
            role_cross_platform[r]["cross"] += 1

    print(f"\nCross-platform edge fraction by role:")
    for role in sorted(role_cross_platform):
        d = role_cross_platform[role]
        frac = d["cross"] / d["total"] if d["total"] > 0 else 0
        print(f"  {role}: {d['cross']}/{d['total']} = {frac:.1%}")

    # Top connectors and hubs
    connectors = [d for d in node_details if "connector" in d["role"] or "hub" in d["role"]]
    connectors.sort(key=lambda x: (-x["participation"], -x["degree"]))
    print(f"\nTop connectors/hubs ({len(connectors)}):")
    for c in connectors[:10]:
        print(f"  [{c['platform'][:2]}] {c['name']}: "
              f"role={c['role']}, deg={c['degree']}, z={c['z_score']:.2f}, P={c['participation']:.3f}")

    # Save results
    result = {
        "experiment": "F48: Concept-Level Guimerà-Amaral Roles",
        "threshold": threshold,
        "network": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": len(set(partition.values())),
        },
        "role_distribution": dict(role_counts),
        "platform_roles": {p: dict(c) for p, c in platform_roles.items()},
        "cross_platform_by_role": {r: d for r, d in role_cross_platform.items()},
        "node_details": sorted(node_details, key=lambda x: (-x["degree"], -x["participation"])),
    }
    with open(RESULTS_DIR / "concept_guimera_amaral_roles.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_guimera_amaral_roles.json")
    return result


# ─── F49: Recursive Abstraction Convergence ──────────────────────────────

def ollama_generate(prompt, model="llama3.2", timeout=300):
    """Call Ollama API."""
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_meta_concepts(concept_names, level_from, level_to, run_id=0):
    """Run one L(n)→L(n+1) abstraction via LLM."""
    names_str = ", ".join(concept_names)
    prompt = f"""These are Level-{level_from} concepts extracted from a knowledge archive:
{names_str}

Identify {max(2, len(concept_names) // 2)}-{max(3, len(concept_names) // 2 + 1)} higher-level themes (Level-{level_to}) that group these concepts.

Return ONLY a JSON array: [{{"name": "theme name", "members": ["concept1", "concept2", ...]}}]"""

    response = ollama_generate(prompt)
    # Parse JSON from response
    import re
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if not match:
        return []
    try:
        themes = json.loads(match.group())
        return [t for t in themes if isinstance(t, dict) and "name" in t]
    except json.JSONDecodeError:
        return []


def run_abstraction_convergence(n_runs=5):
    """F49: Test if L2→L3 compression ratio of 2.0× is robust."""
    print("\n" + "=" * 60)
    print("F49: Recursive Abstraction Convergence")
    print("=" * 60)

    # Load existing hierarchies
    with open(RESULTS_DIR / "concepts_chatgpt_hierarchy.json") as f:
        chatgpt_hier = json.load(f)
    with open(RESULTS_DIR / "concepts_agentic_hierarchy.json") as f:
        agentic_hier = json.load(f)

    results = {"chatgpt": [], "agentic": []}

    for platform, hier in [("chatgpt", chatgpt_hier), ("agentic", agentic_hier)]:
        l2_names = [t["name"] for t in hier["level_2_meta_concepts"]]
        l2_count = len(l2_names)
        print(f"\n{platform}: {l2_count} L2 meta-concepts → L3")

        for run in range(n_runs):
            try:
                l3 = extract_meta_concepts(l2_names, 2, 3, run_id=run)
                l3_count = len(l3)
                ratio = l2_count / l3_count if l3_count > 0 else float("inf")
                l3_names = [t["name"] for t in l3]
                print(f"  Run {run+1}: {l3_count} L3 themes (ratio={ratio:.1f}×) — {l3_names}")
                results[platform].append({
                    "run": run + 1,
                    "l2_count": l2_count,
                    "l3_count": l3_count,
                    "compression_ratio": round(ratio, 2),
                    "l3_themes": l3_names,
                })
            except Exception as e:
                print(f"  Run {run+1}: ERROR — {e}")
                results[platform].append({"run": run + 1, "error": str(e)})

        # Also try L3→L4 to test termination
        l3_sample = results[platform][0].get("l3_themes", [])
        if len(l3_sample) >= 2:
            print(f"\n  L3→L4 test ({len(l3_sample)} themes)...")
            try:
                l4 = extract_meta_concepts(l3_sample, 3, 4)
                l4_count = len(l4)
                l4_names = [t["name"] for t in l4]
                ratio34 = len(l3_sample) / l4_count if l4_count > 0 else float("inf")
                print(f"  L4: {l4_count} themes (ratio={ratio34:.1f}×) — {l4_names}")
                results[platform].append({
                    "run": "L3→L4",
                    "l3_count": len(l3_sample),
                    "l4_count": l4_count,
                    "compression_ratio": round(ratio34, 2),
                    "l4_themes": l4_names,
                })
            except Exception as e:
                print(f"  L3→L4: ERROR — {e}")

    # Summary statistics
    print("\n" + "-" * 40)
    print("Summary:")
    for platform in ["chatgpt", "agentic"]:
        ratios = [r["compression_ratio"] for r in results[platform]
                  if isinstance(r.get("run"), int) and "compression_ratio" in r]
        if ratios:
            print(f"  {platform} L2→L3 compression: "
                  f"mean={np.mean(ratios):.2f}×, std={np.std(ratios):.2f}, "
                  f"range=[{min(ratios):.1f}×, {max(ratios):.1f}×]")

    output = {
        "experiment": "F49: Recursive Abstraction Convergence",
        "n_runs": n_runs,
        "results": results,
    }
    with open(RESULTS_DIR / "abstraction_convergence.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to abstraction_convergence.json")
    return output


# ─── F50: Hierarchical Multiplex ─────────────────────────────────────────

def load_episode_edges(platform):
    """Load episode-level similarity edges."""
    if platform == "chatgpt":
        path = Path(__file__).parent.parent.parent / "data" / "network" / "edges-chatgpt-2to1-t0.9.json"
        if not path.exists():
            # Try dev path
            path = Path(__file__).parent.parent.parent / "dev" / "ablation_study" / "data" / "edges" / "edges_chatgpt-json-llm-user2.0-ai1.0.json"
    else:
        path = Path(__file__).parent.parent / "data" / "edges-t0.9.json"
        if not path.exists():
            path = Path(__file__).parent.parent / "data" / "agentic-edges-text-only-unfiltered.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        edges = json.load(f)
    return edges, str(path)


def _rebuild_episode_communities(edges_path, threshold):
    """Rebuild Louvain partition from edge list to get community→episode mapping."""
    import community.community_louvain as community_louvain
    with open(edges_path) as f:
        edges = json.load(f)
    G = nx.Graph()
    for src, dst, w in edges:
        if w >= threshold:
            G.add_edge(src, dst, weight=w)
    partition = community_louvain.best_partition(G, random_state=42)
    comm_to_episodes = defaultdict(list)
    for ep, comm in partition.items():
        comm_to_episodes[comm].append(ep)
    return G, partition, comm_to_episodes


def run_hierarchical_multiplex():
    """F50: Build and analyze the E—E + C→E + C—C multiplex."""
    print("\n" + "=" * 60)
    print("F50: Hierarchical Multiplex Network")
    print("=" * 60)

    # Use θ=0.70 for C—C to get meaningful community structure
    G_concept, concepts, sim = load_concept_network(threshold=0.70)

    # Rebuild episode-level communities to get full C→E mapping
    chatgpt_edges_path = (RESULTS_DIR.parent.parent.parent /
                          "data" / "network" / "edges_user2.0-ai1.0_t0.9.json")
    agentic_edges_path = RESULTS_DIR.parent.parent / "data" / "edges-all.json"

    chatgpt_ee, chatgpt_part, chatgpt_comm2ep = _rebuild_episode_communities(
        chatgpt_edges_path, 0.9)
    print(f"ChatGPT E—E: {chatgpt_ee.number_of_nodes()} nodes, "
          f"{chatgpt_ee.number_of_edges()} edges, "
          f"{len(set(chatgpt_part.values()))} communities")

    agentic_ee, agentic_part, agentic_comm2ep = None, None, None
    if agentic_edges_path.exists():
        agentic_ee, agentic_part, agentic_comm2ep = _rebuild_episode_communities(
            agentic_edges_path, 0.95)
        print(f"Agentic E—E: {agentic_ee.number_of_nodes()} nodes, "
              f"{agentic_ee.number_of_edges()} edges, "
              f"{len(set(agentic_part.values()))} communities")

    # Build C→E mapping: each concept links to ALL episodes in its source community
    concept_to_episodes = {}
    for c in concepts:
        comm_id = c["source_community"]
        if c["platform"] == "chatgpt":
            episodes = chatgpt_comm2ep.get(comm_id, [])
        elif agentic_comm2ep is not None:
            episodes = agentic_comm2ep.get(comm_id, [])
        else:
            episodes = c.get("example_episodes", [])
        concept_to_episodes[c["index"]] = {
            "concept": c["name"],
            "platform": c["platform"],
            "episodes": episodes,
            "source_community": comm_id,
        }

    # Build C→E bipartite graph
    G_ce = nx.DiGraph()
    total_ce_edges = 0
    for idx, mapping in concept_to_episodes.items():
        c_node = f"C_{idx}"
        G_ce.add_node(c_node, type="concept", name=mapping["concept"],
                       platform=mapping["platform"])
        for ep in mapping["episodes"]:
            e_node = f"E_{ep}"
            G_ce.add_node(e_node, type="episode")
            G_ce.add_edge(c_node, e_node)
            total_ce_edges += 1

    print(f"\nC→E bipartite: {G_ce.number_of_nodes()} nodes, {total_ce_edges} edges")
    concept_nodes = [n for n, d in G_ce.nodes(data=True) if d.get("type") == "concept"]
    episode_nodes = [n for n, d in G_ce.nodes(data=True) if d.get("type") == "episode"]
    print(f"  Concepts: {len(concept_nodes)}, Episodes: {len(episode_nodes)}")

    # C—C layer stats
    partition = louvain_partition(G_concept)
    n_communities = len(set(partition.values()))
    print(f"\nC—C layer: {G_concept.number_of_nodes()} nodes, "
          f"{G_concept.number_of_edges()} edges, {n_communities} communities")

    # Compute fan-out: how many episodes each concept links to
    fanouts = [G_ce.out_degree(n) for n in concept_nodes]
    print(f"\nC→E fan-out: mean={np.mean(fanouts):.1f}, "
          f"median={np.median(fanouts):.0f}, max={max(fanouts)}")

    # Concept participation across episode communities
    # For concepts linked to episodes from multiple E-E communities, compute diversity
    concept_episode_community_diversity = {}
    for idx, mapping in concept_to_episodes.items():
        ep_count = len(mapping["episodes"])
        concept_episode_community_diversity[idx] = {
            "name": mapping["concept"],
            "platform": mapping["platform"],
            "n_episodes": ep_count,
        }

    # Interlayer degree correlation: concept C—C degree vs C→E degree
    cc_degrees = []
    ce_degrees = []
    for idx in range(len(concepts)):
        cc_deg = G_concept.degree(idx)
        ce_deg = len(concept_to_episodes.get(idx, {}).get("episodes", []))
        cc_degrees.append(cc_deg)
        ce_degrees.append(ce_deg)

    if len(cc_degrees) > 2:
        corr, p_val = stats.spearmanr(cc_degrees, ce_degrees)
        print(f"\nInterlayer degree correlation (C—C vs C→E): "
              f"ρ={corr:.3f}, p={p_val:.4f}")
    else:
        corr, p_val = None, None

    # Platform mixing in communities
    comm_platform_mix = defaultdict(lambda: Counter())
    for node, comm in partition.items():
        comm_platform_mix[comm][concepts[node]["platform"]] += 1

    mixed_communities = 0
    for comm, counts in comm_platform_mix.items():
        if len(counts) > 1:
            mixed_communities += 1

    print(f"\nCommunity platform mixing: {mixed_communities}/{n_communities} "
          f"mixed ({mixed_communities/n_communities:.0%})")
    for comm in sorted(comm_platform_mix):
        counts = comm_platform_mix[comm]
        total = sum(counts.values())
        if total >= 3:
            mix_str = ", ".join(f"{p}:{n}" for p, n in counts.items())
            print(f"  Community {comm} ({total}): {mix_str}")

    # Multiplex participation: for each concept, what fraction of its
    # connections are C—C vs C→E?
    multiplex_participation = {}
    for idx in range(len(concepts)):
        cc_deg = G_concept.degree(idx)
        ce_deg = len(concept_to_episodes.get(idx, {}).get("episodes", []))
        total = cc_deg + ce_deg
        if total > 0:
            p = 1.0 - (cc_deg / total) ** 2 - (ce_deg / total) ** 2
        else:
            p = 0.0
        multiplex_participation[idx] = {
            "name": concepts[idx]["name"],
            "platform": concepts[idx]["platform"],
            "cc_degree": cc_deg,
            "ce_degree": ce_deg,
            "participation": round(p, 3),
        }

    # Top multiplex participants
    top_mp = sorted(multiplex_participation.values(),
                    key=lambda x: -x["participation"])
    print(f"\nTop multiplex participants (balanced C—C and C→E):")
    for t in top_mp[:8]:
        print(f"  [{t['platform'][:2]}] {t['name']}: "
              f"P={t['participation']:.3f} (CC={t['cc_degree']}, CE={t['ce_degree']})")

    result = {
        "experiment": "F50: Hierarchical Multiplex E—E + C→E + C—C",
        "layers": {
            "CC": {"nodes": G_concept.number_of_nodes(),
                   "edges": G_concept.number_of_edges(),
                   "communities": n_communities},
            "CE": {"concept_nodes": len(concept_nodes),
                   "episode_nodes": len(episode_nodes),
                   "edges": total_ce_edges},
        },
        "interlayer_degree_correlation": {
            "layers": "C—C vs C→E",
            "spearman_rho": round(corr, 4) if corr is not None else None,
            "p_value": round(p_val, 6) if p_val is not None else None,
        },
        "ce_fanout_stats": {
            "mean": round(float(np.mean(fanouts)), 1),
            "median": float(np.median(fanouts)),
            "max": int(max(fanouts)),
            "min": int(min(fanouts)),
        },
        "community_platform_mixing": {
            "total_communities": n_communities,
            "mixed_communities": mixed_communities,
            "fraction_mixed": round(mixed_communities / n_communities, 3),
        },
        "multiplex_participation": sorted(
            list(multiplex_participation.values()),
            key=lambda x: -x["participation"]
        ),
    }
    with open(RESULTS_DIR / "hierarchical_multiplex.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to hierarchical_multiplex.json")
    return result


# ─── F51: Consensus Concept Extraction ────────────────────────────────────

def run_consensus_extraction(n_runs=5, consensus_threshold=0.85, min_agreement=3):
    """F51: Extract concepts N times, cluster semantically, keep consensus."""
    print("\n" + "=" * 60)
    print(f"F51: Consensus Concept Extraction (N={n_runs}, "
          f"θ={consensus_threshold}, min={min_agreement}/{n_runs})")
    print("=" * 60)

    # Rebuild ChatGPT episode communities and pick top 3 by size
    chatgpt_edges_path = (RESULTS_DIR.parent.parent.parent /
                          "data" / "network" / "edges_user2.0-ai1.0_t0.9.json")
    _, _, comm2ep = _rebuild_episode_communities(chatgpt_edges_path, 0.9)

    # Sort communities by size, take top 3 with ≥30 episodes
    sorted_comms = sorted(comm2ep.items(), key=lambda x: -len(x[1]))
    communities_to_test = [(cid, eps) for cid, eps in sorted_comms if len(eps) >= 30][:3]

    if not communities_to_test:
        print("  No suitable communities found")
        return None

    all_results = []

    for comm_id, episodes_full in communities_to_test:
        sessions = episodes_full[:20]  # Sample top 20
        print(f"\nCommunity {comm_id} ({len(episodes_full)} episodes, using {len(sessions)}): ", end="")

        # Use episode slugs as session titles
        session_summaries = sessions

        titles_str = "\n".join(f"- {t}" for t in session_summaries)
        prompt = f"""These are {len(sessions)} AI conversation sessions from a knowledge community.

Sessions:
{titles_str}

Identify 4-6 abstract concepts or cognitive patterns that characterize this community of conversations.

Return ONLY a JSON array: [{{"name": "concept name", "description": "brief description"}}]"""

        # Run N extractions
        run_concepts = []
        for run in range(n_runs):
            try:
                response = ollama_generate(prompt)
                import re
                match = re.search(r'\[.*\]', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    names = [c["name"] for c in parsed if isinstance(c, dict) and c.get("name")]
                    run_concepts.append(names)
                    print(f"{len(names)}", end=" ")
                else:
                    run_concepts.append([])
                    print("0", end=" ")
            except Exception as e:
                run_concepts.append([])
                print("E", end=" ")

        print()

        # Flatten all concept names
        all_names = []
        name_to_runs = defaultdict(set)
        for run_idx, names in enumerate(run_concepts):
            for name in names:
                all_names.append(name)
                name_to_runs[name.lower().strip()].add(run_idx)

        # Exact-match consensus
        exact_consensus = [name for name, runs in name_to_runs.items()
                          if len(runs) >= min_agreement]
        print(f"  Exact-match consensus (≥{min_agreement}/{n_runs}): "
              f"{len(exact_consensus)} — {exact_consensus}")

        # Semantic clustering: embed all unique concept names, cluster by similarity
        unique_names = list(set(all_names))
        if len(unique_names) >= 2:
            import requests
            # Embed all concept names
            embeddings = []
            for name in unique_names:
                try:
                    resp = requests.post(
                        "http://localhost:11434/api/embed",
                        json={"model": "nomic-embed-text", "input": name},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    embeddings.append(resp.json()["embeddings"][0])
                except Exception:
                    embeddings.append(None)

            # Filter out failures
            valid = [(n, e) for n, e in zip(unique_names, embeddings) if e is not None]
            if len(valid) >= 2:
                valid_names, valid_embs = zip(*valid)
                emb_matrix = np.array(valid_embs)
                # Normalize
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                emb_matrix = emb_matrix / norms
                # Cosine similarity
                sim_matrix = emb_matrix @ emb_matrix.T

                # Greedy clustering at consensus_threshold
                clusters = []
                assigned = set()
                for i in range(len(valid_names)):
                    if i in assigned:
                        continue
                    cluster = [i]
                    assigned.add(i)
                    for j in range(i + 1, len(valid_names)):
                        if j not in assigned and sim_matrix[i, j] >= consensus_threshold:
                            cluster.append(j)
                            assigned.add(j)
                    clusters.append(cluster)

                # For each cluster, check how many runs contributed
                semantic_consensus = []
                for cluster in clusters:
                    cluster_names = [valid_names[i] for i in cluster]
                    contributing_runs = set()
                    for name in cluster_names:
                        for run_idx, run_names in enumerate(run_concepts):
                            if name in run_names:
                                contributing_runs.add(run_idx)
                    if len(contributing_runs) >= min_agreement:
                        # Pick the most common name as representative
                        name_counts = Counter()
                        for name in cluster_names:
                            name_counts[name] += 1
                        representative = name_counts.most_common(1)[0][0]
                        semantic_consensus.append({
                            "representative": representative,
                            "variants": cluster_names,
                            "n_runs": len(contributing_runs),
                            "n_variants": len(cluster_names),
                        })

                print(f"  Semantic consensus (θ≥{consensus_threshold}, ≥{min_agreement}/{n_runs}): "
                      f"{len(semantic_consensus)}")
                for sc in semantic_consensus:
                    print(f"    \"{sc['representative']}\" ({sc['n_runs']}/{n_runs} runs, "
                          f"{sc['n_variants']} variants)")

                all_results.append({
                    "community_id": comm_id,
                    "n_sessions": len(sessions),
                    "n_runs": n_runs,
                    "total_unique_concepts": len(unique_names),
                    "exact_consensus": exact_consensus,
                    "n_exact_consensus": len(exact_consensus),
                    "semantic_consensus": semantic_consensus,
                    "n_semantic_consensus": len(semantic_consensus),
                    "run_details": [{"run": i + 1, "concepts": names}
                                    for i, names in enumerate(run_concepts)],
                })

    # Summary
    if all_results:
        exact_counts = [r["n_exact_consensus"] for r in all_results]
        semantic_counts = [r["n_semantic_consensus"] for r in all_results]
        print(f"\n{'='*40}")
        print(f"Summary across {len(all_results)} communities:")
        print(f"  Exact consensus: mean={np.mean(exact_counts):.1f}")
        print(f"  Semantic consensus: mean={np.mean(semantic_counts):.1f}")

    output = {
        "experiment": "F51: Consensus Concept Extraction",
        "parameters": {
            "n_runs": n_runs,
            "consensus_threshold": consensus_threshold,
            "min_agreement": min_agreement,
        },
        "communities": all_results,
    }
    with open(RESULTS_DIR / "consensus_concepts.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to consensus_concepts.json")
    return output


# ─── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Concept-level experiments F48-F51")
    parser.add_argument("--f48", action="store_true", help="Guimerà-Amaral roles")
    parser.add_argument("--f49", action="store_true", help="Abstraction convergence")
    parser.add_argument("--f50", action="store_true", help="Hierarchical multiplex")
    parser.add_argument("--f51", action="store_true", help="Consensus extraction")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="Concept network threshold (default: 0.8)")
    args = parser.parse_args()

    if not any([args.f48, args.f49, args.f50, args.f51, args.all]):
        parser.print_help()
        sys.exit(0)

    if args.f48 or args.all:
        run_guimera_amaral(args.threshold)
    if args.f49 or args.all:
        run_abstraction_convergence()
    if args.f50 or args.all:
        run_hierarchical_multiplex()
    if args.f51 or args.all:
        run_consensus_extraction()
