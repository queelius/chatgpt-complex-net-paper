"""Concept-level network experiments (F98-F102).

Runs five analyses from the pending experiments list:
  F98: Concept co-occurrence in abstraction hierarchy (L1→L2 collapse patterns)
  F99: Temporal concept emergence order (which concepts appear when?)
  F100: Information-theoretic neighborhood analysis (mutual information, conditional entropy)
  F101: Network motif roles (which concepts participate in which structural motifs?)
  F102: Cross-platform concept mirror analysis (matched vs unmatched concepts)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score

RESULTS_DIR = Path(__file__).parent / "results"


def load_concept_data():
    """Load concept embeddings and similarity matrix."""
    with open(RESULTS_DIR / "concept_embeddings.json") as f:
        meta = json.load(f)
    sim = np.load(RESULTS_DIR / "concept_similarity_matrix.npy")
    return meta["concepts"], sim


def build_network(concepts, sim, threshold=0.70):
    """Build concept network at given threshold."""
    G = nx.Graph()
    for c in concepts:
        G.add_node(c["index"], name=c["name"], platform=c["platform"],
                    source_community=c["source_community"])
    n = len(concepts)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim[i, j]))
    return G


def get_louvain_communities(G, seed=42):
    try:
        import community as community_louvain
        return community_louvain.best_partition(G, random_state=seed)
    except ImportError:
        return {n: 0 for n in G.nodes()}


def run_f98(concepts, sim, threshold=0.70):
    """F98: Concept co-occurrence in abstraction hierarchy.

    Load L1→L2 hierarchy for both platforms. Analyze which L1 concepts
    collapse together at L2. Do cross-platform concepts merge at L2?
    """
    print("=" * 60)
    print("F98: Concept Co-occurrence in Abstraction Hierarchy")
    print("=" * 60)

    # Load hierarchy data
    with open(RESULTS_DIR / "concepts_chatgpt_hierarchy.json") as f:
        cg_hier = json.load(f)
    with open(RESULTS_DIR / "concepts_agentic_hierarchy.json") as f:
        ag_hier = json.load(f)

    cg_l2 = cg_hier["level_2_meta_concepts"]
    ag_l2 = ag_hier["level_2_meta_concepts"]

    print(f"\nChatGPT L2 meta-concepts ({len(cg_l2)}):")
    for mc in cg_l2:
        if isinstance(mc, dict):
            print(f"  {mc.get('name', mc)}")
        else:
            print(f"  {mc}")

    print(f"\nAgentic L2 meta-concepts ({len(ag_l2)}):")
    for mc in ag_l2:
        if isinstance(mc, dict):
            print(f"  {mc.get('name', mc)}")
        else:
            print(f"  {mc}")

    # Build concept network
    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    platforms = {c["index"]: c["platform"] for c in concepts}

    # Map L1 concepts to their source communities
    comm_concepts = defaultdict(list)
    for c in concepts:
        comm_concepts[c["source_community"]].append(c)

    # Which Louvain communities at concept level merge concepts from different
    # source communities? This reveals abstraction collapse.
    concept_comm = defaultdict(list)
    for c in concepts:
        concept_comm[partition.get(c["index"], -1)].append(c)

    print(f"\nConcept-level Louvain communities and their source diversity:")
    merge_patterns = []
    for comm_id in sorted(concept_comm.keys()):
        members = concept_comm[comm_id]
        if len(members) < 2:
            continue
        sources = Counter(m["source_community"] for m in members)
        plat_mix = Counter(m["platform"] for m in members)
        n_sources = len(sources)

        # Compute mean internal similarity
        indices = [m["index"] for m in members]
        internal_sims = []
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i + 1:]:
                internal_sims.append(sim[idx1, idx2])
        mean_sim = np.mean(internal_sims) if internal_sims else 0

        merge_patterns.append({
            "concept_community": comm_id,
            "size": len(members),
            "n_source_communities": n_sources,
            "source_distribution": dict(sources),
            "platform_distribution": dict(plat_mix),
            "mean_internal_similarity": float(mean_sim),
            "member_names": [m["name"] for m in members],
        })

        cg_count = plat_mix.get("chatgpt", 0)
        ag_count = plat_mix.get("agentic", 0)
        print(f"\n  C{comm_id} ({len(members)} concepts, {cg_count}CG+{ag_count}AG, "
              f"from {n_sources} source communities, mean_sim={mean_sim:.3f}):")
        for m in members[:5]:
            print(f"    {m['name']} ({m['platform']}, src=C{m['source_community']})")
        if len(members) > 5:
            print(f"    ... and {len(members) - 5} more")

    # Cross-platform merge rate: what fraction of concept communities contain both platforms?
    mixed = sum(1 for mp in merge_patterns if "chatgpt" in mp["platform_distribution"]
                and "agentic" in mp["platform_distribution"])
    print(f"\nCross-platform communities: {mixed}/{len(merge_patterns)} ({mixed/len(merge_patterns):.1%})")

    # Source community diversity: how many source communities feed into each concept community?
    source_divs = [mp["n_source_communities"] for mp in merge_patterns]
    print(f"Source diversity: mean={np.mean(source_divs):.1f}, max={max(source_divs)}")

    # Cross-platform L2 similarity: embed L2 concept names and compare
    cg_l2_names = [mc.get("name", mc) if isinstance(mc, dict) else mc for mc in cg_l2]
    ag_l2_names = [mc.get("name", mc) if isinstance(mc, dict) else mc for mc in ag_l2]

    # Use concept-level similarity to approximate L2 comparison
    # For each L2 meta-concept, find the L1 concepts in its domain and compute mean similarity
    print(f"\nL2 name comparison (ChatGPT × Agentic):")
    # Simple string-based matching of L2 names
    from difflib import SequenceMatcher
    l2_matches = []
    for cg_name in cg_l2_names:
        best_match = None
        best_score = 0
        for ag_name in ag_l2_names:
            score = SequenceMatcher(None, cg_name.lower(), ag_name.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = ag_name
        l2_matches.append({
            "chatgpt_l2": cg_name,
            "best_agentic_l2": best_match,
            "similarity": best_score,
        })
        print(f"  {cg_name} → {best_match} (sim={best_score:.3f})")

    result = {
        "method": "L1→L2 abstraction hierarchy collapse analysis",
        "chatgpt_l2_count": len(cg_l2_names),
        "agentic_l2_count": len(ag_l2_names),
        "chatgpt_l2": cg_l2_names,
        "agentic_l2": ag_l2_names,
        "merge_patterns": merge_patterns,
        "cross_platform_communities": mixed,
        "cross_platform_fraction": float(mixed / len(merge_patterns)) if merge_patterns else 0,
        "source_diversity_mean": float(np.mean(source_divs)),
        "l2_cross_platform_matches": l2_matches,
    }
    with open(RESULTS_DIR / "concept_hierarchy_collapse.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_hierarchy_collapse.json")


def run_f99(concepts, sim, threshold=0.70):
    """F99: Temporal concept emergence order.

    Using the 'era' information from F60, determine which concepts emerge
    in each era and whether cross-platform concepts emerge earlier or later
    than platform-specific ones.
    """
    print("\n" + "=" * 60)
    print("F99: Temporal Concept Emergence Order")
    print("=" * 60)

    # Load era data from F60
    with open(RESULTS_DIR / "model_era_concept_bridging.json") as f:
        era_data = json.load(f)

    concept_details = era_data["concept_details"]

    # Build name→concept mapping for platform lookup
    name_to_concept = {c["name"].lower(): c for c in concepts}

    # Enrich era data with platform info
    for cd in concept_details:
        matched = name_to_concept.get(cd["name"].lower())
        if matched:
            cd["platform"] = matched["platform"]
            cd["concept"] = cd["name"]
        else:
            cd["platform"] = "unknown"
            cd["concept"] = cd["name"]

    # Map concepts to eras
    era_concepts = defaultdict(list)
    for cd in concept_details:
        era_concepts[cd["era"]].append(cd)

    print(f"Concepts by era:")
    for era in ["early", "middle", "late"]:
        cs = era_concepts[era]
        n_cg = sum(1 for c in cs if c["platform"] == "chatgpt")
        n_ag = sum(1 for c in cs if c["platform"] == "agentic")
        print(f"  {era}: {len(cs)} concepts ({n_cg}CG + {n_ag}AG)")

    # Build concept network and compute metrics
    G = build_network(concepts, sim, threshold)
    partition = get_louvain_communities(G)
    platforms = {c["index"]: c["platform"] for c in concepts}
    bc = nx.betweenness_centrality(G)
    pr = nx.pagerank(G)
    degree = dict(G.degree())

    # Map concept_details to network nodes by name
    name_to_idx = {c["name"].lower(): c["index"] for c in concepts}

    # Cross-platform similarity by era
    for cd in concept_details:
        idx = name_to_idx.get(cd["concept"].lower())
        if idx is not None:
            cd["degree"] = degree.get(idx, 0)
            cd["betweenness"] = bc.get(idx, 0)
            cd["pagerank"] = pr.get(idx, 0)
            cd["community"] = partition.get(idx, -1)
            # Cross-platform degree
            cd["cross_platform_degree"] = sum(
                1 for nb in G.neighbors(idx) if platforms.get(nb) != cd["platform"]
            ) if idx in G else 0
        else:
            cd["degree"] = 0
            cd["betweenness"] = 0
            cd["pagerank"] = 0
            cd["community"] = -1
            cd["cross_platform_degree"] = 0

    # Compare early vs late concepts on network metrics
    print(f"\nNetwork metrics by era:")
    era_metrics = {}
    for era in ["early", "middle", "late"]:
        cs = era_concepts[era]
        if not cs:
            continue
        degs = [c["degree"] for c in cs]
        cpds = [c["cross_platform_degree"] for c in cs]
        bcs = [c["betweenness"] for c in cs]
        era_metrics[era] = {
            "mean_degree": float(np.mean(degs)),
            "mean_cross_platform_degree": float(np.mean(cpds)),
            "mean_betweenness": float(np.mean(bcs)),
            "n_concepts": len(cs),
        }
        print(f"  {era}: deg={np.mean(degs):.1f}, cross_deg={np.mean(cpds):.1f}, "
              f"BC={np.mean(bcs):.4f}")

    # Do early concepts have higher/lower cross-platform connectivity?
    early_cpd = [c["cross_platform_degree"] for c in era_concepts["early"]]
    late_cpd = [c["cross_platform_degree"] for c in era_concepts["late"]]
    if early_cpd and late_cpd:
        mw, mw_p = stats.mannwhitneyu(early_cpd, late_cpd, alternative="two-sided")
        print(f"\nEarly vs Late cross-platform degree: MW p={mw_p:.4e}")
    else:
        mw_p = 1.0

    # Kruskal-Wallis across eras for each metric
    print(f"\nKruskal-Wallis tests (early vs middle vs late):")
    kw_results = {}
    for metric in ["degree", "cross_platform_degree", "betweenness"]:
        groups = [
            [c[metric] for c in era_concepts[era]]
            for era in ["early", "middle", "late"]
            if era in era_concepts and era_concepts[era]
        ]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            kw_results[metric] = {"H": float(h), "p": float(p)}
            print(f"  {metric}: H={h:.3f}, p={p:.4f}")

    # Platform emergence pattern: do agentic concepts appear more in later eras?
    era_order = {"early": 0, "middle": 1, "late": 2}
    ag_eras = [era_order[c["era"]] for c in concept_details if c["platform"] == "agentic"]
    cg_eras = [era_order[c["era"]] for c in concept_details if c["platform"] == "chatgpt"]
    if ag_eras and cg_eras:
        mw2, mw2_p = stats.mannwhitneyu(ag_eras, cg_eras, alternative="two-sided")
        print(f"\nAgentic vs ChatGPT era distribution: MW p={mw2_p:.4e}")
        print(f"  Agentic mean era: {np.mean(ag_eras):.2f}")
        print(f"  ChatGPT mean era: {np.mean(cg_eras):.2f}")

    # Concept "bridging potential" by era: cross_platform_sim from F60 data
    print(f"\nCross-platform similarity by era (from F60):")
    for era in ["early", "middle", "late"]:
        cs = [c for c in concept_details if c["era"] == era and "mean_cross_sim" in c]
        if cs:
            sims = [c["mean_cross_sim"] for c in cs]
            print(f"  {era}: mean={np.mean(sims):.4f}, n={len(cs)}")

    result = {
        "method": "Temporal concept emergence order using era classification from F60",
        "era_distribution": {
            era: {"n": len(cs),
                  "chatgpt": sum(1 for c in cs if c["platform"] == "chatgpt"),
                  "agentic": sum(1 for c in cs if c["platform"] != "chatgpt")}
            for era, cs in era_concepts.items()
        },
        "era_metrics": era_metrics,
        "kruskal_wallis": kw_results,
        "early_vs_late_cross_platform_p": float(mw_p),
        "concept_era_details": [
            {
                "name": cd["concept"],
                "platform": cd["platform"],
                "era": cd["era"],
                "degree": cd["degree"],
                "cross_platform_degree": cd["cross_platform_degree"],
                "betweenness": cd["betweenness"],
            }
            for cd in concept_details if cd["degree"] > 0
        ],
    }
    with open(RESULTS_DIR / "concept_temporal_emergence.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_temporal_emergence.json")


def run_f100(concepts, sim, threshold=0.70):
    """F100: Information-theoretic neighborhood analysis.

    For each node, compute:
    - Neighborhood entropy (platform diversity of neighbors)
    - Mutual information between node platform and neighbor platforms
    - Conditional entropy H(neighbor_platform | node_platform)
    """
    print("\n" + "=" * 60)
    print("F100: Information-Theoretic Neighborhood Analysis")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n = len(concepts)
    platforms = {c["index"]: c["platform"] for c in concepts}
    partition = get_louvain_communities(G)

    def entropy(probs):
        """Shannon entropy in bits."""
        probs = [p for p in probs if p > 0]
        return -sum(p * np.log2(p) for p in probs)

    # Per-node neighborhood platform entropy
    node_entropy = {}
    for v in G.nodes():
        nbs = list(G.neighbors(v))
        if not nbs:
            node_entropy[v] = 0.0
            continue
        nb_plats = [platforms[u] for u in nbs]
        counts = Counter(nb_plats)
        total = len(nb_plats)
        probs = [c / total for c in counts.values()]
        node_entropy[v] = entropy(probs)

    cg_ent = [node_entropy[v] for v in G.nodes() if platforms[v] == "chatgpt"]
    ag_ent = [node_entropy[v] for v in G.nodes() if platforms[v] == "agentic"]
    print(f"Neighborhood platform entropy:")
    print(f"  ChatGPT nodes: mean H={np.mean(cg_ent):.4f}")
    print(f"  Agentic nodes: mean H={np.mean(ag_ent):.4f}")
    mw, mw_p = stats.mannwhitneyu(cg_ent, ag_ent, alternative="two-sided")
    print(f"  Mann-Whitney p={mw_p:.4e}")

    # Global mutual information: I(node_platform; neighbor_platform)
    # Build joint distribution from all edges
    joint_counts = Counter()
    for u, v in G.edges():
        joint_counts[(platforms[u], platforms[v])] += 1
        joint_counts[(platforms[v], platforms[u])] += 1  # both directions

    total_pairs = sum(joint_counts.values())
    joint_probs = {k: v / total_pairs for k, v in joint_counts.items()}

    # Marginals
    marginal_src = Counter()
    marginal_dst = Counter()
    for (s, d), p in joint_probs.items():
        marginal_src[s] += p
        marginal_dst[d] += p

    # MI = sum p(x,y) log(p(x,y) / (p(x) * p(y)))
    mi = 0
    for (s, d), p_joint in joint_probs.items():
        p_s = marginal_src[s]
        p_d = marginal_dst[d]
        if p_joint > 0 and p_s > 0 and p_d > 0:
            mi += p_joint * np.log2(p_joint / (p_s * p_d))

    # H(platform)
    h_platform = entropy(list(marginal_src.values()))
    # Normalized MI
    nmi_val = mi / h_platform if h_platform > 0 else 0

    print(f"\nGlobal mutual information I(src_platform; dst_platform):")
    print(f"  MI = {mi:.4f} bits")
    print(f"  H(platform) = {h_platform:.4f} bits")
    print(f"  Normalized MI = {nmi_val:.4f}")

    # Conditional entropy: H(neighbor_platform | node_platform)
    h_cond = h_platform - mi  # H(Y|X) = H(Y) - I(X;Y) when marginals are same
    print(f"  H(neighbor_platform | node_platform) = {h_cond:.4f} bits")

    # Community-level mutual information
    # I(node_community; neighbor_community) for each node
    comm_joint = Counter()
    for u, v in G.edges():
        cu, cv = partition.get(u, -1), partition.get(v, -1)
        comm_joint[(cu, cv)] += 1
        comm_joint[(cv, cu)] += 1

    total_comm = sum(comm_joint.values())
    comm_joint_p = {k: v / total_comm for k, v in comm_joint.items()}
    comm_marginal = Counter()
    for (s, d), p in comm_joint_p.items():
        comm_marginal[s] += p

    mi_comm = 0
    for (s, d), p_joint in comm_joint_p.items():
        p_s = comm_marginal[s]
        p_d = comm_marginal[d]
        if p_joint > 0 and p_s > 0 and p_d > 0:
            mi_comm += p_joint * np.log2(p_joint / (p_s * p_d))

    h_comm = entropy(list(comm_marginal.values()))
    nmi_comm = mi_comm / h_comm if h_comm > 0 else 0

    print(f"\nCommunity mutual information I(src_comm; dst_comm):")
    print(f"  MI = {mi_comm:.4f} bits")
    print(f"  H(community) = {h_comm:.4f} bits")
    print(f"  Normalized MI = {nmi_comm:.4f}")

    # Comparison: platform MI vs community MI
    print(f"\nPlatform MI ({mi:.4f}) vs Community MI ({mi_comm:.4f})")
    print(f"  Community carries {mi_comm / mi:.1f}× more information about neighbors" if mi > 0 else "")

    # Per-community conditional entropy
    print(f"\nPer-community neighborhood entropy:")
    for comm_id in sorted(set(partition.values())):
        members = [v for v in G.nodes() if partition.get(v) == comm_id]
        if len(members) < 2:
            continue
        ents = [node_entropy[v] for v in members if v in node_entropy]
        n_cg = sum(1 for v in members if platforms[v] == "chatgpt")
        n_ag = len(members) - n_cg
        if ents:
            print(f"  C{comm_id} ({len(members)} nodes, {n_cg}CG+{n_ag}AG): "
                  f"mean H={np.mean(ents):.3f}")

    result = {
        "method": "Information-theoretic analysis of neighborhood structure",
        "neighborhood_entropy": {
            "chatgpt_mean": float(np.mean(cg_ent)),
            "agentic_mean": float(np.mean(ag_ent)),
            "mann_whitney_p": float(mw_p),
        },
        "platform_mutual_information": {
            "mi_bits": float(mi),
            "h_platform": float(h_platform),
            "normalized_mi": float(nmi_val),
            "conditional_entropy": float(h_cond),
        },
        "community_mutual_information": {
            "mi_bits": float(mi_comm),
            "h_community": float(h_comm),
            "normalized_mi": float(nmi_comm),
        },
        "mi_ratio_community_over_platform": float(mi_comm / mi) if mi > 0 else None,
    }
    with open(RESULTS_DIR / "concept_information_theory.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_information_theory.json")


def run_f101(concepts, sim, threshold=0.70):
    """F101: Network motif roles — which concepts participate in which structural motifs?

    For each node, count: triangles, open triads (as center), open triads (as end),
    4-cliques. Correlate motif participation with platform and bridging role.
    """
    print("\n" + "=" * 60)
    print("F101: Network Motif Roles")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    platforms = {c["index"]: c["platform"] for c in concepts}
    partition = get_louvain_communities(G)

    nodes = list(G.nodes())

    # Per-node motif participation
    motif_data = {}
    for v in nodes:
        nbs = set(G.neighbors(v))
        # Triangles: edges among neighbors
        triangles = 0
        for u in nbs:
            for w in nbs:
                if w > u and G.has_edge(u, w):
                    triangles += 1

        # Open triads where v is the center (wedges)
        deg = len(nbs)
        wedges = deg * (deg - 1) // 2

        # 4-cliques containing v
        four_cliques = 0
        nbs_list = list(nbs)
        for i, u in enumerate(nbs_list):
            for j in range(i + 1, len(nbs_list)):
                w = nbs_list[j]
                if not G.has_edge(u, w):
                    continue
                # u, w are mutual neighbors of v and connected
                # Look for 4th node connected to v, u, w
                common = nbs & set(G.neighbors(u)) & set(G.neighbors(w))
                common.discard(v)
                four_cliques += len(common)
        four_cliques //= 3  # each 4-clique counted 3 times

        # Cross-platform triangles
        cross_triangles = 0
        for u in nbs:
            for w in nbs:
                if w > u and G.has_edge(u, w):
                    plats = {platforms[v], platforms[u], platforms[w]}
                    if len(plats) > 1:
                        cross_triangles += 1

        motif_data[v] = {
            "triangles": triangles,
            "wedges": wedges,
            "local_clustering": triangles / wedges if wedges > 0 else 0,
            "four_cliques": four_cliques,
            "cross_platform_triangles": cross_triangles,
            "cross_triangle_fraction": cross_triangles / triangles if triangles > 0 else 0,
        }

    # Platform comparison
    for metric in ["triangles", "wedges", "local_clustering", "four_cliques",
                    "cross_platform_triangles", "cross_triangle_fraction"]:
        cg_vals = [motif_data[v][metric] for v in nodes if platforms[v] == "chatgpt"]
        ag_vals = [motif_data[v][metric] for v in nodes if platforms[v] == "agentic"]
        if cg_vals and ag_vals:
            mw, p = stats.mannwhitneyu(cg_vals, ag_vals, alternative="two-sided")
            sig = "*" if p < 0.05 else ""
            print(f"  {metric:30s}: CG={np.mean(cg_vals):.2f}, AG={np.mean(ag_vals):.2f}, "
                  f"p={p:.4f}{sig}")

    # Top nodes by triangle participation
    top_tri = sorted(nodes, key=lambda v: motif_data[v]["triangles"], reverse=True)[:10]
    print(f"\nTop 10 by triangle participation:")
    for v in top_tri:
        md = motif_data[v]
        print(f"  {md['triangles']:4d} triangles ({md['cross_platform_triangles']} cross): "
              f"{concepts[v]['name']} ({platforms[v]})")

    # Top nodes by 4-clique participation
    top_4c = sorted(nodes, key=lambda v: motif_data[v]["four_cliques"], reverse=True)[:10]
    print(f"\nTop 10 by 4-clique participation:")
    for v in top_4c:
        md = motif_data[v]
        print(f"  {md['four_cliques']:4d} 4-cliques: {concepts[v]['name']} ({platforms[v]})")

    # Cross-platform triangle fraction vs degree
    degs = [G.degree(v) for v in nodes if motif_data[v]["triangles"] > 0]
    ctf = [motif_data[v]["cross_triangle_fraction"] for v in nodes if motif_data[v]["triangles"] > 0]
    if len(degs) > 3:
        rho, p = stats.spearmanr(degs, ctf)
        print(f"\nDegree vs cross-platform triangle fraction: ρ={rho:.3f}, p={p:.4e}")

    result = {
        "method": "Per-node motif census: triangles, wedges, 4-cliques, cross-platform triangles",
        "platform_comparison": {
            metric: {
                "chatgpt_mean": float(np.mean([motif_data[v][metric] for v in nodes if platforms[v] == "chatgpt"])),
                "agentic_mean": float(np.mean([motif_data[v][metric] for v in nodes if platforms[v] == "agentic"])),
            }
            for metric in ["triangles", "wedges", "four_cliques", "cross_triangle_fraction"]
        },
        "top_triangle_nodes": [
            {"name": concepts[v]["name"], "platform": platforms[v],
             "triangles": motif_data[v]["triangles"],
             "cross_triangles": motif_data[v]["cross_platform_triangles"]}
            for v in top_tri
        ],
        "top_4clique_nodes": [
            {"name": concepts[v]["name"], "platform": platforms[v],
             "four_cliques": motif_data[v]["four_cliques"]}
            for v in top_4c
        ],
    }
    with open(RESULTS_DIR / "concept_motif_roles.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_motif_roles.json")


def run_f102(concepts, sim, threshold=0.70):
    """F102: Cross-platform concept mirror analysis.

    Identify "mirror" concept pairs: concepts from different platforms with
    high similarity. Compare matched vs unmatched concepts on network metrics.
    """
    print("\n" + "=" * 60)
    print("F102: Cross-Platform Concept Mirror Analysis")
    print("=" * 60)

    G = build_network(concepts, sim, threshold)
    n = len(concepts)
    platforms = {c["index"]: c["platform"] for c in concepts}
    partition = get_louvain_communities(G)
    bc = nx.betweenness_centrality(G)
    pr = nx.pagerank(G)
    degree = dict(G.degree())
    kcore = nx.core_number(G)

    # Find best cross-platform match for each concept
    cg_concepts = [c for c in concepts if c["platform"] == "chatgpt"]
    ag_concepts = [c for c in concepts if c["platform"] == "agentic"]

    # For each CG concept, find best AG match and vice versa
    mirrors = []
    matched_cg = set()
    matched_ag = set()

    # Greedy matching: take highest-similarity cross-platform pairs
    cross_pairs = []
    for cg in cg_concepts:
        for ag in ag_concepts:
            cross_pairs.append((cg["index"], ag["index"], sim[cg["index"], ag["index"]]))

    cross_pairs.sort(key=lambda x: x[2], reverse=True)

    # Greedy 1-to-1 matching
    for cg_idx, ag_idx, similarity in cross_pairs:
        if cg_idx in matched_cg or ag_idx in matched_ag:
            continue
        mirrors.append({
            "chatgpt": concepts[cg_idx]["name"],
            "chatgpt_idx": cg_idx,
            "agentic": concepts[ag_idx]["name"],
            "agentic_idx": ag_idx,
            "similarity": float(similarity),
            "same_community": partition.get(cg_idx) == partition.get(ag_idx),
        })
        matched_cg.add(cg_idx)
        matched_ag.add(ag_idx)

    # Classify: strong mirrors (sim > 0.80), moderate (0.70-0.80), weak (< 0.70)
    strong = [m for m in mirrors if m["similarity"] >= 0.80]
    moderate = [m for m in mirrors if 0.70 <= m["similarity"] < 0.80]
    weak = [m for m in mirrors if m["similarity"] < 0.70]

    print(f"Mirror pairs: {len(mirrors)} total")
    print(f"  Strong (sim ≥ 0.80): {len(strong)}")
    print(f"  Moderate (0.70 ≤ sim < 0.80): {len(moderate)}")
    print(f"  Weak (sim < 0.70): {len(weak)}")

    print(f"\nTop 15 strongest mirrors:")
    for m in mirrors[:15]:
        same = "✓" if m["same_community"] else "✗"
        print(f"  {m['similarity']:.3f} [{same}]: {m['chatgpt']} ↔ {m['agentic']}")

    # Same-community rate by mirror strength
    strong_same = sum(1 for m in strong if m["same_community"])
    mod_same = sum(1 for m in moderate if m["same_community"])
    weak_same = sum(1 for m in weak if m["same_community"])
    print(f"\nSame-community rate:")
    print(f"  Strong mirrors: {strong_same}/{len(strong)} ({strong_same/len(strong):.1%})" if strong else "")
    print(f"  Moderate mirrors: {mod_same}/{len(moderate)} ({mod_same/len(moderate):.1%})" if moderate else "")
    print(f"  Weak mirrors: {weak_same}/{len(weak)} ({weak_same/len(weak):.1%})" if weak else "")

    # Compare matched (has strong mirror) vs unmatched concepts
    strong_cg_idx = {m["chatgpt_idx"] for m in strong}
    strong_ag_idx = {m["agentic_idx"] for m in strong}
    matched_idx = strong_cg_idx | strong_ag_idx
    unmatched_idx = set(range(n)) - matched_idx

    matched_degs = [degree.get(i, 0) for i in matched_idx]
    unmatched_degs = [degree.get(i, 0) for i in unmatched_idx]
    matched_bc = [bc.get(i, 0) for i in matched_idx]
    unmatched_bc = [bc.get(i, 0) for i in unmatched_idx]
    matched_core = [kcore.get(i, 0) for i in matched_idx]
    unmatched_core = [kcore.get(i, 0) for i in unmatched_idx]

    print(f"\nMatched (strong mirror) vs Unmatched concepts:")
    for metric_name, matched_vals, unmatched_vals in [
        ("Degree", matched_degs, unmatched_degs),
        ("Betweenness", matched_bc, unmatched_bc),
        ("K-core", matched_core, unmatched_core),
    ]:
        if matched_vals and unmatched_vals:
            mw, p = stats.mannwhitneyu(matched_vals, unmatched_vals, alternative="two-sided")
            print(f"  {metric_name}: matched={np.mean(matched_vals):.3f}, "
                  f"unmatched={np.mean(unmatched_vals):.3f}, p={p:.4e}")

    # Unmatched concepts (no strong mirror): what are they?
    unmatched_cg = [c for c in cg_concepts if c["index"] not in strong_cg_idx]
    unmatched_ag = [c for c in ag_concepts if c["index"] not in strong_ag_idx]
    print(f"\nUnmatched ChatGPT concepts ({len(unmatched_cg)}):")
    for c in sorted(unmatched_cg, key=lambda x: degree.get(x["index"], 0), reverse=True)[:10]:
        print(f"  deg={degree.get(c['index'], 0):3d}: {c['name']}")
    print(f"\nUnmatched Agentic concepts ({len(unmatched_ag)}):")
    for c in sorted(unmatched_ag, key=lambda x: degree.get(x["index"], 0), reverse=True)[:10]:
        print(f"  deg={degree.get(c['index'], 0):3d}: {c['name']}")

    result = {
        "method": "Greedy 1-to-1 cross-platform matching by similarity, then compare matched vs unmatched",
        "n_mirrors": len(mirrors),
        "n_strong": len(strong),
        "n_moderate": len(moderate),
        "n_weak": len(weak),
        "same_community_rate": {
            "strong": float(strong_same / len(strong)) if strong else 0,
            "moderate": float(mod_same / len(moderate)) if moderate else 0,
            "weak": float(weak_same / len(weak)) if weak else 0,
        },
        "top_mirrors": mirrors[:20],
        "matched_vs_unmatched": {
            "n_matched": len(matched_idx),
            "n_unmatched": len(unmatched_idx),
            "degree": {
                "matched_mean": float(np.mean(matched_degs)),
                "unmatched_mean": float(np.mean(unmatched_degs)),
            },
            "betweenness": {
                "matched_mean": float(np.mean(matched_bc)),
                "unmatched_mean": float(np.mean(unmatched_bc)),
            },
            "kcore": {
                "matched_mean": float(np.mean(matched_core)),
                "unmatched_mean": float(np.mean(unmatched_core)),
            },
        },
        "unmatched_chatgpt": [c["name"] for c in unmatched_cg],
        "unmatched_agentic": [c["name"] for c in unmatched_ag],
    }
    with open(RESULTS_DIR / "concept_mirrors.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to concept_mirrors.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concept experiments F98-F102")
    parser.add_argument("--f98", action="store_true", help="Hierarchy collapse")
    parser.add_argument("--f99", action="store_true", help="Temporal emergence")
    parser.add_argument("--f100", action="store_true", help="Information theory")
    parser.add_argument("--f101", action="store_true", help="Motif roles")
    parser.add_argument("--f102", action="store_true", help="Mirror analysis")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    if not any([args.f98, args.f99, args.f100, args.f101, args.f102, args.all]):
        parser.print_help()
        sys.exit(1)

    concepts, sim = load_concept_data()
    print(f"Loaded {len(concepts)} concepts, similarity matrix {sim.shape}")

    if args.f98 or args.all:
        run_f98(concepts, sim, args.threshold)
    if args.f99 or args.all:
        run_f99(concepts, sim, args.threshold)
    if args.f100 or args.all:
        run_f100(concepts, sim, args.threshold)
    if args.f101 or args.all:
        run_f101(concepts, sim, args.threshold)
    if args.f102 or args.all:
        run_f102(concepts, sim, args.threshold)


if __name__ == "__main__":
    main()
