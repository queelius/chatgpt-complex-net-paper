---
title: "Dialogue Turn-Taking Motifs in Human-AI Conversation Networks"
status: idea
domain: complex-networks
priority: medium
---

## Core Question

Do recurring structural patterns (motifs) in human-AI turn-taking
sequences encode cognitive patterns? Can motif frequencies distinguish
conversation types and track model evolution?

## Approach

1. Convert each conversation's message sequence to a directed graph of
   role transitions (user -> assistant -> user -> ...)
2. Enumerate 3-turn, 4-turn, 5-turn motifs using subgraph census
   (Nauty/pynauty for isomorphism)
3. Compute motif overrepresentation vs. random baseline (Erdos-Renyi
   turn sequences)
4. Correlate motif frequency with:
   - Community type (theoretical vs. practical domains)
   - Model era (GPT-3.5 / GPT-4 / GPT-4o / o1)
   - Conversation length and bridge type
5. Compute Markov chain entropy rate for turn-taking sequences
   - Transition matrix P(role_t+1 | role_t)
   - Compare entropy rates across domains and model eras

## Why This Works

- 35,244 turns across 27 months of model evolution
- Nobody else has this corpus with temporal coverage
- Subgraph census is standard and fast (pynauty)
- Clean separation between content analysis (existing papers) and
  structural analysis (this paper)

## Data Requirements

- Message role sequences from chatgpt-conversation-corpus
- Community labels from cognitive-MRI analysis
- Model version timestamps

## Target Venue

PLOS Complex Systems, Complex Networks (CompleNet), or Social Network
Analysis and Mining (SNAM)

## References

- Milo et al. (2002) network motifs
- Sacks-Schegloff (1968) conversation analysis
- Towell-Matta (2025) ablation study (2:1 weighting validates user
  intent primacy)
