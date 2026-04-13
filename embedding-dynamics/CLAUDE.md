# CLAUDE.md

## What This Is

Theory paper: "Embedding Dynamics: Treating Text Sequences as Trajectories in Semantic Space."

The core claim: sub-unit embeddings (messages, paragraphs, etc.) are forces/impulses, not positions. Their accumulation is the semantic position. This produces a derivative tower (velocity, acceleration, changepoints) and a family of filters (lenses) parameterized by a single decay rate (the semantic half-life).

## CRITICAL: The Prefix Experiment

**Before writing the paper, run the prefix experiment.**

Script: ~/github/papers/cognitive-mri-ai-conversations/semantic-dynamics/experiments/exp04_prefix_vs_accumulation.py

This experiment tests whether accumulate(embed(msg_1), ..., embed(msg_k)) approximates embed(concat(msg_1, ..., msg_k)). The entire theoretical framework stands or falls on this result.

The user is actively preparing an embedding model to run this. It is the HIGHEST PRIORITY task.

## Paper Structure

Layered: core interpretation + extensions.

- Sections 1-4: The theory and its validation (the contribution)
- Sections 5-6: Applications and a brief case study (consequences)
- Section 7-8: Discussion and conclusion

Full spec: paper/SPEC.md

## Dependencies

- embflow package: ~/github/beta/embflow (35 tests, all passing)
- Analysis DB: ~/github/papers/cognitive-mri-ai-conversations/operational-memex/data/analysis.db
- Existing experiments: semantic-dynamics/experiments/ and operational-memex/experiments/

## Companion Paper

The semantic-dynamics paper (same parent repo, semantic-dynamics/ directory) is the empirical companion. It presents the conversation-archive experiments in detail. This paper presents the theory and validates it.
