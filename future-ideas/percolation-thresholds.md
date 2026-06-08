---
title: "Percolation Phase Transitions in Conversation Similarity Networks"
status: idea
domain: complex-networks
priority: high
---

## Core Question

The ablation study (Towell-Matta 2025) found catastrophic fragmentation
near theta=0.875. Is this a genuine percolation phase transition, and
if so, what universality class does it belong to?

## Approach

1. Finite-size scaling analysis at 15+ thresholds near the critical point
2. Measure critical exponents:
   - P(giant component) ~ (theta - theta_c)^beta
   - Correlation length xi ~ (theta - theta_c)^{-nu}
   - Cluster size distribution P(s) ~ s^{-tau} at criticality
3. Compare against Erdos-Renyi percolation (beta=1, nu=1/2) and
   2D lattice percolation (beta=5/36, nu=4/3)
4. Semantic correlation structure should produce non-classical exponents

## Why This Works

- Data already exists (1,906 conversations, all-pairs cosine)
- Phase transition already documented empirically
- percolation_thresholds.py module already built in open-problems repo
- Clean paper: one focused result with quantitative predictions

## Verification

- Compare against synthetic networks with equivalent density
- Bootstrap confidence intervals on critical exponents
- Test universality: do different embedding models give same exponents?

## Target Venue

Physical Review E (networks + statistical mechanics) or Network Science
