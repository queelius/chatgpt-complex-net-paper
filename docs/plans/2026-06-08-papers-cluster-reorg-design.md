# Design: Reorganize Cognitive-MRI repo into a paper cluster

- **Date:** 2026-06-08
- **Status:** Draft, awaiting user review
- **Author:** Alexander Towell
- **Scope:** One-time structural migration of the `cognitive-mri-ai-conversations` repository.

## Problem

The repository began as a single paper (the published Complex Networks 2025 "Cognitive MRI"
conference paper) and has organically grown into a **research program of six papers**, all built
on the same ChatGPT conversation corpus:

| Current directory | Paper | Self-containment |
|---|---|---|
| `comp-net-2025-camera-ready/` | Conference paper (Springer, **published**) | paper + slides + images |
| `comp-net-2025-journal/` | Journal extension (PLOS, rejected 2026-04-27; seeking venue) | paper + own code + own data |
| `agentic/` | "From Episodes to Abstractions" (ISCS 2026) | own code + experiments + data + paper |
| `embedding-dynamics/` | Embedding Dynamics theory paper | experiments + paper |
| `semantic-dynamics/` | Semantic Dynamics paper | experiments + paper |
| `operational-memex/` | Operational Memex paper (uncommitted) | own code + experiments + data + paper |

Everything lives in **one git repo** (no nested `.git`), with shared scaffolding at the root
(`code/`, `data/`, `dev/`, `docs/`, `future-ideas/`) and root metadata that makes the whole repo a
**published, citeable artifact**: GitHub remote `queelius/cognitive-mri-conversations`,
`CITATION.cff`, and `.zenodo.json` describing the "Cognitive MRI of AI Conversations: Research
Compendium." It is disorganized because it outgrew its original single-paper shape.

## Goals

1. Each paper becomes its **own independent git repository** under a `papers/` directory, matching
   the user's established convention (`~/github/papers/` and `~/github/` are both plain directories
   of independent repos).
2. **Preserve the published compendium identity** non-destructively: the existing GitHub repo,
   Zenodo record, and citation keep resolving.
3. **Preserve git history** for each paper where it is meaningful.
4. Leave the program **buildable and reproducible** after the move (papers still compile; shared
   corpus/pipeline still reachable).

## Non-goals

- Re-hosting the shared corpus or pipeline. The corpus already has a home at
  `~/github/chatgpt-conversation-corpus` (dataset) and the pipeline at `chatgpt-complex-net`
  (Zenodo DOI `10.5281/zenodo.15314235`). We reference, not duplicate.
- Creating *public* remotes. We create *private* remotes for all five children during the
  migration (off-machine backup); making any of them public is deferred per paper, decided when
  each is ready.
- Rewriting history to strip binary bloat (`git filter-repo` surgery). Out of scope; noted as a
  possible future cleanup.
- Moving the cluster to `~/github/` root (as `bernoulli/` and `trapdoor-computing/` sit). Possible
  later; this reorg keeps the cluster at its current path.

## Key decisions (resolved with the user)

1. **Repo model:** parent directory of *separate* repos (not a monorepo, not git submodules).
2. **Umbrella identity:** the parent stays a **thin git repo**, a program landing page. It keeps
   the existing `cognitive-mri-conversations` remote, `CITATION.cff`, and `.zenodo.json`. Its job
   shrinks to: program overview, an index linking each paper repo, and links to the external
   corpus/pipeline repos.
3. **Conference + journal = one repo** (`papers/comp-net/`). The journal extends the conference and
   they share `code/` + `data/` and the citation lineage; splitting them fragments tightly-coupled
   work.
4. **Shared `code/` and `data/`** (the comp-net pipeline) move **into** `papers/comp-net/`. The
   newer papers already carry their own code/data.
5. **`dev/`** (3.6 GB gitignored raw corpus) stays at the umbrella as shared working data; child
   code that reads it uses a configurable path (documented), not hardcoded cross-repo relatives.
6. **Spec/plan location:** `docs/plans/` (this repo's existing convention), not the skill default.
7. **Child dir names:** accept the proposed names, including renaming `agentic/` to
   `hierarchical-memory/`.
8. **comp-net history method:** the clean 4-way `subtree split` + merge (lean repo, scoped log).
9. **GitHub remotes:** create **private** remotes for all five children during the migration and
   push, for immediate off-machine backup. The umbrella keeps its existing
   `cognitive-mri-conversations` remote.
10. **Cluster location:** keep at `~/github/papers/cognitive-mri-ai-conversations/`.

## Target structure

```
cognitive-mri-ai-conversations/        # umbrella: thin git repo (keeps GitHub remote + DOI + CITATION)
├── README.md                          # REWRITTEN: program overview + index of all six papers
├── CITATION.cff                        # umbrella/compendium citation (links to each paper)
├── .zenodo.json                        # unchanged role: compendium record, links out
├── CLAUDE.md                           # REWRITTEN for the cluster layout
├── LICENSE
├── .gitignore                          # ADD: papers/*  (umbrella does not track child repos)
├── docs/                               # program-level design/plan docs (this file lives here)
├── future-ideas/                       # tracked by umbrella (percolation, turn-taking, compsac-2026)
├── dev/                                # gitignored shared raw corpus (working data, unchanged)
└── papers/                             # umbrella .gitignores papers/*; each child is its own repo
    ├── comp-net/                       # INDEPENDENT repo: conference + journal lineage
    │   ├── conference/                 # was comp-net-2025-camera-ready/
    │   │   └── springer-cognitive-mri.pdf   # was root-level
    │   ├── journal/                    # was comp-net-2025-journal/ (keeps its own code/ + data/)
    │   ├── code/                       # was root code/ (shared comp-net pipeline)
    │   └── data/                       # was root data/ (reproducibility artifacts)
    ├── hierarchical-memory/            # INDEPENDENT repo: was agentic/ (ISCS 2026)
    ├── embedding-dynamics/             # INDEPENDENT repo
    ├── semantic-dynamics/              # INDEPENDENT repo
    └── operational-memex/              # INDEPENDENT repo (currently uncommitted)
```

### Child repo local-dir names (proposed; confirm)

`comp-net/`, `hierarchical-memory/` (renamed from `agentic/`), `embedding-dynamics/`,
`semantic-dynamics/`, `operational-memex/`.

### Child GitHub repo names (created private during migration)

A `cmri-` prefix associates them as a program: `cmri-comp-net`, `cmri-hierarchical-memory`,
`cmri-embedding-dynamics`, `cmri-semantic-dynamics`, `cmri-operational-memex`. All created
**private** under `queelius` and pushed during the migration. Visibility (making any public) is
decided per paper later. The plan will confirm these exact names before creating the repos.

## Migration model

**Philosophy: the umbrella is the ancestor; children are extractions.**

- **Umbrella = the current repo, kept in place.** Identity, remote, full commit history, and the
  published-DOI linkage are preserved exactly. Its `.git` stays heavy (acceptable, it is the
  archive of record). Its *working tree* is slimmed to the landing page plus `future-ideas/`,
  `docs/`, and gitignored `dev/`, with `papers/*` gitignored.
- **Each child = a new git repo** at `papers/<name>/`, created with **history preserved** where
  meaningful:
  - `hierarchical-memory` from `git subtree split` of `agentic/` (41 commits).
  - `semantic-dynamics` from `git subtree split` of `semantic-dynamics/` (7 commits).
  - `embedding-dynamics` from `git subtree split` of `embedding-dynamics/` (1 commit) or fresh init.
  - `operational-memex` from **fresh `git init`** (0 commits; nothing to preserve).
  - `comp-net` from the history of its four constituents (`comp-net-2025-camera-ready`,
    `comp-net-2025-journal`, `code`, `data`), preserved. **Recommended:** subtree-split each of the
    four and merge into the subdirs above. **Acceptable fallback if the 4-way merge is troublesome:**
    clone-and-prune (clone the umbrella, `git rm -r` all non-comp-net dirs in one commit), which
    keeps full intertwined history at the cost of carrying other papers' blobs.

Each child keeps a paper-specific log; the umbrella retains the complete provenance.

## Prerequisites (Step 0, before any history surgery)

The working tree is currently dirty; `subtree split` only sees committed history, so uncommitted
work would be dropped from children. Before migrating:

1. **Resolve `compsac-2026`:** it was moved from the repo root into `future-ideas/compsac-2026/`
   but the move is uncommitted (root copy shows as deleted, `future-ideas/` is untracked). Commit
   the move.
2. **Commit modified work:** `agentic/experiments/figures/zoom_in_bridge.*`,
   `comp-net-2025-journal/paper/PLOS/paper-with-figs.pdf`, `semantic-dynamics/paper/paper.pdf`,
   and the various untracked-but-wanted PDFs/templates/scripts under `agentic/`, `comp-net-*`.
3. **Add `operational-memex/` and `future-ideas/`** (track them, since their content must seed the
   new repos and umbrella).
4. **Gitignore build/junk artifacts** rather than commit them: `.coverage`, `.playwright-mcp/`,
   `*/CNALatexTemplate/main.pdf`, `*/html_paper/`, `*/plos-template/`, `*/plos-template.zip`,
   extracted template dirs, and other reproducible build outputs.
5. **Achieve a clean `git status`,** then **tag the rollback point:**
   `git tag pre-cluster-reorg-2026-06-08`.

## Shared corpus / pipeline handling

- `dev/` (raw corpus, gitignored) stays at the umbrella. Child code that reads raw data uses a
  configurable path: an env var (e.g. `CMRI_CORPUS_DIR`) or a documented symlink, defaulting to
  the umbrella `dev/`. No hardcoded `../../dev` cross-repo paths.
- The *published* corpus and pipeline are referenced (not vendored): each child README notes the
  dependency on `chatgpt-conversation-corpus` and `chatgpt-complex-net`.
- `papers/comp-net/code/` is the canonical comp-net pipeline (a working mirror of
  `chatgpt-complex-net`); its README notes the relationship.

## Umbrella landing page (new `README.md`), required contents

- One-paragraph program description (knowledge-network analysis of AI conversations).
- A table indexing the six papers: title, venue/status, path under `papers/`, and (once pushed)
  repo link.
- Links to the shared corpus (`chatgpt-conversation-corpus`) and pipeline (`chatgpt-complex-net`,
  with DOI).
- A "how this repo is organized" note explaining that `papers/*` are independent repos.
- Citation guidance (cite the relevant paper; compendium DOI for the program).

## Safety, reversibility, verification

- **Reversibility:** the `pre-cluster-reorg-2026-06-08` tag captures the exact pre-migration state;
  nothing is removed from the umbrella's *history*, only from its working tree. Children are
  derived, so the migration can be re-run.
- **Order of operations:** create and verify each child repo *before* slimming the umbrella working
  tree, so no data is removed until its extraction is confirmed.
- **Verification checklist (per child):**
  - `git -C papers/<name> log --oneline` shows the expected (paper-specific) history.
  - `git -C papers/<name> status` is clean.
  - The paper still builds (e.g. `pdflatex` in `papers/comp-net/conference/paper/`,
    `papers/hierarchical-memory/paper/`, and so on).
- **Verification (umbrella):** `git remote -v` unchanged; `CITATION.cff` and `.zenodo.json` intact;
  `git status` clean with `papers/*` ignored; README renders the paper index.
- **Catalog:** update the entry in `~/github/CLAUDE.md` to describe the new cluster shape.

## Review outcome

All four open questions were resolved during review (see Key decisions 7 to 10): accept the
proposed names with `agentic/` renamed to `hierarchical-memory/`; use clean subtree-split + merge
for comp-net; create private remotes for all five children and push; keep the cluster under
`papers/`. The design is final and ready for an implementation plan.
