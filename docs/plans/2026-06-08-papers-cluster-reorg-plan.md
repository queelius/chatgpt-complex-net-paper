# Paper-Cluster Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the `cognitive-mri-ai-conversations` repo into five independent paper repos under `papers/`, each with preserved git history and a private GitHub remote, while keeping the parent as a thin, still-published "compendium" landing page.

**Architecture:** The current repo stays in place as the umbrella (it keeps its remote, DOI, CITATION, and full history as the archive of record). Each paper is extracted with `git subtree split` (history-preserving) into a fresh repo at `papers/<name>/`, then untracked working data is layered in via `rsync`. Originals are removed from the umbrella only after each child is verified and pushed. The umbrella then gitignores `papers/*` and is slimmed to a landing page.

**Tech Stack:** git 2.43 (`git-subtree` confirmed present), `rsync`, `gh` CLI (authenticated as `queelius`), bash.

**Design reference:** `docs/plans/2026-06-08-papers-cluster-reorg-design.md`

**Source-to-child mapping:**

| Child repo (`papers/<name>/`) | Built from (umbrella dirs) | GitHub repo (private) | History |
|---|---|---|---|
| `comp-net/` | `comp-net-2025-camera-ready/` -> `conference/`, `comp-net-2025-journal/` -> `journal/`, `code/` -> `code/`, `data/` -> `data/` | `cmri-comp-net` | 4-way subtree-split + subtree-add merge |
| `hierarchical-memory/` | `agentic/` | `cmri-hierarchical-memory` | subtree-split |
| `embedding-dynamics/` | `embedding-dynamics/` | `cmri-embedding-dynamics` | subtree-split |
| `semantic-dynamics/` | `semantic-dynamics/` | `cmri-semantic-dynamics` | subtree-split |
| `operational-memex/` | `operational-memex/` | `cmri-operational-memex` | subtree-split (trivial history) |

**Conventions used in every task below:**
- `UMB` is the absolute umbrella path. Export it once per shell session:
  `export UMB=/home/spinoza/github/papers/cognitive-mri-ai-conversations`
- "Verify" steps replace unit tests for this migration: each is a command plus the expected output.
- Avoid em-dashes in any file you write (a repo hook rejects them); use commas, colons, or parentheses.

---

## Task 0: Preflight checks

**Files:** none (read-only verification).

- [ ] **Step 1: Confirm tools and location**

Run:
```bash
export UMB=/home/spinoza/github/papers/cognitive-mri-ai-conversations
cd "$UMB"
git --version
git subtree --help >/dev/null 2>&1 && echo "subtree OK" || echo "subtree MISSING"
command -v rsync >/dev/null && echo "rsync OK" || echo "rsync MISSING"
gh auth status 2>&1 | grep -q "Logged in" && echo "gh OK" || echo "gh NOT AUTHED"
git remote -v
```
Expected:
```
git version 2.43.0
subtree OK
rsync OK
gh OK
origin	https://github.com/queelius/cognitive-mri-conversations.git (fetch)
origin	https://github.com/queelius/cognitive-mri-conversations.git (push)
```
If any tool is MISSING or gh NOT AUTHED, stop and resolve before continuing.

- [ ] **Step 2: Confirm the five source groups exist**

Run:
```bash
for d in agentic comp-net-2025-camera-ready comp-net-2025-journal embedding-dynamics semantic-dynamics operational-memex code data; do
  [ -d "$UMB/$d" ] && echo "present: $d" || echo "MISSING: $d"
done
```
Expected: all eight print `present:`.

---

## Task 1: Reconcile the working tree and tag a rollback point

**Why:** `git subtree split` only sees committed history. Any uncommitted work would be silently dropped from the child repos. We commit everything we want to keep, ignore the regenerable junk, then tag.

**Files:**
- Modify: `.gitignore`
- Commit: all currently-modified and intended-untracked paths.
- Tag: `pre-cluster-reorg-2026-06-08`

- [ ] **Step 1: Append build/junk patterns to `.gitignore`**

Append these lines to `$UMB/.gitignore` (create the block if absent):
```gitignore
# --- reorg: regenerable build artifacts and caches ---
.coverage
.playwright-mcp/
**/__pycache__/
**/.pytest_cache/
**/venv/
**/.venv/
**/html_paper/
**/plos-template/
**/plos-template.zip
**/CNALatexTemplate/main.pdf
**/*-eps-converted-to.pdf
*.aux
*.bbl
*.blg
*.out
*.synctex.gz
```

- [ ] **Step 2: Review what will be committed**

Run:
```bash
cd "$UMB"
git add -A
git status --short | sed 's/^/  /' | head -80
```
Expected: a list of staged changes. Confirm it includes `operational-memex/`, `future-ideas/`, the moved `compsac-2026` (root deletions plus `future-ideas/compsac-2026/` additions), the modified figures/PDFs, and that it does NOT include `.coverage`, `.playwright-mcp/`, any `venv/`, `__pycache__/`, `html_paper/`, or `plos-template/` (those should now be ignored). If any junk is still staged, add its pattern to `.gitignore`, run `git rm -r --cached <path>`, and re-check.

- [ ] **Step 3: Commit the reconciled tree**

Run:
```bash
cd "$UMB"
git commit -m "chore: reconcile working tree before paper-cluster reorg"
```
Expected: one commit created.

- [ ] **Step 4: Verify clean tree, then tag**

Run:
```bash
cd "$UMB"
git status --porcelain | head
git tag pre-cluster-reorg-2026-06-08
git tag --list 'pre-cluster-reorg-*'
```
Expected: `git status --porcelain` prints nothing (clean), and the tag is listed. A clean tree here is a hard gate; do not proceed otherwise.

---

## Task 2: Create history-preserving split branches

**Why:** Produce one branch per source dir whose history contains only that dir's commits, re-rooted. These branches feed the child repos.

**Files:** none on disk; creates local branches `split/*` in the umbrella.

- [ ] **Step 1: Split each source dir**

Run:
```bash
cd "$UMB"
git subtree split --prefix=agentic                    -b split/hierarchical-memory
git subtree split --prefix=embedding-dynamics         -b split/embedding-dynamics
git subtree split --prefix=semantic-dynamics          -b split/semantic-dynamics
git subtree split --prefix=operational-memex          -b split/operational-memex
git subtree split --prefix=comp-net-2025-camera-ready -b split/cn-conference
git subtree split --prefix=comp-net-2025-journal      -b split/cn-journal
git subtree split --prefix=code                       -b split/cn-code
git subtree split --prefix=data                       -b split/cn-data
```
Expected: each command ends with a line like `1234abc` and the branch name printed (`-b` creates the branch silently on success).

- [ ] **Step 2: Verify the split branches exist and have history**

Run:
```bash
cd "$UMB"
for b in hierarchical-memory embedding-dynamics semantic-dynamics operational-memex cn-conference cn-journal cn-code cn-data; do
  printf "%-22s %s commits\n" "split/$b" "$(git rev-list --count split/$b)"
done
```
Expected (counts approximate; nonzero is what matters): `split/hierarchical-memory` ~41, `split/semantic-dynamics` ~7, `split/cn-conference` ~32, `split/cn-journal` ~15, `split/cn-code` ~8, `split/cn-data` ~4, `split/embedding-dynamics` ~1, `split/operational-memex` ~1.

---

## Task 3: Build the `comp-net` child repo (conference + journal + code + data)

**Files:**
- Create: `papers/comp-net/` (new git repo) with subdirs `conference/`, `journal/`, `code/`, `data/`
- Create: `papers/comp-net/conference/springer-cognitive-mri.pdf` (moved copy of the root PDF)
- Create: `papers/comp-net/README.md`

- [ ] **Step 1: Initialize the repo with an empty root commit**

Run:
```bash
cd "$UMB"
mkdir -p papers/comp-net
cd papers/comp-net
git init -b main
git commit --allow-empty -m "chore: initialize comp-net repository"
```
Expected: `Initialized empty Git repository` and one empty commit.

- [ ] **Step 2: Merge each split branch under its subdir (history preserved)**

Run:
```bash
cd "$UMB/papers/comp-net"
git subtree add --prefix=conference "$UMB" split/cn-conference
git subtree add --prefix=journal    "$UMB" split/cn-journal
git subtree add --prefix=code       "$UMB" split/cn-code
git subtree add --prefix=data       "$UMB" split/cn-data
```
Expected: each prints `Added dir '<prefix>'` and creates a merge commit.

- [ ] **Step 3: Layer in untracked working data (skip regenerable venvs/caches)**

Run:
```bash
cd "$UMB/papers/comp-net"
RS="rsync -a --ignore-existing --exclude venv/ --exclude .venv/ --exclude __pycache__/ --exclude .pytest_cache/ --exclude node_modules/"
$RS "$UMB/comp-net-2025-camera-ready/" conference/
$RS "$UMB/comp-net-2025-journal/"      journal/
$RS "$UMB/code/"                        code/
$RS "$UMB/data/"                        data/
```
Expected: rsync runs without error. (Tracked files already present are skipped; only untracked/ignored data is copied.)

- [ ] **Step 4: Move the root Springer PDF into conference/**

Run:
```bash
cd "$UMB/papers/comp-net"
cp "$UMB/springer-cognitive-mri.pdf" conference/springer-cognitive-mri.pdf
git add conference/springer-cognitive-mri.pdf
git commit -m "chore: place published Springer PDF under conference/"
```
Expected: one commit adding the PDF.

- [ ] **Step 5: Write the child README**

Create `papers/comp-net/README.md` with this content:
```markdown
# Cognitive MRI: Complex-Network Analysis of AI Conversations (comp-net lineage)

Part of the Cognitive MRI of AI Conversations research program
(umbrella: https://github.com/queelius/cognitive-mri-conversations).

This repository holds two papers that share one pipeline and dataset:

- `conference/`: Complex Networks 2025 conference paper (Springer, published).
- `journal/`: journal extension (PLOS submission rejected 2026-04-27; seeking a venue).
- `code/`: the shared analysis pipeline (a working mirror of `chatgpt-complex-net`).
- `data/`: curated reproducibility artifacts.

## Dependencies

- Conversation corpus: https://github.com/queelius/chatgpt-conversation-corpus
- Pipeline of record: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)

Raw corpus is read from `$CMRI_CORPUS_DIR` (defaults to the umbrella `dev/`).
```

- [ ] **Step 6: Commit the README and verify the repo**

Run:
```bash
cd "$UMB/papers/comp-net"
git add README.md
git commit -m "docs: add comp-net repository README"
git status --porcelain
git log --oneline | head -15
ls conference journal code data
```
Expected: `git status --porcelain` is empty (clean; any copied extras are gitignored by the subdir `.gitignore` files that came with history). The log shows the merge commits plus the README/PDF commits. All four subdirs list their expected contents.

- [ ] **Step 7: Verify comp-net LaTeX cross-references survived the move**

Run:
```bash
cd "$UMB/papers/comp-net"
grep -rIn --include='*.tex' -E '\\(input|include|includegraphics|bibliography)\{[^}]*\.\./' conference journal | head -40
```
Expected: review each hit. The risk is references that climbed out of the old `comp-net-2025-camera-ready/` (now `conference/`) toward the old root `code/`, `data/`, or `images/`. If a path points to a sibling that is now at `papers/comp-net/<x>/`, fix the depth (from `conference/paper/`, the sibling `code/` is `../../code`). If no hits, no fix needed. After any edit:
```bash
cd "$UMB/papers/comp-net/conference/paper" 2>/dev/null && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/tmp/cn-conf-build.log 2>&1; tail -3 /tmp/cn-conf-build.log
```
Expected: a PDF is produced (`Output written on paper.pdf`). Pre-existing build issues unrelated to paths are out of scope; the goal is only that the move did not break includes. Commit any path fixes with `git commit -am "fix: repair cross-references after directory move"`.

---

## Task 4: Build the `hierarchical-memory` child repo (from `agentic/`)

**Files:**
- Create: `papers/hierarchical-memory/` (new git repo)
- Create: `papers/hierarchical-memory/README.md` (only if one is not already present from history)

- [ ] **Step 1: Create the repo from the split branch (history preserved)**

Run:
```bash
cd "$UMB"
mkdir -p papers/hierarchical-memory
cd papers/hierarchical-memory
git init -b main
git fetch "$UMB" split/hierarchical-memory
git reset --hard FETCH_HEAD
```
Expected: `HEAD is now at <sha> ...` and a populated working tree.

- [ ] **Step 2: Layer in untracked working data**

Run:
```bash
cd "$UMB/papers/hierarchical-memory"
rsync -a --ignore-existing --exclude venv/ --exclude .venv/ --exclude __pycache__/ \
  --exclude .pytest_cache/ --exclude node_modules/ "$UMB/agentic/" ./
git status --porcelain | head
```
Expected: empty or only paths already covered by the repo's `.gitignore`. If a copied data file shows as untracked and is regenerable, add its pattern to `.gitignore`; if it is source-of-truth, `git add` and commit it with `git commit -m "chore: include generated artifacts"`.

- [ ] **Step 3: Ensure a top-level README exists**

Run:
```bash
cd "$UMB/papers/hierarchical-memory"
test -f README.md && echo "README present" || echo "need README"
```
If "need README", create `README.md`:
```markdown
# From Episodes to Abstractions: Latent Hierarchical Memory in AI Conversations

Part of the Cognitive MRI of AI Conversations research program
(umbrella: https://github.com/queelius/cognitive-mri-conversations).
Target venue: ISCS 2026.

## Dependencies
- Conversation corpus: https://github.com/queelius/chatgpt-conversation-corpus
- Base pipeline: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)
Raw corpus is read from `$CMRI_CORPUS_DIR` (defaults to the umbrella `dev/`).
```
Then:
```bash
git add README.md && git commit -m "docs: add repository README"
```

- [ ] **Step 4: Verify**

Run:
```bash
cd "$UMB/papers/hierarchical-memory"
git log --oneline | wc -l
git status --porcelain
ls paper experiments 2>/dev/null
```
Expected: a nonzero commit count (roughly 41 plus any new commits), clean status, and the paper/experiment dirs present.

---

## Task 5: Build the `embedding-dynamics` child repo

**Files:**
- Create: `papers/embedding-dynamics/` (new git repo)
- Create: `papers/embedding-dynamics/README.md` (only if absent)

- [ ] **Step 1: Create the repo from the split branch**

Run:
```bash
cd "$UMB"
mkdir -p papers/embedding-dynamics
cd papers/embedding-dynamics
git init -b main
git fetch "$UMB" split/embedding-dynamics
git reset --hard FETCH_HEAD
```
Expected: `HEAD is now at <sha> ...`.

- [ ] **Step 2: Layer in untracked working data**

Run:
```bash
cd "$UMB/papers/embedding-dynamics"
rsync -a --ignore-existing --exclude venv/ --exclude .venv/ --exclude __pycache__/ \
  --exclude .pytest_cache/ --exclude node_modules/ "$UMB/embedding-dynamics/" ./
git status --porcelain | head
```
Expected: empty or only gitignored paths (resolve as in Task 4 Step 2).

- [ ] **Step 3: Ensure a top-level README exists**

Run:
```bash
cd "$UMB/papers/embedding-dynamics"
test -f README.md && echo "README present" || echo "need README"
```
If "need README", create `README.md`:
```markdown
# Embedding Dynamics

Part of the Cognitive MRI of AI Conversations research program
(umbrella: https://github.com/queelius/cognitive-mri-conversations).

## Dependencies
- Conversation corpus: https://github.com/queelius/chatgpt-conversation-corpus
- Base pipeline: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)
Raw corpus is read from `$CMRI_CORPUS_DIR` (defaults to the umbrella `dev/`).
```
Then:
```bash
git add README.md && git commit -m "docs: add repository README"
```

- [ ] **Step 4: Verify**

Run:
```bash
cd "$UMB/papers/embedding-dynamics"
git log --oneline | head
git status --porcelain
ls paper experiments 2>/dev/null
```
Expected: history present, clean status, expected dirs present.

---

## Task 6: Build the `semantic-dynamics` child repo

**Files:**
- Create: `papers/semantic-dynamics/` (new git repo)
- Create: `papers/semantic-dynamics/README.md` (only if absent)

- [ ] **Step 1: Create the repo from the split branch**

Run:
```bash
cd "$UMB"
mkdir -p papers/semantic-dynamics
cd papers/semantic-dynamics
git init -b main
git fetch "$UMB" split/semantic-dynamics
git reset --hard FETCH_HEAD
```
Expected: `HEAD is now at <sha> ...`.

- [ ] **Step 2: Layer in untracked working data**

Run:
```bash
cd "$UMB/papers/semantic-dynamics"
rsync -a --ignore-existing --exclude venv/ --exclude .venv/ --exclude __pycache__/ \
  --exclude .pytest_cache/ --exclude node_modules/ "$UMB/semantic-dynamics/" ./
git status --porcelain | head
```
Expected: empty or only gitignored paths (resolve as in Task 4 Step 2).

- [ ] **Step 3: Ensure a top-level README exists**

Run:
```bash
cd "$UMB/papers/semantic-dynamics"
test -f README.md && echo "README present" || echo "need README"
```
If "need README", create `README.md`:
```markdown
# Semantic Dynamics

Part of the Cognitive MRI of AI Conversations research program
(umbrella: https://github.com/queelius/cognitive-mri-conversations).

## Dependencies
- Conversation corpus: https://github.com/queelius/chatgpt-conversation-corpus
- Base pipeline: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)
Raw corpus is read from `$CMRI_CORPUS_DIR` (defaults to the umbrella `dev/`).
```
Then:
```bash
git add README.md && git commit -m "docs: add repository README"
```

- [ ] **Step 4: Verify**

Run:
```bash
cd "$UMB/papers/semantic-dynamics"
git log --oneline | head
git status --porcelain
ls paper experiments 2>/dev/null
```
Expected: history present (roughly 7 commits), clean status, expected dirs present.

---

## Task 7: Build the `operational-memex` child repo

**Files:**
- Create: `papers/operational-memex/` (new git repo)
- Create: `papers/operational-memex/.gitignore` (if absent)
- Create: `papers/operational-memex/README.md` (if absent)

- [ ] **Step 1: Create the repo from the split branch**

Run:
```bash
cd "$UMB"
mkdir -p papers/operational-memex
cd papers/operational-memex
git init -b main
git fetch "$UMB" split/operational-memex
git reset --hard FETCH_HEAD
```
Expected: `HEAD is now at <sha> ...` (this dir had only the Task 1 commit, so the history is short).

- [ ] **Step 2: Layer in untracked working data**

Run:
```bash
cd "$UMB/papers/operational-memex"
rsync -a --ignore-existing --exclude venv/ --exclude .venv/ --exclude __pycache__/ \
  --exclude .pytest_cache/ --exclude node_modules/ "$UMB/operational-memex/" ./
git status --porcelain | head
```
Expected: possibly some untracked files (this paper is new and may lack a `.gitignore`).

- [ ] **Step 3: Ensure `.gitignore` and README exist, then commit any real additions**

If `git status` showed untracked caches/outputs, create `.gitignore`:
```gitignore
__pycache__/
*.pyc
venv/
.venv/
.pytest_cache/
data/embeddings/
output/
```
If `README.md` is absent, create it:
```markdown
# Operational Memex

Part of the Cognitive MRI of AI Conversations research program
(umbrella: https://github.com/queelius/cognitive-mri-conversations).

## Dependencies
- Conversation corpus: https://github.com/queelius/chatgpt-conversation-corpus
- Base pipeline: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)
Raw corpus is read from `$CMRI_CORPUS_DIR` (defaults to the umbrella `dev/`).
```
Then stage source-of-truth files (code, paper) and commit:
```bash
cd "$UMB/papers/operational-memex"
git add -A
git commit -m "chore: add gitignore, README, and tracked sources"
git status --porcelain
```
Expected: clean status after commit.

- [ ] **Step 4: Verify**

Run:
```bash
cd "$UMB/papers/operational-memex"
git log --oneline | head
ls paper 2>/dev/null
```
Expected: at least one commit, paper dir present.

---

## Task 8: Create private GitHub remotes and push all five children

**Why:** Off-machine backup immediately after extraction, before touching the umbrella.

**Files:** none locally; creates five private GitHub repos under `queelius`.

- [ ] **Step 1: Confirm intended repo names with the operator**

The repos to be created (private) are: `cmri-comp-net`, `cmri-hierarchical-memory`, `cmri-embedding-dynamics`, `cmri-semantic-dynamics`, `cmri-operational-memex`. Creating remotes is outward-facing; confirm these names are acceptable before running Step 2. (Renaming later is `gh repo rename`, but confirm now to avoid churn.)

- [ ] **Step 2: Create and push each remote**

Run:
```bash
cd "$UMB/papers/comp-net"            && gh repo create queelius/cmri-comp-net            --private --source=. --remote=origin --push
cd "$UMB/papers/hierarchical-memory" && gh repo create queelius/cmri-hierarchical-memory --private --source=. --remote=origin --push
cd "$UMB/papers/embedding-dynamics"  && gh repo create queelius/cmri-embedding-dynamics  --private --source=. --remote=origin --push
cd "$UMB/papers/semantic-dynamics"   && gh repo create queelius/cmri-semantic-dynamics   --private --source=. --remote=origin --push
cd "$UMB/papers/operational-memex"   && gh repo create queelius/cmri-operational-memex   --private --source=. --remote=origin --push
```
Expected: each prints the created repo URL and a push summary. If a push is rejected for a file over 100 MB, stop: that file needs Git LFS or removal from history (out of scope here; note it and consult the operator). The largest known blobs (Springer PDF 46 MB, agentic `.npy` 35 MB) are under the limit.

- [ ] **Step 3: Verify remotes**

Run:
```bash
for n in comp-net hierarchical-memory embedding-dynamics semantic-dynamics operational-memex; do
  echo "== $n =="; git -C "$UMB/papers/$n" remote -v | head -1
done
gh repo list queelius --limit 100 | grep cmri-
```
Expected: each child has an `origin` pointing at its `cmri-*` repo, and all five appear in `gh repo list`.

---

## Task 9: Slim the umbrella working tree

**Why:** The umbrella now ignores `papers/*` and drops the original source dirs from its working tree (their content lives in the children). History is retained in the umbrella as the archive of record.

**Files:**
- Modify: `$UMB/.gitignore` (add `papers/`)
- Remove (tracked): `agentic`, `comp-net-2025-camera-ready`, `comp-net-2025-journal`, `embedding-dynamics`, `semantic-dynamics`, `operational-memex`, `code`, `data`, `springer-cognitive-mri.pdf`

- [ ] **Step 1: Gate check, children must be pushed**

Run:
```bash
for n in comp-net hierarchical-memory embedding-dynamics semantic-dynamics operational-memex; do
  git -C "$UMB/papers/$n" rev-parse --abbrev-ref HEAD >/dev/null 2>&1 && \
  git -C "$UMB/papers/$n" ls-remote origin -h refs/heads/main >/dev/null 2>&1 && echo "OK: $n pushed" || echo "STOP: $n not pushed"
done
```
Expected: five `OK:` lines. If any says STOP, do not delete originals; finish Task 8 first.

- [ ] **Step 2: Ignore the papers/ tree in the umbrella**

Append to `$UMB/.gitignore`:
```gitignore
# --- reorg: child paper repos are independent, not tracked by the umbrella ---
papers/
```

- [ ] **Step 3: Remove originals from umbrella tracking and disk**

Run:
```bash
cd "$UMB"
git rm -r --quiet agentic comp-net-2025-camera-ready comp-net-2025-journal \
  embedding-dynamics semantic-dynamics operational-memex code data springer-cognitive-mri.pdf
```
Expected: many `rm '...'` lines (tracked files removed from index and working tree).

- [ ] **Step 4: Delete leftover untracked/ignored remnants of the originals**

Run:
```bash
cd "$UMB"
for d in agentic comp-net-2025-camera-ready comp-net-2025-journal embedding-dynamics semantic-dynamics operational-memex code data; do
  [ -d "$d" ] && rm -rf "$d" && echo "removed leftover: $d"
done
ls -d papers/*/ 2>/dev/null
```
Expected: leftovers removed (these held only gitignored files such as venvs, already excluded from children or regenerable). The `papers/` children remain on disk. Confirm `papers/comp-net papers/hierarchical-memory papers/embedding-dynamics papers/semantic-dynamics papers/operational-memex` all still exist before trusting this step.

- [ ] **Step 5: Verify the umbrella sees a clean, slimmed tree**

Run:
```bash
cd "$UMB"
git status --short | head
git check-ignore papers/comp-net && echo "papers/ ignored OK"
```
Expected: staged deletions of the eight dirs plus the PDF and the `.gitignore` change; `papers/` is ignored; no child repo content appears as untracked.

---

## Task 10: Rewrite the umbrella into a landing page and commit

**Files:**
- Rewrite: `$UMB/README.md`
- Rewrite: `$UMB/CLAUDE.md` (structure section only; keep build/pipeline notes that still apply)
- Keep: `$UMB/CITATION.cff`, `$UMB/.zenodo.json` (unchanged role)

- [ ] **Step 1: Write the new umbrella README**

Replace `$UMB/README.md` with:
```markdown
# Cognitive MRI of AI Conversations (research compendium)

A research program applying complex-network analysis to large archives of AI
conversations (built on a corpus of ChatGPT conversations). This repository is the
umbrella: a landing page indexing the individual paper repositories and the shared
data and pipeline.

## Papers

Each paper is its own git repository under `papers/` (not tracked by this umbrella).

| Paper | Venue / status | Path | Repo |
|---|---|---|---|
| Cognitive MRI: complex-network analysis (conference) | Complex Networks 2025, Springer (published) | `papers/comp-net/conference/` | `queelius/cmri-comp-net` |
| Temporal evolution of cognitive knowledge networks (journal) | PLOS submission rejected 2026-04-27; seeking venue | `papers/comp-net/journal/` | `queelius/cmri-comp-net` |
| From Episodes to Abstractions: latent hierarchical memory | ISCS 2026 | `papers/hierarchical-memory/` | `queelius/cmri-hierarchical-memory` |
| Embedding Dynamics | in progress | `papers/embedding-dynamics/` | `queelius/cmri-embedding-dynamics` |
| Semantic Dynamics | in progress | `papers/semantic-dynamics/` | `queelius/cmri-semantic-dynamics` |
| Operational Memex | early draft | `papers/operational-memex/` | `queelius/cmri-operational-memex` |

## Shared data and pipeline

- Conversation corpus (dataset): https://github.com/queelius/chatgpt-conversation-corpus
- Analysis pipeline of record: `chatgpt-complex-net` (Zenodo DOI 10.5281/zenodo.15314235)
- Local working raw corpus lives in `dev/` (gitignored). Child code reads it via
  `$CMRI_CORPUS_DIR`, which defaults to this repo's `dev/`.

## How this repo is organized

`papers/*` are independent repositories. Clone them individually, or clone this umbrella
and populate `papers/` by cloning each `cmri-*` repo into place. The umbrella retains the
full project history (the archive of record) and the compendium citation.

## Citation

Cite the specific paper you use (see each paper repo). For the program as a whole, cite the
compendium via `CITATION.cff` / the Zenodo record.
```

- [ ] **Step 2: Update CLAUDE.md structure section**

In `$UMB/CLAUDE.md`, replace the `## Project Structure` tree (and any directory-specific paths that no longer exist at the umbrella root) with a short description of the new layout: the umbrella holds `README.md`, `CITATION.cff`, `.zenodo.json`, `docs/`, `future-ideas/`, gitignored `dev/`, and `papers/` (independent child repos, gitignored). Note that build instructions and the pipeline description now live inside `papers/comp-net/`. Keep the user-identity and research-parameter sections intact. Do not use em-dashes.

- [ ] **Step 3: Commit the slimmed umbrella**

Run:
```bash
cd "$UMB"
git add -A
git commit -m "refactor: split papers into independent repos under papers/, slim umbrella to compendium landing page"
git status --porcelain
```
Expected: one commit; clean status afterward.

- [ ] **Step 4: Push the umbrella**

Run:
```bash
cd "$UMB"
git push origin HEAD
```
Expected: push succeeds to `cognitive-mri-conversations` (only the new commits transfer; historical blobs already exist on the remote).

---

## Task 11: Update the global repo catalog

**Files:**
- Modify: `/home/spinoza/github/CLAUDE.md`

- [ ] **Step 1: Update the catalog entry**

In `/home/spinoza/github/CLAUDE.md`, find the `papers/` table row for `cognitive-mri-ai-conversations/` and its `agentic/` sub-paper row. Update the description to note it is now a paper cluster (umbrella + `papers/` of independent repos), and replace the single `agentic/` sub-row with the five children: `comp-net` (conference + journal), `hierarchical-memory` (formerly `agentic`), `embedding-dynamics`, `semantic-dynamics`, `operational-memex`. Keep the table format consistent with the surrounding rows. Do not use em-dashes.

- [ ] **Step 2: Verify**

Run:
```bash
grep -n "cmri-\|cognitive-mri\|hierarchical-memory" /home/spinoza/github/CLAUDE.md | head
```
Expected: the new entries are present. (This file may be in its own repo; commit it there only if the operator wants, per the "commit when asked" rule.)

---

## Task 12: Final verification sweep and cleanup

**Files:** none (verification); deletes local `split/*` branches.

- [ ] **Step 1: Per-child sanity**

Run:
```bash
for n in comp-net hierarchical-memory embedding-dynamics semantic-dynamics operational-memex; do
  echo "== $n =="
  git -C "$UMB/papers/$n" status --porcelain | head -3
  git -C "$UMB/papers/$n" log --oneline -1
done
```
Expected: every child is clean and has a sensible HEAD commit.

- [ ] **Step 2: Umbrella sanity**

Run:
```bash
cd "$UMB"
git remote -v | head -1
git status --porcelain
test -f CITATION.cff && test -f .zenodo.json && echo "citation files intact"
git check-ignore papers/comp-net >/dev/null && echo "papers/ ignored"
ls
```
Expected: remote unchanged, clean status, citation files intact, `papers/` ignored, and the top level shows the slimmed layout (`README.md`, `CITATION.cff`, `.zenodo.json`, `docs/`, `future-ideas/`, `papers/`, `dev/`).

- [ ] **Step 3: Confirm rollback tag still points at the pre-reorg state**

Run:
```bash
cd "$UMB"
git show --stat pre-cluster-reorg-2026-06-08 | head -5
```
Expected: the tag resolves to the Task 1 commit (the full pre-split tree is recoverable from here).

- [ ] **Step 4: Delete the temporary split branches**

Run:
```bash
cd "$UMB"
git branch -D split/hierarchical-memory split/embedding-dynamics split/semantic-dynamics \
  split/operational-memex split/cn-conference split/cn-journal split/cn-code split/cn-data
git branch --list 'split/*'
```
Expected: branches deleted; the final `git branch --list` prints nothing.

---

## Done criteria

- Five independent repos exist under `papers/`, each clean, with preserved history and a private `cmri-*` remote that has been pushed.
- `comp-net/` builds the conference paper (or has only pre-existing, non-path build issues).
- The umbrella is slimmed to a landing page, still on its `cognitive-mri-conversations` remote, with `CITATION.cff` / `.zenodo.json` intact and `papers/` gitignored.
- `pre-cluster-reorg-2026-06-08` tags the recoverable pre-migration state.
- The global catalog (`~/github/CLAUDE.md`) reflects the new cluster shape.
