# Paper Spec: Embedding Dynamics

**Title:** Embedding Dynamics: Treating Text Sequences as Trajectories in Semantic Space

**Author:** Alexander Towell, Southern Illinois University Edwardsville

**Type:** Venue-agnostic preprint (arXiv). Theory + validation paper.

**Target length:** 12-18 pages (article class, 11pt, 1-inch margins)

---

## 1. Central Claim

When a text is composed of ordered sub-units (messages in a conversation, paragraphs in a document, sections in a paper), each sub-unit can be embedded independently. The standard practice is to treat these embeddings as the semantic content itself and aggregate them (typically by averaging) to get a document-level representation.

We propose a reinterpretation: **each sub-unit embedding is not the semantic state; it is an impulse (force) that changes the semantic state.** The semantic state at position k is the accumulated effect of all sub-units up to k. This accumulated state is what we call the **trajectory**: the path of the document through embedding space as it unfolds.

This reinterpretation is not merely a relabeling. It produces a different derivative tower, a different notion of changepoint, and a different set of operations on sequences. It also makes a testable empirical prediction: the accumulated embedding at position k should approximate the embedding of the full text from positions 1 through k (the prefix embedding).

## 2. The Foundational Experiment

**This experiment is the linchpin. It must be run before the paper can be written with confidence.**

**Setup:** Take a text sequence (e.g., a conversation) with n sub-units (messages). For each position k from 1 to n:
- Compute embed_prefix(k) = embed(concat(sub_1, ..., sub_k)). This is the "true" semantic state at time k.
- Compute embed_accum(k) = lens.project(embed(sub_1), ..., embed(sub_k)). This is the accumulated embedding at time k, using an exponential lens with decay parameter alpha.

**Measure:** The cosine similarity between embed_prefix(k) and embed_accum(k) at each k, for various alpha values.

**Expected result (assumed for this spec):** Strong agreement (cosine similarity > 0.8) across most positions and conversation types, with the optimal alpha varying by conversation type. This would validate the entire framework.

**Test cases:**
1. Focused conversation (stays on one topic): both methods should agree strongly at all k.
2. Drifting conversation (gradual topic shift): agreement should be high, with the optimal alpha tracking the drift rate.
3. Sharp pivot (abrupt topic change): agreement may temporarily drop at the pivot point, then recover. The accumulation "forgets" the old topic more slowly than the prefix embedding does.
4. Return to topic: tests whether the accumulation captures the return or stays biased toward the intervening topic.

**Script:** Already written at semantic-dynamics/experiments/exp04_prefix_vs_accumulation.py. Needs an embedding API to run.

**Variants to explore:**
- Which alpha gives the best agreement across the full trajectory? (Is there an "optimal filter" for approximating prefix embeddings?)
- Does the optimal alpha match the conversation's adaptive alpha (estimated from the data)?
- Does the agreement depend on the embedding model? (Test with multiple models if possible.)
- How does agreement change with sequence length? (Does it degrade for very long sequences?)

## 3. Paper Structure (Layered: Core + Extensions)

### Layer 1: The Core Interpretation (Sections 1-4)

This is the sharp theoretical claim. It should be comprehensible and compelling on its own.

#### Section 1: Introduction

Open with the problem: text sequences (conversations, documents, transcripts) are composed of ordered sub-units. Standard practice embeds these sub-units and averages them. This discards the temporal structure.

State the core idea in one paragraph: "We propose treating each sub-unit embedding as a force applied to the document's semantic state. The semantic state at position k is the accumulated effect of all forces up to k. This accumulated state traces a trajectory through embedding space. The trajectory has velocity, acceleration, and drift, all of which are computable from the sub-unit embeddings alone."

State the key empirical prediction: "The accumulated embedding at position k should approximate the embedding of the full prefix from 1 to k."

State the validation: "We test this prediction on [N] conversations across [3] AI platforms and find [cosine similarity > 0.8 / strong agreement / etc.]."

State the consequences: "This reinterpretation yields a calculus over embedding sequences with practical applications for changepoint detection, trajectory comparison, and document representation."

#### Section 2: Background and Related Work

**2.1 Embedding aggregation.** Mean pooling, weighted pooling, attention-based aggregation. The observation that averaging embeddings works surprisingly well (the high-dimensional near-orthogonality argument from Arora et al.). Position-weighted schemes (MEGA, DemaFormer, ETSformer).

**2.2 Temporal network embeddings.** weg2vec and its use of exponential decay for event embeddings. The distinction: they embed events in a network, we embed sub-units in a sequence.

**2.3 Time series analysis of text.** Topic segmentation (TextTiling, Bayesian changepoint detection). The distinction: these operate on raw text features, we operate in embedding space.

**2.4 Dynamical systems.** Brief note on the analogy to physics. We are not claiming text IS a dynamical system; we are claiming the same mathematical operations (accumulation, differentiation, filtering) that reveal structure in physical trajectories also reveal structure in embedding trajectories.

#### Section 3: The Embedding Flow Calculus

The theoretical heart of the paper. This section should be 3-4 pages.

**3.1 Setup.** A text sequence consists of n sub-units. Each sub-unit is embedded by a frozen pre-trained model, producing vectors e_1, ..., e_n in R^d. The sub-units are ordered (by position in the text, or by timestamp).

**3.2 The standard interpretation: sub-units as semantic positions.** Under this view, e_k IS the semantic content of sub-unit k. The document embedding is the average: (1/n) sum e_k. Velocity is e_{k+1} - e_k (how content changes). Curvature is the second difference. This is what most prior work assumes, explicitly or implicitly.

**3.3 The proposed interpretation: sub-units as semantic forces.** Under this view, e_k is the impulse applied to the document's semantic state at time k. The semantic state at time k is:

x(k) = normalize(sum_{j=1}^{k} w(j,k) * e_j)

where w(j,k) is a weighting function (the lens). For the uniform lens, w=1 and x(k) is the running mean. For the exponential lens, w(j,k) = alpha^{k-j} and x(k) is an exponentially weighted running average.

The key difference: under the standard view, the "trajectory" is the sequence of raw embeddings (e_1, ..., e_n). Under the proposed view, the trajectory is the sequence of accumulated states (x(1), ..., x(n)).

**3.4 The derivative tower.** Present the full tower with both interpretations side by side:

| Level | Standard (sub-unit = position) | Proposed (sub-unit = force) |
|-------|-------------------------------|----------------------------|
| Integral of position | Not typically computed | "Absement": how long you lingered where |
| Position | e_k (the sub-unit embedding) | x(k) = accumulated embedding |
| Velocity | e_{k+1} - e_k | dx/dk = how the accumulated state changes |
| Acceleration | 2nd diff of e | e_{k+1} - e_k = change in force |
| Jerk | 3rd diff of e | 2nd diff of e = change in acceleration |

The critical observation: under the proposed interpretation, the raw sub-unit difference ||e_{k+1} - e_k|| is the magnitude of ACCELERATION, not velocity. A large value means the force changed abruptly: a changepoint. This gives a dynamics-native changepoint detector that falls directly out of the calculus.

**3.5 Lenses as signal filters.** Different weighting functions (lenses) correspond to different filters applied to the force signal:

- Uniform lens = box filter / simple integral. Position estimate = center of mass.
- Exponential lens = first-order IIR low-pass filter. Alpha = filter coefficient. Half-life = log(0.5)/log(alpha) = time constant.
- Gaussian lens = FIR bandpass filter centered at a focal point.
- Surprise lens = adaptive filter (weights depend on the signal itself).

Lenses compose by pointwise multiplication of weight vectors. Uniform is the identity. This is a monoid.

**3.6 The semantic half-life.** Alpha parameterizes the exponential filter. The half-life h = log(0.5)/log(alpha) is the number of sub-units after which a given sub-unit's influence has halved. This is an interpretable, measurable quantity:
- It can be estimated per-document (adaptive alpha) by fitting the trajectory to predict the next sub-unit.
- It varies systematically by document type (in our case, by AI platform: ChatGPT conversations have shorter half-life than Claude Code sessions).
- It controls both the position estimate AND the noise filtering for changepoint detection. One parameter, dual interpretation.

**3.7 Changepoints from acceleration.** Under the proposed interpretation, a changepoint is a moment of high acceleration: ||e_{k+1} - e_k|| exceeds a threshold. The raw signal is noisy (individual sub-units may be anomalous). Smoothing via the exponential trajectory produces a filtered acceleration signal (the trajectory curvature). Alpha controls the noise/sensitivity tradeoff. At alpha=0.50 (heavy smoothing), 87% of smoothed changepoints agree with raw changepoints. At alpha=0.95 (light smoothing), 55% agree.

**3.8 The continuous-time view (brief).** When sub-units have timestamps, the time between them is not uniform. A message sent 3 hours after the previous one carries more "surprise" than one sent 30 seconds later. The TimeDecay lens handles this: w(j,k) = exp(-(t_k - t_j) / tau), where tau is a time constant in real units. The half-life becomes a physical time (e.g., "this conversation forgets context with a half-life of 45 minutes"). This is noted as a direction for future work, not fully developed.

#### Section 4: Empirical Validation

**4.1 The prefix experiment.** Present the setup, the test cases, and the results. This is the most important section of the paper. Show:
- Cosine similarity between accumulated and prefix embeddings at each k, for each conversation type.
- The optimal alpha for each conversation type.
- How agreement varies with k (does it improve or degrade as the sequence grows?).
- Side-by-side trajectory plots (PCA-projected accumulated trajectory vs prefix trajectory).

**4.2 Embedding model sensitivity (if possible).** Does the agreement hold across different embedding models? If we can test with both OpenAI and a local model (nomic, etc.), this strengthens the claim.

### Layer 2: Extensions (Sections 5-6)

These sections show what follows from the core interpretation. They are consequences, not the contribution itself.

#### Section 5: Applications

**5.1 Changepoint detection via acceleration.** Show that ||e_{k+1} - e_k|| detects meaningful topic shifts. Compare raw vs smoothed (trajectory curvature at various alpha). Show examples. Show that role transitions amplify acceleration by 1.57x. Show the acceleration profile is independent of topic (rho=0.16).

**5.2 Trajectory comparison.** Two sequences can be compared via DTW on their trajectories. Continuation scoring (end of A vs start of B). Shape similarity (curvature profile comparison). Brief demonstration.

**5.3 Multi-lens views.** Different lenses give different views of the same sequence. First-only vs Last-only have 0.07 neighbor overlap: where you started and where you ended are disjoint neighborhoods. Brief demonstration with the 8-lens overlap matrix.

#### Section 6: Case Study (Minimal)

A short section showing the framework applied to an AI conversation archive.

**6.1 Corpus.** 3,097 conversations, 149,623 per-message embeddings, 3 platforms. (Brief; full details are in the companion paper.)

**6.2 Platform-dependent half-life.** ChatGPT: alpha=0.76 (2.4 msg half-life). Claude Code: alpha=0.83 (4.3 msg half-life). The half-life reflects the interaction modality.

**6.3 Acceleration changepoints in practice.** The consciousness conversation (188 messages, 5 episodes). The game development session (1,764 messages, 43 episodes). Brief qualitative validation.

#### Section 7: Discussion

**7.1 What the reinterpretation buys.** The force/position interpretation is not just a relabeling. It changes:
- What "velocity" means (under standard: how content differs message-to-message. Under proposed: how the document's semantic state is evolving).
- What "acceleration" means (under standard: second difference of content. Under proposed: change in force = changepoint).
- What the mean embedding means (under standard: average content. Under proposed: center of mass of the trajectory).
- What alpha controls (under standard: a weighting parameter. Under proposed: the filter bandwidth of a position estimator).

**7.2 When would the approximation fail?** The accumulation-as-position hypothesis assumes that the embedding model's response to a concatenated text is approximately the weighted sum of its responses to the parts. This is known to be roughly true for many embedding models (the "bag of embeddings" literature). It would fail when:
- Context effects are very strong (a word means something completely different in context than in isolation).
- The sub-units are very short (a single word or token; the embedding carries almost no independent signal).
- The embedding model has strong positional biases (weights the beginning or end of the input disproportionately).

Discuss how the experiment results illuminate these failure modes.

**7.3 Connections.** The exponential trajectory is a Kalman filter with a specific noise model. The adaptive alpha is an estimate of the process noise. The lenses are a family of linear filters. These connections to signal processing and control theory are noted for future work.

**7.4 Limitations.** Single user corpus. One embedding model (with sensitivity analysis if possible). The "force" analogy is a metaphor; text is not literally a physical system. N=1 but N=3,097.

#### Section 8: Conclusion

The embedding flow calculus is a framework for treating text sequences as trajectories in semantic space. Its core insight is that sub-unit embeddings are better understood as forces than as positions. The accumulated trajectory is a principled, efficient approximation to the more expensive prefix embedding. From this interpretation, a complete set of operations follows: velocity, acceleration, changepoints, comparison, and filtering, all parameterized by a single decay rate with an interpretable meaning (the semantic half-life).

**Software.** The `embflow` Python package implements the calculus. Available at github.com/queelius/embflow.

---

## 4. Figures (Planned)

1. **Conceptual diagram:** The derivative tower with both interpretations side by side. Not a data figure; a conceptual illustration.

2. **Prefix vs accumulation:** Cosine similarity between embed_prefix(k) and embed_accum(k) as a function of k, for each conversation type and each alpha value. THE key figure.

3. **Trajectory comparison:** PCA projection of the accumulated trajectory vs the prefix trajectory for one conversation. Do they trace the same path?

4. **Acceleration changepoints:** Raw acceleration signal for a conversation, with detected changepoints marked. Show the smoothed version (trajectory curvature) below for comparison.

5. **Lens overlap matrix:** The 8x8 heatmap from the companion experiments.

6. **Alpha distribution:** Adaptive alpha by platform (real data, already generated).

## 5. Dependencies

- **embflow** package (~/github/beta/embflow). Already implemented and tested (35 tests).
- **Prefix experiment** (Exp04). Script written. MUST RUN BEFORE WRITING. Needs embedding API access.
- **Existing experiments** from semantic-dynamics/ and operational-memex/. Results can be referenced; some cherry-picked into this paper.

## 6. Relationship to Other Papers

| Paper | Scope | Shared infrastructure |
|-------|-------|----------------------|
| Paper 1 (comp-net-2025) | Semantic graph on ChatGPT conversations | Same corpus (ChatGPT portion) |
| Paper 2 (ISCS 2026) | Multiplex network on Claude Code | Same corpus (Claude Code portion) |
| Paper A (operational-memex) | Trails, marginalia, cross-source replication | Same embeddings, same analysis DB |
| Paper B (semantic-dynamics) | Orthogonal spaces, archetypes, half-life, continuation | Same experiments, different framing |
| **This paper (embedding-dynamics)** | **The theory: force/position, calculus, validation** | **embflow package, prefix experiment** |

Paper B (semantic-dynamics) becomes the empirical companion to this paper. This paper presents the theory and validates it. Paper B applies it at scale to a conversation archive and reports the detailed empirical findings.

## 7. Writing Principles

- **Theory first, experiments in service of theory.** Every experiment answers a question posed by the theory.
- **The reinterpretation is the contribution.** Not the experiments, not the package, not the conversation analysis. The idea that sub-unit embeddings are forces and their accumulation is position.
- **State the prediction before showing the data.** "If the reinterpretation is correct, we should see X. Figure Y shows that we do."
- **The derivative tower should feel inevitable.** Once the reader accepts messages-as-forces, velocity/acceleration/changepoints should follow naturally. They should not feel like additional claims.
- **Keep the case study brief.** This is not a paper about conversations. It's a paper about embedding dynamics that happens to demonstrate on conversations.
- **Acknowledge the metaphor.** "Force" and "position" are analogies. The math is the same (accumulation, differentiation). The physics vocabulary is adopted for interpretability, not because text literally obeys Newton's laws.

## 8. Risk

The entire paper depends on the prefix experiment (Section 4.1). If accumulation does NOT approximate prefix embedding:
- The "force/position" interpretation loses its empirical ground.
- The paper would need to pivot to: "embedding dynamics as a useful approximation even when the physics analogy breaks down."
- The calculus still works as a set of operations; it just can't claim to approximate the "true" semantic state.

The spec assumes the experiment succeeds. If it doesn't, revisit Section 3.3 and Section 4.1 before proceeding.
