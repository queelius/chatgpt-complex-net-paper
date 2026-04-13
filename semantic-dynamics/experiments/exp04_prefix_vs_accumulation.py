"""Experiment 4: Prefix embedding vs accumulated message embeddings.

Compares two ways of computing "the meaning of a conversation at time k":
1. embed_1(k) = embed(messages[1..k])  (prefix embedding: embed the full text)
2. embed_2(k) = accumulate(embed(msg_1), ..., embed(msg_k))  (accumulated impulses)

If these agree, then the cheap per-message accumulation is a good proxy
for the expensive full-prefix embedding. The lens parameter (alpha)
controls the accumulation, and we can find the alpha that best
approximates the prefix embedding.

Uses OpenAI text-embedding-3-small at 256 dims for both.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.expanduser("~/github/beta/embflow"))
import embflow as ef

# Sample conversations: diverse enough to test the hypothesis
SAMPLE_CONVERSATIONS = [
    {
        "id": "synthetic_focused",
        "title": "Focused technical discussion (stays on topic)",
        "messages": [
            {"role": "user", "content": "How does gradient descent work in neural networks?"},
            {"role": "assistant", "content": "Gradient descent is an optimization algorithm that iteratively adjusts the parameters of a neural network to minimize a loss function. At each step, it computes the gradient of the loss with respect to each parameter, then updates the parameters in the direction that reduces the loss. The learning rate controls the step size."},
            {"role": "user", "content": "What about stochastic gradient descent? How is it different?"},
            {"role": "assistant", "content": "Stochastic gradient descent (SGD) approximates the true gradient by computing it on a random subset (mini-batch) of the training data rather than the entire dataset. This introduces noise but makes each update much cheaper computationally. The noise can actually help escape local minima."},
            {"role": "user", "content": "And Adam optimizer? How does it improve on SGD?"},
            {"role": "assistant", "content": "Adam combines two ideas: momentum (keeping a running average of past gradients) and adaptive learning rates (scaling the learning rate for each parameter based on the variance of recent gradients). It maintains first and second moment estimates of the gradients and uses bias correction for the initial steps."},
            {"role": "user", "content": "When would you use Adam vs plain SGD?"},
            {"role": "assistant", "content": "Adam is generally a good default: it converges quickly and requires less tuning. However, SGD with momentum often generalizes better on well-studied tasks like image classification. The common wisdom is to start with Adam for prototyping, then try SGD with a tuned learning rate schedule for final training."},
            {"role": "user", "content": "What about learning rate scheduling? How does that work with these optimizers?"},
            {"role": "assistant", "content": "Learning rate scheduling reduces the learning rate during training according to a predefined rule. Common schedules include step decay (halve the rate every N epochs), cosine annealing (smoothly decrease and optionally restart), and warmup (start low, ramp up, then decay). Both Adam and SGD benefit from scheduling, though Adam is more robust to the initial learning rate choice."},
        ],
    },
    {
        "id": "synthetic_drift",
        "title": "Drifting conversation (topic shifts gradually)",
        "messages": [
            {"role": "user", "content": "I'm reading about the history of computing. Tell me about Ada Lovelace."},
            {"role": "assistant", "content": "Ada Lovelace, born in 1815, is often considered the first computer programmer. She worked with Charles Babbage on his Analytical Engine and wrote extensive notes including what is recognized as the first algorithm intended for machine processing. She also had remarkable insight into the potential of computing beyond mere calculation."},
            {"role": "user", "content": "She saw that machines could go beyond numbers. That's remarkable for the 1840s. What did she mean exactly?"},
            {"role": "assistant", "content": "Lovelace realized that the Analytical Engine could manipulate any symbols, not just numbers. She wrote that the engine could compose music, produce graphics, and be used for scientific purposes beyond arithmetic. This is essentially the concept of general-purpose computation, articulated a century before Turing."},
            {"role": "user", "content": "Speaking of Turing, how did his ideas build on this?"},
            {"role": "assistant", "content": "Turing formalized computation with his abstract machine in 1936. His key insight was that a single machine could simulate any other computational process, given the right instructions. This universality principle is the foundation of modern computing. Where Lovelace intuited the generality, Turing proved it mathematically."},
            {"role": "user", "content": "And the Turing test. That's about intelligence, not computation. How did he make that leap?"},
            {"role": "assistant", "content": "In his 1950 paper 'Computing Machinery and Intelligence,' Turing asked whether machines could think. Rather than defining intelligence, he proposed a behavioral test: if a human judge cannot distinguish a machine from a human through conversation, the machine should be considered intelligent. This shifted the question from metaphysics to empirical observation."},
            {"role": "user", "content": "Do you think current LLMs pass the Turing test?"},
            {"role": "assistant", "content": "In narrow settings, yes. GPT-4 and Claude can sustain conversations that many judges cannot distinguish from human responses. But this reveals the limitations of the test more than the capabilities of the models. The test measures conversational mimicry, not understanding. Models can produce coherent text without having beliefs, experiences, or genuine comprehension."},
            {"role": "user", "content": "So what would a better test of machine intelligence look like?"},
            {"role": "assistant", "content": "Several alternatives have been proposed. The ARC benchmark tests novel reasoning on visual puzzles. Francois Chollet argues for measuring skill-acquisition efficiency rather than task performance. Others propose testing causal reasoning, transfer learning to truly novel domains, or the ability to generate genuinely new scientific hypotheses. The consensus is shifting toward measuring generalization and reasoning, not fluency."},
            {"role": "user", "content": "That connects to the alignment problem too, doesn't it? If we can't measure intelligence, how do we align it?"},
            {"role": "assistant", "content": "Exactly. Alignment research grapples with specifying what we want from AI systems. If we cannot precisely define intelligence or goals, how can we ensure AI pursues them? Current approaches include RLHF (learning from human feedback), constitutional AI (encoding principles), and interpretability research (understanding what models are actually doing internally). None is fully satisfactory."},
        ],
    },
    {
        "id": "synthetic_pivot",
        "title": "Sharp pivot mid-conversation",
        "messages": [
            {"role": "user", "content": "Help me write a recipe for chocolate chip cookies."},
            {"role": "assistant", "content": "Here's a classic chocolate chip cookie recipe. Preheat oven to 375F. Cream together 1 cup butter, 3/4 cup sugar, and 3/4 cup brown sugar. Beat in 2 eggs and 1 tsp vanilla. In a separate bowl, whisk 2 1/4 cups flour, 1 tsp baking soda, and 1 tsp salt. Gradually combine wet and dry ingredients. Fold in 2 cups chocolate chips. Drop tablespoon-sized balls onto baking sheets. Bake 9-11 minutes until golden."},
            {"role": "user", "content": "What if I want them chewier?"},
            {"role": "assistant", "content": "For chewier cookies: use more brown sugar (1 cup brown, 1/2 cup white), add an extra egg yolk, reduce flour by 2 tablespoons, and refrigerate the dough for at least 1 hour before baking. Also underbake slightly, pulling them out when the edges are set but the centers still look soft. They firm up as they cool."},
            {"role": "user", "content": "Thanks. Actually, completely different topic. Can you explain quantum entanglement?"},
            {"role": "assistant", "content": "Quantum entanglement occurs when two particles become correlated such that the quantum state of one instantly determines the state of the other, regardless of distance. When you measure one entangled particle's spin as 'up,' the other is immediately 'down.' Einstein called this 'spooky action at a distance.' Crucially, this isn't faster-than-light communication because you can't control which outcome you get."},
            {"role": "user", "content": "How does it relate to quantum computing?"},
            {"role": "assistant", "content": "Entanglement is one of the key resources that makes quantum computers powerful. A quantum computer uses qubits that can be entangled, allowing them to represent and process exponentially many states simultaneously. Algorithms like Shor's (factoring) and Grover's (search) exploit entanglement to achieve speedups impossible for classical computers."},
            {"role": "user", "content": "Is quantum supremacy real? Has it been achieved?"},
            {"role": "assistant", "content": "Google claimed quantum supremacy in 2019 with their Sycamore processor, solving a specific problem in 200 seconds that they estimated would take a classical supercomputer 10,000 years. IBM disputed the classical estimate. More recently, experiments with hundreds of qubits have demonstrated advantage on specific tasks. But practical quantum advantage for useful problems remains elusive. We're in the NISQ (noisy intermediate-scale quantum) era."},
        ],
    },
    {
        "id": "synthetic_return",
        "title": "Return to earlier topic",
        "messages": [
            {"role": "user", "content": "Let's talk about the French Revolution. What caused it?"},
            {"role": "assistant", "content": "The French Revolution had multiple causes: financial crisis from war debts, a rigid social system with an overtaxed Third Estate, Enlightenment ideas challenging absolute monarchy, food shortages, and a weak king in Louis XVI. The immediate trigger was the financial crisis of 1789 and the calling of the Estates-General."},
            {"role": "user", "content": "How did it end?"},
            {"role": "assistant", "content": "The Revolution went through several phases: the constitutional monarchy (1789-1792), the radical Republic and Reign of Terror under Robespierre (1793-1794), the Thermidorian reaction and Directory (1794-1799), and finally Napoleon's coup in 1799. Napoleon brought stability but also authoritarian rule, effectively ending the revolutionary period while preserving some of its reforms."},
            {"role": "user", "content": "Speaking of Napoleon, what was his military genius?"},
            {"role": "assistant", "content": "Napoleon's military brilliance centered on speed, concentration of force, and the corps system. He divided his army into self-sufficient corps that could march independently but concentrate rapidly for battle. He excelled at interior lines, defeating enemy forces in detail before they could unite. His use of artillery as a mobile striking force was innovative."},
            {"role": "user", "content": "What was his biggest mistake?"},
            {"role": "assistant", "content": "The invasion of Russia in 1812. Napoleon led over 600,000 troops into Russia, but the Russian army refused to engage in a decisive battle, instead retreating and using scorched-earth tactics. The Battle of Borodino was a pyrrhic victory. When Napoleon reached Moscow, he found it abandoned and burning. The subsequent retreat through the Russian winter destroyed his army. Only about 100,000 returned."},
            {"role": "user", "content": "Going back to the Revolution itself. You mentioned Enlightenment ideas. Which thinkers were most influential?"},
            {"role": "assistant", "content": "Rousseau was perhaps the most influential, with his Social Contract arguing that legitimate political authority rests on a social contract among citizens. Voltaire championed religious tolerance and freedom of speech. Montesquieu advocated separation of powers. The Encyclopedists, led by Diderot, spread rational inquiry and skepticism of authority. These ideas permeated the educated classes and provided the intellectual framework for revolution."},
            {"role": "user", "content": "How do these Enlightenment ideas connect to modern democracy?"},
            {"role": "assistant", "content": "The connection is direct. Montesquieu's separation of powers is built into the US Constitution. Rousseau's popular sovereignty underpins democratic theory. Voltaire's advocacy for civil liberties maps onto the Bill of Rights and similar documents. The Declaration of the Rights of Man (1789) is a revolutionary document that still resonates. Modern liberal democracy is essentially an Enlightenment project, refined over two centuries."},
        ],
    },
]


def main():
    from openai import OpenAI
    import tiktoken

    client = OpenAI()
    enc = tiktoken.encoding_for_model("text-embedding-3-small")

    results = []

    for conv in SAMPLE_CONVERSATIONS:
        print(f"\n{'='*60}")
        print(f"  {conv['title']}")
        print(f"  {len(conv['messages'])} messages")
        print(f"{'='*60}")

        messages = conv["messages"]
        n = len(messages)

        # Step 1: Embed each message individually
        msg_texts = [m["content"] for m in messages]
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=msg_texts,
            dimensions=256,
        )
        msg_embeddings = np.array([d.embedding for d in resp.data])

        # Step 2: Embed each prefix (messages 1..k concatenated)
        prefix_embeddings = []
        for k in range(1, n + 1):
            prefix_text = "\n\n".join(
                f"[{m['role']}]: {m['content']}" for m in messages[:k]
            )
            # Truncate if needed
            tokens = enc.encode(prefix_text)
            if len(tokens) > 8191:
                prefix_text = enc.decode(tokens[:8191])

            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=[prefix_text],
                dimensions=256,
            )
            prefix_embeddings.append(resp.data[0].embedding)

        prefix_embeddings = np.array(prefix_embeddings)

        # Step 3: Compare prefix embeddings to various accumulations
        alphas = [0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

        print(f"\n{'k':>3s}", end="")
        for alpha in alphas:
            label = "mean" if alpha == 1.0 else f"a={alpha}"
            print(f"  {label:>7s}", end="")
        print(f"  {'prefix_drift':>12s}")

        conv_result = {"id": conv["id"], "title": conv["title"], "n": n, "steps": []}

        for k in range(n):
            prefix_emb = prefix_embeddings[k]
            prefix_emb_norm = prefix_emb / np.linalg.norm(prefix_emb)

            step = {"k": k + 1, "similarities": {}}

            print(f"{k+1:>3d}", end="")
            for alpha in alphas:
                if alpha == 1.0:
                    lens = ef.Uniform()
                else:
                    lens = ef.Exponential(alpha)

                # Accumulate messages 0..k
                acc_emb = lens.project(msg_embeddings[:k + 1])

                sim = float(cosine_similarity(
                    acc_emb.reshape(1, -1),
                    prefix_emb_norm.reshape(1, -1)
                )[0, 0])

                label = "mean" if alpha == 1.0 else f"{alpha}"
                step["similarities"][label] = round(sim, 4)
                print(f"  {sim:>7.4f}", end="")

            # Also track how much the prefix embedding itself drifts
            if k > 0:
                drift = 1 - float(cosine_similarity(
                    prefix_embeddings[0].reshape(1, -1),
                    prefix_embeddings[k].reshape(1, -1)
                )[0, 0])
            else:
                drift = 0
            step["prefix_drift"] = round(drift, 4)
            print(f"  {drift:>12.4f}")

            conv_result["steps"].append(step)

        # Summary: which alpha gives the best average agreement?
        print(f"\n  Average similarity to prefix embedding:")
        best_alpha = None
        best_sim = -1
        for alpha in alphas:
            label = "mean" if alpha == 1.0 else f"{alpha}"
            sims = [s["similarities"][label] for s in conv_result["steps"]]
            avg = np.mean(sims)
            if avg > best_sim:
                best_sim = avg
                best_alpha = label
            print(f"    {label:>5s}: {avg:.4f}")
        print(f"  Best: {best_alpha} (avg sim = {best_sim:.4f})")
        conv_result["best_alpha"] = best_alpha
        conv_result["best_sim"] = round(best_sim, 4)

        results.append(conv_result)

    # Save
    os.makedirs("experiments/results", exist_ok=True)
    with open("experiments/results/exp04_prefix_vs_accumulation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/results/exp04_prefix_vs_accumulation.json")


if __name__ == "__main__":
    main()
