# Architecture: How the Problem Space Breaks Down

## What we've built is already the right modular piece

`helpers/` (this folder) is a **geometry engine** — pure math with no opinion about what you use it for. The installable package is **`ophanimus`** under `src/ophanimus/`. It has three clean layers:

```
ophanimus/manifolds/       Pure math: exp, log, distance, transport, conversions
ophanimus/algorithms/      Generic manifold algorithms: regression, clustering, knn, pga
ophanimus/manifold_selection.py  Geometry detection: "which manifold fits my data?"
```

None of this knows about documents, queries, retrieval, embeddings, or Qdrant. That's correct — it shouldn't.

## The retrieval system is a separate project that *consumes* this

The hyperbolic retrieval system (the elevator pitch) is an **application layer** with its own concerns:

```
retrieval-system/
├── entailment_cones.py    # Cone containment logic (uses Poincaré angles)
├── hyperbolic_ann.py      # Approximate nearest neighbor in hyperbolic space
├── hybrid_trainer.py      # The Poincaré <-> Hyperboloid training loop
├── embedding_model.py     # Hyperbolic embedding model (replaces MiniLM)
├── index.py               # Vector DB integration (Qdrant/FAISS adapter)
└── ...
```

That project would depend on **`ophanimus`** (pip) / `import ophanimus` for the math, but owns all the retrieval-specific logic. The hybrid training pipeline maps cleanly:

```python
from ophanimus.manifolds.hyperboloid import exp_map, from_poincare, log_map, to_poincare

# Step A: Lift from Poincaré to Hyperboloid
x_hyp = from_poincare(x_poincare)

# Step B: Gradient step on the Hyperboloid (stable, no NaN)
x_hyp_new = exp_map(x_hyp, -lr * riemannian_grad)

# Step C: Project back to Poincaré for storage/logic
x_poincare_new = to_poincare(x_hyp_new)
```

That's three lines using functions we already built. The retrieval repo doesn't need to reimplement any geometry.

## What does NOT belong in helpers

- Entailment cone logic — that's retrieval-specific (uses Poincaré conformality for angle-based containment)
- Hyperbolic ANN indexing — that's an infrastructure concern
- Embedding model training — that's ML, not geometry
- Qdrant/FAISS adapters — that's integration code

## What might still be missing from helpers

One thing the retrieval system will need that we don't have yet: **entailment cone math**. This is borderline — it's pure geometry (half-aperture angles in the Poincaré ball), but it's also very specific to the retrieval use case. It stays in the retrieval repo unless it gets reused elsewhere.

## The clean split

```
helpers/                        # REPO FOLDER — `pip install -e .` exposes ophanimus
  src/ophanimus/         #   Python package
    manifolds/                  #   "What are the rules of this space?"
    algorithms/                 #   "What can I compute in any space?"
    manifold_selection.py       #   "Which space fits my data?"

hyperbolic-retrieval/           # APPLICATION — specific to search/retrieval
  uses ophanimus.manifolds for  #   Poincaré/Hyperboloid math
  adds its own:                 #   entailment cones, ANN, hybrid training,
                                #   embedding models, DB integration
```

`helpers/` is done as a modular foundation. The retrieval system is a separate project that imports it.

---

## Hyperbolic Embedding Ecosystem & Strategy

### The current state of pre-trained hyperbolic models

The open-source ecosystem is a double-edged sword. While there are tens of thousands of plug-and-play spherical models on Hugging Face (standard `sentence-transformers`), **pre-trained hyperbolic models are still largely confined to academic GitHub repositories.** There is no `Hyperbolic-MiniLM-L6-v2` you can `pip install` and run in two lines — yet.

However, researchers *do* release their weights. Here is what exists, from basic to advanced:

#### 1. Word/Node Embeddings: Gensim & Poincaré GloVe

For single words, concepts, or an existing taxonomy (product catalog, medical terms) — this is solved and ready to go.

- **Gensim:** Built-in `PoincareModel`. Feed it parent-child relationships, get Poincaré disk coordinates out.
- **Poincaré GloVe:** Pre-trained hyperbolic word embeddings (GloVe trained in negative curvature) available from the original FAIR GitHub repositories.

#### 2. Sentence-Level: Hyperbolic BERT

Contextual sentence embeddings built with **Hyperbolic BERT** (HyboNet, Hyperbolic Transformer).

- **Where:** Researchers' GitHub repos (search "Hyperbolic BERT pretrained weights").
- **The catch:** These replace standard Euclidean linear layers and attention with hyperbolic equivalents (Möbius addition). They cannot load into standard Hugging Face `transformers` pipelines — you must use the researchers' custom PyTorch code for the forward pass.

#### 3. The 2024/2025 Wave: Frameworks & Fine-Tuning

Instead of pre-training from scratch, the current approach is taking massive Euclidean models and "bending" them into hyperbolic space.

- **HyperCore / HypLL:** New PyTorch frameworks for building hyperbolic foundation models.
- **HypLoRA:** Using Low-Rank Adaptation (LoRA) to fine-tune standard Euclidean LLMs directly onto a hyperbolic manifold.

### The production strategy: Hyperbolic Projection Head

Because pulling academic code from GitHub is a nightmare for production stability, most engineers **do not use pre-trained hyperbolic transformers**. Instead, they build a **Hyperbolic Projection Head** — getting the benefits of hyperbolic entailment without abandoning the heavily optimized `sentence-transformers` library.

**The playbook:**

1. **Base:** Use a standard, fast Euclidean model (`all-MiniLM-L6-v2`) to generate initial 384-dim embeddings.
2. **Bridge:** Build a tiny 2-layer neural network in PyTorch using the **Geoopt** library.
3. **Training:** Feed Euclidean embeddings into this network. Train with an **Asymmetric Contrastive Loss** — a loss function that penalizes the model if a "parent" document isn't placed closer to the origin than a "child" query.
4. **Output:** The network learns to project flat, spherical vectors into the Poincaré Ball, organizing them hierarchically.

This takes a few hours on a single GPU, requires minimal data (just pairs of broad documents and specific queries), and produces a custom hyperbolic space tailored to your specific data structure.

```
sentence-transformers (Euclidean)     Geoopt projection head     Poincaré Ball
        all-MiniLM-L6-v2        -->   2-layer network with   -->  hierarchical
        384-dim flat vectors          Riemannian optimization     hyperbolic embeddings
```

**This is the recommended path for the retrieval system.** It maps directly to the architecture above:

- `embedding_model.py` wraps `sentence-transformers` as the base encoder
- `hybrid_trainer.py` trains the Geoopt projection head with asymmetric contrastive loss
- `ophanimus.manifolds` provides the Poincaré/Hyperboloid math underneath

### Upstream data validation (already built)

Before committing to any of this, validate that the raw data actually benefits from hyperbolic geometry using the tools already in `ophanimus.manifold_selection`:

```python
from manifold_selection import distance_from_features, select_manifold

# Measure BEFORE any neural network touches the data
D = distance_from_features(tfidf_vectors, metric="cosine")
result = select_manifold(D, dim=10)

# delta_relative < 0.1 → data is tree-like, hyperbolic wins
# delta_relative > 0.25 → data is flat, stick with Euclidean/spherical
print(result["gromov"]["delta_relative"])
print(result["recommendation"])
```

Key insight: **measure the raw data (TF-IDF, BM25) before the neural network normalizes it.** If you measure post-transformer embeddings, you're measuring the architecture (L2-normalization forces everything onto S^n), not the data.


# Misc

In the READE.md, we talked about drone pathing, but there was no mention of quaternions, why? Are they not necessary, and what is the trade-off between the two?