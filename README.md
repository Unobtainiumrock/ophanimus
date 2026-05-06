# Ophanimus

**Riemannian manifold primitives and geodesic algorithms** — pure geometry, no retrieval stack, no vector DB, no training loops for text embeddings.

## Install

```bash
pip install .          # or: pip install -e .  for development
```

Import the **`ophanimus`** package (PyPI name: `ophanimus`).

```python
from ophanimus.manifolds import poincare, hyperboloid
from ophanimus.algorithms import geodesic_regression
```

Layout follows the standard **`src/`** layout so the package is what gets installed, not stray repo files.

## Structure

```
├── pyproject.toml
├── src/
│   └── ophanimus/
│       ├── manifolds/              # Riemannian manifold primitives
│       │   ├── sphere.py           #   Unit sphere S^n
│       │   ├── poincare.py         #   Poincaré ball D^n (variable curvature)
│       │   ├── hyperboloid.py      #   Hyperboloid H^n (Lorentz model)
│       │   ├── so3.py              #   Rotation group SO(3)
│       │   └── spd.py              #   Symmetric positive definite matrices Sym+_n
│       ├── algorithms/             # Downstream tasks using manifold primitives
│       │   ├── frechet_mean.py     #   Manifold centroid
│       │   ├── regression.py       #   Geodesic regression
│       │   ├── clustering.py       #   Manifold k-means
│       │   ├── knn.py              #   Geodesic k-nearest neighbors
│       │   ├── interpolation.py    #   Geodesic interpolation
│       │   ├── pga.py              #   Principal geodesic analysis
│       │   ├── kernels.py          #   Manifold-aware kernels
│       │   └── time_series.py      #   Velocity comparison via parallel transport
│       └── manifold_selection.py   # Distance matrix construction + geometry selection
├── notes.md                # Background theory notes
└── TODO.md
```

Each manifold module exposes four primitives:
- `exp_map(p, v)` — walk from point p along tangent vector v
- `log_map(p, q)` — tangent vector at p pointing toward q (inverse of exp_map)
- `distance(p, q)` — geodesic distance between two points
- `parallel_transport(v, p, q)` — slide tangent vector v from p's tangent space to q's

## Geodesic Regression

`ophanimus/algorithms/regression.py` provides `geodesic_regression(X, Y, exp_map, log_map, distance, parallel_transport)` which fits the model:

```
ŷ_i = Exp_p(x_i · v)
```

This is the manifold analogue of `y = p + x·v`. It works with any manifold — just pass in the three primitives:

```python
from ophanimus.manifolds import sphere
from ophanimus.algorithms import geodesic_regression

p, v, losses = geodesic_regression(X, Y,
    sphere.exp_map, sphere.log_map, sphere.distance, sphere.parallel_transport)
```

## Hyperboloid vs Poincaré: Engine vs Dashboard

We have two models of hyperbolic space: the **Hyperboloid** (Lorentz model) in
`hyperboloid.py` and the **Poincaré ball** in `poincare.py`. They represent the
same underlying geometry — the same way a Mercator map and a globe represent
the same Earth.

**The Hyperboloid is the engine.** Use it for computation — gradient descent,
optimization, training. Its formulas are numerically stable everywhere because
there's no boundary singularity. The Poincaré ball's conformal factor
`2/(1-||x||^2)` explodes as points approach the boundary; the hyperboloid has
no such problem.

**The Poincaré ball is the dashboard.** Use it for visualization, interpretation,
and passing results to other systems. It's bounded (everything fits in a disc/ball),
making it natural for humans to look at and reason about.

In practice: train on the hyperboloid, convert to Poincaré for display.

### Converting between models

```python
from ophanimus.manifolds.hyperboloid import from_poincare, to_poincare

# Hyperboloid (n+1 dims) -> Poincaré ball (n dims)
p = to_poincare(x)   # x[0] is the time component, x[1:] are spatial

# Poincaré ball (n dims) -> Hyperboloid (n+1 dims)
x = from_poincare(p)  # prepends the time component
```

The formulas:
```
Hyperboloid -> Poincaré:   p_i = x_i / (1 + x_0)
Poincaré -> Hyperboloid:   x_0 = (1 + ||p||^2) / (1 - ||p||^2)
                           x_i = 2*p_i / (1 - ||p||^2)
```

Points near the Poincaré boundary (||p|| -> 1) map to large x_0 on the
hyperboloid — they're "high up" on the sheet, far from the origin. The
center of the Poincaré ball (p = 0) maps to the base of the hyperboloid
(x = [1, 0, 0, ...]).

## Learning Curvature for Poincaré Embeddings

The Poincaré ball has a curvature parameter `c` (where K = -c) that controls how aggressively the space expands outward. All four functions in `poincare.py` accept `c` as an optional parameter (defaults to 1.0):

```python
from ophanimus.manifolds import poincare

# Fixed curvature
d = poincare.distance(u, v, c=2.0)
q = poincare.exp_map(p, tangent, c=2.0)
```

### Making c learnable

To learn c alongside p and v in geodesic regression, wrap the Poincaré operations with a shared c and differentiate through it. The idea: c is a scalar parameter in the loss function, so gradient descent can tune it just like any other weight.

```python
import numpy as np
from ophanimus.manifolds import poincare


def learn_curvature(X, Y, lr=1e-3, lr_c=1e-4, steps=1000):
    """Geodesic regression with learnable Poincaré curvature.

    Jointly optimizes base point p, tangent direction v, and
    curvature c via Riemannian gradient descent.
    """
    N = len(Y)
    c = 1.0  # initial curvature

    # Initialize p at Fréchet mean
    p = Y[0].copy().astype(np.float64)
    for _ in range(50):
        tangents = [poincare.log_map(p, Y[i], c) for i in range(N)]
        mean_t = np.mean(tangents, axis=0)
        if np.linalg.norm(mean_t) < 1e-10:
            break
        p = poincare.exp_map(p, mean_t, c)

    # Initialize v via tangent-space OLS
    tangents = [poincare.log_map(p, Y[i], c) for i in range(N)]
    W = np.array(tangents)
    x_sq = np.dot(X, X)
    v = np.tensordot(X, W, axes=([0], [0])) / x_sq if x_sq > 1e-15 else np.zeros_like(p)

    for step in range(steps):
        # Forward pass
        Y_hat = [poincare.exp_map(p, X[i] * v, c) for i in range(N)]
        loss = np.mean([poincare.distance(Y[i], Y_hat[i], c) ** 2 for i in range(N)])

        # Gradient for c via finite differences
        eps = 1e-6
        Y_hat_plus = [poincare.exp_map(p, X[i] * v, c + eps) for i in range(N)]
        loss_plus = np.mean([poincare.distance(Y[i], Y_hat_plus[i], c + eps) ** 2 for i in range(N)])
        grad_c = (loss_plus - loss) / eps

        # Tangent-space gradients for p and v
        errors = [poincare.log_map(p, Y[i], c) - X[i] * v for i in range(N)]
        grad_v = -np.mean([X[i] * errors[i] for i in range(N)], axis=0)
        grad_p = -np.mean(errors, axis=0)

        # Update all three parameters
        v = v - lr * grad_v
        p = poincare.exp_map(p, -lr * grad_p, c)
        c = max(1e-6, c - lr_c * grad_c)  # clamp c > 0

    return p, v, c
```

Key details:

- **Separate learning rate for c** (`lr_c`): curvature changes affect the entire geometry, so it typically needs a smaller step size than p and v to stay stable.
- **Finite-difference gradient for c**: we compute `dL/dc ≈ (L(c+ε) - L(c)) / ε`. This is simple and works well since c is a single scalar. For a production system you could derive the analytic gradient instead.
- **Clamping `c > 0`**: curvature must stay positive (negative curvature space). As c approaches 0, the space flattens to Euclidean — which is a valid outcome if your data isn't hierarchical.
- **What c learns**: high c (steep curvature) if the data has deep hierarchy with exponential branching. Low c (gentle curvature) if the structure is shallow. Near-zero c if the data is essentially flat.

## Defining New Manifolds

To add a new manifold to `src/ophanimus/manifolds/`, you need to derive and implement the same three primitives: `exp_map`, `log_map`, and `distance`. This follows a four-step pipeline from differential geometry.

### Step 1: Define the Space and the Metric

A Riemannian manifold starts with a set of points and a **metric** g_p(u, v) — an inner product on the tangent space at each point p. The metric defines what "length" and "angle" mean in your space.

The metric is what makes each manifold unique. For example, the Poincare disk and flat 2D Euclidean space are both sets of 2D points. The only difference is that the Poincare disk uses a metric that inflates distances near the boundary. Change the metric, change the entire geometry.

For a new manifold, write down g_p(u, v) explicitly. Everything else follows from it.

### Step 2: Derive the Geodesic Equation

Geodesics (shortest paths) are found by minimizing the curve length integral under your metric:

```
L(gamma) = integral of sqrt(g(gamma'(t), gamma'(t))) dt
```

Applying the Euler-Lagrange equations from calculus of variations gives the **geodesic equation**:

```
gamma''_k + sum_ij  Gamma^k_ij  gamma'_i  gamma'_j = 0
```

The Christoffel symbols Gamma^k_ij are computed directly from derivatives of your metric g. This is the ODE that governs "straight-line" motion in your space.

### Step 3: Solve for the Exponential Map (Forward: IVP)

The exp map is an **initial value problem**: place a particle at point p with velocity v and solve the geodesic ODE forward to t=1.

```
gamma(0) = p,  gamma'(0) = v  -->  Exp_p(v) = gamma(1)
```

For simple manifolds (sphere, hyperbolic space), this ODE has a closed-form solution — that's where formulas like `cos(||v||) p + sin(||v||) v/||v||` come from. For complex manifolds where no closed form exists, use a numerical ODE solver (e.g. Runge-Kutta via `scipy.integrate.solve_ivp`).

### Step 4: Invert for the Logarithmic Map (Backward: BVP)

The log map is a **boundary value problem**: given start p and end q, find the initial velocity v that makes the geodesic hit q at t=1.

```
gamma(0) = p,  gamma(1) = q  -->  Log_p(q) = gamma'(0)
```

The norm of the resulting vector ||Log_p(q)|| equals the geodesic distance d(p, q), which is the error signal used in geodesic regression.

For simple manifolds this also has a closed form. For complex ones, use a BVP solver (e.g. `scipy.integrate.solve_bvp`) or shooting methods.

### Adding a new manifold to this codebase

1. Create `src/ophanimus/manifolds/your_manifold.py`
2. Implement `exp_map(p, v)`, `log_map(p, q)`, `distance(p, q)` following the steps above
3. Register the submodule in `manifolds/__init__.py` (see existing entries)
4. It will work immediately with `geodesic_regression`:

```python
from ophanimus.manifolds import your_manifold
p, v, losses = geodesic_regression(X, Y, your_manifold.exp_map, your_manifold.log_map, your_manifold.distance)
```

### When you can't solve analytically

If your metric produces a geodesic equation with no closed-form solution, implement exp_map and log_map numerically:

```python
from scipy.integrate import solve_ivp, solve_bvp

def exp_map(p, v):
    # Solve geodesic ODE as IVP: gamma(0)=p, gamma'(0)=v
    def geodesic_ode(t, state):
        pos, vel = state[:n], state[n:]
        accel = -christoffel_contraction(pos, vel)  # your Gamma terms
        return np.concatenate([vel, accel])

    sol = solve_ivp(geodesic_ode, [0, 1], np.concatenate([p, v]))
    return sol.y[:n, -1]  # position at t=1
```

This makes the framework extensible to any Riemannian manifold, even exotic ones where the geometry is too complex for pen-and-paper solutions.

## Manifold Selection

`ophanimus/manifold_selection.py` helps you determine which geometry fits your data best. The full pipeline takes raw data through to a geometry recommendation:

```
Raw Data ──► Clean ──► Distance Matrix ──► select_manifold() ──► Recommendation
                            |                                          |
                  (from graph, features,                               v
                   or similarity scores)                   Use that manifold's
                                                           exp/log/distance for
                                                           downstream work
                                                           (regression, clustering,
                                                           nearest neighbors, etc.)
```

### Building the Distance Matrix

Everything starts from a pairwise distance matrix — an NxN table that says "how far apart are these two things?" without assuming any geometry. How you build it depends on what your raw data looks like:

**From feature vectors** (embeddings, sensor readings, tabular data):
```python
from ophanimus.manifold_selection import distance_from_features
D = distance_from_features(X, metric="cosine")  # or "euclidean", "correlation"
```

**From a graph** (social network, taxonomy, knowledge graph):
```python
from ophanimus.manifold_selection import distance_from_graph
D = distance_from_graph(adj_matrix, weighted=False)  # BFS shortest paths
```

**From a similarity matrix** (co-occurrence, transition probabilities, ratings):
```python
from ophanimus.manifold_selection import distance_from_similarity
D = distance_from_similarity(S, method="neglog")  # or "subtract", "inverse"
```

### The choice of distance conversion matters

The way you convert raw data into distances is not neutral — it changes what geometry the pipeline will recommend. The same similarity matrix can point to different geometries depending on the conversion:

- **"subtract"** (`d = max(S) - S`): Linear transform. Preserves rank ordering but treats the similarity scale as uniform. Use as a safe default.
- **"inverse"** (`d = 1/S`): Nonlinear. Heavily penalizes low-similarity pairs, pushing them far apart. Use when low similarity should mean very distant.
- **"neglog"** (`d = -log(S)`): Makes distances additive. If S contains transition probabilities, the distance from A to C via B equals d(A,B) + d(B,C). This is exactly how tree distances work, so neglog will reveal hierarchical structure that subtract would miss.

If you are unsure, run `select_manifold` with multiple conversions. If the recommendation is consistent across methods, trust it. If it flips (e.g. neglog says "poincare" but subtract says "euclidean"), that is telling you the geometry is ambiguous at that scale — the conversion is shaping the result more than the data is.

Similarly for feature-based distances: cosine ignores magnitude (good for text embeddings where a long document and short document about the same topic should be similar), Euclidean cares about magnitude (good for physical measurements), and correlation removes the per-point baseline (good for time series patterns).

### Cleaning Before Building Distances

Before constructing the distance matrix:
- **Remove NaN/inf/duplicates** from feature matrices.
- **Subsample if N > ~2000** — the distance matrix is O(N^2), and the embedding step is expensive. 500-2000 points is usually enough to detect the underlying geometry.
- **Check connectivity for graphs** — `distance_from_graph` returns `inf` for unreachable pairs. Either cap these at a large finite value or analyze connected components separately.
- **Normalize if using cosine** — zero vectors produce NaN in cosine distance.

### What select_manifold Returns

```python
from ophanimus.manifold_selection import select_manifold

result = select_manifold(D, dim=10)
```

The function runs two stages:

**Stage 1 — Gromov delta-hyperbolicity** (no embedding needed): Tests how "tree-like" your distance matrix is by sampling quadruples of points and checking the four-point condition. `delta_relative < 0.1` means the data is strongly hierarchical and hyperbolic geometry is indicated. `delta_relative > 0.25` means the data is not tree-like.

**Stage 2 — Competitive embedding**: Embeds the distance matrix into Euclidean, Sphere, and Poincare at the same dimension, then measures distortion for each. Uses two metrics:
- **Stress**: normalized RMSE of distance errors. Single number, lower = better fit.
- **Distortion ratios**: per-pair `d_embedded / d_original`. The max ratio is the worst-case multiplicative distortion. The variance tells you whether distortion is uniform or concentrated on specific pairs.

The result includes a ranking by stress and a recommended geometry:
```python
print(result["recommendation"])              # "poincare"
print(result["gromov"]["delta_relative"])     # 0.04 (tree-like)
print(result["ranking"])                      # ["poincare", "euclidean", "sphere"]
for geom in result["ranking"]:
    r = result["embeddings"][geom]
    print(f"  {geom}: stress={r['stress']:.4f}, ratio_var={r['ratio_variance']:.4f}")
```

## Downstream Tasks

Once you have a geometry recommendation and can embed into it, a family of
algorithms becomes available. They all follow the same pattern: take the
Euclidean version, replace Euclidean operations with manifold primitives.

```
                         ┌───────────────────────┐
                         │   Manifold Primitives │
                         │  exp, log, distance,  │
                         │  parallel_transport   │
                         └──────────┬────────────┘
                                    │
            ┌───────────┬───────────┼───────────┬──────────────┐
            v           v           v           v              v
      ┌───────────┐ ┌────────┐ ┌────────┐ ┌─────────┐ ┌─────────────┐
      │ Fréchet   │ │Geodesic│ │  k-NN  │ │  PGA    │ │   Kernels   │
      │ Mean      │ │Regression│        │ │ (PCA)   │ │ (GP, SVM)   │
      └─────┬─────┘ └────────┘ └────────┘ └─────────┘ └─────────────┘
            │
      ┌─────┴─────┐
      │  k-Means  │
      │ Clustering│
      └───────────┘
```

All downstream tasks live in `ophanimus/algorithms/`:

```
ophanimus/algorithms/
├── frechet_mean.py     # Manifold centroid (used by regression + clustering)
├── regression.py       # Geodesic regression
├── clustering.py       # Manifold k-means (Fréchet mean per cluster)
├── knn.py              # Geodesic k-nearest neighbors
├── interpolation.py    # Walk along geodesics between points
├── pga.py              # Principal geodesic analysis (manifold PCA)
├── kernels.py          # Geodesic distance kernels (for GP, SVM, KDE)
└── time_series.py      # Compare velocities via parallel transport
```

Each algorithm takes manifold primitives as arguments, so they work with
any geometry:

```python
from ophanimus.algorithms.clustering import kmeans
from ophanimus.manifolds import poincare

labels, centers = kmeans(points, k=5,
    exp_map=poincare.exp_map,
    log_map=poincare.log_map,
    distance=poincare.distance)
```
