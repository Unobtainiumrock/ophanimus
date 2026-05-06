# Changelog

All notable changes to ophanimus.

## 0.2.2 — 2026-05-06

### Documentation

- README gains a **Numerical Conventions and Training Dynamics** section
  documenting the K=−1 distance convention, the gyrovector ↔ arccosh
  identity (and the 0.2.0 fix for the form-mixing bug), the Möbius
  gyration concern for adaptive Riemannian optimizers (Riemannian Adam,
  Momentum SGD), the fp64 boundary-promotion contract, and guidance on
  when to prefer `ophanimus.hyperbolic.*` over `ophanimus.manifolds.poincare.*`
  for near-boundary numerical stability.
- CHANGELOG gains explicit migration-notes subsections under 0.2.0 and
  0.2.1 covering autograd gradient-magnitude consequences, fixed-margin
  loss adjustments, RBF/Laplacian bandwidth scaling, and adaptive
  optimizer momentum-buffer guidance.
- Filed `HE-API-DOCS` as a future task to surface these conventions
  prominently in the eventual full docs site.

No code changes vs 0.2.1.

## 0.2.1 — 2026-05-06

### Fixed

- **`ophanimus.manifolds.poincare.parallel_transport`** now includes the
  Möbius gyration term (Ganea, Bécigneul, Hofmann 2018, Eq. 4). Previous
  releases used the conformal-scaling-only formula `(λ_p / λ_q) · v`,
  which preserved the metric inner product magnitude but had the wrong
  direction along the (curved) geodesic.

  The fix routes the c=1 case through the Lorentz hyperboloid:
  `from_poincare → push tangent → hyperboloid PT → pull tangent →
  to_poincare`. The Lorentz connection has no gyrogroup bookkeeping,
  so the gyration falls out of the round-trip exactly. Verified by
  three new tests in `tests/test_manifolds_poincare.py`:
  - `test_parallel_transport_matches_hyperbolic_dispatch` — pair-by-pair
    agreement with `ophanimus.hyperbolic.parallel_transport` within `1e-10`.
  - `test_parallel_transport_round_trip` — `PT(PT(v, p, q), q, p) ≈ v`,
    a property the previous formula failed in 2D+.
  - `test_parallel_transport_self_is_identity`.

  Closes HE-PT-MOBIUS-GYRATION.

  For `c ≠ 1` (variable curvature), the conformal-scaling fallback is
  retained — the hyperboloid module is currently written for `c = 1`
  only, and extending the lift with a c-aware scaling is left as a
  separate task.

### Migration notes (impact on adaptive optimizers)

If you train models with **Riemannian Adam, Riemannian Momentum SGD**,
or any adaptive optimizer that parallel-transports historical state
(momentum, second-moment estimates) between parameter steps:

- **Before 0.2.1**: PT was direction-wrong (conformal-scaling only,
  no gyration). Transported momentum was misaligned with the new
  parameter's local geometry, injecting drift into the update step
  and degrading convergence.
- **From 0.2.1**: PT is direction-correct. Momentum transport now
  matches the geodesic curvature.

If you have model checkpoints + optimizer state from 0.1.x or 0.2.0,
the optimizer state was accumulated under a slightly-wrong PT. You can:
- Continue training (the optimizer will course-correct over a few
  hundred steps as new gradients dominate the buffers), or
- Reset optimizer state and re-warm from the existing model weights.

Plain RSGD without momentum (e.g. `geodesic_regression` with line
search) was unaffected — it never transports historical state.

### Internal

- Factored push/pull tangent-vector helpers between Poincaré and
  hyperboloid into a new internal module `ophanimus._tangent`. Used by
  both `ophanimus.hyperbolic` (engine/dashboard primitives) and the
  rewritten `ophanimus.manifolds.poincare.parallel_transport`. Private
  aliases preserved in `ophanimus.hyperbolic` for back-compat.

## 0.2.0 — 2026-05-06

### Added

- **`ophanimus.hyperbolic`** — Poincaré-coordinated primitives that compute
  through the Lorentz hyperboloid, realizing the "engine in hyperboloid,
  dashboard in Poincaré" pattern documented in the README. Drop-in
  replacement for `ophanimus.manifolds.poincare.*` in algorithm callable
  arguments. Numerically stable up to and well past `||p|| = 0.999`,
  where direct Poincaré compute starts losing precision to the
  `1/(1-||x||²)` blow-up. Includes:
  - `distance(p, q)` — geodesic distance via `arccosh(-<u, v>_M)`
  - `exp_map(p, v)`, `log_map(p, q)` — converted via Jacobian of `from_poincare`
  - `parallel_transport(v, p, q)` — exact Lorentz PT (also fixes a
    silently-wrong direction term in `manifolds.hyperboloid`; see Fixed)

### Fixed

- **`ophanimus.manifolds.hyperboloid.parallel_transport`**: previous
  implementation produced output that was not Minkowski-orthogonal to
  the destination point (i.e. not in the destination tangent space),
  so transported vectors were silently incorrect along long geodesics.
  Now uses the standard formula `PT(v) = v + (<v,q>_M / (1 - <p,q>_M)) * (p+q)`
  per Nickel & Kiela 2018. Norm-preservation and tangent-space tests
  added in `tests/test_manifolds_hyperboloid.py`.

### Changed (BREAKING)

- **`ophanimus.manifolds.poincare.distance`** now returns the standard
  K=−1 geodesic distance with prefactor `1/sqrt(c)`, matching
  `ophanimus.manifolds.hyperboloid.distance` after `from_poincare`
  conversion. Previous releases used `2/sqrt(c)` (the radius-2 ball
  convention), giving values that were exactly twice the K=−1 distance.
  If you were using 0.1.0 and depended on the larger value, halve any
  thresholds in your code, or multiply the result by 2.

### Migration notes (impact on training dynamics)

The distance change is mechanically a 2× rescaling. For most use cases
within ophanimus that's invariant — `kmeans`/`knn` use `argmin`/`argsort`,
`manifold_selection` normalizes by a target-sum scale factor,
`geodesic_regression` computes gradients via `log_map` (not by
differentiating `distance`), and its line search compares
`new_loss < current_loss` where both are rescaled by the same factor of
4 (since loss is `d²`).

For **user code that wraps ophanimus distance in autograd** (PyTorch,
JAX), every gradient flowing back through `poincare.distance` is now
half its previous magnitude:
1. `loss = poincare.distance(y_pred, y_true)**2` is now `1/4` its old value
2. `∇_E loss` (Euclidean gradient) is halved
3. `∇_R loss = (1 − c‖x‖²)²/4 · ∇_E loss` (Riemannian gradient) is halved
4. The optimizer takes a step half the size it previously would have

To recover **exact prior training behavior**, double your learning rate:
`lr ← 2·lr`. To take advantage of the corrected scale (recommended),
leave `lr` and accept the slower-but-more-meaningful steps.

For **fixed-margin losses** (contrastive, triplet, etc.), the distances
in the margin comparison are now half their old values, making any
fixed margin twice as strict in relative terms. Halve your margin
hyperparameter to maintain the same boundary strictness:
`margin ← margin / 2`.

For **RBF / Laplacian kernel bandwidths** (`σ` in `exp(-d²/2σ²)` or
`exp(-d/σ)`): the same kernel value at the same point pair now occurs at
σ/2 vs. before, since `d` is halved. Halve `σ` to preserve prior
similarity behavior.

### Known issues (filed for later)

- `ophanimus.manifolds.poincare.parallel_transport` still uses the
  conformal-scaling-only formula and is missing the Möbius gyration
  term (Ganea, Becigneul, Hofmann 2018, Eq. 4). Norm-preserving but
  direction-incorrect along the geodesic. Workaround: use
  `ophanimus.hyperbolic.parallel_transport` (exact). Tracking issue:
  HE-PT-MOBIUS-GYRATION.

## 0.1.0 — 2026-05-06

Initial release.

- Five Riemannian manifolds: Poincaré ball, Lorentz hyperboloid, sphere,
  SO(3), SPD
- Eight geodesic algorithms: Fréchet mean, geodesic regression, k-means,
  kNN, PGA, kernels, parallel-transport time-series, interpolation
- Closed-loop manifold-selection pipeline: Gromov δ-hyperbolicity plus
  competitive embedding into Euclidean / sphere / Poincaré
- fp64 boundary promotion at every public entry point (~32 functions)
- SciPy integration: `pdist`+`squareform`, `csgraph.shortest_path`
- 73-test pytest suite
- Apache-2.0 licensed
