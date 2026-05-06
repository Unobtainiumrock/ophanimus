# Changelog

All notable changes to ophanimus.

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
