# Changelog

All notable changes to ophanimus.

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
