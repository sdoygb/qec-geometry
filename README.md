# qec-geometry

**Geometric diagnostics for quantum error correction codes.**

A zero-dependency Python toolkit that classifies decoding error patterns by their
topology (A0: local / trivial, A1: non-trivial crossing chains) and predicts
fault-tolerance thresholds from a closed-form formula.

```
qecgeo  ──  Pauli algebra ── stabilizer framework ── code zoo ── error geometry ── threshold closed form ── anyon typing
```

---


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21946023.svg)](https://doi.org/10.5281/zenodo.21946023)

## What it does

### 1. Error-pattern geometry: A0/A1 classification (surface code)

After MWPM decoding of a surface code, every sampled error pattern is classified:

| class | meaning | topology |
|---|---|---|
| **A0** | correctable, local error chains | trivial (Berry phase 0) |
| **A1** | logical errors, chains crossing the code | non-trivial (Berry phase 2π) |

The MWPM **matching layer** separates the two classes sharply:

| feature | A0 median | A1 median | A1/A0 |
|---|---|---|---|
| `total_dist` (sum of pairing-edge lengths) | 2.00 | 6.00 | **3.00×** |
| `n_edges` | 3.00 | 5.00 | 1.67× |
| `cluster` (max excitation cluster) | 2.00 | 4.00 | 2.00× |
| crossing rate | 22% | 47% | 2.1× |

The A1/A0 separation **appears below threshold** (p ≈ 0.005 < p_th ≈ 0.011 for
the surface code): at low noise, logical error chains carry a measurable
topological signature in the pairing layer. At high noise the error patterns
lose structure and the distinction vanishes.

### Phase 2: distance scaling law (L = 4, 6, 8)

As the code distance L grows (fixed physical noise p = 0.005), the **absolute**
topological signal survives and scales linearly with L:

| L | p_L | A0 total_dist | A1 total_dist | **A1 − A0** | A1/A0 | A1 long-chain rate |
|---|---|---|---|---|---|---|
| 4 | 1.48% | 2.0 | 6.0 | **4.0 = L** | 3.00× | 12.9% |
| 6 | 0.49% | 8.0 | 14.0 | **6.0 = L** | 1.75× | 33.9% |
| 8 | 0.17% | 16.0 | 24.0 | **8.0 = L** | 1.50× | 51.2% |

**Scaling law: A1_total_dist − A0_total_dist = L.** The logical-error pairing
chain carries an extra contribution of exactly one code distance — the geometric
cost of the non-trivial homology class (Berry phase 2π). Meanwhile the *ratio*
decays (3.00× → 1.50×) because the local-error background grows as ∝ L²
(error count ∝ code area) and dilutes the relative signal.

Two further consequences:

- **Explicit long chains become more common with L**: the fraction of A1
  samples with a pairing edge ≥ 4 grows 12.9% → 33.9% → 51.2% (answering the
  open question of article 10.54 §6.3).
- **Seed reproducibility**: median features are identical across seeds
  42/123/777 at every L (deterministic sampling via
  `compile_detector_sampler(seed=...)`).

### Cross-code benchmark: surface vs color code

`qecgeo` diagnostics are **circuit-agnostic**: `diagnose_circuit(circuit, ...)`
accepts any stim circuit with detector coordinates, so the same A0/A1 pipeline
runs on structurally different codes. Benchmark (surface 8,000 shots / color 120,000 shots, seed 42, via
`scripts/benchmark_scan.py`):

**Surface code** (L = 4, rounds = 3):

| noise | p_L | crossing lift (A1/A0) | total_dist lift | cluster lift |
|---|---|---|---|---|
| 0.005 | 1.80% | **1.82×** | 3.00× | 2.00× |
| 0.010 | 7.45% | 1.34× | 1.50× | 1.33× |
| 0.020 | 24.5% | 1.07× | 1.00× | 1.20× |
| 0.030 | 38.1% | 1.02× | 1.00× | 1.14× |

**Color code** (diameter = 3, rounds = 3, 120,000 shots; decoder: **chromobius** — the color-code-specific color-matching decoder; MWPM is structurally inapplicable to the three-color structure, its matching graph has boundary-free components for d ≥ 7):

| noise | p_L | crossing lift (A1/A0) | cluster lift |
|---|---|---|---|
| 0.005 | 0.36% | **2.69×** | 3.00× |
| 0.010 | 1.44% | **2.44×** | 1.50× |
| 0.020 | 5.06% | **2.06×** | 1.50× |
| 0.030 | 10.42% | **1.80×** | 1.50× |

Three robust findings:

1. **Crossing lift decays monotonically with noise** in both codes: as physical
   noise grows, error patterns randomize and the A0/A1 geometric separation is
   washed out. The topological signature is strongest **below threshold** —
   exactly the regime QEC operates in.
2. **The color code (d=3) keeps a resolvable separation at every noise level**
   (≥ 1.80× vs surface's collapse to ~1.0× above noise 0.01); the separation
   degrades with code distance (d=5/7/9/11: 1.73/1.27/1.21/0.97 at p=0.03,
   see 10.55 open question 2). The three-color
   (tri-sector) structure constrains A1 chains topologically, making the
   logical-error geometry more persistent in the high-noise regime. This is the
   geometric-theory prediction: richer sector structure ⇒ more robust A0/A1
   separation.
3. **Cluster lift is strongest at low noise (3.0×) and plateaus at 1.5×**: the

 4. **Non-lattice codes (1D coordinates) lose the A0/A1 separation** (NEW,
    cross-code boundary): on the [[15,7,3]] Hamming CSS code the detector
    coordinates are 1D, and the crossing/topology criteria become trivial —
    cross_lift collapses to 1.00× (A0 exc_med = A1 exc_med) vs surface 2.39×.
    The A0/A1 separation requires ≥2D lattice coordinates; the pipeline now
    normalizes arbitrary coordinate dimensions and **reports the degeneration
    honestly** instead of crashing (`_normalize_coords`; see
    `scripts/cross_code_topology_diagnosis.py`). For LDPC/non-lattice codes,
    non-topological criteria (syndrome weight / distance) are the appropriate
    comparators.
   largest excitation cluster of A1 is systematically larger than A0's, but
   cluster separation (1.5–3.0×) is weaker than crossing separation
   (1.8–2.7×) and both decay together with noise — crossing rate is the
   cleaner separator.

*Note on `inf` entries in the scan output*: at low noise most color-code A0
samples have zero excitations, so A0 median chain-length features are 0 and
ratios diverge. Crossing rate and cluster size (rate / median, no pairing
chain needed) are unaffected and are the robust cross-code comparators. With
chromobius decoding there is no MWPM pairing layer, so `total_dist` features
apply to the surface code only.

**Theoretical grounding**: full write-up and tri-sector
prediction in article 10.55 of the conjugate-spectral-geometry library (CN).

### 2. Fault-tolerance threshold: closed form

For single-round optimal decoding with depolarizing noise, the logical error
rate obeys the quadratic law

```
p_L(p) ≈ A·p²,    A = η·C(n,2),    p_th = 1/A
```

where η = fraction of weight-2 errors mis-recovered as logical operators
(exactly enumerated). Verified values:

| code | η | A | p_th |
|---|---|---|---|
| [[5,1,3]] five-qubit | 1.000 | 10.00 | 10.00% |
| [[7,1,3]] Steane | 0.3333 | 7.00 | 14.29% |
| [[15,7,3]] RM(1,4) CSS | 0.3333 | 35.00 | 2.86% |

Monte Carlo confirms p_L ≈ A·p² at low p (ratio ≈ 0.87 at p = 0.01, [7,1,3]).

### 3. Anyon typing

The δ = Cl(8) Majorana 8-cycle permutation closes with zero net phase
(δ⁸ ≡ 0 mod 2π, consistent with the Berry phase 2π) — an **Ising (Majorana)
type** anyon signature, not Fibonacci.

---

## Install & use

```bash
pip install qec-geometry   # from PyPI (numpy auto-installed)
pip install stim pymatching   # needed only for the surface-code demo
pip install chromobius       # needed only for the color-code demo
```

```python
from qecgeo.codes import steane_code
from qecgeo.threshold import analyze_eta

r = analyze_eta(steane_code())
print(r['eta'], r['p_th'])   # 0.3333, 0.1429
```

Surface-code error geometry (reproduces the 3.00× separation):

```bash
cd qec-geometry
python3 scripts/demo_error_geometry.py          # L=4, noise=0.005, 20000 shots
python3 scripts/demo_threshold.py               # η enumeration + MC + anyon
python3 scripts/phase2_scaling.py --scan        # Phase 2: L=4/6/8 × 3 seeds scaling
python3 -m unittest tests.test_qecgeo -v        # 14 tests
```

---

## Validation

- **Threshold closed form**: exact enumeration over all weight-2 errors for
  [[5,1,3]], [[7,1,3]], [[15,7,3]]; Monte Carlo verification of the quadratic
  law; concatenation compression p_{L+1} = A·p_L² shown for p0 = 0.001/0.01/0.05.
- **Error geometry (Phase 1)**: 20,000 samples at p = 0.005 (L=4 surface code),
  the 3.00× total_dist separation is stable across runs.
- **Error geometry (Phase 2 scaling)**: L = 4/6/8 × seeds 42/123/777 at p = 0.005,
  30k–600k shots per point; scaling law A1 − A0 = L exact (4.0/6.0/8.0);
  long-chain rate 12.9% → 51.2%; seed-stable medians (results in `data/phase2/`).
- All numerical results are stored in `data/` (JSON) and reproduced by the
  scripts above.

### 5. PG complete codes + degeneracy ratios (NEW)

Two families are covered by the closed forms:

| Family | Code | d | Closed form | Reference |
|---|---|---|---|---|
| PG | `[[2^m-1, 2^m-1-2m, 3]]` | 3 (locked) | weight-3 logicals `= (2^m-1)(2^m-2)/6` | 10.28 |
| AG | `[[2^m, n-2·dim RM(r,m), 2^{r+1}]]` | arbitrary | full parameter/loss table | 10.30/10.35 |

Degeneracy ratios (10.33): `P_r(m)` (full-degeneracy proportion, = 1 for r ≤ 2,
~1e-4 for r=3,m=8) and `P'_r(m)` (next-order, → θ^{d+2} coefficient).

```python
from qecgeo import pg_params, degeneracy_ratio, degeneracy_ratio_next
pg_params(4)                       # {'n': 15, 'k': 7, 'd': 3, 'w3_logical': 35}
degeneracy_ratio(8, 3)             # 1.007e-4
degeneracy_ratio_next(6, 2)        # 0.08197
```

Run `scripts/compare_families.py` for the side-by-side PG vs AG comparison.

### 6. Logical-operator counting + cross-family benchmark (NEW)

Logical-operator count closed forms (verified against full enumeration):

```python
from qecgeo import logical_operator_count, pg_logical_count
logical_operator_count(5, 1)   # 1240 = AG(5,2) 2-flats  (RM(1,5), verified)
logical_operator_count(6, 1)   # 10416 = AG(6,2) 2-flats (RM(1,6), verified)
pg_logical_count(4)            # 35 = PG(3,2) lines
```

| Family | Logical operators (weight d) | Reference |
|---|---|---|
| AG `[[2^m, k, 2^{r+1}]]` | `2^{m-r-1}·[m choose r+1]_2` (AG flats) | Theorem 10.30.2.04 |
| PG `[[2^m-1, k, 3]]` | `(2^m-1)(2^m-2)/6` (PG lines) | 10.28 |

The same functionality is exposed as a class API (matching the
pyqpanda-algorithm `QECClosedForm` module), for use alongside a simulator:

```python
from qecgeo import QECClosedForm
cf = QECClosedForm(10, 3)     # [[1024, 672, 16]]
cf.code()                     # (1024, 672, 16)
cf.loss(0.01)                 # 1.05e-24
cf.zero_loss_boundary()       # 7
```

Run `scripts/cross_family_benchmark.py` for the unified cross-family
comparison (PG vs AG, parameters + logical counts + loss scaling + degeneracy
ratios), all from combinatorial closed forms.

### 7. Universal loss exponent + detection-rate closed forms (NEW)

Two closed forms from the noise-behaviour series (10.29/10.31):

```python
from qecgeo import loss_exponent, detection_rate, miss_conditional_fidelity
loss_exponent(8)          # 8 = 2·ceil(8/2): loss ~ θ^8 for d=8
detection_rate(0.1)       # 0.002498 = sin²(0.05)
miss_conditional_fidelity()  # 1.0 (missed injection is harmless)
```

| Quantity | Closed form | Reference | Verified |
|---|---|---|---|
| Loss exponent | `2·⌈d/2⌉` (universal, PG & AG) | Theorem 10.31.1.05 | d=3 → slope 3.99; d=8 → 7.96 |
| Detection rate | `sin²(θ/2)` (code-independent) | 10.29 Prop. 2 | dev < 3.8e-16 |
| Miss-path fidelity | `1` (project back to code space) | 10.29 Prop. 2b | dev < 2.2e-16 |

### 8. Open-problem closed forms: RM(1,m) degeneracy + d−1 layer (NEW)

Answers to two open problems of 10.30 §8, verified by full enumeration:

```python
from qecgeo import rm1_w2_degeneracy, ag_dminus1_syndrome
rm1_w2_degeneracy(7)          # 127 classes × 64 pairs each (RM(1,7), verified)
ag_dminus1_syndrome(6, 1)     # 64 classes × 651 (d−1 layer, = PG(5,2) lines)
```

**O1 — RM(1,m) weight-2 degeneracy** (10.30 §8 O1): class count = `2^m − 1`,
size per class = `2^{m−1}`, all classes share a fixed difference vector a⊕b
(parallelogram). Verified m = 4..10; note the 10.30 text's "63" is a typo for
`2^5−1 = 31` (confirmed by enumeration and the QEC paper).

**O2 — weight d−1 layer syndrome distribution** (10.30 §8 O2, r=1 case):
class count = `2^m`, size per class = `(2^m−1)(2^m−2)/6` — exactly the
PG(m−1,2) line count. A deep duality between the two complete families
(AG d−1 layer degeneracy = PG logical count). Verified m = 4..6.

### 9. General RM(r,m) degeneracy classes — O1 for all r (NEW)

The RM(1,m) weight-2 answer (O1) generalizes to **every** r: for CSS(RM(r,m))
the weight-`2^r` degeneracy classes are determined completely by the affine
span dimension of the error support (10.32 inclusion-equivalence, Theorem
10.32.1.01):

| Class type | Affine span of A | # classes | size/class |
|---|---|---|---|
| r-flat class | exactly r (A is an r-flat) | `[m r]_2` (Gaussian binomial) | `2^{m−r}` |
| (r+1)-aff class | exactly r+1 (A in unique (r+1)-flat) | `flats(m,r+1)·E(r+1,2^r)/2` | 2 |

```python
from qecgeo import rm_degeneracy_classes
rm_degeneracy_classes(5, 2)   # RM(2,5): 17515 classes
#  n_flat_classes=155 × size 8 (2^{5−2})  +  n_aff_classes=17360 × size 2
rm_degeneracy_classes(8, 3)   # ratio 1.007e-4  (matches 10.32 closed form)
```

- **r=1 degenerates to O1**: `E(2,2)=0` → only the r-flat class survives,
  class count `2^m−1`, size `2^{m−1}` — exactly `rm1_w2_degeneracy`.
- **Member conservation**: `[m r]_2·2^{m−r} + flats(m,r+1)·E(r+1,2^r)`
  = the 10.33 degeneracy-ratio numerator (full degeneracy for r ≤ 2;
  partial for r ≥ 3, e.g. RM(3,8) ratio ≈ 1.007×10⁻⁴).
- Verified by full enumeration for RM(1,4..6), RM(2,4/5), RM(3,4)
  (`scripts/verify_degeneracy_classes.py`); all member counts + ratios
  conserved up to RM(8,4).

## Honest limitations

- The A0/A1 geometric separation is a **sub-threshold phenomenon**: at noise
  above threshold the error patterns lose structure and the distinction
  vanishes. It is a diagnostic of the *decoding error topology*, not a decoder
  itself.
- Threshold closed form is a **single-round, memoryless optimal-decoding model**
  — it does not capture circuit-level noise correlations or multi-round
  propagation. Real circuit thresholds differ (typically lower).
- [[5,1,3]] has η = 1 (all weight-2 errors are mis-recovered): its p_th = 10%
  is an upper-bound estimate in this model, not a practical circuit threshold.
- Surface-code geometry was characterized for L = 4/6/8 (one code family);
  crossing-rate statistics fluctuate at the few-percent level with shot count
  (1.2–2.1× range). The relative separation (A1/A0) decays with L (background
  dilution); the absolute scaling law (A1 − A0 = L) is the robust observable.
- Color-code numbers above were produced with chromobius (its native decoder):
  MWPM is structurally inapplicable to the three-color structure (d ≥ 7
  matching graph has boundary-free components). For color codes always use
  `decoder='chromobius'`.

### 4. AG complete-code closed forms: parameters without circuits (NEW)

`qecgeo.closedform` computes the full error-correction parameter table for the
AG complete-code family `[[2^m, k, 2^{r+1}]]` **directly from combinatorial
closed forms** — no circuit generation, no simulation:

```python
from qecgeo.closedform import ag_params, zero_loss_boundary, loss_at_theta

p = ag_params(10, 3)                      # [[1024, 672, 16]]
print(p['rate'])                          # encoding rate 0.65625
print(zero_loss_boundary(p['d']))         # 7 = floor((d-1)/2)
print(loss_at_theta(p['c_d'], p['d'], 0.01))  # 1.05e-24
```

| Quantity | Closed form | Reference |
|---|---|---|
| Code `[[n,k,d]]` | `[[2^m, n-2·dim RM(r,m), 2^{r+1}]]` | 10.30 |
| `fail(w0)` | `1 - Pr/(v_r·P(w0)) - Pr1/(v_r1·P(w0))` | Lemma 10.35.2.07 |
| `κ_r(m)` | `2^{(r+1)(m-r-1)} / [m choose r+1]_2` | Lemma 10.35.2.10 |
| `loss(θ)` | `c_d·θ^d`, `c_d = C(n,w0)·P(w0)·fail(w0)·κ·2^{-2w0}` | Theorem 10.35.1.07 |
| Zero-loss boundary | `k_max = ⌊(d-1)/2⌋` | Theorem 10.31.1.01 |
| Transversal gates | `{Pauli, CNOT, H, phase, logical measurement}` | Prop. 10.30.3.01 |

Run `scripts/closedform_table.py --theta 0.01 --verify` for the full table
(24 codes, loss spans ~49 orders of magnitude) plus cross-validation:
[[15,7,3]] exact enumeration (315/945 = 1/3) and threshold p_th = 1/A = 2.86%
— above the surface-code ~1%.

`scripts/verify_closedform.py` independently verifies the closed-form
components against exact enumeration on [[16,6,4]] (Pr, Pr1, P(w0) all match;
zero-loss boundary holds; θ⁴ slope ≈ 4 reproduced on the [[7,1,3]] statevector
simulation).

## Theory source

The A0/A1 classification, threshold closed form, and anyon typing derive from
the Geometric Theory of quantum error correction (Ouyang Guobin):

- Article 10.27 — geometric code construction [[5,1,3]], [[7,1,3]], [[9,1,3]]
- Article 10.28 — PG complete codes [[2^m-1, 2^m-1-2m, 3]], direction completeness
- Article 10.30 — AG complete codes [[2^m, k, 2^{r+1}]] construction
- Article 10.33 — degeneracy-ratio closed forms P_r(m), P'_r(m)
- Article 10.31 — zero-loss injection theorem (k ≤ ⌊(d-1)/2⌋)
- Article 10.35 — loss-scaling closed form loss(θ) = c_d·θ^d
- Article 10.44 — threshold closed form p_th = 1/(η·C(n,2)), Ising anyon typing
- Article 10.54 — error-pattern geometry: A0/A1 pairing-layer separation

## License

MIT © 2026 Ouyang Guobin
