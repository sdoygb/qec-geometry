# qec-geometry

**Geometric diagnostics for quantum error correction codes.**

A zero-dependency Python toolkit that classifies decoding error patterns by their
topology (A0: local / trivial, A1: non-trivial crossing chains), predicts
fault-tolerance thresholds from a closed-form formula, decodes Reed-Muller
CSS codes from their error moments, and analyzes the degeneracy structure of
mainstream codes (surface / LDPC) with the geometric-theory framework.

```
qecgeo  ──  Pauli algebra ── stabilizer framework ── code zoo ── error geometry ── threshold closed form ── anyon typing
         ── RM moment decoder (low-weight lookup + high-weight MILP fallback)
         ── degeneracy-class analysis (AG / surface / LDPC)
         ── stim multi-round simulation + measurement noise
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
python3 -m pytest tests/ -q              # 47 tests
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

### 10. Self-contained lookup decoder + geometric recovery table (NEW)

A decoder **written by us** — no stim/pymatching dependency — that closes the
"execute" loop on a laptop: build a syndrome→recovery table by enumerating all
errors up to weight `w_max`, recovering with the minimum-weight representative
of each syndrome class (`qecgeo.decoder.LookupDecoder`).

```python
from qecgeo import LookupDecoder
from qecgeo.codes import steane_code

code = steane_code()
dec = LookupDecoder(code.gens, code.n, name=code.name)
dec.build(w_max=2)                 # 63 syndromes, ~3 ms on a Mac
dec.decode(code.syndrome_of(some_error))   # → recovery Pauli (O(1) lookup)
dec.fail_rate(2)                   # decoding failure rate at weight 2
dec.weight2_uniqueness()           # 1.0 = zero-degeneracy layer
```

**Design = geometric recovery table** (10.30/10.35): recovery representative =
minimum weight in the syndrome class; decoding fails ⟺ residual ∉ stabilizer
group = logical error. Class structure = the degeneracy classes of Theorem
10.30.2.05.

**New closed form discovered by the decoder**: for AG r=1, weight-2 degeneracy
ratio is exactly **1/3** (m-independent), class size `2^{m−1}`, hence
`fail(2) = 1/3 − 1/(3·2^{m−1})` (verified m = 4, 5 by enumeration). For AG r≥2
the weight-2 layer is zero-degenerate: uniqueness 1.0, fail(2) = 0, class count
= error count (C(32,2)·9 = 4464 for [[32,·,8]]).

Verified (260827, after fixing `decode_error` to use exact group membership):
[[5,1,3]]/[[7,1,3]]/[[15,7,3]]/[[16,6,4]] — **weight-1 errors all recover**
(0 failures); weight-2 failures match the true correctable range (d=3 codes
correct ≤ (d−1)/2 = 1 error, so weight-2 logical failures are expected:
5-qubit 90/105, Steane 147/210, Hamming 735/1020; AG [[16,6,4]] weight-2
fail 315/945 = 1/3). fail(2) formula matches 10.35 Thm 1.02 in its AG
even-distance domain (verified 0.3125/0.2917 exact)
(`scripts/verify_lookup_decoder.py`).

### 11. RM moment decoder: low-weight lookup + high-weight MILP fallback (NEW)

Decode **CSS(RM(r,m)) X-errors from their moments** `m_I = Σ_{a∈A} x_I(a)`
(|I| ≤ r): the error set A is uniquely determined by its moments for weights
≤ (d−1)/2 (the correctable range, guaranteed); at larger weights small m
shows collisions (e.g. r=1 weight-2 collides for all m — linear moments are
insufficient; r=2 m<7 weight-4 collides ~1%). Two regimes:

| regime | method | latency | coverage |
|---|---|---|---|
| |A| ≤ 4 | precomputed single-point-moment lookup table | ms | any n (verified d=32) |
| 5 ≤ \|A\| ≤ 8, n ≤ 128 | scipy.milp minimum-weight solution of G·e ≡ m (mod 2) | seconds | m ≤ 7 |
| n = 256 (16-point) | — | timeout | open (needs true Reed recursion) |

```python
from qecgeo import moments_of, rm_x_decode
A = [3, 17, 54, 91]                      # 4-point error
mm = moments_of(A, m=7, r=3)             # all moments up to degree 3
rm_x_decode(mm, 7, 3)                    # → sorted(A), ms on a Mac
```

**Honest boundary** (verified): the MILP fallback works for n ≤ 128
(m=5/6/7, 5–8 points, seconds); n = 256 16-point errors time out — the
"true Reed recursion" in the moment domain is an **open algorithmic problem**
(equivalent to the coset-leader problem, generally NP-hard). Coverage
recomputed honestly: for n = 1024, p = 0.001 the weight ≤ 4 fast path covers
91.5% of errors — the high-weight path matters for real noise (42% coverage
at p = 0.005 with weight ≤ 4 only).

**Performance cliff** (measured 260827): the weight-3/4 fast paths are slow
at large n — w=3 @ n=1024 ≈ 37 s/decode, w=4 @ n=512 ≈ 179 s, w=4 @ n=1024
≈ 11.7 min. The decoder is practical when errors ≤ 2 dominate (99.89% at
p=0.001); single w=3/4 errors at n ≥ 512 stall decode.

### 12. Degeneracy analysis of mainstream codes: surface & LDPC (NEW)

The geometric-theory degeneracy pipeline (10.30/10.83) applied to **any**
stabilizer code, with the **min-weight decode recovery rate** as the correct
metric (an error is recoverable ⟺ it is the unique minimum-weight member of
its syndrome class — not the naive "pure weight-w class" count):

| code | n | k | d | w1 recovery | w2 recovery |
|---|---|---|---|---|---|
| HGP(rep3) [[13,·,·]] LDPC | 13 | 3 | — | 100% | 58.7% |
| HGP(rep4) [[25,·,·]] LDPC | 25 | 4 | — | 100% | 87.7% |
| AG(4,1) [[16,6,4]] | 16 | 6 | 4 | 100% | 67% |
| AG(6,2) [[64,20,8]] | 64 | 20 | 8 | 100% | **100%** (zero-degenerate) |

**Key findings**:
- All mainstream codes have 100% weight-1 recovery (single-bit errors
  unambiguous — universal good property).
- **Note (260827 erratum)**: the earlier surface w2 = 1/3 figure and
  "surface is structurally weak" conclusion were based on a non-valid
  construction (X/Z stabilizers on the same face — not a legal CSS code,
  4 anticommuting pairs). Withdrawn; the true surface w2 recovery needs an
  authoritative tool and will be reported when available.
- HGP LDPC weight-2 recovery rises with code length (58.7% → 87.7%) —
  probabilistic low degeneracy, approaching AG r≥2 by dilution.
- AG r≥2 is the **only zero-degenerate family** (Theorem 10.30.2.03:
  quadratic monomials block parallelograms → weight-2 syndromes fully unique).

```bash
python3 scripts/surface_degeneracy.py    # surface vs AG vs HGP, same metric
python3 scripts/ldpc_degeneracy.py       # HGP LDPC degeneracy scaling
```

### 13. AG zero-degeneracy under physical noise (NEW)

Zero degeneracy ⟹ all errors of weight ≤ d−1 recoverable without ambiguity.
Under **depolarizing noise** this pushes p_L ≈ P(w ≥ d) exponentially low:

| code | d | p=0.01 | p=0.02 | p=0.03 |
|---|---|---|---|---|
| AG(6,2) [[64,20,8]] (zero-deg., theory) | 8 | ≈0 | 0.00005 | 0.00068 |
| AG(8,3) [[256,70,16]] (zero-deg., theory) | 16 | ≈0 | 0.00004 | 0.00504 |

**stim multi-round + measurement noise, same conditions** (rounds=2
differential, data depolarize + MR flip p_meas=0.01, `scripts/ag_stim_memory.py`):

| code | decoder | p=0.01 | p=0.02 | p=0.03 |
|---|---|---|---|---|
| **AG(4,1) [[16,6,4]]** | **lookup (zero-deg.)** | **0.00320** | **0.01240** | **0.02040** |
| AG(4,1) | pymatching (MWPM) | 0.03120 | 0.05180 | 0.06920 |

Zero degeneracy ⟹ a lookup decoder suffices (no complex decoder needed);
on the AG code the lookup table outperforms MWPM, which is designed for
degenerate/local codes (dense syndromes). Infrastructure verified: single-round circuits give
random X-stabilizer measurements (|0⟩ is not an X eigenstate), so rounds=2
reference + differential detectors are required (noise-free → all-zero
detectors ✓); differential extraction recovers the standard RM generators ✓;
measurement noise correctly raises p_L after the syndrome bit-vector fix ✓.

**Complete spatiotemporal SCL decoding** (`scripts/ag_spatiotemporal_scl.py`,
260828): differential detectors only see d_t = H·e_t ⊕ m_t ⊕ m_{t-1} — the
data error e_t appears once (round t), the measurement error m_t appears twice
(rounds t and t+1). A decoder that only uses the last differential (old
pipeline) silently drops all intermediate-round data errors. The complete
pipeline iterates: per-round SCL moment decode (data errors) → residual
d_t ⊕ H·ê_t ≈ m_t ⊕ m_{t-1} → 1D repetition-code time-chain decode of the
measurement errors (per-stabilizer independent, m[t] = m[0] ⊕ Σ d, min-weight
over m[0]∈{0,1}) → subtract → re-SCL:

| code | condition | old (last-diff only) | complete SCL | gain |
|---|---|---|---|---|
| **AG(6,2) [[64,20,8]]** | p=0.001, p_meas=0, r=3 | 0.0085 | **0.00000** (0/2000) | ∞ |
| AG(6,2) | p=0.001, p_meas=0, r=5 | — | **0.00000** (0/3000) | ∞ |
| AG(6,2) | p=0.001, p_meas=0.001, r=5 | — | **0.00033** (1/3000) | 26× |
| AG(6,2) | p=0.001, p_meas=0.01, r=5 | (was stuck) | **0.00767** (23/3000) | ∞ |
| AG(6,2) | p=0.003, p_meas=0, r=5 | — | **0.00000** (0/2000) | ∞ |
| AG(6,2) | p=0.01, p_meas=0, r=5 | — | **0.00100** (2/2000) | — |
| **AG(4,1) [[16,6,4]]** | p=0.01, p_meas=0, r=3 | 0.0575* | **0.0057** | 10× |
| AG(4,1) | p=0.01, p_meas=0.01, r=4 | 0.075 | **0.0090** | 8.3× |

(*old 0.0575 = last-diff-only baseline after the commutation fix; 0.0085 was
the CNOT-decomposition gate-level number, superseded by the MQ model.)
Implementation notes: detectors must be appended immediately after each round's
MR (rec is relative to the append point; a delayed append makes every detector
identically zero); X-stabilizer measurements detect Z-type errors and
vice-versa (commutation, not label); time-chain decode is the prefix-sum
m[t]=m[0]⊕Σ_{i<t}d[i] with min-weight choice of m[0].

**r=2 四点/三点代数参数化**（`qecgeo/rm_scl_decoder.py`，260828）：4 点
A={a,a⊕p,a⊕q,a⊕r} 的总 XOR = p⊕q⊕r = 线性矩 d。d=0 ⟹ 必为平行四边形
（二次矩只依赖 (p,q)，先筛再解 a）；d≠0 ⟹ 4 点 =
{a,a⊕p,a⊕q,a⊕p⊕q⊕d}，固定 (p,q) 后二次矩方程对 a 线性（≤2 候选）。
3 点 {a,a⊕p,p⊕d} 同理 O(n²)。**O(n⁴) → O(n²·m)**：m=7 (n=128) 一般
4 点 0.52s（旧 C(128,4)≈1e7 不可行）；m=6 全范围 300 次 82s→8s。
验证 m=4/5/6 随机 4 点 300/300 公式成立；tests +3（含 m=7 性能护栏）。

**迭代顺序关键修正**（`scripts/ag_spatiotemporal_scl.py` run_pL，
260828）：必须【先 SCL 后残差时间链】——先时间链会把数据错误
syndrome 也当测量错误吞掉（解码输出恒 0，p_L = obs 率 0.023）；
且 SCL 用候选残差最小化（`_scl_best`，测量噪声污染矩时取第一个
候选常错，选 syndrome 与差分残差权重最小者），无解不中断。这使
p_meas=0.01 的 AG(6,2) 扫描从"卡死 45min"变成 268s 完成，p_L 0.110→0.0077。

**sinter.Decoder integration (complete)** — the custom-decoder interface lives
in **sinter** (bundled with stim 1.16), not stim itself. `LookupSinterDecoder`
(`scripts/sinter_lookup_decoder.py`) wraps the lookup decoder as a
`sinter.Decoder` (file-based b8 decode, 100% agreement with the manual path),
and `sinter.collect` runs the full comparison (`scripts/sinter_collect_demo.py`):

| code | decoder | p=0.01 | p=0.02 | p=0.03 |
|---|---|---|---|---|
| **AG(4,1)** | **lookup (geometry)** | **0.00540** | **0.01260** | **0.02140** |
| AG(4,1) | pymatching (MWPM) | 0.02900 | 0.05300 | 0.06900 |
| surface d=3 | pymatching (MWPM) | 0.01440 | 0.02460 | 0.03800 |

**Note**: the `lookup` rows use a decoder trained on CSS(RM(1,4)); the
`surface + lookup` row is a cross-code control (a decoder trained on one code
applied to another — not a valid benchmark), demonstrating decoder–code
mismatch only. Numbers above are from `data/sinter_benchmark.csv`
(5000 shots/cell, sinter 1.16, reproducible).

**AG vs surface：同一 MQ 门级模型**（260906，`scripts/surface_mq.py` +
`scripts/ag_vs_surface_mq.py`，原始数据 `data/ag_vs_surface_mq_p5r.csv`）：
把 surface 码放进与 AG 完全相同的 MQ 门级模型——每稳定子每轮
1×DEPOLARIZE2(p) on (ancilla, 1 参与数据)（all-to-all 平台每稳定子计
1 个多体门，与支撑权重无关）+ ancilla MR flip + 轮间数据 depolarize，
CNOT 链错误归零。surface 电路 = stim 官方 `rotated_memory_z` 只替换
错误模型（逻辑 Z / 稳定子 / DETECTOR 保持官方构造），解码
pymatching(MWPM)；AG 用自身 SCL 矩解码（`run_pL`, p_gate=p_data=p）。
rounds=5，seed 42，2000 shots/格：

| code | n,k,d | rate | p_meas | p=0.001 | p=0.003 | p=0.01 |
|---|---|---|---|---|---|---|
| **AG(6,2) [[64,20,8]]** | 20/64 | 0.3125 | 0 | 0.00050 | 0.00250 | 0.02100 |
| AG(6,2) | | | p | 0.00050 | 0.00550 | 0.05550 |
| **AG(4,1) [[16,6,4]]** | 6/16 | 0.375 | 0 | 0.00000 | 0.00050 | 0.01300 |
| AG(4,1) | | | p | 0.00000 | 0.00150 | 0.02100 |
| **surface d=3 [[17,1,3]]** | 1/17 | 0.059 | 0 | 0.00200 | 0.00450 | 0.01900 |
| surface d=3 | | | p | 0.00100 | 0.00650 | 0.01650 |
| **surface d=5 [[41,1,5]]** | 1/41 | 0.024 | 0 | 0.00050 | 0.00050 | 0.00100 |
| surface d=5 | | | p | 0.00000 | 0.00000 | 0.00150 |
| **surface d=7 [[97,1,7]]** | 1/97 | 0.010 | 0 | 0.00000 | 0.00000 | 0.00000 |
| surface d=7 | | | p | 0.00000 | 0.00000 | 0.00050 |

实测观察（如实记录，不做超越性结论）：
- 同物理 p 下两族都按 d 分层；在该 MQ 模型 + 原生解码器组合下，小支撑
  稳定子 + MWPM 的 surface（d=5/7）p_L 低于同等 p 的 AG(6,2)（d=8，
  p=0.003: surface d5 0.0005 vs AG(6,2) 0.0025）。AG 的对照优势在参数空间
  而非同 p p_L：`[[64,20,8]]` 以 64 物理比特编码 20 逻辑比特（rate 0.312），
  surface d=7 以 97 比特编码 1 逻辑比特（rate 0.010）。
- 对照含内在混淆，逐项注明：解码器不同（SCL vs MWPM，均为各自原生最优）；
  稳定子支撑权重 AG 16–64 vs surface ≤4，而 MQ 记账每稳定子只计 1 个
  DEPOLARIZE2（对高支撑 AG 有利的假设，未计 w 依赖的门的物理成本）；
  数据 depolarize / MR 位置数 ∝ n。即本表是"模型 + 原生解码器"的联合
  实测，不能单独归因于码族。

## Honest limitations

- The A0/A1 geometric separation is a **sub-threshold phenomenon**: at noise
  above threshold the error patterns lose structure and the distinction
  vanishes. It is a diagnostic of the *decoding error topology*, not a decoder
  itself.
- The `LookupDecoder` enumerates errors up to a chosen weight; it is optimal
  (minimum-weight recovery) **within that weight range** but does not handle
  weights above `w_max` (use MWPM for those). Table size grows as
  `Σ_w C(n,w)·3^w` — practical for small/medium codes, not for
  hundreds-of-qubits LDPC.
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
- **RM moment decoder**: the MILP high-weight fallback covers n ≤ 128 only;
  n = 256 16-point errors time out (true Reed recursion in the moment domain
  is an open problem, equivalent to coset-leader). Low-weight (≤ 4) lookup
  works for any n (d=32 verified).
- **Degeneracy metric**: min-weight decode recovery rate assumes minimum-weight
  decoding; a real decoder that picks a non-minimum representative (or fails
  to pick) will differ. The metric measures the *structural* recovery ceiling.
- **surface code (260827 erratum)**: the earlier "weak code" conclusion
  (w2 recovery 1/3) was based on a non-valid construction and is withdrawn.
  The surface w2 recovery requires an authoritative extraction; no claim is
  made until verified.
- **AG p_L numbers**: zero-degeneracy values are theoretical (w < d fully
  recovered by the zero-degeneracy theorem, w ≥ d conservatively counted as
  failure); the stim multi-round AG(4,1) lookup-vs-pymatching comparison is
  fully simulated (same circuit-level noise model).
- **AG vs surface MQ (260906)**: surface rows use the stim-official circuit
  transformed to the MQ gate model + pymatching; AG rows use its own SCL
  decoder. The cross-family table compares *equal physical p* under the
  MQ gate-cost assumption (one 2-body DEPOLARIZE2 per stabilizer, independent
  of stabilizer weight) — not a rate- or decoder-matched benchmark.

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
- Article 10.83 — full-type failure-rate closed form + RM moment decoder
- Article 10.84 — mainstream-code bridge: degeneracy of surface/LDPC + AG
  zero-degeneracy under depolarizing noise

## License

MIT © 2026 Ouyang Guobin
