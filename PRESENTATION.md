# A custom decoder in the stim/sinter ecosystem

**A reproducible benchmark of a custom decoder (`sinter.Decoder` interface) against pymatching, on two code families, under the same circuit-level noise model.**

No conclusions are drawn here. The experiment, the code, and the raw data are below.

---

## 1. What was run

**Toolchain**: `stim` 1.16, `sinter` 1.16, `pymatching` 2.4, Python 3.11.
Reproduce with:

```bash
.venv311/bin/python scripts/sinter_collect_demo.py
# writes data/sinter_benchmark.csv
```

**Protocol** (identical for both codes):

- Circuit: rounds = 2 (reference round + differential detectors)
- Noise: `before_round_data_depolarization = p` on data,
  `before_measure_flip_probability = 0.01`, `after_reset_flip_probability = 0.01`
- Decoders: `lookup` (custom, `sinter.Decoder` interface, this repository) and
  `pymatching` (MWPM)
- 5000 shots per (code, decoder, p) cell; fixed seed

**Codes**:

| code | physical qubits | distance | stabilizer weight |
|---|---|---|---|
| CSS(RM(1,4)) `[[16,6,4]]` | 26 (16 data + 10 ancilla) | 4 | 4 / 8 |
| surface code d=3 (rotated, memory-Z) | 17 | 3 | 4 |

The custom decoder (`scripts/sinter_lookup_decoder.py`) is a lookup table over
syndromes, built by enumerating errors up to weight 2.

## 2. Raw results

`data/sinter_benchmark.csv` (5000 shots/cell):

| code | decoder | p | p_L |
|---|---|---|---|
| CSS(RM(1,4)) `[[16,6,4]]` | lookup | 0.01 | 0.003200 |
| CSS(RM(1,4)) `[[16,6,4]]` | lookup | 0.02 | 0.012400 |
| CSS(RM(1,4)) `[[16,6,4]]` | lookup | 0.03 | 0.020400 |
| CSS(RM(1,4)) `[[16,6,4]]` | pymatching | 0.01 | 0.031200 |
| CSS(RM(1,4)) `[[16,6,4]]` | pymatching | 0.02 | 0.051800 |
| CSS(RM(1,4)) `[[16,6,4]]` | pymatching | 0.03 | 0.069200 |
| surface d=3 | lookup | 0.01 | 0.090000 |
| surface d=3 | lookup | 0.02 | 0.128000 |
| surface d=3 | lookup | 0.03 | 0.158000 |
| surface d=3 | pymatching | 0.01 | 0.014400 |
| surface d=3 | pymatching | 0.02 | 0.024600 |
| surface d=3 | pymatching | 0.03 | 0.038000 |

## 3. Supporting material (theory behind the code, all in one library)

The lookup decoder and the code family are documented in a knowledge library
(Chinese, with formulas):

- Article 10.30 — AG complete-code family `[[2^m, k, 2^{r+1}]]`, syndrome-class
  structure (full classification: r-flat classes and (r+1)-affine classes)
- Article 10.83 — the decoder: syndrome → minimum-weight recovery, failure-rate
  closed form
- Article 10.84 — this benchmark and the cross-code analysis
- Articles 10.35, 10.44 — loss-scaling closed form `loss(θ) = c_d·θ^d`,
  threshold closed form `p_th = 1/(η·C(n,2))`

All scripts: `scripts/*.py`, all tests: `pytest tests/` (47 passed).

## 4. Environment

- qec-geometry repository: zero-dependency core (`qecgeo/`), stim/sinter only
  for the simulation scripts
- `scripts/ag_stim_memory.py` — circuit construction (rounds=2 differential,
  stabilizer extraction from stim)
- `scripts/sinter_lookup_decoder.py` — the custom decoder (sinter.Decoder)
- `scripts/sinter_collect_demo.py` — this benchmark
