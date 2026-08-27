# GitHub Discussion 草案 —— 给 tqec 维护者的开场

> 用法：在 tqec 仓库开一个 Discussion（或发邮件给维护者），标题 + 正文如下。
> 语气：只陈述事实，邀请对方看数据，不评价他们的项目。

---

## 标题（三选一）

- (A) A custom sinter.Decoder with raw benchmark data — for your interest
- (B) A lookup-table decoder via the sinter.Decoder interface: benchmark data included
- (C) [Data] Custom decoder vs pymatching: reproducible numbers

## 正文（英文）

Hi tqec team,

I built a decoder that plugs into the stim/sinter ecosystem through the
`sinter.Decoder` interface (file-based `decode_via_files`), and ran a
reproducible benchmark against pymatching. Since your project uses sinter for
decoding, you may find the numbers interesting.

What's in the repo (all reproducible):

- `scripts/sinter_lookup_decoder.py` — a `sinter.Decoder` subclass: a lookup
  table over syndromes (errors enumerated up to weight 2), with the b8
  file-based interface.
- `scripts/sinter_collect_demo.py` — `sinter.collect` run with
  `custom_decoders={'lookup': ...}` on two code families under the same
  circuit-level noise model (rounds=2 differential, data depolarization +
  measurement flip, 5000 shots/cell, fixed seed).
- `PRESENTATION.md` — the raw results table (no analysis).
- `data/sinter_benchmark.csv` — the raw numbers.

Codes covered: CSS(RM(1,4)) `[[16,6,4]]` (26-qubit circuit) and surface d=3
(rotated memory-Z). Decoders: the custom lookup table and pymatching.

If any of this is useful for your decoder benchmarking, feel free to use it.
Happy to provide more data points (larger distances, more noise levels) on
request.

Best,
Ouyang Guobin
(Geometric Theory of QEC; zero-dependency core at
https://github.com/sdoygb/conjugate-spectral-geometry — the theory write-up
is in the articles/ directory of that repository)

---

## 备选：更简版（如果 Discussion 被拒/不看）

> Hi, I have a custom `sinter.Decoder` with a reproducible benchmark
> (CSS(RM(1,4)) vs surface d=3, vs pymatching, 5000 shots/cell).
> Raw data: PRESENTATION.md. Let me know if you'd like larger runs.
