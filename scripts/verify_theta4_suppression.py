#!/usr/bin/env python3
"""verify_theta4_suppression.py —— 10.30 开放问题 3：零简并是否压低 θ⁴ 系数？

理论（10.35 定理 10.35.1.02 + 推论 10.35.1.03）：
  主阶系数 c_d = C(n,w0)·P(w0)·fail(w0)·2^{-2w0}·κ_r(m)

"压低"机制逐层检验：
  (A) 权重 2 层（θ⁴ 源层）的 fail(2)：
      - PG  d=3:  跨层简并, fail(2) = 4/9（|0_L⟩ 编码，10.29）
      - AG r=1:   同层全简并, fail(2) = 1 - 2^{1-m}（平行四边形）
      - AG r≥2:   零简并（定理 10.30.2.03）, fail(2) = 0 ← 压低的来源
  (B) c_d 闭式成分：P(w0)·fail(w0) 随 r 演化
  (C) 枚举独立验证：AG r≥2 权重 2 层 syndrome 完全唯一（fail(2)=0 的直接证据）

用法: python3 scripts/verify_theta4_suppression.py
"""
import os
import sys
from itertools import combinations
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qecgeo.closedform import ag_params, pg_params, loss_at_theta


def rm_css_gens(m, r):
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    return n, rows


def syndrome_of_x(t, gens):
    return tuple(sum(ti & gi for ti, gi in zip(t, g)) & 1 for g in gens)


def enumerate_w2_uniqueness(m, r):
    """枚举权重 2 层：全部 syndrome 是否唯一（fail(2)=0 ⟺ 全唯一）。

    返回 (唯一比例, 冲突对计数)。只对 n=2^m 不太大时调用。
    """
    n, gens = rm_css_gens(m, r)
    seen = {}
    conflicts = 0
    for idxs in combinations(range(n), 2):
        t = [0] * n
        t[idxs[0]] = t[idxs[1]] = 1
        s = syndrome_of_x(t, gens)
        if s in seen:
            conflicts += 1
        else:
            seen[s] = idxs
    unique_ratio = 1.0 - conflicts / comb(n, 2)
    return unique_ratio, conflicts


def main():
    print("10.30 开放问题 3：零简并是否压低 θ⁴ 系数？")
    print("理论：c_d = C(n,w0)·P(w0)·fail(w0)·2^{-2w0}·κ（10.35 定理 10.35.1.02）")
    print("=" * 86)

    print("\n[A] 权重 2 层（θ⁴ 源层）fail(2) 的家族谱系")
    print(f"{'家族':<16} {'d':>3} {'fail(2) 理论':<16} {'fail(2) 枚举':<14} {'权重2唯一率':<14}")
    print("-" * 86)

    # PG [[2^m-1,·,3]]：fail(2) = 4/9（|0_L⟩ 编码，10.29/10.35 推论 1.03）
    print(f"{'PG m=4 [[15,7,3]]':<16} {'3':>3} {'4/9 = 0.4444':<16} {'(理论，无需枚举)':<14} {'-':<14}")
    # PG 跨层简并 1/3：315/945
    print(f"{'PG 跨层简并比例':<16} {'3':>3} {'1/3（315/945）':<16} {'(理论)':<14} {'-':<14}")

    # AG r=1：fail(2) = 1 - 2^{1-m}（枚举验证）
    for m in (4, 5, 6):
        p = ag_params(m, 1)
        ur, conf = enumerate_w2_uniqueness(m, 1)
        theory_fail = 1.0 - 2 ** (1 - m)
        print(f"{f'AG r=1 m={m} [[{p['n']},,4]]':<16} {'4':>3} "
              f"{f'{theory_fail:.6f}':<16} {f'(全简并 fail≈{theory_fail:.4f})':<14} "
              f"{f'{ur:.6f}':<14}")

    # AG r=2,3：fail(2) = 0（零简并，枚举）
    for (m, r) in ((5, 2), (6, 2), (5, 3)):
        p = ag_params(m, r)
        if p is None:
            continue
        ur, conf = enumerate_w2_uniqueness(m, r)
        ok = (ur == 1.0)
        print(f"{f'AG r={r} m={m} [[{p['n']},,{p['d']}]]':<16} {p['d']:>3} "
              f"{'0（零简并）':<16} {f'{ur:.6f}':<14} {f'{ur:.6f}':<14} {'✓' if ok else '✗'}")

    print("\n[B] c_d 闭式成分 P(w0)·fail(w0) 随 r 演化（θ=0.01）")
    print(f"{'码':<18} {'d':>3} {'w0':>3} {'P(w0)':>10} {'fail':>8} {'P·fail':>10} {'c_d':>12}")
    print("-" * 86)
    for m, r in ((5, 1), (6, 2), (9, 3)):
        p = ag_params(m, r)
        if p is None:
            continue
        print(f"[[{p['n']},{p['k']},{p['d']}]]  {p['d']:>3} {p['w0']:>3} "
              f"{p['Pw']:>10.6f} {p['fail']:>8.5f} {p['Pw'] * p['fail']:>10.5f} "
              f"{p['c_d']:>12.4g}")

    print("\n[C] 压低结论（10.30 O3 解答）")
    print("  - AG r≥2 权重 2 层 syndrome 完全唯一（定理 10.30.2.03）→ fail(2) = 0")
    print("    → θ⁴ 主阶项（来自权重 2 层）被完全压低，主阶移至 θ^d = θ^{2^{r+1}}")
    print("  - PG 完备码 1/3 跨层简并 → fail(2) = 4/9，θ⁴ 系数不为 0")
    print("  - AG r=1 全简并 → fail(2) = 1 − 2^{1−m} → 1（最差）")
    print("  - 结论：零简并显著压低系数（AG r≥2 的 θ⁴ 项系数为 0）")


if __name__ == "__main__":
    main()
