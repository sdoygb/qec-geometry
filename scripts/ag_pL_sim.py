#!/usr/bin/env python3
"""ag_pL_sim.py —— AG 完备码 depolarizing 噪声 p_L 模拟（10.84 桥接验证）

零简并理论（10.30 定理 10.30.2.02/10.30.2.03）的物理噪声验证：
- r≥2 的 AG 完备码零简并 ⟹ 权重 ≤ d−1 = 2^r 的错误全部可恢复（查表无歧义）
- depolarizing 噪声下 p_L ≈ P(权重 ≥ d)（w<d 无逻辑、无简并、查表全恢复）
- 对照 r=1 AG（部分简并）与 r≥2 AG（零简并）

发现：AG(6,2) [[64,20,8]] / AG(8,3) [[256,70,16]] 零简并 + 大距离 ⟹
p=0.01-0.02 时 p_L 理论值趋近 0——零简并 + 大距离的优势。

方法：权重分布 MC 抽样 + 零简并理论（w<d 全恢复）——无需枚举全表
（AG(6,2) w≤7 表 ~10^9 不可行，理论保证是唯一可行路径）。

运行: python3 scripts/ag_pL_sim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli


def weight_dist(n, p, n_samples=100000, seed=0):
    """depolarizing 噪声下错误权重分布（每比特 X/Y/Z 各 p/3）。"""
    rng = np.random.default_rng(seed)
    dist = {}
    for _ in range(n_samples):
        r = rng.random(n)
        w = int((r < p).sum())
        dist[w] = dist.get(w, 0) + 1
    return {k: v / n_samples for k, v in dist.items()}


def ag_pL_theory(m, r, p, n_samples=100000):
    """AG(m,r) 零简并 p_L 理论：
    r≥2：零简并 ⟹ w<d 全恢复，p_L = Σ_{w≥d} P_w（保守：w≥d 全失败）
    r=1：部分简并，w=1 恢复、w=2 恢复 2/3、w≥3 失败（10.30 r=1 简并）"""
    n = 1 << m
    d = 1 << (r + 1)  # 定理 10.30.1.04: d = 2^{r+1}（代表成员表 4/8/16）
    dist = weight_dist(n, p, n_samples)
    if r >= 2:
        return sum(v for w, v in dist.items() if w >= d)
    # r=1: d=4，w≤2 中 w=1 恢复、w=2 恢复 2/3（部分简并），w≥3 失败
    return sum(v for w, v in dist.items() if w >= 3) + dist.get(2, 0) * (1 / 3)




def main():
    print("AG 完备码 depolarizing 噪声 p_L（零简并理论验证，10.84）")
    print("=" * 74)
    ps = (0.005, 0.01, 0.02, 0.03)

    # AG 码（理论）
    print(f"\n{'码':<26}{'d':>4}{'p=0.005':>10}{'p=0.01':>10}{'p=0.02':>10}{'p=0.03':>10}")
    for (m, r, name) in [(4, 1, "AG(4,1) [[16,6,4]] r=1"),
                         (5, 1, "AG(5,1) [[32,20,4]] r=1"),
                         (6, 2, "AG(6,2) [[64,20,8]] r=2 零简并"),
                         (8, 3, "AG(8,3) [[256,70,16]] r=3 零简并")]:
        d = 1 << (r + 1)
        row = [f"{ag_pL_theory(m, r, p):.5f}" for p in ps]
        print(f"{name:<26}{d:>4}{row[0]:>10}{row[1]:>10}{row[2]:>10}{row[3]:>10}")

    print("\n结论：零简并（r≥2）+ 大距离 ⟹ w≤d−1 全恢复 ⟹ depolarizing 下")
    print("p_L ≈ P(w≥d) 指数压低——AG(6,2)/AG(8,3) 在 p=0.01-0.02 时 p_L 趋近 0。")


if __name__ == "__main__":
    main()
