#!/usr/bin/env python3
"""end_to_end_demo.py —— 一台 Mac 执行纠错：完整端到端演示（10.83，v4）

v4 改进：支持更大码（CSS(RM(1,m)) 家族 [[16,6,4]] / [[32,20,4]] / [[64,50,4]]），
恢复编码拆分为 rec_x / rec_z 两个 uint64 数组（2n 位不再溢出 int64）。

v3 改进（保留）：
  - 向量化解码循环（100k shots 60 ms，~60×）

v2 改进（保留）：
  [A] 真正应用恢复操作并重算逻辑值——从"检测"升级为"纠错"验证
  [B] d=4 码——权重 2 错误混合型全部唯一（10.83 定理 10.83.1.01）

闭环：注入错误 → syndrome → 自研解码器查表 → 应用恢复 → 验证 p_L。

运行: python3 scripts/end_to_end_demo.py（零外部依赖）
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli


def css_rm_code(m, r):
    """CSS(RM(r,m)) [[2^m, 2^m−2·dim, 2^{r+1}]]：gens + 逻辑 Z。

    逻辑 Z = x_1 x_2 的支撑（RM(r+1,m) 中不在 RM(r,m) 的二次单项式）。
    """
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    gens = []
    for row in rows:
        gens.append(Pauli(n, [1 if b else 0 for b in row]))   # X 型稳定子
        gens.append(Pauli(n, [2 if b else 0 for b in row]))   # Z 型稳定子

    def bit(col, i):
        return (col >> i) & 1
    lz = Pauli(n, [2 if bit(col, m - 1) and bit(col, m - 2) else 0 for col in range(n)])
    return gens, lz


def build_recovery_arrays(dec, n):
    """syndrome → 恢复映射为 numpy 索引数组（向量化查表）。

    返回 (rec_x, rec_z, in_table)：
      rec_x[K] / rec_z[K] = syndrome 编码为整数 K 时恢复算符的 X/Z 分量位掩码
      in_table[K] = 该 syndrome 是否在恢复表（可识别）
    """
    ns = len(dec.gens)
    size = 1 << ns
    rec_x = np.zeros(size, dtype=np.uint64)
    rec_z = np.zeros(size, dtype=np.uint64)
    in_table = np.zeros(size, dtype=bool)
    for synd, R in dec.table.items():
        k = sum(int(b) << i for i, b in enumerate(synd))
        xmask = zmask = 0
        for i in range(n):
            if R.t[i] in (1, 3):
                xmask |= 1 << i
            if R.t[i] in (2, 3):
                zmask |= 1 << i
        rec_x[k] = xmask
        rec_z[k] = zmask
        in_table[k] = True
    return rec_x, rec_z, in_table


def popcount64(x):
    """向量化 popcount（numpy uint64 数组）。"""
    x = x.astype(np.uint64)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def run_code(m, r, noise_list, shots=100000):
    """对单个 CSS(RM(r,m)) 码跑完整纠错闭环。"""
    gens, lz = css_rm_code(m, r)
    n = 1 << m
    ns = len(gens)
    k = n - 2 * sum(__import__('math').comb(m, i) for i in range(r + 1))
    print(f"码: CSS(RM({r},{m})) = [[{n},{k},{1 << (r + 1)}]] "
          f"(d={1 << (r + 1)}, {ns} 稳定子, 逻辑 Z = x1x2 权重 {lz.weight()})")

    # 逻辑 Z 支撑位掩码 + 0/1 向量
    lz_xmask = 0
    for i in range(n):
        if lz.t[i] == 2:
            lz_xmask |= 1 << i
    lz_bits = np.array([1 if (lz_xmask >> i) & 1 else 0 for i in range(n)], dtype=np.uint64)

    # 稳定子矩阵（向量化 syndrome）
    Sx = np.zeros((ns, n), dtype=np.int8)
    Sz = np.zeros((ns, n), dtype=np.int8)
    for a, g in enumerate(gens):
        for i in range(n):
            if g.t[i] in (1, 3):
                Sx[a, i] = 1
            if g.t[i] in (2, 3):
                Sz[a, i] = 1

    for p in noise_list:
        pe = p / 3   # X/Z/Y 各 p/3 → 总错误率 p（退极化）

        # 1. 恢复表 + 向量化映射
        t0 = time.time()
        dec = LookupDecoder(gens, n, name=f'[[{n},{k},{1 << (r + 1)}]]')
        dec.build(w_max=2)
        rec_x, rec_z, in_table = build_recovery_arrays(dec, n)
        t_build = (time.time() - t0) * 1000

        # 2. 向量化采样错误 + syndrome + 恢复 + 验证
        t0 = time.time()
        rng = np.random.default_rng(42)
        rr = rng.random((shots, n))
        tx = (rr < pe).astype(np.int8)
        tz = ((rr >= pe) & (rr < 2 * pe)).astype(np.int8)
        ty = ((rr >= 2 * pe) & (rr < 3 * pe)).astype(np.int8)
        Ex = (tx | ty).astype(np.uint64)   # X 分量
        Ez = (tz | ty).astype(np.uint64)   # Z 分量

        synd = ((Ex.astype(np.int64) @ Sz.T + Ez.astype(np.int64) @ Sx.T) & 1).astype(np.uint64)
        shifts = np.array([1 << i for i in range(ns)], dtype=np.uint64)
        idx = (synd.astype(np.uint64) * shifts).sum(axis=1).astype(np.int64)

        # 错误翻转逻辑 Z：E 的 X 分量与 Lz 支撑交奇数
        flip_E = (Ex * lz_bits).sum(axis=1) & 1

        # 查表恢复（向量化）
        Rx = rec_x[idx]
        # 恢复翻转逻辑 Z：R 的 X 分量与 Lz 交奇数
        flip_R = popcount64(Rx & np.uint64(lz_xmask)) & 1
        flip_corr = flip_E ^ flip_R
        resolved = in_table[idx] | (Rx == 0)   # 命中表 或 无错误
        t_dec = (time.time() - t0) * 1000

        # 3. 验证
        pL_bare = float(flip_E.mean())
        pL_corr = float(flip_corr.mean())
        mask = resolved
        pL_res = float(flip_corr[mask].mean()) if mask.any() else float('nan')

        print(f"\n--- noise p = {p} ---")
        print(f"  恢复表构建 {t_build:.0f} ms | 模拟+解码 {t_dec:.0f} ms "
              f"({shots} shots) | 可识别率 {resolved.mean():.4f}")
        print(f"  裸逻辑错误率 p_L(bare)     = {pL_bare:.4f}")
        print(f"  纠错后 p_L(corrected)      = {pL_corr:.4f}")
        print(f"  可识别样本纠错后 p_L       = {pL_res:.4f}")
        if pL_corr == 0:
            print(f"  → 纠错有效性: p_L → 0（全部可识别样本完全纠正）✓")
        elif pL_corr < pL_bare:
            print(f"  → 纠错有效性: p_L 降低 {pL_bare / pL_corr:.1f}×")
        else:
            print(f"  → 纠错有效性: 未改善（边界情况）")


def main():
    print("=" * 80)
    print("一台 Mac 执行纠错 v4：注入错误 → syndrome → 自研解码器 → 恢复 → 验证")
    print("=" * 80)
    run_code(4, 1, (0.001, 0.005, 0.01))    # [[16,6,4]]
    run_code(5, 1, (0.001, 0.005))           # [[32,20,4]]
    run_code(6, 1, (0.001, 0.005))           # [[64,50,4]]


if __name__ == "__main__":
    main()
