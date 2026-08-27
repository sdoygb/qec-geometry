#!/usr/bin/env python3
"""end_to_end_demo.py —— 一台 Mac 执行纠错：完整端到端演示（10.83，v3）

v3 改进：向量化解码循环（相对 v2 的 3.9s/100k shots 纯 Python 循环）。
  - syndrome 计算：numpy 位运算批量（错误矩阵 × 稳定子矩阵 mod 2）
  - 查表：syndrome → 恢复的映射预构建为 numpy 索引数组（O(1) 向量化）
  - 逻辑翻转：X 分量与逻辑 Z 支撑的交按位与 + popcount

v2 改进（保留）：
  [A] 真正应用恢复操作并重算逻辑值——从"检测"升级为"纠错"验证
  [B] 码升级到 [[16,6,4]]（CSS(RM(1,4))，d=4）

模拟方式（码容量，诚实标注）：stim 的 detector_sampler 把显式 `X` 门当作
"预期操作"吸收进测量基（不触发探测器），故本 demo 不依赖 stim 探测器采样，
而用自洽的码容量模拟：随机注入 Pauli 错误 → syndrome（模拟量子硬件读出）
→ 查表恢复 → 应用恢复 → 重算逻辑值。

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

    返回 (rec_idx, in_table)：
      rec_idx[K] = syndrome 编码为整数 K 时恢复算符的索引（0 = 无恢复）
      in_table[K] = 该 syndrome 是否在恢复表（可识别）
    """
    ns = len(dec.gens)
    size = 1 << ns
    rec_idx = np.zeros(size, dtype=np.int64)
    in_table = np.zeros(size, dtype=bool)
    for synd, R in dec.table.items():
        k = sum(int(b) << i for i, b in enumerate(synd))
        # 恢复算符索引：把 Pauli 编码为单个整数（X 分量位掩码 + Z 分量位掩码）
        xmask = zmask = 0
        for i in range(n):
            if R.t[i] in (1, 3):
                xmask |= 1 << i
            if R.t[i] in (2, 3):
                zmask |= 1 << i
        rec_idx[k] = (xmask << n) | zmask
        in_table[k] = True
    return rec_idx, in_table


def popcount64(x):
    """向量化 popcount（numpy int64 数组）。"""
    x = x.astype(np.uint64)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def main():
    print("=" * 80)
    print("一台 Mac 执行纠错 v3：注入错误 → syndrome → 自研解码器 → 恢复 → 验证")
    print("=" * 80)
    m, r = 4, 1
    gens, lz = css_rm_code(m, r)
    n = 1 << m
    ns = len(gens)
    # 逻辑 Z 支撑位掩码（X 分量与 Lz 交 → 翻转）
    lz_xmask = 0
    for i in range(n):
        if lz.t[i] == 2:
            lz_xmask |= 1 << i
    lz_bits = np.array([1 if (lz_xmask >> i) & 1 else 0 for i in range(n)], dtype=np.uint64)
    print(f"码: CSS(RM(1,4)) = [[{n},{n - ns},{1 << (r + 1)}]] "
          f"(d=4, 逻辑 Z = x1x2 权重 {lz.weight()})")

    # 稳定子矩阵（向量化 syndrome）
    Sx = np.zeros((ns, n), dtype=np.int8)   # 稳定子的 X 分量
    Sz = np.zeros((ns, n), dtype=np.int8)   # 稳定子的 Z 分量
    for a, g in enumerate(gens):
        for i in range(n):
            if g.t[i] in (1, 3):
                Sx[a, i] = 1
            if g.t[i] in (2, 3):
                Sz[a, i] = 1

    for p in (0.001, 0.005, 0.01):
        shots = 100000
        pe = p / 3   # X/Z/Y 各 p/3 → 总错误率 p（与 v2 对齐可比）

        # 1. 恢复表 + 向量化映射
        t0 = time.time()
        dec = LookupDecoder(gens, n, name='[[16,6,4]]')
        dec.build(w_max=2)
        rec_idx, in_table = build_recovery_arrays(dec, n)
        t_build = (time.time() - t0) * 1000

        # 2. 向量化采样错误 + syndrome + 恢复 + 验证
        t0 = time.time()
        rng = np.random.default_rng(42)
        r = rng.random((shots, n))
        # 每比特: X/Z/Y 各 pe = p/3 概率（退极化，总错误率 p）
        tx = (r < pe).astype(np.int8)
        tz = ((r >= pe) & (r < 2 * pe)).astype(np.int8)
        ty = ((r >= 2 * pe) & (r < 3 * pe)).astype(np.int8)
        Ex = (tx | ty).astype(np.uint64)   # X 分量
        Ez = (tz | ty).astype(np.uint64)   # Z 分量

        # syndrome = (Ex @ Sz^T + Ez @ Sx^T) mod 2（X 错误 vs Z 稳定子 + Z 错误 vs X 稳定子）
        synd = ((Ex.astype(np.int64) @ Sz.T + Ez.astype(np.int64) @ Sx.T) & 1).astype(np.uint64)
        # 编码为整数索引（向量化查表）
        shifts = np.array([1 << i for i in range(ns)], dtype=np.uint64)
        idx = (synd.astype(np.uint64) * shifts).sum(axis=1).astype(np.int64)

        # 错误翻转逻辑 Z：E 的 X 分量与 Lz 支撑交奇数（逐位乘 + sum mod 2）
        flip_E = (Ex * lz_bits).sum(axis=1) & 1

        # 查表恢复（向量化）
        R_enc = rec_idx[idx]                    # 恢复算符编码（0 = 无）
        xmask_rec = (R_enc >> n).astype(np.uint64)
        # 恢复翻转逻辑 Z：R 的 X 分量与 Lz 交奇数
        flip_R = popcount64(xmask_rec & np.uint64(lz_xmask)) & 1
        # 纠错后逻辑值
        flip_corr = flip_E ^ flip_R
        resolved = in_table[idx] | (R_enc == 0)  # 命中表 或 无错误
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


if __name__ == "__main__":
    main()
