#!/usr/bin/env python3
"""ag_stim_sim.py —— AG 完备码 stim 物理噪声模拟（10.84 桥接验证）

目标：用 stim 标准流程（depolarizing noise）模拟 AG 完备码 CSS(RM(r,m))，
得到 p_L vs p 曲线，验证零简并理论在物理噪声下的恢复优势（对照 surface code）。

设计：
- 单轮 CSS 记忆电路：数据 |0> + 稳定子测量（X 稳定子 ancilla H-CNOT-H + Z 稳定子
  ancilla CNOT）+ 数据 depolarize + 辅助 MR + 差分探测器（纯空间 syndrome）。
  纯空间 syndrome 匹配 LookupDecoder 的静态查表（零简并理论适用域）。
- 逻辑 Z observable：次数 r+1 单项式 x_1...x_{r+1} 的评估向量
  （正交于全部稳定子，权重 2^{m-r-1}；AG(6,2) 时 = 8 = d）。
- 解码：stim 探测器翻转 → LookupDecoder 查表恢复 → 修正逻辑值 → p_L。

运行: python3 scripts/ag_stim_sim.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from itertools import combinations

import stim

from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli


def rm_single_monomials(m, max_deg):
    """全部次数 ≤ max_deg 单项式的评估向量（AG(m,2) 全部 2^m 点）。"""
    n = 1 << m
    gens = []
    for deg in range(0, max_deg + 1):
        for I in combinations(range(m), deg):
            g = np.zeros(n, dtype=int)
            for a in range(n):
                val = 1
                for i in I:
                    val &= (a >> (m - 1 - i)) & 1
                g[a] = val
            gens.append(g)
    return gens


def logical_z_vector(m, r):
    """逻辑 Z = 次数 r+1 单项式 x_1...x_{r+1} 的评估向量（权重 2^{m-r-1}）。
    正交于全部次数 ≤ r 单项式（RM(r,m)），故为合法逻辑算符。"""
    n = 1 << m
    I = tuple(range(r + 1))
    v = np.zeros(n, dtype=int)
    for a in range(n):
        val = 1
        for i in I:
            val &= (a >> (m - 1 - i)) & 1
        v[a] = val
    return v


def build_ag_circuit(m, r, p, seed=0):
    """AG 码 CSS(RM(r,m)) 单轮记忆电路（X 错误由 Z 稳定子检测，反之亦然）。"""
    n = 1 << m
    gx = rm_single_monomials(m, r)      # X 稳定子（次数 ≤ r）
    gz = rm_single_monomials(m, r)      # Z 稳定子（同支撑，CSS(H,H)）
    nx = len(gx)
    c = stim.Circuit()
    anc = n
    # 数据初始化
    for j in range(n):
        c.append("R", [j])
    # 数据 depolarize（物理噪声，在稳定子测量之前——X 错误只被 Z 稳定子检测）
    for j in range(n):
        c.append("DEPOLARIZE1", [j], p)
    # X 稳定子测量（ancilla H-CNOT-H）
    for a in range(nx):
        ax = anc + a
        c.append("R", [ax])
        c.append("H", [ax])
        for j in range(n):
            if gx[a][j]:
                c.append("CNOT", [ax, j])
        c.append("H", [ax])
    # Z 稳定子测量
    for a in range(nx):
        az = anc + nx + a
        c.append("R", [az])
        for j in range(n):
            if gz[a][j]:
                c.append("CNOT", [j, az])
    # 测量辅助（稳定子）
    for a in range(nx):
        c.append("MR", [anc + a])
        c.append("DETECTOR", [stim.target_rec(-1)], [a])
    for a in range(nx):
        c.append("MR", [anc + nx + a])
        c.append("DETECTOR", [stim.target_rec(-1)], [nx + a])
    # 数据测量（逻辑 Z observable）
    lz = logical_z_vector(m, r)
    mrec = []
    for j in range(n):
        c.append("M", [j])
        mrec.append(stim.target_rec(-(n - j)))
    c.append("OBSERVABLE_INCLUDE", mrec, 0)
    # 逻辑 Z 值 = Σ lz[j] * M_j mod 2（用 X_ERROR 探测时用 lz 系数）
    return c, lz


def decode_with_lookup(dets, gens_z, gens_x, dec_z, dec_x, n):
    """stim 探测器 → LookupDecoder 恢复。
    探测器布局：前 nx = X 稳定子（检测 Z 错误），后 nx = Z 稳定子（检测 X 错误）。
    dec_z 用 Z 稳定子 syndrome（X 错误）→ 对应 dets 的后 nx 位；
    dec_x 用 X 稳定子 syndrome（Z 错误）→ 对应 dets 的前 nx 位。
    返回 (rec_z, rec_x)：X/Z 恢复的比特掩码。"""
    nx = len(gens_z)
    rec_z = np.zeros(dets.shape[0], dtype=np.int64)   # X 错误恢复位（数据比特）
    rec_x = np.zeros(dets.shape[0], dtype=np.int64)   # Z 错误恢复位
    for i in range(dets.shape[0]):
        sx = tuple(np.where(dets[i, :nx])[0])        # X 稳定子 → Z 错误
        sz = tuple(np.where(dets[i, nx:])[0])        # Z 稳定子 → X 错误
        ez = dec_z.decode(sz)   # X 错误（Z 稳定子检测）→ 恢复用 X
        ex = dec_x.decode(sx)   # Z 错误（X 稳定子检测）→ 恢复用 Z
        if ez is not None and ez.t is not None:
            for j in range(n):
                if ez.t[j]:
                    rec_z[i] ^= 1 << j
        if ex is not None and ex.t is not None:
            for j in range(n):
                if ex.t[j]:
                    rec_x[i] ^= 1 << j
    return rec_z, rec_x


def main():
    print("AG 完备码 stim 物理噪声模拟（10.84 桥接验证）")
    print("=" * 72)

    for (m, r, name) in [(4, 1, "AG(4,1) [[16,6,4]]"), (5, 1, "AG(5,1) [[32,20,4]]"),
                         (6, 2, "AG(6,2) [[64,20,8]]")]:
        n = 1 << m
        gx = rm_single_monomials(m, r)
        # 构建 LookupDecoder（X 侧：Z 稳定子检测 X 错误）
        gens_z = [Pauli(n, [2 if x else 0 for x in g]) for g in gx]
        gens_x = [Pauli(n, [1 if x else 0 for x in g]) for g in gx]
        dec_z = LookupDecoder(gens_z, n, name=f"{name} Z")
        dec_z.build(w_max=2)
        dec_x = LookupDecoder(gens_x, n, name=f"{name} X")
        dec_x.build(w_max=2)
        lz = logical_z_vector(m, r)

        print(f"\n{name}: n={n} | Z 稳定子 {len(gx)} 个 | 逻辑 Z 权重 {lz.sum()}")
        for p in (0.01, 0.02, 0.03, 0.05):
            c, _ = build_ag_circuit(m, r, p)
            sampler = c.compile_detector_sampler(seed=0)
            dets, obs = sampler.sample(20000, separate_observables=True)
            rec_z, rec_x = decode_with_lookup(dets, gx, gx, dec_z, dec_x, n)
            # 恢复后逻辑值：obs ⊕ Σ lz[j]*X_j(rec_z)（X 恢复翻转 Z 逻辑）
            # 简化：只算 X 恢复对逻辑 Z 的影响（Z 恢复不影响 Z 逻辑）
            n_fail = 0
            for i in range(dets.shape[0]):
                # 恢复前逻辑值
                log = obs[i, 0]
                # X 恢复（rec_z）翻转逻辑 Z（因为 X_j 与 Z 逻辑反交换若 lz[j]=1）
                corr = bin(rec_z[i] & int("".join(str(x) for x in lz), 2)).count("1") % 2
                if (log ^ corr) != 0:
                    n_fail += 1
            pL = n_fail / dets.shape[0]
            print(f"  p={p:.2f}: p_L = {pL:.5f}")


if __name__ == "__main__":
    main()
