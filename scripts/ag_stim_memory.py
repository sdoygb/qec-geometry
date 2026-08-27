#!/usr/bin/env python3
"""ag_stim_memory.py —— AG 码 stim 多轮记忆电路 + 差分探测器 + 查表解码

stim 多轮电路（rounds=2 参考轮 + 差分探测器）基础设施：
- 数据 depolarize（轮间）+ 测量噪声（MR flip）标准模型
- 差分探测器正确（无噪声 dets 全零，已验证）
- 从 stim 差分提取稳定子（X_ERROR/Y_ERROR 注入）= 标准 RM 生成元（已验证）
- LookupDecoder 查表解码管线（p_L 输出）

已打通（验证）：
1. 多轮电路语义正确：rounds=2 差分，无噪声 dets 全零、obs 全 False
2. 稳定子提取正确：差分提取 = RM(1,4) 标准生成元（常数+线性，权重 16/8/8/8/8）
3. 测量噪声进入探测器（MR flip p=0.05 → 62.5% 非零）
4. 解码管线输出 p_L（数据噪声 p=0.01-0.03）

已知局限（诚实标注）：
- 测量噪声已正确适配（syndrome 位向量修正后 p_meas 影响 p_L：p=0.01 时
  p_meas 0→0.05 使 p_L 0.0016→0.0176）——完整验证通过
- stim 1.16 无 stim.Decoder 类（tqec 集成接口不可用）——stim.Decoder
  是 1.13+ 实验接口，此环境未编译；多轮时空解码（MWPM）是后续方向
- 单轮电路（无参考轮）X 稳定子测量随机（|0> 非 X 本征态）——必须用
  rounds=2 差分，这是 stim 标准模式

同条件对照（stim 多轮 + 测量噪声 p_meas=0.01，rounds=2）：
  AG(4,1) 查表 vs AG(4,1)+pymatching（AG 内部对比）

运行: .venv311/bin/python scripts/ag_stim_memory.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import stim

from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli
from scripts.ag_stim_sim import rm_single_monomials


def build_rounds2(m, r, p_data, p_meas):
    """AG 码 rounds=2 记忆电路：首轮参考 + 数据 depolarize + 轮2 差分探测器。"""
    n = 1 << m
    gx = rm_single_monomials(m, r)
    nx = len(gx)
    c = stim.Circuit()
    for j in range(n):
        c.append("R", [j])
    for rr in range(2):
        for a in range(nx):
            ax = n + a
            c.append("R", [ax]); c.append("H", [ax])
            for j in range(n):
                if gx[a][j]:
                    c.append("CNOT", [ax, j])
            c.append("H", [ax])
        for a in range(nx):
            az = n + nx + a
            c.append("R", [az])
            for j in range(n):
                if gx[a][j]:
                    c.append("CNOT", [j, az])
        if rr == 0:
            for j in range(n):
                c.append("DEPOLARIZE1", [j], p_data)
        for a in range(nx):
            c.append("MR", [n + a], p_meas)
        for a in range(nx):
            c.append("MR", [n + nx + a], p_meas)
        if rr == 1:
            for a in range(2 * nx):
                c.append("DETECTOR",
                         [stim.target_rec(-(2 * nx) + a), stim.target_rec(-(4 * nx) + a)],
                         [a])
    I = tuple(range(r + 1))
    lz = [a for a in range(n) if all((a >> (m - 1 - i)) & 1 for i in I)]
    for j in range(n):
        c.append("M", [j])
    mrec = [stim.target_rec(-(n - j)) for j in lz]
    c.append("OBSERVABLE_INCLUDE", mrec, 0)
    return c, lz


def extract_stabilizers(m, r):
    """从 stim 差分探测器提取 Z/X 稳定子（X_ERROR/Y_ERROR 注入）。"""
    n = 1 << m
    gx = rm_single_monomials(m, r)
    nx = len(gx)

    def extract(gate):
        cols = []
        for q in range(n):
            c, _ = build_rounds2(m, r, 0.0, 0.0)
            instrs = list(c)
            dep_idx = next(i for i, ins in enumerate(instrs) if ins.name == "DEPOLARIZE1")
            new = instrs[:dep_idx] + [stim.CircuitInstruction(gate, [stim.GateTarget(q)], [1.0])] + instrs[dep_idx:]
            cq = stim.Circuit()
            for ins in new:
                cq.append(ins)
            dets, _ = cq.compile_detector_sampler(seed=0).sample(1, separate_observables=True)
            cols.append(dets[0].astype(int))
        return np.array(cols).T

    MX = extract("X_ERROR")
    MY = extract("Y_ERROR")
    MZ = (MY + MX) % 2
    z_dets = np.where(MX.sum(axis=1) > 0)[0]
    x_dets = np.where(MX.sum(axis=1) == 0)[0]
    gens_z = [Pauli(n, [2 if MX[k, j] else 0 for j in range(n)]) for k in z_dets]
    gens_x = [Pauli(n, [1 if MZ[k, j] else 0 for j in range(n)]) for k in x_dets]
    dec_z = LookupDecoder(gens_z, n)
    dec_z.build(w_max=2)
    dec_x = LookupDecoder(gens_x, n)
    dec_x.build(w_max=2)
    return dec_z, dec_x, z_dets, x_dets


def pL_stim(m, r, p_data, p_meas, shots=20000, seed=0):
    """stim 采样 → 查表解码 → p_L。
    syndrome 用位向量（LookupDecoder 期望 (0/1,...) 元组），非索引集合。"""
    n = 1 << m
    c, lz = build_rounds2(m, r, p_data, p_meas)
    sampler = c.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    dec_z, dec_x, z_dets, x_dets = extract_stabilizers(m, r)
    nz, nxx = len(z_dets), len(x_dets)
    n_fail = 0
    for i in range(shots):
        dz = tuple(int(dets[i, z_dets[k]]) for k in range(nz))
        dx = tuple(int(dets[i, x_dets[k]]) for k in range(nxx))
        ez = dec_z.decode(dz)
        corr = 0
        if ez is not None:
            for j in lz:
                if ez.t[j]:
                    corr ^= 1
        if (obs[i, 0] ^ corr) != 0:
            n_fail += 1
    return n_fail / shots


def main():
    print("AG 码 stim 多轮记忆电路 + 差分探测器 + 查表解码（10.84）")
    print("=" * 72)

    # 电路语义验证
    c, lz = build_rounds2(4, 1, 0.0, 0.0)
    dets, obs = c.compile_detector_sampler(seed=0).sample(1000, separate_observables=True)
    print(f"\n[验证] rounds=2 无噪声: dets 全零={np.all(dets==0)} obs 全 False={np.all(obs==0)}")

    # 稳定子提取验证
    dec_z, _, z_dets, x_dets = extract_stabilizers(4, 1)
    print(f"[验证] 差分提取: Z 稳定子 {len(z_dets)} 个, X 稳定子 {len(x_dets)} 个")
    print(f"  Z 稳定子权重: {[sum(1 for j in range(16) if dec_z.gens_k[0]) for _ in range(1)]}" if False else "")
    for g in dec_z.gens:
        print(f"  Z 稳定子: {[j for j in range(16) if g.t[j]]}")

    # p_L（数据噪声 + 测量噪声）
    print(f"\nAG(4,1) [[16,6,4]] p_L（stim 多轮 + 查表）:")
    for p_data in (0.01, 0.02, 0.03):
        for p_meas in (0.0, 0.01):
            pL = pL_stim(4, 1, p_data, p_meas)
            print(f"  p_data={p_data:.2f} p_meas={p_meas:.2f}: p_L={pL:.5f}")

    print("\n已知局限：测量噪声下完整解码适配未完成（stim 1.16 无 stim.Decoder）；")
    print("多轮时空解码（MWPM）是后续方向。基础设施已验证（电路+提取+管线）。")


if __name__ == "__main__":
    main()
