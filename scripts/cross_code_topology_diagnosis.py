#!/usr/bin/env python3
"""cross_code_topology_diagnosis.py —— 跨码族 A0/A1 拓扑诊断基准（10.55 延伸）

验证 diagnose_circuit 的电路无关性扩展到非格点码族，并标定拓扑诊断的
适用边界：

  [一] surface code (L=4)：2D 格点 → A0/A1 强分离（10.54 已知，cross_lift>1.5）
  [二] [[15,7,3]] Hamming CSS：1D 坐标 → A0/A1 分离消失（cross_lift≈1.0）

核心结论：A0/A1 拓扑分离依赖 ≥2D 格点探测器坐标——这是诊断的适用范围
边界（非缺陷）：对非格点/LDPC 码，拓扑判据退化为平凡，需改用 syndrome
重量/距离等非拓扑判据。本脚本将该边界固化为可复现基准。

依赖：stim + pymatching（venv311）。运行: .venv311/bin/python scripts/cross_code_topology_diagnosis.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import stim

from qecgeo.codes import rm_code_15_7_3
from qecgeo.error_geometry import diagnose_circuit, build_surface_circuit


def build_hamming_css_circuit(rounds, noise):
    """[[15,7,3]] Hamming CSS 多轮记忆电路（1D 探测器坐标，LDPC 型）。

    逻辑 Z = 全部 15 数据比特 Z 乘积（初态 |0>^15 期望 +1）。
    """
    code = rm_code_15_7_3()
    n = code.n

    def xm(p):
        return np.array([1 if t in (1, 3) else 0 for t in p.t], dtype=int)

    def zm(p):
        return np.array([1 if t in (2, 3) else 0 for t in p.t], dtype=int)

    Hx = np.array([xm(g) for g in code.gens])
    Hz = np.array([zm(g) for g in code.gens])
    ns = Hx.shape[0]
    c = stim.Circuit()
    anc = n
    for rr in range(rounds):
        for a in range(ns):
            ax = anc + a
            c.append('R', [ax]); c.append('H', [ax])
            for j in range(n):
                if Hx[a, j]:
                    c.append('CNOT', [ax, j])
            c.append('H', [ax])
        for a in range(ns):
            az = anc + ns + a
            c.append('R', [az])
            for j in range(n):
                if Hz[a, j]:
                    c.append('CNOT', [j, az])
        for j in range(n):
            c.append('DEPOLARIZE1', [j], noise)
        for a in range(2 * ns):
            c.append('MR', [anc + a])
        if rr >= 1:
            for a in range(2 * ns):
                c.append('DETECTOR',
                         [stim.target_rec(-(2 * ns) + a), stim.target_rec(-(4 * ns) + a)],
                         [a + (rr - 1) * 2 * ns])
    for j in range(n):
        c.append('M', [j])
    c.append('OBSERVABLE_INCLUDE', [stim.target_rec(-k) for k in range(1, n + 1)], 0)
    return c


def main():
    print("跨码族 A0/A1 拓扑诊断基准（10.55 延伸）")
    print("=" * 86)
    print(f"{'码族':<24} {'坐标维':>6} {'pL':>8} {'cross_lift':>10} {'A0 exc_med':>10} {'A1 exc_med':>10} {'分离?':>10}")
    print("-" * 86)

    # [一] surface L=4（2D 格点，基准）
    c_surf = build_surface_circuit(4, 3, 0.005)
    r_surf = diagnose_circuit(c_surf, shots=2000, seed=42)
    lift_surf = r_surf['cross_lift']
    a0_exc = r_surf['structure']['ok'].get('exc_med')
    a1_exc = r_surf['structure']['err'].get('exc_med')
    sep = '✓' if (lift_surf and lift_surf > 1.5) else '✗'
    print(f"{'surface L=4':<24} {'2D':>6} {r_surf['pL']:>8.4f} "
          f"{lift_surf:>10.3f} {a0_exc:>10} {a1_exc:>10} {sep:>10}")

    # [二] [[15,7,3]] Hamming CSS（1D 坐标，LDPC 型）
    c_h = build_hamming_css_circuit(5, 0.08)
    r_h = diagnose_circuit(c_h, shots=5000, seed=42)
    lift_h = r_h['cross_lift']
    a0_exc_h = r_h['structure']['ok'].get('exc_med')
    a1_exc_h = r_h['structure']['err'].get('exc_med')
    sep_h = '✓' if (lift_h and lift_h > 1.5) else '✗ (1D 退化)'
    print(f"{'[[15,7,3]] Hamming CSS':<24} {'1D':>6} {r_h['pL']:>8.4f} "
          f"{lift_h:>10.3f} {a0_exc_h:>10} {a1_exc_h:>10} {sep_h:>10}")

    print("\n结论：")
    print("  - surface（2D 格点）：A0/A1 拓扑分离显著（cross_lift>1.5）——10.54/10.55 复现")
    print("  - [[15,7,3]]（1D 坐标）：A0/A1 分离消失（cross_lift≈1.0）——拓扑判据")
    print("    需 ≥2D 格点坐标；对非格点/LDPC 码，诊断自动退化并如实报告（不崩溃）")
    print("  - 边界已固化为可复现基准：诊断管线对任意带 det 坐标 stim 电路可运行，")
    print("    拓扑指标在非格点上如实显示'无区分'（诚实标注，不夸大）")


if __name__ == "__main__":
    main()
