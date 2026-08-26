#!/usr/bin/env python3
"""
cross_family_benchmark.py —— 几何论闭式：全码族统一基准（零电路零模拟）

延续 10.55 的"跨码基准"精神，但全部用组合闭式（10.28–10.36），一台普通
电脑秒级完成。统一对比三个维度的纠错实用性：

  [一] 码参数：n/k/d/编码率
  [二] 逻辑算符计数（定理 10.30.2.04 / 10.28）
  [三] 损失标度 loss(θ)=c_d·θ^d（定理 10.35.1.07）
  [四] 零损失边界 + 简并比例（10.31/10.33）

运行: python3 scripts/cross_family_benchmark.py [--theta 0.01]
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse

from qecgeo.closedform import (ag_params, pg_params, zero_loss_boundary,
                               loss_at_theta, degeneracy_ratio,
                               degeneracy_ratio_next,
                               logical_operator_count, pg_logical_count,
                               loss_exponent, detection_rate)


def main():
    ap = argparse.ArgumentParser(description="几何论闭式：全码族统一基准")
    ap.add_argument("--theta", type=float, default=0.01)
    args = ap.parse_args()
    theta = args.theta

    print(f"几何论闭式：全码族统一基准（零电路零模拟，θ={theta}）")
    print("=" * 88)

    print("\n[一] 码参数与逻辑算符计数")
    print(f"{'族':<4} {'码':<20} {'d':>3} {'率':>7} {'逻辑算符计数':>12}")
    print("-" * 52)
    for m in range(3, 9):
        p = pg_params(m)
        code = f"$[[{p['n']},{p['k']},{p['d']}]]$"
        print(f"PG {m:<2} {code:<18} {p['d']:>3} {p['rate']:>7.3f} "
              f"{pg_logical_count(m):>12}")
    for m in range(4, 11):
        for r in range(1, min(3, (m - 1) // 2) + 1):
            p = ag_params(m, r)
            if p is None:
                continue
            code = f"$[[{p['n']},{p['k']},{p['d']}]]$"
            print(f"AG {m:<2} {code:<18} {p['d']:>3} {p['rate']:>7.3f} "
                  f"{logical_operator_count(m, r):>12}")

    print("\n[二] 损失标度 loss(θ)=c_d·θ^d（AG 族）")
    print(f"{'码':<20} {'d':>3} {'零损':>5} {'loss(θ)':>16} {'P_r(m) 简并':>12}")
    print("-" * 62)
    for m in range(4, 11):
        for r in range(1, min(3, (m - 1) // 2) + 1):
            p = ag_params(m, r)
            if p is None:
                continue
            loss = loss_at_theta(p["c_d"], p["d"], theta)
            code = f"$[[{p['n']},{p['k']},{p['d']}]]$"
            print(f"{code:<20} {p['d']:>3} {zero_loss_boundary(p['d']):>5} "
                  f"{loss:>16.4g} {degeneracy_ratio(m, r):>12.6g}")

    print("\n[三] 关键对比：同规模码的实用性（含通用损失指数 10.31.1.05）")
    print(f"{'码':<20} {'族':>3} {'d':>3} {'率':>7} {'指数 2⌈d/2⌉':>10} {'loss(θ)':>14}")
    print("-" * 62)
    pairs = [
        ("$[[15,7,3]]$", "PG"), ("$[[16,6,4]]$", "AG"),
        ("$[[63,51,3]]$", "PG"), ("$[[64,50,4]]$", "AG"),
        ("$[[255,239,3]]$", "PG"), ("$[[256,238,4]]$", "AG"),
        ("$[[1023,1003,3]]$", "PG"), ("$[[1024,1002,4]]$", "AG"),
    ]
    for code, fam in pairs:
        import re as _re
        n_val = int(_re.search(r"\[\[(\d+)", code).group(1))
        if fam == "PG":
            m = {15: 4, 63: 6, 255: 8, 1023: 10}[n_val]
            p = pg_params(m)
            loss = float("nan")  # PG 无 c_d 闭式（系数），但指数有
            ls = f"—（系数无闭式，指数 {loss_exponent(p['d'])}）"
        else:
            m, r = {16: (4, 1), 64: (6, 1), 256: (8, 1), 1024: (10, 1)}[n_val]
            p = ag_params(m, r)
            loss = loss_at_theta(p["c_d"], p["d"], theta)
            ls = f"{loss:.4g}"
        print(f"{code:<20} {fam:>3} {p['d']:>3} {p['rate']:>7.3f} "
              f"{loss_exponent(p['d']):>10} {ls:>14}")

    print("\n[四] 检测率闭式（10.29 预言 2）：p_det(θ) = sin²(θ/2)")
    print(f"{'θ':>8} {'p_det':>12}")
    print("-" * 24)
    for th in (0.01, 0.05, 0.1, 0.2, 0.4):
        print(f"{th:>8.2f} {detection_rate(th):>12.6f}")

    print("\n结论：AG 族在距离/损失/零损失边界全面优于同规模 PG 族；"
          "PG 优势在构造最简（方向完备）。通用损失指数 2⌈d/2⌉（10.31.1.05）"
          "覆盖两族，检测率闭式 sin²(θ/2) 与码无关。全部组合闭式，零模拟。")


if __name__ == "__main__":
    main()
