#!/usr/bin/env python3
"""
compare_families.py —— PG 完备码族 vs AG 完备码族：完整纠错参数对比
（几何论闭式，零电路零模拟；10.28/10.30/10.33/10.35）

展示两族互补性：
  - PG  [[2^m-1, 2^m-1-2m, 3]]：方向完备，d=3 锁死，距离小但构造最简
  - AG  [[2^m, n-2·dim RM(r,m), 2^{r+1}]]：距离可任意大，编码率可极高
外加简并比例闭式（10.33）：P_r(m) / P'_r(m)

运行: python3 scripts/compare_families.py
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from qecgeo.closedform import (ag_params, pg_params, zero_loss_boundary,
                               loss_at_theta, degeneracy_ratio,
                               degeneracy_ratio_next)


def main():
    theta = 0.01
    print("几何论闭式：PG vs AG 完备码族完整对比（零电路零模拟）")
    print("=" * 78)

    print("\n[一] PG 完备码族 [[2^m-1, 2^m-1-2m, 3]]（10.28）")
    print(f"{'m':>3} {'码':<18} {'率':>7} {'权重3逻辑':>10}")
    print("-" * 44)
    for m in range(3, 11):
        p = pg_params(m)
        print(f"{m:>3} $[[{p['n']},{p['k']},{p['d']}]]$".ljust(22)
              + f"{p['rate']:>7.3f} {p['w3_logical']:>10}")

    print("\n[二] AG 完备码族（10.30）损失闭式 loss(θ)=c_d·θ^d")
    print(f"{'码':<18} {'d':>3} {'率':>7} {'零损':>5} {'loss(θ=0.01)':>16}")
    print("-" * 55)
    for m in range(4, 11):
        for r in range(1, min(3, (m - 1) // 2) + 1):
            p = ag_params(m, r)
            if p is None:
                continue
            loss = loss_at_theta(p["c_d"], p["d"], theta)
            code = f"$[[{p['n']},{p['k']},{p['d']}]]$"
            print(code.ljust(18) + f"{p['d']:>3} {p['rate']:>7.3f} "
                  + f"{zero_loss_boundary(p['d']):>5} {loss:>16.4g}")

    print("\n[三] 简并比例闭式（10.33）")
    print(f"{'m':>3} {'r':>2} {'P_r(m) 全简并比例':>18} {'P\'_r(m) 次主阶':>16}")
    print("-" * 44)
    for m, r in [(6, 2), (7, 2), (8, 3), (10, 3)]:
        print(f"{m:>3} {r:>2} {degeneracy_ratio(m, r):>18.6g} "
              + f"{degeneracy_ratio_next(m, r):>16.6g}")

    print("\n结论：两族互补——PG 构造最简（d=3 锁死），AG 距离可任意大且"
          "编码率可极高（m=20 时 99.996%）；简并比例闭式给出全简并边界"
          "（r≤2 恒 1，r≥3 骤降）。全部由组合闭式直接计算。")


if __name__ == "__main__":
    main()
