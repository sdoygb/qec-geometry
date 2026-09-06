#!/usr/bin/env python3
"""ag_vs_surface_mq.py —— AG 码 vs surface 码：同一 MQ 门级噪声模型下的 p_L 扫描

模型（两侧逐条一致）：
  - 每稳定子每轮 1× DEPOLARIZE2(p_gate) on (ancilla, 1 参与数据)  （MQ 门成本）
  - 每次 ancilla MR flip p_meas
  - 轮间数据 depolarize p_data
  AG 侧：scripts/ag_spatiotemporal_scl.py build_memory_circuit + SCL 解码
  surface 侧：scripts/surface_mq.py（stim 官方电路 + MQ 噪声）+ pymatching (MWPM)

解码器 = 各自原生解码器（AG: 矩/SCL；surface: MWPM）。注意码参数差异
（n/k/d/稳定子权重）在表中并列标注，不做超越性结论。

用法: .venv311/bin/python scripts/ag_vs_surface_mq.py [--shots 2000] [--rounds 5]
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ag_spatiotemporal_scl import run_pL
from scripts.surface_mq import pL_surface

CODE_PARAMS = {
    "AG(6,2) [[64,20,8]]": {
        "fn": lambda p, q, s, rnd: run_pL(
            6,
            2,
            rounds=rnd,
            p_data=p,
            p_meas=q,
            shots=s,
            seed=42,
            use_scl=True,
            p_gate=p,
        ),
        "n": 64,
        "k": 20,
        "d": 8,
    },
    "AG(4,1) [[16,6,4]]": {
        "fn": lambda p, q, s, rnd: run_pL(
            4,
            1,
            rounds=rnd,
            p_data=p,
            p_meas=q,
            shots=s,
            seed=42,
            use_scl=True,
            p_gate=p,
        ),
        "n": 16,
        "k": 6,
        "d": 4,
    },
    "surface d=3 [[17,1,3]]": {
        "fn": lambda p, q, s, rnd: pL_surface(3, rnd, p, q, s, seed=42),
        "n": 17,
        "k": 1,
        "d": 3,
    },
    "surface d=5 [[41,1,5]]": {
        "fn": lambda p, q, s, rnd: pL_surface(5, rnd, p, q, s, seed=42),
        "n": 41,
        "k": 1,
        "d": 5,
    },
    "surface d=7 [[97,1,7]]": {
        "fn": lambda p, q, s, rnd: pL_surface(7, rnd, p, q, s, seed=42),
        "n": 97,
        "k": 1,
        "d": 7,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--p-list", default="0.001,0.003,0.01")
    args = ap.parse_args()
    p_list = [float(x) for x in args.p_list.split(",")]
    shots, rounds = args.shots, args.rounds
    out_csv = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        f"ag_vs_surface_mq_p{rounds}r.csv",
    )
    rows = []
    t_start = time.time()
    print(
        f"AG vs surface MQ 同模型扫描  rounds={rounds} shots={shots} "
        f"p∈{p_list} p_meas∈{{0,p}}  每格 p_gate=p_data=p"
    )
    print("=" * 90)
    for name, prm in CODE_PARAMS.items():
        for p in p_list:
            for q in (0.0, p):
                t0 = time.time()
                pL, nf = prm["fn"](p, q, shots, rounds)
                rows.append(
                    {
                        "code": name,
                        "n": prm["n"],
                        "k": prm["k"],
                        "d": prm["d"],
                        "rate": round(prm["k"] / prm["n"], 4),
                        "p": p,
                        "p_meas": q,
                        "p_gate": p,
                        "p_data": p,
                        "rounds": rounds,
                        "shots": shots,
                        "errors": nf,
                        "p_L": round(pL, 6),
                    }
                )
                print(
                    f"  {name:<24} p={p:<6} p_meas={q:<6} p_L={pL:.5f} "
                    f"({nf}/{shots})  {time.time() - t0:.0f}s"
                )
        print("-" * 90)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"总耗时 {time.time() - t_start:.0f}s → 数据已存 {out_csv}")
    print("\n对照表（同 p，p_meas=0）:")
    print(
        f"  {'code':<24} {'rate':>6} {'d':>3} | " + " | ".join(f"p={p}" for p in p_list)
    )
    for name, prm in CODE_PARAMS.items():
        cell = [
            next(
                (
                    r["p_L"]
                    for r in rows
                    if r["code"] == name and r["p"] == p and r["p_meas"] == 0
                ),
                None,
            )
            for p in p_list
        ]
        print(
            f"  {name:<24} {prm['k'] / prm['n']:>6.3f} {prm['d']:>3} | "
            + " | ".join(f"{c:.5f}" if c is not None else "  n/a " for c in cell)
        )


if __name__ == "__main__":
    main()
