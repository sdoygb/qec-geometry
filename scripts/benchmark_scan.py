"""跨码多噪声扫描：surface code vs color code 的 A0/A1 几何诊断区分度

用法（在装有 stim+pymatching+chromobius 的环境）：
    python scripts/benchmark_scan.py [--shots 10000] [--seed 42]

输出 benchmark 表：pL, cross_lift, total_dist_ratio, exc_ratio, cluster_ratio
"""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "/tmp/chromobius_src/src")  # clorco（若存在）

from qecgeo.error_geometry import (
    diagnose_circuit,
    build_surface_circuit,
    _global_extent,
)

NOISES = [0.005, 0.01, 0.02, 0.03]


def _safe_ratio(v1, v0):
    """A1/A0 比值；A0 为 0 时返回 None（不参与统计）"""
    if v0 is None or v1 is None or v0 == 0:
        return None
    return v1 / v0


def summarize(res):
    """从 diagnose_circuit 结果提取 benchmark 关键量"""
    out = {"pL": res["pL"], "cross_lift": res.get("cross_lift")}
    # 从 ratios 里找 total_dist / exc / cluster
    for r in res.get("ratios", []):
        f = r["field"]
        if f == "total_dist":
            out["total_dist"] = r["ratio"]
        elif f == "exc":
            out["exc"] = r["ratio"]
        elif f == "cluster":
            out["cluster"] = r["ratio"]
    # A0/A1 样本数
    st = res["structure"]
    out["nA0"] = st.get("ok", {}).get("n") if isinstance(st.get("ok"), dict) else None
    out["nA1"] = st.get("err", {}).get("n") if isinstance(st.get("err"), dict) else None
    return out


def scan_surface(shots, seed):
    print("\n=== SURFACE CODE (L=4, rounds=3) ===")
    print(f"{'noise':>6} {'pL':>8} {'cross':>7} {'total_d':>8} {'exc':>6} {'cluster':>8} {'nA0':>6} {'nA1':>6}")
    rows = []
    for noise in NOISES:
        circuit = build_surface_circuit(4, 3, noise)
        res = diagnose_circuit(circuit, shots=shots, with_edges=True, seed=seed)
        s = summarize(res)
        rows.append(s)
        print(f"{noise:>6.3f} {s['pL']:>8.5f} {s.get('cross_lift', float('nan')):>7.2f} "
              f"{str(s.get('total_dist', 'inf')):>8} {str(s.get('exc', 'inf')):>6} "
              f"{str(s.get('cluster', 'inf')):>8} {s.get('nA0', 0):>6} {s.get('nA1', 0):>6}")
    return rows


def scan_color(shots, seed):
    try:
        from clorco import color_code
        from clorco import _make_circuit_params as mcp
        import gen
    except ImportError:
        print("\n[skip] clorco 不可用，跳过 color code 扫描")
        return []

    constructions = color_code.make_named_color_code_constructions()
    print("\n=== COLOR CODE (diameter=3, rounds=3) ===")
    print(f"{'noise':>6} {'pL':>8} {'cross':>7} {'total_d':>8} {'exc':>6} {'cluster':>8} {'nA0':>6} {'nA1':>6}")
    rows = []
    for noise in NOISES:
        params = mcp.Params(
            style="phenom",
            rounds=3,
            diameter=3,
            noise_strength=noise,
            noise_model=gen.NoiseModel.uniform_depolarizing(noise),
            debug_out_dir=None,
            convert_to_cz=False,
            editable_extras={},
        )
        circuit = constructions["phenom_color_code"](params)
        res = diagnose_circuit(circuit, shots=shots, with_edges=True, seed=seed)
        s = summarize(res)
        rows.append(s)
        print(f"{noise:>6.3f} {s['pL']:>8.5f} {s.get('cross_lift', float('nan')):>7.2f} "
              f"{str(s.get('total_dist', 'inf')):>8} {str(s.get('exc', 'inf')):>6} "
              f"{str(s.get('cluster', 'inf')):>8} {s.get('nA0', 0):>6} {s.get('nA1', 0):>6}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    surf = scan_surface(args.shots, args.seed)
    color = scan_color(args.shots, args.seed)
    print(f"\n[耗时 {time.time()-t0:.1f}s]")

    # 汇总判断：cross_lift 是否在两个码上都稳定 > 1.5
    print("\n=== 判定 ===")
    for name, rows in [("surface", surf), ("color", color)]:
        if not rows:
            continue
        cls = [r.get("cross_lift") for r in rows if r.get("cross_lift") is not None]
        if cls:
            print(f"{name}: cross_lift 均值 {np.mean(cls):.2f}，"
                  f"全部 {'有区分度(>1.5)' if all(c > 1.5 for c in cls) else '部分未达 1.5'}")


if __name__ == "__main__":
    main()
