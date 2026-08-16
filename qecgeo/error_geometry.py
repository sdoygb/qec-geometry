"""error_geometry.py —— 解码错误模式的几何分类（A0/A1，10.54）

几何论视角：
  - A0 类（局域错误）：激发局域、可修正 → 平凡拓扑（Berry 相位 0）
  - A1 类（逻辑错误）：错误链跨边界非平凡 → 整体拓扑（Berry 相位 2π）

特征提取（对每个采样错误模式）：
  激发层特征：exc（激发数）/ min_pair（最近激发对距离）/ n_layers（时间层数）
              / diam（空间直径）/ bdry（边界距离）/ cluster（最大簇大小）
              / cross（穿越判据：错误链同时接触相对边界 → A1 本质特征）
  配对边特征（MWPM）：n_edges / bdry_edges / total_dist / max_dist
              / long_straight（最长内部边直度）/ long_rate

已知结果（10.54，surface code L=4）：配对层 total_dist A1/A0 ≈ 3.00×，
cross 穿越率 A1 远高于 A0 —— 配对层是 A0/A1 分类的最强区分层。

依赖：stim + pymatching（可选，仅 surface code 演示需要）；chromobius（可选，仅 color code 解码需要）。
"""
import numpy as np
from collections import defaultdict

# ---------- 1. 电路生成 + 解码 ----------

def build_surface_circuit(L, rounds, noise):
    """surface code 电路（stim 内置生成器），噪声 = 统一 depolarizing/翻转率"""
    import stim
    return stim.Circuit.generated(
        "surface_code:unrotated_memory_z",
        distance=L,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_round_data_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
    )


def get_detector_coords(circuit):
    """stim 1.15 返回 {det_idx: [x, y, t]}"""
    try:
        return circuit.get_detector_coordinates()
    except Exception:
        return None


def decode_surface(L, rounds, noise, shots, seed=None):
    """surface code 解码全流程：电路 → DEM → 采样 → MWPM 解码 → 逻辑错误标记

    返回 dict：circuit/dets/obs/preds/le/coords/matching/n_det/dt
    """
    import pymatching
    import time
    t0 = time.time()
    circuit = build_surface_circuit(L, rounds, noise)
    dem = circuit.detector_error_model(decompose_errors=False)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    preds = matching.decode_batch(dets)
    le = np.any(preds != obs, axis=1)
    coords = get_detector_coords(circuit)
    return dict(circuit=circuit, dets=dets, obs=obs, preds=preds, le=le,
                coords=coords, matching=matching, n_det=dets.shape[1],
                dt=time.time() - t0)


def _pack_lsb(dets):
    """unpacked (shots, n_det) uint8 → LSB-first bit-packed (shots, ceil(n/8))

    stim 的 bit_packed 约定（bit i → byte i//8 的 bit i%8），与 chromobius
    predict_obs_flips_from_dets_bit_packed 的输入格式一致（已验证）。
    """
    n = dets.shape[1]
    nb = (n + 7) // 8
    bp = np.zeros((dets.shape[0], nb), dtype=np.uint8)
    for b in range(nb):
        col = dets[:, b * 8:(b + 1) * 8].astype(np.uint8)
        w = (1 << np.arange(min(8, n - b * 8))).astype(np.uint8)
        bp[:, b] = (col * w).sum(axis=1).astype(np.uint8)
    return bp


def decode_circuit(circuit, shots, seed=None, decoder='pymatching'):
    """任意 stim 电路解码全流程：电路 → DEM → 采样 → 解码 → 逻辑错误标记

    与 decode_surface 相同，但接受任意 stim.Circuit（surface code / color code
    / 任意 stabilizer 电路），电路生成与解码解耦。decoder 可选：
      - 'pymatching'：MWPM（surface code 最优；对三色码 d≥7 结构性失效，
        匹配图含无边界连通分量，奇校验样本无法配对）
      - 'chromobius'：彩色匹配（color code 专用；无配对边输出 → matching=None，
        配对层特征（total_dist 等）不可用，激发层特征不受影响）
    返回字段同 decode_surface（chromobius 时 matching=None，加 decoder 字段）。
    """
    import time
    t0 = time.time()
    dem = circuit.detector_error_model(decompose_errors=False)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    if decoder == 'chromobius':
        import chromobius
        dec = chromobius.compile_decoder_for_dem(dem)
        preds = dec.predict_obs_flips_from_dets_bit_packed(_pack_lsb(dets))
        # preds 是 bit-packed（观测数 → ceil(n_obs/8) 字节），obs 是 unpacked；
        # 统一打包后比较（bit i = 观测 i 是否翻转）
        le = np.any(preds != _pack_lsb(obs), axis=1)
        matching = None
    else:
        import pymatching
        matching = pymatching.Matching.from_detector_error_model(dem)
        preds = matching.decode_batch(dets)
        le = np.any(preds != obs, axis=1)
    coords = get_detector_coords(circuit)
    return dict(circuit=circuit, dets=dets, obs=obs, preds=preds, le=le,
                coords=coords, matching=matching, n_det=dets.shape[1],
                decoder=decoder, dt=time.time() - t0)


# ---------- 2. 错误模式空间结构分析 ----------

def _cluster_sizes(ps):
    """ps: [(x,y),...] → 连通分量大小列表（曼哈顿距离 ≤ 2 连通）

    注：单比特错误产生相邻稳定子激发对，稳定子坐标曼哈顿距离 = 2
    （第一版 min_pair_med=2.0 的观测证实），故邻域半径取 2。
    """
    if not ps:
        return []
    n = len(ps)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if abs(ps[i][0] - ps[j][0]) + abs(ps[i][1] - ps[j][1]) <= 2:
                union(i, j)
    cnt = defaultdict(int)
    for i in range(n):
        cnt[find(i)] += 1
    return list(cnt.values())


def _global_extent(coords):
    """码的全局边界（任意坐标原点）：(xmin, ymin, xmax, ymax)

    surface code 稳定子坐标从 0 开始 → (0, 0, gx, gy)；
    color code 坐标从 1.0 开始 → (1.0, 1.0, 2.125, 2.0)。
    相对边界使穿越判据与码的绝对坐标原点无关（几何论 A1 拓扑判据）。
    """
    if coords is None:
        return 0, 0, 0, 0
    all_x = [c[0] for c in coords.values() if len(c) >= 3]
    all_y = [c[1] for c in coords.values() if len(c) >= 3]
    if not all_x:
        return 0, 0, 0, 0
    return min(all_x), min(all_y), max(all_x), max(all_y)


def analyze_error_structure(dets, coords, le):
    """错误模式空间结构分析（几何论 A0/A1 视角）

    返回 dict(ok=汇总A0, err=汇总A1)，每侧含 med/q90 统计 + cross_rate
    """
    n = len(dets)
    exc_counts = dets.sum(axis=1)
    x0, y0, gx, gy = _global_extent(coords)
    per_sample = []
    for s in range(n):
        exc_idx = np.where(dets[s])[0]
        base = dict(exc=int(exc_counts[s]), min_pair=None, n_layers=None,
                    diam=None, bdry=None, cluster=None, cross=None)
        if coords is None or len(exc_idx) == 0:
            per_sample.append(base)
            continue
        # 每激发点 (x, y, t)
        pts = []
        for d in exc_idx:
            c = coords.get(d)
            if c is None or len(c) < 3:
                continue
            pts.append((int(c[0]), int(c[1]), int(c[2])))
        if not pts:
            per_sample.append(base)
            continue
        # 按时间层分组
        layers = defaultdict(list)
        for x, y, t in pts:
            layers[t].append((x, y))
        # 最近激发对（同层内平面曼哈顿距离）
        min_pair = None
        diam = 0.0
        for t, ps in layers.items():
            for i in range(len(ps)):
                for j in range(i + 1, len(ps)):
                    dist = abs(ps[i][0] - ps[j][0]) + abs(ps[i][1] - ps[j][1])
                    if min_pair is None or dist < min_pair:
                        min_pair = dist
                    if dist > diam:
                        diam = dist
        # 边界距离（激发点到码边界的最小距离，相对坐标）
        bdry = min(min(x - x0, gx - x, y - y0, gy - y) for x, y, _ in pts)
        # 穿越判据（A1 拓扑本质：错误链同时接触相对边界）
        xs_all = [p[0] for p in pts]
        ys_all = [p[1] for p in pts]
        width_x = (gx - x0) if gx > x0 else 0.0
        width_y = (gy - y0) if gy > y0 else 0.0
        cross = int((max(xs_all) - min(xs_all) >= width_x - 1e-6) or
                    (max(ys_all) - min(ys_all) >= width_y - 1e-6))
        # 每层最大簇
        max_cluster = 0
        for t, ps in layers.items():
            sizes = _cluster_sizes(ps)
            if sizes:
                max_cluster = max(max_cluster, max(sizes))
        per_sample.append(dict(exc=int(exc_counts[s]), min_pair=min_pair,
                               n_layers=len(layers), diam=diam, bdry=bdry,
                               cluster=max_cluster, cross=cross))
    # 汇总（按逻辑错误分组）
    grp_ok, grp_err = [], []
    for s in range(n):
        (grp_err if le[s] else grp_ok).append(per_sample[s])

    def summ(grp):
        if not grp:
            return {}
        f = {}
        for field in ("exc", "min_pair", "n_layers", "diam", "bdry", "cluster"):
            v = [g[field] for g in grp if g[field] is not None]
            f[field + "_med"] = float(np.median(v)) if v else None
            f[field + "_q90"] = float(np.percentile(v, 90)) if v else None
        cr = [g["cross"] for g in grp if g.get("cross") is not None]
        f["cross_rate"] = float(np.mean(cr)) if cr else None
        return dict(n=len(grp), **f)

    return dict(ok=summ(grp_ok), err=summ(grp_err))


def analyze_edges(dets, matching, coords, le, max_ok_samples=None):
    """MWPM 配对边纯链分析（A1 非平凡拓扑链应在配对层显现）

    decode_to_edges_array 返回配对节点对（无权重列）→ 用 detector 坐标
    计算每条配对边的空间距离作为链长代理。A1 逻辑链跨越整个码 →
    应有更长距离的配对边、更多边界连接。

    容错：三色码（d>=7）的无边界连通分量上 MWPM 无法配对（奇校验）时
    捕获 ValueError，该样本标记 unpaired 并单独计数，不污染配对统计。
    max_ok_samples：A0 侧抽样上限（大样本时 A0 中位数统计无需全量）。
    """
    n = len(dets)
    nd = matching.num_detectors
    # 色码格坐标：x 方向 0.125/0.875 交替（六边形格内色分裂），映射为整数格
    coords = {k: (round(v[0]), round(v[1]), *v[2:]) for k, v in coords.items()}
    x0, y0, gx, gy = _global_extent(coords)

    def edge_dist(e):
        a, b = int(e[0]), int(e[1])
        if a >= nd and b >= nd:
            return 0.0
        if a >= nd:  # 边界-探测器：探测器到最近边界的距离（相对坐标）
            c = coords.get(b)
            if c is None or len(c) < 3:
                return 0.0
            return min(c[0] - x0, gx - c[0], c[1] - y0, gy - c[1])
        if b >= nd:
            c = coords.get(a)
            if c is None or len(c) < 3:
                return 0.0
            return min(c[0] - x0, gx - c[0], c[1] - y0, gy - c[1])
        ca, cb = coords.get(a), coords.get(b)
        if ca is None or cb is None or len(ca) < 3 or len(cb) < 3:
            return 0.0
        return abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])  # 空间曼哈顿距离

    per = [None] * n
    n_unpaired = 0
    n_ok_done = 0
    for s in range(n):
        if not le[s] and max_ok_samples is not None and n_ok_done >= max_ok_samples:
            continue  # A0 抽样：跳过后续 A0 样本
        try:
            edges = matching.decode_to_edges_array(dets[s])
        except ValueError:
            # 无边界连通分量奇校验：MWPM 不适用（chromobius 可解）
            n_unpaired += 1
            per[s] = dict(n_edges=-1, bdry_edges=-1, total_dist=-1.0, max_dist=-1.0,
                          long_straight=None, long_rate=0.0, unpaired=True)
            continue
        if not le[s]:
            n_ok_done += 1
        if len(edges) == 0:
            per[s] = dict(n_edges=0, bdry_edges=0, total_dist=0.0, max_dist=0.0,
                          long_straight=None, long_rate=0.0)
            continue
        bdry = int(np.sum((edges[:, 0] >= nd) | (edges[:, 1] >= nd)))
        dists = [edge_dist(e) for e in edges]
        mx = float(max(dists))
        # 最长内部 detector-detector 边的直度（A1 穿越链应更"直"）
        best = None
        best_dist = -1.0
        for e in edges:
            a, b = int(e[0]), int(e[1])
            if a >= nd or b >= nd:
                continue
            ca, cb = coords.get(a), coords.get(b)
            if ca is None or cb is None or len(ca) < 3 or len(cb) < 3:
                continue
            d = abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])
            if d > best_dist:
                best_dist = d
                best = (d, abs(ca[0] - cb[0]), abs(ca[1] - cb[1]))
        straight = (max(best[1], best[2]) / best[0]) if best is not None and best[0] > 0 else None
        per[s] = dict(n_edges=int(len(edges)), bdry_edges=bdry,
                      total_dist=float(sum(dists)), max_dist=mx,
                      long_straight=straight, long_rate=float(mx >= 4.0))
    grp_ok, grp_err = [], []
    for s in range(n):
        if per[s] is None:
            continue  # A0 抽样跳过的样本
        (grp_err if le[s] else grp_ok).append(per[s])

    def summ(grp):
        valid = [g for g in grp if g.get("total_dist", -1.0) >= 0.0]
        if not valid:
            return {}
        f = {}
        for field in ("n_edges", "bdry_edges", "total_dist", "max_dist", "long_rate"):
            v = [g[field] for g in valid]
            f[field + "_med"] = float(np.median(v))
            f[field + "_q90"] = float(np.percentile(v, 90))
        ls = [g["long_straight"] for g in valid if g["long_straight"] is not None]
        f["long_straight_med"] = float(np.median(ls)) if ls else None
        return dict(n=len(valid), **f)

    return dict(ok=summ(grp_ok), err=summ(grp_err), unpaired=n_unpaired)


def edge_maxdist_distribution(matching, dets, coords, le):
    """每样本配对边最大距离分布（按 A0/A1 分组）——显式长链型统计

    10.54 §6.3：显式长链型（max_dist >= 4）比例是否随码距 L 增大？
    返回 dict(ok=[maxd...], err=[maxd...], n_ok, n_err)
    """
    nd = matching.num_detectors
    coords = {k: (round(v[0]), round(v[1]), *v[2:]) for k, v in coords.items()}
    x0, y0, gx, gy = _global_extent(coords)

    def edge_dist(e):
        a, b = int(e[0]), int(e[1])
        if a >= nd and b >= nd:
            return 0.0
        if a >= nd:
            c = coords.get(b)
            if c is None or len(c) < 3:
                return 0.0
            return min(c[0] - x0, gx - c[0], c[1] - y0, gy - c[1])
        if b >= nd:
            c = coords.get(a)
            if c is None or len(c) < 3:
                return 0.0
            return min(c[0] - x0, gx - c[0], c[1] - y0, gy - c[1])
        ca, cb = coords.get(a), coords.get(b)
        if ca is None or cb is None or len(ca) < 3 or len(cb) < 3:
            return 0.0
        return abs(ca[0] - cb[0]) + abs(ca[1] - cb[1])

    n = len(dets)
    ok, err = [], []
    for s in range(n):
        edges = matching.decode_to_edges_array(dets[s])
        mx = max((edge_dist(e) for e in edges), default=0.0)
        (err if le[s] else ok).append(float(mx))
    return dict(ok=ok, err=err, n_ok=len(ok), n_err=len(err))


def _ratio_line(field, ok, err, threshold=1.5):
    """A1/A0 比值判定（有区分度阈值 1.5 / 0.67）"""
    m0, m1 = ok.get(field + "_med"), err.get(field + "_med")
    if m0 is None or m1 is None:
        return None
    if m0 == 0 and m1 == 0:
        return dict(field=field, ratio=None, direction="=", verdict="无区分度（均为 0）")
    ratio = m1 / m0 if m0 > 0 else float("inf")
    arrow = "↑" if m1 > m0 else ("↓" if m1 < m0 else "=")
    verdict = "有区分度" if ratio > threshold or ratio < 1 / threshold else "无区分度"
    return dict(field=field, ratio=float(ratio), direction=arrow, verdict=verdict)


def diagnose_circuit(circuit, shots, with_edges=True, seed=None, decoder='pymatching'):
    """电路无关的诊断入口：任意 stim 电路 → A0/A1 几何分析（10.54 方法）

    surface code / color code 通用。输入任意 stim.Circuit（带探测器坐标），
    输出与 run_error_geometry 相同的诊断结构。decoder：
      - 'pymatching'：MWPM 配对（surface code；with_edges=True 报告配对层）
      - 'chromobius'：色码专用彩色匹配（无配对边 → with_edges 自动关闭，
        只报告激发层特征；色码必须用它，见 analyze_edges 注释）
    返回 dict：pL, structure(A0/A1 激发层特征), edges(配对边特征, chromobius
    时为 None), ratios(区分度判定), cross_lift(穿越率提升), n_det, n_qubits,
    decoder。
    """
    res = decode_circuit(circuit, shots, seed=seed, decoder=decoder)
    st = analyze_error_structure(res["dets"], res["coords"], res["le"])
    out = dict(pL=float(res["le"].mean()), structure=st, edges=None,
               ratios=[], cross_lift=None, n_det=res["n_det"],
               n_qubits=res["circuit"].num_qubits, decoder=decoder)
    # 穿越率提升（A1 本质特征）
    c0 = st["ok"].get("cross_rate")
    c1 = st["err"].get("cross_rate")
    if c0 is not None and c1 is not None and c0 > 0:
        out["cross_lift"] = float(c1 / c0)
    # 激发层区分度
    for field in ("exc", "min_pair", "n_layers", "diam", "bdry", "cluster"):
        r = _ratio_line(field, st["ok"], st["err"])
        if r is not None:
            out["ratios"].append(r)
    if with_edges and res["matching"] is not None:
        st2 = analyze_edges(res["dets"], res["matching"], res["coords"], res["le"])
        out["edges"] = st2
        for field in ("n_edges", "bdry_edges", "total_dist", "max_dist", "long_rate"):
            r = _ratio_line(field, st2["ok"], st2["err"])
            if r is not None:
                out["ratios"].append(r)
        # 配对层直度
        ls0, ls1 = st2["ok"].get("long_straight_med"), st2["err"].get("long_straight_med")
        if ls0 is not None and ls1 is not None and ls0 > 0:
            out["ratios"].append(dict(field="long_straight",
                                      ratio=float(ls1 / ls0),
                                      direction="↑" if ls1 > ls0 else "↓",
                                      verdict="有区分度" if ls1 / ls0 > 1.5 or ls1 / ls0 < 0.67 else "无区分度"))
    return out


def run_error_geometry(L=4, rounds=3, noise=0.03, shots=20000, with_edges=True, seed=None):
    """surface code 便捷入口：错误模式几何分析（10.54 复现）"""
    circuit = build_surface_circuit(L, rounds, noise)
    return diagnose_circuit(circuit, shots, with_edges=with_edges, seed=seed)
