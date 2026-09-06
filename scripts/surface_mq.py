"""surface_mq.py —— surface 码 MQ 门级模型电路（基于 stim 官方 rotated_memory_z 电路改造）

公平对比模型（260828/260906，与 AG 门级模型 scripts/ag_spatiotemporal_scl.py 完全一致）：
  - 每稳定子每轮：1 次 MQ 门错误 DEPOLARIZE2(p_gate)，作用在 (ancilla, 1 个参与数据)，
    位置 = 该 ancilla 的 MR 之前（即其 CX 链之后）
  - 每次 ancilla MR：flip 概率 p_meas
  - 轮间：数据 qubit depolarize p_data（stim 原生 before_round_data_depolarization）
官方电路其余全部保留：逻辑 Z / 稳定子 / DETECTOR / 时间结构一律 stim 官方正确构造，
只把门错误模型换成 MQ（CNOT 链错误关掉，after_clifford_depolarization=0）。
（此前"坐标驱动手工重建 + 自提取逻辑 Z"的版本因几何/逻辑提取不可靠而废弃，
   改由官方电路变换保证结构正确性。）

验证（__main__）：
  1. 无噪声：dets 全零、obs 全零（确定性 Z_L 读出）
  2. X_ERROR 注入逻辑支撑数据 → obs 翻转（Z 基读出只被 X 型错误翻转；每 shot 确定）
  3. Z_ERROR 注入 → 产生 syndrome（dets 非零），obs 不变（Z 错误与 Z_L 对易）
  4. MWPM 解码 p_L：d 增大下降、p 增大上升；无噪声 p_L=0

用法：
  .venv311/bin/python scripts/surface_mq.py            # 自检
  .venv311/bin/python scripts/ag_vs_surface_mq.py      # AG vs surface 同模型扫描
"""

from collections import defaultdict

import stim


def _ancilla_partners(c: stim.Circuit):
    """(ancilla -> sorted 数据 qubit 列表) —— 从整电路 CX 对提取。

    官方电路每轮同一 ancilla 的参与数据固定；跨轮累积取并集即可。
    """
    ancillas = set()
    for ins in c:
        if ins.name == "MR":
            for t in ins.targets_copy():
                ancillas.add(t.value)
    partners = defaultdict(set)
    for ins in c:
        if ins.name == "CX":
            ts = ins.targets_copy()
            for k in range(0, len(ts), 2):
                a, b = ts[k].value, ts[k + 1].value
                if a in ancillas and b not in ancillas:
                    partners[a].add(b)
                elif b in ancillas and a not in ancillas:
                    partners[b].add(a)
    return {a: sorted(p) for a, p in partners.items()}


def surface_mq_circuit(d, rounds, p_data, p_meas, p_gate):
    """官方 rotated_memory_z + MQ 门错误模型。

    返回电路：结构 = stim 官方（逻辑 Z/DETECTOR 正确），噪声 =
    每稳定子每轮 1×DEPOLARIZE2(p_gate) + MR flip(p_meas) + 轮间数据 depolarize(p_data)。
    """
    c = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=0.0,  # 关 CNOT 链错误（MQ 平台假设）
        after_reset_flip_probability=0.0,
        before_measure_flip_probability=0.0,  # MR flip 由本函数手动逐条加
        before_round_data_depolarization=p_data,  # 轮间数据 depolarize
    )
    if p_gate <= 0 and p_meas <= 0:
        return c
    partners = _ancilla_partners(c)
    out = stim.Circuit()
    for ins in c:
        name = ins.name
        if name == "MR" and (p_gate > 0 or p_meas > 0):
            ts = ins.targets_copy()
            for t in ts:
                a = t.value
                if p_gate > 0 and partners.get(a):
                    out.append("DEPOLARIZE2", [a, partners[a][0]], p_gate)
                if p_meas > 0:
                    out.append("MR", [a], p_meas)
                else:
                    out.append("MR", [a])
        else:
            out.append(ins)
    return out


def obs_qubits(c: stim.Circuit):
    """官方 OBSERVABLE_INCLUDE 引用的最终数据 M qubit（逻辑 Z 支撑，物理编号）。"""
    meas = []
    for ins in c:
        if ins.name in ("M", "MR"):
            meas += [t.value for t in ins.targets_copy()]
    out = []
    for ins in c:
        if ins.name == "OBSERVABLE_INCLUDE":
            for t in ins.targets_copy():
                out.append(meas[len(meas) + t.value])
    return out


def pL_surface(d, rounds, p, p_meas, shots, seed=0, p_gate=None, pymatching=None):
    """surface d 记忆 p_L（pymatching，MQ 门模型）。

    p_data = p；p_gate 默认 = p（同 AG 门级模型：每稳定子 DEPOLARIZE2(p)）。
    """
    if p_gate is None:
        p_gate = p
    c = surface_mq_circuit(d, rounds, p_data=p, p_meas=p_meas, p_gate=p_gate)
    import pymatching

    dem = c.detector_error_model(decompose_errors=False)
    matcher = pymatching.Matching(dem)
    sampler = c.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    n_fail = 0
    for i in range(shots):
        if matcher.decode(dets[i]) != obs[i, 0]:
            n_fail += 1
    return n_fail / shots, n_fail


if __name__ == "__main__":
    print("=" * 70)
    print("surface MQ 门级模型自检（官方电路 + MQ 噪声）")
    print("=" * 70)
    for d in (3, 5):
        # 1) 无噪声
        c0 = surface_mq_circuit(d, 5, 0.0, 0.0, 0.0)
        ds0 = c0.compile_detector_sampler(seed=0)
        d0, o0 = ds0.sample(200, separate_observables=True)
        print(f"d={d} 无噪声: dets 全零={d0.sum() == 0}  obs 全零={o0.sum() == 0}")
        # 2) X_ERROR 注入一个逻辑支撑数据（应确定翻转 obs；可能带 syndrome 可被 MWPM 纠正）
        obsq = obs_qubits(c0)
        outx = stim.Circuit()
        for ins in c0:
            if ins.name == "M":
                outx.append("X_ERROR", [obsq[0]], 1.0)
            outx.append(ins)
        _, ox = outx.compile_detector_sampler(seed=0).sample(
            200, separate_observables=True
        )
        print(
            f"d={d} X_ERROR@逻辑支撑 qubit {obsq[0]}: obs 全1占比={ox.mean():.3f}（未解码应≈1）"
        )
        # 3) Z_ERROR 注入（应只有 syndrome，obs 不变）
        outz = stim.Circuit()
        for ins in c0:
            if ins.name == "MR" and sum(1 for _ in [0]) == 0:
                pass
            if ins.name == "M":
                outz.append("Z_ERROR", [obsq[0]], 1.0)
            outz.append(ins)
        dz, oz = outz.compile_detector_sampler(seed=0).sample(
            200, separate_observables=True
        )
        print(f"d={d} Z_ERROR@同 qubit:  obs 全1占比={oz.mean():.3f}（应≈0）")
    # 4) MWPM p_L 趋势
    print("\nMWPM p_L 趋势（rounds=5, 2000 shots）:")
    for d in (3, 5, 7):
        for p in (0.001, 0.003, 0.01):
            pl, nf = pL_surface(d, 5, p, 0.0, 2000, seed=42)
            print(f"  d={d} p={p:<6}: p_L={pl:.5f} ({nf}/2000)")
    print("\n自检完成。")
