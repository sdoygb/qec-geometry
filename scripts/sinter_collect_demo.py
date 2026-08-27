#!/usr/bin/env python3
"""sinter_collect_demo.py —— sinter.collect 集成：AG 零简并查表 vs surface MWPM

用 sinter 生态的标准流程（sinter.collect）跑同条件对比：
- AG(4,1) [[16,6,4]]：自定义解码器 LookupSinterDecoder（几何论查表，零简并）
- surface d=3：pymatching（工业标准 MWPM）
- 同一噪声模型：rounds=2 差分 + 数据 depolarize + 测量翻转（before_measure）
- 输出：各噪声下的 p_L 曲线

这证明几何论解码器可通过 sinter.Decoder 接口无缝融入 stim 生态
（含 tqec 若其基于 sinter）。

运行: .venv311/bin/python scripts/sinter_collect_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sinter
import stim

from scripts.ag_stim_memory import build_rounds2, extract_stabilizers
from scripts.sinter_lookup_decoder import LookupSinterDecoder


def ag_tasks(p_data, p_meas, shots):
    """AG(4,1) sinter 任务：自定义解码器 'lookup'。"""
    m, r = 4, 1
    n = 1 << m
    c, _ = build_rounds2(m, r, p_data, p_meas)
    dec_z, dec_x, z_dets, x_dets = extract_stabilizers(m, r)
    I = tuple(range(r + 1))
    lz = [a for a in range(n) if all((a >> (m - 1 - i)) & 1 for i in I)]
    dec = LookupSinterDecoder(dec_z=dec_z, dec_x=dec_x,
                              z_det_indices=list(z_dets), x_det_indices=list(x_dets),
                              obs_map_z=[lz], obs_map_x=[[]])
    return sinter.Task(
        circuit=c,
        json_metadata={'code': 'AG(4,1)', 'p_data': p_data, 'p_meas': p_meas},
    ), {'lookup': dec}


def surface_task(d, p_data, p_meas, shots):
    """surface d=3 sinter 任务：pymatching 解码器。"""
    c = stim.Circuit.generated('surface_code:rotated_memory_z',
        distance=d, rounds=2,
        after_clifford_depolarization=0.0,
        before_round_data_depolarization=p_data,
        after_reset_flip_probability=p_meas,
        before_measure_flip_probability=p_meas)
    return sinter.Task(
        circuit=c,
        json_metadata={'code': f'surface d={d}', 'p_data': p_data, 'p_meas': p_meas},
    )


def main():
    print("sinter.collect 集成：AG 零简并查表 vs surface MWPM（10.84）")
    print("=" * 74)

    # AG(4,1) 任务（自定义解码器 lookup）
    ag_tasks_list = []
    ag_decs = {}
    for p in (0.01, 0.02, 0.03):
        task, decs = ag_tasks(p, 0.01, 5000)
        ag_tasks_list.append(task)
        ag_decs['lookup'] = decs['lookup']

    # surface 任务（pymatching）
    sf_tasks = [surface_task(3, p, 0.01, 5000) for p in (0.01, 0.02, 0.03)]

    # 运行 sinter.collect
    all_tasks = ag_tasks_list + sf_tasks
    stats = sinter.collect(
        num_workers=2,
        tasks=all_tasks,
        max_shots=5000,
        decoders=['lookup', 'pymatching'],
        custom_decoders=ag_decs,
        print_progress=False,
    )

    print(f'\n{"code":<14}{"decoder":<12}{"p":>6}{"shots":>8}{"p_L":>12}')
    print('-' * 56)
    for st in sorted(stats, key=lambda s: (s.json_metadata['code'], s.json_metadata['p_data'])):
        code = st.json_metadata['code']
        p = st.json_metadata['p_data']
        pL = st.errors / st.shots if st.shots else float('nan')
        print(f'{code:<14}{st.decoder:<12}{p:>6.2f}{st.shots:>8}{pL:>12.5f}')


if __name__ == "__main__":
    main()
