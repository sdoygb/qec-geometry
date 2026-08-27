#!/usr/bin/env python3
"""sinter_collect_demo.py —— sinter.collect：自定义解码器 vs pymatching 对照

可复现对照实验（sinter 标准流程）：
- 码 1：CSS(RM(1,4)) [[16,6,4]]（26 qubit 电路，rounds=2 差分，数据 depolarize + 测量翻转）
  - 解码器 A：本库查表解码器（LookupSinterDecoder，sinter.Decoder 接口）
  - 解码器 B：pymatching (MWPM)
- 解码器 A：本库查表解码器（LookupSinterDecoder，sinter.Decoder 接口）
  - 解码器 B：pymatching (MWPM)

输出：data/sinter_benchmark.csv（code, decoder, p_data, p_meas, shots, errors, p_L）
无任何判断句——纯数据，可复现。

注意：lookup 解码器用 CSS(RM(1,4)) [[16,6,4]] 训练（z/x 探测器索引 + 恢复表
来自该码）。
的解码器套到 B 码上，非有效基准，仅演示"解码器-码不匹配"行为。

运行: .venv311/bin/python scripts/sinter_collect_demo.py
"""
import csv
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sinter
import stim

from scripts.ag_stim_memory import build_rounds2, extract_stabilizers
from scripts.sinter_lookup_decoder import LookupSinterDecoder

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def ag_task(p_data, p_meas, shots):
    """CSS(RM(1,4)) [[16,6,4]] 任务 + 查表解码器（sinter.Decoder 接口）。"""
    m, r = 4, 1
    n = 1 << m
    c, _ = build_rounds2(m, r, p_data, p_meas)
    dec_z, dec_x, z_dets, x_dets = extract_stabilizers(m, r)
    I = tuple(range(r + 1))
    lz = [a for a in range(n) if all((a >> (m - 1 - i)) & 1 for i in I)]
    dec = LookupSinterDecoder(dec_z=dec_z, dec_x=dec_x,
                              z_det_indices=list(z_dets), x_det_indices=list(x_dets),
                              obs_map_z=[lz], obs_map_x=[[]])
    task = sinter.Task(
        circuit=c,
        json_metadata={"code": "CSS(RM(1,4)) [[16,6,4]]", "p_data": p_data, "p_meas": p_meas},
    )
    return task, {"lookup": dec}



def main():
    ps = (0.01, 0.02, 0.03)
    p_meas = 0.01
    shots = 5000

    # 组装任务
    tasks = []
    custom = {}
    for p in ps:
        task, decs = ag_task(p, p_meas, shots)
        tasks.append(task)
        custom["lookup"] = decs["lookup"]

    # sinter.collect（标准流程，固定解码器集合）
    stats = sinter.collect(
        num_workers=2,
        tasks=tasks,
        max_shots=shots,
        decoders=["lookup", "pymatching"],
        custom_decoders=custom,
        print_progress=False,
    )

    # 结构化输出
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "sinter_benchmark.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "decoder", "p_data", "p_meas", "shots", "errors", "p_L"])
        rows = []
        for st in sorted(stats, key=lambda s: (s.json_metadata["code"], s.json_metadata["p_data"])):
            pL = st.errors / st.shots if st.shots else float("nan")
            w.writerow([st.json_metadata["code"], st.decoder,
                        st.json_metadata["p_data"], p_meas, st.shots, st.errors, f"{pL:.6f}"])
            rows.append((st.json_metadata["code"], st.decoder, st.json_metadata["p_data"], pL))
    print(f"结果已写入 {out_path}")

    print(f"\n{'code':<24}{'decoder':<10}{'p':>6}{'shots':>8}{'p_L':>12}")
    print("-" * 62)
    for code, decoder, p, pL in sorted(rows):
        print(f"{code:<24}{decoder:<10}{p:>6.2f}{shots:>8}{pL:>12.6f}")


if __name__ == "__main__":
    main()
