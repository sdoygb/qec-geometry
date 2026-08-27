#!/usr/bin/env python3
"""end_to_end_demo.py —— 一台 Mac 执行纠错：完整端到端演示（10.83，v2）

v2 两个改进（相对 v1）：
  [A] 真正应用恢复操作并重算逻辑值——从"检测"升级为"纠错"验证
  [B] 码升级到 [[16,6,4]]（CSS(RM(1,4))，d=4）——权重 2 错误混合型全部
      唯一（10.83 定理 10.83.1.01），恢复表 w_max=2 覆盖更多可纠错误

模拟方式（码容量，诚实标注）：stim 的 detector_sampler 把显式 `X` 门当作
"预期操作"吸收进测量基（不触发探测器），故本 demo 不依赖 stim 探测器采样，
而用**自洽的码容量模拟**：随机注入 Pauli 错误 → `dec.syndrome_of` 计算
syndrome（模拟"量子硬件读出"）→ 查表恢复 → 应用恢复 → 重算逻辑值。
stabilizer 模拟器（stim.TableauSimulator）可用于独立验证恢复正确性。

闭环：注入错误 → syndrome → 自研解码器查表 → 应用恢复 → 验证 p_L。

运行: python3 scripts/end_to_end_demo.py（零外部依赖）
"""
import os
import sys
import time
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qecgeo import LookupDecoder
from qecgeo.pauli import Pauli


def css_rm_code(m, r):
    """CSS(RM(r,m)) [[2^m, 2^m−2·dim, 2^{r+1}]]：gens + 逻辑 Z。

    逻辑 Z = x_1 x_2 的支撑（RM(r+1,m) 中不在 RM(r,m) 的二次单项式）。
    """
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    gens = []
    for row in rows:
        gens.append(Pauli(n, [1 if b else 0 for b in row]))   # X 型稳定子
        gens.append(Pauli(n, [2 if b else 0 for b in row]))   # Z 型稳定子

    def bit(col, i):
        return (col >> i) & 1
    lz = Pauli(n, [2 if bit(col, m - 1) and bit(col, m - 2) else 0 for col in range(n)])
    return gens, lz


def sample_error(n, p, rng):
    """码容量噪声：每比特独立 p 概率的 X/Z/Y 错误。返回组合 Pauli。"""
    t = [0] * n
    for j in range(n):
        if rng.random() < p:
            t[j] = rng.choice((1, 2, 3))
    return Pauli(n, t)


def main():
    print("=" * 80)
    print("一台 Mac 执行纠错 v2：注入错误 → syndrome → 自研解码器 → 恢复 → 验证")
    print("=" * 80)
    m, r = 4, 1
    gens, lz = css_rm_code(m, r)
    n = 1 << m
    lz_support = set(i for i in range(n) if lz.t[i] == 2)
    print(f"码: CSS(RM(1,4)) = [[{n},{n - len(gens)},{1 << (r + 1)}]] "
          f"(d=4, 逻辑 Z = x1x2 权重 {lz.weight()})")

    for p in (0.001, 0.005, 0.01):
        shots = 100000
        rng = Random(42)

        # 1. 恢复表（几何论恢复表设计）
        t0 = time.time()
        dec = LookupDecoder(gens, n, name='[[16,6,4]]')
        dec.build(w_max=2)
        t_build = (time.time() - t0) * 1000

        # 2. 模拟 shots 次：注入错误 → syndrome（模拟量子硬件读出）
        t0 = time.time()
        flip_E = np.zeros(shots, dtype=int)    # 错误本身翻转逻辑 Z
        flip_corr = np.zeros(shots, dtype=int) # 纠错后逻辑值
        resolved = np.zeros(shots, dtype=bool) # syndrome 可识别（恢复成功）
        for s in range(shots):
            E = sample_error(n, p, rng)
            synd = dec.syndrome_of(E)
            # 错误翻转逻辑 Z ⟺ E 的 X 分量与 Lz 支撑交奇数
            ex = set(i for i in range(n) if E.t[i] in (1, 3))
            flip_E[s] = len(ex & lz_support) % 2
            # 查表恢复（最小权重代表）
            R = dec.decode(synd)
            if R.weight() == 0:
                # 无错误（syndrome 0）：不恢复
                flip_corr[s] = flip_E[s]
                resolved[s] = True
            else:
                # 应用恢复：纠错后逻辑值 = E 翻转 ⊕ R 翻转（R 正确则 = 0）
                rx = set(i for i in range(n) if R.t[i] in (1, 3))
                flip_R = len(rx & lz_support) % 2
                flip_corr[s] = flip_E[s] ^ flip_R
                resolved[s] = synd in dec.table
        t_dec = (time.time() - t0) * 1000

        # 3. 验证
        pL_bare = float(flip_E.mean())
        pL_corr = float(flip_corr.mean())
        mask = resolved
        pL_res = float(flip_corr[mask].mean()) if mask.any() else float('nan')

        print(f"\n--- noise p = {p} ---")
        print(f"  恢复表构建 {t_build:.0f} ms | 模拟+解码 {t_dec:.0f} ms "
              f"({shots} shots) | 可识别率 {resolved.mean():.4f}")
        print(f"  裸逻辑错误率 p_L(bare)     = {pL_bare:.4f}")
        print(f"  纠错后 p_L(corrected)      = {pL_corr:.4f}")
        print(f"  可识别样本纠错后 p_L       = {pL_res:.4f}")
        if pL_corr == 0:
            print(f"  → 纠错有效性: p_L → 0（全部可识别样本完全纠正）✓")
        elif pL_corr < pL_bare:
            print(f"  → 纠错有效性: p_L 降低 {pL_bare / pL_corr:.1f}×")
        else:
            print(f"  → 纠错有效性: 未改善（边界情况）")


if __name__ == "__main__":
    main()
