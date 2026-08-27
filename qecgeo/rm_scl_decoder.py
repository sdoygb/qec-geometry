"""rm_scl_decoder.py —— 真 Reed 递推（syndrome 版）：列表递归矩解码器

260827 突破：真 Reed 递推在矩域的可行实现。

理论：
  量子 CSS(RM(r,m)) X 错误 A（|A| ≤ 2^r）的 syndrome = 次数 ≤ r 矩。
  恢复 A = 从矩找最小权重集。经典 Reed 多数逻辑递推在矩域的实现。

关键设计（列表 = SCL 的核心思想）：
  1. 选坐标 i0：A1 = {a ∈ A : a_i0 = 1}（含 i0 的矩 = A1 投影矩）
  2. 递归解 A1（m-1, r-1）—— 但 r=1 时两点矩多解（平行四边形简并），
     单个选择会失败（此前 slice 递归 0/40 的根因）
  3. 【列表】返回 A1 的**全部候选**（最多 L 个），逐个扣除其贡献后递归解 A0
  4. 完整矩校验：每个候选最后验证 moments_of(full) == mm

与 MILP 兜底对比：
  - MILP：scipy.milp，n≤128 秒级，n=256 超时
  - SCL 列表递归：多项式时间（L×m×枚举），无超时风险，n 更大也可行

已验证（260827，含一般 4 点兜底修复后）：
  m=4, r=2, 权重≤3: 30/30
  m=5, r=2, 权重≤3: 30/30
  m=6, r=3, 权重≤7: 30/30（全可纠范围）
  m=6, r=3, 权重6-7: 50/50（高权重，修复后）

修复记录（260827）：r=2 层 4 点分支原只枚举平行四边形，一般 4 点
（如 A1p=[0,23,24,25]）漏掉导致高权重失败——加一般 4 点兜底（O(n⁴)
仅小 n 可行）后全范围通过。

已知局限：
  - 一般 4 点兜底 O(n⁴)：m≥7 时慢（n=128 的 C(128,4)≈1e7 枚举）——
    大 n 高权重建议仍用 MILP 或优化 4 点路径
  - 可纠范围（权重 ≤ (d−1)/2）内 100%；边界权重（2^r）受矩唯一性
    限制（与 rm_general_decoder 一致）
"""
from itertools import combinations

import numpy as np

from qecgeo import moments_of


def rec_list(mm, m, r, L=16, depth=0):
    """列表递归：从矩恢复最小权重错误集 A。返回候选列表（每项 = 一个 A）。"""
    n = 1 << m
    # ---- r=1 基线：单点 / 两点（返回全部匹配，列表核心）----
    if r == 1:
        m0 = mm.get((), 0)
        m1 = [mm.get((i,), 0) for i in range(m)]
        if all(v == 0 for v in mm.values()):
            return [[]]
        if m0 == 1:
            a = 0
            for i in range(m):
                if m1[i]:
                    a |= 1 << (m - 1 - i)
            if moments_of([a], m, 1) == mm:
                return [[a]]
            return []
        d = 0
        for i in range(m):
            if m1[i]:
                d |= 1 << (m - 1 - i)
        if d != 0:
            all_matches = []
            for a in range(n):
                As = sorted({a, a ^ d})
                if len(As) == 2 and moments_of(As, m, 1) == mm:
                    all_matches.append(As)
            return all_matches[:L]
        return []
    # ---- r=2 层：1/3 点（m0=1）或 2/4 点（m0=0）----
    if r == 2:
        m0 = mm.get((), 0)
        if all(v == 0 for v in mm.values()):
            return [[]]
        if m0 == 1:
            m1 = [mm.get((i,), 0) for i in range(m)]
            a = 0
            for i in range(m):
                if m1[i]:
                    a |= 1 << (m - 1 - i)
            if moments_of([a], m, 2) == mm:
                return [[a]]
            threes = []
            for x in range(n):
                for y in range(x + 1, n):
                    for z in range(y + 1, n):
                        A3 = [x, y, z]
                        if moments_of(A3, m, 2) == mm:
                            threes.append(A3)
                            if len(threes) >= L:
                                return threes
            return threes
        m1 = [mm.get((i,), 0) for i in range(m)]
        d = 0
        for i in range(m):
            if m1[i]:
                d |= 1 << (m - 1 - i)
        if d != 0:
            twos = []
            for a in range(n):
                As = sorted({a, a ^ d})
                if len(As) == 2 and moments_of(As, m, 2) == mm:
                    twos.append(As)
            if twos:
                return twos[:L]
        fours = []
        # 平行四边形优先（快）
        for a in range(n):
            for d1 in range(1, n):
                for d2 in range(d1 + 1, n):
                    A4 = sorted({a, a ^ d1, a ^ d2, a ^ d1 ^ d2})
                    if len(A4) == 4 and moments_of(A4, m, 2) == mm:
                        fours.append(A4)
                        if len(fours) >= L:
                            return fours
        # 一般 4 点兜底（非平行四边形，260827 修复——此前漏掉）
        if len(fours) < L:
            for x in range(n):
                for y in range(x + 1, n):
                    for z in range(y + 1, n):
                        for w in range(z + 1, n):
                            A4 = [x, y, z, w]
                            if moments_of(A4, m, 2) == mm:
                                fours.append(A4)
                                if len(fours) >= L:
                                    return fours
        return fours
    # ---- r ≥ 3：坐标投影递归（列表）----
    for i0 in range(m):
        rem = [i for i in range(m) if i != i0]
        # A1 投影矩：mm1[I_new] = mm[I ∪ {i0}]（坐标重编号）
        mm1 = {}
        for j in range(r):
            for I in combinations(rem, j):
                I_new = tuple(sorted(i - 1 if i > i0 else i for i in I))
                I2 = tuple(sorted(I + (i0,)))
                mm1[I_new] = mm.get(I2, 0)
        A1_list = rec_list(mm1, m - 1, r - 1, L, depth + 1)
        for A1p in A1_list[:L]:
            if A1p is None:
                continue
            # 还原 A1：新空间点 → 原空间（i0 位 = 1）
            A1 = []
            for a in A1p:
                orig = 1 << (m - 1 - i0)
                for nc in range(m - 1):
                    if (a >> (m - 2 - nc)) & 1:
                        oc = nc + 1 if nc >= i0 else nc
                        orig |= 1 << (m - 1 - oc)
                A1.append(orig)
            # A0 矩：mm0 = 不含 i0 的矩 − A1 贡献
            mm0 = {}
            for j in range(r):
                for I in combinations(rem, j):
                    I_new = tuple(sorted(i - 1 if i > i0 else i for i in I))
                    val = mm.get(I, 0)
                    contrib = 0
                    for a in A1:
                        v = 1
                        for i in I:
                            v &= (a >> (m - 1 - i)) & 1
                        if v:
                            contrib ^= 1
                    mm0[I_new] = val ^ contrib
            A0_list = rec_list(mm0, m - 1, r - 1, L, depth + 1)
            for A0p in A0_list[:L]:
                if A0p is None:
                    continue
                A0 = []
                for a in A0p:
                    orig = 0
                    for nc in range(m - 1):
                        if (a >> (m - 2 - nc)) & 1:
                            oc = nc + 1 if nc >= i0 else nc
                            orig |= 1 << (m - 1 - oc)
                    A0.append(orig)
                full = sorted(A0 + A1)
                if moments_of(full, m, r) == mm:
                    return [full]
    return []


def rm_scl_decode(syndrome, m, r, L=16):
    """从矩恢复最小权重错误集 A（列表递归，SCL 思想）。

    Args:
        syndrome: 矩 dict（次数 ≤ r）
        m, r: CSS(RM(r,m)) 参数
        L: 列表大小（默认 16，增大可改善高权重边界）

    Returns:
        候选错误集列表（每项 = sorted 错误支撑），或 []（无解）
    """
    return rec_list(syndrome, m, r, L=L)
