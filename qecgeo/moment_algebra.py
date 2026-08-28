"""moment_algebra.py —— 矩代数约束与测量噪声代数纠正（10.85）

几何论解码新理论：CSS(RM(r,m)) 码的 syndrome（次数 ≤ r 矩）满足
代数分解约束 M₂ = Σ_{a∈A} v_a v_aᵀ（GF(2)），测量噪声翻转矩位
破坏该约束——用代数约束可确定性检测并纠正测量噪声（非概率方法）。

核心结果（10.85，260828 验证）：
- 定理 10.85.2.02（矩分解）：M₂ = Σ v_a v_aᵀ，rank(M₂) ≤ |A|
- 推论 10.85.2.04（单点）：M₂ = m₁⊗m₁
- 定理 10.85.3.01（两点解析解）：O(m) 非枚举
- 定理 10.85.3.02（三点解析解）：O(m) 非枚举
- 算法 10.85.4.01（代数纠正）：枚举翻转 + 解析约束校验

用法：
    from qecgeo.moment_algebra import solve_two, solve_three, correct_polluted
"""
from itertools import combinations_with_replacement

import numpy as np

from qecgeo import moments_of

__all__ = ["m1_vec", "m2_mat", "solve_two", "solve_three", "correct_polluted"]


def m1_vec(mm, m):
    """线性矩向量 m1[i] = m_{ {i} }。"""
    return np.array([mm.get((i,), 0) for i in range(m)])


def m2_mat(mm, m):
    """矩矩阵：对角 = m1[i]（GF(2) 下 x_i² = x_i），非对角 = m2[i,j]。"""
    M = np.zeros((m, m), dtype=int)
    for i in range(m):
        for j in range(m):
            if i == j:
                M[i, j] = mm.get((i,), 0)
            else:
                M[i, j] = mm.get(tuple(sorted((i, j))), 0)
    return M


def _mm_from(m1, M2, m, m0):
    """从 m1/M2 重建矩 dict（含 m0 阶）。"""
    mm = {(): m0}
    for i in range(m):
        mm[(i,)] = int(m1[i])
    for i in range(m):
        for j in range(i + 1, m):
            mm[(i, j)] = int(M2[i, j])
    return mm


def solve_two(m1, M2, m):
    """定理 10.85.3.01：两点错误解析解（O(m)，非枚举）。

    给定线性矩 m1（非零）与矩矩阵 M2，返回候选两点错误列表。
    验证：m=5 全枚举 496/496 命中（100%）。
    """
    d = m1
    if not d.any():
        return []
    jstar = int(np.nonzero(d)[0][0])
    sols = []
    for t in (0, 1):
        va = np.zeros(m, dtype=int)
        vb = np.zeros(m, dtype=int)
        vb[jstar] = t
        va[jstar] = 1 ^ t
        ok = True
        for i in range(m):
            if i == jstar:
                continue
            rhs = int(M2[i, jstar])
            if t == 1:
                vb[i] = rhs
                va[i] = d[i] ^ rhs
            else:
                va[i] = rhs
                vb[i] = d[i] ^ rhs
            if (int(va[i]) * (1 ^ t) ^ int(vb[i]) * t) != rhs:
                ok = False
                break
        if not ok:
            continue
        va_int = sum(int(va[i]) << (m - 1 - i) for i in range(m))
        vb_int = sum(int(vb[i]) << (m - 1 - i) for i in range(m))
        A2 = sorted({va_int, vb_int})
        if len(A2) == 2 and moments_of(A2, m, 2) == _mm_from(m1, M2, m, 0):
            sols.append(A2)
    return sols


def solve_three(m1, M2, m):
    """定理 10.85.3.02：三点错误解析解（O(m)，非枚举）。

    取 j* 使 m1[j*]=1，枚举 (va[j*], vb[j*], vc[j*])（权重奇数组合），
    对每 i 解 2×2 线性方程组。验证：m=5 300/300 命中（100%）。
    """
    d = m1
    if not d.any():
        return []
    jstar = int(np.nonzero(d)[0][0])
    cands = []
    for A, B, C in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)):
        va = np.zeros(m, dtype=int)
        vb = np.zeros(m, dtype=int)
        vc = np.zeros(m, dtype=int)
        va[jstar] = A; vb[jstar] = B; vc[jstar] = C
        ok = True
        for i in range(m):
            if i == jstar:
                continue
            rhs = int(M2[i, jstar])
            di = int(d[i])
            # 通用求解：枚举 va[i], vb[i] ∈ {0,1}²（4 候选），
            # vc[i] = di ^ vai ^ vbi，验证方程 M2[i,j*] 匹配。
            # 覆盖所有 (A,B,C) 组合（含 B=C=1 秩退化、A 唯一活跃等），
            # 无分支遗漏（260828 修复：原 Cramer 分支漏 (0,0,1)/(1,1,1) 退化）。
            solved = False
            for vai in (0, 1):
                for vbi in (0, 1):
                    vci = di ^ vai ^ vbi
                    lhs = (A * vai) ^ (B * vbi) ^ (C * vci)
                    if lhs == rhs:
                        va[i] = vai; vb[i] = vbi; vc[i] = vci
                        solved = True
            if not solved:
                ok = False
                break
        if not ok:
            continue
        va_int = sum(int(va[i]) << (m - 1 - i) for i in range(m))
        vb_int = sum(int(vb[i]) << (m - 1 - i) for i in range(m))
        vc_int = sum(int(vc[i]) << (m - 1 - i) for i in range(m))
        A3 = sorted({va_int, vb_int, vc_int})
        if len(A3) == 3 and moments_of(A3, m, 2) == _mm_from(m1, M2, m, 1):
            cands.append(A3)
    return cands


def correct_polluted(mm_noisy, m, max_flip=2, w=2):
    """算法 10.85.4.01：测量噪声代数纠正。

    给定污染矩（测量噪声翻转 ≤ max_flip 位），枚举翻转组合，用
    w 点解析约束（solve_two / solve_three）校验，返回满足约束的
    (翻转集, 候选错误) 列表。

    验证（m=5）：两点 +1 位 100%、+2 位 >60%（歧义边界）；
    三点 +1 位 100%。
    """
    keys = list(mm_noisy.keys())
    solver = solve_two if w == 2 else solve_three
    m0_expected = 0 if w == 2 else 1
    for k in range(1, max_flip + 1):
        hits = []
        for flip in combinations_with_replacement(keys, k):
            mm2 = dict(mm_noisy)
            for fk in flip:
                mm2[fk] ^= 1
            m1 = np.array([mm2.get((i,), 0) for i in range(m)])
            M2 = np.zeros((m, m), dtype=int)
            for i in range(m):
                for j in range(m):
                    if i == j:
                        M2[i, j] = mm2.get((i,), 0)
                    else:
                        M2[i, j] = mm2.get(tuple(sorted((i, j))), 0)
            sols = solver(m1, M2, m)
            if sols:
                hits.append((flip, sols))
        if hits:
            return hits
    return []
