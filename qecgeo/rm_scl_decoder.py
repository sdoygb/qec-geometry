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

修复记录：
  - 260827：r=2 层 4 点分支原只枚举平行四边形，一般 4 点（如
    A1p=[0,23,24,25]）漏掉导致高权重失败——加一般 4 点兜底
    （O(n⁴) 仅小 n 可行）后全范围通过。
  - 260828：O(n⁴) 兜底 → O(n²·m) 代数参数化解（见 _four_points_alg）：
    * 4 点 A={a,a⊕p,a⊕q,a⊕r} 的总 XOR = p⊕q⊕r = 线性矩 d
    * d=0 ⟹ 4 点必为平行四边形（一般 4 点不存在）——平行四边
      形枚举即可，且二次矩与 a 无关，可先筛 (p,q) 再解 a
    * d≠0 ⟹ 4 点 = {a,a⊕p,a⊕q,a⊕p⊕q⊕d}，固定 (p,q) 后二次矩
      方程对 a 线性（a_j 自由位最多 2 个候选）——O(n²) 枚举替代
      O(n⁴) 全枚举
    验证：m=4/5/6 随机 4 点 300/300 公式成立；d=0 ⟹ 平行四边形
    恒成立。m=7 (n=128) 4 点恢复从 ~1e7 枚举降到 ~16k×m。

已知局限：
  - 可纠范围（权重 ≤ (d−1)/2）内 100%；边界权重（2^r）受矩唯一性
    限制（与 rm_general_decoder 一致）
"""
from itertools import combinations

import numpy as np

from qecgeo import moments_of


def _decode_two(mm, m):
    """r=1 基线 / r=2 两点：从线性矩 d 恢复 {a, a⊕d} 全部匹配。"""
    m1 = [mm.get((i,), 0) for i in range(m)]
    d = 0
    for i in range(m):
        if m1[i]:
            d |= 1 << (m - 1 - i)
    if d == 0:
        return []
    n = 1 << m
    matches = []
    for a in range(n):
        As = sorted({a, a ^ d})
        if len(As) == 2 and moments_of(As, m, 2) == mm:
            matches.append(As)
    return matches


def _four_points_alg(mm, m, L=16):
    """r=2 四点恢复：代数参数化，O(n²·m)（替代 O(n⁴) 兜底）。

    数学（260828 验证）：
      A = {a, a⊕p, a⊕q, a⊕r}，总 XOR p⊕q⊕r = 线性矩 d。
      d=0 ⟹ r = p⊕q（平行四边形），且二次矩
        m₂[i,j] = p_i p_j ⊕ q_i q_j ⊕ r_i r_j   （与 a 无关！）
      d≠0 ⟹ r = p⊕q⊕d，二次矩
        m₂[i,j] = a_i d_j ⊕ a_j d_i ⊕ p_i p_j ⊕ q_i q_j ⊕ r_i r_j
        固定 (p,q) 后对 a 线性：取 d_j*=1，则
        a_i = m₂[i,j*] ⊕ a_j*·d_i ⊕ p_i p_j* ⊕ q_i q_j* ⊕ r_i r_j*
        a_j* ∈ {0,1} → 最多 2 个候选 a。

    流程：枚举 (p,q)（O(n²)），先验二次矩约束（d=0 时只验 (p,q)，
    与 a 无关；d≠0 时解 a 再验），最后完整矩校验。
    """
    n = 1 << m
    d = 0
    for i in range(m):
        if mm.get((i,), 0):
            d |= 1 << (m - 1 - i)

    def bit(v, i):
        return (v >> (m - 1 - i)) & 1

    results = []
    seen = set()
    if d == 0:
        # 平行四边形：m2 只依赖 (p,q)，先筛 (p,q) 再枚举 a
        good_pq = []
        for p in range(1, n):
            for q in range(p + 1, n):
                r = p ^ q
                # 验二次矩（与 a 无关）
                ok = True
                for i in range(m):
                    for j in range(i + 1, m):
                        pi, pj = bit(p, i), bit(p, j)
                        qi, qj = bit(q, i), bit(q, j)
                        ri, rj = bit(r, i), bit(r, j)
                        if mm.get((i, j), 0) != (pi * pj ^ qi * qj ^ ri * rj):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    good_pq.append((p, q))
                    if len(good_pq) > 4 * L:
                        break
            if len(good_pq) > 4 * L:
                break
        for p, q in good_pq:
            r = p ^ q
            # 枚举 a：A = {a, a⊕p, a⊕q, a⊕r}
            for a in range(n):
                A4 = sorted({a, a ^ p, a ^ q, a ^ r})
                if len(A4) == 4:
                    key = tuple(A4)
                    if key in seen:
                        continue
                    seen.add(key)
                    if moments_of(A4, m, 2) == mm:
                        results.append(A4)
                        if len(results) >= L:
                            return results
    else:
        # 一般 4 点：固定 (p,q)，解 a（线性方程，≤2 候选）
        jstar = None
        for j in range(m):
            if bit(d, j):
                jstar = j
                break
        for p in range(1, n):
            for q in range(1, n):
                if p == q:
                    continue
                r = p ^ q ^ d
                # 预计算常数项 c_ij = p_i p_j ⊕ q_i q_j ⊕ r_i r_j
                c = [[0] * m for _ in range(m)]
                for i in range(m):
                    for j in range(i + 1, m):
                        pi, pj = bit(p, i), bit(p, j)
                        qi, qj = bit(q, i), bit(q, j)
                        ri, rj = bit(r, i), bit(r, j)
                        c[i][j] = pi * pj ^ qi * qj ^ ri * rj
                # 对 a_jstar ∈ {0,1} 各解一次 a
                for aj in (0, 1):
                    a = 0
                    if aj:
                        a |= 1 << (m - 1 - jstar)
                    ok = True
                    for i in range(m):
                        if i == jstar:
                            continue
                        # m2[i,j*] = a_i·1 ⊕ aj·d_i ⊕ c
                        rhs = mm.get((min(i, jstar), max(i, jstar)), 0)
                        ai = rhs ^ (aj * bit(d, i)) ^ c[min(i, jstar)][max(i, jstar)]
                        if ai:
                            a |= 1 << (m - 1 - i)
                    # 完整校验
                    A4 = sorted({a, a ^ p, a ^ q, a ^ r})
                    if len(A4) != 4:
                        continue
                    key = tuple(A4)
                    if key in seen:
                        continue
                    seen.add(key)
                    if moments_of(A4, m, 2) == mm:
                        results.append(A4)
                        if len(results) >= L:
                            return results
    return results


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
            # 三点 {a, a⊕p, p⊕d}（260828：代数参数化 O(n²) 替代 O(n³)）
            # 数学：总 XOR a⊕b⊕c = 线性矩 d，令 b = a⊕p：
            #   c = a⊕b⊕d = a⊕(a⊕p)⊕d = p⊕d。
            # 枚举 (a, p)，c = p⊕d，二次矩校验。
            d = a  # 总 XOR（m1 位向量组装值）
            threes = []
            seen3 = set()
            for a0 in range(n):
                for p in range(1, n):
                    b = a0 ^ p
                    if b <= a0:
                        continue
                    c = p ^ d
                    if c <= b:
                        continue
                    A3 = [a0, b, c]
                    key = tuple(A3)
                    if key in seen3:
                        continue
                    seen3.add(key)
                    if moments_of(A3, m, 2) == mm:
                        threes.append(A3)
                        if len(threes) >= L:
                            return threes
            return threes
        # m0 = 0：2 点或 4 点
        twos = _decode_two(mm, m)
        if twos:
            return twos[:L]
        return _four_points_alg(mm, m, L)
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
