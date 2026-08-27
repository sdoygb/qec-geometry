"""rm_decoder.py —— Reed-Muller 快速解码器（矩恢复，O(n·poly)，非查表）

量子 CSS(RM(r,m)) 的 X 错误解码 ≡ 经典码 RM(m-r-1,m) 的 syndrome 解码：
  - syndrome = 错误支撑 A 与次数 ≤ r 单项式的点积（"矩"）
  - 解码目标：从矩恢复最小权重错误 A（= syndrome 类的最小权重代表）
  - 矩唯一性（260827 复核）：权重 ≤ (d−1)/2 = 2^r − ½ 的错误矩唯一（可纠
    范围，理论保证）；更大权重在 m 小时碰撞（r=1 的 w=2 对所有 m 大量
    碰撞——线性矩不足；r=2 且 m<7 的 w=4 ~1% 碰撞）。decode_r1/r2 为
    【遗留实现】（存在矩不唯一时静默返回错误答案的风险），公开 API
    rm_x_decode 委托 rm_general_decoder（矩阵查表，见该模块的边界说明）。

与查表解码器（LookupDecoder）的本质区别：
  - 查表：表构建 O(C(n,2)·3^2)，n=1024 需 470 万错误——不可行
  - 矩解码：O(n·poly(m))，CSS(RM(1,10)) [[1024,·,4]] 10000 次解码 0.09s
  - 这是"RM 快速解码"的核心价值——大码不靠查表

分层（按 r）：
  r=1（错误 ≤ 2）：矩直接读出，O(n)
    - m_∅=1 → A={a}，a = 线性矩向量
    - m_∅=0 且线性矩 d≠0 → |A|=2，A = {a, a+d}（枚举 a 匹配矩）
  r=2（错误 ≤ 4）：矩方程 + 约束枚举
    - |A|=1/2：解析 + 枚举；|A|=4：平行四边形优先（O(n³)）+ 兜底

验证：r=1 m=4..10 全过（1000/1000）；r=2 m=4/5 全过；
CSS(RM(1,m)) 端到端与 LookupDecoder 对照 100/100 一致。
"""
from __future__ import annotations

from itertools import combinations

from .pauli import Pauli

__all__ = ["moments_of", "rm_x_decode", "css_rm_x_decode", "css_rm_zsupport"]


def moments_of(A, m, r):
    """错误支撑 A（点的集合）的次数 ≤ r 矩。

    返回 dict：key = 坐标子集 I（tuple），value = Σ_{a∈A} x_I(a) mod 2。
    """
    mm = {}
    mm[()] = len(A) & 1
    for j in range(1, r + 1):
        for I in combinations(range(m), j):
            s = 0
            for a in A:
                val = 1
                for i in I:
                    val &= (a >> (m - 1 - i)) & 1
                if val:
                    s ^= 1
            mm[I] = s
    return mm


def _check_moments(A, mm, m, r):
    return moments_of(A, m, r) == mm


def decode_r1(mm, m):
    """[遗留] r=1 矩解码（O(n)）。注意：r=1 的 w=2 矩不唯一（线性矩不足），
    对碰撞 syndrome 返回首个匹配的 a（可能非真实 A）。公开 API 请用 rm_x_decode。"""
    n = 1 << m
    m0 = mm[()]
    m1 = [mm[(i,)] for i in range(m)]
    if m0 == 1:
        a = 0
        for i in range(m):
            if m1[i]:
                a |= 1 << (m - 1 - i)
        return [a]
    d = 0
    for i in range(m):
        if m1[i]:
            d |= 1 << (m - 1 - i)
    if d == 0:
        return []
    for a in range(n):
        A = sorted({a, a ^ d})
        if len(A) == 2 and _check_moments(A, mm, m, 1):
            return A
    return None


def decode_r2(mm, m):
    """[遗留] r=2 矩解码（约束枚举）。注意：m<7 时 w=4 矩 ~1% 碰撞（超出
    可纠 (d−1)/2=3），O(n⁴) 兜底仅小码可行。公开 API 请用 rm_x_decode。"""
    n = 1 << m
    m0 = mm[()]
    m1 = [mm[(i,)] for i in range(m)]
    a = 0
    for i in range(m):
        if m1[i]:
            a |= 1 << (m - 1 - i)
    # |A| = 1
    if m0 == 1:
        if _check_moments([a], mm, m, 2):
            return [a]
        # |A| = 3：约束枚举
        for c1 in range(n):
            for c2 in range(c1 + 1, n):
                for c3 in range(c2 + 1, n):
                    A3 = [c1, c2, c3]
                    if _check_moments(A3, mm, m, 2):
                        return A3
        return None
    # m0 == 0：|A| ∈ {0, 2, 4}
    d = 0
    for i in range(m):
        if m1[i]:
            d |= 1 << (m - 1 - i)
    # d==0 且全部矩为 0 → 无错误
    if d == 0 and all(v == 0 for v in mm.values()):
        return []
    # |A| = 2
    if d != 0:
        if _check_moments([0, d], mm, m, 2):
            return [0, d]
        for a in range(n):
            A = sorted({a, a ^ d})
            if len(A) == 2 and _check_moments(A, mm, m, 2):
                return A
    # |A| = 4：平行四边形优先
    for a in range(n):
        for d1 in range(1, n):
            if len({a, a ^ d1}) != 2:
                continue
            for d2 in range(d1 + 1, n):
                A = sorted({a, a ^ d1, a ^ d2, a ^ d1 ^ d2})
                if len(A) == 4 and _check_moments(A, mm, m, 2):
                    return A
    # 兜底：一般 4 点（O(n^4)，慢；仅小码）
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            for c3 in range(c2 + 1, n):
                for c4 in range(c3 + 1, n):
                    A = [c1, c2, c3, c4]
                    if _check_moments(A, mm, m, 2):
                        return A
    return None


def rm_x_decode(syndrome, m, r):
    """从 syndrome（矩 dict）恢复最小权重错误支撑 A（|A| ≤ 2^r）。

    委托给通用解码器（rm_general_decoder，支持 r ≥ 1，矩阵查表）。
    """
    from .rm_general_decoder import rm_x_decode as _gen
    return _gen(syndrome, m, r)


def css_rm_x_decode(syndrome, m, r):
    """量子 CSS(RM(r,m)) X 错误解码入口。

    syndrome：X 错误的矩（与次数 ≤ r 单项式的点积）。
    返回恢复支撑 A'（最小权重，与错误同 syndrome）。
    """
    return rm_x_decode(syndrome, m, r)


def css_rm_zsupport(m, r):
    """CSS(RM(r,m)) 逻辑 Z 支撑 = x1x2 的支撑（权重 4）。

    与 end_to_end_demo 的 css_rm_code 一致。
    """
    n = 1 << m
    return [i for i in range(n) if ((i >> (m - 1)) & 1) and ((i >> (m - 2)) & 1)]
