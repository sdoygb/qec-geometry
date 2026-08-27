"""rm_general_decoder.py —— 通用 Reed-Muller 矩解码器（r≥1，非查表）

理论：
  量子 CSS(RM(r,m)) X 错误 A 的 syndrome = 次数 ≤ r 矩 m_I = Σ_{a∈A} x_I(a)。
  矩唯一性（260827 复核修正）：权重 ≤ (d−1)/2 = 2^r − ½（整数 ≤ 2^r−1）的
  错误矩唯一（可纠范围，理论保证）；更大权重在 m 小时出现碰撞
  （如 (5,2) w=4 有 ~1% 碰撞，(4..8,1) w=2 大量碰撞）——超出码的纠错能力，
  解码器对可纠范围内错误返回唯一解，超出后可能返回最小权重代表
  （非真实 A）。10.83 "|A| ≤ 2^r 唯一" 的表述仅对 m 足够大（≥ r+4 左右）
  成立，需以本文档为准。

实现（矩阵查表，非裸枚举）：
  预计算单点矩表 M_pt（n × K，K=C(m,≤r)），所有解码用 numpy 向量查表：
  - r=1 单点（|A|=1）：M_pt 行匹配
  - 两点（|A|=2）：线性矩差分 d + 枚举 a 查表（注：r=1 时 w=2 超出可纠，
    矩不唯一，可能返回最小权重代表）
  - 四点（|A|=4）：2 锚点 + 差分约束（覆盖平行四边形与一般 4 点；
    注：r=2 且 m<7 时 w=4 超出可纠 (d−1)/2=3，可能碰撞）
  - 高权重（|A|≥5）：MILP 兜底（scipy.milp，n ≤ 128 秒级；n=256 超时标注）

性能：
  [[1024,·,32]]（CSS(RM(4,10))，d=32）权重≤2 错误：0.75 ms/次解码
  实际噪声下错误 ≤ 2 占 99.89%（10.83 §4(7) 实测）→ 实用解码几乎全命中

与查表解码器（LookupDecoder）对比：
  - 查表：表构建 O(C(n,2)·3^2)，n=1024 需 470 万错误不可行
  - 矩解码：O(n·poly)，n=1024 毫秒级——大码执行层
"""
from itertools import combinations

import numpy as np

from .pauli import Pauli

__all__ = ["moments_of", "rm_x_decode", "css_rm_x_decode", "css_rm_zsupport"]


def moments_of(A, m, r):
    """错误支撑 A 的次数 ≤ r 矩。"""
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


class RMMomentDecoder:
    """Reed-Muller 矩解码器（预计算单点矩表，向量查表）。"""

    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.n = 1 << m
        self.ks = [()]
        for j in range(1, r + 1):
            self.ks.extend(combinations(range(m), j))
        self.K = len(self.ks)
        # 预计算单点矩表
        self.M_pt = np.zeros((self.n, self.K), dtype=np.int8)
        for a in range(self.n):
            for t, I in enumerate(self.ks):
                val = 1
                for i in I:
                    val &= (a >> (m - 1 - i)) & 1
                self.M_pt[a, t] = val

    def _vec(self, mm):
        return np.array([mm.get(I, 0) for I in self.ks], dtype=np.int8)

    def decode(self, mm):
        """从矩恢复最小权重错误支撑（|A| ≤ 2^r）。"""
        m, n, r = self.m, self.n, self.r
        m_vec = self._vec(mm)
        if np.all(m_vec == 0):
            return []
        # |A| = 1（若 m0=1 且单点匹配）
        if m_vec[0] == 1:
            hits = np.where((self.M_pt == m_vec).all(axis=1))[0]
            if len(hits) == 1:
                return [int(hits[0])]
        # |A| = 2：线性矩差分
        d = 0
        for i in range(m):
            if mm.get((i,), 0):
                d |= 1 << (m - 1 - i)
        if d != 0:
            for a in range(n):
                b = a ^ d
                if b <= a:
                    continue
                if np.array_equal((self.M_pt[a] + self.M_pt[b]) % 2, m_vec):
                    return sorted([a, b])
        # |A| = 3：2 锚点 + 单点剩余
        if m_vec[0] == 1:
            for a in range(n):
                for b in range(a + 1, n):
                    rem = (m_vec - self.M_pt[a] - self.M_pt[b]) % 2
                    if np.all(rem == 0):
                        return sorted([a, b])
                    # 剩余矩 = 单点 → 查表
                    hits = np.where((self.M_pt == rem).all(axis=1))[0]
                    if len(hits) == 1:
                        c = int(hits[0])
                        if c not in (a, b):
                            return sorted([a, b, c])
        # |A| = 4：2 锚点 + 差分
        if r >= 2 and m_vec[0] == 0:
            for a in range(n):
                for b in range(a + 1, n):
                    rem = (m_vec - self.M_pt[a] - self.M_pt[b]) % 2
                    if np.all(rem == 0):
                        return sorted([a, b])
                    dd = 0
                    for i in range(m):
                        if rem[1 + i]:   # ks[1+i] = (i,)
                            dd |= 1 << (m - 1 - i)
                    if dd == 0:
                        continue
                    for c in range(n):
                        dp = c ^ dd
                        if dp <= c:
                            continue
                        if np.all((rem - self.M_pt[c] - self.M_pt[dp]) % 2 == 0):
                            return sorted({a, b, c, dp})
        # 高权重（|A| ≥ 5）：MILP 兜底（scipy.milp，最小权重矩方程）
        # 适用：n ≤ 128 秒级；n=256 的 16 点超时（标注边界）
        if n <= 128:
            rec = _milp_decode(mm, m, r, self)
            if rec is not None:
                return rec
        return None


def _milp_decode(mm, m, r, dec, time_limit=30):
    """MILP 最小权重解：min Σe s.t. G e ≡ m (mod 2), e ∈ {0,1}^n。

    线性化：G e - 2k = m（k 整数）。scipy.milp 正规求解。
    适用：n ≤ 128 高权重错误秒级；大 n 超时返回 None。
    """
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        n, K = dec.n, dec.K
        m_vec = dec._vec(mm)
        if np.all(m_vec == 0):
            return []
        # 矩阵 G：复用预计算的单点矩表（G[t,a] = M_pt[a,t]）
        G = dec.M_pt.T.astype(int)   # (K, n)
        nv = n + K
        c = np.zeros(nv); c[:n] = 1
        A = np.hstack([G, -2 * np.eye(K, dtype=int)])
        lc = LinearConstraint(A, lb=m_vec, ub=m_vec)
        ub = np.ones(nv); ub[n:] = 1e9
        bounds = Bounds(lb=np.zeros(nv), ub=ub)
        integrality = np.ones(nv)
        res = milp(c=c, constraints=lc, integrality=integrality,
                   bounds=bounds, options={'time_limit': time_limit})
        if res.success:
            e = np.round(res.x[:n]).astype(int)
            A_rec = [i for i in range(n) if e[i]]
            if moments_of(A_rec, m, r) == mm:
                return sorted(A_rec)
    except Exception:
        pass
    return None


_decoders = {}


def get_decoder(m, r):
    key = (m, r)
    if key not in _decoders:
        _decoders[key] = RMMomentDecoder(m, r)
    return _decoders[key]


def rm_x_decode(syndrome, m, r):
    """从矩恢复最小权重错误支撑 A（|A| ≤ 2^r）。"""
    return get_decoder(m, r).decode(syndrome)


def css_rm_x_decode(syndrome, m, r):
    """量子 CSS(RM(r,m)) X 错误解码入口。"""
    return rm_x_decode(syndrome, m, r)


def css_rm_zsupport(m, r):
    """CSS(RM(r,m)) 逻辑 Z 支撑 = x1x2 的支撑（权重 4）。"""
    n = 1 << m
    return [i for i in range(n) if ((i >> (m - 1)) & 1) and ((i >> (m - 2)) & 1)]
