"""qecgeo/closedform.py —— AG 完备码族闭式参数（几何论 10.27–10.36）

零电路、零模拟：由组合闭式直接给出码参数、失败率、损失标度、零损失边界。
定理引用：
  - 码参数   [[2^m, k, 2^{r+1}]]: 10.30
  - fail(w0) 引理 10.35.2.07
  - kappa    引理 10.35.2.10
  - loss(θ)  定理 10.35.1.07
  - 零损失   定理 10.31.1.01
"""
from __future__ import annotations

import math
from math import comb
from fractions import Fraction

def gb(m, k):
    """高斯二项 [m k]_2（正整数）"""
    if k < 0 or k > m:
        return 0
    num = den = 1
    for i in range(k):
        num *= (1 << (m - i)) - 1
        den *= (1 << (k - i)) - 1
    return num // den


def flats(m, k):
    """m 维 F_2 空间中 k-平坦数 = 2^{m-k} [m k]_2"""
    return (1 << (m - k)) * gb(m, k)


_E = {}


def E(k, s):
    """k 维平坦内、仿射包恰 k 维的 s 点子集数（递推）"""
    key = (k, s)
    if key in _E:
        return _E[key]
    if s == 1:
        r = 1 if k == 0 else 0
        _E[key] = r
        return r
    if k == 0 or s > (1 << k):
        _E[key] = 0
        return 0
    total = comb(1 << k, s)
    for j in range(k):
        total -= flats(k, j) * E(j, s)
    _E[key] = total
    return total


def dim_rm(m, r):
    """RM(r,m) 维数"""
    return sum(comb(m, i) for i in range(r + 1))


def ag_params(m, r):
    """AG 完备码 [[2^m, k, 2^{r+1}]] 全套闭式参数（10.35）。"""
    n = 1 << m
    d = 1 << (r + 1)
    w0 = 1 << r
    k = n - 2 * dim_rm(m, r)
    if k < 1:
        return None
    Pw = Fraction(flats(m, r + 1) * E(r + 1, w0) + flats(m, r), comb(n, w0))
    Pr = Fraction(flats(m, r), comb(n, w0))
    Pr1 = Fraction(flats(m, r + 1) * E(r + 1, w0), comb(n, w0))
    v_r, v_r1 = 1 << (m - r), 2
    fail = Fraction(1) - Pr / (v_r * Pw) - Pr1 / (v_r1 * Pw)
    kap = (1 << ((r + 1) * (m - r - 1))) / gb(m, r + 1)
    Pprime = Fraction(flats(m, r + 1) * comb(1 << (r + 1), w0 + 1), comb(n, w0 + 1))
    c_d = float(Fraction(comb(n, w0)) * Pw * fail / (1 << (2 * w0))) * kap
    c_nx = float(Fraction(comb(n, w0 + 1)) * Pprime / (1 << (2 * (w0 + 1)))) * kap
    return dict(m=m, r=r, n=n, k=k, d=d, w0=w0,
                Pw=float(Pw), fail=float(fail), kap=kap,
                c_d=c_d, ln_cd=math.log(c_d), c_nx=c_nx, rho=c_nx / c_d,
                rate=k / n)


def zero_loss_boundary(d):
    """定理 10.31.1.01：注入 k ≤ ⌊(d-1)/2⌋ 比特相干旋转 → 损失恒 0。"""
    return (d - 1) // 2


def loss_at_theta(c_d, d, theta):
    """损失标度 loss(θ) = c_d·θ^d（定理 10.35.1.07 主阶）。"""
    return c_d * (theta ** d)


TRANSVERSAL_GATES = "{Pauli, CNOT, H, 对角相位门, 逻辑测量}"  # 命题 10.30.3.01




# ============ PG 完备码族（10.28）============

def pg_params(m):
    """PG 完备码 [[2^m-1, 2^m-1-2m, 3]] 参数（10.28）。

    方向完备码：H 列取遍 F2^m\\{0}（PG(m-1,2) 列互异），d=3 锁死。
      - 权重 2 检测: 列互异 O(n²)
      - 权重 3 逻辑: (2^m-1)(2^m-2)/6 = PG(m-1,2) 线数
    """
    n = (1 << m) - 1
    k = n - 2 * m
    d = 3
    w3_logical = n * (n - 1) // 6
    return dict(family="PG", m=m, n=n, k=k, d=d, rate=k / n,
                w3_logical=w3_logical)


# ============ 简并比例闭式（10.33）============

def degeneracy_ratio(m, r):
    """简并比例 P_r(m)（10.33 定理）：

    P_r(m) = [flats(m,r+1)·E(r+1,2^r) + flats(m,r)] / C(2^m, 2^r)
    权重 2^r 层错误与更低层共享 syndrome 的比例（AG 完备码）。
    r=1,2 恒为 1（全简并）；r=3,m=8 为 ~1e-4。
    """
    w0 = 1 << r
    num = flats(m, r + 1) * E(r + 1, w0) + flats(m, r)
    den = comb(1 << m, w0)
    return float(Fraction(num, den))


def degeneracy_ratio_next(m, r):
    """次主阶简并比例 P'_r(m)（10.33）：

    P'_r(m) = flats(m,r+1)·C(2^{r+1}, 2^r+1) / C(2^m, 2^r+1)
    权重 2^r+1 层简并 → θ^{d+2} 次主阶系数。
    """
    w0 = 1 << r
    num = flats(m, r + 1) * comb(1 << (r + 1), w0 + 1)
    den = comb(1 << m, w0 + 1)
    return float(Fraction(num, den))


# ============ 逻辑算符计数闭式（定理 10.30.2.04）============

def logical_operator_count(m, r):
    """AG 完备码 CSS(RM(r,m)) 的权重 d=2^{r+1} 逻辑算符数（定理 10.30.2.04）。

    N_logic = 2^{m-r-1}·[m choose r+1]_2
            = 2^{m-r-1}·∏_{i=0}^{r} (2^{m-i}-1)/(2^{r+1-i}-1)
    即 AG(m,2) 中 (r+1)-维仿射平坦数。
    已验证：RM(1,5) → 1240；RM(1,6) → 10416（全量枚举吻合）。
    """
    gaussian = gb(m, r + 1)
    return (1 << (m - r - 1)) * gaussian


def pg_logical_count(m):
    """PG 完备码权重 d=3 逻辑算符数 = PG(m-1,2) 线数（10.28）。

    N = (2^m-1)(2^m-2)/6
    """
    n = (1 << m) - 1
    return n * (n - 1) // 6


# ============ 距离-噪声标度律 + 检测率闭式（10.31/10.29）============

def loss_exponent(d):
    """距离-噪声标度指数（定理 10.31.1.05）：loss ~ θ_max^(2⌈d/2⌉)

    完备码家族（PG/AG）通用：d=3 → 4（θ⁴，10.29 复现斜率 3.99），
    d=8 → 8（θ⁸，斜率 7.96）。
    """
    return 2 * ((d + 1) // 2)


def detection_rate(theta):
    """检测率闭式（10.29 预言 2a）：p_det(θ) = sin²(θ/2)

    对任意单比特 Pauli 型旋转注入，与码及 syndrome 线无关。
    程序验证偏差 < 3.8e-16。
    """
    import math as _m
    return _m.sin(theta / 2) ** 2


def miss_conditional_fidelity():
    """漏检无害（10.29 预言 2b）：漏检路径条件保真度恒 1

    未检测注入将态投影回码空间，不破坏逻辑信息。
    程序验证偏差 < 2.2e-16。
    """
    return 1.0


# ============ 开放问题闭式（10.30 §8 解答）============

def rm1_w2_degeneracy(m):
    """RM(1,m) 权重 2 简并类闭式（10.30 开放问题 1 解答）。

    类数 = 2^m − 1，每类大小 = 2^{m−1}（m=4..10 全量枚举验证）。
    类内共享差分向量 a⊕b（平行四边形结构）。守恒: C(2^m,2) = (2^m−1)·2^{m−1}。
    """
    return dict(classes=(1 << m) - 1, size_per_class=1 << (m - 1))


def rm_degeneracy_classes(m, r):
    """RM(r,m) 权重 2^r 层简并类结构闭式（10.30 开放问题 1 的 r≥1 通用化）。

    类 = 共享 syndrome 的权重 w0=2^r X 错误集（10.32 定理 10.32.1.01 包含性等价）。
    类结构由仿射包维数完全决定：

      (a) r-平坦类：A 本身是 r-平坦（仿射包恰 r 维）。
          类数 = [m r]_2（高斯二项），每类大小 = 2^{m−r}
          （= A 自身 1 + 含 A 的 (r+1)-平坦数 (2^{m−r}−1) 个补集伙伴 P∖A）。
      (b) (r+1)-仿射包类：A 仿射包恰 r+1 维（含于唯一 (r+1)-平坦 P=aff(A)）。
          伙伴唯一 B = P∖A，每类大小 = 2。
          类数 = flats(m,r+1)·E(r+1,2^r)/2。

    成员守恒：类数×大小 之和 = flats(m,r) + flats(m,r+1)·E(r+1,2^r)
              = 10.33 简并比例分子 P_r(m)·C(2^m, 2^r)（r≤2 全简并时 = C(2^m,2^r)；
              r=3 部分简并，m=5: 796700 < C(32,8)=10518300）。

    r=1 退化：E(2,2)=0（2 点子集仿射包 ≤1 维）→ 仅 (a) 类，类数 2^m−1、
    大小 2^{m−1}，精确回到 rm1_w2_degeneracy。
    验证：RM(1,4..6) 15/31/63 类，RM(2,4)=875、RM(2,5)=17515、RM(3,4)=6435，
    全量枚举一致（qec-geometry/scripts/verify_degeneracy_classes.py）。
    """
    w0 = 1 << r
    n_flat = gb(m, r)                      # (a) r-平坦类数
    size_flat = 1 << (m - r)               # (a) 每类大小 2^{m−r}
    n_aff = flats(m, r + 1) * E(r + 1, w0) // 2   # (b) (r+1)-仿射包类数
    total = n_flat + n_aff
    members = n_flat * size_flat + n_aff * 2
    return dict(family=f"RM({r},{m})", m=m, r=r, w0=w0,
                n_classes=total,
                n_flat_classes=n_flat, size_flat_class=size_flat,
                n_aff_classes=n_aff, size_aff_class=2,
                members=members,
                degenerate_ratio=Fraction(members, comb(1 << m, w0)),
                uniform=(n_aff == 0))


def ag_dminus1_syndrome(m, r):
    """AG 完备码权重 d−1 层 syndrome 分布闭式（10.30 开放问题 2 解答）。

    对 r=1（d=4, w=3）：类数 = 2^m，每类大小 = (2^m−1)(2^m−2)/6
    = PG(m−1,2) 线数 —— 两个完备族的深层对偶（m=4..6 全量枚举验证）。
    """
    if r != 1:
        raise NotImplementedError("r=1 已验证；r≥2 的 d−1 层分布见 m=5,r=2 探索（类大小 84/106/155）")
    n = 1 << m
    size = (n - 1) * (n - 2) // 6
    return dict(classes=n, size_per_class=size,
                pg_lines=size)  # = PG(m−1,2) 线数


class QECClosedForm:
    """AG 完备码族 [[2^m, k, 2^{r+1}]] 闭式纠错参数预测器（类封装 API）。

    与 pyqpanda-algorithm 的 QECClosedForm 模块接口一致（同步维护）：
    QECNoise（模拟验证 θ⁴）的预测层配套——闭式秒算 loss(θ)=c_d·θ^d。

    示例::

        cf = QECClosedForm(10, 3)          # [[1024, 672, 16]]
        cf.code()                          # (1024, 672, 16)
        cf.loss(0.01)                      # 1.05e-24
    """

    def __init__(self, m, r):
        self.m = m
        self.r = r
        p = ag_params(m, r)
        if p is None:
            raise ValueError(f"参数 m={m}, r={r} 给出非正逻辑比特数")
        self.n, self.k, self.d = p["n"], p["k"], p["d"]
        self.w0 = p["w0"]
        self.fail = p["fail"]
        self.kap = p["kap"]
        self.c_d = p["c_d"]

    def code(self):
        """返回码参数 (n, k, d)。"""
        return self.n, self.k, self.d

    def encoding_rate(self):
        """编码率 k/n。"""
        return self.k / self.n

    def zero_loss_boundary(self):
        """注入零损失边界 k_max = ⌊(d-1)/2⌋（定理 10.31.1.01）。"""
        return zero_loss_boundary(self.d)

    def loss(self, theta):
        """逻辑损失闭式 loss(θ) = c_d·θ^d（定理 10.35.1.07）。"""
        return loss_at_theta(self.c_d, self.d, theta)

    def logical_operator_count(self):
        """权重 d 逻辑算符数（定理 10.30.2.04）。"""
        return logical_operator_count(self.m, self.r)

    def degeneracy_classes(self):
        """权重 2^r 层简并类结构（rm_degeneracy_classes 通用闭式）。"""
        return rm_degeneracy_classes(self.m, self.r)

    @staticmethod
    def detection_rate(theta):
        """检测率闭式 p_det(θ) = sin²(θ/2)（10.29 预言 2a）。"""
        return detection_rate(theta)

    def summary(self):
        """返回一行可读的参数摘要。"""
        return (f"[[{self.n},{self.k},{self.d}]] rate={self.encoding_rate():.4f} "
                f"w0={self.w0} fail={self.fail:.4f} κ={self.kap:.4f} "
                f"c_d={self.c_d:.4g} zero-loss≤{self.zero_loss_boundary()} "
                f"logicals={self.logical_operator_count()}")


__all__ = ["ag_params", "pg_params", "zero_loss_boundary", "loss_at_theta",
           "degeneracy_ratio", "degeneracy_ratio_next", "logical_operator_count",
           "pg_logical_count", "loss_exponent", "detection_rate",
           "miss_conditional_fidelity", "rm1_w2_degeneracy",
           "rm_degeneracy_classes", "ag_dminus1_syndrome", "QECClosedForm"]
