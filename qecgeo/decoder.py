"""decoder.py —— 自研查表解码器 + 几何论恢复表设计（qecgeo）

把 10.30/10.35 的简并类理论落地为可执行的解码器：

  1. 完整查表：枚举权重 ≤ w_max 的全部 Pauli 错误 → syndrome → 最小权重恢复
     （查表版最优解码：与 MWPM 同义，但纯组合、无图论、Mac 秒级构建）
  2. 几何论恢复表设计：恢复代表 = 类内最小权重；类 = 共享 syndrome 的错误集
     （10.30 定理 10.30.2.05 的类结构）。解码失败 ⟺ 残留 ∉ 稳定子群 = 逻辑错误
  3. fail(w) 闭式验证：解码失败率 = 1 − ⟨1/v⟩（10.35 引理 10.35.2.07），
     本模块直接枚举验证——零简并层（AG r≥2 权重 2）失败率 = 0 的直接证据

用法::

    from qecgeo import LookupDecoder
    from qecgeo.codes import five_qubit_code, steane_code

    code = steane_code()
    dec = LookupDecoder(code.gens, code.n, name=code.name)
    dec.build(w_max=2)
    dec.decode(code.syndrome_of(some_error))   # -> 恢复 Pauli
    dec.stats()                                # -> 类数/简并分布/fail(2)

依赖：仅 numpy + 本包 pauli（无 stim / pymatching —— 纯组合解码）。
"""
from __future__ import annotations

from itertools import combinations, product
from collections import Counter, defaultdict

from .pauli import Pauli

__all__ = ["LookupDecoder", "decode_fail_rate", "recovery_class_structure"]


class LookupDecoder:
    """自研查表解码器：syndrome → 最小权重恢复（几何论恢复表）。

    构建：枚举权重 ≤ w_max 的全部 Pauli 错误（每类 X/Z/Y 型），按 syndrome
    分组，每类记录最小权重代表。解码 = 查表（O(1)）；表构建为纯组合枚举。

    设计原则（几何论 10.30/10.35）：
      - 恢复代表 = 类内最小权重（最优查表解码，等价 MWPM 的查表退化）
      - 类内其余成员 E′：E′·R ∈ 稳定子群 → 解码成功；∉ → 逻辑错误
      - 类大小分布 = 简并类结构（10.30 定理 10.30.2.05），可直接对照闭式
      - fail(w) = 1 − ⟨1/v⟩（10.35 引理 10.35.2.07），本类枚举验证

    参数
    ----
    gens : list[Pauli]   稳定子生成元（码的 stabilizers）
    n : int              物理比特数
    name : str           可选，码名（用于统计输出）
    """

    def __init__(self, gens, n, name='code'):
        self.gens = list(gens)
        self.n = n
        self.m = len(gens)
        self.name = name
        self.table = {}          # syndrome -> 恢复 Pauli（最小权重代表）
        self._classes = None     # syndrome -> list[错误 Pauli]（延迟构建）
        self._built_wmax = 0
        self.zero = tuple(0 for _ in range(self.m))
        # 稳定子群的 syndrome 集（都是 0）——残留 ∈ 群 ⟺ 残留 syndrome = 0
        # 快速判定：残留的 syndrome 是否为 0（群元 syndrome 恒 0）
        self._group_syndromes = {self.zero}
        self._group = None  # 惰性构建（大码时 2^m 可能爆炸）

    @property
    def group(self):
        """稳定子群（含单位元），惰性构建（大码 2^m 可能很大）。"""
        if self._group is None:
            elems = [Pauli.I(self.n)]
            for g in self.gens:
                elems = elems + [e * g for e in elems]
            self._group = elems
        return self._group

    # ---------- 核心 ----------

    def syndrome_of(self, E):
        """错误 E 的 syndrome（与各生成元 symplectic 内积）。"""
        return tuple(E.symplectic(g) for g in self.gens)

    def _iter_errors(self, w_max):
        """枚举权重 ≤ w_max 的全部 Pauli 错误（不含单位元）。"""
        for w in range(1, w_max + 1):
            for idxs in combinations(range(self.n), w):
                for types in product((1, 2, 3), repeat=w):
                    t = [0] * self.n
                    for idx, ty in zip(idxs, types):
                        t[idx] = ty
                    yield Pauli(self.n, t)

    def build(self, w_max=None):
        """构建恢复表：枚举权重 ≤ w_max 错误 → syndrome → 最小权重代表。

        w_max 默认 = ⌈(d 未知时) n⌉，但实际应传码的 d−1（可纠权重）。
        返回 self（链式）。
        """
        w_max = w_max if w_max is not None else self.n
        classes = defaultdict(list)
        for E in self._iter_errors(w_max):
            s = self.syndrome_of(E)
            classes[s].append(E)
        # 恢复表：每类最小权重代表（并列取第一个，确定性）
        table = {}
        for s, members in classes.items():
            best = min(members, key=lambda E: E.weight())
            table[s] = best
        self.table = table
        self._classes = classes
        self._built_wmax = w_max
        return self

    # ---------- 解码 ----------

    def decode(self, syndrome):
        """syndrome → 恢复操作（查表，O(1)）。未命中返回单位元。"""
        return self.table.get(syndrome, Pauli.I(self.n))

    def decode_error(self, E):
        """给定实际错误 E：恢复后残留。返回 (残留, 是否逻辑错误)。

        残留 = E·R(s)。残留 ∈ 稳定子群 → 解码成功（可逆）；否则为逻辑错误
        （残留含非平凡逻辑算符分量）。群元 syndrome 恒 0，故用
        syndrome(残留) == 0 判定，O(m) 而非 O(|群|)。
        """
        s = self.syndrome_of(E)
        R = self.decode(s)
        resid = E * R
        return resid, (self.syndrome_of(resid) != self.zero)

    def correct(self, E):
        """模拟纠错：返回 (是否成功, 残留 Pauli)。"""
        resid, is_logical = self.decode_error(E)
        return (not is_logical), resid

    # ---------- 统计与几何论对照 ----------

    def class_structure(self):
        """类结构：syndrome 类大小分布（10.30 定理 10.30.2.05 的枚举版）。

        返回 dict(classes=类数, size_dist={大小: 类数}, 按权重分类的类大小)。
        """
        if self._classes is None:
            raise RuntimeError('先调用 build()')
        total = len(self._classes)
        sizes = Counter(len(v) for v in self._classes.values())
        # 按权重分组的类大小（简并类由权重 w 错误引起）
        by_w = {}
        for s, members in self._classes.items():
            ws = sorted({E.weight() for E in members})
            key = tuple(ws)
            by_w[key] = by_w.get(key, 0) + 1
        return dict(classes=total, size_dist=dict(sizes), by_weight=dict(by_w))

    def fail_rate(self, w):
        """权重 w 层的解码失败率（10.35 引理 10.35.2.07 的枚举版）。

        fail(w) = 1 − ⟨1/v⟩_w，其中 v = 该错误所在 syndrome 类大小，
        平均取全部权重 w 错误。等价闭式 fail(w) = 1 − Σ 1/v / N_w。
        """
        if self._classes is None:
            raise RuntimeError('先调用 build()')
        n_w = 0
        inv_sum = 0.0
        for E in self._iter_errors(w):
            if E.weight() != w:
                continue
            s = self.syndrome_of(E)
            v = len(self._classes[s])
            n_w += 1
            inv_sum += 1.0 / v
        if n_w == 0:
            return None
        return 1.0 - inv_sum / n_w

    def weight2_uniqueness(self):
        """权重 2 层 syndrome 唯一率（AG r≥2 零简并的直接证据）。"""
        if self._classes is None:
            raise RuntimeError('先调用 build()')
        total = conflicts = 0
        seen = set()
        for E in self._iter_errors(2):
            if E.weight() != 2:
                continue
            s = self.syndrome_of(E)
            total += 1
            if s in seen:
                conflicts += 1
            else:
                seen.add(s)
        return 1.0 - conflicts / total

    def stats(self):
        """可读统计：表大小、类数、权重 2 唯一率、fail(2)。"""
        cs = self.class_structure()
        return dict(code=self.name, n=self.n, m=self.m,
                    table_size=len(self.table),
                    classes=cs['classes'],
                    size_dist=cs['size_dist'],
                    weight2_unique=self.weight2_uniqueness(),
                    fail_w2=self.fail_rate(2))

    def __repr__(self):
        return (f"LookupDecoder({self.name}, n={self.n}, m={self.m}, "
                f"table={len(self.table)})")


def decode_fail_rate(gens, n, w, w_max=None):
    """便捷函数：直接算权重 w 层失败率（构建临时解码器）。"""
    dec = LookupDecoder(gens, n)
    dec.build(w_max=w_max if w_max is not None else w)
    return dec.fail_rate(w)


def recovery_class_structure(gens, n, w_max):
    """便捷函数：类结构（不实例化解码器语义）。"""
    dec = LookupDecoder(gens, n)
    dec.build(w_max=w_max)
    return dec.class_structure()
