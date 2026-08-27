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

    def build_fast(self, w_max=2):
        """向量化恢复表构建（numpy 批量 syndrome，26× 加速）。

        与 build 等价（表逐项一致），但 syndrome 用矩阵乘法批量计算：
          syndrome = (Ex @ Sz^T + Ez @ Sx^T) mod 2
        错误按权重升序枚举 → 首次出现的 syndrome 即最小权重代表。
        实测：[[128,70,8]] 119.6s → 4.6s；[[64,20,8]] 11.4s → 0.6s。
        """
        import numpy as np
        from itertools import combinations, product
        n, ns = self.n, self.m
        # 稳定子矩阵
        Sx = np.zeros((ns, n), dtype=np.int8)
        Sz = np.zeros((ns, n), dtype=np.int8)
        for a, g in enumerate(self.gens):
            for i in range(n):
                if g.t[i] in (1, 3):
                    Sx[a, i] = 1
                if g.t[i] in (2, 3):
                    Sz[a, i] = 1
        # 枚举全部错误（权重升序 → 首次出现 = 最小权重代表）
        rows = []
        for w in range(1, w_max + 1):
            for idxs in combinations(range(n), w):
                for types in product((1, 2, 3), repeat=w):
                    t = [0] * n
                    for idx, ty in zip(idxs, types):
                        t[idx] = ty
                    x = [1 if v in (1, 3) else 0 for v in t]
                    z = [1 if v in (2, 3) else 0 for v in t]
                    rows.append((x, z))
        N = len(rows)
        Ex = np.array([r[0] for r in rows], dtype=np.int8)  # (N, n)
        Ez = np.array([r[1] for r in rows], dtype=np.int8)
        synd = (Ex @ Sz.T + Ez @ Sx.T) & 1                   # (N, ns)
        # 首次出现（np.unique 返回每个 syndrome 首次出现的行索引）
        shifts = np.array([1 << i for i in range(ns)], dtype=np.int64)
        idx = (synd.astype(np.int64) * shifts).sum(axis=1)
        uniq, first_idx = np.unique(idx, return_index=True)
        # 构建表（首次出现行 = 最小权重恢复）
        table = {}
        for j, k in enumerate(first_idx):
            xr, zr = rows[int(k)]
            t = [0] * n
            for i in range(n):
                if xr[i] and zr[i]:
                    t[i] = 3
                elif xr[i]:
                    t[i] = 1
                elif zr[i]:
                    t[i] = 2
            row_synd = synd[int(k)]
            table[tuple(int(b) for b in row_synd)] = Pauli(n, t)
        self.table = table
        # _classes 仅记录首次出现（fail_rate/class_structure 用 len(_classes[s]) 需完整类
        # —— 此处保留惰性：仅当调用 fail_rate 时重建完整类（罕见）
        self._classes = None
        self._built_wmax = w_max
        return self

    def _ensure_classes(self):
        """惰性构建完整 syndrome 类（build_fast 后按需，build 后跳过）。"""
        if self._classes is None:
            w_max = self._built_wmax if self._built_wmax > 0 else self.n
            classes = defaultdict(list)
            for E in self._iter_errors(w_max):
                classes[self.syndrome_of(E)].append(E)
            self._classes = classes

    # ---------- 解码 ----------

    def decode(self, syndrome):
        """syndrome → 恢复操作（查表，O(1)）。未命中返回单位元。"""
        return self.table.get(syndrome, Pauli.I(self.n))

    def _ensure_group(self):
        """构建稳定子群（惰性）。群大小 2^m 可能爆炸——仅当需要精确判定时调用。

        保护（260827）：m > 20（群 > 100 万）时拒绝构建，decode_error 需
        回退到 syndrome==0 近似（见 decode_error 的说明）。
        """
        if self._group is None:
            if self.m > 20:
                raise MemoryError(
                    f'稳定子群大小 2^{self.m} 过大，无法精确判定。'
                    f'decode_error/correct 对大码(m>20)请用 syndrome==0 近似或逐错误处理。')
            group = [Pauli.I(self.n)]
            for g in self.gens:
                group = group + [e * g for e in group]
            self._group = group
        return self._group

    def in_group(self, E):
        """精确群成员判定（残留 ∈ 稳定子群，忽略相位）。

        注意（260827 修复）：稳定子群的物理元素可带 ±1/±i 相位（Pauli
        乘积的全局相位），`E == s` 的相位精确比较会漏掉"稳定子×相位"的
        等价残留（误判为逻辑错误）。正确判定 = 忽略相位的 t-vector 比较：
        残留的支撑 pattern 是某稳定子的支撑 ⟺ 物理上可逆（相位无关）。
        O(|群|)——小码可用，大码慎用（群大小 2^m 爆炸）。
        """
        return any(E.t == s.t for s in self._ensure_group())

    def decode_error(self, E):
        """给定实际错误 E：恢复后残留。返回 (残留, 是否逻辑错误)。

        残留 = E·R(s)。残留 ∈ 稳定子群 → 解码成功（可逆）；否则为逻辑错误
        （残留含非平凡逻辑算符分量）。

        注意（260827 修复）：syndrome(残留)==0 只说明残留 ∈ normalizer
        （与全部生成元对易），**不等于**残留 ∈ 稳定子群——残留可能是非平凡
        逻辑算符（如 Steane 的权重 3 X 逻辑）。正确判定 = 精确群成员检查：
        残留 ∈ group ⟺ 解码成功。代价 O(|群|)=2^m（小码可行；大码建议用
        残留 syndrome==0 的快速近似，并知晓误报风险）。
        """
        s = self.syndrome_of(E)
        R = self.decode(s)
        resid = E * R
        if self.syndrome_of(resid) != self.zero:
            return resid, True  # 残留仍触发 syndrome → 明确逻辑错误
        # syndrome==0：残留 ∈ normalizer。区分"群元"（成功）vs"非平凡逻辑"（失败）
        try:
            return resid, not self.in_group(resid)
        except MemoryError:
            # 大码（m>20）无法构建群：回退 syndrome==0 近似（残留 syndrome 0
            # 视为群元——可能漏报逻辑错误，仅在无法精确判定时使用）
            return resid, False

    def correct(self, E):
        """模拟纠错：返回 (是否成功, 残留 Pauli)。"""
        resid, is_logical = self.decode_error(E)
        return (not is_logical), resid

    # ---------- 统计与几何论对照 ----------

    def class_structure(self):
        """类结构：syndrome 类大小分布（10.30 定理 10.30.2.05 的枚举版）。

        返回 dict(classes=类数, size_dist={大小: 类数}, 按权重分类的类大小)。
        """
        self._ensure_classes()
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

        适用域（260827 复核）：v 取**全类大小**（含跨层成员），故该公式仅
        在 AG 偶距离码的主阶层（同层简并、无跨层共享）与真实失败率一致
        （实测 AG r=1 [[16,6,4]] fail(2)=0.2917 精确匹配）；对 d 为奇数的码
        （Steane/五比特，权重 2 与权重 1 跨层共享）或类内相差稳定子的码
        （Shor）失真（方向不定）。真实失败率用 decode_error/correct 枚举。
        """
        if w > self._built_wmax:
            raise ValueError(
                f'fail_rate({w}) 超出已构建 w_max={self._built_wmax}——先 build(w_max≥{w})')
        self._ensure_classes()
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
        if self._built_wmax < 2:
            raise RuntimeError('先调用 build(w_max≥2) 或 build_fast(w_max≥2)')
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
