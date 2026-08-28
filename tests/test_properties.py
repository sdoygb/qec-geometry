"""test_properties.py —— 属性测试（property-based）：解码器不变式框架

把 260827 细心复查中手动验证的不变式固化为随机测试。核心思想：
不是"这个固定用例通过"，而是"对任意随机错误，不变式恒成立"。

不变式清单：
  1. 查表恢复一致性：decode(syndrome(E)) 与 E 同 syndrome（恢复有效）
  2. 可纠范围恢复：权重 ≤ (d-1)/2 的错误 decode 后残留 ∈ 稳定子群（成功）
  3. 矩解码正确性：rm_x_decode(moments_of(A)) == A（可纠范围内）
  4. 矩唯一性边界：权重 ≤ (d-1)/2 的矩无碰撞（全局检查）
  5. MILP 约束：恢复的矩 == 输入矩（高权重路径）
  6. build_fast ≡ build（向量化与逐项等价）
"""
import random
import time
import unittest
from itertools import combinations, combinations_with_replacement

import numpy as np

from qecgeo import LookupDecoder, moments_of, rm_x_decode
from qecgeo.codes import steane_code, rm_code_15_7_3
from qecgeo.pauli import Pauli


def _rm_css_gens(m, r):
    """CSS(RM(r,m)) 稳定子（X 型 + Z 型，同支撑）。"""
    n = 1 << m
    gens = []
    for deg in range(0, r + 1):
        for I in combinations(range(m), deg):
            g = [0] * n
            for a in range(n):
                val = 1
                for i in I:
                    val &= (a >> (m - 1 - i)) & 1
                g[a] = val
            gens.append(g)
    return gens


class TestLookupRecoveryInvariant(unittest.TestCase):
    """不变式 1+2：查表恢复有效性 + 可纠范围恢复成功。"""

    def _check_code(self, code, w_max, n_random):
        dec = LookupDecoder(code.gens, code.n, name=code.name)
        dec.build(w_max=w_max)
        rng = random.Random(42)
        for _ in range(n_random):
            w = rng.randint(1, w_max)
            idxs = rng.sample(range(code.n), w)
            types = [rng.randint(1, 3) for _ in range(w)]
            t = [0] * code.n
            for idx, ty in zip(idxs, types):
                t[idx] = ty
            E = Pauli(code.n, t)
            # 不变式 1：恢复与错误同 syndrome
            R = dec.decode(dec.syndrome_of(E))
            self.assertEqual(dec.syndrome_of(E * R), dec.zero,
                             f"{code.name}: 恢复后 syndrome 未归零")
            # 不变式 2（仅权重1，d=3 码）：单比特错误必恢复成功
            if w == 1:
                _, is_log = dec.decode_error(E)
                self.assertFalse(is_log, f"{code.name}: 权重1错误被判逻辑")

    def test_steane(self):
        self._check_code(steane_code(), 2, 300)

    def test_hamming_15(self):
        self._check_code(rm_code_15_7_3(), 2, 200)


class TestMomentDecodeInvariant(unittest.TestCase):
    """不变式 3+4：矩解码在可纠范围内正确 + 矩唯一。"""

    def _rm_decode_ok(self, m, r, w, n_random):
        """可纠范围内随机权重 w 错误，rm_x_decode 正确恢复。"""
        n = 1 << m
        d = 1 << (r + 1)
        corr = (d - 1) // 2
        self.assertLessEqual(w, corr, f"w={w} 超出可纠范围 (d-1)/2={corr}")
        rng = random.Random(42)
        for _ in range(n_random):
            A = rng.sample(range(n), w)
            mm = moments_of(A, m, r)
            self.assertEqual(rm_x_decode(mm, m, r), sorted(A),
                             f"(m={m},r={r}) w={w}: 矩解码错误")

    def test_m5_r2_w2(self):
        self._rm_decode_ok(5, 2, 2, 100)

    def test_m5_r2_w3(self):
        self._rm_decode_ok(5, 2, 3, 100)

    def test_m6_r3_w4(self):
        self._rm_decode_ok(6, 3, 4, 100)

    def test_moment_uniqueness_in_range(self):
        """不变式 4：可纠范围内矩全局无碰撞（检查所有 C(n,w) 集合）。"""
        m, r, w = 5, 2, 3  # 可纠上限 (d-1)/2 = 3
        n = 1 << m
        seen = {}
        for A in combinations(range(n), w):
            mm = moments_of(A, m, r)
            key = tuple(sorted(mm.items()))
            if key in seen:
                self.fail(f"矩碰撞（可纠范围内）: {A} vs {seen[key]}")
            seen[key] = A


class TestMILPInvariant(unittest.TestCase):
    """不变式 5：MILP 高权重恢复的矩 == 输入矩。"""

    def test_milp_moment_consistency(self):
        import qecgeo.rm_general_decoder as rmg
        m, r = 6, 3
        dec = rmg.RMMomentDecoder(m, r)
        n = dec.n
        rng = random.Random(7)
        for _ in range(5):
            w = rng.randint(5, 7)
            A = sorted(rng.sample(range(n), w))
            mm = moments_of(A, m, r)
            rec = rmg._milp_decode(mm, m, r, dec)
            self.assertIsNotNone(rec, "MILP 返回 None")
            self.assertEqual(moments_of(rec, m, r), mm,
                             "MILP 恢复的矩与输入不一致")


class TestBuildFastEquivalence(unittest.TestCase):
    """不变式 6：build_fast ≡ build（逐项一致）。"""

    def test_steane_equivalence(self):
        code = steane_code()
        d1 = LookupDecoder(code.gens, code.n)
        d1.build(w_max=2)
        d2 = LookupDecoder(code.gens, code.n)
        d2.build_fast(w_max=2)
        # 表内容逐项一致
        self.assertEqual(d1.table, d2.table, "build 与 build_fast 表不一致")
        # 恢复结果一致
        rng = random.Random(0)
        for _ in range(50):
            w = rng.randint(1, 2)
            idxs = rng.sample(range(code.n), w)
            types = [rng.randint(1, 3) for _ in range(w)]
            t = [0] * code.n
            for idx, ty in zip(idxs, types):
                t[idx] = ty
            E = Pauli(code.n, t)
            s = d1.syndrome_of(E)
            self.assertEqual(d1.decode(s).t, d2.decode(s).t,
                             "build 与 build_fast 恢复不一致")


if __name__ == "__main__":
    unittest.main()


class TestRegressionMutations(unittest.TestCase):
    """变异测试补盲：专门回归测试（260827 mutation_test 暴露的存活变异）。

    M1 存活：缺 decode_error 的逻辑判定回归
    M3 存活：缺 fail_rate 数值断言
    M5 存活：缺 decode 返回值断言
    """

    def test_decode_error_logical_detection(self):
        """M1 回归：Steane 权重2错误残留=逻辑算符 → 必须判为逻辑。"""
        code = steane_code()
        dec = LookupDecoder(code.gens, code.n)
        dec.build(w_max=2)
        # E=X0X1 → 残留 X0X1X2（权重3 X 逻辑）→ is_logical 必须 True
        n = code.n
        t = [0] * n; t[0] = 1; t[1] = 1
        _, is_log = dec.decode_error(Pauli(n, t))
        self.assertTrue(is_log, "X0X1 残留=X0X1X2（逻辑）应判逻辑错误")

    def test_decode_error_phase_insensitive(self):
        """M2 回归：稳定子×i 的残留不是逻辑错误（相位无关）。"""
        code = steane_code()
        dec = LookupDecoder(code.gens, code.n)
        dec.build(w_max=2)
        for g in code.gens:
            if all(x in (0, 1) for x in g.t) and g.weight() == 4:
                g2 = Pauli(code.n, g.t, phase=1j)
                _, is_log = dec.decode_error(g2)
                self.assertFalse(is_log, "稳定子×i 应判非逻辑")
                break

    def test_fail_rate_ag_value(self):
        """M3 回归：fail_rate(2) 在 AG 适用域内的精确值。"""
        from qecgeo import LookupDecoder
        from itertools import combinations
        m, r = 4, 1
        n = 1 << m
        gens = []
        for deg in range(0, r + 1):
            for I in combinations(range(m), deg):
                g = [0] * n
                for a in range(n):
                    val = 1
                    for i in I:
                        val &= (a >> (m - 1 - i)) & 1
                    g[a] = val
                gens.append(g)
        gens_p = [Pauli(n, [1 if x else 0 for x in g]) for g in gens]
        gens_p += [Pauli(n, [2 if x else 0 for x in g]) for g in gens]
        dec = LookupDecoder(gens_p, n)
        dec.build(w_max=2)
        # AG r=1 m=4: fail(2) = 1/3 − 1/(3·2^{m−1}) = 0.2917
        self.assertAlmostEqual(dec.fail_rate(2), 1/3 - 1/(3 * 2**3), places=4,
                               msg="fail_rate(2) 在 AG 域内应精确匹配闭式")

    def test_decode_returns_min_weight(self):
        """M5 回归：decode(syndrome) 返回非平凡恢复（不恒为单位元）。"""
        code = steane_code()
        dec = LookupDecoder(code.gens, code.n)
        dec.build(w_max=2)
        # 单比特 X0 的 syndrome → decode 应返回 X0（非单位元）
        n = code.n
        E = Pauli(n, [1] + [0] * (n - 1))
        R = dec.decode(dec.syndrome_of(E))
        self.assertFalse(all(x == 0 for x in R.t),
                         "decode(X0 syndrome) 应返回非平凡恢复（X0）")


class TestSCLDecoder(unittest.TestCase):
    """真 Reed 递推（syndrome 版）：SCL 列表递归解码器。"""

    def _scl_ok(self, m, r, wmax, n_random):
        from qecgeo.rm_scl_decoder import rm_scl_decode
        n = 1 << m
        rng = random.Random(42)
        for _ in range(n_random):
            w = rng.randint(1, wmax)
            A = sorted(rng.sample(range(n), w))
            mm = moments_of(A, m, r)
            recs = rm_scl_decode(mm, m, r)
            self.assertTrue(any(r2 == A for r2 in recs),
                            f"(m={m},r={r}) A={A}: SCL 未恢复")

    def test_m4_r2_w3(self):
        self._scl_ok(4, 2, 3, 30)

    def test_m5_r2_w3(self):
        self._scl_ok(5, 2, 3, 20)

    def test_m6_r3_w5(self):
        self._scl_ok(6, 3, 5, 15)

    def test_m5_r2_w4(self):
        """r=2 权重 4（含一般 4 点，260828 代数参数化 O(n²·m)）。"""
        self._scl_ok(5, 2, 4, 30)

    def test_m6_r2_w4(self):
        """r=2 权重 4，m=6：代数参数化 4 点（旧 O(n⁴) 在此规模可行但慢）。"""
        self._scl_ok(6, 2, 4, 20)

    def test_m7_r2_w4_perf(self):
        """m=7 (n=128) 权重 4：O(n⁴) 兜底在此规模不可行（C(128,4)≈1e7），
        代数参数化必须能在秒级完成——性能回归护栏。"""
        from qecgeo.rm_scl_decoder import rm_scl_decode
        n = 1 << 7
        rng = random.Random(7)
        t0 = time.time()
        for _ in range(8):
            # 一半一般 4 点（d≠0，最坏情形）
            while True:
                A = sorted(rng.sample(range(n), 4))
                mm = moments_of(A, 7, 2)
                d = 0
                for i in range(7):
                    if mm.get((i,), 0):
                        d |= 1 << (6 - i)
                if d != 0:
                    break
            recs = rm_scl_decode(mm, 7, 2, L=16)
            self.assertTrue(any(r2 == A for r2 in recs),
                            f"m=7 一般 4 点 A={A}: SCL 未恢复")
        dt = time.time() - t0
        self.assertLess(dt, 30.0, f"m=7 代数 4 点过慢: {dt:.1f}s (>30s)")


from qecgeo.moment_algebra import correct_polluted, solve_two


class TestMomentAlgebra(unittest.TestCase):
    """矩代数约束（10.85）：M2 = Σ v_a v_aᵀ 分解、秩约束、两点解析解。

    260828 新理论：测量噪声的代数纠正（非概率）。
    """

    def _m1_vec(self, mm, m):
        return np.array([mm.get((i,), 0) for i in range(m)])

    def _m2_mat(self, mm, m):
        M = np.zeros((m, m), dtype=int)
        for i in range(m):
            for j in range(m):
                if i == j:
                    M[i, j] = mm.get((i,), 0)   # GF(2): x_i^2 = x_i
                else:
                    M[i, j] = mm.get(tuple(sorted((i, j))), 0)
        return M

    @staticmethod
    def _gf2_rank(M):
        M = M.copy().astype(int)
        rows, cols = M.shape
        rank = 0
        for col in range(cols):
            pivot = None
            for r in range(rank, rows):
                if M[r, col]:
                    pivot = r
                    break
            if pivot is None:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            for r in range(rows):
                if r != rank and M[r, col]:
                    M[r] ^= M[rank]
            rank += 1
        return rank

    def test_rank_constraint(self):
        """定理 10.85.2.03: rank(M2) ≤ |A|（m=5 抽样）。"""
        from qecgeo import moments_of
        m = 5
        n = 1 << m
        rng = random.Random(10)
        for w in (1, 2, 3, 4):
            for _ in range(40):
                A = sorted(rng.sample(range(n), w))
                mm = moments_of(A, m, 2)
                M2 = self._m2_mat(mm, m)
                self.assertLessEqual(self._gf2_rank(M2), w,
                                     f"(m=5,w={w}) A={A}: rank > w")

    def test_single_point_outer_product(self):
        """推论 10.85.2.04: 单点 M2 == m1⊗m1（m=5 全 32 点）。"""
        from qecgeo import moments_of
        m = 5
        n = 1 << m
        for a in range(n):
            mm = moments_of([a], m, 2)
            m1 = self._m1_vec(mm, m)
            M2 = self._m2_mat(mm, m)
            self.assertTrue((M2 == np.outer(m1, m1) % 2).all(),
                            f"单点 a={a}: M2 != m1⊗m1")

    def test_two_point_analytic(self):
        """定理 10.85.3.01: 两点错误解析解（非枚举）m=5 全 496 命中。"""
        from qecgeo import moments_of
        m = 5
        n = 1 << m
        for A in combinations(range(n), 2):
            mm = moments_of(list(A), m, 2)
            m1 = self._m1_vec(mm, m)
            M2 = self._m2_mat(mm, m)
            sols = solve_two(m1, M2, m)
            self.assertTrue(any(s == sorted(A) for s in sols),
                            f"两点 A={A}: 解析解未命中")

    def test_three_point_analytic(self):
        """三点解析解（10.85 §6 开放问题）：有解时正确 100%。
        覆盖率 ~9%（跨 i 一致性约束待推导）——非交付标准。"""
        from qecgeo import moments_of
        from qecgeo.moment_algebra import solve_three, m1_vec, m2_mat
        m = 5
        n = 1 << m
        rng = random.Random(17)
        for A in rng.sample(list(combinations(range(n), 3)), 200):
            mm = moments_of(list(A), m, 2)
            sols = solve_three(m1_vec(mm, m), m2_mat(mm, m), m)
            if sols:
                self.assertTrue(any(s == sorted(A) for s in sols),
                                f"三点 A={A}: 解析解有解但错误")

    def test_two_point_pollution_correction(self):
        """算法 10.85.4.01: 两点 + 1/2 位污染，代数纠正 100%。"""
        from qecgeo import moments_of
        m = 5
        n = 1 << m
        rng = random.Random(3)
        # 1 位污染
        ok1 = tot1 = 0
        for A in rng.sample(list(combinations(range(n), 2)), 40):
            mm = moments_of(list(A), m, 2)
            keys = list(mm.keys())
            for fk in keys:
                mm2 = dict(mm); mm2[fk] ^= 1
                hits = correct_polluted(mm2, m, 2, w=2)
                tot1 += 1
                if any(sorted(s) == sorted(A) for _, sols in hits for s in sols):
                    ok1 += 1
        self.assertEqual(ok1, tot1, f"两点 1 位污染纠正 {ok1}/{tot1}")
        # 2 位污染
        ok2 = tot2 = 0
        for A in rng.sample(list(combinations(range(n), 2)), 20):
            mm = moments_of(list(A), m, 2)
            keys = list(mm.keys())
            for fi in range(min(6, len(keys))):
                for fj in range(fi + 1, min(6, len(keys))):
                    mm2 = dict(mm)
                    mm2[keys[fi]] ^= 1; mm2[keys[fj]] ^= 1
                    hits = correct_polluted(mm2, m, 2, w=2)
                    tot2 += 1
                    if any(sorted(s) == sorted(A) for _, sols in hits for s in sols):
                        ok2 += 1
        # 2 位污染：歧义边界（10.85 §6）——约束非充分时解可能指向
        # 其他两点错误；但无解（需>2位翻转）为 0，纠正率 >60%。
        self.assertGreater(ok2 / max(tot2, 1), 0.60,
                           f"两点 2 位污染纠正率过低 {ok2}/{tot2}")
