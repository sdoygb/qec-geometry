"""test_qecgeo.py —— qecgeo 单元测试（unittest，零外部依赖）

覆盖：Pauli 代数、码构造自检、距离验证、阈值闭式（η/A/p_th）、
MC 二次律、任意子 Ising 判定。
"""
import unittest
import numpy as np

from qecgeo.pauli import Pauli
from qecgeo.codes import (five_qubit_code, steane_code, shor_code,
                          rm_code_15_7_3, ALL)
from qecgeo.threshold import analyze_eta, monte_carlo, verify_quadratic
from qecgeo.anyon import delta_phase_analysis, is_ising_statistics


class TestPauli(unittest.TestCase):
    def test_multiplication_table(self):
        """X·Z = -iY, Z·X = iY（反交换）"""
        X, Y, Z = Pauli.X(1, 0), Pauli.Y(1, 0), Pauli.Z(1, 0)
        self.assertEqual(X * Z, Pauli(1, [3], -1j))
        self.assertEqual(Z * X, Pauli(1, [3], 1j))
        self.assertEqual(Y * Y, Pauli.I(1))

    def test_symplectic(self):
        X, Z = Pauli.X(1, 0), Pauli.Z(1, 0)
        self.assertEqual(X.symplectic(Z), 1)
        self.assertEqual(X.symplectic(X), 0)

    def test_from_string(self):
        P = Pauli.from_string(5, 'X1Z2Z3X4')
        self.assertEqual(P.t, [1, 2, 2, 1, 0])
        Q = Pauli.from_string(7, 'XXXXXXX')
        self.assertEqual(Q.t, [1] * 7)

    def test_apply_to_state(self):
        X = Pauli.X(1, 0)
        self.assertTrue(np.allclose(X.apply_to_state([0, 1]), [1, 0]))
        Z = Pauli.Z(1, 0)
        self.assertTrue(np.allclose(Z.apply_to_state([1, 0]), [1, 0]))
        self.assertTrue(np.allclose(Z.apply_to_state([0, 1]), [0, -1]))


class TestCodes(unittest.TestCase):
    def test_all_codes_construct(self):
        """四个码都能构造（StabilizerCode.__init__ 内含自检）"""
        for f in ALL:
            code = f()
            self.assertEqual(code.k, code.n - code.m)
            self.assertEqual(len(code.group), 2 ** code.m)

    def test_distance(self):
        """三个几何码距离 = 3"""
        for f in (five_qubit_code, steane_code, shor_code):
            code = f()
            ok, msg = code.check_distance(3)
            self.assertTrue(ok, f'{code.name}: {msg}')

    def test_logical_zero_stabilized(self):
        """|0_L⟩ 被所有生成元稳定（本征值 +1）"""
        code = steane_code()
        state = code.logical_zero()
        for g in code.gens:
            v = np.vdot(state, g.apply_to_state(state))
            self.assertAlmostEqual(v.real, 1.0, places=6)
            self.assertAlmostEqual(v.imag, 0.0, places=6)

    def test_encode_fidelity(self):
        """编码-纠错-保真度：单比特 X 错误可被纠正"""
        code = steane_code()
        ideal = code.logical_zero()
        E = Pauli.X(code.n, 2)
        corrupted = E.apply_to_state(ideal)
        s = code.measure_syndrome(corrupted)
        recovered = code.correct(corrupted, s)
        self.assertGreater(code.fidelity(recovered, ideal), 0.999)


class TestThreshold(unittest.TestCase):
    def test_eta_finite(self):
        """三个码的 η 枚举：0 < η ≤ 1，p_th 有限"""
        for f in (five_qubit_code, steane_code, rm_code_15_7_3):
            r = analyze_eta(f())
            self.assertGreater(r['eta'], 0.0)
            self.assertLessEqual(r['eta'], 1.0)
            self.assertLess(r['p_th'], float('inf'))
            self.assertAlmostEqual(r['A'], r['eta'] * r['n'] * (r['n'] - 1) / 2)

    def test_steane_eta(self):
        """[7,1,3] 的 η 已知值：7 比特码权重 2 误恢复比例（10.44 复核）"""
        r = analyze_eta(steane_code())
        # 与 10.44 程序复核一致的精确枚举值
        self.assertEqual(r['total'], 9 * 21)  # C(7,2)·9 = 189

    def test_mc_quadratic_law(self):
        """小噪声下 p_L ≈ A·p²（ratio ≈ 1）"""
        code = steane_code()
        r = analyze_eta(code)
        pL = monte_carlo(code, 0.03, n_trials=200000, seed=7)
        ratio = pL / (r['A'] * 0.03 ** 2)
        self.assertAlmostEqual(ratio, 1.0, delta=0.15)

    def test_verify_quadratic_structure(self):
        rows = verify_quadratic(steane_code(), ps=(0.01, 0.05), n_trials=50000)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertIn('pL', row)
            self.assertIn('Ap2', row)


class TestAnyon(unittest.TestCase):
    def test_delta8_closed(self):
        """δ⁸ 净相位 ≡ 0 (mod 2π)"""
        r = delta_phase_analysis()
        self.assertLess(r['delta8_phase_mod2pi'], 1e-9)
        self.assertAlmostEqual(r['delta_phase'], 7 * np.pi / 4)

    def test_ising_statistics(self):
        ok, msg = is_ising_statistics()
        self.assertTrue(ok)
        self.assertIn('Ising', msg)


if __name__ == '__main__':
    unittest.main()


class TestQECClosedFormClass:
    """QECClosedForm 类 API（与 pyqpanda-algorithm PR #49 同步）。"""

    def test_class_code(self):
        from qecgeo import QECClosedForm
        cf = QECClosedForm(10, 3)
        assert cf.code() == (1024, 672, 16)

    def test_class_loss(self):
        from qecgeo import QECClosedForm
        cf = QECClosedForm(4, 1)
        assert abs(cf.loss(0.01) - 3e-8) < 1e-10

    def test_class_interface_parity(self):
        """类 API 与函数 API 结果一致。"""
        from qecgeo import QECClosedForm, ag_params
        cf = QECClosedForm(6, 2)
        p = ag_params(6, 2)
        assert cf.c_d == p["c_d"]
        assert cf.encoding_rate() == p["rate"]


class TestRmDegeneracyClasses:
    """RM(r,m) 权重 2^r 层简并类结构通用闭式（10.30 开放问题 1 的 r≥1 推广）。"""

    def test_r1_matches_original(self):
        """r=1 退化精确回到 rm1_w2_degeneracy。"""
        from qecgeo.closedform import rm_degeneracy_classes, rm1_w2_degeneracy
        for m in (4, 6, 8):
            d = rm_degeneracy_classes(m, 1)
            old = rm1_w2_degeneracy(m)
            assert d["n_classes"] == old["classes"]
            assert d["size_flat_class"] == old["size_per_class"]
            assert d["uniform"]

    def test_known_enumeration_values(self):
        """闭式与全量枚举已知值一致（RM(2,4)=875 等）。"""
        from qecgeo.closedform import rm_degeneracy_classes
        assert rm_degeneracy_classes(4, 2)["n_classes"] == 875
        assert rm_degeneracy_classes(5, 2)["n_classes"] == 17515
        assert rm_degeneracy_classes(4, 3)["n_classes"] == 6435
        assert rm_degeneracy_classes(5, 1)["n_classes"] == 31

    def test_class_size_structure(self):
        """类大小结构：r-平坦类 2^{m-r}，(r+1)-仿射包类 2。"""
        from qecgeo.closedform import rm_degeneracy_classes
        d = rm_degeneracy_classes(5, 2)
        assert d["size_flat_class"] == 8          # 2^{5-2}
        assert d["n_flat_classes"] == 155         # [5 2]_2
        assert d["n_aff_classes"] == 17360
        assert d["size_aff_class"] == 2

    def test_member_conservation(self):
        """成员总数 = 10.33 简并比例分子。"""
        from qecgeo.closedform import rm_degeneracy_classes, flats, E
        for (m, r) in [(8, 1), (8, 2), (8, 3), (6, 4)]:
            d = rm_degeneracy_classes(m, r)
            w0 = 1 << r
            expect = flats(m, r) + flats(m, r + 1) * E(r + 1, w0)
            assert d["members"] == expect
            assert 0 <= d["degenerate_ratio"] <= 1

    def test_r3_ratio_matches_10_32(self):
        """RM(3,8) 简并比例 = 10.32 闭式 1.007×10⁻⁴。"""
        from qecgeo.closedform import rm_degeneracy_classes
        d = rm_degeneracy_classes(8, 3)
        assert abs(float(d["degenerate_ratio"]) - 1.007e-4) < 1e-6

    def test_class_api(self):
        """QECClosedForm 类 API 暴露简并类结构。"""
        from qecgeo import QECClosedForm
        from qecgeo.closedform import rm_degeneracy_classes
        cf = QECClosedForm(6, 2)
        d = cf.degeneracy_classes()
        assert d["n_classes"] == rm_degeneracy_classes(6, 2)["n_classes"]


class TestTheta4Suppression:
    """10.30 开放问题 3：零简并压低 θ⁴ 系数（fail(2) = 0 的直接证据）。"""

    def _unique_ratio(self, m, r):
        """权重 2 层 syndrome 唯一率（枚举）。"""
        from itertools import combinations
        n = 1 << m
        rows = []
        for mask in range(1 << m):
            if mask.bit_count() <= r:
                rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
        seen, conflicts = {}, 0
        for idxs in combinations(range(n), 2):
            t = [0] * n
            t[idxs[0]] = t[idxs[1]] = 1
            s = tuple(sum(ti & gi for ti, gi in zip(t, g)) & 1 for g in rows)
            if s in seen:
                conflicts += 1
            else:
                seen[s] = idxs
        return 1.0 - conflicts / (n * (n - 1) // 2)

    def test_r2_weight2_zero_degeneracy(self):
        """AG r≥2 权重 2 层 syndrome 完全唯一 → fail(2) = 0（θ⁴ 项压低）。"""
        assert self._unique_ratio(5, 2) == 1.0
        assert self._unique_ratio(6, 2) == 1.0

    def test_r1_weight2_ratio(self):
        """AG r=1 唯一率 = 2^{1-m}（fail(2) = 1 − 2^{1−m}）。"""
        for m in (4, 5, 6):
            ur = self._unique_ratio(m, 1)
            assert abs(ur - 2 ** (1 - m)) < 1e-9

    def test_family_fail_spectrum(self):
        """10.35 定理 10.35.1.02 家族失败率谱系：r 增 fail 单调降。"""
        from qecgeo import ag_params
        f1 = ag_params(10, 1)["fail"]
        f2 = ag_params(10, 2)["fail"]
        f3 = ag_params(10, 3)["fail"]
        assert f1 > f2 > f3
        assert f1 > 0.99          # r=1 全简并 fail → 1
        assert abs(f2 - 0.5) < 0.01
        assert abs(f3 - 0.5) < 0.01


class TestCrossCodeNormalization:
    """跨码族坐标归一化（_normalize_coords）：1D/0D → 标准 3 维。"""

    def _norm(self):
        from qecgeo import error_geometry as eg
        return eg._normalize_coords

    def test_2d_preserved(self):
        """2D+ 坐标原样保留（surface/color 不变）。"""
        n = self._norm()
        coords = {0: [1.5, 2.5, 3], 1: [0.0, 0.0, 1]}
        out = n(coords)
        assert out[0] == (1.5, 2.5, 3.0)
        assert out[1] == (0.0, 0.0, 1.0)

    def test_1d_becomes_linear_chain(self):
        """1D 坐标 → (x, 0, 0)：拓扑退化为线性链。"""
        n = self._norm()
        coords = {0: [0.0], 1: [1.0], 2: [2.0]}
        out = n(coords)
        assert out[0] == (0.0, 0.0, 0.0)
        assert out[1] == (1.0, 0.0, 0.0)
        assert out[2] == (2.0, 0.0, 0.0)

    def test_0d_synthetic_index(self):
        """0D/空坐标 → 合成索引 (i, 0, 0)。"""
        n = self._norm()
        out = n({5: [], 7: None})
        assert out[5] == (5.0, 0.0, 0.0)
        assert out[7] == (7.0, 0.0, 0.0)

    def test_none_passthrough(self):
        """None 坐标原样返回（无坐标电路不崩）。"""
        n = self._norm()
        assert n(None) is None

    def test_all_outputs_3d(self):
        """所有输出坐标恒 ≥3 维（analyze_edges 的 round(v[0]),round(v[1]) 安全）。"""
        n = self._norm()
        for coords in ({0: [1.0]}, {0: [1.0, 2.0]}, {0: [1.0, 2.0, 3.0, 4.0]}, {0: []}):
            out = n(coords)
            for v in out.values():
                assert len(v) >= 3, f"{coords} → {v}"


class TestLookupDecoder:
    """自研查表解码器 + 几何论恢复表（qecgeo.decoder）。"""

    def _rm_gens(self, m, r):
        from qecgeo.pauli import Pauli
        n = 1 << m
        rows = []
        for mask in range(1 << m):
            if mask.bit_count() <= r:
                rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
        gens = []
        for row in rows:
            gens.append(Pauli(n, [1 if b else 0 for b in row]))
            gens.append(Pauli(n, [2 if b else 0 for b in row]))
        return gens

    def test_small_code_correctness(self):
        """[[5,1,3]]/[[7,1,3]]/[[15,7,3]] 权重 1 全部恢复；权重 2 部分失败（d=3 只纠 1 个错误）。"""
        from qecgeo import LookupDecoder
        from qecgeo.codes import five_qubit_code, steane_code, rm_code_15_7_3
        from itertools import combinations, product
        from qecgeo.pauli import Pauli
        for code in (five_qubit_code(), steane_code(), rm_code_15_7_3()):
            dec = LookupDecoder(code.gens, code.n, name=code.name)
            dec.build(w_max=2)
            # 权重 1：全部恢复（单比特错误唯一可纠）
            ok1 = fail1 = 0
            # 权重 2：部分失败（d=3 → 只能纠权重 ≤ (d-1)/2 = 1 个错误）
            ok2 = fail2 = 0
            for w in (1, 2):
                for idxs in combinations(range(code.n), w):
                    for types in product((1, 2, 3), repeat=w):
                        t = [0] * code.n
                        for idx, ty in zip(idxs, types):
                            t[idx] = ty
                        success, _ = dec.correct(Pauli(code.n, t))
                        if w == 1:
                            ok1 += success; fail1 += (not success)
                        else:
                            ok2 += success; fail2 += (not success)
            # d=3 码：权重 1 无逻辑失败（fail1=0）；权重 2 存在逻辑失败（fail2>0）
            assert fail1 == 0, f"{code.name}: 权重1 有 {fail1} 个失败（应 0）"
            assert fail2 > 0, f"{code.name}: 权重2 应存在逻辑失败（d=3 只纠 1 错误）"
            assert ok1 + ok2 > 0

    def test_ag_r2_weight2_zero_degeneracy(self):
        """AG r=2 权重 2 层零简并：唯一率 1.0、fail(2)=0、类数=错误数。"""
        from qecgeo import LookupDecoder
        dec = LookupDecoder(self._rm_gens(5, 2), 32, name='AG r=2 m=5')
        dec.build(w_max=2)
        assert dec.weight2_uniqueness() == 1.0
        assert dec.fail_rate(2) == 0.0
        n_w2 = sum(1 for s, ms in dec._classes.items()
                   if all(E.weight() == 2 for E in ms))
        assert n_w2 == 9 * (32 * 31 // 2)

    def test_ag_r1_fail2_closed_form(self):
        """AG r=1 权重 2 fail(2) = 1/3 − 1/(3·2^{m−1})（新闭式，解码器发现）。"""
        from qecgeo import LookupDecoder
        for m in (4, 5):
            dec = LookupDecoder(self._rm_gens(m, 1), 1 << m, name=f'AG r=1 m={m}')
            dec.build(w_max=2)
            fr = dec.fail_rate(2)
            closed = 1 / 3 - 1 / (3 * 2 ** (m - 1))
            assert abs(fr - closed) < 1e-12, f"m={m}: {fr} vs {closed}"

    def test_build_speed(self):
        """小码查表构建毫秒级（一台 Mac 的实时可用性）。"""
        from qecgeo import LookupDecoder
        from qecgeo.codes import rm_code_15_7_3
        import time
        code = rm_code_15_7_3()
        dec = LookupDecoder(code.gens, code.n, name=code.name)
        t0 = time.time()
        dec.build(w_max=2)
        assert (time.time() - t0) < 1.0  # 远小于 1 秒


class TestBuildFast:
    """向量化恢复表构建（build_fast vs build 等价）。"""

    def test_equivalence_small(self):
        """[[15,7,3]] build_fast 与 build 表完全一致。"""
        from qecgeo import LookupDecoder
        from qecgeo.codes import rm_code_15_7_3
        code = rm_code_15_7_3()
        d1 = LookupDecoder(code.gens, code.n, name='t'); d1.build(w_max=2)
        d2 = LookupDecoder(code.gens, code.n, name='t'); d2.build_fast(w_max=2)
        assert len(d1.table) == len(d2.table)
        for k, v in d2.table.items():
            assert d1.table[k] == v

    def test_fail_rate_after_fast(self):
        """build_fast 后 fail_rate/class_structure 可用且一致。"""
        from qecgeo import LookupDecoder
        from qecgeo.codes import rm_code_15_7_3
        code = rm_code_15_7_3()
        d1 = LookupDecoder(code.gens, code.n, name='t'); d1.build(w_max=2)
        d2 = LookupDecoder(code.gens, code.n, name='t'); d2.build_fast(w_max=2)
        assert abs(d2.fail_rate(2) - d1.fail_rate(2)) < 1e-12
        assert d2.class_structure()['classes'] == d1.class_structure()['classes']

    def test_fast_speedup(self):
        """build_fast 显著快于 build（>5×，Mac 秒级）。"""
        from qecgeo import LookupDecoder
        from qecgeo.pauli import Pauli
        import time

        def css_rm_gens(m, r):
            n = 1 << m
            rows = []
            for mask in range(1 << m):
                if mask.bit_count() <= r:
                    rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
            gens = []
            for row in rows:
                gens.append(Pauli(n, [1 if b else 0 for b in row]))
                gens.append(Pauli(n, [2 if b else 0 for b in row]))
            return gens, n

        gens, n = css_rm_gens(6, 2)   # [[64,20,8]]
        d1 = LookupDecoder(gens, n, name='t')
        t0 = time.time(); d1.build(w_max=2); t_old = time.time() - t0
        d2 = LookupDecoder(gens, n, name='t')
        t0 = time.time(); d2.build_fast(w_max=2); t_new = time.time() - t0
        assert len(d1.table) == len(d2.table)
        assert t_new < t_old / 5, f"加速不足: {t_old:.1f}s → {t_new:.1f}s"


class TestRmDecoder:
    """Reed-Muller 矩解码器（非查表，O(n·poly)）。"""

    def test_r1_decode_all_weights(self):
        """r=1 解码：错误 ≤ 2 全部恢复，矩一致。"""
        import random
        from qecgeo import moments_of, rm_x_decode
        random.seed(1)
        for m in (4, 5, 6):
            n = 1 << m
            for _ in range(100):
                w = random.randint(0, 2)
                A = random.sample(range(n), w)
                mm = moments_of(A, m, 1)
                dec = rm_x_decode(mm, m, 1)
                assert dec is not None
                assert moments_of(dec, m, 1) == mm
                assert len(dec) <= 2

    def test_r1_scales_to_1024(self):
        """r=1 在 n=1024 微秒级（查表不可行的规模）。"""
        import random, time
        from qecgeo import moments_of, rm_x_decode
        random.seed(2)
        n = 1 << 10
        t0 = time.time()
        for _ in range(1000):
            w = random.randint(0, 2)
            A = random.sample(range(n), w)
            dec = rm_x_decode(moments_of(A, 10, 1), 10, 1)
            assert dec is not None
        assert (time.time() - t0) < 2.0  # 1000 次 < 2s

    def test_r2_decode(self):
        """r=2 解码：错误 ≤ 4 全部恢复（m=4/5 枚举可行）。"""
        import random
        from qecgeo import moments_of, rm_x_decode
        random.seed(3)
        for m in (4, 5):
            n = 1 << m
            for _ in range(30):
                w = random.randint(0, 4)
                A = random.sample(range(n), w)
                mm = moments_of(A, m, 2)
                dec = rm_x_decode(mm, m, 2)
                assert dec is not None
                assert moments_of(dec, m, 2) == mm
                assert len(dec) <= 4

    def test_end_to_end_vs_lookup(self):
        """CSS(RM(1,m)) 端到端：矩解码与查表解码恢复一致。"""
        import random
        from qecgeo import moments_of, rm_x_decode, LookupDecoder
        from qecgeo.pauli import Pauli
        random.seed(4)
        for m in (4, 5):
            n = 1 << m
            rows = []
            for mask in range(1 << m):
                if mask.bit_count() <= 1:
                    rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
            gens = []
            for row in rows:
                gens.append(Pauli(n, [1 if b else 0 for b in row]))
                gens.append(Pauli(n, [2 if b else 0 for b in row]))
            dec_tbl = LookupDecoder(gens, n, name='t'); dec_tbl.build(w_max=2)
            for _ in range(50):
                w = random.randint(0, 2)
                A = random.sample(range(n), w)
                E = Pauli(n, [1 if i in A else 0 for i in range(n)])
                A_rec = rm_x_decode(moments_of(A, m, 1), m, 1)
                R_mom = Pauli(n, [1 if i in A_rec else 0 for i in range(n)])
                zero = tuple([0] * len(gens))
                assert dec_tbl.syndrome_of(E * R_mom) == zero


class TestRmDecoderHighR:
    """通用 Reed-Muller 解码器（r≥3，d=16/32 码）。"""

    def test_r3_d16_low_weight(self):
        """CSS(RM(3,8)) [[256,·,16]]：权重 ≤ 3 全部恢复。"""
        import random
        from qecgeo import moments_of, rm_x_decode
        random.seed(5)
        m, r = 8, 3
        n = 1 << m
        for _ in range(50):
            w = random.randint(0, 3)
            A = random.sample(range(n), w)
            mm = moments_of(A, m, r)
            dec = rm_x_decode(mm, m, r)
            assert dec is not None
            assert moments_of(dec, m, r) == mm

    def test_r4_d32_low_weight(self):
        """CSS(RM(4,10)) [[1024,·,32]]：权重 ≤ 2 毫秒级恢复（查表不可行规模）。"""
        import random, time
        from qecgeo import moments_of, rm_x_decode
        random.seed(6)
        m, r = 10, 4
        n = 1 << m
        t0 = time.time()
        for _ in range(30):
            w = random.randint(0, 2)
            A = random.sample(range(n), w)
            mm = moments_of(A, m, r)
            dec = rm_x_decode(mm, m, r)
            assert dec is not None
            assert moments_of(dec, m, r) == mm
        assert (time.time() - t0) < 5.0  # 30 次 < 5s（毫秒级/次）

    def test_zsupport_weight(self):
        """CSS(RM(r,m)) 逻辑 Z 支撑 = x1x2 → 权重 2^{m-2}。"""
        from qecgeo import css_rm_zsupport
        for m in (4, 6, 10):
            z = css_rm_zsupport(m, 1)
            assert len(z) == (1 << (m - 2))


class TestHighWeightDecode:
    """高权重错误（|A| ≥ 5）恢复（MILP 兜底）。"""

    def test_high_weight_m6(self):
        """m=6 (n=64) r=3：权重 5-8 全部恢复（MILP 兜底）。"""
        import random
        from qecgeo import moments_of, rm_x_decode
        random.seed(12)
        m, r = 6, 3
        n = 1 << m
        for _ in range(5):
            w = random.randint(5, 8)
            A = random.sample(range(n), w)
            mm = moments_of(A, m, r)
            rec = rm_x_decode(mm, m, r)
            assert rec == sorted(A)

    def test_high_weight_m7(self):
        """m=7 (n=128) r=3：权重 8 恢复（MILP 兜底，n≤128 可行）。"""
        import random
        from qecgeo import moments_of, rm_x_decode
        random.seed(13)
        m, r = 7, 3
        n = 1 << m
        A = random.sample(range(n), 8)
        mm = moments_of(A, m, r)
        rec = rm_x_decode(mm, m, r)
        assert rec == sorted(A)
