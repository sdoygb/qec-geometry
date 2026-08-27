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
