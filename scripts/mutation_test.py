#!/usr/bin/env python3
"""mutation_test.py —— 变异测试（手动注入 bug，验证测试抓错能力）

不依赖 mutmut（其依赖 libcst 在此环境构建失败）。原理相同：
对源码注入已知变异（bug），跑测试套件，统计"测试能否抓住"。
存活变异 = 测试盲区（需要补测试）。

变异清单（每类注入 decoder.py 的一个真实 bug 模式）：
  M1  syndrome==0 误判（decode_error 旧 bug）——应被抓
  M2  in_group 相位比较（旧 bug）——应被抓
  M3  fail_rate 的 v 少算（类大小-1）——应被抓
  M4  build 枚举跳过 Y 型错误（t=3）——应被抓
  M5  decode 返回错误恢复（返回首个而非最小权重）——应被抓

用法: python3 scripts/mutation_test.py
输出: 每个变异的 [存活/被杀] + 总变异得分（被杀/总数）
"""
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECODER = os.path.join(REPO, 'qecgeo', 'decoder.py')

# ---- 中断恢复保证（260827 加固）----
# 变异注入期间任何中断（Ctrl-C / SIGTERM / 异常 / 退出）都必须恢复原始源码。
# 机制：启动时快照原始内容；signal handler + atexit + finally 三层兜底。
_ORIGINAL_SRC = None


def _restore_decoder():
    """把 decoder.py 恢复为启动时的原始内容（幂等，可多次调用）。"""
    global _ORIGINAL_SRC
    if _ORIGINAL_SRC is not None:
        try:
            open(DECODER, 'w', encoding='utf-8').write(_ORIGINAL_SRC)
        except Exception:
            pass  # 恢复失败也不掩盖原始错误


def _signal_handler(signum, frame):
    """SIGINT/SIGTERM：立即恢复源码再退出。"""
    _restore_decoder()
    print(f"\n[变异测试] 收到信号 {signum}，已恢复原始源码，退出")
    sys.exit(128 + signum)


def apply_mutation(src, name):
    """注入变异，返回变异后的源码（或 None 若该模式不适用）。"""
    if name == 'M1_syndrome_zero':
        # 旧 bug：decode_error 用 syndrome==0 判定（不查群成员）
        old = "        if self.syndrome_of(resid) != self.zero:\n            return resid, True  # 残留仍触发 syndrome → 明确逻辑错误\n        # syndrome==0：残留 ∈ normalizer。区分\"群元\"（成功）vs\"非平凡逻辑\"（失败）\n        try:\n            return resid, not self.in_group(resid)\n        except MemoryError:\n            # 大码（m>20）无法构建群：回退 syndrome==0 近似（残留 syndrome 0\n            # 视为群元——可能漏报逻辑错误，仅在无法精确判定时使用）\n            return resid, False"
        new = "        return resid, (self.syndrome_of(resid) != self.zero)"
        if old in src:
            return src.replace(old, new)
        return None
    if name == 'M2_in_group_phase':
        # 旧 bug：in_group 相位精确比较
        old = "return any(E.t == s.t for s in self._ensure_group())"
        new = "return any(E == s for s in self._ensure_group())"
        if old in src:
            return src.replace(old, new)
        return None
    if name == 'M3_fail_rate_v':
        # bug：fail_rate 的 v 减 1（少算类成员）
        old = "            v = len(self._classes[s])"
        new = "            v = max(len(self._classes[s]) - 1, 1)"
        if old in src:
            return src.replace(old, new)
        return None
    if name == 'M4_skip_Y':
        # bug：build 枚举跳过 Y 型错误
        old = "product((1, 2, 3), repeat=w)"
        new = "product((1, 2), repeat=w)"
        if old in src:
            return src.replace(old, new)
        return None
    if name == 'M5_decode_wrong':
        # bug：decode 返回类中最后一个而非最小权重
        old = "        return self.table.get(syndrome, Pauli.I(self.n))"
        new = "        return self.table.get(syndrome, Pauli.I(self.n)) if False else Pauli.I(self.n)"
        if old in src:
            return src.replace(old, new)
        return None
    return None


def run_tests():
    """跑与 decoder 相关的测试子集（快），返回 (通过, 总数)。
    判定用 returncode（pytest 失败返回非 0），不用字符串——'2 failed,
    8 passed' 也含 'passed'，字符串判定会误判。"""
    r = subprocess.run(
        [sys.executable, '-m', 'pytest',
         'tests/test_qecgeo.py::TestLookupDecoder',
         'tests/test_properties.py::TestLookupRecoveryInvariant',
         'tests/test_properties.py::TestBuildFastEquivalence',
         'tests/test_properties.py::TestRegressionMutations',
         '-q', '--no-header'],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    import re
    m = re.search(r'(\d+) passed', r.stdout)
    n = int(m.group(1)) if m else 0
    return r.returncode == 0, n


def main():
    global _ORIGINAL_SRC
    # 1. 快照原始源码（启动时）
    _ORIGINAL_SRC = open(DECODER, encoding='utf-8').read()
    # 2. 注册中断恢复：signal handler + atexit 兜底
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_restore_decoder)

    print("变异测试：注入 bug → 跑测试 → 统计抓错能力")
    print("=" * 60)
    results = []
    for name in ('M1_syndrome_zero', 'M2_in_group_phase', 'M3_fail_rate_v',
                 'M4_skip_Y', 'M5_decode_wrong'):
        mutated = apply_mutation(_ORIGINAL_SRC, name)
        if mutated is None:
            print(f"  {name:<20} [跳过: 模式不适用]")
            continue
        # 写入变异（每次从原始快照重新注入，避免累积）
        open(DECODER, 'w', encoding='utf-8').write(mutated)
        try:
            passed, n = run_tests()
        except subprocess.TimeoutExpired:
            passed, n = False, 0
        finally:
            _restore_decoder()  # 恢复（幂等）
        killed = not passed
        results.append((name, killed, n))
        print(f"  {name:<20} [{('被杀' if killed else '存活!')}] {n} 测试")

    score = sum(1 for _, k, _ in results if k) / len(results) if results else 0
    print("\n" + "=" * 60)
    print(f"变异得分: {score:.0%}（被杀/总数）—— 100% = 测试抓住所有注入 bug")
    alive = [n for n, k, _ in results if not k]
    if alive:
        print(f"存活变异（测试盲区）: {alive}")
        print("→ 需要补测试覆盖这些盲区")
    else:
        print("→ 无存活变异：测试体系抓住所有注入 bug")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
