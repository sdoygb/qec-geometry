#!/usr/bin/env python3
"""verify_degeneracy_classes.py —— RM(r,m) 权重 2^r 层简并类结构闭式验证

对照 rm_degeneracy_classes（10.30 开放问题 1 的 r≥1 通用化）：
  - 精确枚举（小参数）：类数、类大小分布逐项对比
  - 守恒律：闭式成员总数 = 10.33 简并比例分子
  - r=1 退化：精确回到 rm1_w2_degeneracy
  - 均匀性：r=1（全均匀）与 r=2（混合）边界行为

用法: python3 scripts/verify_degeneracy_classes.py
"""
import sys
from itertools import combinations
from collections import defaultdict, Counter
from math import comb

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from qecgeo.closedform import rm_degeneracy_classes, rm1_w2_degeneracy


def rm_css_gens(m, r):
    """CSS(RM(r,m)) 的稳定子生成元（与 verify_closedform.py 一致）。"""
    n = 1 << m
    rows = []
    for mask in range(1 << m):
        if mask.bit_count() <= r:
            rows.append([1 if (col & mask) == mask else 0 for col in range(n)])
    return n, rows


def syndrome_of_x(t, gens):
    """X 错误 t 的 syndrome（与 Z 稳定子的标量积）。"""
    out = []
    for g in gens:
        s = sum(ti & gi for ti, gi in zip(t, g)) & 1
        out.append(s)
    return tuple(out)


def enumerate_classes(m, r):
    """精确枚举：syndrome → 权重 2^r X 错误集。返回 (类数, 类大小分布)。"""
    n, gens = rm_css_gens(m, r)
    w0 = 1 << r
    classes = defaultdict(list)
    for idxs in combinations(range(n), w0):
        t = [0] * n
        for i in idxs:
            t[i] = 1
        classes[syndrome_of_x(t, gens)].append(idxs)
    sizes = Counter(len(v) for v in classes.values())
    return len(classes), sizes


def closed_form_dist(m, r):
    """闭式给出的类大小分布（size → 类数）。"""
    d = rm_degeneracy_classes(m, r)
    dist = {}
    if d["n_flat_classes"]:
        dist[d["size_flat_class"]] = dist.get(d["size_flat_class"], 0) + d["n_flat_classes"]
    if d["n_aff_classes"]:
        dist[2] = dist.get(2, 0) + d["n_aff_classes"]
    return dist


def main():
    print("== 1) 精确枚举对比（小参数）==")
    ok_all = True
    for (m, r) in [(4, 1), (5, 1), (6, 1), (4, 2), (5, 2), (4, 3)]:
        ec, esizes = enumerate_classes(m, r)
        cf = rm_degeneracy_classes(m, r)
        cfdist = closed_form_dist(m, r)
        ok = (ec == cf["n_classes"]) and (dict(esizes) == cfdist)
        ok_all &= ok
        tag = "✓" if ok else "✗"
        print(f"  RM({r},{m}): 类数 枚举={ec} 闭式={cf['n_classes']} "
              f"| 分布 枚举={dict(esizes)} 闭式={cfdist} {tag}")

    print("\n== 2) 守恒律（大参数，无枚举）==")
    for (m, r) in [(8, 1), (8, 2), (8, 3), (10, 3), (6, 4), (8, 4)]:
        d = rm_degeneracy_classes(m, r)
        members = d["members"]
        # 10.33 简并比例分子: flats(m,r) + flats(m,r+1)·E(r+1,2^r)
        from qecgeo.closedform import flats, E
        w0 = 1 << r
        expect = flats(m, r) + flats(m, r + 1) * E(r + 1, w0)
        ok = (members == expect) and (0 <= d["degenerate_ratio"] <= 1)
        ok_all &= ok
        print(f"  RM({r},{m}): 成员={members} 守恒={expect} "
              f"ratio={float(d['degenerate_ratio']):.4g} {'✓' if ok else '✗'}")

    print("\n== 3) r=1 退化 → rm1_w2_degeneracy ==")
    for m in [4, 6, 8]:
        d = rm_degeneracy_classes(m, 1)
        old = rm1_w2_degeneracy(m)
        ok = (d["n_classes"] == old["classes"]
              and d["size_flat_class"] == old["size_per_class"]
              and d["uniform"])
        ok_all &= ok
        print(f"  m={m}: 通用 {d['n_classes']}/{d['size_flat_class']} "
              f"== 原闭式 {old['classes']}/{old['size_per_class']} "
              f"均匀={d['uniform']} {'✓' if ok else '✗'}")

    print("\n== 4) 均匀性边界 ==")
    u1 = rm_degeneracy_classes(10, 1)
    u2 = rm_degeneracy_classes(10, 2)
    print(f"  RM(1,10) 均匀={u1['uniform']}（预期 True，全 r-平坦类）")
    print(f"  RM(2,10) 均匀={u2['uniform']}（预期 False，r-平坦 + (r+1)-仿射包混合）")
    ok_all &= u1["uniform"] and not u2["uniform"]

    print("\n" + ("✅ 全部通过" if ok_all else "❌ 存在不一致"))


if __name__ == "__main__":
    main()
