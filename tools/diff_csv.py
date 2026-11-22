#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比两次 --csv 输出，找出哪一层变慢/变快了。

用法:
    python tools/diff_csv.py out/layers_before.csv out/layers_after.csv

为什么需要它：控制台那张表只有绝对值，改完代码看一眼「好像差不多」就过去了。
性能回归往往就是某一层慢 15%，肉眼看不出来，diff 一下立刻现形。
"""
import csv
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["layer"]: float(r["ms"]) for r in rows}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = [k for k in a if k in b]
    if not common:
        print("两个文件的 layer 名对不上，检查是否同一份代码生成的")
        return 1

    print(f"{'layer':<22}{'before':>10}{'after':>10}{'delta':>10}{'%':>9}")
    print("-" * 61)
    worst = None
    for k in common:
        d = b[k] - a[k]
        pct = (d / a[k] * 100) if a[k] else 0.0
        flag = ""
        # 只按百分比会误判：maxpool 只有 0.2ms，抖 0.02ms 就是 10%。
        # 所以要求「相对 + 绝对」同时超阈值才算退化。
        if pct > 5 and d >= 0.15:
            flag = "  ← 变慢"
            if worst is None or pct > worst[1]:
                worst = (k, pct)
        elif pct > 5:
            flag = "  (幅度小，可能是噪声)"
        print(f"{k:<22}{a[k]:>10.3f}{b[k]:>10.3f}{d:>+10.3f}{pct:>+8.1f}%{flag}")

    print("-" * 61)
    if worst:
        print(f"最需要看的一层：{worst[0]}（+{worst[1]:.1f}%）")
        return 2          # 有超过 5% 的退化，CI 里可以直接当失败
    print("没有超过 5% 的退化")
    return 0


if __name__ == "__main__":
    sys.exit(main())
