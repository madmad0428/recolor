#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d5 분석 (x0.125). API는 부르지 않고 파일만 읽는다.

필요한 파일 (같은 폴더)
  d3_manifest.csv   d3 이미지의 버튼 좌표 (로그인 gray가 어느 버튼을 골랐는지 계산하는 데 씀)
  d4_pilot.csv      로그인 gray, 로그인 T(d3의 h65/h245 = 조건 '65','245')   (x0.125)
  d4_label.csv      홈·설정 gray
  d5_results.csv    d5 실행 결과 (홈·설정의 T/D, 로그인의 D)

단위 = 블록 (레이아웃, 거울, 목표라벨). 한 블록에는 gray 1장과 조건별 이미지가 있다.
같은 블록 안에서 조건 - gray 차이를 구하므로 레이아웃 난이도는 상쇄된다.

검정 (둘 다 블록 차이에 대해)
  - 부호 뒤집기 순열검정: 귀무가설(효과 없음)에서는 차이의 부호가 대칭이므로 부호를 무작위로 뒤집어 분포를 만든다.
  - 레이아웃 단위 부트스트랩 구간: 같은 레이아웃의 거울 쌍과 목표 라벨들이 서로 닮았으니 레이아웃을 한 덩어리로 다시 뽑는다.

사전에 정한 주 질문 (결과를 보고 바꾸지 않는다)
  Q1 오도: D 조건에서 '오도 버튼을 고른 비율'이 같은 블록 gray에서 그 버튼을 고른 비율보다 큰가. (세 목표 합산)
  Q2 일반화: 목표만 유채색(T)이 gray보다 정확한가. (홈·설정 합산이 일반화 질문, 로그인은 d4에서 이미 확인)
  보조: D에서 목표 정확도가 gray보다 떨어지는가, 라벨별 결과, 색조(65 vs 245) 비교는 탐색용.
"""
import csv, json, random, os, sys
from collections import defaultdict

SCALE = sys.argv[1] if len(sys.argv) > 1 else "0.125"   # 예: python d5_analyze.py 1.0
NPERM = 10000
rng = random.Random(2026)


def mean(v): return sum(v) / len(v)


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def label_at(boxes, px, py):
    ins = [b["label"] for b in boxes if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
    return ins[0] if ins else "(버튼 밖)"


def load():
    d3 = {r["file"]: json.loads(r["boxes"]) for r in read("d3_manifest.csv")}
    blocks = defaultdict(lambda: {"gray": None, "T": [], "D": []})
    # 로그인 gray, 로그인 T (d4_pilot.csv, 배율 0.125)
    for r in read("d4_pilot.csv"):
        if r["scale"] != SCALE or r["status"] != "ok":
            continue
        key = (int(r["layout"]), int(r["mirror"]), "로그인")
        picked = label_at(d3[r["file"]], float(r["pred_x"]), float(r["pred_y"]))
        rec = {"hit": int(r["box_in"] == "1"), "picked": picked}
        if r["cond"] == "gray":
            blocks[key]["gray"] = rec
        elif r["cond"] in ("65", "245"):
            blocks[key]["T"].append(rec)
    # 홈, 설정 gray (d4_label.csv) — 이 파일은 x0.125에서만 쟀으므로 그 배율일 때만 쓴다
    for r in (read("d4_label.csv") if SCALE == "0.125" else []):
        if r["picked"] == "(파싱 실패)":
            continue
        blocks[(int(r["layout"]), int(r["mirror"]), r["target"])]["gray"] = {"hit": int(r["hit"]), "picked": r["picked"]}
    # d5 결과
    for r in read("d5_results.csv"):
        if r["scale"] != SCALE or r["status"] != "ok":
            continue
        key = (int(r["layout"]), int(r["mirror"]), r["target"])
        rec = {"hit": int(r["hit"]), "picked": r["picked"], "decoy": r["decoy"]}
        blocks[key]["T" if r["kind"] == "target" else "D"].append(rec)
    return blocks


def test(items, label):
    """items = [(layout, 차이)]. 부호 뒤집기 순열검정 + 레이아웃 부트스트랩."""
    n = len(items)
    if n < 5:
        print(f"  {label}: 블록 {n}개 — 너무 적어 검정하지 않음")
        return
    d = [x for _, x in items]
    obs = mean(d)
    ge = 0
    for _ in range(NPERM):
        s = mean([x if rng.random() < 0.5 else -x for x in d])
        ge += abs(s) >= abs(obs) - 1e-12
    p = (1 + ge) / (1 + NPERM)
    by = defaultdict(list)
    for l, x in items:
        by[l].append(x)
    L = list(by)
    boots = []
    for _ in range(NPERM // 2):
        s = []
        for _ in L:
            s += by[rng.choice(L)]
        boots.append(mean(s))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    print(f"  {label}: 블록 {n}개, 평균 차이 {obs*100:+.1f}%p, 순열검정 p = {p:.4f}, 레이아웃 부트스트랩 95% [{lo*100:+.1f}, {hi*100:+.1f}]%p")


def main():
    if not os.path.exists("d5_results.csv"):
        raise SystemExit("d5_results.csv가 없습니다. 먼저 d5_run.py를 돌리세요.")
    blocks = load()
    targets = ["로그인", "홈", "설정"]
    print(f"분석 배율: x{SCALE}" + ("" if SCALE == "0.125" else "   (홈·설정 gray 기준선은 x0.125에서만 있어 이 배율에서는 로그인만 해당)"))

    print("=== 진행 상황 (블록 = 레이아웃x거울x목표, 목표당 40개) ===")
    for t in targets:
        ks = [k for k in blocks if k[2] == t]
        print(f"  {t}: gray {sum(1 for k in ks if blocks[k]['gray'])}, T 완성(2색조) {sum(1 for k in ks if len(blocks[k]['T']) == 2)}, "
              f"D 완성(2색조) {sum(1 for k in ks if len(blocks[k]['D']) == 2)}")

    # ---- Q1 오도 ----
    print("\n=== Q1 오도: 오도 버튼을 고른 비율 (D 평균) - 같은 블록 gray가 그 버튼을 고른 비율 ===")
    pooled = []
    for t in targets:
        items = []
        for k, b in blocks.items():
            if k[2] != t or not b["gray"] or len(b["D"]) != 2:
                continue
            dec = b["D"][0]["decoy"]
            d_pick = mean([int(r["picked"] == dec) for r in b["D"]])
            g_pick = int(b["gray"]["picked"] == dec)
            items.append((k[0], d_pick - g_pick))
        test(items, f"목표 {t}")
        pooled += items
    test(pooled, "세 목표 합산 (주 검정)")

    print("\n=== 보조: D에서 목표 정확도 - gray 정확도 ===")
    pooled = []
    for t in targets:
        items = [(k[0], mean([r["hit"] for r in b["D"]]) - b["gray"]["hit"])
                 for k, b in blocks.items() if k[2] == t and b["gray"] and len(b["D"]) == 2]
        test(items, f"목표 {t}")
        pooled += items
    test(pooled, "세 목표 합산")

    # ---- Q2 T vs gray ----
    print("\n=== Q2 목표만 유채색(T) 정확도 - gray 정확도 ===")
    for t in targets:
        items = [(k[0], mean([r["hit"] for r in b["T"]]) - b["gray"]["hit"])
                 for k, b in blocks.items() if k[2] == t and b["gray"] and len(b["T"]) == 2]
        test(items, f"목표 {t}" + ("  (d4 결과 재계산)" if t == "로그인" else ""))
    items = [(k[0], mean([r["hit"] for r in b["T"]]) - b["gray"]["hit"])
             for k, b in blocks.items() if k[2] in ("홈", "설정") and b["gray"] and len(b["T"]) == 2]
    test(items, "홈+설정 합산 (일반화 주 검정)")
    print("  ※ 홈·설정은 gray가 이미 약 72%라 올라갈 여지가 28%p뿐입니다(천장에 가까움).")

    # ---- 참고: 라벨별 gray ----
    print("\n=== 참고: gray에서 목표 정확도 ===")
    for t in targets:
        g = [b["gray"]["hit"] for k, b in blocks.items() if k[2] == t and b["gray"]]
        if g:
            print(f"  {t}: {sum(g)}/{len(g)}")


if __name__ == "__main__":
    main()