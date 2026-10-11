#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d5 분석. API는 부르지 않고 파일만 읽는다.

사용
  python d5_analyze.py            # x0.125 전체 분석 (주 검정 Q1, Q2)
  python d5_analyze.py 0.2        # 다른 배율의 분석 (로그인 목표만 해당)
  python d5_analyze.py --trend    # 로그인 목표의 오도 끌림을 배율별로 한 표에
  python d5_analyze.py --func 0.13  # 기능 설명 지시문(d5_results_func.csv) vs 글자 지시문, 같은 이미지끼리 짝지어

필요한 파일 (같은 폴더)
  d3_manifest.csv   d3 이미지의 버튼 좌표 (d4 gray가 어느 버튼을 골랐는지 계산하는 데 씀)
  d4_pilot.csv      로그인 gray(배율 1.0, 0.5, 0.25, 0.125), 로그인 T(d3의 h65/h245 = 조건 '65','245', 0.125)
  d4_label.csv      홈·설정 gray (x0.125에서만)
  d5_results.csv    d5 실행 결과. 로그인 gray가 있으면(중간 배율) d4 gray 대신 그것을 쓴다.

단위 = 블록 (레이아웃, 거울, 목표라벨). 한 블록에는 gray 1장과 조건별 이미지가 있다.
같은 블록 안에서 조건 - gray 차이를 구하므로 레이아웃 난이도는 상쇄된다.

검정 (둘 다 블록 차이에 대해)
  - 부호 뒤집기 순열검정: 귀무가설(효과 없음)에서는 차이의 부호가 대칭이므로 부호를 무작위로 뒤집어 분포를 만든다.
  - 레이아웃 단위 부트스트랩 구간: 같은 레이아웃의 거울 쌍과 목표 라벨들이 서로 닮았으니 레이아웃을 한 덩어리로 다시 뽑는다.
  - "지지"는 순열검정 p < 0.05 이고 부트스트랩 구간이 0을 포함하지 않을 때만. 둘이 갈리면 "불확정".

사전에 정한 주 질문 (결과를 보고 바꾸지 않는다)
  Q1 오도: D 조건에서 '오도 버튼을 고른 비율'이 같은 블록 gray에서 그 버튼을 고른 비율보다 큰가. (x0.125, 세 목표 합산)
  Q2 일반화: 목표만 유채색(T)이 gray보다 정확한가. (x0.125, 홈·설정 합산)
  추세: 로그인 목표에서 배율이 낮아질수록(1.0 -> 0.25 -> 0.2 -> 0.15 -> 0.14 -> 0.13 -> 0.125) 오도 끌림이 커지는가. 기술적 비교이며 단계별 검정은 탐색용.
"""
import csv, json, random, os, sys
from collections import defaultdict

NPERM = 10000
rng = random.Random(2026)
TREND_SCALES = ["1.0", "0.25", "0.2", "0.15", "0.14", "0.13", "0.125"]


def mean(v): return sum(v) / len(v)


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def label_at(boxes, px, py):
    ins = [b["label"] for b in boxes if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
    return ins[0] if ins else "(버튼 밖)"


def load(scale, path="d5_results.csv", use_d4=True):
    d3 = {r["file"]: json.loads(r["boxes"]) for r in read("d3_manifest.csv")}
    blocks = defaultdict(lambda: {"gray": None, "T": [], "D": []})
    # 로그인 gray, 로그인 T (d4_pilot.csv)
    for r in (read("d4_pilot.csv") if use_d4 else []):
        if r["scale"] != scale or r["status"] != "ok":
            continue
        key = (int(r["layout"]), int(r["mirror"]), "로그인")
        picked = label_at(d3[r["file"]], float(r["pred_x"]), float(r["pred_y"]))
        rec = {"hit": int(r["box_in"] == "1"), "picked": picked}
        if r["cond"] == "gray":
            blocks[key]["gray"] = rec
        elif r["cond"] in ("65", "245"):
            blocks[key]["T"].append(rec)
    # 홈, 설정 gray (d4_label.csv) — x0.125에서만 쟀음
    for r in (read("d4_label.csv") if (scale == "0.125" and use_d4) else []):
        if r["picked"] == "(파싱 실패)":
            continue
        blocks[(int(r["layout"]), int(r["mirror"]), r["target"])]["gray"] = {"hit": int(r["hit"]), "picked": r["picked"]}
    # d5 결과 (로그인 gray가 있으면 d4 gray를 덮어씀)
    for r in read(path):
        if r["scale"] != scale or r["status"] != "ok":
            continue
        key = (int(r["layout"]), int(r["mirror"]), r["target"])
        rec = {"hit": int(r["hit"]), "picked": r["picked"], "decoy": r["decoy"], "cond": r["cond"]}
        if r["kind"] == "gray":
            blocks[key]["gray"] = rec
        elif r["kind"] == "target":
            blocks[key]["T"].append(rec)
        else:
            blocks[key]["D"].append(rec)
    return blocks


def stats(items):
    """items = [(layout, 차이)] -> (n, 평균, p, lo, hi) 또는 None"""
    n = len(items)
    if n < 5:
        return None
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
    return n, obs, p, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]


def test(items, label):
    st = stats(items)
    if st is None:
        print(f"  {label}: 블록 {len(items)}개 — 너무 적어 검정하지 않음")
        return
    n, obs, p, lo, hi = st
    print(f"  {label}: 블록 {n}개, 평균 차이 {obs*100:+.1f}%p, 순열검정 p = {p:.4f}, 레이아웃 부트스트랩 95% [{lo*100:+.1f}, {hi*100:+.1f}]%p")


def d_pick_items(blocks, target):
    items = []
    for k, b in blocks.items():
        if k[2] != target or not b["gray"] or len(b["D"]) != 2:
            continue
        dec = b["D"][0]["decoy"]
        items.append((k[0], mean([int(r["picked"] == dec) for r in b["D"]]) - int(b["gray"]["picked"] == dec)))
    return items


def trend():
    print("=== 추세: 로그인 목표, 오도 버튼을 고른 비율의 배율별 변화 ===")
    print("(오도 끌림 = D에서 오도 버튼을 고른 비율 - 같은 블록 gray가 그 버튼을 고른 비율)\n")
    print(f"{'배율':>6} | {'블록':>4} | {'gray 정확도':>10} | {'D 정확도':>9} | {'gray가 오도버튼':>13} | {'D가 오도버튼':>11} | 끌림 (95% 구간, 순열검정 p)")
    print("-" * 112)
    for sc in TREND_SCALES:
        blocks = load(sc)
        ks = [k for k, b in blocks.items() if k[2] == "로그인" and b["gray"] and len(b["D"]) == 2]
        if not ks:
            print(f"{'x' + sc:>6} | 데이터 없음 (gray 또는 D가 완성되지 않음)")
            continue
        ga = mean([blocks[k]["gray"]["hit"] for k in ks])
        da = mean([mean([r["hit"] for r in blocks[k]["D"]]) for k in ks])
        gp = mean([int(blocks[k]["gray"]["picked"] == blocks[k]["D"][0]["decoy"]) for k in ks])
        dp = mean([mean([int(r["picked"] == r["decoy"]) for r in blocks[k]["D"]]) for k in ks])
        st = stats(d_pick_items(blocks, "로그인"))
        tail = (f"{st[1]*100:+.1f}%p [{st[3]*100:+.1f}, {st[4]*100:+.1f}], p = {st[2]:.4f}" if st else "검정 불가")
        print(f"{'x' + sc:>6} | {len(ks):>4} | {ga*100:>9.1f}% | {da*100:>8.1f}% | {gp*100:>12.1f}% | {dp*100:>10.1f}% | {tail}")
    print("\n※ gray가 거의 100%인 배율에서는 gray가 오도 버튼을 고른 비율이 0이라 끌림은 곧 D에서 오도 버튼을 고른 비율입니다.")
    print("※ 정확도가 100%에 붙은 배율의 '끌림 0'은 '색 영향이 정확히 없다'가 아니라 '정답 여부로는 보이지 않는다'입니다.")



def func_report(scale):
    """기능 설명 지시문(d5_results_func.csv) 분석. 같은 이미지에 두 지시문을 모두 물었으므로 블록 안에서 짝지어 비교한다."""
    if not os.path.exists("d5_results_func.csv"):
        raise SystemExit("d5_results_func.csv가 없습니다. 먼저 d5_run.py --instr func 를 돌리세요.")
    lit = load(scale)                                            # 글자 지시문 (d5_results.csv + d4)
    fun = load(scale, "d5_results_func.csv", use_d4=False)       # 기능 설명 지시문

    def dpick(b): return mean([int(r["picked"] == r["decoy"]) for r in b["D"]])
    def dhit(b): return mean([r["hit"] for r in b["D"]])
    def gpick(b): return int(b["gray"]["picked"] == b["D"][0]["decoy"])

    def ok(k):
        return (fun[k]["gray"] and len(fun[k]["D"]) == 2 and k in lit and lit[k]["gray"] and len(lit[k]["D"]) == 2)

    print(f"분석 배율: x{scale}   기능 설명 vs 글자 지시문 (같은 이미지끼리 짝지음)")
    groups = []
    for t in ("로그인", "홈", "설정"):
        ks = [k for k in list(fun) if k[2] == t and ok(k)]
        n_any = sum(1 for k in fun if k[2] == t)
        if n_any:
            print(f"  {t}: 두 지시문 모두 완성된 블록 {len(ks)}개" + ("" if len(ks) == 40 else "  (40개가 아니면 아직 덜 돈 것 — 해석하지 말 것)"))
        if len(ks) >= 5:
            groups.append((t, ks))
    if "홈" in dict(groups) and "설정" in dict(groups):
        groups.append(("홈+설정 합산 (주 검정)", dict(groups)["홈"] + dict(groups)["설정"]))
    if not groups:
        raise SystemExit("완성된 블록이 너무 적습니다.")

    for name, ks in groups:
        print(f"\n################ {name} (블록 {len(ks)}개) ################")
        print(f"  {'':16}{'gray 정확도':>12}{'D 정확도':>10}{'gray가 오도버튼':>16}{'D가 오도버튼':>13}")
        for nm, src in (("글자 지시문", lit), ("기능 설명 지시문", fun)):
            print(f"  {nm:16}{mean([src[k]['gray']['hit'] for k in ks])*100:>11.1f}%{mean([dhit(src[k]) for k in ks])*100:>9.1f}%"
                  f"{mean([gpick(src[k]) for k in ks])*100:>15.1f}%{mean([dpick(src[k]) for k in ks])*100:>12.1f}%")
        print("  -- 오도 끌림 (D가 오도버튼 - 같은 블록 gray가 오도버튼)")
        test([(k[0], dpick(fun[k]) - gpick(fun[k])) for k in ks], "기능 설명")
        test([(k[0], dpick(lit[k]) - gpick(lit[k])) for k in ks], "글자 (참고)")
        print("  -- 지시문 효과 (같은 이미지, 기능 설명 - 글자)  ← 핵심")
        test([(k[0], dpick(fun[k]) - dpick(lit[k])) for k in ks], "D에서 오도 버튼을 고른 비율")
        test([(k[0], (dpick(fun[k]) - gpick(fun[k])) - (dpick(lit[k]) - gpick(lit[k]))) for k in ks], "오도 끌림의 차이 (gray 선택까지 뺀 값)")
        test([(k[0], dhit(fun[k]) - dhit(lit[k])) for k in ks], "D 정확도")
        test([(k[0], fun[k]["gray"]["hit"] - lit[k]["gray"]["hit"]) for k in ks], "gray 정확도 (크게 다르면 gray 난이도가 섞임)")
        print("  -- 기능 설명: D 정확도 - gray 정확도")
        test([(k[0], dhit(fun[k]) - fun[k]["gray"]["hit"]) for k in ks], "기능 설명")
        print("  -- 색조별 오도 버튼 선택 (기술용)")
        for nm, src in (("글자", lit), ("기능", fun)):
            for c in ("D65", "D245"):
                v = [int(r["picked"] == r["decoy"]) for k in ks for r in src[k]["D"] if r.get("cond") == c]
                if v:
                    print(f"    {nm} {c}: {sum(v)}/{len(v)}")
    print("\n※ 홈·설정의 글자 지시문 gray는 d4_label.csv(같은 gray 그림) 결과입니다. 두 지시문은 서로 다른 날 호출이라 호출 간 흔들림이 섞여 있습니다.")
    print("※ 합산 검정에서 같은 레이아웃의 홈·설정 블록은 부트스트랩에서 한 덩어리로 다시 뽑힙니다.")


def main(scale):
    if not os.path.exists("d5_results.csv"):
        raise SystemExit("d5_results.csv가 없습니다. 먼저 d5_run.py를 돌리세요.")
    blocks = load(scale)
    targets = ["로그인", "홈", "설정"]
    print(f"분석 배율: x{scale}" + ("" if scale == "0.125" else "   (홈·설정 gray 기준선은 x0.125에서만 있어 이 배율에서는 로그인만 해당)"))

    print("=== 진행 상황 (블록 = 레이아웃x거울x목표, 목표당 40개) ===")
    for t in targets:
        ks = [k for k in blocks if k[2] == t]
        print(f"  {t}: gray {sum(1 for k in ks if blocks[k]['gray'])}, T 완성(2색조) {sum(1 for k in ks if len(blocks[k]['T']) == 2)}, "
              f"D 완성(2색조) {sum(1 for k in ks if len(blocks[k]['D']) == 2)}")

    print("\n=== Q1 오도: 오도 버튼을 고른 비율 (D 평균) - 같은 블록 gray가 그 버튼을 고른 비율 ===")
    pooled = []
    for t in targets:
        items = d_pick_items(blocks, t)
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

    print("\n=== Q2 목표만 유채색(T) 정확도 - gray 정확도 ===")
    for t in targets:
        items = [(k[0], mean([r["hit"] for r in b["T"]]) - b["gray"]["hit"])
                 for k, b in blocks.items() if k[2] == t and b["gray"] and len(b["T"]) == 2]
        test(items, f"목표 {t}" + ("  (d4 결과 재계산)" if t == "로그인" else ""))
    items = [(k[0], mean([r["hit"] for r in b["T"]]) - b["gray"]["hit"])
             for k, b in blocks.items() if k[2] in ("홈", "설정") and b["gray"] and len(b["T"]) == 2]
    test(items, "홈+설정 합산 (일반화 주 검정)")
    print("  ※ 홈·설정은 gray가 이미 약 72%라 올라갈 여지가 28%p뿐입니다(천장에 가까움).")

    print("\n=== 참고: gray에서 목표 정확도 ===")
    for t in targets:
        g = [b["gray"]["hit"] for k, b in blocks.items() if k[2] == t and b["gray"]]
        if g:
            print(f"  {t}: {sum(g)}/{len(g)}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--trend":
        trend()
    elif a and a[0] == "--func":
        func_report(a[1] if len(a) > 1 else "0.13")
    else:
        main(a[0] if a else "0.125")