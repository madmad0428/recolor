#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d7 분석 (3a: 모양 끌림). API는 부르지 않고 d7_results.csv만 읽는다.

사용
  python d7_analyze.py              # 사전 점검: 배율 x 조건별 정확도·방해 자리 선택
  python d7_analyze.py 0.13         # 본 분석 (배율은 d6와 같은 x0.13으로 미리 고정)

조건 (방해 자리 = 레이아웃마다 정해진 중립 버튼 한 자리, d6와 같은 자리)
  Tgray/Tred/Tblue : 그 자리가 글자 버튼 (회/빨/파)
  Igray/Ired/Iblue : 그 자리가 아이콘 버튼 (회/빨/파)
  '방해 선택' = 그 자리를 골랐는가. 6조건 모두 같은 기준.

단위와 검정 (d6와 같음)
  단위 = (레이아웃, 거울). 레이아웃 단위 부호 뒤집기 순열검정 + 레이아웃 부트스트랩 95% 구간.
  "지지" = p < 기준 이고 부트스트랩 구간이 0을 포함하지 않을 때만.

미리 정한 판정 (결과를 보고 바꾸지 않는다)
  [안전 점검]  x1.0 Igray를 돌렸다면, 그 정확도가 90% 미만이면 아이콘 화면 자체가 모델을 헷갈리게 하는 것이라
              모양 끌림 해석을 보류한다. (x1.0을 안 돌렸으면 이 점검은 '안 함'으로 표시)
  [조작 확인]  글자 색 끌림 = 평균(Tred, Tblue) - Tgray 방해 선택. d6 재현 확인용.
  [주 검정 1]  모양 끌림 = Igray - Tgray 방해 선택. > 0 이고 지지되면 "혼자 모양이 다른 버튼에도 오답이 몰린다".
  [주 검정 2]  모양 x 색 = (평균(Ired,Iblue) - Igray) - (평균(Tred,Tblue) - Tgray). 양쪽 검정.
              < 0 지지: 아이콘에서는 색이 덜 더해짐(이미 눈에 띄는 버튼엔 색 효과가 줄어듦)
              > 0 지지: 아이콘에서 색이 더 크게 더해짐
              지지 안 됨: 색 효과가 형태에 따라 다르다는 증거 없음 (더해진다고 '증명'된 것은 아님)
              [조작 확인]이 지지되지 않으면 주 검정 2는 해석 보류.
  주 검정이 2개이므로 Holm 보정: 작은 p는 0.025, 큰 p는 0.05 미만이어야 지지.
  [보조, 탐색]  아이콘 색 끌림, 모양+색 합계(Icolor - Tgray), 정확도 차이, 빨강 vs 파랑, 아이콘 종류별 모양 끌림(기술만).
"""
import csv, os, random, sys
from collections import defaultdict

NPERM = 20000
rng = random.Random(2027)
RES = "d7_results.csv"
CONDS = ("Tgray", "Tred", "Tblue", "Igray", "Ired", "Iblue")
TARGETS = ("삭제", "저장")


def mean(v): return sum(v) / len(v)


def load():
    if not os.path.exists(RES):
        raise SystemExit(f"{RES}가 없습니다. 먼저 d5_run.py --set d7 로 돌리세요.")
    return [r for r in csv.DictReader(open(RES, encoding="utf-8-sig")) if r["status"] == "ok"]


def pick(r): return int(r["picked"] == r["decoy"])
def hit(r): return int(r["hit"])


def precheck(rows):
    print("=== 사전 점검: 배율 x 조건 (두 목표 합산) ===")
    g = defaultdict(list)
    for r in rows:
        g[(r["scale"], r["cond"])].append(r)
    for sc in sorted({k[0] for k in g}, key=float, reverse=True):
        print(f"  x{sc}")
        for c in CONDS:
            v = g.get((sc, c), [])
            if v:
                print(f"    {c:6} n={len(v):3}  정확도 {mean([hit(r) for r in v])*100:5.1f}%   "
                      f"방해 자리 선택 {mean([pick(r) for r in v])*100:5.1f}%")
    v = g.get(("1.0", "Igray"), [])
    if v:
        a = mean([hit(r) for r in v])
        print(f"\n[안전 점검] x1.0 Igray 정확도 {a*100:.1f}% → {'통과' if a >= 0.9 else '90% 미만: 모양 끌림 해석 보류'}")


def stats(items):
    by = defaultdict(list)
    for l, x in items:
        by[l].append(x)
    L = list(by)
    if len(L) < 5:
        return None
    obs = mean([x for _, x in items])
    ge = 0
    for _ in range(NPERM):
        s = []
        for l in L:
            f = 1 if rng.random() < 0.5 else -1
            s += [f * x for x in by[l]]
        ge += abs(mean(s)) >= abs(obs) - 1e-12
    p = (1 + ge) / (1 + NPERM)
    boots = []
    for _ in range(NPERM // 4):
        s = []
        for _ in L:
            s += by[rng.choice(L)]
        boots.append(mean(s))
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
    return len(items), len(L), obs, p, lo, hi


def show(label, st, alpha=0.05):
    if st is None:
        print(f"  {label}: 레이아웃이 너무 적어 검정하지 않음"); return None
    n, nl, obs, p, lo, hi = st
    sup = p < alpha and (lo > 0 or hi < 0)
    print(f"  {label}: 단위 {n}개(레이아웃 {nl}), 평균 {obs*100:+.1f}%p, 순열검정 p = {p:.4f}, "
          f"부트스트랩 95% [{lo*100:+.1f}, {hi*100:+.1f}]%p  → {'지지' if sup else '지지 안 됨'}"
          + (f" (Holm 기준 p<{alpha})" if alpha not in (0.05, 0.0) else " (Holm: 다른 주 검정이 먼저 통과 못 해 지지 불가)" if alpha == 0.0 else ""))
    return sup, obs


def main(scale):
    rows = load()
    U = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["scale"] == scale:
            U[(int(r["layout"]), int(r["mirror"]))][r["target"]][r["cond"]] = r
    U = {k: v for k, v in U.items() if all(len(v[t]) == 6 for t in TARGETS)}
    print(f"분석 배율: x{scale}   완성된 단위(레이아웃x거울) {len(U)}개"
          + ("" if len(U) == 40 else "  ← 40개가 아님: 덜 돈 상태라 해석하지 말 것"))
    if not U:
        return
    v1 = [r for r in rows if r["scale"] == "1.0" and r["cond"] == "Igray"]
    if v1:
        a = mean([hit(r) for r in v1])
        safe = a >= 0.9
        print(f"[안전 점검] x1.0 Igray 정확도 {a*100:.1f}% (n={len(v1)}) → {'통과' if safe else '90% 미만'}")
    else:
        safe = True
        print("[안전 점검] x1.0 Igray 안 함")

    def m(k, c, f=pick): return mean([f(U[k][t][c]) for t in TARGETS])     # 두 목표 평균

    print("\n=== 기술 통계 (두 목표 합산) ===")
    print(f"  {'조건':6} {'정확도':>7} {'방해 자리 선택':>14}")
    for c in CONDS:
        print(f"  {c:6} {mean([m(k, c, hit) for k in U])*100:6.1f}% {mean([m(k, c) for k in U])*100:13.1f}%")

    col = lambda k, f: (m(k, f + "red") + m(k, f + "blue")) / 2
    print("\n=== [조작 확인] 글자 색 끌림: 평균(Tred,Tblue) - Tgray ===")
    man = show("글자 색 끌림", stats([(k[0], col(k, "T") - m(k, "Tgray")) for k in U]))

    st1 = stats([(k[0], m(k, "Igray") - m(k, "Tgray")) for k in U])
    st2 = stats([(k[0], (col(k, "I") - m(k, "Igray")) - (col(k, "T") - m(k, "Tgray"))) for k in U])
    # Holm: p가 작은 쪽은 0.025, 큰 쪽은 (작은 쪽이 0.025 미만일 때만) 0.05
    if st1[3] <= st2[3]:
        a1, a2 = 0.025, (0.05 if st1[3] < 0.025 else 0.0)
    else:
        a2, a1 = 0.025, (0.05 if st2[3] < 0.025 else 0.0)
    print("\n=== [주 검정 1] 모양 끌림: Igray - Tgray (방해 자리 선택) ===")
    r1 = show("모양 끌림", st1, a1)
    print("\n=== [주 검정 2] 모양 x 색: (아이콘 색 끌림) - (글자 색 끌림) ===")
    r2 = show("모양 x 색", st2, a2)

    print("\n=== [보조, 탐색] ===")
    show("아이콘 색 끌림: 평균(Ired,Iblue) - Igray", stats([(k[0], col(k, "I") - m(k, "Igray")) for k in U]))
    show("모양+색 합계: 평균(Ired,Iblue) - Tgray", stats([(k[0], col(k, "I") - m(k, "Tgray")) for k in U]))
    show("정확도: Igray - Tgray", stats([(k[0], m(k, "Igray", hit) - m(k, "Tgray", hit)) for k in U]))
    show("빨강 - 파랑 (두 형태 평균)", stats([(k[0], (m(k, "Tred") + m(k, "Ired") - m(k, "Tblue") - m(k, "Iblue")) / 2) for k in U]))
    by_icon = defaultdict(list)
    icon_of = {}
    if os.path.exists("d7_manifest.csv"):
        icon_of = {int(r["layout"]): r["icon"] for r in csv.DictReader(open("d7_manifest.csv", encoding="utf-8-sig"))
                   if r["form"] == "I"}
    for k in U:
        by_icon[icon_of.get(k[0], "?")].append(m(k, "Igray") - m(k, "Tgray"))
    if any(by_icon):
        print("  아이콘 종류별 모양 끌림 (기술만, 종류당 레이아웃 5개라 검정 안 함):")
        for ic_, v in sorted(by_icon.items()):
            print(f"    {ic_:10} {mean(v)*100:+.1f}%p  (단위 {len(v)})")

    print("\n=== 판정 (미리 정한 규칙) ===")
    if not safe:
        print("  x1.0에서 아이콘 화면 정확도가 90% 미만 → 모양 끌림 해석 보류.")
    elif r1 and r1[0] and r1[1] > 0:
        print("  [1] 모양 끌림 지지 → 혼자 모양이 다른(아이콘) 버튼에도 오답이 몰린다. 색만의 효과가 아님.")
    elif r1 and r1[0] and r1[1] < 0:
        print("  [1] 모양 끌림이 음수로 지지 → 아이콘 자리가 오히려 덜 골라짐(예측과 반대, 그대로 보고).")
    else:
        print("  [1] 모양 끌림이 0과 구분되지 않음 → 이 조건에서 모양 끌림은 확인되지 않음(효과 없음과는 다름).")
    if not man or not man[0]:
        print("  [2] 글자 색 끌림이 이번에 지지되지 않음 → 모양 x 색 해석 보류.")
    elif r2 and r2[0]:
        print(f"  [2] 모양 x 색 {'< 0' if r2[1] < 0 else '> 0'} 지지 → 색 효과 크기가 형태에 따라 다름"
              f"({'아이콘에서 더 작음' if r2[1] < 0 else '아이콘에서 더 큼'}).")
    else:
        print("  [2] 모양 x 색이 0과 구분되지 않음 → 색 효과가 형태에 따라 다르다는 증거 없음.")
    print("\n※ 단일 모델·합성 UI·글자 지시문·라벨과 같은 뜻의 아이콘 4종(톱니·돋보기·집·물음표)·한 배율 기준. 아이콘은 뭉개지면 '진한 덩어리'로 남아 글자와 보이는 방식이 다름(모양 차이에 포함).")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        precheck(load())
    else:
        main(a[0])