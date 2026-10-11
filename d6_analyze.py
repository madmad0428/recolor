#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d6 분석 (2부: 색 관습). API는 부르지 않고 d6_results.csv만 읽는다.

사용
  python d6_analyze.py                 # 사전 점검: 배율·목표별 gray 정확도 (배율 고르기용)
  python d6_analyze.py 0.13            # 한 배율 본 분석
  python d6_analyze.py 0.13,0.125      # 두 배율 합산 본 분석 (미리 정한 두 배율일 때만)

단위와 검정
  단위 = (레이아웃, 거울, 배율). 한 단위에 gray, Dred, Dblue 이미지가 있고 각각을 목표 '삭제', '저장'으로 물었다.
  검정은 레이아웃 단위로 묶어서 한다(같은 레이아웃의 거울·배율은 서로 닮았으므로).
    - 레이아웃 단위 부호 뒤집기 순열검정: 레이아웃마다 부호를 함께 뒤집는다.
    - 레이아웃 단위 부트스트랩 95% 구간.
  "지지" = 순열검정 p < 0.05 이고 부트스트랩 구간이 0을 포함하지 않을 때만.

미리 정한 판정 (결과를 보고 바꾸지 않는다)
  [조작 확인] 색 끌림 = (D에서 방해 버튼 선택) - (gray에서 그 버튼 선택), 두 목표·두 색 평균.
             이게 지지되지 않으면 그 배율에서는 색이 선택을 끌지 않는 것이라 교차 검정을 해석하지 않는다(보류).
  [주 검정]   교차 = (삭제: Dred 선택 - Dblue 선택) - (저장: Dred 선택 - Dblue 선택)
             > 0 이고 지지되면 "기능에 어울리는 색의 버튼에 더 끌린다(색 관습 사용)".
             지지되지 않으면 "이 조건에서 색 관습 효과는 확인되지 않음". 0과 구분 안 되는 것을 '효과 없음'으로 쓰지 않는다.
  [보조]     빨강 주효과(빨강 - 파랑, 두 목표 평균) = 빨강이 원래 더 끄는가. 교차와 별개.
             목표별 단순효과, 정확도로 본 교차(관습 색 방해가 정확도를 더 떨어뜨리는가).
"""
import csv, os, random, sys
from collections import defaultdict

NPERM = 20000
rng = random.Random(2026)
RES = "d6_results.csv"
TARGETS = ("삭제", "저장")


def mean(v): return sum(v) / len(v)


def load():
    if not os.path.exists(RES):
        raise SystemExit(f"{RES}가 없습니다. 먼저 d5_run.py --set d6 로 돌리세요.")
    return [r for r in csv.DictReader(open(RES, encoding="utf-8-sig")) if r["status"] == "ok"]


def precheck(rows):
    print("=== 사전 점검: gray 정확도 (배율 x 목표) ===")
    print("목표 구간: 60~70% 근처. 너무 높으면 색이 끼어들 여지가 없고, 너무 낮으면 찍기라 해석이 어렵다.\n")
    g = defaultdict(list)
    for r in rows:
        if r["cond"] == "gray":
            g[(r["scale"], r["target"])].append(int(r["hit"]))
    for sc in sorted({k[0] for k in g}, key=float, reverse=True):
        line = f"  x{sc:<6}"
        for t in TARGETS:
            v = g.get((sc, t), [])
            line += f"  {t}: {sum(v)}/{len(v)}" + (f" ({sum(v)/len(v)*100:.1f}%)" if v else "") + "   "
        print(line)
    print("\n※ 두 목표 모두 40/40개가 찬 배율만 판단에 쓰세요.")


def units(rows, scales):
    """(layout, mirror, scale) -> {target: {cond: rec}}"""
    u = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["scale"] in scales:
            u[(int(r["layout"]), int(r["mirror"]), r["scale"])][r["target"]][r["cond"]] = r
    full = {k: v for k, v in u.items() if all(len(v[t]) == 3 for t in TARGETS)}
    return full


def stats(items):
    """items = [(layout, 값)]. 레이아웃 단위 부호 뒤집기 + 레이아웃 부트스트랩."""
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


def test(items, label):
    st = stats(items)
    if st is None:
        print(f"  {label}: 레이아웃이 너무 적어 검정하지 않음")
        return None
    n, nl, obs, p, lo, hi = st
    sup = p < 0.05 and (lo > 0 or hi < 0)
    print(f"  {label}: 단위 {n}개(레이아웃 {nl}), 평균 {obs*100:+.1f}%p, 순열검정 p = {p:.4f}, "
          f"부트스트랩 95% [{lo*100:+.1f}, {hi*100:+.1f}]%p  → {'지지' if sup else '지지 안 됨'}")
    return sup, obs


def main(scales):
    rows = load()
    U = units(rows, scales)
    print(f"분석 배율: {', '.join('x' + s for s in scales)}   완성된 단위(레이아웃x거울x배율) {len(U)}개"
          + ("" if len(U) == 40 * len(scales) else f"  ← {40*len(scales)}개가 아님: 덜 돈 상태라 해석하지 말 것"))
    if not U:
        return

    def pick(r): return int(r["picked"] == r["decoy"])
    def hit(r): return int(r["hit"])

    print("\n=== 기술 통계 ===")
    print(f"  {'목표':4} {'조건':6} {'정확도':>8} {'방해버튼 선택':>12}")
    for t in TARGETS:
        for c in ("gray", "Dred", "Dblue"):
            v = [U[k][t][c] for k in U]
            print(f"  {t:4} {c:6} {mean([hit(r) for r in v])*100:7.1f}% {mean([pick(r) for r in v])*100:11.1f}%")

    print("\n=== [조작 확인] 색 끌림 (D 방해선택 - gray 방해선택, 두 목표·두 색 평균) ===")
    man = test([(k[0], mean([pick(U[k][t][c]) - pick(U[k][t]["gray"]) for t in TARGETS for c in ("Dred", "Dblue")]))
                for k in U], "색 끌림")

    print("\n=== [주 검정] 교차: (삭제: 빨강-파랑) - (저장: 빨강-파랑), 방해 버튼 선택 ===")
    dd = lambda k, t: pick(U[k][t]["Dred"]) - pick(U[k][t]["Dblue"])
    main_res = test([(k[0], dd(k, "삭제") - dd(k, "저장")) for k in U], "교차")

    print("\n=== [보조] ===")
    test([(k[0], mean([dd(k, t) for t in TARGETS])) for k in U], "빨강 주효과 (빨강-파랑, 두 목표 평균)")
    test([(k[0], dd(k, "삭제")) for k in U], "삭제에서 빨강-파랑 (예측: +)")
    test([(k[0], dd(k, "저장")) for k in U], "저장에서 빨강-파랑 (예측: -)")
    da = lambda k, t: hit(U[k][t]["Dred"]) - hit(U[k][t]["Dblue"])
    test([(k[0], da(k, "삭제") - da(k, "저장")) for k in U], "정확도 교차 (예측: -, 관습 색 방해가 정확도를 더 깎음)")

    print("\n=== 판정 (미리 정한 규칙) ===")
    if not man or not man[0]:
        print("  색 끌림이 지지되지 않음 → 이 배율에서는 색이 선택을 끌지 않음. 교차 검정은 해석 보류.")
    elif main_res and main_res[0] and main_res[1] > 0:
        print("  교차 > 0 지지 → 기능에 어울리는 색(삭제-빨강, 저장-파랑)의 버튼에 더 끌림. 색 관습 사용을 지지.")
    elif main_res and main_res[0] and main_res[1] < 0:
        print("  교차 < 0 지지 → 관습과 반대 방향. 예측과 다름(그대로 보고).")
    else:
        print("  교차가 0과 구분되지 않음 → 이 조건에서 색 관습 효과는 확인되지 않음(효과 없음과는 다름).")
    print("\n※ 단일 모델·합성 UI·글자 지시문·빨강/파랑 각 한 색 기준의 결과입니다.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        precheck(load())
    else:
        main(a[0].split(","))