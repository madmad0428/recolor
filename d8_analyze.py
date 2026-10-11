#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d8 분석 (3부 재설계: 목표 종류 x 방해 형태 x 방해 색, 기능 지시문). API는 부르지 않고 d8_results.csv만 읽는다.

사용
  python d8_analyze.py                    # 사전 점검: x1.0 안전 점검 + 층별 배율 자동 권장
  python d8_analyze.py TT=0.13 IT=0.15    # 본 분석 (층마다 사전 점검이 권장한 배율)
  python d8_analyze.py TT=0.13            # 한 층만

층: TT = 목표가 글자 버튼, IT = 목표가 아이콘 버튼. 두 층은 따로 분석한다(ScreenSpot-Pro가 글자/아이콘을 나눠 보고하듯).
조건: <층>_<방해형태 T/I><색 gray/red/blue>. '방해 선택' = 방해 자리(레이아웃마다 정해진 한 자리)를 골랐는가.

미리 정한 규칙 (결과를 보고 바꾸지 않는다)
  레이아웃 60개. 사전 점검(안전 점검·배율 선택)은 0~19번 레이아웃만으로 하고, 본 분석은 60개 전부로 한다.
  [안전 점검]  x1.0에서 층마다 Tgray, Igray 정확도(두 목표 합)가 90% 이상이어야 그 층을 해석한다.
              미만이면 기능 지시문 자체를 모델이 못 따라가는 것이라 그 층 해석 보류.
  [배율 선택]  층마다 Tgray 정확도로 고른다. 후보 = 두 목표 모두 40개가 차 있고, 목표별 정확도가 각각 50~80%,
              두 목표 차이 30%p 미만인 배율. 그중 두 목표 합 정확도가 65%에 가장 가까운 배율(같으면 큰 배율).
              후보가 없으면 배율을 더 찾는다. 고른 배율은 본 실험 전에 고정.
  층마다 검정 (단위 = 레이아웃 x 거울, 두 목표 평균. 레이아웃 단위 부호 뒤집기 순열검정 + 레이아웃 부트스트랩)
  [조작 확인]  색 끌림 = 평균(red, blue) - gray, 두 방해 형태 평균. 지지되지 않으면 주 검정 2, 3은 해석 보류.
  [주 검정 1]  모양 끌림 = Igray - Tgray. 양쪽 검정. > 0 지지: 혼자 모양이 다른 버튼에 오답이 몰림.
  [주 검정 2]  모양 x 색 = (I의 색 끌림) - (T의 색 끌림). 양쪽 검정.
  [주 검정 3]  색 관습 = (삭제: red - blue) - (저장: red - blue), 두 방해 형태 평균. 예측 > 0.
  층 안의 주 검정 3개는 Holm 보정(가장 작은 p부터 0.0167, 0.025, 0.05). 지지 = Holm 통과 + 부트스트랩 구간이 0 제외.
  두 층은 서로 다른 질문(글자 목표 / 아이콘 목표)으로 보고 층 사이 보정은 하지 않는다. 층 비교는 같은 배율일 때만 탐색으로.
"""
import csv, json, os, random, sys
from collections import defaultdict

NPERM = 20000
rng = random.Random(2031)
RES = "d8_results.csv"
TARGETS = ("삭제", "저장")
FORMS = ("T", "I")
COLS = ("gray", "red", "blue")
STRATA = ("TT", "IT")
NAME = {"TT": "TT (목표 = 글자 버튼)", "IT": "IT (목표 = 아이콘 버튼)"}


def mean(v): return sum(v) / len(v)
def pick(r): return int(r["picked"] == r["decoy"])
def hit(r): return int(r["hit"])


def load():
    if not os.path.exists(RES):
        raise SystemExit(f"{RES}가 없습니다. 먼저 d5_run.py --set d8 로 돌리세요.")
    return [r for r in csv.DictReader(open(RES, encoding="utf-8-sig")) if r["status"] == "ok"]


N_CAL = 20          # 사전 점검에 쓰는 레이아웃 (0~19)
N_LAY = 60


def cell(rows, st, cond, scale, target=None, cal=True):
    return [r for r in rows if r["cond"] == f"{st}_{cond}" and r["scale"] == scale
            and (target is None or r["target"] == target) and (not cal or int(r["layout"]) < N_CAL)]


def safety(rows, st):
    out = {}
    for c in ("Tgray", "Igray"):
        v = cell(rows, st, c, "1.0")
        out[c] = (mean([hit(r) for r in v]), len(v)) if v else None
    ok = all(x is not None and x[0] >= 0.9 for x in out.values())
    return ok, out


def choose_scale(rows, st):
    sc = sorted({r["scale"] for r in rows if r["cond"] == f"{st}_Tgray" and r["scale"] != "1.0"
                 and int(r["layout"]) < N_CAL}, key=float, reverse=True)
    lines, cands = [], []
    for s in sc:
        acc = {}
        for t in TARGETS:
            v = cell(rows, st, "Tgray", s, t)
            acc[t] = (mean([hit(r) for r in v]) if v else None, len(v))
        full = all(acc[t][1] == 40 for t in TARGETS)
        a = [acc[t][0] for t in TARGETS]
        tot = mean(a) if None not in a else None
        ok = full and all(0.5 <= x <= 0.8 for x in a) and abs(a[0] - a[1]) < 0.3
        lines.append(f"    x{s:<6} " + "  ".join(f"{t} {acc[t][0]*100:5.1f}% ({acc[t][1]}개)" if acc[t][0] is not None
                                                 else f"{t} -" for t in TARGETS)
                     + (f"   합 {tot*100:5.1f}%" if tot is not None else "") + ("   ← 후보" if ok else "")
                     + ("" if full else "   (40개가 안 참)"))
        if ok:
            cands.append((abs(tot - 0.65), -float(s), s))
    best = sorted(cands)[0][2] if cands else None
    return lines, best, sc


def precheck(rows):
    print("=== 사전 점검 (레이아웃 0~19번만 사용) ===")
    for st in STRATA:
        print(f"\n[{NAME[st]}]")
        ok, out = safety(rows, st)
        for c, x in out.items():
            print(f"  x1.0 {c}: " + (f"{x[0]*100:.1f}% (n={x[1]})" if x else "안 함"))
        print(f"  안전 점검 → {'통과' if ok else ('아직 안 함/덜 됨' if None in out.values() else '90% 미만: 이 층 해석 보류')}")
        lines, best, sc = choose_scale(rows, st)
        if not sc:
            print("  배율 탐색 결과 없음")
            continue
        print("  Tgray 정확도 (배율 선택용):")
        print("\n".join(lines))
        if best:
            print(f"  → 규칙상 권장 배율: x{best}")
        else:
            accs = [mean([hit(r) for r in cell(rows, st, "Tgray", s)]) for s in sc]
            hint = "더 작은 배율(더 어렵게)" if min(accs) > 0.8 else "더 큰 배율(더 쉽게)" if max(accs) < 0.5 else "사이 배율"
            print(f"  → 후보 없음. {hint}을 더 재야 함")


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
    return dict(n=len(items), nl=len(L), obs=obs, p=p, lo=lo, hi=hi)


def line(label, st, alpha=0.05, holm=False):
    if st is None:
        print(f"  {label}: 레이아웃이 너무 적어 검정하지 않음"); return False
    sup = st["p"] < alpha and (st["lo"] > 0 or st["hi"] < 0)
    tag = f" (Holm 기준 p<{alpha:.4f})" if holm and alpha > 0 else " (Holm: 앞 단계에서 멈춰 지지 불가)" if holm else ""
    print(f"  {label}: 단위 {st['n']}개(레이아웃 {st['nl']}), 평균 {st['obs']*100:+.1f}%p, 순열검정 p = {st['p']:.4f}, "
          f"부트스트랩 95% [{st['lo']*100:+.1f}, {st['hi']*100:+.1f}]%p  → {'지지' if sup else '지지 안 됨'}{tag}")
    return sup


def holm(ps):
    """ps: {key: p}. 반환 {key: 그 검정의 기준 alpha (0이면 앞 단계 실패)}"""
    order = sorted(ps, key=lambda k: ps[k]); m = len(order); out = {}; alive = True
    for i, k in enumerate(order):
        a = 0.05 / (m - i)
        out[k] = a if alive else 0.0
        if not (ps[k] < a):
            alive = False
    return out


def units(rows, st, scale):
    U = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["scale"] == scale and r["cond"].startswith(st + "_"):
            U[(int(r["layout"]), int(r["mirror"]))][r["target"]][r["cond"][3:]] = r
    return {k: v for k, v in U.items() if all(len(v[t]) == 6 for t in TARGETS)}


def analyze(rows, st, scale):
    print("\n" + "=" * 70 + f"\n{NAME[st]}   배율 x{scale}")
    ok, out = safety(rows, st)
    print(f"[안전 점검] x1.0 " + ", ".join(f"{c} {x[0]*100:.1f}%" if x else f"{c} 안 함" for c, x in out.items())
          + f" → {'통과' if ok else '통과 못 함: 이 층은 해석 보류 (숫자는 참고로만 출력)'}")
    U = units(rows, st, scale)
    print(f"완성된 단위(레이아웃x거울) {len(U)}개" + ("" if len(U) == 2 * N_LAY else f"  ← {2*N_LAY}개가 아님: 덜 돈 상태라 해석하지 말 것"))
    if not U:
        return None

    def m(k, c, f=pick, t=None):
        return mean([f(U[k][tt][c]) for tt in ((t,) if t else TARGETS)])

    print("\n  기술 통계 (두 목표 합산)        정확도   방해 선택")
    for f in FORMS:
        for c in COLS:
            print(f"    {f}{c:5} ({'방해=글자' if f == 'T' else '방해=아이콘'}, {c:4})   "
                  f"{mean([m(k, f + c, hit) for k in U])*100:6.1f}%   {mean([m(k, f + c) for k in U])*100:6.1f}%")
    print("  목표별 방해 선택 (색 관습 확인용):")
    for t in TARGETS:
        print(f"    {t}: " + "  ".join(f"{f}{c} {mean([m(k, f + c, t=t) for k in U])*100:4.1f}%" for f in FORMS for c in COLS))

    colpull = lambda k, f: (m(k, f + "red") + m(k, f + "blue")) / 2 - m(k, f + "gray")
    rb = lambda k, f, t: m(k, f + "red", t=t) - m(k, f + "blue", t=t)

    print("\n  [조작 확인] 색 끌림 = 평균(red,blue) - gray, 두 방해 형태 평균")
    man = line("색 끌림", stats([(k[0], mean([colpull(k, f) for f in FORMS])) for k in U]))

    S = {1: stats([(k[0], m(k, "Igray") - m(k, "Tgray")) for k in U]),
         2: stats([(k[0], colpull(k, "I") - colpull(k, "T")) for k in U]),
         3: stats([(k[0], mean([rb(k, f, "삭제") - rb(k, f, "저장") for f in FORMS])) for k in U])}
    A = holm({i: S[i]["p"] for i in S})
    print("\n  [주 검정 1] 모양 끌림 = Igray - Tgray")
    r1 = line("모양 끌림", S[1], A[1], True)
    print("  [주 검정 2] 모양 x 색 = (아이콘 방해의 색 끌림) - (글자 방해의 색 끌림)")
    r2 = line("모양 x 색", S[2], A[2], True)
    print("  [주 검정 3] 색 관습 = (삭제: 빨-파) - (저장: 빨-파), 두 방해 형태 평균 (예측 +)")
    r3 = line("색 관습", S[3], A[3], True)

    print("\n  [보조, 탐색]")
    for f in FORMS:
        line(f"색 끌림, 방해={'글자' if f == 'T' else '아이콘'}", stats([(k[0], colpull(k, f)) for k in U]))
    line("빨강 - 파랑 (두 목표·두 형태 평균)", stats([(k[0], mean([rb(k, f, t) for f in FORMS for t in TARGETS])) for k in U]))
    line("삭제에서 빨-파 (예측 +)", stats([(k[0], mean([rb(k, f, "삭제") for f in FORMS])) for k in U]))
    line("저장에서 빨-파 (예측 -)", stats([(k[0], mean([rb(k, f, "저장") for f in FORMS])) for k in U]))
    for f in FORMS:
        line(f"색 관습, 방해={'글자' if f == 'T' else '아이콘'}만",
             stats([(k[0], rb(k, f, "삭제") - rb(k, f, "저장")) for k in U]))
    line("정확도 Igray - Tgray", stats([(k[0], m(k, "Igray", hit) - m(k, "Tgray", hit)) for k in U]))
    line("정확도 평균(색) - gray, 두 형태 평균",
         stats([(k[0], mean([(m(k, f + "red", hit) + m(k, f + "blue", hit)) / 2 - m(k, f + "gray", hit) for f in FORMS]))
                for k in U]))

    print("\n  [판정 (미리 정한 규칙)]")
    if not ok:
        print("    안전 점검 통과 못 함 → 이 층은 해석 보류.")
        return dict(U=U, m=m)
    s1 = S[1]["obs"]
    print("    [1] " + (f"모양 끌림 {'> 0' if s1 > 0 else '< 0'} 지지 → "
                         + ("혼자 모양이 다른 버튼에 오답이 몰린다." if s1 > 0 else "아이콘 방해가 오히려 덜 골라진다(그대로 보고).")
                         if r1 else "모양 끌림이 0과 구분되지 않음 → 확인되지 않음(효과 없음과는 다름)."))
    if not man:
        print("    [2][3] 색 끌림이 지지되지 않음 → 모양 x 색, 색 관습은 해석 보류.")
    else:
        print("    [2] " + (f"모양 x 색 {'> 0' if S[2]['obs'] > 0 else '< 0'} 지지 → 색 효과가 방해 형태에 따라 다름"
                             f"({'아이콘에서 더 큼' if S[2]['obs'] > 0 else '아이콘에서 더 작음'})." if r2
                             else "모양 x 색이 0과 구분되지 않음 → 색 효과가 형태에 따라 다르다는 증거 없음."))
        print("    [3] " + ("색 관습 > 0 지지 → 기능에 어울리는 색(삭제-빨강, 저장-파랑)의 방해 버튼에 더 끌림."
                             if r3 and S[3]["obs"] > 0 else
                             "색 관습 < 0 지지 → 관습과 반대 방향(그대로 보고)." if r3 else
                             "색 관습이 0과 구분되지 않음 → 이 층에서 색 관습 효과는 확인되지 않음."))
    return dict(U=U, m=m)


def main(args):
    rows = load()
    sc = dict(a.split("=") for a in args)
    res = {}
    for st in STRATA:
        if st in sc:
            res[st] = analyze(rows, st, sc[st])
    if len(res) == 2 and all(res.values()) and sc["TT"] == sc["IT"]:
        print("\n" + "=" * 70 + "\n[층 비교, 탐색] 같은 배율이라 비교 가능: (IT 값) - (TT 값), 같은 레이아웃끼리 짝")
        UT, UI = res["TT"]["U"], res["IT"]["U"]
        mt, mi = res["TT"]["m"], res["IT"]["m"]
        ks = [k for k in UT if k in UI]
        line("모양 끌림 차이", stats([(k[0], (mi(k, "Igray") - mi(k, "Tgray")) - (mt(k, "Igray") - mt(k, "Tgray"))) for k in ks]))
        cp = lambda mm, k: mean([(mm(k, f + "red") + mm(k, f + "blue")) / 2 - mm(k, f + "gray") for f in FORMS])
        line("색 끌림 차이", stats([(k[0], cp(mi, k) - cp(mt, k)) for k in ks]))
        line("정확도 차이 (Tgray)", stats([(k[0], mi(k, "Tgray", hit) - mt(k, "Tgray", hit)) for k in ks]))
    elif len(res) == 2:
        print("\n[층 비교] 두 층의 배율이 달라 층끼리 숫자를 직접 비교하지 않는다(난이도가 다름).")
    print("\n※ 단일 모델·합성 UI·기능 지시문·한 층당 한 배율·레이아웃 60개 기준.")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        precheck(load())
    else:
        main(a)