#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d3 분석.

주 지표  : 박스 밖(= 1 - 박스-안) 비율 — ScreenSpot 방식의 '실패'. 조건별 Wilson 95% 신뢰구간.
보조 지표: 완전실패(오차 > 화면 짧은 변의 절반), 오차 중앙값(완전실패 제외), dx/dy 중앙값.

검정: 층화 순열검정. 설계가 (레이아웃, 거울) 블록마다 9조건을 한 번씩 담고 있으므로,
      '같은 블록 안에서만' 조건 라벨을 섞어 귀무분포를 만든다 -> 레이아웃·거울 난이도 차이가 자동으로 통제됨.
      카이제곱 임계값 비교를 쓰지 않는 이유: 정답률이 천장 근처면 기대 빈도가 5 미만이라 근사가 성립하지 않음.
  (a) 색조 효과 : 8색조 사이에 실패 빈도 차이가 있는가
  (b) 기준선    : 회색 목표 vs 색 있는 목표(8색조 합)의 실패율 차이

사용: python d3_analyze.py [결과.csv] [--perm 5000]
"""
import csv, math, random, statistics, sys

Z = 1.96

def wilson(k, n):
    if n == 0: return float('nan'), float('nan'), float('nan')
    p = k / n; d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z / d * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)

def load(path):
    rows = []
    for r in csv.DictReader(open(path, encoding='utf-8-sig')):
        r['layout'] = int(r['layout']); r['mirror'] = int(r['mirror'])
        r['miss'] = 0 if r['box_in'] == '1' else 1
        r['cfail'] = int(r['fail'])
        rows.append(r)
    return rows

def blocks_of(rows):
    b = {}
    for r in rows: b.setdefault((r['layout'], r['mirror']), []).append(r)
    return list(b.values())

def hue_test(rows, key, nperm, rng):
    hr = [r for r in rows if r['cond'] != 'gray']
    N = len(hr); M = sum(r[key] for r in hr)
    if N == 0 or M == 0 or M == N: return None
    hues = sorted({r['cond'] for r in hr}, key=int)
    nh = {h: sum(1 for r in hr if r['cond'] == h) for h in hues}
    exp = {h: M * nh[h] / N for h in hues}
    bl = [([r['cond'] for r in b], [r[key] for r in b]) for b in blocks_of(hr)]
    def stat(blocks):
        cnt = {h: 0 for h in hues}
        for labs, ys in blocks:
            for l, y in zip(labs, ys): cnt[l] += y
        return sum((cnt[h] - exp[h]) ** 2 / exp[h] for h in hues)
    obs = stat(bl); ge = 0
    for _ in range(nperm):
        perm = []
        for labs, ys in bl:
            l2 = labs[:]; rng.shuffle(l2); perm.append((l2, ys))
        if stat(perm) >= obs - 1e-9: ge += 1
    return obs, (1 + ge) / (1 + nperm)

def baseline_test(rows, key, nperm, rng):
    if not any(r['cond'] == 'gray' for r in rows) or not any(r['cond'] != 'gray' for r in rows): return None
    M = sum(r[key] for r in rows)
    if M == 0 or M == len(rows): return None
    bl = [([r['cond'] == 'gray' for r in b], [r[key] for r in b]) for b in blocks_of(rows)]
    def stat(blocks):
        sg = ng = sc = nc = 0
        for labs, ys in blocks:
            for l, y in zip(labs, ys):
                if l: sg += y; ng += 1
                else: sc += y; nc += 1
        return (sg / ng - sc / nc) if ng and nc else 0.0
    obs = stat(bl); ge = 0
    for _ in range(nperm):
        perm = []
        for labs, ys in bl:
            l2 = labs[:]; rng.shuffle(l2); perm.append((l2, ys))
        if abs(stat(perm)) >= abs(obs) - 1e-9: ge += 1
    return obs, (1 + ge) / (1 + nperm)

def main(path="d3_results.csv", nperm=5000):
    import os
    if not os.path.exists(path):
        print(f"결과 파일이 없습니다: {path}"); return
    rows = load(path)
    if not rows: print("결과 없음"); return
    rng = random.Random(2026)
    n = len(rows); M = sum(r['miss'] for r in rows); C = sum(r['cfail'] for r in rows)
    npz = sum(1 for r in rows if r.get('status') == 'no_parse')
    print(f"\n{'=' * 74}\n분석: {path}   n={n}\n{'=' * 74}")
    print(f"전체: 박스 밖 {M}/{n} ({M / n * 100:.1f}%)  완전실패 {C}/{n} ({C / n * 100:.1f}%)  좌표 못 냄(no_parse) {npz}")

    conds = (['gray'] if any(r['cond'] == 'gray' for r in rows) else []) + sorted({r['cond'] for r in rows if r['cond'] != 'gray'}, key=int)
    print(f"\n{'조건':>6} | {'n':>3} | {'박스밖':>6} | {'실패율':>6} | {'95% CI':>15} | {'완전실패':>6} | {'오차중앙':>7} | {'dx':>6} | {'dy':>6}")
    print('-' * 80)
    uppers = []
    for c in conds:
        g = [r for r in rows if r['cond'] == c]; k = sum(r['miss'] for r in g)
        p, lo, hi = wilson(k, len(g)); uppers.append(hi)
        good = [r for r in g if r.get('status') == 'ok' and r['cfail'] == 0]
        med = statistics.median([float(r['err']) for r in good]) if good else float('nan')
        dxm = statistics.median([float(r['dx']) for r in good]) if good else float('nan')
        dym = statistics.median([float(r['dy']) for r in good]) if good else float('nan')
        print(f"{c:>6} | {len(g):>3} | {k:>3}/{len(g):<2} | {p * 100:>5.1f}% | [{lo * 100:>4.1f},{hi * 100:>5.1f}]% | "
              f"{sum(r['cfail'] for r in g):>3}/{len(g):<2} | {med:>5.1f}px | {dxm:>+5.1f} | {dym:>+5.1f}")

    # 천장 판정
    print()
    if M == 0:
        print(f"⚠ 천장: 박스 밖이 0건입니다. 모든 조건에서 정답률이 상한이라 어떤 효과도 검정할 수 없습니다.")
        print(f"  이 데이터로 말할 수 있는 건 '조건별 실패율의 95% 상한이 최대 {max(uppers) * 100:.1f}%'뿐입니다.")
        print("  (= 그보다 큰 효과는 배제되지만, 그보다 작은 효과는 이 난이도에서 볼 수 없음. '효과 없음'이 아님)")
        return
    if M / n < 0.05:
        print(f"⚠ 천장 근처: 전체 실패율 {M / n * 100:.1f}%. 검정력이 매우 낮습니다. 결과를 '효과 없음'으로 읽지 마세요.")

    # 검정
    print(f"층화 순열검정 (블록=레이아웃x거울, {nperm}회)")
    for key, name in (('miss', '박스 밖'), ('cfail', '완전실패')):
        h = hue_test(rows, key, nperm, rng)
        if h is None: print(f"  (a) 색조 효과[{name}]: 해당 사건이 없거나 전부여서 검정 불가")
        else:
            obs, p = h
            print(f"  (a) 색조 효과[{name}]: 통계량 {obs:.2f}, p = {p:.4f}  → "
                  + ("색조 간 차이가 우연으로 설명되지 않음(탐색적 — 후속 확인 필요)" if p < 0.05 else "색조 간 차이를 우연과 구분할 수 없음"))
        b = baseline_test(rows, key, nperm, rng)
        if b is None: print(f"  (b) 기준선[{name}]: 회색 조건이 없거나 검정 불가")
        else:
            d, p = b
            print(f"  (b) 기준선[{name}]: 회색 - 색조합 = {d * 100:+.1f}%p, p = {p:.4f}  → "
                  + ("차이가 우연으로 설명되지 않음" if p < 0.05 else "차이를 우연과 구분할 수 없음"))
    print("  ※ (b)는 '색조의 종류'가 아니라 '목표가 방해 버튼들 사이에서 색을 가졌는가'(눈에 띔 포함)의 효과입니다.")
    print("    색조 자체의 효과는 (a)로만 판단하세요.")

    # 진단: 실패가 특정 레이아웃/거울에 몰렸는지
    by_l = {}
    for r in rows: by_l[r['layout']] = by_l.get(r['layout'], 0) + r['miss']
    top = sorted(by_l.items(), key=lambda x: -x[1])[:5]
    share = sum(v for _, v in top[:3]) / M
    print(f"\n[진단] 박스 밖이 많은 레이아웃 상위: " + ", ".join(f"L{l:02d}={v}" for l, v in top) + f"  (상위 3개가 전체 실패의 {share * 100:.0f}%)")
    if M >= 6 and share > 0.5:
        print("  → 실패가 소수 레이아웃에 몰려 있습니다. 색보다 배치 난이도가 지배적일 수 있으니 그 이미지를 직접 확인하세요.")
    for m in (0, 1):
        g = [r for r in rows if r['mirror'] == m]
        if g: print(f"[진단] 거울={m}: 박스 밖 {sum(r['miss'] for r in g)}/{len(g)}")

if __name__ == "__main__":
    a = sys.argv[1:]; npz = 5000
    if '--perm' in a:
        i = a.index('--perm'); npz = int(a[i + 1]); del a[i:i + 2]
    main(a[0] if a else "d3_results.csv", npz)
