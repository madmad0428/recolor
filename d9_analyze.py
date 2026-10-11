#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d9 분석 — 선생님 메모 검증: 8색조 + 회색 기준선, 색별 성공률(윌슨 구간·카이제곱), 실패 분류. API는 부르지 않는다.

사용
  python d9_analyze.py                         # 사전 점검: 회색 정확도 x 배율 -> 난이도(배율) 고르기
  python d9_analyze.py 0.13                    # 본 분석 (배율은 사전 점검으로 미리 고정)
  python d9_analyze.py 0.13 --compare d9_results_run2.csv   # 같은 실험을 한 번 더 돌렸을 때 1회차와 비교(기술만)
  옵션: --file 결과파일 (기본 d9_results.csv)

정답 판정: 모델이 낸 좌표(0~1000 정규화, x 먼저)를 이미지 크기로 바꾼 점이 목표 버튼 사각형 안(경계 포함)이면 정답.
          좌표를 못 읽은 응답(파싱 실패)은 오답으로 센다(전체 1,080개 중 몇 개인지 따로 표시).

단위·검정 (결과를 보고 바꾸지 않는다)
  단위 = (레이아웃, 거울) 블록 하나 = 같은 배치에서 9조건이 한 묶음. 같은 배치의 두 거울은 독립이 아니므로
  신뢰구간·순열검정은 레이아웃 단위로 한다.
  [사전 점검]  회색 정확도가 50~80%인 배율 중 65%에 가장 가까운 것(동률이면 큰 배율). 없으면 멈추고 의논.
  [주 검정 1]  색 있는 8색 평균 - 회색 정확도. 레이아웃 단위 부호 뒤집기 순열검정 + 레이아웃 부트스트랩 95%.
  [주 검정 2]  8색 사이 차이(전체 검정). 블록 안 순열 카이제곱(주 판정) + 일반 카이제곱(참고; 천장 근처라 기대도수<5가 생길 수 있고 배치 짝 구조를 무시).
  주 검정 2개 -> Holm: 작은 p는 0.025, 큰 p는 0.05 미만이어야 '지지'.
  [색별 탐색]  8색 각각 - 회색. Holm(8개). 색당 120장이라 작은 차이는 검출력이 낮다(미검출 = 효과 없음 아님).
  '지지' = 보정된 기준을 통과하고 부트스트랩 구간이 0을 포함하지 않을 때만.

실패 분류 (오답 하나에 하나씩, 위에서부터 먼저 해당하는 것)
  1 응답 형식      : 좌표를 못 읽음
  2 좌표 변환 의심 : 값이 0~1000 밖이거나, x/y를 바꾸거나 '줄인 그림의 픽셀' 또는 '원본 픽셀'로 읽으면 목표 버튼 안
  3 배치 문제 의심 : 그 배치의 18장(9조건x거울2) 중 80% 이상이 오답 (색과 무관하게 어려운 배치)
  4a 다른 버튼 선택 / 4b 버튼 밖(목표에서 40px 이내) / 4c 버튼 밖(멀리)  = 모델 오답
"""
import csv, json, math, os, re, sys
from collections import Counter, defaultdict
import numpy as np

W, H = 880, 600
HUES = ["20", "65", "110", "155", "200", "245", "290", "335"]
CONDS = HUES + ["gray"]
RES = "d9_results.csv"
MAN = "d9_manifest.csv"
IMG = "d9_img"
NPERM = 20000
rng = np.random.default_rng(2031)


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def chi2_sf(x, k):
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, k))
    except Exception:
        a, xx = k / 2, x / 2                       # 정규화된 불완전 감마 Q(a, x)
        if xx <= 0: return 1.0
        if xx < a + 1:
            s = t = 1 / a; n = a
            for _ in range(1000):
                n += 1; t *= xx / n; s += t
                if abs(t) < abs(s) * 1e-14: break
            return 1 - s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        b = xx + 1 - a; c = 1e300; d = 1 / b; h = d
        for i in range(1, 1000):
            an = -i * (i - a); b += 2
            d = an * d + b; d = 1e-300 if abs(d) < 1e-300 else d
            c = b + an / c; c = 1e-300 if abs(c) < 1e-300 else c
            d = 1 / d; de = d * c; h *= de
            if abs(de - 1) < 1e-14: break
        return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def load(path=None):
    path = path or arg("--file", RES)
    if not os.path.exists(path):
        raise SystemExit(f"{path}가 없습니다. 먼저 d5_run.py --set d9 로 돌리세요.")
    return [r for r in csv.DictReader(open(path, encoding="utf-8-sig"))]


def hit(r): return int(r["hit"])


# ----------------------------------------------------------------------------- 사전 점검
def precheck(rows):
    print("=== 사전 점검: 회색 기준선 정확도 x 배율 (난이도 고르기) ===")
    g = defaultdict(list)
    for r in rows:
        if r["cond"] == "gray": g[r["scale"]].append(r)
    if not g:
        print("  회색 결과가 아직 없습니다. py d5_run.py --set d9 --conds gray --layouts 0-29 --scales 0.14,0.135,0.13,0.125")
        return
    cand = []
    for sc in sorted(g, key=float, reverse=True):
        v = g[sc]; k = sum(hit(r) for r in v); p, lo, hi = wilson(k, len(v))
        npar = sum(1 for r in v if r["status"] != "ok")
        print(f"  x{sc:<6} n={len(v):3}  정확도 {p*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]  ({k}/{len(v)})"
              + (f"   좌표 못 읽음 {npar}" if npar else ""))
        if len(v) >= 50 and 0.50 <= p <= 0.80: cand.append((abs(p - 0.65), -float(sc), sc, p))
    print("\n[규칙] 회색 정확도 50~80% 중 65%에 가장 가까운 배율 (동률이면 큰 배율), 회색 n>=50일 때만 후보")
    if cand:
        cand.sort(); sc, p = cand[0][2], cand[0][3]
        print(f"  → 권고: x{sc} (회색 {p*100:.1f}%)")
        print(f"  다음: py d5_run.py --set d9 --scales {sc}     (회색 60장은 이미 해서 건너뛰고 나머지를 돈다)")
        print(f"        py d9_analyze.py {sc}")
    else:
        print("  → 조건을 만족하는 배율이 없음 (또는 회색 n<50). 멈추고 의논.")


# ----------------------------------------------------------------------------- 레이아웃 단위 검정
def layout_stats(items):
    """items = [(layout, value)]. 레이아웃 단위 부호 뒤집기 순열검정 + 레이아웃 부트스트랩."""
    by = defaultdict(list)
    for l, x in items: by[l].append(x)
    L = sorted(by)
    S = np.array([sum(by[l]) for l in L], float); Nn = np.array([len(by[l]) for l in L], float)
    obs = S.sum() / Nn.sum()
    ge = 0
    for _ in range(NPERM // 5000):
        f = rng.choice([-1.0, 1.0], size=(5000, len(L)))
        ge += int((np.abs(f @ S / Nn.sum()) >= abs(obs) - 1e-12).sum())
    p = (1 + ge) / (1 + NPERM)
    idx = rng.integers(0, len(L), size=(NPERM // 4, len(L)))
    bm = S[idx].sum(1) / Nn[idx].sum(1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return dict(n=len(items), nl=len(L), obs=obs, p=p, lo=float(lo), hi=float(hi))


def verdict(st, alpha):
    return alpha > 0 and st["p"] < alpha and (st["lo"] > 0 or st["hi"] < 0)


def show(label, st, alpha=0.05, note=""):
    sup = verdict(st, alpha)
    print(f"  {label}: 블록 {st['n']}개(레이아웃 {st['nl']}), 평균 {st['obs']*100:+.1f}%p, 순열검정 p = {st['p']:.4f}, "
          f"부트스트랩 95% [{st['lo']*100:+.1f}, {st['hi']*100:+.1f}]%p → {'지지' if sup else '지지 안 됨'}"
          + (f" (기준 p<{alpha:.4g})" if alpha not in (0.05,) else "") + note)
    return sup


def block_perm_chi2(M):
    """M[B,8] 0/1 정답표. 블록 안에서 8색 값을 섞는 순열검정. 통계량 = 색별 정답 수의 카이제곱."""
    B, K = M.shape; tot = M.sum()
    def T(cols):
        e = tot / K
        return float(((cols - e) ** 2 / e).sum() + (((B - cols) - (B - e)) ** 2 / (B - e)).sum()) if 0 < e < B else 0.0
    obs = T(M.sum(0))
    ge = 0; done = 0
    while done < NPERM:
        c = min(2000, NPERM - done)
        order = np.argsort(rng.random((c, B, K)), axis=2)
        P = np.take_along_axis(np.broadcast_to(M, (c, B, K)), order, axis=2)   # 블록마다 색 위치를 섞음
        cols = P.sum(1)                                                       # [c, K]  (섞은 뒤 색별 정답 수)
        e = tot / K
        t = (((cols - e) ** 2 / e).sum(1) + (((B - cols) - (B - e)) ** 2 / (B - e)).sum(1)) if 0 < e < B else np.zeros(c)
        ge += int((t >= obs - 1e-9).sum()); done += c
    return obs, (1 + ge) / (1 + NPERM)


# ----------------------------------------------------------------------------- 실패 분류
def nums(raw):
    mx = re.search(r'"x"\s*:\s*(-?\d+(?:\.\d+)?)', raw); my = re.search(r'"y"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if mx and my: return float(mx.group(1)), float(my.group(1))
    v = re.findall(r'-?\d+(?:\.\d+)?', raw)
    return (float(v[0]), float(v[1])) if len(v) >= 2 else None


def inside(px, py, b): return b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]


def dist_rect(px, py, b):
    dx = max(b["x0"] - px, 0, px - b["x1"]); dy = max(b["y0"] - py, 0, py - b["y1"])
    return math.hypot(dx, dy)


def classify(rows_miss, man, scale, layout_fail):
    out = []
    for r in rows_miss:
        m = man[r["file"]]; boxes = json.loads(m["boxes"]); tb = boxes[0]
        cat = sub = ""
        ab = nums(r["raw"]) if r["status"] == "ok" else None
        if r["status"] != "ok":
            cat, sub = "1 응답 형식", "좌표를 못 읽음"
        else:
            a, b = ab if ab else (None, None)
            alts = []
            if a is not None:
                if not (0 <= a <= 1000 and 0 <= b <= 1000): alts.append("범위 밖 값")
                if inside(b / 1000 * W, a / 1000 * H, tb): alts.append("x·y 바꿔 읽으면 정답")
                if inside(a / scale, b / scale, tb): alts.append("줄인 그림 픽셀로 읽으면 정답")
                if inside(a, b, tb): alts.append("원본 픽셀로 읽으면 정답")
            if alts:
                cat, sub = "2 좌표 변환 의심", "; ".join(alts)
            elif layout_fail[int(r["layout"])] >= 0.8:
                cat, sub = "3 배치 문제 의심", f"그 배치 오답률 {layout_fail[int(r['layout'])]*100:.0f}%"
            else:
                px, py = float(r["pred_x"]), float(r["pred_y"])
                if r["picked"] != "(버튼 밖)":
                    ctr = lambda bb: ((bb["x0"] + bb["x1"]) / 2, (bb["y0"] + bb["y1"]) / 2)
                    tc = ctr(tb)
                    order = sorted(boxes[1:], key=lambda bb: math.hypot(ctr(bb)[0] - tc[0], ctr(bb)[1] - tc[1]))
                    near = order[0]["label"] == r["picked"]
                    cat, sub = "4a 다른 버튼 선택", ("가장 가까운 버튼" if near else "가장 가까운 버튼은 아님")
                else:
                    d = dist_rect(px, py, tb)
                    cat, sub = ("4b 버튼 밖(근접)", f"{d:.0f}px") if d <= 40 else ("4c 버튼 밖(멀리)", f"{d:.0f}px")
        out.append(dict(file=r["file"], layout=r["layout"], mirror=r["mirror"], cond=r["cond"], category=cat, detail=sub,
                        picked=r["picked"], pred_x=r["pred_x"], pred_y=r["pred_y"], err=r["err"], raw=r["raw"]))
    return out


def gallery(fails, scale, path="d9_failures.png"):
    from PIL import Image, ImageDraw, ImageFont
    fp = next((p for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "C:/Windows/Fonts/malgun.ttf",
                           "/System/Library/Fonts/AppleSDGothicNeo.ttc"] if os.path.exists(p)), None)
    font = ImageFont.truetype(fp, 15) if fp else ImageFont.load_default()
    tw, th, per = 352, 240, 3
    sheet = Image.new("RGB", (tw * per, (th + 22) * len(CONDS)), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    man = {r["file"]: r for r in csv.DictReader(open(MAN, encoding="utf-8-sig"))}
    for ri, c in enumerate(CONDS):
        fl = sorted([f for f in fails if f["cond"] == c], key=lambda f: (f["layout"], f["mirror"]))
        pick = [fl[i] for i in np.linspace(0, len(fl) - 1, min(per, len(fl))).astype(int)] if fl else []
        d.text((6, ri * (th + 22) + 2), f"{'회색' if c == 'gray' else '색조 ' + c}  — 오답 {len(fl)}개 중 {len(pick)}개 (모델이 본 줄인 그림, 초록=정답 버튼, 빨강 +=모델 답)", fill=(0, 0, 0), font=font)
        for ci, f in enumerate(pick):
            im = Image.open(os.path.join(IMG, f["file"])).convert("RGB")
            im = im.resize((round(W * scale), round(H * scale)), Image.LANCZOS).resize((tw, th), Image.NEAREST)
            dd = ImageDraw.Draw(im); b = json.loads(man[f["file"]]["boxes"])[0]; k = tw / W
            dd.rectangle([b["x0"] * k, b["y0"] * k, b["x1"] * k, b["y1"] * k], outline=(0, 200, 0), width=2)
            try:
                px, py = float(f["pred_x"]) * k, float(f["pred_y"]) * k
                dd.line([px - 7, py, px + 7, py], fill=(255, 0, 0), width=2); dd.line([px, py - 7, px, py + 7], fill=(255, 0, 0), width=2)
            except ValueError:
                pass
            dd.rectangle([0, th - 18, tw, th], fill=(255, 255, 255)); dd.text((3, th - 17), f"L{int(f['layout']):02d} {'M' if f['mirror'] == '1' else 'N'} {f['category']}", fill=(0, 0, 0), font=font)
            sheet.paste(im, (ci * tw, ri * (th + 22) + 22))
    sheet.save(path)
    return path


# ----------------------------------------------------------------------------- 본 분석
def main(scale):
    rows = [r for r in load() if r["scale"] == scale and r["cond"] in CONDS]
    man = {r["file"]: r for r in csv.DictReader(open(MAN, encoding="utf-8-sig"))}
    U = defaultdict(dict)
    for r in rows: U[(int(r["layout"]), int(r["mirror"]))][r["cond"]] = r
    U = {k: v for k, v in U.items() if len(v) == 9}
    rows = [r for k in U for r in U[k].values()]
    print(f"분석 배율: x{scale}   완성된 블록(레이아웃x거울) {len(U)}개 / 120" + ("" if len(U) == 120 else "  ← 120개가 아님: 덜 돈 상태라 해석하지 말 것"))
    if len(U) < 20: return
    npar = sum(1 for r in rows if r["status"] != "ok")
    print(f"좌표를 못 읽은 응답 {npar}/{len(rows)}개 (오답으로 셈)")
    print("정답 판정: 모델 좌표(0~1000 정규화, x 먼저)를 이미지 크기로 바꾼 점이 목표 버튼 사각형 안(경계 포함)이면 정답.")

    # --- 색별 정답률
    print("\n=== 1. 색별 정답률 (윌슨 95% 구간) ===")
    print(f"  {'조건':>6} {'n':>4} {'정답':>5} {'오답':>5} {'정답률':>7}   윌슨 95%        회색 대비")
    tab = {}
    g_acc = np.mean([hit(U[k]['gray']) for k in U])
    for c in CONDS:
        k = sum(hit(U[b][c]) for b in U); n = len(U); p, lo, hi = wilson(k, n); tab[c] = k
        print(f"  {c:>6} {n:4} {k:5} {n-k:5} {p*100:6.1f}%   [{lo*100:5.1f}, {hi*100:5.1f}]   " + ("(기준선)" if c == "gray" else f"{(p-g_acc)*100:+6.1f}%p"))
    if not (0.45 <= g_acc <= 0.85): print(f"  ⚠ 회색 정확도 {g_acc*100:.1f}%: 사전 점검 목표(50~80%) 밖. 천장/바닥 효과로 해석 주의.")
    if max(tab[c] / len(U) for c in CONDS) >= 0.98: print("  ⚠ 정답률 98% 이상인 조건이 있음: 천장.")

    # --- 카이제곱 (일반)
    M = np.array([[hit(U[b][c]) for c in HUES] for b in sorted(U)])         # [블록, 8색]
    obs_cols = M.sum(0); B = len(M); tot = obs_cols.sum()
    e_hit, e_miss = tot / 8, (B * 8 - tot) / 8
    chi = float(((obs_cols - e_hit) ** 2 / e_hit).sum() + (((B - obs_cols) - e_miss) ** 2 / e_miss).sum())
    p_chi = chi2_sf(chi, 7)
    print("\n=== 2. 8색 사이 차이 (전체 검정) ===")
    print(f"  [참고] 일반 카이제곱(정답/오답 x 8색, 자유도 7): χ² = {chi:.2f}, p = {p_chi:.4f}   "
          f"(기대 오답 수/색 {e_miss:.1f}{' — 5 미만이라 근사 불안' if e_miss < 5 else ''})")
    obs_t, p_bp = block_perm_chi2(M)
    print(f"  [주 판정] 블록 안 순열 카이제곱: χ² = {obs_t:.2f}, p = {p_bp:.4f}   (같은 배치끼리 묶인 구조를 반영)")

    # --- 주 검정
    col_ = lambda b: float(np.mean([hit(U[b][c]) for c in HUES]))
    st1 = layout_stats([(b[0], col_(b) - hit(U[b]['gray'])) for b in U])
    st2 = dict(p=p_bp, n=len(U), nl=len({b[0] for b in U}), obs=float('nan'), lo=0, hi=0)
    if st1["p"] <= p_bp: a1, a2 = 0.025, (0.05 if st1["p"] < 0.025 else 0.0)
    else: a2, a1 = 0.025, (0.05 if p_bp < 0.025 else 0.0)
    print("\n=== 3. [주 검정 1] 색 있는 버튼 vs 회색: 8색 평균 - 회색 (정확도) ===")
    s1 = show("색 vs 회색", st1, a1, note="" if a1 else " (Holm: 다른 주 검정이 먼저 통과 못 해 지지 불가)")
    print("\n=== 4. [주 검정 2] 8색 사이 차이: 위 순열 카이제곱 ===")
    s2 = p_bp < a2
    print(f"  p = {p_bp:.4f} (기준 p<{a2:.4g}) → {'지지: 색조에 따라 정답률이 다르다' if s2 else '지지 안 됨: 8색 사이 차이가 있다는 증거 없음'}")

    # --- 색별 탐색
    print("\n=== 5. [색별 탐색] 각 색 - 회색 (Holm 8개) ===")
    res = []
    for c in HUES:
        st = layout_stats([(b[0], hit(U[b][c]) - hit(U[b]['gray'])) for b in U]); res.append((c, st))
    order = sorted(range(8), key=lambda i: res[i][1]["p"]); thr = {}
    for rank, i in enumerate(order): thr[i] = 0.05 / (8 - rank)
    stop = False
    for rank, i in enumerate(order):                                          # Holm: 처음 실패하면 이후 전부 불가
        if stop or res[i][1]["p"] >= thr[i]: stop = True; thr[i] = 0.0
    for i, (c, st) in enumerate(res):
        sup = verdict(st, thr[i] if thr[i] else 0)
        print(f"  색조 {c:>3}: 평균 {st['obs']*100:+5.1f}%p, p = {st['p']:.4f}, 95% [{st['lo']*100:+5.1f}, {st['hi']*100:+5.1f}]%p → {'지지' if sup else '지지 안 됨'}")

    # --- 실패 분류
    miss = [r for r in rows if hit(r) == 0]
    byl = defaultdict(list)
    for r in rows: byl[int(r["layout"])].append(hit(r))
    layout_fail = {l: 1 - float(np.mean(v)) for l, v in byl.items()}
    fails = classify(miss, man, float(scale), layout_fail)
    cats = sorted({f["category"] for f in fails})
    print(f"\n=== 6. 실패 분류: 오답 {len(fails)}개 (전체 {len(rows)}장 중 {len(fails)/len(rows)*100:.1f}%) ===")
    print(f"  {'분류':<18}" + "".join(f"{c:>6}" for c in CONDS) + f"{'합계':>6}")
    for ct in cats:
        cnt = Counter(f["cond"] for f in fails if f["category"] == ct)
        print(f"  {ct:<18}" + "".join(f"{cnt.get(c, 0):>6}" for c in CONDS) + f"{sum(cnt.values()):>6}")
    cnt = Counter(f["cond"] for f in fails)
    print(f"  {'오답 합계':<17}" + "".join(f"{cnt.get(c, 0):>6}" for c in CONDS) + f"{len(fails):>6}")
    chance = (150 * 54) / (W * H)
    n2 = sum(1 for f in fails if f["category"].startswith("2"))
    print(f"\n  · 좌표 변환 의심 {n2}개. 아무 점이나 찍어도 목표 버튼에 들어갈 확률은 약 {chance*100:.1f}%(버튼 면적/이미지 면적)라, "
          f"오답 {len(fails)}개 중 우연히 {len(fails)*chance*3:.1f}개쯤은 세 가지 해석 중 하나에 걸릴 수 있다. 이보다 훨씬 많을 때만 변환 문제를 의심.")
    flagged = sorted([l for l, v in layout_fail.items() if v >= 0.8])
    print(f"  · 배치 문제 의심(그 배치 오답률 80%+): 레이아웃 {flagged if flagged else '없음'}")
    wrong_lab = Counter(f["picked"] for f in fails if f["category"].startswith("4a"))
    if wrong_lab: print("  · 다른 버튼 선택 시 고른 버튼: " + ", ".join(f"{k} {v}" for k, v in wrong_lab.most_common()))
    if flagged:
        keep = [b for b in U if b[0] not in flagged]
        st = layout_stats([(b[0], col_(b) - hit(U[b]['gray'])) for b in keep])
        print("  · 민감도 점검(배치 문제 의심 레이아웃 제외):", end=" "); show("색 vs 회색", st, 0.05)
    with open("d9_failures.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fails[0].keys())) if fails else None
        if w: w.writeheader(); w.writerows(sorted(fails, key=lambda x: (x["cond"], int(x["layout"]), int(x["mirror"]))))
    print("  · 오답 전체 목록: d9_failures.csv")
    if fails: print("  · 오답 갤러리:", gallery(fails, float(scale)))

    # --- 판정
    print("\n=== 판정 (미리 정한 규칙) ===")
    if s1 and st1["obs"] > 0:
        print("  [1] 색 있는 버튼이 회색보다 더 정확히 찾아진다 (지지).")
    elif s1:
        print("  [1] 색 있는 버튼이 회색보다 덜 정확 (지지, 예측과 반대 — 그대로 보고).")
    else:
        print("  [1] 색 vs 회색 차이는 0과 구분되지 않음 (효과 없음과는 다름).")
    print("  [2] " + ("8색 사이에 차이가 있다 — 어느 색인지는 5번 표 참고(탐색)." if s2 else "8색 사이 차이가 있다는 증거 없음 (색당 120장의 검출력 한계 포함)."))
    print("\n※ 단일 모델(Gemini 3.5 Flash-Lite)·합성 UI·글자 지시문(로그인)·한 배율·1회 실행. 같은 그림의 응답 흔들림은 이 결과에 섞여 있음(--compare로 재실행 시 별도 확인).")


# ----------------------------------------------------------------------------- 재실행 비교
def compare(scale, other):
    a = {(r["file"]): r for r in load() if r["scale"] == scale}
    b = {(r["file"]): r for r in load(other) if r["scale"] == scale}
    both = sorted(set(a) & set(b))
    print(f"=== 재실행 비교: 1회차 vs {other}  (같은 그림 {len(both)}장) ===")
    if not both: return
    flip = [f for f in both if hit(a[f]) != hit(b[f])]
    print(f"  정오답이 바뀐 그림: {len(flip)}/{len(both)} = {len(flip)/len(both)*100:.1f}%  "
          f"(정답→오답 {sum(1 for f in flip if hit(a[f]) == 1)}, 오답→정답 {sum(1 for f in flip if hit(a[f]) == 0)})")
    pa = np.mean([hit(a[f]) for f in both]); pb = np.mean([hit(b[f]) for f in both])
    print(f"  전체 정답률 {pa*100:.1f}% → {pb*100:.1f}%")
    print(f"  {'조건':>6} {'n':>4} {'1회차':>7} {'2회차':>7} {'바뀐 비율':>9}")
    for c in CONDS:
        fs = [f for f in both if a[f]["cond"] == c]
        if fs: print(f"  {c:>6} {len(fs):4} {np.mean([hit(a[f]) for f in fs])*100:6.1f}% {np.mean([hit(b[f]) for f in fs])*100:6.1f}% "
                     f"{sum(1 for f in fs if hit(a[f]) != hit(b[f]))/len(fs)*100:8.1f}%")
    print("  ※ 색별 차이가 '바뀐 비율'보다 작으면 모델 흔들림과 구분하기 어렵다. 기술만(검정 안 함).")


if __name__ == "__main__":
    pos = [x for x in sys.argv[1:] if not x.startswith("--") and not x.endswith(".csv")]
    if not pos:
        precheck(load())
    elif arg("--compare"):
        compare(pos[0], arg("--compare"))
    else:
        main(pos[0])