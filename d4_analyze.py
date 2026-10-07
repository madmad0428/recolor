import csv, json, random
from collections import Counter, defaultdict
from d3_analyze import wilson

SCALE, NPERM = "0.125", 5000
rng = random.Random(2026)

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")) if r["scale"] == SCALE]
for r in rows:
    r["hit"] = int(r["box_in"] == "1")
    r["layout"], r["mirror"] = int(r["layout"]), int(r["mirror"])
    r["picked"] = "(파싱 실패)"
    if r["status"] == "ok":
        px, py = float(r["pred_x"]), float(r["pred_y"])
        ins = [b["label"] for b in json.loads(man[r["file"]]["boxes"])
               if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
        r["picked"] = ins[0] if ins else "(버튼 밖)"

blocks = defaultdict(list)
for r in rows:
    blocks[(r["layout"], r["mirror"])].append(r)
full = {k: v for k, v in blocks.items() if len(v) == 9 and len({x["cond"] for x in v}) == 9}
print(f"블록 {len(blocks)}개 중 조건 9개가 모두 있는 블록 {len(full)}개만 사용\n")

gray_rows = [x for v in full.values() for x in v if x["cond"] == "gray"]
col_rows = [x for v in full.values() for x in v if x["cond"] != "gray"]
G = [x["hit"] for x in gray_rows]; C = [x["hit"] for x in col_rows]
pg, lg, hg = wilson(sum(G), len(G)); pc, lc, hc = wilson(sum(C), len(C))
print(f"gray   정확도 {sum(G)}/{len(G)} = {pg*100:.1f}% [{lg*100:.1f}, {hg*100:.1f}]")
print(f"색 합산 정확도 {sum(C)}/{len(C)} = {pc*100:.1f}% [{lc*100:.1f}, {hc*100:.1f}]")

# --- A. 주 검정: 같은 블록 안에서 (색 8장 평균 - gray) ---
hits_by_block = [[x["hit"] for x in v] for v in full.values()]
def block_diff(v):
    g = next(x["hit"] for x in v if x["cond"] == "gray")
    c = [x["hit"] for x in v if x["cond"] != "gray"]
    return sum(c) / len(c) - g
diffs = {k: block_diff(v) for k, v in full.items()}
obs = sum(diffs.values()) / len(diffs)

def perm_stat():
    t = 0
    for h in hits_by_block:
        i = rng.randrange(9)                     # 블록 안에서 9장 중 아무거나를 'gray'로 가정
        t += (sum(h) - h[i]) / 8 - h[i]
    return t / len(hits_by_block)
ge = sum(abs(perm_stat()) >= abs(obs) - 1e-12 for _ in range(NPERM))
p_main = (1 + ge) / (1 + NPERM)

by_layout = defaultdict(list)
for (l, m), d in diffs.items():
    by_layout[l].append(d)
L = list(by_layout); boots = []
for _ in range(NPERM):                           # 레이아웃 단위로 다시 뽑기(거울 쌍은 같이 이동)
    s = []
    for _ in L:
        s += by_layout[rng.choice(L)]
    boots.append(sum(s) / len(s))
boots.sort()
lo, hi = boots[int(0.025 * NPERM)], boots[int(0.975 * NPERM) - 1]
print(f"\n[A] 같은 블록 안 차이 (색 - gray) = {obs*100:+.1f}%p, 층화 순열검정 p = {p_main:.4f}")
print(f"    레이아웃 단위 부트스트랩 95% 구간 [{lo*100:+.1f}, {hi*100:+.1f}]%p")

# --- B. 색조 간 차이 (탐색용) ---
hues = sorted({x["cond"] for x in col_rows}, key=int)
cb = [[(x["cond"], x["hit"]) for x in v if x["cond"] != "gray"] for v in full.values()]
Mtot = sum(h for b in cb for _, h in b)
def chi(blocks):
    cnt = Counter()
    for b in blocks:
        for c, h in b:
            cnt[c] += h
    e = Mtot / 8
    return sum((cnt[c] - e) ** 2 / e for c in hues)
if 0 < Mtot < 8 * len(cb):
    o = chi(cb); ge = 0
    for _ in range(NPERM):
        pb = []
        for b in cb:
            labs = [c for c, _ in b]; rng.shuffle(labs)
            pb.append([(l, h) for l, (_, h) in zip(labs, b)])
        ge += chi(pb) >= o - 1e-9
    print(f"[B] 색조 간 차이(탐색용): 통계량 {o:.2f}, 블록 안 순열검정 p = {(1 + ge) / (1 + NPERM):.4f}")

# --- C. 어느 버튼을 골랐나 ---
def share(g):
    c = Counter(x["picked"] for x in g); n = len(g)
    return "  ".join(f"{k} {v / n * 100:.0f}%" for k, v in c.most_common())
print("\n[C] 고른 버튼")
print("    gray  :", share(gray_rows))
print("    색 합산:", share(col_rows))

# --- D. 레이아웃별 ---
print("\n[D] 레이아웃별 (gray는 거울 2장 중 맞은 수, 색 합산은 16장 중)")
for l in sorted({k[0] for k in full}):
    vs = [v for k, v in full.items() if k[0] == l]
    g = sum(x["hit"] for v in vs for x in v if x["cond"] == "gray")
    c = sum(x["hit"] for v in vs for x in v if x["cond"] != "gray")
    print(f"    L{l:02d}  gray {g}/{len(vs)}  색 합산 {c}/{8 * len(vs)}")