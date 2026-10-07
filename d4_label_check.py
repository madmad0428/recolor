import csv, json, math
from collections import Counter
from itertools import combinations
from math import comb

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
hit = {"로그인": {}, "홈": {}, "설정": {}}
for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")):
    if r["scale"] == "0.125" and r["cond"] == "gray":
        hit["로그인"][r["file"]] = int(r["box_in"] == "1")
for r in csv.DictReader(open("d4_label.csv", encoding="utf-8-sig")):
    hit[r["target"]][r["file"]] = int(r["hit"])
files = sorted(set(hit["로그인"]) & set(hit["홈"]) & set(hit["설정"]))
print(len(files), "장 공통\n")

def mcnemar(a, b):
    x = sum(1 for f in files if a[f] and not b[f])
    y = sum(1 for f in files if b[f] and not a[f])
    n = x + y
    if n == 0:
        return x, y, 1.0
    p = sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, 2 * p)

print("[1] 같은 이미지에서 라벨끼리 비교 (정확 McNemar, Bonferroni 기준 0.0167)")
for a, b in combinations(hit, 2):
    x, y, p = mcnemar(hit[a], hit[b])
    print(f"    {a}만 맞음 {x} : {b}만 맞음 {y}   p = {p:.4f}")

print("\n[2] 이미지별 (로그인, 홈, 설정) 맞음 패턴 (1=맞음)")
pat = Counter((hit["로그인"][f], hit["홈"][f], hit["설정"][f]) for f in files)
for k, v in sorted(pat.items(), reverse=True):
    print(f"    {k}: {v}장")

def phi(a, b):
    n11 = sum(1 for f in files if a[f] and b[f]); n10 = sum(1 for f in files if a[f] and not b[f])
    n01 = sum(1 for f in files if not a[f] and b[f]); n00 = sum(1 for f in files if not a[f] and not b[f])
    d = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return (n11 * n00 - n10 * n01) / d if d else float("nan")
print("\n[3] 라벨 간 상관(phi): 이미지가 어려우면 라벨이 달라도 같이 틀리는가")
for a, b in combinations(hit, 2):
    print(f"    {a}-{b}: {phi(hit[a], hit[b]):+.2f}")

print("\n[4] 라벨별 버튼 위치와 주변 간격 (40장 평균)")
for lab in hit:
    xs, ys, nn = [], [], []
    for f in files:
        bs = json.loads(man[f]["boxes"])
        t = next(b for b in bs if b["label"] == lab)
        cx, cy = (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2
        d = min(math.hypot(cx - (b["x0"] + b["x1"]) / 2, cy - (b["y0"] + b["y1"]) / 2)
                for b in bs if b is not t)
        xs.append(cx); ys.append(cy); nn.append(d)
    n = len(files)
    print(f"    {lab}: 평균 중심 ({sum(xs)/n:.0f}, {sum(ys)/n:.0f}), "
          f"가장 가까운 다른 버튼까지 {sum(nn)/n:.0f}px, 정확도 {sum(hit[lab][f] for f in files)}/{n}")