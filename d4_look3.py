import csv, json
from collections import Counter

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")) if r["scale"] == "0.125"]

# 이 레이아웃에 쓰인 라벨 전체 (첫 레이아웃 기준)
m0 = next(iter(man.values()))
print("라벨 목록(레이아웃", m0["layout"], "):", [b["label"] for b in json.loads(m0["boxes"])])

picked = Counter()
for r in rows:
    if r["box_in"] == "1":
        continue
    m = man[r["file"]]
    px, py = float(r["pred_x"]), float(r["pred_y"])
    hit = [b["label"] for b in json.loads(m["boxes"])
           if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
    lab = hit[0] if hit else "(버튼 밖)"
    picked[lab] += 1
    print(f"L{int(m['layout']):02d} {'M' if m['mirror']=='1' else 'N'}  → {lab}")

print("\n골라진 라벨 횟수:", dict(picked))