import csv, json, math, statistics as st

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")) if r["scale"] == "0.125"]

m0 = next(iter(man.values()))
b0 = json.loads(m0["boxes"])
print(f"버튼 개수 {len(b0)}, 첫 번째 목표 버튼 크기 {int(m0['x1'])-int(m0['x0'])}x{int(m0['y1'])-int(m0['y0'])}px")

def center(b): return (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2

inside_other = outside = 0
nearest_d = []
for r in rows:
    if r["box_in"] == "1": continue
    m = man[r["file"]]
    px, py = float(r["pred_x"]), float(r["pred_y"])
    tgt = (int(m["x0"]), int(m["y0"]), int(m["x1"]), int(m["y1"]))
    others = [b for b in json.loads(m["boxes"]) if (b["x0"], b["y0"], b["x1"], b["y1"]) != tgt]
    inside = [b for b in others if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
    d = min(math.hypot(px - center(b)[0], py - center(b)[1]) for b in others)
    nearest_d.append(d)
    if inside: inside_other += 1
    else: outside += 1
    print(f"L{int(m['layout']):02d} {'M' if m['mirror']=='1' else 'N'} err={float(r['err']):6.1f}  "
          f"다른 버튼 안={'O' if inside else 'X'}  가장 가까운 방해 버튼까지 {d:5.1f}px")

print(f"\n방해 버튼 안에 찍음 {inside_other}, 버튼 밖(빈 공간) {outside}")
print(f"가장 가까운 방해 버튼까지 거리 중앙값 {st.median(nearest_d):.1f}px")
