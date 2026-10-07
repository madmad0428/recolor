import csv, json
from collections import Counter

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
rows = list(csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")))
for s in sorted({r["scale"] for r in rows}, key=float, reverse=True):
    c = Counter()
    for r in rows:
        if r["scale"] != s: continue
        if r["status"] != "ok":
            c["(파싱 실패)"] += 1; continue
        m = man[r["file"]]; px, py = float(r["pred_x"]), float(r["pred_y"])
        hit = [b["label"] for b in json.loads(m["boxes"])
               if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
        c[hit[0] if hit else "(버튼 밖)"] += 1
    print(f"x{s:<6}", dict(c.most_common()))