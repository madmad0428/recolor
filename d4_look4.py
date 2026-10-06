import csv, json, math, statistics as st

man = {r["file"]: r for r in csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))}
rows = [r for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")) if r["scale"] == "0.125"]
W, H = 880, 600

def c(b): return (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2

labels = sorted({b["label"] for m in man.values() for b in json.loads(m["boxes"])})
print("라벨 종류:", labels, "\n")
print("레이아웃 거울 | 로그인 중심(x,y) | 도움말 중심(x,y) | 서로 거리 | 결과")

grp = {"맞음": [], "실패": []}
for r in sorted(rows, key=lambda r: (int(r["layout"]), r["mirror"])):
    bs = json.loads(man[r["file"]]["boxes"])
    t = next(b for b in bs if b["label"] == "로그인")
    h = next((b for b in bs if b["label"] == "도움말"), None)
    tx, ty = c(t)
    res = "맞음" if r["box_in"] == "1" else "실패"
    grp[res].append((tx, ty))
    if h:
        hx, hy = c(h)
        print(f"L{int(r['layout']):02d} {'M' if r['mirror']=='1' else 'N'}  "
              f"({tx:3.0f},{ty:3.0f})  ({hx:3.0f},{hy:3.0f})  {math.hypot(tx-hx, ty-hy):5.0f}px  {res}")
    else:
        print(f"L{int(r['layout']):02d} {'M' if r['mirror']=='1' else 'N'}  ({tx:3.0f},{ty:3.0f})  도움말 없음  {res}")

print()
for k, v in grp.items():
    if v:
        print(f"{k} {len(v)}장: 로그인 평균 위치 x={st.mean(p[0] for p in v):.0f}, y={st.mean(p[1] for p in v):.0f}")