import csv, statistics as st
rows = [r for r in csv.DictReader(open("d4_pilot.csv", encoding="utf-8-sig")) if r["scale"] == "0.125"]
miss = sorted(float(r["err"]) for r in rows if r["box_in"] != "1" and r["err"] != "")
print("miss", len(miss), "/ no_parse", sum(r["status"] == "no_parse" for r in rows))
print("오차(px, 원본 880x600 기준):", [round(e) for e in miss])
print("중앙값:", round(st.median(miss), 1))