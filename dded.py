import csv, json
m = next(iter(csv.DictReader(open("d3_manifest.csv", encoding="utf-8-sig"))))
print(json.loads(m["boxes"])[0])