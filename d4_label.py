import csv, json, os, time
from collections import Counter
import d3_run as D
import d4_pilot as P
from d3_analyze import wilson

LABELS = ["홈", "설정"]        # 로그인은 이미 d4_pilot.csv에 있음 (17/40)
SCALE = 0.125
OUT = "d4_label.csv"
FIELDS = ["file", "layout", "mirror", "target", "picked", "hit", "pred_x", "pred_y", "raw"]
BASE = P.PROMPT

def summary():
    if not os.path.exists(OUT):
        return
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    print("\n목표 라벨 / 정확도 (Wilson 95% CI) / 고른 라벨   (참고: 로그인은 17/40 = 42.5%)")
    for lab in LABELS:
        g = [r for r in rows if r["target"] == lab]
        if not g:
            continue
        k = sum(int(r["hit"]) for r in g)
        p, lo, hi = wilson(k, len(g))
        c = Counter(r["picked"] for r in g)
        print(f"  {lab}: {k}/{len(g)} = {p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]  {dict(c.most_common())}")

def main():
    rows = [r for r in D.load_manifest() if r["cond"] == "gray"]
    done = set()
    if os.path.exists(OUT):
        done = {(r["file"], r["target"]) for r in csv.DictReader(open(OUT, encoding="utf-8-sig"))}
    todo = [(r, lab) for lab in LABELS for r in rows if (r["file"], lab) not in done]
    print(f"이번에 호출할 것 {len(todo)}")
    client = D.make_client() if todo else None
    consec = 0
    for n, (r, lab) in enumerate(todo, 1):
        img, _ = P.prep(r, SCALE)
        P.PROMPT = BASE.replace("로그인", lab)
        try:
            txt, err = P.ask(client, img)
        except P.DailyQuota:
            print("하루 한도에 걸렸습니다. 한도가 풀린 뒤 같은 명령으로 이어서 하세요.")
            break
        if err:
            consec += 1
            print("  [API 오류 — 저장 안 함]", err[:120], flush=True)
            if consec >= 2:
                print("연속 2회 실패 — 중단합니다. 같은 명령으로 이어서 실행하세요.")
                break
            continue
        consec = 0
        ab = D.parse_xy(txt)
        boxes = json.loads(r["boxes"])
        picked, hit, px, py = "(파싱 실패)", 0, "", ""
        if ab:
            px, py = round(ab[0] / 1000 * D.W, 1), round(ab[1] / 1000 * D.H, 1)
            inside = [b["label"] for b in boxes if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
            picked = inside[0] if inside else "(버튼 밖)"
            hit = int(picked == lab)
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({"file": r["file"], "layout": r["layout"], "mirror": r["mirror"], "target": lab,
                        "picked": picked, "hit": hit, "pred_x": px, "pred_y": py,
                        "raw": txt.replace("\n", " ")[:100]})
        print(f"[{n}/{len(todo)}] L{int(r['layout']):02d} {'M' if r['mirror'] == '1' else 'N'} "
              f"{lab} → {picked}", flush=True)
        time.sleep(P.SLEEP)
    summary()

if __name__ == "__main__":
    main()