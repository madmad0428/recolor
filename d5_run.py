#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d5 실행기: d5_manifest.csv의 이미지를 (줄여서) Gemini에 보내고 채점해 d5_results.csv에 쌓는다.

필요: 같은 폴더에 d3_run.py, d4_pilot.py (둘 다 새 버전), d3_analyze.py, make_d5.py가 만든 d5_img/, d5_manifest.csv

사용 예
  python d5_run.py --summary                         # API 호출 없이 지금까지 요약만
  python d5_run.py --conds gray --scales 0.25,0.2,0.15,0.125 --max 60
                                                     # 사전 점검: gray만, 여러 배율 (난이도 구간 찾기)
  python d5_run.py --scales 0.15 --max 100           # 본 실행 (배율은 사전 점검 결과로 정함)
  python d5_run.py --targets 로그인 --conds D65,D245 --scales 1.0,0.25
                                                     # 목표 라벨 필터: 로그인 목표의 오도 조건만, 배율 2개

규칙
  - 이미 한 (파일, 배율)은 건너뛴다. 끊겨도 같은 명령으로 이어서 한다.
  - API 오류는 저장하지 않는다. 연속 2회 실패하면 멈춘다.
  - 실행 순서는 매니페스트의 run_order(전체를 섞은 순서)를 따른다.
  - 지시문은 d4와 같고 목표 라벨만 바뀐다. 색 단어가 없는지 시작할 때 검사한다.
"""
import csv, io, json, math, os, sys, time, datetime
from collections import Counter, defaultdict
from PIL import Image
import d3_run as D
import d4_pilot as P
from d3_analyze import wilson

MANIFEST, OUT, IMG_DIR = "d5_manifest.csv", "d5_results.csv", "d5_img"
FIELDS = ["file", "layout", "mirror", "cond", "kind", "target", "decoy", "scale", "picked", "hit",
          "err", "pred_x", "pred_y", "status", "raw", "ts"]
BASE_PROMPT = P.PROMPT


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["run_order"]))
    return rows


def image_bytes(row, s):
    im = Image.open(os.path.join(IMG_DIR, row["file"])).convert("RGB")
    if s != 1.0:
        im = im.resize((round(D.W * s), round(D.H * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def summary():
    if not os.path.exists(OUT):
        print("결과 파일이 아직 없습니다.")
        return
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    print("(gray 기준선은 d3와 같은 그림이라 d4 결과를 씀: x0.125에서 로그인 17/40, 홈 29/40, 설정 29/40)")
    for s in sorted({r["scale"] for r in rows}, key=float, reverse=True):
        g = [r for r in rows if r["scale"] == s]
        print(f"\n=== 배율 x{s} (n={len(g)}) ===")
        for t in sorted({r["target"] for r in g}):
            print(f"  목표 {t}")
            for c in sorted({r["cond"] for r in g if r["target"] == t}):
                h = [r for r in g if r["target"] == t and r["cond"] == c]
                k = sum(int(r["hit"]) for r in h)
                p, lo, hi = wilson(k, len(h))
                line = f"    {c:>5}: {k:>3}/{len(h):<3} 정확도 {p*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
                if c.startswith("D"):
                    dk = sum(1 for r in h if r["picked"] == r["decoy"])
                    line += f"   오도 버튼을 고른 비율 {dk}/{len(h)} = {dk/len(h)*100:.0f}%"
                print(line)
        bad = sum(1 for r in g if r["status"] != "ok")
        if bad:
            print(f"  좌표를 못 낸 응답 {bad}개")


def main():
    D.assert_no_color_words(BASE_PROMPT)
    if "--summary" in sys.argv:
        summary()
        return
    conds = arg("--conds", "all")
    scales = [float(x) for x in arg("--scales", "0.125").split(",")]
    mx = int(arg("--max")) if arg("--max") else None
    targets = arg("--targets")
    tset = set(targets.split(",")) if targets else None
    rows = [r for r in load_manifest()
            if (conds == "all" or r["cond"] in conds.split(","))
            and (tset is None or r["target"] in tset)]
    done = set()
    if os.path.exists(OUT):
        done = {(r["file"], r["scale"]) for r in csv.DictReader(open(OUT, encoding="utf-8-sig"))}
    todo = [(r, s) for r in rows for s in scales if (r["file"], str(s)) not in done]
    if mx:
        todo = todo[:mx]
    print(f"대상 {len(rows)}장 x {len(scales)}배율 / 이번에 호출할 것 {len(todo)}"
          + (f"   (목표 라벨 필터: {','.join(sorted(tset))})" if tset else ""))
    if not rows:
        print("조건에 맞는 이미지가 없습니다. --targets / --conds 철자를 확인하세요 (예: 로그인, 홈, 설정 / D65, T245).")
        return
    client = D.make_client() if todo else None
    consec = 0
    for n, (r, s) in enumerate(todo, 1):
        P.PROMPT = BASE_PROMPT.replace("로그인", r["target"])
        try:
            txt, err = P.ask(client, image_bytes(r, s))
        except P.DailyQuota:
            print("\n하루 한도에 걸렸습니다. 태평양 시간 자정(서머타임 기간 한국 시간 오후 4시경) 이후 같은 명령으로 이어서 하세요.")
            break
        if err:
            consec += 1
            print("  [API 오류 — 저장 안 함, 다음 실행에서 재시도]", err[:120], flush=True)
            if consec >= 2:
                print("연속 2회 실패 — 중단합니다. 같은 명령으로 이어서 실행하세요.")
                break
            continue
        consec = 0
        boxes = json.loads(r["boxes"])
        ab = D.parse_xy(txt)
        rec = {"file": r["file"], "layout": r["layout"], "mirror": r["mirror"], "cond": r["cond"], "kind": r["kind"],
               "target": r["target"], "decoy": r["decoy"], "scale": str(s),
               "raw": txt.replace("\n", " ")[:100], "ts": datetime.datetime.now().isoformat(timespec="seconds")}
        if ab is None:
            rec.update(picked="(파싱 실패)", hit=0, err="", pred_x="", pred_y="", status="no_parse")
        else:
            px, py = ab[0] / 1000 * D.W, ab[1] / 1000 * D.H           # 0~1000 정규화, x 먼저 (d4와 같은 규칙)
            ins = [b["label"] for b in boxes if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
            picked = ins[0] if ins else "(버튼 밖)"
            cx, cy = int(r["cx"]), int(r["cy"])
            rec.update(picked=picked, hit=int(picked == r["target"]), err=round(math.hypot(px - cx, py - cy), 1),
                       pred_x=round(px, 1), pred_y=round(py, 1), status="ok")
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{n}/{len(todo)}] L{int(r['layout']):02d} {'M' if r['mirror'] == '1' else 'N'} "
              f"{r['cond']:>5} x{s:<5} 목표 {r['target']} → {rec['picked']}", flush=True)
        time.sleep(P.SLEEP)
    summary()


if __name__ == "__main__":
    main()