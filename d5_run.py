#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d5 실행기: d5_manifest.csv의 이미지를 (줄여서) Gemini에 보내고 채점해 d5_results.csv에 쌓는다.

필요: 같은 폴더에 d3_run.py, d4_pilot.py (둘 다 새 버전), d3_analyze.py, make_d5.py가 만든 d5_img/, d5_manifest.csv

사용 예
  python d5_run.py --summary                         # API 호출 없이 지금까지 요약만
  python d5_run.py --set d6 --targets 삭제,저장 --conds gray --scales 0.13,0.125
                                                     # 2부(색 관습): d6_manifest.csv / d6_img / d6_results.csv 사용
  python d5_run.py --set d7 --targets 삭제,저장 --scales 0.13
                                                     # 3a(모양 끌림): d7_manifest.csv / d7_img / d7_results.csv 사용
  python d5_run.py --set d8 --targets 삭제,저장 --conds TT_Tgray,TT_Igray,IT_Tgray,IT_Igray --scales 1.0 --layouts 0-19
  python d5_run.py --set d8 --targets 삭제,저장 --conds TT --scales 0.13      # --conds TT = TT_로 시작하는 6조건 전부
                                                     # 3부 재설계: d8_*, 기능 지시문(첫 문장 중립)으로 고정
  python d5_run.py --set d9 --conds gray --layouts 0-29 --scales 0.14,0.135,0.13,0.125
                                                     # d9 사전 점검: 회색 60장 x 배율 4개 (난이도 고르기)
  python d5_run.py --set d9 --scales 0.13            # d9 본 실행 (배율은 사전 점검 결과로 정함, 1,080장)
  python d5_run.py --set d9 --scales 0.13 --run 2    # 같은 실험 한 번 더: d9_results_run2.csv에 따로 저장
  python d5_run.py --conds gray --scales 0.25,0.2,0.15,0.125 --max 60
                                                     # 사전 점검: gray만, 여러 배율 (난이도 구간 찾기)
  python d5_run.py --scales 0.15 --max 100           # 본 실행 (배율은 사전 점검 결과로 정함)
  python d5_run.py --targets 로그인 --conds D65,D245 --scales 1.0,0.25
                                                     # 목표 라벨 필터: 로그인 목표의 오도 조건만, 배율 2개

규칙
  - 이미 한 (파일, 배율, 목표)는 건너뛴다. 끊겨도 같은 명령으로 이어서 한다.
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

SET = "d5"                                   # --set d6 이면 2부(색 관습) 파일을 쓴다
MANIFEST, OUT, IMG_DIR = "d5_manifest.csv", "d5_results.csv", "d5_img"
# 기능 설명 지시문 (--instr func). 라벨 대신 '무슨 일을 하는 버튼인지'로 목표를 말한다. 글자 일치로 찾을 수 없게 한다.
FUNC = {"로그인": "계정에 접속하는", "홈": "처음 화면으로 돌아가는", "저장": "작업 내용을 보관하는",
        "삭제": "항목을 지우는", "검색": "원하는 내용을 찾는", "설정": "환경을 조정하는",
        "취소": "하던 작업을 그만두는", "도움말": "사용 방법을 안내받는"}
INSTR = "literal"

FIELDS = ["file", "layout", "mirror", "cond", "kind", "target", "decoy", "scale", "picked", "hit",
          "err", "pred_x", "pred_y", "status", "raw", "ts"]
BASE_PROMPT = P.PROMPT


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["run_order"]))
    # gray 그림은 목표와 무관하게 같은 그림이다(d3 gray와 픽셀 동일). 매니페스트에는 로그인 목표로만 들어 있으므로
    # 홈·설정 목표의 gray 행을 같은 파일로 만들어 붙인다. 정답 좌표만 그 라벨의 버튼으로 바꾼다.
    extra = []
    if SET != "d5":                          # d6·d7 매니페스트에는 목표별 gray 행이 이미 있음
        return rows
    for r in [r for r in rows if r["kind"] == "gray"]:
        boxes = json.loads(r["boxes"])
        for t in ("홈", "설정"):
            b = next(b for b in boxes if b["label"] == t)
            e = dict(r)
            e.update(target=t, cx=(b["x0"] + b["x1"]) // 2, cy=(b["y0"] + b["y1"]) // 2,
                     x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"])
            extra.append(e)
    return rows + extra


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
    if SET == "d5":
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
                if c.startswith("D") or SET in ("d7", "d8"):
                    dk = sum(1 for r in h if r["picked"] == r["decoy"])
                    line += f"   오도 버튼을 고른 비율 {dk}/{len(h)} = {dk/len(h)*100:.0f}%"
                print(line)
        bad = sum(1 for r in g if r["status"] != "ok")
        if bad:
            print(f"  좌표를 못 낸 응답 {bad}개")


def make_prompt(target):
    if INSTR == "literal":
        p = BASE_PROMPT.replace("로그인", target)
        last = target[-1]
        if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 == 0:   # 받침 없는 말(삭제, 취소): '이라고' -> '라고'
            p = p.replace(f"'{target}' 이라고", f"'{target}' 라고")
        return p
    if INSTR == "func8":
        # d8: 첫 문장의 '글자가 적힌 버튼'을 '버튼'으로 바꿔 아이콘 버튼도 같은 자격의 후보가 되게 하고, 목표는 기능으로 말한다.
        a, b = "이 이미지에는 글자가 적힌 버튼이 여러 개 있습니다.", "이 이미지에는 버튼이 여러 개 있습니다."
        old = "'로그인' 이라고 적힌 버튼"
        if a not in BASE_PROMPT or old not in BASE_PROMPT:
            raise SystemExit("d4_pilot.PROMPT의 문장이 예상과 달라 d8 지시문을 만들 수 없습니다. 지시문을 붙여 보내주세요.")
        return BASE_PROMPT.replace(a, b).replace(old, FUNC[target] + " 버튼")
    # 기능 설명: "'로그인' 이라고 적힌 버튼" 구절 전체를 "<기능>하는 버튼"으로 통째로 바꾼다.
    old = "'로그인' 이라고 적힌 버튼"
    if old not in BASE_PROMPT:
        raise SystemExit("d4_pilot.PROMPT에 \"'로그인' 이라고 적힌 버튼\" 구절이 없습니다. 지시문 형태가 달라 기능 설명 치환을 못 합니다.")
    return BASE_PROMPT.replace(old, FUNC[target] + " 버튼")


def main():
    global OUT, INSTR, SET, MANIFEST, IMG_DIR
    SET = arg("--set", "d5")
    if SET not in ("d5", "d6", "d7", "d8", "d9"):
        raise SystemExit("--set 은 d5, d6, d7, d8, d9 중 하나")
    MANIFEST, OUT, IMG_DIR = f"{SET}_manifest.csv", f"{SET}_results.csv", f"{SET}_img"
    run = arg("--run")                       # 재실행: --run 2 -> {SET}_results_run2.csv 에 따로 저장 (1회차 파일은 건드리지 않음)
    if run:
        if not run.isdigit() or int(run) < 2:
            raise SystemExit("--run 은 2 이상의 정수 (1회차는 옵션 없이 실행)")
        OUT = f"{SET}_results_run{run}.csv"
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST}가 없습니다. 이미지 묶음을 이 폴더에 풀었는지 확인하세요.")
    INSTR = arg("--instr", "literal")
    if SET == "d8":                          # 3부 재설계: 기능 지시문 + 중립 첫 문장으로 고정 (글자/아이콘 목표에 공정하게)
        if arg("--instr") not in (None, "func8"):
            raise SystemExit("--set d8 은 기능 지시문(func8)으로만 돈다. --instr 를 빼고 실행하세요.")
        INSTR = "func8"
    elif INSTR not in ("literal", "func"):
        raise SystemExit("--instr 는 literal 또는 func")
    if INSTR == "func":
        OUT = f"{SET}_results_func.csv"      # 글자 지시문 결과와 섞이지 않게 따로 저장
        for t in FUNC:
            p = make_prompt(t)
            leaked = [l for l in FUNC if l in p]
            assert not leaked, f"기능 설명 지시문에 라벨 글자가 남음: {leaked} / {p}"
    if INSTR == "func8":
        for t in FUNC:
            p = make_prompt(t)
            leaked = [l for l in FUNC if l in p]
            assert not leaked, f"기능 설명 지시문에 라벨 글자가 남음: {leaked} / {p}"
            assert "글자" not in p, f"d8 지시문에 '글자'가 남음: {p}"
    if "--summary" not in sys.argv:
        for t in (arg("--targets") or "로그인").split(","):     # 실제로 모델에 갈 지시문을 먼저 보여줌
            print(f"[{'글자' if INSTR == 'literal' else '기능 설명'} 지시문 — 목표 {t}]")
            print(make_prompt(t))
        print(f"(결과는 {OUT}에 저장)\n")
    D.assert_no_color_words(BASE_PROMPT)
    if "--summary" in sys.argv:
        summary()
        return
    conds = arg("--conds", "all")
    scales = [float(x) for x in arg("--scales", "0.125").split(",")]
    mx = int(arg("--max")) if arg("--max") else None
    targets = arg("--targets")
    tset = set(targets.split(",")) if targets else None
    lay = arg("--layouts")                   # 예: 0-19  (d8 사전 점검은 0~19번 레이아웃만)
    lset = set(range(int(lay.split("-")[0]), int(lay.split("-")[1]) + 1)) if lay else None
    cl = conds.split(",")
    rows = [r for r in load_manifest()
            if (conds == "all" or r["cond"] in cl or any(r["cond"].startswith(c + "_") for c in cl))
            and (tset is None or r["target"] in tset)
            and (lset is None or int(r["layout"]) in lset)]
    done = set()
    if os.path.exists(OUT):
        done = {(r["file"], r["scale"], r["target"]) for r in csv.DictReader(open(OUT, encoding="utf-8-sig"))}
    todo = [(r, s) for r in rows for s in scales if (r["file"], str(s), r["target"]) not in done]
    import random
    random.Random(2026).shuffle(todo)          # 중간에 끊겨도 목표·조건·레이아웃이 한쪽으로 몰리지 않게 섞음
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
        P.PROMPT = make_prompt(r["target"])
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