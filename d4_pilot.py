import csv, os, sys, time, math, datetime, re
from PIL import Image
import d3_run as D
from d3_analyze import wilson

SCALES = [0.125]
ALL_CONDS = "gray,20,65,110,155,200,245,290,335"
SLEEP = 4                      # 호출 사이 대기(초). 분당 한도에 걸리면 늘리세요
OUT, IMG_OUT = "d4_pilot.csv", "d4_img"
PROMPT = ("이 이미지에는 글자가 적힌 버튼이 여러 개 있습니다.\n"
          "그중 '로그인' 이라고 적힌 버튼의 중심 좌표를 구하세요.\n"
          "좌표는 이미지 왼쪽 위가 (0, 0), 오른쪽 아래가 (1000, 1000)인 정규화 좌표이며 x는 가로, y는 세로입니다.\n"
          "설명 없이 JSON만 출력: {\"x\": <정수>, \"y\": <정수>}")
FIELDS = ['file', 'layout', 'mirror', 'cond', 'scale', 'sw', 'sh', 'pred_x', 'pred_y',
          'err', 'box_in', 'echo', 'status', 'raw', 'ts']
_shown = False

class DailyQuota(Exception):
    pass

def prep(row, s):
    src = os.path.join(D.IMG_DIR, row["file"])
    if s == 1.0:
        return open(src, "rb").read(), (D.W, D.H)
    os.makedirs(IMG_OUT, exist_ok=True)
    sw, sh = round(D.W * s), round(D.H * s)
    dst = os.path.join(IMG_OUT, f"{os.path.splitext(row['file'])[0]}_s{s}.png")
    if not os.path.exists(dst):
        Image.open(src).convert("RGB").resize((sw, sh), Image.LANCZOS).save(dst)
    return open(dst, "rb").read(), (sw, sh)

def retry_wait(msg, default):
    m = (re.search(r"retry in ([\d.]+)\s*s", msg, re.I)
         or re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", msg, re.I))
    if m:
        try:
            return min(float(m.group(1)) + 2, 120)
        except ValueError:
            pass
    return default

def ask(client, img):
    global _shown
    from google.genai import types
    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=D.MODEL,
                contents=[types.Part.from_bytes(data=img, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(
                    temperature=D.TEMPERATURE, max_output_tokens=D.MAX_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_level=D.THINKING)))
            return r.text or "", None
        except Exception as e:
            m = str(e)
            if "429" in m or "RESOURCE_EXHAUSTED" in m:
                if not _shown:
                    print("  429 전문:", m[:600], flush=True)
                    _shown = True
                if "PerDay" in m or "per day" in m.lower():
                    raise DailyQuota(m)
                wait = retry_wait(m, 25 * (attempt + 1))
                print(f"  (속도제한 {wait:.0f}s 대기)", flush=True)
                time.sleep(wait)
                continue
            return "", m
    return "", "rate limit retries exhausted"

def summary():
    if not os.path.exists(OUT):
        return
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    print("\n조건 / 배율 / 박스 밖 비율 (Wilson 95% CI)")
    for s in sorted({r['scale'] for r in rows}, key=float, reverse=True):
        for c in sorted({r['cond'] for r in rows}, key=lambda x: (x != 'gray', x.zfill(4))):
            g = [r for r in rows if r['cond'] == c and r['scale'] == s]
            if not g:
                continue
            k = sum(r['box_in'] != '1' for r in g)
            p, lo, hi = wilson(k, len(g))
            ec = sum(r['echo'] == '1' for r in g)
            print(f"  {c:>5} x{s:<6} {k:>3}/{len(g):<3} {p*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  에코 {ec}")

def main():
    args = sys.argv[1:]
    mx = None
    if "--max" in args:
        i = args.index("--max")
        mx = int(args[i + 1])
        del args[i:i + 2]
    conds = (args[0] if args else ALL_CONDS).split(",")
    D.assert_no_color_words(PROMPT)
    rows = [r for r in D.load_manifest() if r["cond"] in conds]
    done = set()
    if os.path.exists(OUT):
        done = {(r["file"], r["scale"]) for r in csv.DictReader(open(OUT, encoding="utf-8-sig"))}
    todo = [(r, s) for r in rows for s in SCALES if (r["file"], str(s)) not in done]
    if mx:
        todo = todo[:mx]
    print(f"대상 {len(rows)}장 x {len(SCALES)}배율 / 이번에 호출할 것 {len(todo)}")
    client = D.make_client() if todo else None
    consec = 0
    for n, (r, s) in enumerate(todo, 1):
        img, (sw, sh) = prep(r, s)
        try:
            txt, err = ask(client, img)
        except DailyQuota:
            print("\n하루 한도에 걸렸습니다. 태평양 시간 자정(서머타임 기간 한국 시간 오후 4시경) 이후 "
                  "같은 명령으로 이어서 하세요.")
            break
        if err:
            consec += 1
            print("  [API 오류 — 저장 안 함, 다음 실행에서 재시도]", err[:120], flush=True)
            if consec >= 2:
                print("연속 2회 실패 — 중단합니다. 한도가 풀린 뒤 같은 명령으로 이어서 실행하세요.")
                break
            continue
        consec = 0
        ab = D.parse_xy(txt)
        rec = {'file': r['file'], 'layout': r['layout'], 'mirror': r['mirror'], 'cond': r['cond'],
               'scale': str(s), 'sw': sw, 'sh': sh, 'raw': txt.replace('\n', ' ')[:100],
               'ts': datetime.datetime.now().isoformat(timespec='seconds')}
        if ab is None:
            rec.update(pred_x='', pred_y='', err='', box_in=0, echo=0, status='no_parse')
        else:
            px, py = ab[0] / 1000 * D.W, ab[1] / 1000 * D.H
            cx, cy = int(r['cx']), int(r['cy'])
            hw = (int(r['x1']) - int(r['x0'])) / 2
            hh = (int(r['y1']) - int(r['y0'])) / 2
            echo = int(round(ab[0]) in (sw, D.W) and round(ab[1]) in (sh, D.H))
            rec.update(pred_x=round(px, 1), pred_y=round(py, 1), err=round(math.hypot(px - cx, py - cy), 1),
                       box_in=int(abs(px - cx) <= hw and abs(py - cy) <= hh), echo=echo, status='ok')
        new = not os.path.exists(OUT)
        with open(OUT, 'a', newline='', encoding='utf-8-sig') as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{n}/{len(todo)}] L{int(r['layout']):02d} {r['cond']:>4} "
              f"{'M' if r['mirror'] == '1' else 'N'} x{s:<5} box_in={rec['box_in']} {rec['status']}", flush=True)
        time.sleep(SLEEP)
    summary()

if __name__ == "__main__":
    main()