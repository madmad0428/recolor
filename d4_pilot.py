import csv, os, sys, time, math, datetime
from PIL import Image
import d3_run as D
from d3_analyze import wilson

SCALES = [1.0, 0.5, 0.25, 0.125]
OUT, IMG_OUT = "d4_pilot.csv", "d4_img"
PROMPT = ("이 이미지에는 글자가 적힌 버튼이 여러 개 있습니다.\n"
          "그중 '로그인' 이라고 적힌 버튼의 중심 좌표를 구하세요.\n"
          "좌표는 이미지 왼쪽 위가 (0, 0), 오른쪽 아래가 (1000, 1000)인 정규화 좌표이며 x는 가로, y는 세로입니다.\n"
          "설명 없이 JSON만 출력: {\"x\": <정수>, \"y\": <정수>}")
FIELDS = ['file', 'layout', 'mirror', 'cond', 'scale', 'sw', 'sh', 'pred_x', 'pred_y',
          'err', 'box_in', 'echo', 'status', 'raw', 'ts']

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

def ask(client, img):
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
                time.sleep(25 * (attempt + 1)); continue
            return "", m
    return "", "rate limit retries exhausted"

def summary():
    rows = list(csv.DictReader(open(OUT, encoding="utf-8-sig")))
    print("\n조건 / 배율 / 박스 밖 비율 (Wilson 95% CI)")
    for c in sorted({r['cond'] for r in rows}):
        for s in SCALES:
            g = [r for r in rows if r['cond'] == c and r['scale'] == str(s)]
            if not g: continue
            k = sum(r['box_in'] != '1' for r in g)
            p, lo, hi = wilson(k, len(g))
            ec = sum(r['echo'] == '1' for r in g)
            print(f"  {c:>5} x{s:<6} {k:>3}/{len(g):<3} {p*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]  에코 {ec}")

def main():
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["gray"]
    D.assert_no_color_words(PROMPT)
    rows = [r for r in D.load_manifest() if r["cond"] in conds]
    done = set()
    if os.path.exists(OUT):
        done = {(r["file"], r["scale"]) for r in csv.DictReader(open(OUT, encoding="utf-8-sig"))}
    client = D.make_client()
    print(f"대상 {len(rows)}장 x {len(SCALES)}배율, 이미 한 것 {len(done)}")
    for r in rows:
        for s in SCALES:
            if (r["file"], str(s)) in done: continue
            img, (sw, sh) = prep(r, s)
            txt, err = ask(client, img)
            if err:
                print("  [API 오류 — 저장 안 함, 다음 실행에서 재시도]", err[:120]); continue
            ab = D.parse_xy(txt)
            rec = {'file': r['file'], 'layout': r['layout'], 'mirror': r['mirror'], 'cond': r['cond'],
                   'scale': str(s), 'sw': sw, 'sh': sh, 'raw': txt.replace('\n', ' ')[:100],
                   'ts': datetime.datetime.now().isoformat(timespec='seconds')}
            if ab is None:
                rec.update(pred_x='', pred_y='', err='', box_in=0, echo=0, status='no_parse')
            else:
                px, py = ab[0] / 1000 * D.W, ab[1] / 1000 * D.H      # 원본 좌표계로 환산
                cx, cy = int(r['cx']), int(r['cy'])
                hw = (int(r['x1']) - int(r['x0'])) / 2; hh = (int(r['y1']) - int(r['y0'])) / 2
                echo = int(round(ab[0]) in (sw, D.W) and round(ab[1]) in (sh, D.H))
                rec.update(pred_x=round(px, 1), pred_y=round(py, 1), err=round(math.hypot(px - cx, py - cy), 1),
                           box_in=int(abs(px - cx) <= hw and abs(py - cy) <= hh), echo=echo, status='ok')
            new = not os.path.exists(OUT)
            with open(OUT, 'a', newline='', encoding='utf-8-sig') as fp:
                w = csv.DictWriter(fp, fieldnames=FIELDS)
                if new: w.writeheader()
                w.writerow(rec)
            time.sleep(D.SLEEP)
    summary()

if __name__ == "__main__":
    main()