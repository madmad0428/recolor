#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d3 실험 실행 — 지시문을 주고 모델이 답한 좌표를 기록한다.

사용:
  python d3_run.py --selftest null     # API 없이 파이프라인 전체 점검(가짜 모델). 결과는 d3_results_selftest.csv
  python d3_run.py --det               # 같은 이미지를 3번씩 물어 매번 같은 답인지 확인 (9장 x 3회)
  python d3_run.py                     # 본 실행. 끊겨도 같은 명령을 다시 치면 이어서 진행
  python d3_analyze.py                 # 분석

설계 규칙
  - 지시문에는 색 단어가 없어야 한다(시작할 때 자동 검사. 있으면 중단).
  - API 오류는 결과로 저장하지 않는다 -> 다음 실행에서 자동 재시도.
    (모델이 좌표를 못 내놓은 경우는 'no_parse'로 저장하고 실패로 센다.)
  - 최초 실행 때 모델·온도·지시문·매니페스트 해시를 d3_run_config.json에 기록하고,
    이후 실행에서 바뀌면 중단한다(--force로 무시).
"""
import csv, json, os, re, math, sys, time, hashlib, platform, datetime, random

MANIFEST = "d3_manifest.csv"
IMG_DIR = "."
RESULTS = "d3_results.csv"
SELFTEST = "d3_results_selftest.csv"
DET_FILE = "d3_det.csv"
CONFIG = "d3_run_config.json"

MODEL = "gemini-3.5-flash-lite"
TEMPERATURE = 0.0
MAX_TOKENS = 300
THINKING = "minimal"
API_KEY = ""
SLEEP = 4                 # 호출 사이 대기(초)
MAX_PER_RUN = None        # 무료 한도가 작으면 18 등으로 제한. None = 전부

W, H = 880, 600

# 지시문: 글자 라벨로만 지정. 색에 대한 언급 없음.
PROMPT = ("이 이미지에는 글자가 적힌 버튼이 여러 개 있습니다.\n"
          "그중 '로그인' 이라고 적힌 버튼의 '중심'의 픽셀 좌표를 구하세요. 이미지 크기는 가로 880, 세로 600입니다.\n"
          "설명 없이 JSON만 출력: {\"x\": <정수>, \"y\": <정수>}")

COLOR_WORDS = ['색', '컬러', '빨강', '빨간', '노랑', '노란', '초록', '녹색', '파랑', '파란', '청록', '보라', '자주',
               '분홍', '주황', '회색', '검정', '검은', '흰색', '흑백', '하늘색',
               'red', 'green', 'blue', 'yellow', 'purple', 'pink', 'orange', 'gray', 'grey', 'color', 'colour', 'cyan', 'magenta']

def assert_no_color_words(prompt):
    low = prompt.lower()
    hits = [w for w in COLOR_WORDS if w in low]
    if hits:
        raise SystemExit(f"[중단] 지시문에 색 관련 단어가 들어 있습니다: {hits}")

def sha1(path):
    return hashlib.sha1(open(path, 'rb').read()).hexdigest()

# ---------------- 매니페스트 / 설정 기록 ----------------
def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r['run_order']))
    return rows

def current_config():
    return {"model": MODEL, "temperature": TEMPERATURE, "max_output_tokens": MAX_TOKENS,
            "thinking_level": THINKING, "prompt": PROMPT, "image_size": [W, H],
            "manifest_sha1": sha1(MANIFEST),
            "coordinate_rule": "0-1000 normalized, x first (calib-confirmed for gemini-3.5-flash-lite)"}

def check_config(force):
    cur = current_config()
    if not os.path.exists(CONFIG):
        info = dict(cur)
        info["first_run"] = datetime.datetime.now().isoformat(timespec='seconds')
        info["python"] = platform.python_version()
        try:
            from importlib.metadata import version
            info["google_genai"] = version("google-genai")
        except Exception:
            pass
        json.dump(info, open(CONFIG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"설정을 {CONFIG}에 기록했습니다.")
        return
    old = json.load(open(CONFIG, encoding="utf-8"))
    diffs = [k for k in cur if old.get(k) != cur[k]]
    if diffs and not force:
        print(f"[중단] 이전 실행과 설정이 다릅니다: {diffs}")
        print("  같은 실험을 이어서 하려면 값을 되돌리세요. 새 실험이면 결과 파일과 d3_run_config.json을 치우세요.")
        print("  (알고 바꾼 거라면 --force)")
        sys.exit(1)

# ---------------- 모델 호출 ----------------
def make_client():
    key = API_KEY if API_KEY and API_KEY != "여기에_키" else os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("[중단] API 키가 없습니다. d3_run.py 위쪽 API_KEY에 넣으세요. (키는 채팅에 붙여넣지 마세요)")
    from google import genai
    return genai.Client(api_key=key)

def ask_real(client, row, img_bytes):
    """반환 (응답텍스트, 오류문자열 또는 None)"""
    from google.genai import types
    for attempt in range(3):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/png"), PROMPT],
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE, max_output_tokens=MAX_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING)))
            return (resp.text or ""), None
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                wait = 25 * (attempt + 1); print(f"  (속도제한 {wait}s 대기)"); time.sleep(wait); continue
            return "", msg
    return "", "rate limit retries exhausted"

FAKE_RNG = random.Random(123)
def ask_fake(scenario, row):
    """오프라인 점검용 가짜 모델. 시나리오: null / effect / baseline / ceiling"""
    cond = row['cond']
    p = {'null': 0.15, 'ceiling': 0.0,
         'effect': 0.60 if cond == '245' else 0.10,
         'baseline': 0.40 if cond == 'gray' else 0.10}[scenario]
    if scenario != 'ceiling' and FAKE_RNG.random() < 0.01:
        return "죄송합니다. 위치를 알 수 없습니다.", None
    if FAKE_RNG.random() < p:
        o = json.loads(row['boxes'])[FAKE_RNG.randrange(1, 8)]
        cx, cy = (o['x0'] + o['x1']) / 2, (o['y0'] + o['y1']) / 2
    else:
        cx, cy = int(row['cx']) + FAKE_RNG.uniform(-1.5, 1.5), int(row['cy']) + FAKE_RNG.uniform(-1, 1)
    return '```json\n{"x": %d, "y": %d}\n```' % (round(cx / W * 1000), round(cy / H * 1000)), None

# ---------------- 파싱 / 채점 ----------------
def parse_xy(text):
    """JSON {x,y} 또는 'x: 숫자 ... y: 숫자' 꼴만 인정. 그 밖의 숫자(이미지 크기 언급 등)는 좌표로 보지 않는다."""
    m = re.search(r'\{[^{}]*\}', text, re.S)
    if m:
        try:
            o = json.loads(m.group(0)); return float(o['x']), float(o['y'])
        except Exception:
            pass
    m = re.search(r'["\']?x["\']?\s*[:=]\s*(-?\d+\.?\d*).{0,30}?["\']?y["\']?\s*[:=]\s*(-?\d+\.?\d*)', text, re.S | re.I)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def evaluate(r, txt):
    """채점: 예측점이 목표 버튼 박스 안이면 성공(box_in). 오차가 화면 짧은 변의 절반을 넘으면 완전실패(fail)."""
    tx, ty = int(r['cx']), int(r['cy'])
    hw = (int(r['x1']) - int(r['x0'])) / 2; hh = (int(r['y1']) - int(r['y0'])) / 2
    rec = {'file': r['file'], 'layout': r['layout'], 'mirror': r['mirror'], 'cond': r['cond'],
           'run_order': r['run_order'], 'true_x': tx, 'true_y': ty,
           'raw': txt.replace('\n', ' ')[:100], 'ts': datetime.datetime.now().isoformat(timespec='seconds')}
    ab = parse_xy(txt)
    if ab is None:
        rec.update(pred_x='', pred_y='', err='', dx='', dy='', box_in=0, fail=1, status='no_parse')
    else:
        px, py = ab[0] / 1000 * W, ab[1] / 1000 * H      # Gemini: 0~1000 정규화, x먼저
        dx, dy = px - tx, py - ty; e = math.hypot(dx, dy)
        rec.update(pred_x=round(px, 1), pred_y=round(py, 1), err=round(e, 1), dx=round(dx, 1), dy=round(dy, 1),
                   box_in=int(abs(dx) <= hw and abs(dy) <= hh), fail=int(e > min(W, H) / 2), status='ok')
    return rec

FIELDS = ['file', 'layout', 'mirror', 'cond', 'run_order', 'true_x', 'true_y', 'pred_x', 'pred_y',
          'err', 'dx', 'dy', 'box_in', 'fail', 'status', 'raw', 'ts']

def load_done(path):
    if not os.path.exists(path): return set()
    return {r['file'] for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

def append_row(path, rec):
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        if new: w.writeheader()
        w.writerow(rec)

# ---------------- 실행 ----------------
def run(scenario, force):
    assert_no_color_words(PROMPT)
    rows = load_manifest()
    path = RESULTS if scenario is None else SELFTEST
    if scenario is not None and os.path.exists(path): os.remove(path)
    client = None
    if scenario is None:
        check_config(force); client = make_client()
    done = load_done(path)
    todo = [r for r in rows if r['file'] not in done]
    if MAX_PER_RUN and scenario is None: todo = todo[:MAX_PER_RUN]
    print(f"전체 {len(rows)} / 완료 {len(done)} / 이번에 돌릴 것 {len(todo)}   (결과: {path})")
    errors = consec = 0
    for i, r in enumerate(todo):
        if scenario is None:
            with open(os.path.join(IMG_DIR, r['file']), 'rb') as f: img = f.read()
            txt, err = ask_real(client, r, img)
        else:
            txt, err = ask_fake(scenario, r)
        if err:
            errors += 1; consec += 1
            print(f"  [API 오류 — 저장 안 함, 다음 실행에서 재시도] {err[:140]}")
            if consec >= 5: print("연속 5회 오류 — 중단합니다."); break
            continue
        consec = 0
        rec = evaluate(r, txt); append_row(path, rec)
        print(f"[{i + 1}/{len(todo)}] L{int(r['layout']):02d} {r['cond']:>4} {'M' if r['mirror'] == '1' else 'N'}  "
              f"err={rec['err']} box_in={rec['box_in']} {rec['status']}", flush=True)
        if scenario is None and i < len(todo) - 1: time.sleep(SLEEP)
    done_now = len(load_done(path))
    print(f"\n완료 {done_now}/{len(rows)}" + (f"  (API 오류 {errors}건은 미저장 → 다시 실행하면 재시도)" if errors else ""))
    if not os.path.exists(path):
        print("저장된 결과가 없어 분석은 건너뜁니다. 위의 [API 오류] 메시지를 확인하세요 (키, 모델명, 한도).")
        return
    try:
        import d3_analyze
        d3_analyze.main(path)
    except ImportError:
        print("d3_analyze.py 가 같은 폴더에 없어 분석은 건너뜁니다.")

def determinism(force):
    """같은 이미지를 3번씩 물어 매번 같은 답이 나오는지. 같으면 반복은 표본이 안 되므로 새 레이아웃으로 늘려야 한다."""
    assert_no_color_words(PROMPT)
    rows = [r for r in load_manifest() if r['layout'] == '0' and r['mirror'] == '0']
    check_config(force); client = make_client()
    out = []; same = 0
    for r in rows:
        img = open(os.path.join(IMG_DIR, r['file']), 'rb').read(); preds = []
        for k in range(3):
            txt, err = ask_real(client, r, img)
            ab = parse_xy(txt) if not err else None
            preds.append(ab); out.append({'file': r['file'], 'cond': r['cond'], 'rep': k, 'pred': ab, 'raw': (txt or err or '')[:80]})
            time.sleep(SLEEP)
        ident = len({p for p in preds}) == 1 and preds[0] is not None
        same += ident
        print(f"  {r['cond']:>5}: {preds}  {'동일' if ident else '** 다름 **'}")
    with open(DET_FILE, 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.DictWriter(fp, fieldnames=['file', 'cond', 'rep', 'pred', 'raw']); w.writeheader(); w.writerows(out)
    print(f"\n9장 중 3회 모두 동일: {same}장")
    if same == 9:
        print("→ 이 설정(temperature 고정)에서는 같은 이미지 반복이 새 표본이 되지 않음. 표본을 늘리려면 새 레이아웃을 추가하세요.")
    else:
        print("→ 같은 이미지에서도 답이 달라짐. 이 흔들림이 '잡음 바닥'이므로 분석 때 함께 봐야 함.")

if __name__ == "__main__":
    a = sys.argv[1:]; force = '--force' in a
    if '--selftest' in a:
        i = a.index('--selftest'); run(a[i + 1] if i + 1 < len(a) else 'null', True)
    elif '--det' in a:
        determinism(force)
    else:
        run(None, force)