#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d9 실행기 (다른 모델용, 기본 gemini-3.1-pro-preview).
d9 Flash-Lite 실행(d5_run.py --set d9)과 지시문·이미지 축소·채점이 같다. 바뀌는 것은 모델과 호출 설정뿐이다.

순서: 난이도(배율)를 먼저 맞추고, 설정 비교는 그 난이도에서 한다.
  (Pro가 쉬운 배율에서 90%대 천장이면 설정 사이 차이가 안 보여 설정 파일럿이 헛돈다.)

0단계: 스모크 테스트 (작동 확인, 기본 40회). 배율과 무관한 확인이라 난이도보다 먼저 한다.
       키·결제, 이미지별 해상도 지정이 거부(400)되는지, 해상도마다 입력 토큰이 실제로 달라지는지를 본다.
       결과는 d9s_*.csv (파일럿·본 실행과 섞이지 않음)
  python d9_pro_run.py --pilot --smoke           # --max N 으로 횟수 조절
  python d9_pro_run.py --pilot-summary --smoke
1단계: 난이도(배율) 사전 점검 — 기본 설정(high/high/온도 1.0), 회색 60장 x 여러 배율
  python d9_pro_run.py --conds gray --layouts 0-29 --scales 0.13,0.10,0.08,0.06     # 넓게 먼저
  python d9_analyze.py --file d9_results_g31pro_high_high_t1.0.csv                   # 회색 50~80% 중 65%에 가장 가까운 배율 권고
  (50~80% 구간이 사다리 틈에 끼면 사이 배율을 더해 같은 명령을 다시 실행. 이미 한 것은 건너뜀)
2단계: 설정 파일럿 — 1단계에서 고른 배율에서 8칸(해상도 4 x 생각 2, 온도 1.0) x 회색 60장 = 480회. 결과는 d9p_*.csv
  python d9_pro_run.py --pilot --pilot-scale 0.08
  python d9_pro_run.py --pilot-summary --pilot-scale 0.08    # 칸별 정확도·토큰·지연·비용 표
  python d9_pro_run.py --pilot --pilot-scale 0.08 --pilot-res high --pilot-think low,high --pilot-temp 1.0
                                                # 격자를 줄이거나 바꿀 때
  고른 설정이 기본값(high/high/1.0)이 아니면 그 설정으로 1단계를 다시 해서 배율이 여전히 맞는지 확인한다.
3단계: 본 실행 (1,080장)
  python d9_pro_run.py --res high --think high --temp 1.0 --scales 0.13
  python d9_analyze.py 0.13 --file d9_results_g31pro_high_high_t1.0.csv
  python d9_pro_run.py ... --run 2              # 같은 설정 반복 -> ..._run2.csv (d9_analyze --compare 용)

기타
  --summary      API 호출 없이 해당 설정 결과 요약
  --dry          API 호출 없이 지시문·요청 설정·이미지 크기만 출력
  --workers N    동시 호출 수 (기본 4). 429가 잦으면 줄이세요
  --model 이름   다른 모델로 같은 실험 (예: gemini-3.8-flash)

규칙 (d5_run.py와 동일)
  - 이미 한 (파일, 배율, 목표)는 건너뛴다. 끊겨도 같은 명령으로 이어서 한다.
  - API 오류(예외)는 저장하지 않는다 -> 다음 실행에서 재시도. 좌표를 못 낸 응답은 no_parse로 저장하고 오답으로 센다.
  - 결과 파일마다 설정을 *_config.json에 기록하고, 다음 실행에서 설정이 바뀌면 중단한다 (--force로 무시).
  - 응답 전문·토큰 사용량·종료 사유·실제 모델 버전은 *_log.jsonl에 따로 남긴다.
"""
import csv, datetime, hashlib, io, json, math, os, platform, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import d3_run as D            # W, H, parse_xy, assert_no_color_words (d9 Flash-Lite 실행과 같은 함수)
import d4_pilot as P          # PROMPT (d9 Flash-Lite 실행과 같은 지시문)

MANIFEST, IMG_DIR = "d9_manifest.csv", "d9_img"
W, H = D.W, D.H
MODEL = "gemini-3.1-pro-preview"
MAX_TOKENS = 32768            # 생각(thinking) 토큰이 출력 한도를 먹을 수 있어 넉넉히. 잘리면 finish_reason에 MAX_TOKENS로 남는다
PRICE_IN, PRICE_OUT = 2.0, 12.0   # 3.1 Pro, 프롬프트 20만 토큰 미만, 100만 토큰당 달러 (추정용)
RES_LEVELS = ["unspecified", "low", "medium", "high", "ultra_high"]
THINK_LEVELS = ["low", "medium", "high"]   # 3.1 Pro는 minimal 미지원
PILOT_SCALE = 0.13
PILOT_LAYOUTS = range(0, 30)
# 기본 파일럿 격자: 해상도 4 x 생각 2, 온도는 1.0 고정 (구글이 Gemini 3 전 모델에 1.0 유지를 강하게 권장; 낮추면 루핑·성능 저하 가능).
# 온도 0.0 칸은 뺐다 — 1.0으로 갈 것이라 고를 수 없는 칸이다. 보고 싶으면 --pilot-temp 0.0,1.0 (해상도·생각 격자 전체가 두 배가 됨)
PILOT_GRID = [(r, t, 1.0) for r in ["low", "medium", "high", "ultra_high"] for t in ["low", "high"]]

FIELDS = ["file", "layout", "mirror", "cond", "kind", "target", "decoy", "scale", "picked", "hit",
          "err", "pred_x", "pred_y", "status", "raw", "ts",
          # 여기부터 추가 열 (d9_analyze.py는 위 열만 읽으므로 그대로 분석 가능)
          "model", "res", "think", "temp", "finish", "in_tok", "thought_tok", "out_tok", "latency_s", "model_version"]


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a and a.index(name) + 1 < len(a) else default


def sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


def short(model):
    m = re.match(r"gemini-(\d+)\.(\d+)-([a-z]+)(?:-([a-z]+))?", model)
    return f"g{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4) or ''}".replace("preview", "") if m else re.sub(r"\W", "", model)


def tag(model, res, think, temp):
    return f"{short(model)}_{res}_{think}_t{temp}"


def out_path(prefix, model, res, think, temp, run=None):
    return f"{prefix}_{tag(model, res, think, temp)}" + (f"_run{run}" if run else "") + ".csv"


# ----------------------------------------------------------------------------- 입력 (d5_run.py와 같은 방식)
def load_manifest():
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["run_order"]))
    return rows


_img_cache = {}
def image_bytes(row, s):
    """d5_run.image_bytes와 같은 처리: RGB 변환 -> LANCZOS 축소 -> PNG."""
    key = (row["file"], s)
    if key not in _img_cache:
        im = Image.open(os.path.join(IMG_DIR, row["file"])).convert("RGB")
        if s != 1.0:
            im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, "PNG")
        _img_cache[key] = buf.getvalue()
    return _img_cache[key]


def make_prompt(target):
    """d5_run.make_prompt (literal)과 같다. d9의 목표는 전부 '로그인'이라 지시문은 d4_pilot.PROMPT 그대로."""
    p = P.PROMPT.replace("로그인", target)
    last = target[-1]
    if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28 == 0:
        p = p.replace(f"'{target}' 이라고", f"'{target}' 라고")
    return p


# ----------------------------------------------------------------------------- 설정 기록
def config_dict(model, res, think, temp, api_version):
    return {"model": model, "media_resolution": res, "thinking_level": think, "temperature": temp,
            "max_output_tokens": MAX_TOKENS, "prompt": P.PROMPT, "image_size": [W, H],
            "manifest_sha1": sha1(MANIFEST), "api_version": api_version,
            "coordinate_rule": "0-1000 normalized, x first (지시문에 명시; d4_pilot.PROMPT)"}


def check_config(path, cur, force):
    cpath = path.replace(".csv", "_config.json")
    if not os.path.exists(cpath):
        info = dict(cur, first_run=datetime.datetime.now().isoformat(timespec="seconds"), python=platform.python_version())
        try:
            from importlib.metadata import version
            info["google_genai"] = version("google-genai")
        except Exception:
            pass
        json.dump(info, open(cpath, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return
    old = json.load(open(cpath, encoding="utf-8"))
    diffs = [k for k in cur if old.get(k) != cur[k]]
    if diffs and not force:
        raise SystemExit(f"[중단] {path}의 이전 실행과 설정이 다릅니다: {diffs}  (알고 바꾼 거라면 --force)")


# ----------------------------------------------------------------------------- 모델 호출
class DailyQuota(Exception):
    pass


def make_client(api_version):
    key = os.environ.get("GEMINI_API_KEY") or (D.API_KEY if D.API_KEY and D.API_KEY != "여기에_키" else None)
    if not key:
        raise SystemExit("[중단] API 키가 없습니다. 환경변수 GEMINI_API_KEY 또는 d3_run.py의 API_KEY에 넣으세요. (키는 채팅에 붙여넣지 마세요)")
    from google import genai
    from google.genai import types
    ho = types.HttpOptions(api_version=api_version) if api_version else None
    return genai.Client(api_key=key, http_options=ho)


def build_request(img, prompt, res, think, temp):
    from google.genai import types
    part = types.Part.from_bytes(data=img, mime_type="image/png")
    if res != "unspecified":     # 이미지 한 장(part) 단위로 지정. ultra_high는 part 단위에서만 가능
        part.media_resolution = types.PartMediaResolution(level=f"MEDIA_RESOLUTION_{res.upper()}")
    cfg = types.GenerateContentConfig(
        temperature=temp, max_output_tokens=MAX_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_level=think.upper()))
    return [part, prompt], cfg


_shown = False
def ask(client, model, img, prompt, res, think, temp):
    """반환 (결과 dict, 오류 문자열 또는 None). 결과 dict: text, finish, in_tok, thought_tok, out_tok, latency_s, model_version"""
    global _shown
    contents, cfg = build_request(img, prompt, res, think, temp)
    for attempt in range(5):
        t0 = time.time()
        try:
            r = client.models.generate_content(model=model, contents=contents, config=cfg)
        except Exception as e:
            m = str(e)
            if "429" in m or "RESOURCE_EXHAUSTED" in m:
                if not _shown:
                    print("  429 전문:", m[:600], flush=True); _shown = True
                if "PerDay" in m or "per day" in m.lower():
                    raise DailyQuota(m)
                wait = P.retry_wait(m, 20 * (attempt + 1)) + random.uniform(0, 3)
                print(f"  (속도제한 {wait:.0f}s 대기)", flush=True); time.sleep(wait); continue
            if any(s in m for s in ("500", "503", "INTERNAL", "UNAVAILABLE", "DEADLINE")):
                time.sleep(10 * (attempt + 1)); continue
            if "media_resolution" in m.lower() or "mediaresolution" in m.lower():
                m += "  → 이미지별 해상도 지정이 이 API 버전에서 거부됨. --api-version v1alpha 로 다시 시도해 보세요."
            return None, m
        u = r.usage_metadata
        cand = (r.candidates or [None])[0]
        try:
            text = r.text or ""
        except Exception:
            text = ""
        return {"text": text,
                "finish": str(getattr(cand, "finish_reason", "") or "").replace("FinishReason.", ""),
                "in_tok": getattr(u, "prompt_token_count", None) if u else None,
                "thought_tok": (getattr(u, "thoughts_token_count", None) or 0) if u else None,
                "out_tok": (getattr(u, "candidates_token_count", None) or 0) if u else None,
                "latency_s": round(time.time() - t0, 2),
                "model_version": getattr(r, "model_version", "") or ""}, None
    return None, "retries exhausted (rate limit / server error)"


def score(r, s, res_d, model, res, think, temp):
    """d5_run.py와 같은 채점: 0~1000 정규화(x 먼저)를 880x600으로 바꾼 점이 들어간 버튼(첫 번째)을 고른 것으로 본다."""
    boxes = json.loads(r["boxes"])
    txt = res_d["text"]
    ab = D.parse_xy(txt)
    rec = {"file": r["file"], "layout": r["layout"], "mirror": r["mirror"], "cond": r["cond"], "kind": r["kind"],
           "target": r["target"], "decoy": r["decoy"], "scale": str(s),
           "raw": txt.replace("\n", " ")[:300], "ts": datetime.datetime.now().isoformat(timespec="seconds"),
           "model": model, "res": res, "think": think, "temp": temp,
           **{k: res_d[k] for k in ("finish", "in_tok", "thought_tok", "out_tok", "latency_s", "model_version")}}
    if ab is None:
        rec.update(picked="(파싱 실패)", hit=0, err="", pred_x="", pred_y="", status="no_parse")
    else:
        px, py = ab[0] / 1000 * W, ab[1] / 1000 * H
        ins = [b["label"] for b in boxes if b["x0"] <= px <= b["x1"] and b["y0"] <= py <= b["y1"]]
        picked = ins[0] if ins else "(버튼 밖)"
        cx, cy = int(r["cx"]), int(r["cy"])
        rec.update(picked=picked, hit=int(picked == r["target"]), err=round(math.hypot(px - cx, py - cy), 1),
                   pred_x=round(px, 1), pred_y=round(py, 1), status="ok")
    return rec


# ----------------------------------------------------------------------------- 실행 엔진
_lock = threading.Lock()

def append(path, rec, log):
    with _lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=FIELDS)
            if new: w.writeheader()
            w.writerow(rec)
        with open(path.replace(".csv", "_log.jsonl"), "a", encoding="utf-8") as fp:
            fp.write(json.dumps(log, ensure_ascii=False) + "\n")


def done_keys(path):
    if not os.path.exists(path): return set()
    return {(r["file"], r["scale"], r["target"]) for r in csv.DictReader(open(path, encoding="utf-8-sig"))}


def run_jobs(jobs, client, model, workers):
    """jobs = [(path, row, scale, res, think, temp)]. 섞인 순서로 동시에 호출하고, 끝나는 대로 저장."""
    stop = threading.Event(); n_err = [0]; consec = [0]; total = len(jobs); cnt = [0]

    def one(job):
        path, r, s, res, think, temp = job
        if stop.is_set(): return
        prompt = make_prompt(r["target"])
        try:
            out, err = ask(client, model, image_bytes(r, s), prompt, res, think, temp)
        except DailyQuota:
            if not stop.is_set():
                print("\n하루 한도에 걸렸습니다. 한도가 풀린 뒤 같은 명령으로 이어서 하세요.")
            stop.set(); return
        with _lock:
            cnt[0] += 1; k = cnt[0]
            if err:
                n_err[0] += 1; consec[0] += 1
                print(f"  [API 오류 — 저장 안 함, 다음 실행에서 재시도] {err[:200]}", flush=True)
                if consec[0] >= max(3, workers + 1):
                    print("연속 오류 — 중단합니다. 같은 명령으로 이어서 실행하세요."); stop.set()
                return
            consec[0] = 0
        rec = score(r, s, out, model, res, think, temp)
        append(path, rec, {"file": r["file"], "scale": s, "res": res, "think": think, "temp": temp,
                           "prompt": prompt, **out, "ts": rec["ts"]})
        line = (f"[{k}/{total}] {res:>10}/{think:<6} t{temp} L{int(r['layout']):02d} {'M' if r['mirror'] == '1' else 'N'} "
                f"{r['cond']:>5} x{s:<5} → {rec['picked']}  ({out['latency_s']}s, 생각 {out['thought_tok']}토큰, {out['finish']})")
        with _lock:                            # 스레드끼리 출력이 한 줄에 붙지 않게 한 번에 쓴다
            sys.stdout.write(line + "\n"); sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for f in as_completed(futs):
            f.result()
    if n_err[0]:
        print(f"\nAPI 오류 {n_err[0]}건은 저장하지 않았습니다 → 같은 명령을 다시 실행하면 재시도.")


# ----------------------------------------------------------------------------- 요약
def wilson(k, n, z=1.96):
    if n == 0: return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


def conv_suspect(r, man):
    """오답인데 좌표를 '줄인 그림 픽셀'·'원본 픽셀'·'x/y 뒤바뀜'으로 읽으면 정답인 경우 (d9_analyze 분류 2와 같은 기준)."""
    if r["status"] != "ok" or r["hit"] == "1": return False
    ab = D.parse_xy(r["raw"])
    if not ab: return False
    a, b = ab; s = float(r["scale"]); tb = json.loads(man[r["file"]]["boxes"])[0]
    inside = lambda x, y: tb["x0"] <= x <= tb["x1"] and tb["y0"] <= y <= tb["y1"]
    return (not (0 <= a <= 1000 and 0 <= b <= 1000)) or inside(b / 1000 * W, a / 1000 * H) or inside(a / s, b / s) or inside(a, b)


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


def table_line(rows, man):
    n = len(rows); k = sum(int(r["hit"]) for r in rows); p, lo, hi = wilson(k, n)
    npar = sum(r["status"] != "ok" for r in rows)
    trunc = sum("MAX_TOKENS" in (r.get("finish") or "") for r in rows)
    conv = sum(conv_suspect(r, man) for r in rows)
    it = [fnum(r.get("in_tok")) for r in rows]; it = [x for x in it if x is not None]
    th = [fnum(r.get("thought_tok")) for r in rows]; th = [x for x in th if x is not None]
    ot = [fnum(r.get("out_tok")) for r in rows]; ot = [x for x in ot if x is not None]
    la = sorted(x for x in (fnum(r.get("latency_s")) for r in rows) if x is not None)
    cost = (sum(it) * PRICE_IN + (sum(th) + sum(ot)) * PRICE_OUT) / 1e6
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    return (f"{k:>3}/{n:<3} {p*100:5.1f}% [{lo*100:4.1f},{hi*100:5.1f}]  파싱실패 {npar:>2} 잘림 {trunc:>2} 변환의심 {conv:>2}  "
            f"입력 {mean(it):6.0f} 생각 {mean(th):6.0f}  지연 중앙 {la[len(la)//2] if la else float('nan'):5.1f}s  ${cost:5.2f}")


def pilot_summary(prefix="d9p", only_scale=None):
    """prefix d9p = 설정 파일럿, d9s = 스모크 테스트. 배율별로 따로 표를 찍는다(배율이 다른 칸을 섞지 않음)."""
    man = {r["file"]: r for r in load_manifest()}
    files = sorted(f for f in os.listdir(".") if f.startswith(prefix + "_") and f.endswith(".csv"))
    name = "스모크 테스트" if prefix == "d9s" else "설정 파일럿"
    if not files:
        print(f"{name} 결과가 없습니다. python d9_pro_run.py --pilot" + (" --smoke" if prefix == "d9s" else " --pilot-scale 배율")); return
    data = {f: list(csv.DictReader(open(f, encoding="utf-8-sig"))) for f in files}
    scales = sorted({r["scale"] for rows in data.values() for r in rows}, key=float, reverse=True)
    if only_scale is not None:
        scales = [s for s in scales if float(s) == only_scale]
    fl_all = ([r for r in csv.DictReader(open("d9_results.csv", encoding="utf-8-sig"))
               if r["cond"] == "gray" and int(r["layout"]) in PILOT_LAYOUTS] if os.path.exists("d9_results.csv") else [])
    for s in scales:
        print(f"\n=== {name}: 회색 기준선, 배율 x{s}, 칸마다 같은 회색 그림 ===")
        print(f"  {'설정(해상도/생각/온도)':<26} {'정답':>7} {'정확도':>6} {'윌슨 95%':>13}   입력·생각 = 호출당 평균 토큰")
        for f in files:
            g = [r for r in data[f] if r["scale"] == s]
            if g:
                r0 = g[0]
                print(f"  {r0['res'] + '/' + r0['think'] + '/t' + r0['temp']:<26} " + table_line(g, man))
        fl = [r for r in fl_all if r["scale"] == s]
        if fl:
            k = sum(int(r["hit"]) for r in fl); p, lo, hi = wilson(k, len(fl))
            print(f"  {'(참고) Flash-Lite 3.5':<26} {k:>3}/{len(fl):<3} {p*100:5.1f}% [{lo*100:4.1f},{hi*100:5.1f}]")
    print("\n  읽는 법: 정확도가 90%대(천장)거나 20% 아래(바닥)면 이 배율은 설정 비교에 못 씀 → 1단계로 돌아가 배율을 다시 맞출 것.")
    print("          변환의심이 많으면 정답 판정 전에 좌표 규칙부터 확인. 잘림이 있으면 MAX_TOKENS를 늘릴 것.")
    print("          '입력' 토큰이 해상도마다 달라지는지 = 작은 그림에도 media_resolution이 실제로 먹히는지.")


def summary(path):
    if not os.path.exists(path):
        print(f"{path}가 아직 없습니다."); return
    man = {r["file"]: r for r in load_manifest()}
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    print(f"\n=== {path} ===")
    for s in sorted({r["scale"] for r in rows}, key=float, reverse=True):
        g = [r for r in rows if r["scale"] == s]
        print(f"  배율 x{s} (n={len(g)})")
        for c in sorted({r["cond"] for r in g}, key=lambda x: (x != "gray", x.zfill(4))):
            print(f"    {c:>5}: " + table_line([r for r in g if r["cond"] == c], man))


# ----------------------------------------------------------------------------- 진입점
def main():
    global MODEL
    MODEL = arg("--model", MODEL)
    api_version = arg("--api-version")
    workers = int(arg("--workers", "4"))
    force = "--force" in sys.argv
    D.assert_no_color_words(P.PROMPT)
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST}가 없습니다. 레포 폴더에서 실행하세요.")

    smoke = "--smoke" in sys.argv
    ps = arg("--pilot-scale")
    if "--pilot-summary" in sys.argv:
        pilot_summary("d9s" if smoke else "d9p", float(ps) if ps else None); return

    rows_all = load_manifest()
    if "--pilot" in sys.argv:
        if ps is None and not smoke:
            raise SystemExit("[중단] --pilot-scale 이 필요합니다. 난이도(배율)를 먼저 1단계로 맞추고, 그 배율을 넣으세요.\n"
                             "       작동 확인만 하려면 --smoke (배율 무관, 40회).")
        pscale = float(ps) if ps else PILOT_SCALE        # 스모크는 배율이 결과 해석에 안 쓰임
        prefix = "d9s" if smoke else "d9p"
        rl = arg("--pilot-res"); tl = arg("--pilot-think"); te = arg("--pilot-temp")
        grid = PILOT_GRID
        if rl or tl or te:
            grid = [(r, t, float(x)) for r in (rl or "low,medium,high,ultra_high").split(",")
                    for t in (tl or "low,high").split(",") for x in (te or "1.0").split(",")]
        for r, t, _ in grid:
            assert r in RES_LEVELS and t in THINK_LEVELS, f"잘못된 설정: {r}/{t}"
        rows = [r for r in rows_all if r["cond"] == "gray" and int(r["layout"]) in PILOT_LAYOUTS]
        jobs = []
        for res, think, temp in grid:
            path = out_path(prefix, MODEL, res, think, temp)
            check_config(path, config_dict(MODEL, res, think, temp, api_version), force)
            done = done_keys(path)
            jobs += [(path, r, pscale, res, think, temp) for r in rows
                     if (r["file"], str(pscale), r["target"]) not in done]
        random.Random(2026).shuffle(jobs)     # 끊겨도 칸마다 고르게 진행되도록 칸·그림을 섞음
        mx = arg("--max", "40" if smoke else None)
        if mx: jobs = jobs[:int(mx)]
        print(f"[{'스모크 테스트' if smoke else '설정 파일럿'}] {MODEL}  격자 {len(grid)}칸 x 회색 {len(rows)}장 (배율 x{pscale}) / 이번에 호출할 것 {len(jobs)}")
        print("[지시문]\n" + make_prompt("로그인") + "\n")
        if "--dry" in sys.argv:
            print("(--dry: 호출 안 함)"); return
        if jobs:
            run_jobs(jobs, make_client(api_version), MODEL, workers)
        pilot_summary(prefix, pscale); return

    res = arg("--res", "high"); think = arg("--think", "high"); temp = float(arg("--temp", "1.0"))
    if res not in RES_LEVELS: raise SystemExit(f"--res 는 {RES_LEVELS} 중 하나")
    if think not in THINK_LEVELS: raise SystemExit(f"--think 는 {THINK_LEVELS} 중 하나 (3.1 Pro는 minimal 미지원)")
    run = arg("--run")
    if run and (not run.isdigit() or int(run) < 2): raise SystemExit("--run 은 2 이상의 정수")
    path = out_path("d9_results", MODEL, res, think, temp, run)
    if "--summary" in sys.argv:
        summary(path); return

    conds = arg("--conds", "all"); cl = conds.split(",")
    scales = [float(x) for x in arg("--scales", "0.13").split(",")]
    lay = arg("--layouts")
    lset = set(range(int(lay.split("-")[0]), int(lay.split("-")[1]) + 1)) if lay else None
    rows = [r for r in rows_all if (conds == "all" or r["cond"] in cl) and (lset is None or int(r["layout"]) in lset)]
    check_config(path, config_dict(MODEL, res, think, temp, api_version), force)
    done = done_keys(path)
    jobs = [(path, r, s, res, think, temp) for r in rows for s in scales if (r["file"], str(s), r["target"]) not in done]
    random.Random(2026).shuffle(jobs)
    if arg("--max"): jobs = jobs[:int(arg("--max"))]
    print(f"[{MODEL}] 해상도 {res} / 생각 {think} / 온도 {temp}  → {path}")
    print(f"대상 {len(rows)}장 x {len(scales)}배율 / 이번에 호출할 것 {len(jobs)}")
    print("[지시문]\n" + make_prompt("로그인") + "\n")
    if "--dry" in sys.argv:
        for s in scales:
            im = Image.open(io.BytesIO(image_bytes(rows[0], s)))
            print(f"  x{s}: 모델에 가는 그림 {im.size[0]}x{im.size[1]} px")
        print("(--dry: 호출 안 함)"); return
    if not rows:
        print("조건에 맞는 이미지가 없습니다."); return
    if jobs:
        run_jobs(jobs, make_client(api_version), MODEL, workers)
    summary(path)


if __name__ == "__main__":
    main()