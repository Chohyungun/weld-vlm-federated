"""카나리아-1 — 사전학습 모델의 네이티브 좌표 규약 실측 판별 (게이트 G5 · 함정 #4).

## 왜 이것을 재는가

`vlm/coords.py` 의 `NORM_1000` 채택 근거가 **문서적**이었다(Qwen3VLProcessor 공유 +
쿡북의 0~1000 명시). 61번이 스스로 "본실험 착수 게이트"로 등급을 매겨 놓고 미실시로
남긴 항목이고, 67번 §5-1 이 재확인했다. 함정 #4 는 규약이 어긋나면 **IoU 가
0.938 → 0.055 로 붕괴**하는 구간이라 문서 근거로 넘길 자리가 아니다.

66번이 확인한 "매칭쌍 IoU 0.31~0.34" 는 **우리 파이프라인 내부 정합**이다 — 우리가
NORM_1000 으로 만든 타깃을 NORM_1000 으로 되읽으면 당연히 맞는다. 사전학습 모델이
어느 규약을 쓰는지는 그 실험으로 알 수 없다.

## 판별 A — 크기 스케일링 (주 판정)

합성 도형을 **같은 상대 위치로 여러 이미지 크기에 렌더링**하고, 모델이 낸 숫자를 정답
상대좌표로 나눈다. 그 몫이 모델이 상정한 **캔버스 크기**다.

    상정 캔버스 = 생성 좌표 / 정답 상대좌표

- 이미지 크기를 따라가면 → `ABS_ORIG`
- 크기와 무관하게 1000 근처면 → `NORM_1000`
- 프로세서 리사이즈 치수를 따라가면 → `ABS_RESIZED`

**이 판별은 토큰 사전확률에 오염되지 않는다.** 아래 B 가 오염될 수 있어서 A 를 주 판정으로 둔다.

## 판별 B — 우도 argmax (보조) + 그 자체의 대조군

같은 프롬프트에 세 규약의 정답 문자열을 붙여 교사 강제 로그우도를 잰다. 61번이
"argmax 판별"이라고 부른 것이 이쪽이다. 그런데 **이 검사에는 함정이 있다.**

1차 실행에서 정답 상대 박스가 (0.25, 0.30, 0.55, 0.70) 이었다. NORM_1000 표현이
(250, 300, 550, 700) — **전부 반올림 수**다. 언어 모델은 이미지와 무관하게 둥근 수를
선호하므로, 그 선호가 규약 판별로 오독될 수 있었다. 두 가지로 막는다.

1. 정답 박스를 **1000 스케일에서 둥글지 않은 값**으로 바꿨다(237/313/561/688).
2. **대조군을 함께 잰다** — 일부러 틀린 박스를 세 규약으로 만들어 같이 채점한다.
   규약 선호가 진짜라면 정답에서만 나타나야 하고, 틀린 박스에서도 같은 규약이 이기면
   그것은 규약 판별이 아니라 **숫자 크기 사전확률**이다.

## 프롬프트에 범위를 적지 않는다

학습 프롬프트(`vlm/prompts/unified_pilot_v1.txt`)는 "좌표는 0에서 1000 사이의 정수다"
라고 **지시한다.** 그 프롬프트로 재면 네이티브 규약이 아니라 지시 따르기를 재게 된다.
카나리아는 범위를 말하지 않는 중립 프롬프트를 쓴다.

    uv run python scripts/canary_coords.py

산출: outputs/probe_c/canary1_coords/report.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

OUT = Path("outputs/probe_c/canary1_coords")

#: 판별할 모델. 파일럿 모델과 본실험 모델을 둘 다 본다 — 둘이 갈리면 그 자체가 결과다.
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]

#: 정답 상대 박스. **1000 을 곱해도 둥글지 않게** 골랐다(237/313/561/688) — 둥근 수
#: 선호가 규약 판별로 오독되는 것을 막는다(1차 실행에서 실제로 그 위험이 있었다).
REL_BOX = (0.237, 0.313, 0.561, 0.688)

#: 대조군용 오답 박스. 규약 선호가 정답에서만 나타나는지 확인한다.
REL_BOX_WRONG = (0.618, 0.121, 0.884, 0.446)

#: 1000 보다 작은 이미지·큰 이미지·비정방 둘. 비정방은 x·y 배율이 갈리는 경우를 만든다.
SIZES = [(448, 448), (896, 896), (672, 1120), (1344, 672)]

PROMPT = (
    "Locate the solid red rectangle in this image. "
    'Reply with only this JSON, no explanation: {"bbox_2d": [x1, y1, x2, y2]}'
)

MAX_NEW = 192          # 4B 가 서술형으로 흐르는 것을 1차 실행에서 봤다. 넉넉히 준다.
SPACES = ("NORM_1000", "ABS_ORIG", "ABS_RESIZED")


# --------------------------------------------------------------------------
# 합성 도형 · 기하
# --------------------------------------------------------------------------

def render(w: int, h: int, path: Path) -> tuple[float, float, float, float]:
    """흰 배경 + 빨간 사각형 하나. 정답 픽셀 박스를 돌려준다."""
    img = Image.new("RGB", (w, h), "white")
    box = (REL_BOX[0] * w, REL_BOX[1] * h, REL_BOX[2] * w, REL_BOX[3] * h)
    ImageDraw.Draw(img).rectangle([round(v) for v in box], fill=(220, 30, 30))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return box


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = ua + ub - inter
    return float(inter / den) if den > 0 else 0.0


def resized_dims(inputs) -> tuple[int, int] | None:
    """프로세서가 실제로 쓴 리사이즈 치수. `smart_resize` 를 재구현하지 않는다 —
    재구현이 어긋나도 아무도 모르는 상태가 함정 #4 의 원래 경로다(`vlm/coords.py`)."""
    thw = inputs.get("image_grid_thw")
    if thw is None:
        return None
    _t, gh, gw = [int(v) for v in thw[0]]
    return gw * 14, gh * 14          # patch_size 14 — Qwen VL 계열 공통


# --------------------------------------------------------------------------
# 파싱 — 1차 실행에서 여기가 틀렸다
# --------------------------------------------------------------------------

#: `bbox_2d` 의 **`2` 가 첫 숫자로 잡히는** 사고가 1차 실행에서 났다. 키 이름에 숫자가
#: 있으므로 전체 문자열을 숫자로 훑으면 안 된다. 배열을 먼저 집는다.
_ARR = re.compile(r"bbox_2d\s*\"?\s*:\s*\[([^\]]*)\]")
_BARE = re.compile(r"\[\s*-?\d[^\[\]]*?\]")
#: 서술형 답. **순서를 가정하지 않는다** — 1차 실행의 4B 는 "x1=110 to x2=250, and
#: y1=134" 처럼 x1·x2·y1·y2 순으로 냈다. 이름으로 하나씩 찾는다.
_XY = {k: re.compile(rf"\b{k}\s*[=:]\s*(-?\d+(?:\.\d+)?)") for k in ("x1", "y1", "x2", "y2")}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_box(text: str) -> list[float]:
    """생성문에서 bbox 네 숫자를 뽑는다. 세 형태를 순서대로 시도한다."""
    m = _ARR.search(text)
    if m:
        nums = [float(x) for x in _NUM.findall(m.group(1))]
        if len(nums) >= 4:
            return nums[:4]
    hits = {k: r.search(text) for k, r in _XY.items()}   # 4B 는 서술형으로 흐른다
    if all(hits.values()):
        return [float(hits[k].group(1)) for k in ("x1", "y1", "x2", "y2")]
    m = _BARE.search(text)                     # 키 없이 배열만
    if m:
        nums = [float(x) for x in _NUM.findall(m.group(0))]
        if len(nums) >= 4:
            return nums[:4]
    return []


def implied_canvas(nums, rel=REL_BOX) -> list[float] | None:
    """네 숫자를 정답 상대좌표로 나눈 값 — **모델이 상정한 캔버스 크기**다.

    주 판정의 근거이며 토큰 사전확률에 오염되지 않는다.
    """
    if len(nums) != 4:
        return None
    return [round(n / v, 1) for n, v in zip(nums, rel)]


#: 상정 캔버스가 어느 후보와도 이만큼 넘게 어긋나면 **모델이 물체를 못 찾은 것**이지
#: 규약이 다른 것이 아니다. 접지 실패에 표를 주면 판별이 오염된다 — 1차 교정본에서
#: 4B 의 672x1120 이 x2=841(이미지 폭 672 초과)인 엉터리 박스를 내고도 한 표를 던졌다.
CANVAS_TOL = 0.15


def classify_canvas(canvas, W: int, H: int, rw, rh) -> tuple[str | None, dict, bool]:
    """상정 캔버스를 세 후보 [W,H,W,H] · [1000]*4 · [rw,rh,rw,rh] 와 대조한다.

    Returns:
        (판정, 후보별 상대오차, 표를 줄 만한가). 셋째가 거짓이면 접지 실패다.
    """
    if canvas is None:
        return None, {}, False
    cands = {"ABS_ORIG": [W, H, W, H], "NORM_1000": [1000.0] * 4}
    if rw:
        cands["ABS_RESIZED"] = [rw, rh, rw, rh]
    err = {k: round(sum(abs(c - v) / v for c, v in zip(canvas, ref)) / 4, 4)
           for k, ref in cands.items()}
    best = min(err, key=err.get)
    return best, err, err[best] <= CANVAS_TOL


# --------------------------------------------------------------------------
# 우도 후보 문자열
# --------------------------------------------------------------------------

def as_px(nums, space: str, W: int, H: int, rw, rh):
    """모델이 낸 네 숫자를 `space` 규약으로 읽었을 때의 원본 픽셀 박스."""
    x1, y1, x2, y2 = nums
    if space == "ABS_ORIG":
        return x1, y1, x2, y2
    if space == "NORM_1000":
        return x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H
    if not rw:
        return None
    return x1 * W / rw, y1 * H / rh, x2 * W / rw, y2 * H / rh


def candidate(space: str, box_px, W: int, H: int, rw, rh) -> str | None:
    """박스를 `space` 규약의 답 문자열로 만든다."""
    x1, y1, x2, y2 = box_px
    if space == "ABS_ORIG":
        v = (x1, y1, x2, y2)
    elif space == "NORM_1000":
        v = (x1 / W * 1000, y1 / H * 1000, x2 / W * 1000, y2 / H * 1000)
    else:
        if not rw:
            return None
        v = (x1 * rw / W, y1 * rh / H, x2 * rw / W, y2 * rh / H)
    return json.dumps({"bbox_2d": [round(n) for n in v]}, separators=(",", ":"))


# --------------------------------------------------------------------------

def probe_model(model_id: str, images: list[dict]) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    proc = AutoProcessor.from_pretrained(model_id)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    # 학습 경로와 같은 4bit 로 띄운다. 규약은 양자화로 바뀌지 않지만, 실제로 돌릴
    # 구성에서 재는 편이 낫다.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0})
    model.eval()

    rows = []
    for rec in images:
        img = Image.open(rec["path"]).convert("RGB")
        W, H = img.size
        user = {"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": PROMPT}]}
        enc0 = proc.apply_chat_template([user], tokenize=True, return_dict=True,
                                        return_tensors="pt", add_generation_prompt=True)
        dims = resized_dims(enc0)
        rw, rh = dims if dims else (None, None)
        n_prompt = int(enc0["input_ids"].shape[1])
        enc = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in enc0.items()}

        # -- A. 생성 + 스케일 판별 (greedy 1회, 재시도 없음) ------------------
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        text = proc.batch_decode(gen[:, n_prompt:], skip_special_tokens=True)[0].strip()
        nums = parse_box(text)
        canvas = implied_canvas(nums)
        canvas_verdict, canvas_err, canvas_ok = classify_canvas(canvas, W, H, rw, rh)
        gen_iou = {}
        if len(nums) == 4:
            for sp in SPACES:
                px = as_px(nums, sp, W, H, rw, rh)
                gen_iou[sp] = round(iou(px, rec["gt_px"]), 4) if px else None

        # -- B. 우도 argmax + 대조군 ------------------------------------------
        def score(box_px) -> dict:
            out_ = {}
            for sp in SPACES:
                ans = candidate(sp, box_px, W, H, rw, rh)
                if ans is None:
                    out_[sp] = None
                    continue
                full = proc.apply_chat_template(
                    [user, {"role": "assistant", "content": [{"type": "text", "text": ans}]}],
                    tokenize=True, return_dict=True, return_tensors="pt")
                ids = full["input_ids"]
                cu = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in full.items()}
                # 답 구간의 로짓만 물질화한다(명세 판정 11 과 같은 기제). 전 위치 x vocab
                # 을 뜨면 4B·이미지 970토큰에서 수 GB 가 되고 16GB 에서 잴 수 없다.
                n_keep = int(ids.shape[1] - n_prompt + 1)
                with torch.no_grad():
                    o = model(**cu, logits_to_keep=n_keep)
                logits = o.logits[:, :-1].float()
                tgt = ids[:, -(n_keep - 1):].to("cuda")   # 남긴 로짓과 정확히 겹친다
                lp = torch.log_softmax(logits, dim=-1)
                tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).reshape(-1)
                out_[sp] = {"sum_logprob": round(float(tok.sum()), 4),
                            "mean_logprob": round(float(tok.mean()), 4),
                            "n_tokens": int(tok.numel()), "answer": ans}
            return out_

        ll = score(rec["gt_px"])
        ll_ctrl = score(rec["wrong_px"])

        def am(d, key):
            sc = {k: v[key] for k, v in d.items() if v}
            return max(sc, key=sc.get) if sc else None

        rows.append({
            "size": [W, H], "resized": [rw, rh],
            "gt_px": [round(v, 1) for v in rec["gt_px"]],
            "generated": text[:300], "parsed_numbers": nums,
            "상정_캔버스": canvas,
            "상정_캔버스_판정": canvas_verdict if canvas_ok else None,
            "상정_캔버스_최근접": canvas_verdict,
            "상정_캔버스_상대오차": canvas_err,
            # 거짓이면 **접지 실패**다 — 규약이 다른 것이 아니라 물체를 못 찾은 것이라
            # 표를 주지 않는다. 어느 후보와도 CANVAS_TOL 넘게 어긋난 경우다.
            "판별가능": canvas_ok,
            "생성_IoU_규약별": gen_iou,
            "생성_IoU_argmax": max(gen_iou, key=lambda k: (gen_iou[k] or -1)) if gen_iou else None,
            "우도_정답": ll, "우도_대조군_오답박스": ll_ctrl,
            "우도_argmax_합": am(ll, "sum_logprob") if canvas_ok else None,
            "우도_argmax_평균": am(ll, "mean_logprob") if canvas_ok else None,
            "우도_argmax_합_무관문": am(ll, "sum_logprob"),
            "우도_대조군_argmax_합": am(ll_ctrl, "sum_logprob"),
        })
        print(f"  {W}x{H} (resized {rw}x{rh}) 정답 {[round(v) for v in rec['gt_px']]}\n"
              f"     생성 {nums}  → 상정 캔버스 {canvas}  (이미지 [{W},{H},{W},{H}])\n"
              f"     캔버스 최근접 {canvas_verdict}  상대오차 {canvas_err}"
              f"  {'→ 표 인정' if canvas_ok else '→ **접지 실패, 표 제외**'}\n"
              f"     IoU {gen_iou}\n"
              f"     우도 argmax 합 {rows[-1]['우도_argmax_합']} · "
              f"평균 {rows[-1]['우도_argmax_평균']} · "
              f"대조군(오답박스) {rows[-1]['우도_대조군_argmax_합']}", flush=True)

    del model
    torch.cuda.empty_cache()

    def vote(key):
        vals = [r[key] for r in rows if r[key]]
        return {s: vals.count(s) for s in SPACES if vals.count(s)}

    return {
        "model_id": model_id, "이미지별": rows,
        "판별가능_장수": sum(1 for r in rows if r["판별가능"]),
        "캔버스_투표_주판정": vote("상정_캔버스_판정"),
        "생성IoU_투표": vote("생성_IoU_argmax"),
        "우도_투표_합": vote("우도_argmax_합"),
        "우도_투표_평균": vote("우도_argmax_평균"),
        "우도_대조군_투표": vote("우도_대조군_argmax_합"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images = []
    for w, h in SIZES:
        p = OUT / f"synth_{w}x{h}.png"
        images.append({
            "path": p, "gt_px": render(w, h, p),
            "wrong_px": tuple(r * d for r, d in zip(REL_BOX_WRONG, (w, h, w, h))),
        })
    print(f"합성 도형 {len(images)}장 · 정답 상대 박스 {REL_BOX} "
          f"(1000 스케일 {[round(v*1000) for v in REL_BOX]} — 둥근 수 아님)", flush=True)

    t0 = time.perf_counter()
    results = []
    for mid in MODELS:
        print(f"\n=== {mid} ===", flush=True)
        try:
            results.append(probe_model(mid, images))
        except Exception as e:                      # noqa: BLE001
            print(f"  실패: {type(e).__name__}: {e}", flush=True)
            results.append({"model_id": mid, "error": f"{type(e).__name__}: {e}"})

    rep = {
        "무엇을_쟀나": "미세조정 없는 사전학습 모델의 네이티브 bbox 좌표 규약",
        "주판정": "상정 캔버스 = 생성 좌표 / 정답 상대좌표. 토큰 사전확률에 오염되지 않는다",
        "보조판정": "우도 argmax + 오답 박스 대조군(규약 선호가 정답에서만 나타나는지)",
        "프롬프트": PROMPT,
        "주의": "학습 프롬프트와 다르다 — 학습 프롬프트는 0~1000 을 지시하므로 "
                "그것으로 재면 네이티브 규약이 아니라 지시 따르기를 재게 된다.",
        "정답_상대박스": REL_BOX, "대조군_오답_상대박스": REL_BOX_WRONG,
        "크기": SIZES, "max_new_tokens": MAX_NEW,
        "결과": results,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n총 {rep['wall_s']:.0f}s → {OUT / 'report.json'}")
    for r in results:
        print(f"  {r['model_id']}: 판별가능 {r.get('판별가능_장수')}/4 · "
              f"캔버스(주) {r.get('캔버스_투표_주판정')} / "
              f"우도합 {r.get('우도_투표_합')} / 대조군 {r.get('우도_대조군_투표')}")


if __name__ == "__main__":
    main()
