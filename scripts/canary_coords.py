"""카나리아-1 — 사전학습 모델의 네이티브 좌표 규약 실측 판별 (게이트 G5 · 함정 #4).

## 왜 이것을 재는가

`vlm/coords.py` 의 `NORM_1000` 채택 근거가 **문서적**이었다(Qwen3VLProcessor 공유 +
쿡북의 0~1000 명시). 61번이 스스로 "본실험 착수 게이트"로 등급을 매겨 놓고 미실시로
남긴 항목이고, 67번 §5-1 이 재확인했다. 좌표 규약이 어긋나면 지표가 붕괴하는 구간이라
문서 근거로 넘길 자리가 아니다.

66번이 확인한 "매칭쌍 IoU 0.31~0.34" 는 **우리 파이프라인 내부 정합**이다 — 우리가
`NORM_1000` 으로 만든 타깃을 `NORM_1000` 으로 되읽으면 당연히 맞는다. 사전학습 모델이
어느 규약을 쓰는지는 그 실험으로 알 수 없다.

## 2차 개정 (2026-09-21) — 1차 실행은 세 규약 중 둘을 판별하지 못했다

1차 실행(`outputs/probe_c/canary1_coords/`)은 `NORM_1000` 을 배제하고 "절대 픽셀"까지
판별했다. 그 결론은 유지된다. 그러나 **절대 픽셀이 원본 프레임인지 프로세서가 실제로
본 리사이즈 프레임인지는 판별하지 못했다.** 이유가 둘이다(02번 I-5).

1. `resized_dims` 가 격자에 **14** 를 곱했다. 실제 설정은 패치 16 · 병합 2 다
   (`preprocessor_config.json`). 그래서 리사이즈 후보 자체가 틀린 값이었고,
   "리사이즈라면 588·980 이어야 한다" 는 배제 논증도 그 틀린 값 위에 섰다.
2. 1차의 네 크기(448² · 896² · 672×1120 · 1344×672)가 **모두 32 의 배수**여서
   프로세서 리사이즈가 **항등**이었다. 두 후보가 같은 값이므로 판별할 것이 없었다.

이 판은 셋을 고친다.

- 리사이즈 치수를 프로세서 설정에서 읽는다(상수를 코드에 박지 않는다).
- 크기를 **판별력**으로 고른다 — 리사이즈 배율이 원본과 크게 다른 크기를 넣고,
  항등인 크기는 검사기 건전성 대조군으로만 쓴다.
- 본실험 크기 **1280×720** 은 y 배율 차가 2.2 % 뿐이라 한 장으로는 판별이 서지 않는다.
  박스를 여러 개 넣어 **분포로** 판정한다.

## 판별 A — 상정 캔버스 (주 판정)

합성 도형을 **같은 상대 위치로 여러 이미지 크기에 렌더링**하고, 모델이 낸 숫자를 정답
상대좌표로 나눈다. 그 몫이 모델이 상정한 **캔버스 크기**다.

    상정 캔버스 = 생성 좌표 / 정답 상대좌표

- 이미지 크기를 따라가면 → `ABS_ORIG`
- 크기와 무관하게 1000 근처면 → `NORM_1000`
- 프로세서 리사이즈 치수를 따라가면 → `ABS_RESIZED`

**이 판별은 토큰 사전확률에 오염되지 않는다.** 아래 B 가 오염될 수 있어서 A 를 주 판정으로 둔다.
축을 따로 본다 — 리사이즈는 x·y 배율이 다를 수 있고(1280×720 은 y 만 줄어든다), 축을
합치면 그 신호가 묻힌다.

## 판별 B — 우도 argmax (보조, 기본 꺼짐)

같은 프롬프트에 세 규약의 정답 문자열을 붙여 교사 강제 로그우도를 잰다. 1차에서 이
검사의 함정을 확인했다 — 정답 상대 박스가 둥근 수였고, 언어 모델은 이미지와 무관하게
둥근 수를 선호한다. 그래서 정답 박스를 1000 스케일에서 둥글지 않게 잡고(237/313/561/688)
**오답 박스 대조군**을 함께 잰다. 규약 선호가 진짜라면 정답에서만 나타나야 한다.

2차에서 기본을 끈 이유: `ABS_ORIG` 와 `ABS_RESIZED` 의 답 문자열이 본실험 크기에서
2 % 밖에 다르지 않아 **이 축의 판별에는 힘이 없다**. 배율 차가 큰 크기에서만 의미가 있어
`--likelihood` 로 켠다.

## 프롬프트에 범위를 적지 않는다

학습 프롬프트는 "좌표는 …" 범위를 **지시한다.** 그 프롬프트로 재면 네이티브 규약이 아니라
지시 따르기를 재게 된다. 카나리아는 범위를 말하지 않는 중립 프롬프트를 쓴다.

채팅 템플릿은 **`enable_thinking=False` 로 고정한다**(02번 I-4 · `scripts/probe/template_prefix.py`).
미지정이면 4B 는 생각 모드로 흘러 좌표가 사고문 안에 섞이고, 그것은 학습·추론이 쓸 설정이 아니다.

    uv run python scripts/canary_coords.py                 # 4B 만, 판별 A
    uv run python scripts/canary_coords.py --models both   # 0.8B 도
    uv run python scripts/canary_coords.py --likelihood    # 판별 B 도 (배율 차 큰 크기만)

산출: `outputs/probe_c/canary2_coords/report.json`. 1차 산출물은 건드리지 않는다.
평가셋 이미지·정답은 쓰지 않는다 — 전부 합성 도형이다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image, ImageDraw

OUT = Path("outputs/probe_c/canary2_coords")

MODEL_MAIN = "Qwen/Qwen3.5-4B"        # 본실험 모델
MODEL_PILOT = "Qwen/Qwen3.5-0.8B"     # 파일럿 모델 — 둘이 갈리면 그 자체가 결과다

#: 규약 판별용 박스. **1000 을 곱해도 둥글지 않게** 골랐다 — 둥근 수 선호가 규약 판별로
#: 오독되는 것을 막는다(1차 실행에서 실제로 그 위험이 있었다).
REL_BOXES = [
    (0.237, 0.313, 0.561, 0.688),
    (0.114, 0.142, 0.387, 0.469),
    (0.523, 0.561, 0.842, 0.913),
]

#: 대조군용 오답 박스(판별 B). 규약 선호가 정답에서만 나타나는지 확인한다.
REL_BOX_WRONG = (0.618, 0.121, 0.884, 0.446)

#: 본실험 크기에서 쓰는 박스. y 배율 차가 2.2 % 뿐이라 **분포로** 판정한다.
#: y 를 넓게 흩어 놓는다 — 상단·중단·하단에서 같은 방향으로 쏠려야 신호다.
REL_BOXES_MAIN = [
    (0.237, 0.313, 0.561, 0.688), (0.114, 0.142, 0.387, 0.469),
    (0.523, 0.561, 0.842, 0.913), (0.312, 0.081, 0.688, 0.237),
    (0.156, 0.719, 0.437, 0.912), (0.641, 0.187, 0.913, 0.431),
    (0.081, 0.437, 0.312, 0.656), (0.437, 0.656, 0.719, 0.869),
    (0.269, 0.194, 0.731, 0.806), (0.587, 0.306, 0.869, 0.594),
]

#: 모드 비교군 — 1차 실행(생각 모드)과 2차(비생각)의 판정이 갈린 원인을 가린다.
#: 448² 은 1차가 `ABS_ORIG` 로 읽은 크기, 1280×720 은 본실험 크기, 240×180 은 판별력이 큰 크기다.
SIZES_MODECMP = {"modecmp_448": (448, 448), "modecmp_main": (1280, 720),
                 "modecmp_small": (240, 180)}

#: 판별력 = max(|1−x배율|, |1−y배율|). 프로세서 실측으로 고른 크기다.
#: 배율 차가 큰 것이 **규칙 판별**을 세우고, 항등인 것이 검사기 건전성을 본다.
SIZES_RULE = [(240, 180), (272, 208), (224, 224), (320, 200), (200, 360)]
SIZE_MAIN = (1280, 720)                      # 본실험 전 이미지가 이 크기다
SIZES_CONTROL = [(896, 896), (640, 480)]     # 리사이즈 항등 — 판별 불가가 정상

PROMPT = (
    "Locate the solid red rectangle in this image. "
    'Reply with only this JSON, no explanation: {"bbox_2d": [x1, y1, x2, y2]}'
)

MAX_NEW = 64            # 비생각 모드에서 JSON 한 줄이면 넉넉하다
MAX_NEW_THINK = 640     # 생각 모드는 사고문 뒤에 답이 온다
SPACES = ("NORM_1000", "ABS_ORIG", "ABS_RESIZED")

#: 접지 성공 문턱. 어느 해석으로도 이보다 못 겹치면 **물체를 못 찾은 것**이므로 표를
#: 주지 않는다. 1차 교정본에서 이미지 폭을 넘는 엉터리 박스가 한 표를 던졌다.
GROUND_IOU = 0.30
#: 상정 캔버스가 최근접 후보와도 이만큼 넘게 어긋나면 표를 주지 않는다 — 규약이 다른 것이
#: 아니라 물체를 못 찾은 것이다. 1차 교정본에서 이미지 폭을 넘는 박스가 한 표를 던졌다.
CANVAS_TOL = 0.15
#: 축별 배율 판정 문턱. 두 후보의 배율 차가 이보다 작으면 그 축은 판별 불가로 센다.
RATIO_GAP_MIN = 0.03


def render(w: int, h: int, rel, path: Path) -> tuple[float, float, float, float]:
    """흰 배경 + 빨간 사각형 하나. 정답 픽셀 박스를 돌려준다."""
    img = Image.new("RGB", (w, h), "white")
    box = (rel[0] * w, rel[1] * h, rel[2] * w, rel[3] * h)
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


def resized_dims(inputs, patch: int) -> tuple[int, int] | None:
    """프로세서가 실제로 쓴 리사이즈 치수. `smart_resize` 를 재구현하지 않는다 —
    재구현이 어긋나도 아무도 모르는 상태가 함정 #4 의 원래 경로다.

    `patch` 는 **프로세서 설정에서 읽어 넘긴다.** 1차 판은 여기에 14 를 박아 두었고
    실제 값은 16 이었다(02번 I-5).
    """
    thw = inputs.get("image_grid_thw")
    if thw is None:
        return None
    _t, gh, gw = [int(v) for v in thw[0]]
    return gw * patch, gh * patch


#: `bbox_2d` 의 **`2` 가 첫 숫자로 잡히는** 사고가 1차 실행에서 났다. 배열을 먼저 집는다.
_ARR = re.compile(r"bbox_2d\s*\"?\s*:\s*\[([^\]]*)\]")
_BARE = re.compile(r"\[\s*-?\d[^\[\]]*?\]")
#: 서술형 답. **순서를 가정하지 않는다** — 1차의 4B 는 x1·x2·y1·y2 순으로 냈다.
_XY = {k: re.compile(rf"\b{k}\s*[=:]\s*(-?\d+(?:\.\d+)?)") for k in ("x1", "y1", "x2", "y2")}
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_box(text: str) -> list[float]:
    """생성문에서 bbox 네 숫자를 뽑는다. 세 형태를 순서대로 시도한다."""
    m = _ARR.search(text)
    if m:
        nums = [float(x) for x in _NUM.findall(m.group(1))]
        if len(nums) >= 4:
            return nums[:4]
    hits = {k: r.search(text) for k, r in _XY.items()}
    if all(hits.values()):
        return [float(hits[k].group(1)) for k in ("x1", "y1", "x2", "y2")]
    m = _BARE.search(text)
    if m:
        nums = [float(x) for x in _NUM.findall(m.group(0))]
        if len(nums) >= 4:
            return nums[:4]
    return []


def implied_canvas(nums, rel) -> list[float] | None:
    """네 숫자를 정답 상대좌표로 나눈 값 — **모델이 상정한 캔버스 크기**다."""
    if len(nums) != 4:
        return None
    return [round(n / v, 1) for n, v in zip(nums, rel)]


def axis_ratios(canvas, W: int, H: int) -> dict[str, float] | None:
    """상정 캔버스를 원본 크기로 나눈 **축별 배율**. 1 이면 원본 프레임이다."""
    if canvas is None:
        return None
    return {"x": round((canvas[0] + canvas[2]) / 2 / W, 4),
            "y": round((canvas[1] + canvas[3]) / 2 / H, 4)}


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
    """박스를 `space` 규약의 답 문자열로 만든다(판별 B)."""
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


def classify_canvas(canvas, W: int, H: int, rw, rh) -> tuple[str | None, dict, bool]:
    """상정 캔버스를 **세 후보** 전부와 대조한다 — 정규화 · 원본 픽셀 · 리사이즈 픽셀.

    2차 초판에서 여기를 원본·리사이즈 **둘만** 비교하게 짰다가 오판했다. 작은 이미지에서
    상정 캔버스가 1000 근처로 나오는데(정규화), 배율 4.17 이 1.0 보다 1.333(=320/240)에
    가깝다는 이유로 `ABS_RESIZED` 표를 줬다. 후보를 빼면 남은 후보 중 하나가 반드시 이긴다.

    Returns:
        (최근접 후보, 후보별 상대오차, 표를 줄 만한가). 셋째가 거짓이면 접지 실패이거나
        최근접 후보조차 `CANVAS_TOL` 넘게 어긋난 경우다.
    """
    if canvas is None:
        return None, {}, False
    cx, cy = (canvas[0] + canvas[2]) / 2, (canvas[1] + canvas[3]) / 2
    cands = {"NORM_1000": (1000.0, 1000.0), "ABS_ORIG": (float(W), float(H))}
    if rw:
        cands["ABS_RESIZED"] = (float(rw), float(rh))
    err = {k: round((abs(cx - v[0]) / v[0] + abs(cy - v[1]) / v[1]) / 2, 4)
           for k, v in cands.items()}
    best = min(err, key=err.get)
    return best, err, err[best] <= CANVAS_TOL


def _axis_gap(orig: float, resz: float) -> bool:
    """원본과 리사이즈 후보가 이 축에서 구분될 만큼 떨어져 있는가."""
    return abs(1.0 - resz / orig) >= RATIO_GAP_MIN


def probe_model(model_id: str, images: list[dict], *, likelihood: bool) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    proc = AutoProcessor.from_pretrained(model_id)
    patch = int(proc.image_processor.patch_size)
    merge = int(proc.image_processor.merge_size)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    # 학습 경로와 같은 4bit 로 띄운다. 규약은 양자화로 바뀌지 않지만, 실제로 돌릴
    # 구성에서 재는 편이 낫다.
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, device_map={"": 0})
    model.eval()
    print(f"  프로세서 patch={patch} merge={merge}", flush=True)

    rows = []
    t_model = time.perf_counter()
    for k, rec in enumerate(images):
        img = Image.open(rec["path"]).convert("RGB")
        W, H = img.size
        user = {"role": "user", "content": [{"type": "image", "image": img},
                                            {"type": "text", "text": PROMPT}]}
        # 템플릿 고정안 — 학습·추론이 쓸 설정으로 잰다(02번 I-4).
        # 모드 비교군만 생각 모드로도 잰다(1차 실행이 그 모드였다).
        think = rec.get("mode") == "think"
        enc0 = proc.apply_chat_template([user], tokenize=True, return_dict=True,
                                        return_tensors="pt", add_generation_prompt=True,
                                        enable_thinking=think)
        dims = resized_dims(enc0, patch)
        rw, rh = dims if dims else (None, None)
        n_prompt = int(enc0["input_ids"].shape[1])
        enc = {kk: (v.to("cuda") if hasattr(v, "to") else v) for kk, v in enc0.items()}

        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False,
                                 max_new_tokens=MAX_NEW_THINK if think else MAX_NEW)
        text = proc.batch_decode(gen[:, n_prompt:], skip_special_tokens=True)[0].strip()
        nums = parse_box(text)
        canvas = implied_canvas(nums, rec["rel"])
        ratios = axis_ratios(canvas, W, H)

        gen_iou = {}
        if len(nums) == 4:
            for sp in SPACES:
                px = as_px(nums, sp, W, H, rw, rh)
                gen_iou[sp] = round(iou(px, rec["gt_px"]), 4) if px else None
        best_iou = max([v for v in gen_iou.values() if v is not None], default=0.0)
        grounded = best_iou >= GROUND_IOU

        verdict, canvas_err, canvas_ok = classify_canvas(canvas, W, H, rw, rh)
        # 원본·리사이즈가 이 크기에서 구분되는 축이 있는가 — 없으면 그 둘은 갈리지 않는다.
        sep = {"x": _axis_gap(W, rw) if rw else False,
               "y": _axis_gap(H, rh) if rh else False}

        ll = ll_ctrl = None
        if likelihood and rec.get("likelihood"):
            def score(box_px) -> dict:
                out_ = {}
                for sp in SPACES:
                    ans = candidate(sp, box_px, W, H, rw, rh)
                    if ans is None:
                        out_[sp] = None
                        continue
                    full = proc.apply_chat_template(
                        [user, {"role": "assistant", "content": [{"type": "text", "text": ans}]}],
                        tokenize=True, return_dict=True, return_tensors="pt",
                        enable_thinking=False)
                    ids = full["input_ids"]
                    cu = {kk: (v.to("cuda") if hasattr(v, "to") else v)
                          for kk, v in full.items()}
                    # 답 구간의 로짓만 물질화한다(명세 판정 11 과 같은 기제).
                    n_keep = int(ids.shape[1] - n_prompt + 1)
                    with torch.no_grad():
                        o = model(**cu, logits_to_keep=n_keep)
                    logits = o.logits[:, :-1].float()
                    tgt = ids[:, -(n_keep - 1):].to("cuda")
                    lp = torch.log_softmax(logits, dim=-1)
                    tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).reshape(-1)
                    out_[sp] = {"sum_logprob": round(float(tok.sum()), 4),
                                "mean_logprob": round(float(tok.mean()), 4),
                                "n_tokens": int(tok.numel()), "answer": ans}
                return out_

            ll, ll_ctrl = score(rec["gt_px"]), score(rec["wrong_px"])

        def am(d, key):
            if not d:
                return None
            sc = {kk: v[key] for kk, v in d.items() if v}
            return max(sc, key=sc.get) if sc else None

        rows.append({
            "군": rec["group"], "size": [W, H], "resized": [rw, rh],
            "리사이즈_배율": None if not rw else {"x": round(rw / W, 4), "y": round(rh / H, 4)},
            "rel": list(rec["rel"]), "gt_px": [round(v, 1) for v in rec["gt_px"]],
            "생성": text[:200], "파싱": nums,
            "상정_캔버스": canvas, "축별_배율": ratios,
            "최고_IoU": best_iou, "접지": grounded,
            "생성_IoU_규약별": gen_iou,
            "생성_IoU_최고규약": (max(gen_iou, key=lambda k: (gen_iou[k] or -1))
                            if gen_iou else None),
            "캔버스_상대오차": canvas_err,
            "캔버스_판정": verdict if (grounded and canvas_ok) else None,
            "캔버스_최근접": verdict,
            "원본_리사이즈_구분되는_축": sep,
            "모드": rec.get("mode", "no_think"),
            "우도_정답": ll, "우도_대조군_오답박스": ll_ctrl,
            "우도_argmax_합": am(ll, "sum_logprob") if grounded else None,
            "우도_대조군_argmax_합": am(ll_ctrl, "sum_logprob"),
        })
        cvs = (f"캔버스 x{(canvas[0]+canvas[2])/2:.0f} y{(canvas[1]+canvas[3])/2:.0f}"
               if canvas else "캔버스 없음(파싱 실패)")
        print(f"  [{k+1:2d}/{len(images)}] {rec['group']:13s} {rec.get('mode','no_think'):8s} "
              f"{W}x{H}→{rw}x{rh} {cvs} IoU {best_iou:.3f}"
              f"({rows[-1]['생성_IoU_최고규약']}) "
              f"{'판정 ' + str(verdict) if grounded and canvas_ok else '**판별불가**'}",
              flush=True)

    del model
    torch.cuda.empty_cache()

    def tally(group: str, mode: str | None = None) -> dict:
        grp = [r for r in rows if r["군"] == group
               and (mode is None or r["모드"] == mode)]
        sel = [r for r in grp if r["접지"] and r["캔버스_판정"]]
        cnt: dict[str, int] = {}
        for r in sel:
            cnt[r["캔버스_판정"]] = cnt.get(r["캔버스_판정"], 0) + 1
        iou_med: dict[str, float] = {}
        if grp:
            for sp in SPACES:
                vals = sorted((r["생성_IoU_규약별"].get(sp) or 0.0) for r in grp)
                iou_med[sp] = round(vals[len(vals) // 2], 4)
        return {"총_장수": len(grp), "판정_장수": len(sel),
                "캔버스_판정_표": cnt, "규약별_IoU_중앙값": iou_med}

    # 본실험 크기 — y 축 배율의 분포로 판정한다(두 후보 1.0 대 0.9778).
    main_rows = [r for r in rows if r["군"] == "main" and r["접지"] and r["축별_배율"]]
    main_stat = None
    if main_rows:
        ys = [r["축별_배율"]["y"] for r in main_rows]
        xs = [r["축별_배율"]["x"] for r in main_rows]
        rt = main_rows[0]["리사이즈_배율"]["y"]
        closer_resz = sum(1 for y in ys if abs(y - rt) < abs(y - 1.0))
        main_stat = {
            "n": len(ys),
            "y배율_중앙값": round(statistics.median(ys), 4),
            "y배율_평균": round(statistics.fmean(ys), 4),
            "y배율_표준편차": round(statistics.pstdev(ys), 4) if len(ys) > 1 else None,
            "y배율_최소최대": [round(min(ys), 4), round(max(ys), 4)],
            "x배율_중앙값": round(statistics.median(xs), 4),
            "두_후보": {"ABS_ORIG": 1.0, "ABS_RESIZED": rt},
            "리사이즈에_더_가까운_장수": closer_resz,
            "원본에_더_가까운_장수": len(ys) - closer_resz,
            # 잡음이 두 후보 간격(2.2 %)보다 크면 이 크기 단독으로는 판별이 서지 않는다.
            "잡음이_간격보다_큰가": (round(statistics.pstdev(ys), 4) > abs(1.0 - rt)
                              if len(ys) > 1 else None),
        }

    return {
        "model_id": model_id, "patch": patch, "merge": merge,
        "이미지별": rows,
        "규칙군_판정": tally("rule"),
        "대조군_판정": tally("control"),
        "본실험크기_판정": tally("main"),
        "모드비교": {m: {g: tally(g, m) for g in ("modecmp_448", "modecmp_main", "modecmp_small")}
                 for m in ("no_think", "think")} if any(r["군"].startswith("modecmp") for r in rows) else None,
        "본실험크기_통계": main_stat,
        "wall_s": round(time.perf_counter() - t_model, 1),
    }


def build_images() -> list[dict]:
    images = []
    for w, h in SIZES_RULE:
        for i, rel in enumerate(REL_BOXES):
            p = OUT / f"rule_{w}x{h}_{i}.png"
            images.append({"group": "rule", "path": p, "rel": rel,
                           "gt_px": render(w, h, rel, p), "likelihood": True,
                           "wrong_px": tuple(r * d for r, d in
                                             zip(REL_BOX_WRONG, (w, h, w, h)))})
    w, h = SIZE_MAIN
    for i, rel in enumerate(REL_BOXES_MAIN):
        p = OUT / f"main_{w}x{h}_{i}.png"
        images.append({"group": "main", "path": p, "rel": rel,
                       "gt_px": render(w, h, rel, p), "likelihood": False,
                       "wrong_px": tuple(r * d for r, d in
                                         zip(REL_BOX_WRONG, (w, h, w, h)))})
    for w, h in SIZES_CONTROL:
        for i, rel in enumerate(REL_BOXES[:2]):
            p = OUT / f"ctrl_{w}x{h}_{i}.png"
            images.append({"group": "control", "path": p, "rel": rel,
                           "gt_px": render(w, h, rel, p), "likelihood": False,
                           "wrong_px": tuple(r * d for r, d in
                                             zip(REL_BOX_WRONG, (w, h, w, h)))})
    return images


def build_modecmp() -> list[dict]:
    """같은 이미지를 두 템플릿 모드로 잰다 — 규약이 모드에 따라 갈리는지 본다."""
    images = []
    for mode in ("no_think", "think"):
        for group, (w, h) in SIZES_MODECMP.items():
            for i, rel in enumerate(REL_BOXES):
                p = OUT / f"{group}_{w}x{h}_{i}.png"
                images.append({"group": group, "mode": mode, "path": p, "rel": rel,
                               "gt_px": render(w, h, rel, p), "likelihood": False,
                               "wrong_px": tuple(r * d for r, d in
                                                 zip(REL_BOX_WRONG, (w, h, w, h)))})
    return images


def selftest() -> int:
    """GPU 없이 판정 함수만 검사한다. **모델을 올리기 전에 반드시 통과해야 한다.**

    2차-b 실패가 이것이 없어서 났다. 판정 함수를 고친 뒤 상수 복원을 빠뜨렸고, 모델 로드에
    GPU 9분을 쓴 다음 첫 장에서 `NameError` 로 죽었다. 여기서 세 규약의 답을 각각 만들어
    되짚으면 그런 실수가 3초 안에 드러난다.
    """
    W, H = 1280, 720
    rw, rh = 1280, 704                  # 패치 16 · 병합 2 의 실제 리사이즈
    rel = (0.25, 0.30, 0.55, 0.70)
    box_px = (rel[0] * W, rel[1] * H, rel[2] * W, rel[3] * H)
    fails = []

    # (1) 세 규약의 답을 만들어 되짚으면 그 규약으로 판정돼야 한다.
    for space in ("NORM_1000", "ABS_ORIG", "ABS_RESIZED"):
        txt = candidate(space, box_px, W, H, rw, rh)
        nums = parse_box(txt)
        got, err, ok = classify_canvas(implied_canvas(nums, rel), W, H, rw, rh)
        if not ok or got != space:
            fails.append(f"왕복 판정: {space} → {got} (ok={ok}, 오차 {err})")

    # (2) 접지 실패(캔버스가 어느 후보와도 멀다)는 표를 주지 않아야 한다.
    _, _, ok = classify_canvas([0, 0, 2500.0, 2500.0], W, H, rw, rh)
    if ok:
        fails.append("접지 실패인데 표를 줬다")

    # (3) 축 간격. 리사이즈는 32 의 배수로 **반올림**하므로 축별 어긋남이 16 px 를 넘지 않는다.
    #     그래서 두 절대 픽셀 후보의 상대 간격은 16/치수 로 묶인다 — 큰 이미지에서는 구조적으로 작다.
    #     1280×720 은 x 0 % · y 2.22 % 라 둘 다 문턱(3 %) 아래다. 이 사실이 §2-3-2 판별 불가의 근거다.
    if _axis_gap(720.0, 704.0):
        fails.append("1280×720 의 y 간격 2.22 % 가 문턱을 넘는다고 한다")
    if _axis_gap(1280.0, 1280.0):
        fails.append("x 축이 항등인데 갈린다고 한다")
    if not _axis_gap(180.0, 224.0):
        fails.append("240×180 의 y 간격 24 % 가 문턱을 못 넘는다고 한다")
    for dim in (534, 640, 1280):
        if 16.0 / dim >= RATIO_GAP_MIN:
            fails.append(f"{dim} px 축의 상한 {16/dim:.4f} 이 문턱 이상이다 — 상한 산술을 다시 본다")

    # (4) 이미지 수가 기대와 같다.
    n_fixed, n_cmp = len(build_images()), len(build_modecmp())
    want_fixed = len(SIZES_RULE) * len(REL_BOXES) + len(REL_BOXES_MAIN) + len(SIZES_CONTROL) * 2
    want_cmp = len(SIZES_MODECMP) * len(REL_BOXES) * 2
    if (n_fixed, n_cmp) != (want_fixed, want_cmp):
        fails.append(f"이미지 수 {n_fixed}+{n_cmp} ≠ {want_fixed}+{want_cmp}")

    for f in fails:
        print(f"  FAIL {f}", flush=True)
    print(f"스모크: {'통과' if not fails else f'{len(fails)}건 실패'} "
          f"(고정 {n_fixed}장 · 모드비교 {n_cmp}장)", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", choices=["main", "both"], default="main",
                    help="main=4B 만(기본) · both=0.8B 도")
    ap.add_argument("--likelihood", action="store_true",
                    help="판별 B(우도 argmax)도 잰다 — 배율 차 큰 크기에서만")
    ap.add_argument("--modes", choices=["fixed", "both"], default="both",
                    help="both=생각·비생각 두 모드 비교군을 추가로 잰다(기본)")
    ap.add_argument("--selftest", action="store_true",
                    help="GPU 없이 판정 함수만 검사하고 끝낸다 — 스크립트를 고쳤으면 먼저 돌린다")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if selftest() != 0:
        print("스모크가 실패했다 — GPU 를 쓰지 않고 멈춘다", flush=True)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    images = build_images() + (build_modecmp() if args.modes == "both" else [])
    print(f"합성 도형 {len(images)}장 "
          f"(규칙군 {len(SIZES_RULE)}크기×{len(REL_BOXES)} · "
          f"본실험크기 {len(REL_BOXES_MAIN)} · 대조군 {len(SIZES_CONTROL)}×2)", flush=True)

    models = [MODEL_MAIN] + ([MODEL_PILOT] if args.models == "both" else [])
    t0 = time.perf_counter()
    results = []
    for mid in models:
        print(f"\n=== {mid} ===", flush=True)
        try:
            results.append(probe_model(mid, images, likelihood=args.likelihood))
        except Exception as e:                      # noqa: BLE001
            print(f"  실패: {type(e).__name__}: {e}", flush=True)
            results.append({"model_id": mid, "error": f"{type(e).__name__}: {e}"})

    rep = {
        "무엇을_쟀나": "미세조정 없는 사전학습 모델의 네이티브 bbox 좌표 규약 — "
                   "세 후보(정규화·원본 픽셀·리사이즈 픽셀)와 템플릿 모드 의존성",
        "1차와_무엇이_다른가": [
            "리사이즈 치수를 프로세서 설정에서 읽는다(1차는 격자×14, 실제는 패치 16)",
            "크기를 판별력으로 골랐다(1차 네 크기는 전부 32의 배수라 리사이즈가 항등이었다)",
            "본실험 크기 1280×720 을 박스 10개로 분포 판정한다(y 배율 차 2.2 %)",
            "채팅 템플릿을 enable_thinking=False 로 고정한다(02번 I-4)",
        ],
        "주판정": "상정 캔버스 = 생성 좌표 / 정답 상대좌표. 축별로 본다",
        "보조판정": "우도 argmax + 오답 박스 대조군(--likelihood)",
        "프롬프트": PROMPT,
        "주의": "학습 프롬프트와 다르다 — 학습 프롬프트는 좌표 범위를 지시하므로 "
                "그것으로 재면 네이티브 규약이 아니라 지시 따르기를 재게 된다.",
        "정답_상대박스": REL_BOXES, "본실험크기_상대박스": REL_BOXES_MAIN,
        "대조군_오답_상대박스": REL_BOX_WRONG,
        "크기_규칙군": SIZES_RULE, "크기_본실험": list(SIZE_MAIN),
        "크기_대조군": SIZES_CONTROL,
        "접지_IoU_문턱": GROUND_IOU, "축_판정_최소간격": RATIO_GAP_MIN,
        "max_new_tokens": MAX_NEW,
        "결과": results,
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"\n총 {rep['wall_s']:.0f}s → {OUT / 'report.json'}")
    for r in results:
        if "error" in r:
            print(f"  {r['model_id']}: {r['error']}")
            continue
        print(f"  {r['model_id']}: 규칙군 {r['규칙군_판정']}")
        print(f"     대조군 {r['대조군_판정']}")
        print(f"     본실험크기 {r['본실험크기_판정']}")
        if r.get("모드비교"):
            for mode, per in r["모드비교"].items():
                print(f"     모드 {mode}: "
                      + " · ".join(f"{g.replace('modecmp_','')} {v['캔버스_판정_표']}"
                                   f" IoU {v['규약별_IoU_중앙값']}" for g, v in per.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
