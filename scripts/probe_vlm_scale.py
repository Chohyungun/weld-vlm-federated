"""프로브 2 — 모델 크기-시간 곡선 + 과제 4(감독 위치 한정 로짓) 이행 전후 실측.

## 왜 곡선을 잡는가

61번·67번의 통합형 외삽이 "모델 배율 ×3~5" 라는 **폭 2배짜리 상수**에 걸려 있다.
0.8B 실측 한 점만 있으니 4B 를 추정할 근거가 문서적 관례뿐이었다. 같은 코드 경로에서
0.8B·2B·4B 세 점을 재면 그 폭이 실측으로 닫힌다.

세 점 모두 **같은 경로**를 통과해야 한다 — 같은 QLoRA 4bit·같은 LoRA 접미사·같은
gradient checkpointing·같은 micro/accum·같은 프롬프트·같은 표본. 하나라도 다르면
곡선이 아니라 서로 다른 실험 셋이 된다.

## 판정 11 이행 전후

30번 명세 판정 11(감독 위치 한정 로짓)은 최적화가 아니라 **미이행 명세**였다. 이행 전은
전 위치 × vocab 248,320 로짓을 물질화한다. 이행 후는 감독 구간(프롬프트 뒤 접미)만 뽑는다.

**등가부터 확인한다.** 빨라졌는데 손실이 달라졌으면 그건 최적화가 아니라 버그다.
같은 표본·같은 시드로 두 경로의 epoch 손실과 감독 토큰 수를 대조한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from vlm.pilot_vlm import load_pairs, train_rounds

OUT = Path("outputs/probe_c/probe2_vlm_scale.json").resolve()
N_ROWS = 48          # 유효 배치 32 → 1.5 스텝. 크기별 상대비교에는 충분하다
WARM_ROWS = 0        # train_rounds 는 표본 단위 계측을 노출하지 않는다 — epoch 평균으로 본다

#: (라벨, model_id, supervised_logits_only)
LEGS = [
    ("0.8B_before", "Qwen/Qwen3.5-0.8B", False),
    ("0.8B_after", "Qwen/Qwen3.5-0.8B", True),
    ("2B_after", "Qwen/Qwen3.5-2B", True),
    ("4B_after", "Qwen/Qwen3.5-4B", True),
]


def run(label: str, model_id: str, sup_only: bool, rows: list[dict]) -> dict:
    seen: list[dict] = []

    def log_cb(ep, ce, steps, wall):
        seen.append({"epoch": ep, "ce": round(ce, 6), "steps": steps, "wall_s": round(wall, 1)})

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    _, _, metrics, _ = train_rounds(
        rows=rows, epochs=1, round_idx=0, client_idx=0, base_seed=0,
        log_cb=log_cb, model_id=model_id, supervised_logits_only=sup_only,
    )
    wall_total = time.perf_counter() - t0
    out = {
        "leg": label, "model_id": model_id, "supervised_logits_only": sup_only,
        "n_rows": len(rows),
        "train_wall_s": round(metrics["wall_s"], 2),
        "s_per_sample": round(metrics["wall_s"] / len(rows), 4),
        "incl_load_wall_s": round(wall_total, 1),
        "peak_vram_gb": round(metrics["peak_vram_gb"], 3),
        "supervised_tokens": metrics["supervised_tokens"],
        "optimizer_steps": metrics["optimizer_steps"],
        "epoch_ce": seen[-1]["ce"] if seen else None,
        "payload_mb": round(metrics["payload_bytes"] / 1e6, 2),
    }
    torch.cuda.empty_cache()
    return out


def main() -> None:
    rows = load_pairs("train")[:N_ROWS]
    print(f"표본 {len(rows)}행", flush=True)
    legs = []
    for label, mid, sup in LEGS:
        print(f"\n=== {label} ({mid}, supervised_logits_only={sup}) ===", flush=True)
        try:
            r = run(label, mid, sup, rows)
        except Exception as exc:  # noqa: BLE001 - 한 다리 실패로 나머지를 잃지 않는다
            r = {"leg": label, "model_id": mid, "error": f"{type(exc).__name__}: {exc}"}
            print(f"  실패: {r['error']}", flush=True)
            torch.cuda.empty_cache()
        legs.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    by = {l["leg"]: l for l in legs if "error" not in l}
    report = {"n_rows": len(rows), "legs": legs}

    # (1) 판정 11 등가 — 손실·감독 토큰이 같아야 한다
    if "0.8B_before" in by and "0.8B_after" in by:
        b, a = by["0.8B_before"], by["0.8B_after"]
        report["ruling11"] = {
            "ce_before": b["epoch_ce"], "ce_after": a["epoch_ce"],
            "ce_abs_diff": (None if None in (b["epoch_ce"], a["epoch_ce"])
                            else round(abs(b["epoch_ce"] - a["epoch_ce"]), 8)),
            "supervised_tokens_equal": b["supervised_tokens"] == a["supervised_tokens"],
            "s_per_sample_before": b["s_per_sample"], "s_per_sample_after": a["s_per_sample"],
            "speedup": round(b["s_per_sample"] / a["s_per_sample"], 3),
            "peak_vram_before_gb": b["peak_vram_gb"], "peak_vram_after_gb": a["peak_vram_gb"],
            "vram_saved_gb": round(b["peak_vram_gb"] - a["peak_vram_gb"], 3),
        }

    # (2) 크기-시간 곡선. 두 점 사이 기울기를 로그-로그에서 잡고 4B 실측과 대조한다
    pts = [(0.8, by["0.8B_after"]["s_per_sample"]) if "0.8B_after" in by else None,
           (2.0, by["2B_after"]["s_per_sample"]) if "2B_after" in by else None,
           (4.0, by["4B_after"]["s_per_sample"]) if "4B_after" in by else None]
    pts = [p for p in pts if p]
    if len(pts) >= 2:
        import math
        curve = {"points_b_vs_s_per_sample": pts}
        (x0, y0), (x1, y1) = pts[0], pts[1]
        alpha = math.log(y1 / y0) / math.log(x1 / x0)
        curve["exponent_first_two"] = round(alpha, 3)
        curve["pred_4B_from_first_two"] = round(y0 * (4.0 / x0) ** alpha, 4)
        if len(pts) == 3:
            curve["measured_4B"] = pts[2][1]
            curve["ratio_4B_over_0.8B"] = round(pts[2][1] / pts[0][1], 3)
            curve["pred_error_pct"] = round(
                100 * (curve["pred_4B_from_first_two"] - pts[2][1]) / pts[2][1], 1)
        report["scaling"] = curve

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {OUT}", flush=True)
    for k in ("ruling11", "scaling"):
        if k in report:
            print(k, json.dumps(report[k], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
