"""본실험 소요 재외삽 — 67번 합산식.

61번의 곱셈식이 무너진 이유는 산수 자체가 아니라 **손으로 계산했다는 것**이다. 표 헤더에
적어 둔 배율이 값에 곱해지지 않았고(오류 1), 고정비까지 배율에 태웠고(오류 2), 두 오차가
반대 방향이라 자릿수가 우연히 맞아 검산을 통과했다(오류 3).

그래서 이번에는 계산을 스크립트에 둔다. 입력은 프로브 JSON 실물이고, 중간값을 전부
출력한다. 검산하는 사람이 어느 항이 어디서 왔는지 볼 수 있어야 한다.

    시간 = Σ스텝 × 스텝당시간 + 런수 × 런당기동비

**배율이 없다.** 본실험 프로파일에서 직접 쟀으므로 배치배율·FLOPs배율·효율이득이 모두
1이다. 추정이 남는 곳은 통합형의 모델 크기뿐이고 그것은 프로브 2 가 실측으로 좁힌다.

    uv run python scripts/extrapolate_main.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROBE1 = Path("outputs/probe_c/probe1_det_steptime.json")
PROBE1B = Path("outputs/probe_c/probe1b_fullepoch.json")
PROBE2 = Path("outputs/probe_c/probe2_vlm_scale.json")
OUT = Path("outputs/probe_c/extrapolation.json")

#: 동결본(58번). train 만 학습 스텝을 만든다 — `val: False` 다.
TRAIN_TOTAL = 44_846
CLIENT_TRAIN = {"C1": 23_807, "C2": 14_629, "C3": 6_410}
EVAL_IMAGES = 12_461

BATCH = 32
N_EPOCHS = 100          # 전역 예산 N
SEEDS_DET = 3
SEEDS_UNI = 1


def steps_per_epoch(n: int) -> int:
    """Ultralytics 학습 로더는 `drop_last=False` 다 — 마지막 부분 배치도 돈다."""
    return math.ceil(n / BATCH)


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def detection(step_s: float, startup_s: float) -> dict:
    per_client = {c: steps_per_epoch(n) for c, n in CLIENT_TRAIN.items()}
    central_spe = steps_per_epoch(TRAIN_TOTAL)
    local_spe = sum(per_client.values())

    rows = {}
    # ③ 분리·중앙 — 시드당 런 1개
    rows["③ 분리·중앙"] = {
        "steps_per_seed": central_spe * N_EPOCHS, "runs_per_seed": 1, "seeds": SEEDS_DET,
    }
    # ② 분리·로컬 — 시드당 런 3개(클라이언트마다)
    rows["② 분리·로컬"] = {
        "steps_per_seed": local_spe * N_EPOCHS, "runs_per_seed": 3, "seeds": SEEDS_DET,
    }
    # ④ 분리·연합 — 스텝 수는 ②와 같다(R×E=N). 런 수만 R×클라이언트다.
    for R, E in ((50, 2), (100, 1)):
        rows[f"④ 분리·연합 (R={R}·E={E})"] = {
            "steps_per_seed": local_spe * N_EPOCHS, "runs_per_seed": R * 3, "seeds": SEEDS_DET,
        }

    for name, r in rows.items():
        step_time = r["steps_per_seed"] * step_s
        fixed = r["runs_per_seed"] * startup_s
        r["step_time_h_per_seed"] = round(step_time / 3600, 2)
        r["startup_h_per_seed"] = round(fixed / 3600, 2)
        r["total_h_per_seed"] = round((step_time + fixed) / 3600, 2)
        r["total_h_all_seeds"] = round((step_time + fixed) * r["seeds"] / 3600, 1)
        r["total_days_all_seeds"] = round((step_time + fixed) * r["seeds"] / 86400, 2)
        r["startup_share"] = round(fixed / (step_time + fixed), 3)

    return {
        "steps_per_epoch": {"central": central_spe, **per_client, "local_sum": local_spe},
        "step_s": step_s, "startup_s": startup_s, "rows": rows,
    }


def main() -> None:
    p1, p1b = load(PROBE1), load(PROBE1B)
    if p1b is None:
        raise SystemExit("프로브 1b 가 아직 없다. 전체 규모 1 epoch 실측이 있어야 한다.")

    # 스텝당 시간은 **전체 규모 실측(1b)** 을 쓴다. 프로브 1 은 목록 앞부분 표본이라
    # 배경 이미지 비율이 68.6% 로 전체 49.8% 와 다르고, 배경은 손실 계산이 가볍다.
    step_s = float(p1b["step_s_p50"])
    startup_s = float(p1b["startup_s"])

    report = {
        "출처": {
            "step_s_p50": f"{PROBE1B.name} (전체 중앙 뷰 1 epoch, 44,846장)",
            "startup_s": f"{PROBE1B.name} (wall − 배치 구간)",
        },
        "검출": detection(step_s, startup_s),
    }

    if p1 is not None:
        report["대조_부분표본"] = {
            "mosaic_on_p50": p1["legs"][0]["step_s_p50"],
            "mosaic_off_p50": p1["legs"][1]["step_s_p50"],
            "주의": "9,602장 앞부분 표본. 배경 비율이 달라 전체 규모와 직접 비교하지 않는다",
        }

    p2 = load(PROBE2)
    if p2 and "scaling" in p2:
        sc = p2["scaling"]
        report["통합형_크기곡선"] = sc

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
