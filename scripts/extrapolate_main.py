"""본실험 소요 재외삽 — 67번 합산식.

61번의 곱셈식이 무너진 이유는 산수 자체가 아니라 **손으로 계산했다는 것**이다. 표 헤더에
적어 둔 배율이 값에 곱해지지 않았고(오류 1), 고정비까지 배율에 태웠고(오류 2), 두 오차가
반대 방향이라 자릿수가 우연히 맞아 검산을 통과했다(오류 3).

그래서 계산을 스크립트에 둔다. 입력은 프로브 JSON 실물이고 중간값을 전부 출력한다.
검산하는 사람이 어느 항이 어디서 왔는지 볼 수 있어야 한다.

    시간(런) = 런당 기동비 + 첫 epoch(콜드) + (E−1) × 이후 epoch(웜)

**배율이 없다.** 본실험 프로파일에서 직접 쟀으므로 배치배율·FLOPs배율·효율이득이 모두 1이다.

## 콜드와 웜을 왜 나누는가

프로브 1b 에서 전체 규모 첫 epoch 의 로더 대기가 38.5% 였다. 디스크에서 처음 읽는
비용이다. 학습 풀 이미지가 약 2GB 이고 이 기계의 여유 RAM 이 19GB 이므로 OS 페이지
캐시에 들어간다 — **둘째 epoch 부터는 웜이라고 볼 근거가 있지만, 근거와 실측은 다르다.**
프로브 1d 가 2 epoch 을 돌려 실측한다.

클라이언트 이미지 집합은 서로 겹치지 않으므로 **클라이언트마다 첫 epoch 한 번씩** 콜드다.
④ 연합은 라운드 1 에서 세 클라이언트가 각각 한 번 치르고, 그 뒤 라운드는 캐시가 살아 있다.

    uv run python scripts/extrapolate_main.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

P1 = Path("outputs/probe_c/probe1_det_steptime.json")
P1B = Path("outputs/probe_c/probe1b_fullepoch.json")
P1D = Path("outputs/probe_c/probe1d_warm.json")
P2 = Path("outputs/probe_c/probe2_vlm_scale.json")
OUT = Path("outputs/probe_c/extrapolation.json")

#: 동결본(58번). train 만 학습 스텝을 만든다 — `val: False` 다.
TRAIN_TOTAL = 44_846
CLIENT_TRAIN = {"C1": 23_807, "C2": 14_629, "C3": 6_410}
EVAL_IMAGES = 12_461

BATCH = 32
N_EPOCHS = 100
SEEDS_DET = 3


def spe(n: int) -> int:
    """Ultralytics 학습 로더는 `drop_last=False` 다 — 마지막 부분 배치도 돈다."""
    return math.ceil(n / BATCH)


def load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def detection(cold_s: float, warm_s: float, startup_first_s: float,
              startup_warm_s: float) -> dict:
    """칸별 벽시계. 스텝 시간과 기동비를 **따로 누적한다** — 61번이 고정비를 배율에
    태워 무너진 자리라 어느 쪽이 얼마인지 보이게 둔다."""
    per_client = {c: spe(n) for c, n in CLIENT_TRAIN.items()}
    central = spe(TRAIN_TOTAL)
    rows: dict[str, dict] = {}

    def add(name: str, step_s: float, startup_s: float, runs: int, steps: int) -> None:
        rows[name] = {"runs_per_seed": runs, "steps_per_seed": steps,
                      "_step_s": step_s, "_startup_s": startup_s}

    # ③ 분리·중앙 — 런 1개. 첫 epoch 만 콜드
    add("③ 분리·중앙",
        central * (cold_s + (N_EPOCHS - 1) * warm_s), startup_first_s, 1, central * N_EPOCHS)

    # ② 분리·로컬 — 클라이언트마다 독립 런. 이미지 집합이 겹치지 않아 각각 첫 epoch 이 콜드
    add("② 분리·로컬",
        sum(s * (cold_s + (N_EPOCHS - 1) * warm_s) for s in per_client.values()),
        3 * startup_first_s, 3, sum(per_client.values()) * N_EPOCHS)

    # ④ 분리·연합 — 스텝 수는 ②와 같다(R×E=N). 런 수만 R×클라이언트다.
    # 라운드 1 에서만 콜드고, 이후 라운드는 페이지 캐시·라벨 캐시가 살아 있다.
    for R, E in ((50, 2), (100, 1)):
        add(f"④ 분리·연합 (R={R}·E={E})",
            sum(s * (cold_s + (N_EPOCHS - 1) * warm_s) for s in per_client.values()),
            3 * startup_first_s + (R - 1) * 3 * startup_warm_s,
            R * 3, sum(per_client.values()) * N_EPOCHS)

    for r in rows.values():
        sec = r.pop("_step_s") + r.pop("_startup_s")
        r["h_per_seed"] = round(sec / 3600, 2)
        r["h_all_seeds"] = round(sec * SEEDS_DET / 3600, 1)
        r["days_all_seeds"] = round(sec * SEEDS_DET / 86400, 2)

    total_days = round(sum(v["days_all_seeds"] for k, v in rows.items()
                           if "R=100" not in k), 2)
    return {
        "steps_per_epoch": {"central": central, **per_client},
        "step_s_cold": cold_s, "step_s_warm": warm_s,
        "startup_first_s": startup_first_s, "startup_warm_s": startup_warm_s,
        "rows": rows,
        "검출_3칸_합계_days_R50E2": total_days,
    }


#: ⑥ 파일럿 실측(61번): 36,767초 / (2,273행 × 6 epoch) = 2.696초/샘플.
#: 0.8B · 판정 11 **이행 전** 값이다.
PILOT_S_PER_SAMPLE = 36_767 / (2_273 * 6)
PAIRS_MAIN = 44_846      # D4 페어 본실험 규모 = train 전량


def unified(p2: dict, n_epochs_list=(3, 6)) -> dict:
    """통합형 소요. **비율만 쓴다.**

    프로브 2 의 절대값(48행 부분표본)을 본실험에 그대로 곱하지 않는다. 표본이 다르면
    이미지 크기 분포가 달라 샘플당 시간이 달라진다. 파일럿 전량 실측(2.696초/샘플)에
    프로브가 잰 **두 비율**만 곱한다.
    """
    r11 = p2["ruling11"]["speedup"]           # 판정 11 이행 이득
    scale = p2["scaling"]["ratio_4B_over_0.8B"]   # 0.8B → 4B 배율
    s_per_sample = PILOT_S_PER_SAMPLE / r11 * scale
    out = {
        "파일럿_0.8B_판정11_전_s_per_sample": round(PILOT_S_PER_SAMPLE, 4),
        "판정11_이득": r11, "모델배율_4B_대_0.8B": scale,
        "본실험_4B_판정11_후_s_per_sample": round(s_per_sample, 4),
        "epoch당_시간_h": round(PAIRS_MAIN * s_per_sample / 3600, 1),
        "칸별": {},
    }
    for n in n_epochs_list:
        sec = PAIRS_MAIN * n * s_per_sample
        out["칸별"][f"N={n} (시드 1)"] = {
            "⑥_통합·중앙_days": round(sec / 86400, 2),
            "⑦_통합·연합_days": round(sec / 86400, 2),
            "두_칸_합_days": round(2 * sec / 86400, 2),
        }
    return out


def main() -> None:
    p1b, p1d = load(P1B), load(P1D)
    if p1b is None:
        raise SystemExit("프로브 1b 가 없다.")

    cold_s = float(p1b["step_s_mean"])          # 첫 epoch 실측 평균
    startup_first = float(p1b["startup_s"])     # 라벨 전수 스캔 포함
    if p1d and len(p1d.get("epochs", [])) >= 2:
        # **epoch 전체 구간을 스텝 수로 나눈다.** 스텝시간 중앙값은 꼬리를 버리는데,
        # 그 꼬리(로더 대기)가 실제 벽시계의 3~4할이다. 벽시계를 쓰는 것이 맞다.
        eps = p1d["epochs"]
        cold_s = eps[0]["epoch_span_s"] / eps[0]["n_steps"]
        warm_s = eps[-1]["epoch_span_s"] / eps[-1]["n_steps"]
        # 1d 는 라벨 캐시가 이미 있는 상태로 떴다 — 이 값이 ④ 의 라운드당 기동비다.
        startup_warm = max(0.0, float(p1d["wall_s"]) - sum(e["epoch_span_s"] for e in eps))
        source = "프로브 1d (2 epoch 실측, epoch 구간/스텝 수)"
    else:
        warm_s = float(p1b["step_s_p50"])
        startup_warm = startup_first
        source = "프로브 1b 만 (1d 미실측 — 콜드=평균, 웜=중앙값 가정)"

    rep = {
        "출처": source,
        "검출": detection(cold_s, warm_s, startup_first, startup_warm),
        "상한_전부_콜드": detection(cold_s, cold_s, startup_first, startup_first)["rows"],
        "하한_전부_웜": detection(warm_s, warm_s, startup_first, startup_warm)["rows"],
    }
    p2 = load(P2)
    if p2 and "scaling" in p2 and "ruling11" in p2:
        rep["통합형_크기곡선"] = p2["scaling"]
        rep["판정11"] = p2["ruling11"]
        rep["통합형"] = unified(p2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
