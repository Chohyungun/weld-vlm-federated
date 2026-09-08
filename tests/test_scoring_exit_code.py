"""채점 전 조치 3건 — 13_2차폐쇄검증 §3 말미.

| # | 조치 | 여기서 고정하는 것 |
|---|---|---|
| D-7 | 차단 게이트를 `score` 종료 코드에 반영 | 차단 실패 → 2, 대조 불일치 → 1, 정상 → 0. **실제 프로세스**에서 클라우드 로깅 환경변수 하나로 2 가 나온다 |
| D-1 | 층화 채점을 채점 경로에 통합 | `score_cells_v1.json` 안에 `stratified` 블록, 지름길 행 lift 0, `stratified_scoring` 게이트 통과 |
| D-8 파생 | `prereg_recomputed_v1.json` 선배치 | 채점 디렉터리에 없으면 채점기가 만들고, prereg 게이트가 skipped 가 아니라 **판정**한다 |

파일럿 산출물(`outputs/pilot_c`·`outputs/pilot_d`)이 있는 기계에서만 도는 실행 시험이
절반이다 — "실행이 밟아야 드러난다"(13번 §5). 없으면 skip 이고 그 사실이 보고에 남는다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluation.params import FROZEN_SNAPSHOT, PILOT_SEED, PILOT_SNAPSHOT
from scripts.probe.score_cells import (
    EXIT_GATE_BLOCKED,
    EXIT_OK,
    EXIT_REGRESSION,
    exit_code,
)

PILOT_D = REPO / "outputs" / "pilot_d"
PILOT_C = REPO / "outputs" / "pilot_c"
DET_TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
DET_FILES = [f"{t}_s{PILOT_SEED}.jsonl" for t in DET_TAGS]
RAW_FILES = [f"{t}_raw_s{PILOT_SEED}.jsonl" for t in DET_TAGS]
"""하한 레코드 — **곡선의 입력**이다(총괄 판정 1). 없으면 `sweep_curve_recorded` 가 차단한다."""


# --------------------------------------------------------------------------------------
# D-7 — 종료 코드 결정 (순수 함수)
# --------------------------------------------------------------------------------------

def test_blocking_failure_is_nonzero() -> None:
    code, why = exit_code({"blocking_failures": ["no_cloud_logging"]}, {})
    assert code == EXIT_GATE_BLOCKED
    assert "no_cloud_logging" in why and "결과로 쓰지 마라" in why


def test_regression_mismatch_is_nonzero() -> None:
    reg = {"65번": {"checked": True, "identical": False, "diffs": []}}
    code, _ = exit_code({"blocking_failures": []}, reg)
    assert code == EXIT_REGRESSION


def test_gate_block_outranks_regression() -> None:
    reg = {"65번": {"checked": True, "identical": False, "diffs": []}}
    code, _ = exit_code({"blocking_failures": ["scoring_population"]}, reg)
    assert code == EXIT_GATE_BLOCKED


def test_clean_run_is_zero_and_unchecked_regression_is_not_a_failure() -> None:
    reg = {"65번": {"checked": False, "reason": "없음"},
           "66번": {"checked": True, "identical": True, "diffs": []}}
    code, _ = exit_code({"blocking_failures": []}, reg)
    assert code == EXIT_OK


# --------------------------------------------------------------------------------------
# D-8 파생 — 선배치
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(not (REPO / FROZEN_SNAPSHOT / "SNAPSHOT.sha256").exists(),
                    reason="동결 스냅샷 없음")
def test_prereg_constants_are_materialized_into_scoring_dir(tmp_path: Path) -> None:
    """채점 디렉터리가 비어 있어도 prereg 게이트 입력이 생긴다 — skipped 로 갈리지 않는다."""
    from evaluation.params import ScoringParams
    from scripts.probe.score_cells import _measured_prereg

    params = ScoringParams(snapshot=Path(PILOT_SNAPSHOT), pilot=PILOT_C, out=tmp_path)
    assert not (tmp_path / "prereg_recomputed_v1.json").exists()
    m = _measured_prereg(params)
    assert (tmp_path / "prereg_recomputed_v1.json").exists()
    assert m["all_positive_macro_f1"] == pytest.approx(0.21111837, abs=1e-8)
    assert m["snapshot_digest"]
    # 두 번째 호출은 되읽는다 — 값이 같다
    assert _measured_prereg(params) == m


# --------------------------------------------------------------------------------------
# 실행 시험 — 실제 프로세스에서 종료 코드·층화 블록·prereg 판정을 본다
# --------------------------------------------------------------------------------------

def _have_pilot_inputs() -> bool:
    need = [
        REPO / PILOT_SNAPSHOT / "SNAPSHOT.sha256",
        REPO / FROZEN_SNAPSHOT / "SNAPSHOT.sha256",
        PILOT_C / "predictions" / "uni_central.generations.jsonl",
        PILOT_C / "predictions" / "uni_fed.generations.jsonl",
        *[PILOT_D / f for f in DET_FILES],
        *[PILOT_D / "sweep" / f for f in RAW_FILES],
    ]
    return all(p.exists() for p in need)


def _run_score(out: Path, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MLFLOW_", "WANDB_", "COMET_", "NEPTUNE_", "CLEARML_"))}
    env.update({"PYTHONIOENCODING": "utf-8", **env_extra})
    return subprocess.run(
        [sys.executable, "scripts/probe/score_cells.py", "score", "--out", str(out)],
        cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=1800, check=False,   # 종료 코드가 곧 시험 대상이다
    )


@pytest.fixture(scope="module")
def scoring_runs(tmp_path_factory):
    if not _have_pilot_inputs():
        pytest.skip("파일럿 산출물(outputs/pilot_c·pilot_d)이 없다 — 실행 시험 불가")
    base = tmp_path_factory.mktemp("score")
    runs = {}
    for name, env_extra in (
        ("blocked", {"MLFLOW_TRACKING_URI": "https://cloud.example/mlflow"}),
        ("clean", {}),
    ):
        out = base / name
        out.mkdir()
        for f in DET_FILES:
            shutil.copy(PILOT_D / f, out / f)
        (out / "sweep").mkdir()
        for f in RAW_FILES:
            shutil.copy(PILOT_D / "sweep" / f, out / "sweep" / f)
        proc = _run_score(out, env_extra)
        runs[name] = (proc, out)
    return runs


def test_cloud_logging_env_blocks_the_scoring_process(scoring_runs) -> None:
    """**D-7 실측.** `MLFLOW_TRACKING_URI=https://…` 하나로 `score` 가 2 로 죽는다.
    산출물은 증거로 남고 그 안에 차단 사실이 적혀 있다."""
    proc, out = scoring_runs["blocked"]
    assert proc.returncode == EXIT_GATE_BLOCKED, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    assert payload["exit_code"] == EXIT_GATE_BLOCKED
    assert payload["gates_evaluated"]["ok"] is False
    assert "no_cloud_logging" in payload["gates_evaluated"]["blocking_failures"]
    assert "종료 코드 2" in proc.stdout


def test_clean_scoring_process_exits_zero_with_stratified_block(scoring_runs) -> None:
    """**D-1·D-8 실측.** 깨끗한 환경에서 0 이고, 같은 산출물 안에 층화 블록이 있으며,
    prereg 게이트가 skipped 가 아니라 판정했다."""
    proc, out = scoring_runs["clean"]
    assert proc.returncode == EXIT_OK, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    assert payload["exit_code"] == EXIT_OK

    # D-1 — 층화 블록이 같은 산출물 안에 있고 지름길 규칙의 lift 는 정확히 0
    strata = payload["stratified"]
    k0 = str(strata["default_k"])
    rows = strata["by_k"][k0]
    assert set(rows) >= {*DET_TAGS, "uni_central", "uni_fed", "__shortcut__"}
    assert rows["__shortcut__"]["stratified_lift"] == 0.0
    assert rows["__shortcut__"]["stratified_lift_weighted"] == 0.0
    assert strata["axis"] == "idq"

    gates = {r["name"]: r for r in payload["gates_evaluated"]["results"]}
    assert gates["stratified_scoring"]["passed"] and not gates["stratified_scoring"]["skipped"]

    # D-8 파생 — prereg 게이트가 skipped 가 아니라 판정했고, 파일이 채점 디렉터리에 남았다
    assert (out / "prereg_recomputed_v1.json").exists()
    pr = gates["prereg_constants_reproduced"]
    assert pr["passed"] and not pr["skipped"]
    assert pr["value"]["measured"]["snapshot_digest"]

    # D-8 봉인 — 저장본이 최종 코드로 산출됐다는 4키
    for key in ("profile", "model_cfg", "predict_chunk", "imgsz_source"):
        assert key in payload["params"], key


# --------------------------------------------------------------------------------------
# 칸 선택 — 빠진 칸이 조용히 사라지지 않게 (17번 시드 1 첫 채점)
# --------------------------------------------------------------------------------------

def _args(cells: str):
    from types import SimpleNamespace

    return SimpleNamespace(cells=cells)


def test_cells_det_은_검출만_고르고_기본은_전_칸이다() -> None:
    from evaluation.cells import DET_TAGS, UNI_TAGS
    from scripts.probe.score_cells import selected_tags

    assert selected_tags(_args("det")) == (DET_TAGS, ())
    assert selected_tags(_args("all")) == (DET_TAGS, UNI_TAGS)
    # 인자가 아예 없는 호출부(구 스크립트)는 전 칸이다 — 조용히 줄어들지 않는다
    assert selected_tags(object()) == (DET_TAGS, UNI_TAGS)


def test_통합형이_없으면_회복률이_0_을_지어내지_않는다() -> None:
    """부재와 0 을 섞으면 유지율이 거짓 수를 낸다."""
    from scripts.probe.score_cells import recovery

    det = {"sep_central": {"macro_f1": 0.8}, "sep_fed": {"macro_f1": 0.7},
           "sep_local_C1": {"macro_f1": 0.6}, "sep_local_C2": {"macro_f1": 0.5},
           "sep_local_C3": {"macro_f1": 0.4}}
    r = recovery(det)
    assert r["unified"]["central"] is None and r["unified"]["retention_pct"] is None
    assert "부재" in r["unified"]["note"]
    # 분리형 축은 그대로 산출된다
    assert r["separated"]["local_mean"] == pytest.approx(0.5)
    assert r["separated"]["recovery_pct"] == pytest.approx((0.7 - 0.5) / (0.8 - 0.5) * 100)

    full = {**det, "uni_central": {"macro_f1": 0.6}, "uni_fed": {"macro_f1": 0.3}}
    r2 = recovery(full)
    assert r2["unified"]["retention_pct"] == pytest.approx(50.0)


# --------------------------------------------------------------------------------------
# 총괄 판정 1·2 이행 — 곡선·임계 독립 지표·규격 지름길 각주가 **산출물에 실재**하는가
# (22번 §1-2·§2-2-2. 실제 프로세스에서 본다 — 단위 시험은 블록을 손으로 만들어 줄 수 있다)
# --------------------------------------------------------------------------------------

def test_곡선과_임계_독립_지표가_정규_산출물에_있다(scoring_runs) -> None:
    from evaluation.params import ScoringParams

    _, out = scoring_runs["clean"]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    grid = [f"{t:.2f}" for t in ScoringParams(snapshot=".", pilot=".", out=".").conf_sweep]

    curve = payload["curve"]
    assert curve["missing_tags"] == []
    assert [f"{t:.2f}" for t in curve["grid"]] == grid
    for tag in DET_TAGS:
        pts = curve["by_tag"][tag]["by_threshold"]
        assert set(pts) == set(grid), f"{tag}: 격자 전 점이 있어야 한다"
        for k in grid:
            assert {"n_boxes", "macro_f1", "miss_rate"} <= set(pts[k])
    # 임계가 오르면 박스는 단조 감소한다 — 곡선이 실제로 임계를 반영했다는 표시
    n = [curve["by_tag"]["sep_fed"]["by_threshold"][k]["n_boxes"] for k in grid]
    assert n == sorted(n, reverse=True)

    indep = payload["threshold_independent"]
    assert indep["primary"] == "map_50"
    assert indep["recovery"]["map_50"]["recovery_pct"] is not None
    for tag in DET_TAGS:
        assert indep["per_tag"][tag]["macro_ap"] is not None
    # **"임계 독립"이 과장되지 않게 남는 의존을 산출물이 스스로 싣는다**(22번 과제 2)
    assert {"conf_floor", "nms_iou", "match_iou", "max_det"} <= set(indep["still_depends_on"])

    gates = {r["name"]: r for r in payload["gates_evaluated"]["results"]}
    for name in ("sweep_curve_recorded", "p9_source_separation"):
        assert gates[name]["passed"] and not gates[name]["skipped"], name


def test_규격_지름길_각주가_자동으로_달린다(scoring_runs) -> None:
    """전역 지표 표를 읽는 사람이 함정 #11·#12 를 모르고 지나가지 않게 한다(판정 2)."""
    _, out = scoring_runs["clean"]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    note = payload["shortcut_footnote"]
    assert note["applies_to"] == "metrics (전역 지표 표)"
    assert len(note["lines"]) == 2
    assert any("함정 #11" in ln for ln in note["lines"])
    assert any("함정 #12" in ln for ln in note["lines"])
    # 운용점 표가 스스로 "확증 기준 아님" 을 말한다
    assert "확증적 기준 아님" in payload["metrics_role"]
    assert "확증적 기준 아님" in payload["params"]["conf"]["role"]


# --------------------------------------------------------------------------------------
# 22번 §5 — 판별력 Δ 배선(과제 4) · 귀속 축 분해(과제 5)가 본채점 산출물에 실린다
# --------------------------------------------------------------------------------------

def test_판별력_델타가_본채점_산출물에_실린다(scoring_runs) -> None:
    """`evaluation/discrimination.py` 는 09-02 에 구현·시험까지 끝났는데 본채점 진입점이
    부르지 않아 값이 산출물에 실린 적이 없었다. 그 배선을 여기서 고정한다."""
    _, out = scoring_runs["clean"]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    d = payload["discrimination"]
    assert d["provenance"] == "N-crop"
    tags = {r["cell"] if not r.get("client") else f"{r['cell']}_{r['client']}"
            for r in d["results"]}
    # **칸 이름으로 분기하지 않는다**(불변조건 3-7) — 채점된 칸 전부가 같은 함수를 탄다.
    # 이 픽스처에는 통합형 두 칸도 있으므로 검출 5칸은 부분집합이지 전체가 아니다.
    assert set(DET_TAGS) <= tags, tags
    assert len(tags) == len(d["results"]), "칸이 중복으로 실렸다"
    for r in d["results"]:
        # Δ = 결함 발화율 − 정상 발화율. 셋이 서로 맞아야 값이 실제로 계산된 것이다
        assert r["delta"]["point"] == pytest.approx(
            r["fire_rate_defect"] - r["fire_rate_normal"], abs=1e-9)
        # `Interval.as_dict` 의 키는 `ci_lo`/`ci_hi` 다 — 이름이 갈리면 출력이 죽는다
        assert r["delta"]["ci_lo"] <= r["delta"]["point"] <= r["delta"]["ci_hi"]
        assert r["n_defect"] > 0 and r["n_normal"] > 0


def test_귀속_축_분해가_재질로_나오고_한계를_밝힌다(scoring_runs) -> None:
    """평가셋에 `client` 열이 없으므로 3분할 분해는 불가능하다 — 그 사실이 산출물에 적혀야
    분해표를 클라이언트 분해로 오독하지 않는다."""
    _, out = scoring_runs["clean"]
    payload = json.loads((out / "score_cells_v1.json").read_text(encoding="utf-8"))
    dec = payload["decomposition"]
    assert dec["axis"] == "material"
    assert "client" in dec["limitation"] and "C3" in dec["client_mapping"]["AL"]
    for g in dec["by_group"].values():
        assert g["n_images"] == g["n_defect"] + g["n_normal"]
        for tag, v in g["per_tag"].items():
            assert tag in DET_TAGS
            assert v["n_images"] == g["n_images"]
        if g["per_tag"]:
            assert set(g["recovery"]) == {"map_50", "macro_ap", "macro_f1", "miss_rate"}
