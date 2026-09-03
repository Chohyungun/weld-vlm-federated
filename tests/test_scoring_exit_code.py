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
