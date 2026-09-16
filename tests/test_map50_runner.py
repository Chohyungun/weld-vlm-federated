"""`map_50` 독립 검산 실행기 — 판정·입력 대조·코드 대조·출력 보호 (D 53번 · 총괄 57번).

지키는 것:
1. **판정은 전부 허용 안이어야 통과다**(53번 §4) — `map_50`·카테고리 평균 판·카테고리별 AP·R_s·R̄. 허용 차 1e−12.
2. **채택 조건을 닫는 것은 I 모드다**(57번). S 는 진단이고, 둘이 갈리면 그 사실을 적는다.
3. 입력 해시가 v3 기록과 다르거나 없으면 **계산하지 않는다.**
4. 채점 코드는 합산 지문이 아니라 **파일별 해시**로 v3 와 맞댄다. S 경로 파일이 바뀌면 드러난다.
5. 출력은 판 번호로 짓고, 이미 있으면 **계산 전에** 멈춘다.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.recovery_ci import POINT_TOLERANCE, recovery_from
from scripts.probe import map50_independent as run

REPO = Path(__file__).resolve().parents[1]
TAGS = run.DET_TAGS
BASE = {"sep_central": 0.4926, "sep_local_C1": 0.3584, "sep_local_C2": 0.1358,
        "sep_local_C3": 0.0932, "sep_fed": 0.2820}
CLASSES = ("100", "2011", "401", "301")


def _v3() -> dict[int, dict[str, float]]:
    return {n: {t: v * s for t, v in BASE.items()} for n, s in ((1, 1.0), (2, 0.98), (3, 1.01))}


def _ap(n: int, t: str) -> dict[str, float]:
    return {c: 0.1 * (k + 1) + 0.001 * n + 0.0001 * TAGS.index(t) for k, c in enumerate(CLASSES)}


def _mode(v3, **bump) -> dict:
    """모드 결과 모양. `bump` 로 한 곳씩 흔든다: map=(n, t, d) · cm=(n, t, d) · ap=(n, t, c, d)."""
    out = {n: {t: {"map_50": v, "map_50_class_mean": v, "ap_by_class": dict(_ap(n, t))} for t, v in m.items()}
           for n, m in v3.items()}
    if "map" in bump:
        n, t, d = bump["map"]
        out[n][t]["map_50"] += d
    if "cm" in bump:
        n, t, d = bump["cm"]
        out[n][t]["map_50_class_mean"] += d
    if "ap" in bump:
        n, t, c, d = bump["ap"]
        out[n][t]["ap_by_class"][c] = None if d is None else out[n][t]["ap_by_class"][c] + d
    return out


def _ref(v3) -> dict:
    return {n: {t: dict(_ap(n, t)) for t in m} for n, m in v3.items()}


# ---- 1. 판정 ----------------------------------------------------------------------------

def test_허용_차는_POINT_TOLERANCE_와_같은_1e_12():
    assert run.TOLERANCE == POINT_TOLERANCE == 1e-12


def test_같은_값이면_통과하고_비트_일치를_센다():
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3), _ref(v3))
    assert j["pass"] is True
    assert j["n_cells"] == 15 and j["n_bit_equal_map_50"] == 15
    assert j["first_mismatch"] is None
    assert j["worst_abs_diff"] == {"map_50": 0.0, "map_50_class_mean": 0.0, "ap_by_class": 0.0, "R": 0.0, "R_bar": 0.0}


@pytest.mark.parametrize("delta,ok", [(5e-17, True), (5e-13, True), (5e-12, False), (1e-6, False)])
def test_map_50_허용_경계(delta, ok):
    """중앙 칸 — 회복률에 옮겨가는 배율이 1 보다 작아(∂R/∂c = −R/D ≈ −0.8) map_50 경계가 그대로 판정 경계다."""
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, map=(2, "sep_central", delta)), _ref(v3))
    assert j["pass"] is ok, j["worst_abs_diff"]
    assert j["n_bit_equal_map_50"] == 14                  # 허용 안이어도 비트 일치는 아니다
    if not ok:
        assert "map_50" in j["failed_quantities"]
        assert (j["first_mismatch"]["seed"], j["first_mismatch"]["tag"]) == (2, "sep_central")


def test_연합_칸의_차는_1_over_D_배로_회복률에_옮겨가_더_엄격하다():
    """연합 칸 map_50 차 5e−13 은 map_50 기준 안이지만 R_s 차는 5e−13/D ≈ 1.7e−12 로 R 기준(1e−12)을 넘는다.
    판정은 둘 다 보므로 연합 칸의 실질 허용 차는 약 D·1e−12 ≈ 3e−13 이다(53번 §4 를 그대로 적용한 결과)."""
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, map=(2, "sep_fed", 5e-13)), _ref(v3))
    assert j["worst_abs_diff"]["map_50"] <= run.TOLERANCE
    assert j["failed_quantities"] == ["R"] or set(j["failed_quantities"]) == {"R", "R_bar"}
    assert j["pass"] is False
    ok = run.judge_mode(v3, _mode(v3, map=(2, "sep_fed", 2e-13)), _ref(v3))
    assert ok["pass"] is True


def test_연합_칸이_흔들리면_회복률도_판정에_걸린다():
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, map=(1, "sep_fed", 1e-6)), _ref(v3))
    assert {"map_50", "R", "R_bar"} <= set(j["failed_quantities"])
    d = j["recovery_mode"]["R_by_seed"]["1"] - j["recovery_v3"]["R_by_seed"]["1"]
    assert d == pytest.approx(1e-6 / j["recovery_mode"]["D_by_seed"]["1"], rel=1e-6)


def test_카테고리_평균_판이_다르면_실패():
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, cm=(3, "sep_local_C2", 1e-9)), _ref(v3))
    assert j["pass"] is False and j["failed_quantities"] == ["map_50_class_mean"]


@pytest.mark.parametrize("delta", [1e-9, None])
def test_카테고리별_AP_가_재계산과_다르면_실패(delta):
    """None 은 한쪽만 '정답 없는 카테고리' 로 본 경우다 — 무한대 차로 센다."""
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, ap=(1, "sep_central", "401", delta)), _ref(v3))
    assert j["pass"] is False and j["failed_quantities"] == ["ap_by_class"]
    if delta is None:
        assert math.isinf(j["worst_abs_diff"]["ap_by_class"])


def test_재계산_참조가_없으면_카테고리별_AP_는_보지_않는다():
    v3 = _v3()
    j = run.judge_mode(v3, _mode(v3, ap=(1, "sep_central", "401", 1.0)), None)
    assert j["pass"] is True


def test_회복률은_생성기와_같은_계산이다():
    v3 = _v3()
    r = run.recoveries(v3)
    for n, m in v3.items():
        assert (r["R"][n], r["D"][n]) == recovery_from(m["sep_central"], m["sep_fed"],
                                                       [m[t] for t in run.LOCALS])
    assert r["R_bar"] == float(np.mean([r["R"][n] for n in sorted(v3)]))
    bad = _v3()
    bad[2]["sep_central"] = 0.10                        # 분모 ≤ 0 → 미정의
    rb = run.recoveries(bad)
    assert math.isnan(rb["R"][2]) and math.isnan(rb["R_bar"])
    j = run.judge_mode(bad, _mode(bad), _ref(bad))      # 양쪽 다 미정의면 같은 것으로 본다
    assert j["pass"] is True


# ---- 2. 채택 조건 ------------------------------------------------------------------------

@pytest.mark.parametrize("s,i,closed,noted", [
    (True, True, True, False),
    (False, True, True, True),       # S 만 불일치 — I 가 닫되, 갈린 사실을 적는다
    (True, False, False, True),
    (False, False, False, False),
    (True, None, False, False),      # I 를 돌리지 않으면 닫히지 않는다
    (None, True, True, False),
])
def test_채택_조건은_I_모드만_닫는다(s, i, closed, noted):
    v = run.overall_verdict(None if s is None else {"pass": s}, None if i is None else {"pass": i})
    assert v["adoption_condition_closed_by_I"] is closed
    assert (v["note"] is not None) is noted
    assert "총괄" in v["decision"]


# ---- 3. 입력 · 코드 대조 ------------------------------------------------------------------

def test_칸과_참여자를_태그에서_가른다():
    assert run.tag_identity("sep_local_C1") == ("sep_local", "C1")
    assert run.tag_identity("sep_central") == ("sep_central", None)
    assert run.tag_identity("sep_fed") == ("sep_fed", None)


def _records(seed_value: int) -> dict[str, str]:
    out = {}
    for t in TAGS:
        out[f"outputs/main_d/seed1/{t}_s{seed_value}.jsonl"] = "a" * 64
        out[f"outputs/main_d/seed1/sweep/{t}_raw_s{seed_value}.jsonl"] = "b" * 64
    return out


def test_하한_레코드만_고른다():
    keys = run.raw_record_keys(_records(20260828), 20260828)
    assert list(keys) == list(TAGS)
    assert all("/sweep/" in k and "_raw_s20260828" in k for k in keys.values())
    with pytest.raises(SystemExit, match="하나로 정할 수 없다"):
        run.raw_record_keys(_records(20260828), 20260829)       # 다른 시드 값
    recs = _records(20260828)
    recs["outputs/other/sweep/sep_fed_raw_s20260828.jsonl"] = "c" * 64
    with pytest.raises(SystemExit, match="sep_fed"):
        run.raw_record_keys(recs, 20260828)                    # 둘이면 고르지 않는다


def test_입력_해시가_다르거나_없으면_계산하지_않는다(tmp_path):
    from evaluation.provenance import hash_files

    f1 = tmp_path / "outputs" / "a.jsonl"
    f2 = tmp_path / "outputs" / "sweep" / "b.jsonl"
    for f, body in ((f1, b"{\"x\": 1}\n"), (f2, b"{\"y\": 2}\n")):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(body)
    recorded = hash_files([f1, f2], tmp_path)
    assert run.check_input_hashes({1: recorded}, tmp_path) == {"n_checked": 2, "all_equal": True}

    f2.write_bytes(b"{\"y\": 3}\n")                              # 한 바이트
    with pytest.raises(SystemExit, match="해시가 v3 기록과 다르다"):
        run.check_input_hashes({1: recorded}, tmp_path)
    f2.write_bytes(b"{\"y\": 2}\n")
    with pytest.raises(SystemExit, match="계산하지 않는다"):
        run.check_input_hashes({1: {**recorded, "outputs/none.jsonl": "d" * 64}}, tmp_path)   # 없는 파일
    with pytest.raises(SystemExit):
        run.check_input_hashes({1: {k: None for k in recorded}}, tmp_path)                    # 기록 없음


def test_코드는_파일별로_맞대고_S_경로_변경을_드러낸다():
    v3 = {f: f"h-{f}" for f in (*run.S_PATH_FILES, "evaluation/recovery_ci.py", "scripts/probe/score_cells.py")}
    now = dict(v3)
    now["evaluation/recovery_ci.py"] = "changed"                # S 경로 밖
    now[run.EXPECTED_ADDED[0]] = "new"
    d = run.code_drift(v3, now)
    assert d["s_path_unchanged"] is True and d["added_as_expected"] is True
    assert d["changed"] == ["evaluation/recovery_ci.py"]

    now["evaluation/metrics/localization.py"] = "changed"       # S 경로 안
    d = run.code_drift(v3, now)
    assert d["s_path_unchanged"] is False
    assert d["s_path_changed"] == ["evaluation/metrics/localization.py"]

    del now["evaluation/cells.py"]                              # S 경로 파일이 사라짐
    now["evaluation/extra.py"] = "x"
    d = run.code_drift(v3, now)
    assert "evaluation/cells.py" in d["s_path_changed"]
    assert d["added_as_expected"] is False


# ---- 4. 출력 보호 ------------------------------------------------------------------------

def _run_main(*extra):
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONIOENCODING": "utf-8"}
    return subprocess.run([sys.executable, str(REPO / "scripts/probe/map50_independent.py"), *extra],
                          capture_output=True, text=True, encoding="utf-8", env=env, cwd=REPO, check=False)


def test_같은_판_출력이_있으면_계산_전에_멈춘다(tmp_path):
    dest = tmp_path / "seed3set"
    dest.mkdir()
    existing = dest / "map50_independent_v3.json"
    existing.write_bytes(b"old\n")
    proc = _run_main("--root", str(tmp_path / "no_root"), "--dest", str(dest), "--artifact", "score_cells_v3.json")
    assert proc.returncode != 0
    assert "map50_independent_v3.json 이 이미 있다" in proc.stderr, proc.stderr[-600:]
    assert "Traceback" not in proc.stderr
    assert existing.read_bytes() == b"old\n"
    assert sorted(p.name for p in dest.iterdir()) == [existing.name]


@pytest.mark.parametrize("extra,msg", [
    (("--artifact", "score_cells.json"), "판 번호"),
    (("--modes", "X"), "--modes"),
])
def test_잘못된_인자는_읽기_전에_거부(tmp_path, extra, msg):
    proc = _run_main("--root", str(tmp_path / "no_root"), "--dest", str(tmp_path / "d"), *extra)
    assert proc.returncode != 0
    assert msg in proc.stderr, proc.stderr[-600:]
    assert "Traceback" not in proc.stderr
    assert not (tmp_path / "d").exists()
