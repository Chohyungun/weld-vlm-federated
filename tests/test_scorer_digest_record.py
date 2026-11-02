"""채점 지문의 **당시 값 · 지금 값** 병기(35번 §1-5) — 기록이 커밋의 트리에서 나온 값인지 다시 계산해 맞댄다.

표본 채점 연결(외부 검토 137-2 I-05)로 `scripts/probe/score_unified.py`(`SCORER_FILES`)와 `evaluation/**`(`SCORER_TREES`)가 바뀌어 채점
지문(`scorer_code.combined`)이 바뀌었다. 과거 채점 산출 · 동결 영수증 · map_50 검산이 싣는 지문은 그때 코드의 기록이라 고치지 않는다 —
그 자리는 기록의 `unchanged_records` 가 경로로만 가리킨다(값을 옮겨 적지 않는다). 두 값은 사람이 옮긴 것이 아니라 커밋의 트리에서
다시 계산한 것이어야 한다 — 이 시험이 git 트리에서 지문을 다시 내어 본다.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

from evaluation import provenance
from evaluation.provenance import (
    SCORER_FILES,
    SCORER_FILES_AT_MAIN1,
    SCORER_TREES,
    scorer_code_digest,
)

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs/dev_log/2026-10-07-본실험코퍼스/35b_채점지문_병기_D.json"
RECORD_03J = REPO / "docs/dev_log/2026-10-20-설계복구/03j_채점지문_병기_D.json"
RECORD_03R = REPO / "docs/dev_log/2026-10-20-설계복구/03r_채점지문_병기_D.json"
"""같은 꼴 — main 에서 이 묶음(구현 · 검사기 v3 · 엄격 경로 · 진입점)의 마지막 코드 커밋까지."""
RECORD_03W = REPO / "docs/dev_log/2026-10-20-설계복구/03w_채점지문_병기_v3_1_D.json"
"""같은 꼴 — main 에서 검사기 v3.1 커밋까지."""
RECORD_03AB = REPO / "docs/dev_log/2026-10-20-설계복구/03ab_채점지문_병기_1023c_D.json"
"""같은 꼴 — main 에서 지시 1023c 의 코드 커밋까지. 병합 뒤 wt/D-narr tip 을 지금 목록으로 잰 `merge_base` 를 덧붙인다."""
RECORD_03AE = REPO / "docs/dev_log/2026-10-20-설계복구/03ae_채점지문_병기_1023d_D.json"
"""같은 꼴 — main 에서 지시 1023d 의 코드 커밋까지(병합 뒤 tip 은 1023c 가 병합된 main)."""
RECORDS = [RECORD_03J, RECORD_03R, RECORD_03W, RECORD_03AB, RECORD_03AE]
RECORD_IDS = ["03j", "03r", "03w", "03ab", "03ae"]
"""2026-10-22 — 허용치 규칙 모듈 넷을 트리 밖 목록에 더하고 서술 · 조항 축 · 등록 판 6 모듈을 더했다. 앞 커밋은 그때 목록
(`SCORER_FILES_AT_MAIN1`), 뒤 커밋은 지금 목록(`SCORER_FILES`)으로 잰다 — 기록의 `files_constant` 가 어느 쪽인지 적는다."""
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)


def digest_at(commit: str, tmp: Path, files: tuple[str, ...] = SCORER_FILES) -> tuple[str, int]:
    """커밋의 트리에서 `SCORER_TREES` + 트리 밖 목록 `files` 로 지문을 낸다 — 작업 트리를 읽지 않는다.

    기록은 **그 기록이 쓴 목록**으로 다시 잰다 — 35b 는 1차 본실험의 목록(`SCORER_FILES_AT_MAIN1`)이다. 목록이 넓어진 뒤(2026-10-22,
    허용치 규칙 모듈 넷) 지금 목록으로 옛 커밋을 재면 그 커밋에도 있는 새 파일까지 들어가 기록과 달라진다."""
    present = [p for p in (*SCORER_TREES, *files) if _git("cat-file", "-e", f"{commit}:{p}").returncode == 0]
    raw = _git("archive", "--format=tar", commit, "--", *present).stdout
    with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(tmp, filter="data")
        else:                                            # 필터가 없는 판 — 저장소가 만든 tar 다
            tf.extractall(tmp)  # noqa: S202
    d = scorer_code_digest(tmp, include_files=False, files=files)
    return d["combined"], d["n_files"]


def record() -> dict:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_기록은_당시_값과_지금_값을_함께_싣는다() -> None:
    rec = record()
    assert {"before", "after", "unchanged_records", "rule", "why"} <= set(rec)
    for side in ("before", "after"):
        assert _HEX40.match(rec[side]["commit"]) and _HEX64.match(rec[side]["combined"])
        assert type(rec[side]["n_files"]) is int and rec[side]["n_files"] > 20
    assert rec["before"]["combined"] != rec["after"]["combined"]
    assert rec["rule"] == "scorer_code_digest(git archive <commit> -- SCORER_TREES SCORER_FILES)['combined']"
    assert all(isinstance(p, str) and not re.search(r"[0-9a-f]{64}", p) for p in rec["unchanged_records"])  # 경로만


@pytest.mark.parametrize("side", ["before", "after"])
def test_기록의_값은_그_커밋의_트리에서_다시_계산한_값이다(side: str, tmp_path: Path) -> None:
    rec = record()[side]
    if _git("cat-file", "-e", f"{rec['commit']}^{{commit}}").returncode != 0:
        pytest.skip(f"이 체크아웃에 커밋 {rec['commit'][:12]} 이 없다")
    assert digest_at(rec["commit"], tmp_path, SCORER_FILES_AT_MAIN1) == (rec["combined"], rec["n_files"])


@pytest.mark.parametrize("path", RECORDS, ids=RECORD_IDS)
def test_03j_기록은_칸마다_목록을_밝히고_두_값을_싣는다(path: Path) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert {"before", "after", "unchanged_records", "rule", "rule_strings", "why"} <= set(rec)
    assert (rec["before"]["files_constant"], rec["after"]["files_constant"]) == ("SCORER_FILES_AT_MAIN1", "SCORER_FILES")
    for side in ("before", "after"):
        assert _HEX40.match(rec[side]["commit"]) and _HEX64.match(rec[side]["combined"])
        assert type(rec[side]["n_files"]) is int and rec[side]["n_files"] > 20
    assert rec["before"]["combined"] != rec["after"]["combined"]
    assert rec["rule_strings"]["SCORER_FILES_AT_MAIN1"] == provenance._rule_text(SCORER_FILES_AT_MAIN1)
    assert all(f in rec["rule_strings"]["SCORER_FILES"] for f in provenance.LIMITS_RULE_FILES)
    assert all(isinstance(p, str) and not re.search(r"[0-9a-f]{64}", p) for p in rec["unchanged_records"])  # 경로만


@pytest.mark.parametrize("path", RECORDS, ids=RECORD_IDS)
@pytest.mark.parametrize("side", ["before", "after"])
def test_03j_의_값은_그_커밋의_트리에서_기록의_목록으로_다시_계산한_값이다(side: str, path: Path, tmp_path: Path) -> None:
    rec = json.loads(path.read_text(encoding="utf-8"))[side]
    if _git("cat-file", "-e", f"{rec['commit']}^{{commit}}").returncode != 0:
        pytest.skip(f"이 체크아웃에 커밋 {rec['commit'][:12]} 이 없다")
    files = getattr(provenance, rec["files_constant"])
    assert digest_at(rec["commit"], tmp_path, files) == (rec["combined"], rec["n_files"])


@pytest.mark.parametrize("path,prev", [(RECORD_03AB, RECORD_03W), (RECORD_03AE, RECORD_03AB)], ids=["03ab", "03ae"])
def test_병합_뒤_tip_은_지금_목록으로_다시_계산한_값이고_앞_기록의_뒤와_같다(path: Path, prev: Path, tmp_path: Path) -> None:
    """병합 뒤 tip(`merge_base`)은 앞 기록의 뒤와 채점 코드가 같다 — 03ab 는 03w 의 뒤(검사기 v3.1 커밋), 03ae 는 03ab 의 뒤(1023c
    코드 커밋, main 에 병합됨). main 병합(C 의 학습 쪽 파일 · 문서)은 지문에 들지 않는다."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    base = rec["merge_base"]
    assert base["files_constant"] == "SCORER_FILES" and _HEX40.match(base["commit"])
    assert len({rec["before"]["combined"], base["combined"], rec["after"]["combined"]}) == 3
    assert base["combined"] == json.loads(prev.read_text(encoding="utf-8"))["after"]["combined"]
    if _git("cat-file", "-e", f"{base['commit']}^{{commit}}").returncode != 0:
        pytest.skip(f"이 체크아웃에 커밋 {base['commit'][:12]} 이 없다")
    assert digest_at(base["commit"], tmp_path, SCORER_FILES) == (base["combined"], base["n_files"])
