"""검사기 판 · 표 핀 · 닫힌 목록의 고정 조합(07 4판 §19-1 · 1판 §5-2, 인터페이스 문서 07e §6).

한 경로가 같다는 사실만으로 내용이 호환되지 않는다. 검사기 모듈의 상수(구문 트리로 읽는다 — 실행하지 않는다)가 고정표의 한 행이어야 하고,
닫힌 목록의 출처 표가 그 행의 표 핀이어야 하며, 판 6 등록의 (검사기 판, 표, 닫힌 목록)도 그 표의 한 행이어야 한다. 구판 재현용(v3.1 파일 ·
그 상수 · 닫힌 목록 v3-3)은 새 표로 바꾸지 않는다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from evaluation import narrative_checker as NC
from evaluation import narrative_checker_v4 as NC4
from evaluation import narrative_registration as NR
from tests.test_prereg_v6 import v6

REPO = Path(__file__).resolve().parents[1]
PIN = NR.CHECKER_PINS[0]
PIN_V4 = NR.CHECKER_PINS[1]


def copy_pinned(root: Path) -> None:
    for pin in NR.CHECKER_PINS:
        for rel in (pin.module, pin.lists_rel, pin.limits_rel):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, root / rel)


def test_지금_트리의_검사기_핀은_고정표의_한_행이다() -> None:
    assert NR.checker_pin_problems(REPO) == []
    assert [p.module for p in NR.CHECKER_PINS] == [NR.CHECKER_REL, "evaluation/narrative_checker_v4.py"]


def test_고정표의_v3_1_행은_검사기_모듈의_상수와_같다() -> None:
    assert (PIN.checker_id, PIN.limits_rel, PIN.limits_sha256, PIN.lists_rel, PIN.lists_sha256) == (
        NC.CHECKER_ID, NC.LIMITS_REL, NC.LIMITS_SHA256, NC.LISTS_REL, NC.LISTS_SHA256)


def test_고정표의_v4_행은_검사기_v4_모듈의_상수와_같다() -> None:
    """10 7판 §1 — v4 행을 더하고 v3.1 행은 그대로 둔다. v4 는 2 판 표 · 닫힌 목록 v4-lists-2 다."""
    assert (PIN_V4.checker_id, PIN_V4.limits_rel, PIN_V4.limits_sha256, PIN_V4.lists_rel, PIN_V4.lists_sha256) == (
        NC4.CHECKER_ID, NC4.LIMITS_REL, NC4.LIMITS_SHA256, NC4.LISTS_REL, NC4.LISTS_SHA256)
    assert PIN_V4.limits_rel != PIN.limits_rel and PIN_V4.lists_rel != PIN.lists_rel


def test_표_경로는_검사기의_표_핀_한_자리에서_읽는다() -> None:
    """07 1판 §5-2 — 신원 대조의 표 경로가 검사기의 핀과 따로 적힌 둘째 상수였다. 이제 같은 값을 가져다 쓴다."""
    assert NR.LIMITS_CSV_REL == NC.LIMITS_REL


def test_표_sha_만_바꾼_검사기는_멈춘다(tmp_path: Path) -> None:
    copy_pinned(tmp_path)
    module = tmp_path / PIN.module
    text = module.read_text(encoding="utf-8")
    assert text.count(PIN.limits_sha256) == 1
    module.write_text(text.replace(PIN.limits_sha256, "0" * 64), encoding="utf-8")
    got = NR.checker_pin_problems(tmp_path)
    assert got and all(p.startswith("pin_combo: ") for p in got)
    assert any("LIMITS_SHA256" in p for p in got)


def test_고정표에_없는_검사기_모듈은_멈춘다(tmp_path: Path) -> None:
    copy_pinned(tmp_path)
    (tmp_path / "evaluation" / "narrative_checker_v9.py").write_text('CHECKER_ID = "narrative_checker_v9"\n', encoding="utf-8")
    assert any("narrative_checker_v9.py" in p for p in NR.checker_pin_problems(tmp_path))


def test_닫힌_목록의_출처_표가_표_핀과_다르면_멈춘다(tmp_path: Path) -> None:
    copy_pinned(tmp_path)
    lists = tmp_path / PIN.lists_rel
    body = json.loads(lists.read_text(encoding="utf-8"))
    body["sources"]["limits_csv"]["sha256"] = "0" * 64
    lists.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    got = NR.checker_pin_problems(tmp_path)
    assert any("출처 표" in p for p in got) and any("바이트가 고정표와 다르다" in p for p in got)


def test_판_6_의_검사기_표_목록_짝이_고정표에_없으면_멈춘다() -> None:
    ok = replace(v6(), narrative=NR.narrative_block(REPO), clause_axis=NR.clause_axis_block(REPO))
    assert NR.registration_pin_problems(ok) == []
    bad = replace(ok, narrative=replace(ok.narrative, limits_sha256="0" * 64))
    assert [p.split(":", 1)[0] for p in NR.registration_pin_problems(bad)] == ["pin_combo"]


def test_M4_검사기_v3_1_에_다른_표_sha_를_적은_판_6_은_채점_진입점이_거부한다(tmp_path: Path) -> None:
    """07 1판 §5-3 M4 — 검사기 v3.1(1 판 표 핀)인데 등록의 표 sha 가 다른 표다. 실제 바이트 대조와 핀 조합 둘 다 멈춘다."""
    contract = tmp_path / "c.json"
    contract.write_bytes(b"{}\n")
    full = NR.narrative_block(REPO, confirmation_path=contract, checker_vectors_path=contract, inputs_contract_path=contract)
    reg = replace(v6(), narrative=replace(full, limits_sha256="1" * 64),
                  clause_axis=replace(NR.clause_axis_block(REPO), limits_sha256="1" * 64))
    got = NR.identity_problems(reg)
    assert any(p.startswith("narrative.limits_sha256") for p in got)
    assert any(p.startswith("pin_combo: ") for p in got)


# ════════════════════════════════════════════════════════════ 판 6 연결 2 판 — 프로필 조합(§3-2 · R8 · R9 의 공통 함수)

import pytest

from tests import v6_link_world as VW


def _train_entry(root: Path) -> dict:
    from vlm.registration import registration_from_bytes

    return registration_from_bytes((root / "configs/train_registration.yaml").read_bytes())


def test_v6_link_profile_tuple_mixing(tmp_path: Path) -> None:
    """R8 — v4 에 1 판 표 · v3-3 목록, 새 계약 결속에 v3.1, 결속 없는 v4, 모르는 행은 모두 `profile_mismatch`. 실제 검사기 적재 전이다."""
    VW.copy_pinned(tmp_path)
    entry = _train_entry(tmp_path)
    new = VW.regime2_registration(tmp_path)
    assert NR.profile_for(new, train_entry=entry).profile_id == NR.PROFILE_REGIME2
    old = replace(v6(), narrative=NR.narrative_block(tmp_path), clause_axis=NR.clause_axis_block(tmp_path))     # 구판 — 결속 없음
    assert NR.profile_for(old, train_entry=entry).profile_id == NR.PROFILE_LEGACY
    assert NR.profile_for(old, train_entry=None).profile_id == NR.PROFILE_LEGACY
    mixes = {
        "v4_with_v0_table": replace(new, narrative=replace(new.narrative, limits_sha256=PIN.limits_sha256)),
        "v4_with_v3_lists": replace(new, narrative=replace(new.narrative, lists_sha256=PIN.lists_sha256)),
        "new_contract_with_v3_1": replace(old, narrative=replace(old.narrative,
                                                                 inputs_contract_sha256=new.narrative.inputs_contract_sha256)),
        "v4_without_contract": replace(new, narrative=replace(new.narrative, inputs_contract_sha256="a" * 64)),
        "v4_contract_unbound": replace(new, narrative=replace(new.narrative, inputs_contract_sha256=None)),
        "unknown_row": replace(new, narrative=replace(new.narrative, checker_id="narrative_checker_v9", limits_sha256="0" * 64,
                                                      lists_sha256="1" * 64)),
    }
    for name, reg in mixes.items():
        with pytest.raises(NR.ProfileRefused) as exc:
            NR.profile_for(reg, train_entry=entry)
        assert exc.value.code == "profile_mismatch", name
    from tests.test_prereg_unified import filled

    with pytest.raises(NR.ProfileRefused):
        NR.profile_for(filled(), train_entry=entry)                     # 판 5 — 프로필이 없다
    with pytest.raises(NR.ProfileRefused):
        NR.resolve_profile("regime3_v5")
    with pytest.raises(NR.ProfileRefused):
        NR.resolve_profile(NR.NARRATIVE_PROFILES[NR.PROFILE_REGIME2]._replace(generator_rule="lf_single_v1"))


def test_v6_link_load_profile_checker_is_closed_and_runtime_checked(tmp_path: Path, monkeypatch) -> None:
    """닫힌 사상에서만 검사기를 고른다. 표 sha 만 바꾼 모듈은 적재 전 `pin_combo`, 불러온 모듈의 상수가 핀과 다른 대역은 `profile_runtime`."""
    for pid, modname in NR.CHECKER_MODULES.items():
        checker = NR.load_profile_checker(pid, repo=REPO)
        assert type(checker).__module__ == modname
    copy_pinned(tmp_path)
    module = tmp_path / PIN_V4.module
    text = module.read_text(encoding="utf-8")
    module.write_text(text.replace(PIN_V4.limits_sha256, "0" * 64), encoding="utf-8")
    with pytest.raises(NR.ProfileRefused) as exc:
        NR.load_profile_checker(NR.PROFILE_REGIME2, repo=tmp_path)
    assert exc.value.code == "pin_combo"
    monkeypatch.setattr(NC4, "LIMITS_SHA256", "0" * 64)
    with pytest.raises(NR.ProfileRefused) as exc2:
        NR.load_profile_checker(NR.PROFILE_REGIME2, repo=REPO)
    assert exc2.value.code == "profile_runtime"
