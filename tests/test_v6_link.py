"""판 6 연결 2 판 — 묶음 2 의 거부 · 보존 시험(§8 R6 · R17 · R18 · R21 과 추가 접점 echo_bytes). 모델 · GPU 없이.

세계는 `tests.test_regime2_end_to_end.e2e`(2 차 체제의 같은 임시 세계 사슬)다 — 여섯 파일 · 커밋 후 검수 · 영수증 · export · 보조 입력이
실제 함수로 선 뒤, 칸 하나씩 바꿔 어느 소비자가 어느 사유 코드로 멈추는지 본다. 커밋 소비 시험은 임시 저장소의 실제 blob 을 읽는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.probe.score_unified as SU
from evaluation import eval_registration as ER
from evaluation import narrative_registration as NR
from evaluation import prereg_unified as P
from evaluation.entry_gate import EntryRejected, load_registration_for
from evaluation.prereg_unified import dump_registration, read_receipt
from tests import uni_export_world as W
from tests.test_regime2_end_to_end import (
    REG_REL,
    STEM,
    _commit,
    _export,
    _loaded,
    _no_cuda,
    _register,
    approved,
    e2e,
)
from vlm import export_preflight as XP
from vlm import main_registration as MR

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = (e2e, approved, _no_cuda)
"""가져온 fixture — pytest 는 모듈 이름 공간의 fixture 를 이름으로 모은다."""
BEFORE_LINK = "1ee8dcf2"
"""판 6 연결 묶음 1 앞의 본줄기 커밋 — 구판(판 5 · 역사적 판 6)의 바이트 · 사유 코드를 그 모듈로 다시 내 맞댄다(R21)."""


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _receipt_for(e, reg, head: str) -> Path:
    """손으로 쓴 영수증 — 커밋 후 검수를 지나지 못하는 anchor 를 가리키는 불일치 시험용."""
    raw = dump_registration(reg)
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "main",
               "registration_path": REG_REL, "registration_file_sha256": _sha(raw)}
    rp = e.w.repo / "main_in" / f"{STEM}-{head[:8]}.receipt.json"
    rp.write_text(json.dumps(receipt), encoding="utf-8")
    return rp


def _strict_population_code(e, rp: Path) -> str:
    """엄격 채점 진입점의 단계 1(모집단 모드) — 결정적 평가 설정 판독까지만. 거부면 `[코드]` 가 든 EntryRejected 의 문장."""
    rc = read_receipt(rp)
    args = argparse.Namespace(side=None, baseline=None, ecc_permit=None, frame_admission=None, ecc_waiver=None,
                              out=e.w.repo / "outputs" / "main_u")
    ctx = SU.Ctx(args=args, checkout=e.w.repo, repo=e.w.repo, receipt=rc, reg=load_registration_for(rc, repo=e.w.repo))
    with pytest.raises(EntryRejected) as exc:
        SU._strict_population(ctx)
    assert exc.value.code == "EVAL_REGISTRATION"
    return exc.value.reason


def test_v6_link_anchor_bytes_not_worktree(e2e, tmp_path):
    """R6 — 경로는 같고 anchor 의 평가 설정 바이트만 출처가 묶은 sha 와 다르다(작업 트리는 정상) → 네 소비자(커밋 후 검수 · 영수증 ·
    export · 엄격 채점)가 모두 `eval_bytes`. 역방향(anchor 정상 · 작업 트리 변조)은 양성 — 같은 anchor SHA 를 그대로 쓴다."""
    e = e2e
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    good = W.git(e.w.repo, "rev-parse", "HEAD")
    sidecar = e.w.repo / ER.eval_sidecar_rel(REG_REL)
    sidecar.write_bytes(e.cand_raw + b"# comment\n")                     # 작업 트리만 변조 — anchor 는 그대로
    assert MR.validate_committed_main_registration(registration_rel=REG_REL, commit=good, checkout=e.w.repo).sha256 == _sha(e.cand_raw)
    assert _export(e, rp, tmp_path / "ok") is None
    W.git(e.w.repo, "add", ER.eval_sidecar_rel(REG_REL))
    W.git(e.w.repo, "commit", "-q", "-m", "곁 파일 바이트만 다른 anchor")
    bad = W.git(e.w.repo, "rev-parse", "HEAD")
    rp_bad = _receipt_for(e, reg, bad)
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        MR.validate_committed_main_registration(registration_rel=REG_REL, commit=bad, checkout=e.w.repo)
    assert exc.value.code == "eval_bytes"
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        MR.write_main_receipt(registration_rel=REG_REL, commit=bad, checkout=e.w.repo)
    assert exc2.value.code == "eval_bytes"
    calls = dict(e.s.calls)
    assert _export(e, rp_bad, tmp_path / "bad") == "eval_bytes" and dict(e.s.calls) == calls
    assert "[eval_bytes]" in _strict_population_code(e, rp_bad)


def test_v6_link_export_contract_called_before_writer(e2e, tmp_path, monkeypatch):
    """R17 — 계약 변조 입력으로 실제 run_export 경로: `code_bytes:`(계약 실물) · `pin_contract:`(착수 고정의 계약 결속) ·
    `v6_contract_sha:`(판 6 의 계약 sha) 가 export 거부에 보존되고 생성 · writer.begin · 새 행이 0 이다. 잘못된 HEAD reader 면 우연히
    통과하는 anchor 구성(HEAD 의 계약이 바뀜)은 양성이고, 호출을 떼면 시험이 실패한다."""
    e = e2e
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    folder = tmp_path / "exp"                                          # tmp_path/out 은 시드 폴더(착수 고정)가 쓴다
    gen0 = e.w.repo / NR.GENERATOR_FILES_V1[0]
    body = gen0.read_bytes()
    gen0.write_bytes(body + b"\n# x\n")                                   # 계약 실물(생성 경로 한 파일) 변조
    assert _export(e, rp, folder) == "code_bytes" and not _loaded(e) and not folder.exists()
    gen0.write_bytes(body)
    pin = json.loads(e.pin_raw.decode("utf-8"))
    pin["common"]["registration"]["narrative_inputs"]["sha256"] = "0" * 64
    other_pin = json.dumps(pin, sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
    (e.train_root / "PIN.json").write_bytes(other_pin)                    # 착수 고정의 계약 결속이 다르다 — meta 는 그 PIN 의 sha
    e.write_metas(e.tc, start_gate={**e.gate, "pin_sha256": _sha(other_pin)})
    assert _export(e, rp, folder) == "pin_contract" and not _loaded(e) and not folder.exists()
    (e.train_root / "PIN.json").write_bytes(e.pin_raw)
    e.write_metas(e.tc)
    reg_sha = replace(reg, narrative=replace(reg.narrative, inputs_contract_sha256="1" * 64))
    raw_sha = dump_registration(reg_sha)
    (e.w.repo / REG_REL).write_bytes(raw_sha)
    W.git(e.w.repo, "add", REG_REL)
    W.git(e.w.repo, "commit", "-q", "-m", "판 6 의 계약 sha 가 다른 등록")
    head_sha = W.git(e.w.repo, "rev-parse", "HEAD")
    rp_sha = _receipt_for(e, reg_sha, head_sha)
    # 등록의 계약 sha 만 다른 판 6 — 공통 판독(프로필 결속)이 생성기 · writer 전에 멈추고(profile_mismatch), export 가 부르는 계약 ↔ PIN
    # 대조 함수는 그 자리의 사유 `v6_contract_sha:` 를 낸다(R15)
    assert _export(e, rp_sha, folder) == "profile_mismatch" and not _loaded(e) and not folder.exists()
    from evaluation.entry_gate import blob_at
    from vlm.narrative_inputs import v6_problems

    meta = json.loads((e.train_root / "uni_local_C1_s1" / "adapter_last.meta.json").read_text(encoding="utf-8"))
    lines = v6_problems(reg_sha, read_at_receipt=lambda p: blob_at(e.w.repo, head_sha, p), adapter_dir=e.train_root / "uni_local_C1_s1",
                        adapter_meta=meta, repo=e.w.repo)
    assert lines and lines[0].startswith("narrative_inputs: v6_contract_sha:")
    W.git(e.w.repo, "checkout", "-q", "--", REG_REL)
    # HEAD 의 계약을 바꿔도 export 는 영수증 anchor 의 blob 을 읽는다 — 양성
    (e.w.repo / NR.INPUTS_CONTRACT_REL).write_bytes(b'{"changed": true}\n')
    W.git(e.w.repo, "add", NR.INPUTS_CONTRACT_REL)
    W.git(e.w.repo, "commit", "-q", "-m", "HEAD 의 계약이 바뀜")
    W.git(e.w.repo, "checkout", "-q", "HEAD~1", "--", NR.INPUTS_CONTRACT_REL)   # 작업 트리의 실물은 anchor 와 같게
    assert _export(e, rp, folder) is None and e.s.calls["loader"] == 1
    # 호출을 떼면 변조된 계약 실물이 export 를 지난다 — 이 검사가 멈추게 하는 것임을 보인다
    gen0.write_bytes(body + b"\n# x\n")
    monkeypatch.setattr(XP, "_v6_regime_checks", lambda *a, **k: None)
    assert _export(e, rp, tmp_path / "bypass") is None


def test_v6_link_scoring_change_not_generation_only(e2e):
    """R18 — 서술 블록만 바꾼 blob 을 옛 영수증과 묶으면 생성 지문은 같아도 기존 `REGISTRATION_FILE_HASH` 로 거부한다. raw 대조를
    지난 합성 변형은 채점 지문 대조도 실패한다 — 생성 지문이 같음을 성공 근거로 쓰지 않는다."""
    e = e2e
    reg, raw, tc_text = _register(e)
    rp = _commit(e, reg, raw, tc_text)
    h = NR.OPEN_HOLES_V4
    altered = replace(reg, narrative=replace(reg.narrative, open_holes=(*h[1:], h[0])))
    assert altered.generation_sha256() == reg.generation_sha256() and altered.scoring_sha256() != reg.scoring_sha256()
    (e.w.repo / REG_REL).write_bytes(dump_registration(altered))
    W.git(e.w.repo, "add", REG_REL)
    W.git(e.w.repo, "commit", "-q", "-m", "서술 블록만 다른 등록")
    rc = read_receipt(rp)
    forged = SimpleNamespace(**{**rc.__dict__, "main_commit": W.git(e.w.repo, "rev-parse", "HEAD")})
    with pytest.raises(EntryRejected) as exc:
        load_registration_for(forged, repo=e.w.repo)
    assert exc.value.code == "REGISTRATION_FILE_HASH"
    assert _sha(dump_registration(altered)) != rc.registration_file_sha256


def test_v6_link_echo_bytes_preserved(e2e):
    """D6 — 2 차 등록의 에코 목록은 1 차 등록 커밋의 원본 raw 그대로다. 짓기에서 다른 raw 는 `echo_bytes`, 커밋 후 검수도 raw 를 본다."""
    from evaluation.eval_list import canonical_bytes

    e = e2e
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _register(e, echo_raw=canonical_bytes(list(W.IDS[:3])))
    assert exc.value.code == "echo_bytes"
    reg, raw, tc_text = _register(e)
    _commit(e, reg, raw, tc_text)
    head = W.git(e.w.repo, "rev-parse", "HEAD")
    assert MR.validate_committed_main_registration(registration_rel=REG_REL, commit=head, checkout=e.w.repo).profile_id == "regime2_v4"
    MR.ECHO_LIST_RAW_SHA256_V1 = "0" * 64                                  # 원본 raw 지문이 다른 것으로 — 커밋한 에코는 그 원본이 아니다
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        MR.validate_committed_main_registration(registration_rel=REG_REL, commit=head, checkout=e.w.repo)
    assert exc2.value.code == "echo_bytes"


def _old(tmp: Path, name: str):
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BEFORE_LINK}:evaluation/narrative_registration.py"],
                         capture_output=True, check=False)
    if src.returncode != 0:
        pytest.skip(f"이 체크아웃에 {BEFORE_LINK} 가 없다")
    p = tmp / f"{name}.py"
    p.write_bytes(src.stdout)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_v6_link_legacy_golden_unchanged(tmp_path):
    """R21 — 보존된 구판(판 5 · 역사적 판 6)의 바이트 · 지문 · 거부 코드는 판 6 연결 앞의 모듈이 낸 것과 같다. 새 런타임 거부 코드를
    구판에 덧씌우지 않고, 판 5 길은 보조 import 0 이다."""
    old = _old(tmp_path, "narrative_registration_before_link")
    assert old.actual_identity(ROOT) == NR.actual_identity(ROOT) == NR.actual_identity(ROOT, profile=NR.PROFILE_LEGACY)
    inputs = tmp_path / "v6_inputs"
    files = {k: inputs / n for k, n in W.V6_SYNTHETIC_FILES.items()}
    inputs.mkdir()
    for k, p in files.items():
        p.write_bytes(W.synthetic_contract(NR) if k == "inputs_contract_path" else json.dumps({"synthetic": k}).encode("utf-8"))
    assert old.narrative_block(ROOT, **files) == NR.narrative_block(ROOT, **files)
    assert old.clause_axis_block(ROOT) == NR.clause_axis_block(ROOT)
    r = W.registration(SimpleNamespace(digest="d" * 64, elist=SimpleNamespace(file_sha256="f" * 64, set_sha256="s" * 64),
                                       plan=files["inputs_contract_path"]))
    W.to_v6(r, inputs_dir=inputs)
    raw_new = dump_registration(r)
    old_r = replace(r, narrative=old.narrative_block(ROOT, **files), clause_axis=old.clause_axis_block(ROOT))
    assert dump_registration(old_r) == raw_new and old_r.scoring_sha256() == r.scoring_sha256()
    for bad in (r, replace(r, narrative=replace(r.narrative, limits_sha256="1" * 64)),
                replace(r, narrative=replace(r.narrative, open_holes=NR.OPEN_HOLES[:-1])),
                replace(r, narrative=replace(r.narrative, generator_sha256="2" * 64))):
        assert old.identity_problems(bad, repo=ROOT) == NR.identity_problems(bad, repo=ROOT)
    contract = files["inputs_contract_path"].read_bytes()
    pin_raw = json.dumps({"common": {"registration": {"narrative_inputs": {"version": 1, "sha256": _sha(contract)}}}}).encode("utf-8")
    for adapters in ([_sha(pin_raw)] * 5, [], ["0" * 64] * 5):
        assert old.v6_contract_problems(r, contract, pin_raw=pin_raw, adapter_pin_sha256=adapters) == \
            NR.v6_contract_problems(r, contract, pin_raw=pin_raw, adapter_pin_sha256=adapters)
    from tests.test_prereg_unified import filled

    assert old.identity_problems(filled()) == NR.identity_problems(filled())


def test_v6_link_committed_validation_is_blob_not_memory(e2e):
    """커밋 후 검수는 메모리 객체가 아니라 blob 을 읽는다 — 등록 · 원문 · 목록 · 곁 파일 하나씩 바꿔 커밋하면 그 자리의 코드로 멈춘다."""
    e = e2e
    reg, raw, tc_text = _register(e)
    _commit(e, reg, raw, tc_text)
    base = W.git(e.w.repo, "rev-parse", "HEAD")
    d = e.w.repo / "configs" / "registration"
    cases = {
        f"{STEM}.train_config.json": (tc_text + " ").encode("utf-8"), f"{STEM}.generation.list": e.echo_raw,
        f"{STEM}.v6_sources.json": b"{}\n",
    }
    want = {f"{STEM}.train_config.json": "train_config_vs_registration", f"{STEM}.generation.list": "list_vs_registration",
            f"{STEM}.v6_sources.json": "v6_sources_mismatch"}
    for name, body in cases.items():
        W.git(e.w.repo, "checkout", "-q", base)
        (d / name).write_bytes(body)
        W.git(e.w.repo, "add", f"configs/registration/{name}")
        W.git(e.w.repo, "commit", "-q", "-m", name)
        head = W.git(e.w.repo, "rev-parse", "HEAD")
        with pytest.raises(MR.MainRegistrationRefused) as exc:
            MR.validate_committed_main_registration(registration_rel=REG_REL, commit=head, checkout=e.w.repo)
        assert exc.value.code == want[name], name
    W.git(e.w.repo, "checkout", "-q", "main")
    (d / f"{STEM}.v6_sources.json").unlink()
    W.git(e.w.repo, "commit", "-q", "-am", "출처 파일 없음")
    with pytest.raises(MR.MainRegistrationRefused) as exc2:
        MR.validate_committed_main_registration(registration_rel=REG_REL, commit="HEAD", checkout=e.w.repo)
    assert exc2.value.code == "v6_sources_missing"


def test_v6_link_v6_code_and_completion_units():
    """export 거부 코드 추출(`_v6_code`)과 완료 표지(`v6_completion`)의 단위 거동 — 이름만 채운 wanted 는 완료가 아니다."""
    assert XP._v6_code("narrative_inputs: pin_contract: 설명") == "pin_contract"
    assert XP._v6_code("narrative_inputs: 평가 쪽 함수가 없다") == "narrative_inputs"
    assert XP._v6_code("code_bytes: 표가 없다") == "code_bytes"
    cells = [("uni_central", 1), ("uni_fed", 1), ("uni_local_C1", 1), ("uni_local_C2", 1), ("uni_local_C3", 1)]
    names = {f"{t}_s{s}" for t, s in cells}
    full = {"narrative": {"models": {n: {} for n in names}, "paired": {"a": 1}},
            "clause_axis": {"models": {n: {} for n in names}, "paired": {"a": 1}}}
    ok = SimpleNamespace(computed_names=lambda: list(ER.AUXILIARY_V6))
    got = SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=ok, report=full)
    assert got["complete"] is True and got["reasons"] == [] and got["required_names"] == list(ER.AUXILIARY_V6)
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=None, report=full)["reasons"] == ["aux_not_computed"]
    refused = SimpleNamespace(problems=("x",), computed_names=list)
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=refused, report=full)["reasons"] == ["aux_identity_refused"]
    partial = SimpleNamespace(computed_names=lambda: list(ER.AUXILIARY_V6)[:3])
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=partial, report=full)["reasons"] == ["aux_incomplete"]
    missing_model = {"narrative": {"models": {}, "paired": {}}, "clause_axis": full["clause_axis"]}
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=ok, report=missing_model)["reasons"] == ["aux_incomplete"]
    env = {"narrative": {**full["narrative"], "paired": {"status": "산출 불가", "reason": "env_mismatch"}}, "clause_axis": full["clause_axis"]}
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=ok, report=env)["reasons"] == ["aux_incomplete"]
    got = SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells[:4], aux=ok, report=full)
    assert got["complete"] is False and got["reasons"][0].startswith("cells_missing:1")


def test_v6_link_frame_gate_not_waived_in_registration(e2e):
    """D7 — 판 6 신판 등록도 `canary.frame_diag_sha256` 를 비울 수 없고, 작성기의 canary 필수 칸 검사는 그대로다."""
    e = e2e
    scoring = json.loads(json.dumps(e.scoring))
    scoring["canary"]["frame_diag_sha256"] = None
    with pytest.raises(MR.MainRegistrationRefused) as exc:
        _register(e, scoring=scoring)
    assert exc.value.code == "registration_incomplete" and "canary.frame_diag_sha256" in exc.value.detail
    assert P.VOCAB.get("canary.frame_diag_sha256") is None and "canary.frame_diag_sha256" in P.HEX64_FIELDS
