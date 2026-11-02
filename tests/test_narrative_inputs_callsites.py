"""학습 전 계약 대조의 학습 쪽 호출 자리(C 08 2 판 §5-3 · 07e §2-5 · §4 — 시험 14 · 16 의 생성 쪽 몫).

- 학습 등록에 `narrative_inputs: {version, sha256}` 칸이 있으면 등록 검증(`registration_problems` — 착수 고정을 쓰기 전)과 모델 적재 전
  공통 점검(`main_run_problems`)이 평가 쪽 `narrative_inputs_problems` 를 부른다. 칸이 없으면 지금 그대로다.
- 연합 서버는 앱 묶음에서 돌므로 실행 설정의 저장소 뿌리(`uni-repo-root`)와 서버가 맞댄 학습 등록을 그 점검에 넘긴다(09 §10 I2).
- 평가 쪽 함수가 없거나(평가 쪽 2 단계 전) 예외로 멈추면 **닫힌 쪽**이다.
- 넘기는 사실(`VerifiedPairs`)은 이 쪽이 실제 바이트를 본 결과다 — 시험은 평가 쪽 함수를 대역으로 갈아 끼워 넘긴 값을 본다. 평가 쪽
  사례 파일과의 결합(사례 (나)의 세계로 실제 함수를 부르는 것)은 평가 쪽 2 단계 병합 뒤의 통합 커밋이다.
- 판 6 ↔ 계약 ↔ 학습 PIN(`v6_problems`)은 내보내기 사전 점검이 본실험 모델 묶음에서 부를 함수다. 판 6 에 계약 칸이 있을 때만이다.
  배선은 판 6 발급 구현(2 차 등록 묶음)과 함께 한다 — 지금 판 6 은 평가 쪽 대조에서 `hash_rule` 로 멈춘다(07e §8 의 2).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import narrative_registration as NR
from fl.uni_fed import UniFedRejected
from tests.test_fed_token_table import V3_PROMPT, _main_get, _registration, _run
from tests.test_train_registration import _reg, _rule, _uni, _write
from vlm import narrative_inputs as NI
from vlm import registration as R

ENTRY = {"version": 1, "sha256": "c" * 64}


@dataclasses.dataclass(frozen=True)
class FakeVerifiedPairs:
    """평가 쪽 `VerifiedPairs` 의 대역 — 07e §2-3 의 칸 그대로."""

    path: str
    snapshot_digest: str
    snapshot_problems: tuple
    sealed_status: str | None
    pairs_meta_raw: bytes
    snapshot_pairs_meta_sha256: str
    config_pairs_digest: str
    token_table_pairs_digest: str


@pytest.fixture
def no_eval_functions(monkeypatch):
    for name in ("narrative_inputs_problems", "VerifiedPairs", "v6_contract_problems"):
        monkeypatch.delattr(NR, name, raising=False)


def _world(root: Path, *, digest_ok: bool = True) -> dict:
    """합성 저장소 — 계약 파일 · 페어 폴더(계약서 포함) · 행별 토큰 표의 출처. 내용은 시험용 합성이다."""
    from vlm.budget import PROVENANCE_SUFFIX, TOKEN_TABLES

    (root / "configs").mkdir(parents=True)
    contract = b'{"synthetic": "narrative_inputs"}\n'
    (root / NI.contract_rel(1)).write_bytes(contract)
    folder = root / "data/processed/pairs_main_v2"
    folder.mkdir(parents=True)
    members = {"pairs.jsonl": b"{}\n", "counts.json": b"{}\n", "discarded.jsonl": b"", "PAIRS_META.json": b'{"m": 1}\n'}
    hashes = []
    for name, raw in members.items():
        (folder / name).write_bytes(raw)
        hashes.append((hashlib.sha256(raw).hexdigest(), name))
    digest = hashlib.sha256("".join(h for h, _ in hashes).encode()).hexdigest()
    (folder / "SNAPSHOT.sha256").write_text("\n".join(f"{h}  {n}" for h, n in hashes) + f"\n# snapshot_digest {digest}\n",
                                            encoding="utf-8", newline="\n")
    table = root / TOKEN_TABLES["json_line_then_basis"][0]
    table.parent.mkdir(parents=True)
    table.with_name(table.name + PROVENANCE_SUFFIX).write_text(json.dumps({"pairs_digest": "t" * 64}), encoding="utf-8")
    return {"contract": contract, "digest": digest if digest_ok else "0" * 64, "pairs": "data/processed/pairs_main_v2/pairs.jsonl",
            "meta": members["PAIRS_META.json"]}


# ---------------------------------------------------------------- 칸의 꼴

@pytest.mark.parametrize("entry", [None, {"version": 1}, {"version": 1, "sha256": "c" * 64, "x": 1},
                                   {"version": True, "sha256": "c" * 64}, {"version": 0, "sha256": "c" * 64},
                                   {"version": 1, "sha256": "C" * 64}, {"version": 1, "sha256": "c" * 63}])
def test_칸의_꼴이_아니면_대조_전에_거부한다(entry):
    assert NI.entry_problems(entry)
    assert NI.problems(entry, repo=Path("."), pairs_path="x", config_pairs_digest=None, recipe="json_line_then_basis")


def test_등록_파일은_새_칸을_받는다(tmp_path):
    rule = _rule(tmp_path)
    p = _write(tmp_path, _reg(tmp_path, rule, narrative_inputs=dict(ENTRY)))
    reg, _ = R.load_registration(p)
    assert reg["narrative_inputs"] == ENTRY and "narrative_inputs" in R.OPTIONAL_KEYS


# ---------------------------------------------------------------- 닫힌 쪽

def test_평가_쪽_함수가_없으면_닫힌_쪽이다(tmp_path, no_eval_functions):
    w = _world(tmp_path)
    got = NI.problems(ENTRY, repo=tmp_path, pairs_path=w["pairs"], config_pairs_digest=w["digest"],
                      recipe="json_line_then_basis")
    assert len(got) == 1 and got[0].startswith("narrative_inputs: ") and "닫힌 쪽" in got[0]


def test_등록_검증은_칸이_있으면_착수_고정_전에_거부한다(tmp_path, monkeypatch, no_eval_functions):
    rule = _rule(tmp_path)
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: _uni())
    plain = _write(tmp_path, _reg(tmp_path, rule))
    assert R.registration_probe(None, path=plain, rule_path=rule)["registration"].get("narrative_inputs") is None
    p = _write(tmp_path, _reg(tmp_path, rule, narrative_inputs=dict(ENTRY)))
    with pytest.raises(R.RegistrationRejected) as e:
        R.registration_probe(None, path=p, rule_path=rule)
    assert any(x.startswith("narrative_inputs: ") for x in e.value.problems)


def test_모델_적재_전_점검도_칸이_있으면_거부하고_없으면_그대로다(tmp_path, monkeypatch, no_eval_functions):
    from vlm.uni_config import load_uni_config, main_run_problems

    cfg = load_uni_config()
    rule = _rule(tmp_path)
    kw = {"model_id": cfg.model_id, "model_revision": cfg.model_revision, "pairs_path": cfg.pairs_path,
          "pairs_digest": cfg.pairs_digest, "chat_template_kwargs": cfg.chat_template_kwargs, "num_rounds": cfg.num_rounds,
          "local_epochs": cfg.local_epochs, "total_epochs": cfg.total_epochs, "seed_index": 1,
          "seed_value": int(cfg.seeds[0]), "snapshot_digest": cfg.snapshot_digest, "prompt_path": cfg.prompt_path,
          "coord_space": cfg.coord_space, "output_paths": []}
    monkeypatch.setattr("vlm.budget.expected_tokens", lambda per_epoch, **_k: per_epoch)
    with_entry = _write(tmp_path, _reg(tmp_path, rule, narrative_inputs=dict(ENTRY)))
    got = main_run_problems(cfg, registration_path=with_entry, **kw)
    assert [x for x in got if x.startswith("narrative_inputs: ")]
    plain = tmp_path / "plain"
    plain.mkdir()
    no_entry = _write(plain, _reg(plain, rule))
    assert not [x for x in main_run_problems(cfg, registration_path=no_entry, **kw) if x.startswith("narrative_inputs")]


# ---------------------------------------------------------------- 넘기는 사실(평가 쪽 함수 대역)

def test_넘기는_사실은_실제_바이트에서_읽은_값이다(tmp_path, monkeypatch):
    from data import frozen_guard as FG

    # 이 사례는 미등록 폴더의 사실 전달을 검사하므로 합성 세계에서만 명부 행을 뺀다.
    monkeypatch.delitem(FG.EXPECTED_SEALED, "data/processed/pairs_main_v2", raising=False)
    w = _world(tmp_path)
    seen = {}

    def fake(contract_raw, *, registered_sha256, repo, pairs):
        seen.update(contract_raw=contract_raw, registered_sha256=registered_sha256, repo=repo, pairs=pairs)
        return ["pairs_digest: 합성 사유"]

    monkeypatch.setattr(NR, "VerifiedPairs", FakeVerifiedPairs, raising=False)
    monkeypatch.setattr(NR, "narrative_inputs_problems", fake, raising=False)
    got = NI.problems(ENTRY, repo=tmp_path, pairs_path=w["pairs"], config_pairs_digest=w["digest"],
                      recipe="json_line_then_basis")
    assert got == ["narrative_inputs: pairs_digest: 합성 사유"]
    assert seen["contract_raw"] == w["contract"] and seen["registered_sha256"] == ENTRY["sha256"] and seen["repo"] == tmp_path
    p = seen["pairs"]
    assert (p.path, p.snapshot_digest, p.snapshot_problems) == ("data/processed/pairs_main_v2", w["digest"], ())
    assert p.pairs_meta_raw == w["meta"] and p.snapshot_pairs_meta_sha256 == hashlib.sha256(w["meta"]).hexdigest()
    assert (p.config_pairs_digest, p.token_table_pairs_digest) == (w["digest"], "t" * 64)
    assert p.sealed_status is None                                    # 합성 폴더는 봉인 명부에 없다 — 평가 쪽이 거부할 사실이다


def test_스냅샷_대조_문제는_빈_목록이_아니라_그대로_넘긴다(tmp_path, monkeypatch):
    w = _world(tmp_path, digest_ok=False)
    seen = {}
    monkeypatch.setattr(NR, "VerifiedPairs", FakeVerifiedPairs, raising=False)
    monkeypatch.setattr(NR, "narrative_inputs_problems", lambda *a, **k: seen.update(k) or [], raising=False)
    assert NI.problems(ENTRY, repo=tmp_path, pairs_path=w["pairs"], config_pairs_digest=w["digest"],
                       recipe="json_line_then_basis") == []
    assert seen["pairs"].snapshot_problems and seen["pairs"].config_pairs_digest == "0" * 64


def test_대조가_예외로_멈추면_닫힌_쪽이다(tmp_path, monkeypatch):
    w = _world(tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError("합성 예외")

    monkeypatch.setattr(NR, "VerifiedPairs", FakeVerifiedPairs, raising=False)
    monkeypatch.setattr(NR, "narrative_inputs_problems", boom, raising=False)
    got = NI.problems(ENTRY, repo=tmp_path, pairs_path=w["pairs"], config_pairs_digest=w["digest"],
                      recipe="json_line_then_basis")
    assert len(got) == 1 and "예외로 멈췄다" in got[0]


def test_계약_파일이_없으면_거부한다(tmp_path, monkeypatch):
    monkeypatch.setattr(NR, "VerifiedPairs", FakeVerifiedPairs, raising=False)
    monkeypatch.setattr(NR, "narrative_inputs_problems", lambda *a, **k: [], raising=False)
    got = NI.problems({"version": 2, "sha256": "c" * 64}, repo=tmp_path, pairs_path="x/pairs.jsonl",
                      config_pairs_digest=None, recipe="json_line_then_basis")
    assert got and "계약 파일이 없다" in got[0]


# ---------------------------------------------------------------- 판 6 ↔ 계약 ↔ 학습 PIN(내보내기 사전 점검)

def _reg6(sha: str | None):
    return SimpleNamespace(narrative=SimpleNamespace(inputs_contract_sha256=sha))


def _receipt_files(tmp_path, entry=ENTRY) -> dict:
    import yaml

    rule = _rule(tmp_path)
    body = _reg(tmp_path, rule, **({"narrative_inputs": dict(entry)} if entry else {}))
    return {"configs/train_registration.yaml": yaml.safe_dump(body, allow_unicode=True, sort_keys=False).encode("utf-8"),
            NI.contract_rel(1): b'{"synthetic": "contract"}\n'}


def _adapter(tmp_path) -> Path:
    seed = tmp_path / "seed1"
    cell = seed / "uni_local_C1_s1"
    cell.mkdir(parents=True)
    (seed / NI.PIN_NAME).write_bytes(b'{"schema": "pin"}\n')
    return cell


def test_판_6_에_계약_칸이_없으면_부르지_않는다(tmp_path, no_eval_functions):
    assert NI.v6_problems(_reg6(None), read_at_receipt=lambda rel: b"", adapter_dir=tmp_path, adapter_meta={}) == []
    assert NI.v6_problems(SimpleNamespace(), read_at_receipt=lambda rel: b"", adapter_dir=tmp_path, adapter_meta={}) == []


def test_판_6_에_계약_칸이_있는데_평가_쪽_함수가_없으면_닫힌_쪽이다(tmp_path, no_eval_functions):
    got = NI.v6_problems(_reg6("a" * 64), read_at_receipt=lambda rel: b"", adapter_dir=tmp_path, adapter_meta={})
    assert len(got) == 1 and "닫힌 쪽" in got[0]


def test_판_6_대조에_넘기는_것은_영수증_커밋의_계약과_시드_폴더의_PIN_이다(tmp_path, monkeypatch):
    files = _receipt_files(tmp_path)
    cell = _adapter(tmp_path)
    seen = {}

    def fake(registration, contract_raw, *, pin_raw, adapter_pin_sha256):
        seen.update(registration=registration, contract_raw=contract_raw, pin_raw=pin_raw, pins=list(adapter_pin_sha256))
        return ["pin_contract: 합성 사유"]

    monkeypatch.setattr(NR, "v6_contract_problems", fake, raising=False)
    reg = _reg6("a" * 64)
    got = NI.v6_problems(reg, read_at_receipt=files.__getitem__, adapter_dir=cell,
                         adapter_meta={"start_gate": {"pin_sha256": "p" * 64}})
    assert got == ["narrative_inputs: pin_contract: 합성 사유"]
    assert seen["registration"] is reg and seen["contract_raw"] == files[NI.contract_rel(1)]
    assert seen["pin_raw"] == (tmp_path / "seed1" / NI.PIN_NAME).read_bytes() and seen["pins"] == ["p" * 64]


def test_영수증_커밋의_학습_등록에_칸이_없으면_판_6_을_거부한다(tmp_path, monkeypatch):
    monkeypatch.setattr(NR, "v6_contract_problems", lambda *a, **k: [], raising=False)
    files = _receipt_files(tmp_path, entry=None)
    got = NI.v6_problems(_reg6("a" * 64), read_at_receipt=files.__getitem__, adapter_dir=_adapter(tmp_path),
                         adapter_meta={})
    assert got and "칸이 없다" in got[0]


def test_영수증_커밋에서_읽지_못하면_판_6_을_거부한다(tmp_path, monkeypatch):
    monkeypatch.setattr(NR, "v6_contract_problems", lambda *a, **k: [], raising=False)

    def missing(rel):
        raise RuntimeError(f"{rel} 를 그 커밋에서 읽지 못했다")

    got = NI.v6_problems(_reg6("a" * 64), read_at_receipt=missing, adapter_dir=_adapter(tmp_path), adapter_meta={})
    assert got and "읽지 못했다" in got[0]


# ---------------------------------------------------------------- 연합 서버(09 §10 I2)

def test_연합_서버는_저장소_뿌리와_맞댄_등록을_점검에_넘긴다(tmp_path, monkeypatch):
    """서버가 맞댄 학습 등록과 실행 설정의 `uni-repo-root` 를 `main_run_problems` 에 넘긴다. 뿌리가 없는 1 차 실행(비율 1)은
    `repo=None`(이 모듈의 뿌리) · 기본 등록 그대로다."""
    import vlm.uni_config as UC

    seen = {}
    monkeypatch.setattr(UC, "main_run_problems", lambda cfg, **kw: seen.update(kw) or [])
    repo = tmp_path / "checkout"
    reg = _registration(tmp_path, "json_only")
    _run(tmp_path, _main_get(tmp_path, **{"uni-repo-root": repo.as_posix()}), registration_path=reg)
    assert (seen["repo"], seen["registration_path"]) == (repo, reg)
    seen.clear()
    from tests.registration_history import legacy_default

    legacy_default(monkeypatch, tmp_path)
    _run(tmp_path, _main_get(tmp_path, **{"uni-budget-fraction": ""}))
    assert (seen["repo"], seen["registration_path"]) == (None, None)


@pytest.mark.parametrize("with_entry", [True, False])
def test_연합_서버의_계약_대조는_앱_묶음이_아니라_저장소_뿌리를_읽는다(tmp_path, monkeypatch, with_entry):
    """앱 묶음 경로(이 모듈의 뿌리)와 실제 저장소 경로(`uni-repo-root`)가 다를 때 — 칸이 있는 등록이면 평가 쪽 대조가 실제 저장소
    뿌리로 불리고, 칸이 없는 등록은 대조를 부르지 않는다(지금 그대로). 진짜 `main_run_problems` 를 거친다 — 합성 설정의 다른 칸이
    어긋나 서버는 `main_config_mismatch` 로 멈추지만, 이 시험이 보는 것은 대조에 넘어간 뿌리다."""
    import vlm.uni_config as UC

    calls = []
    monkeypatch.setattr(NI, "problems", lambda entry, **kw: calls.append(kw) or [])
    monkeypatch.setattr("vlm.budget.expected_tokens", lambda per_epoch, **_k: per_epoch)
    repo = tmp_path / "checkout"
    assert repo.resolve() != Path(UC.REPO_ROOT).resolve()
    rule = _rule(tmp_path)
    reg = _write(tmp_path, _reg(tmp_path, rule, **({"narrative_inputs": dict(ENTRY)} if with_entry else {})))
    with pytest.raises(UniFedRejected) as exc:
        _run(tmp_path, _main_get(tmp_path, **{"uni-repo-root": repo.as_posix()}), registration_path=reg)
    assert exc.value.code == "main_config_mismatch"
    assert [c["repo"] for c in calls] == ([repo] if with_entry else [])


# ---------------------------------------------------------------- 결합 — 실제 평가 쪽 함수 · 실제 봉인 메타(대역 없음, 지시 1024i)

JOINT_PAIRS = Path(__file__).resolve().parents[1] / "data/processed/pairs_main_v2"
"""결합 시험의 페어 — 최종 봉인 폴더의 다섯 파일을 읽기만 한다."""
WORLD_PAIRS_REL = "data/processed/pairs_main_v2"
FIVE = ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json", "SNAPSHOT.sha256")
V2_RECIPE = "json_line_then_basis"


@pytest.fixture(scope="module")
def sealed_files(tmp_path_factory):
    """봉인 페어의 다섯 파일을 한 번만 옮긴 사본 — 시험마다 세계에 걸어 쓴다. 없으면 자산 부재다."""
    if not all((JOINT_PAIRS / n).is_file() for n in FIVE):
        pytest.fail("봉인 페어가 이 트리에 없다 — 자산 부재를 먼저 의심한다: data/processed/pairs_main_v2")
    base = tmp_path_factory.mktemp("sealed_pairs")
    for n in FIVE:
        shutil.copyfile(JOINT_PAIRS / n, base / n)
    return base


def _place(src: Path, dst: Path) -> None:
    """큰 페어 파일은 걸기만 한다(같은 볼륨이면 하드 링크) — 결합 시험은 이 파일을 고치지 않는다."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _joint_world(root: Path, b1p: Path, *, edit_pairs=None) -> dict:
    """실제 표 · 로더 · 생성 경로 · 봉인 페어 다섯 파일 · 평가 쪽 작성기의 정규형 계약 · 토큰 표 출처로 지은 저장소.

    `edit_pairs` 는 계약을 쓰기 **전에** 페어 폴더를 고치는 자리다(고친 뒤 스냅샷을 다시 내는 것도 그 함수의 몫)."""
    from vlm.budget import PROVENANCE_SUFFIX, TOKEN_TABLES

    repo = Path(__file__).resolve().parents[1]
    table_rel = json.loads((b1p / "PAIRS_META.json").read_bytes())["limits_table"]["path"]
    for rel in (*NR.LIMITS_LOADER_FILES, *NR.GENERATOR_FILES_V1, table_rel):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / rel, root / rel)
    folder = root / WORLD_PAIRS_REL
    folder.mkdir(parents=True)
    for n in FIVE:
        (_place if n == "pairs.jsonl" else shutil.copyfile)(b1p / n, folder / n)
    if edit_pairs is not None:
        edit_pairs(folder)
    meta_raw = (folder / "PAIRS_META.json").read_bytes()
    digest = NI._snapshot(folder)[0]
    contract = NR.dump_narrative_inputs(NR.narrative_inputs_block(root, limits_rel=table_rel, pairs_path=WORLD_PAIRS_REL,
                                                                  snapshot_digest=digest, pairs_meta_raw=meta_raw))
    (root / NI.contract_rel(1)).parent.mkdir(parents=True, exist_ok=True)
    (root / NI.contract_rel(1)).write_bytes(contract)
    table = root / TOKEN_TABLES[V2_RECIPE][0]
    table.parent.mkdir(parents=True, exist_ok=True)
    table.with_name(table.name + PROVENANCE_SUFFIX).write_bytes(json.dumps({"pairs_digest": digest}).encode("utf-8"))
    return {"root": root, "entry": {"version": 1, "sha256": hashlib.sha256(contract).hexdigest()}, "digest": digest,
            "meta_raw": meta_raw, "table_rel": table_rel, "pairs": f"{WORLD_PAIRS_REL}/pairs.jsonl"}


def _app_bundle(tmp: Path, w: dict) -> Path:
    """풀어 놓은 앱 묶음의 모사 — py · configs json 만 있고 표(csv) · 페어 · 토큰 표 출처는 없다(앱 묶음의 포함 규칙)."""
    app = tmp / "flwr_home/apps/weld-fl.weld-fl.0.1.0.deadbeef"
    for rel in (*NR.LIMITS_LOADER_FILES, *NR.GENERATOR_FILES_V1, NI.contract_rel(1)):
        (app / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(w["root"] / rel, app / rel)
    return app


def _sealed(monkeypatch, on: bool) -> None:
    """봉인 명부(데이터 쪽 상수)에 세계의 페어 폴더를 둔다 · 뺀다 — 파일은 고치지 않고 이 시험 안에서만."""
    from data import frozen_guard as FG

    if on:
        monkeypatch.setitem(FG.EXPECTED_SEALED, WORLD_PAIRS_REL, {"status": "expected", "owner": "C", "record": "결합 시험 세계"})
    else:
        monkeypatch.delitem(FG.EXPECTED_SEALED, WORLD_PAIRS_REL, raising=False)


def _spy_ni(monkeypatch) -> list:
    """실제 호출 자리가 실제 대조에 닿았는지 — 진짜 함수를 감싸 부른 횟수만 센다(대역이 아니다)."""
    calls, real = [], NI.problems

    def wrapped(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(NI, "problems", wrapped)
    return calls


def _ni(w: dict, repo: Path | None = None) -> list[str]:
    return NI.problems(w["entry"], repo=repo or w["root"], pairs_path=w["pairs"], config_pairs_digest=w["digest"], recipe=V2_RECIPE)


def _codes(lines: list[str]) -> set[str]:
    assert all(x.startswith("narrative_inputs: ") for x in lines), lines
    return {x.removeprefix("narrative_inputs: ").split(":", 1)[0] for x in lines}


def test_결합_실제_b1p_메타와_평가_쪽_작성기의_계약을_실제_대조가_받는다(tmp_path, monkeypatch, sealed_files):
    w = _joint_world(tmp_path / "repo", sealed_files)
    _sealed(monkeypatch, True)
    assert _ni(w) == []
    meta = json.loads(w["meta_raw"])
    assert meta["limits_table"].get("status")                                          # I-02 — status 는 제자리에 남는다
    assert meta["identity"]["limits"] == {k: meta["limits_table"][k] for k in ("path", "sha256")}


def _edit_loader(w: dict) -> None:
    p = w["root"] / "corpus/rules/schema.py"
    p.write_bytes(p.read_bytes() + "\n# 결합 시험 — 코드 바이트 한 줄\n".encode())


def _edit_condition(w: dict) -> None:
    p = w["root"] / w["table_rel"]
    text = p.read_bytes().decode("utf-8")
    assert "|무등급전제" in text
    p.write_bytes(text.replace("|무등급전제", "", 1).encode("utf-8"))


def _edit_snapshot(w: dict) -> None:
    p = w["root"] / WORLD_PAIRS_REL / "counts.json"
    p.write_bytes(p.read_bytes() + b" ")


def _edit_identity(folder: Path) -> None:
    """PAIRS_META 신원 블록의 한 칸 — 스냅샷 · 계약은 고친 바이트로 다시 낸다(메타 바이트 · digest 는 맞고 신원만 다르다)."""
    from corpus.generate.make_pairs_pilot import snapshot_digest

    meta = json.loads((folder / "PAIRS_META.json").read_bytes())
    meta["identity"]["generator"]["files"][0]["sha256"] = "0" * 64
    (folder / "PAIRS_META.json").write_bytes(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8") + b"\n")
    lines = (folder / "SNAPSHOT.sha256").read_text(encoding="utf-8").splitlines()
    snapshot_digest(folder, [ln.split(None, 1)[1].strip() for ln in lines if ln.strip() and not ln.startswith("#")])


@pytest.mark.parametrize(("case", "code"), [("코드 바이트", "code_bytes"), ("조건 바이트", "code_bytes"),
                                            ("메타 신원", "pairs_meta_identity"), ("미봉인", "pairs_unverified"),
                                            ("스냅샷", "pairs_unverified"), ("앱 묶음 뿌리", "code_bytes")])
def test_결합_어긋나면_실제_대조가_그_사유로_거부한다(tmp_path, monkeypatch, sealed_files, case, code):
    w = _joint_world(tmp_path / "repo", sealed_files, edit_pairs=_edit_identity if case == "메타 신원" else None)
    _sealed(monkeypatch, case != "미봉인")
    repo = None
    if case == "코드 바이트":
        _edit_loader(w)
    elif case == "조건 바이트":
        _edit_condition(w)
    elif case == "스냅샷":
        _edit_snapshot(w)
    elif case == "앱 묶음 뿌리":
        repo = _app_bundle(tmp_path, w)
    assert _codes(_ni(w, repo)) == {code}


@pytest.mark.parametrize("sealed", [True, False])
def test_결합_학습_등록_검증이_실제_대조를_부른다(tmp_path, monkeypatch, sealed_files, sealed):
    w = _joint_world(tmp_path / "repo", sealed_files)
    _sealed(monkeypatch, sealed)
    calls = _spy_ni(monkeypatch)
    monkeypatch.setattr(R, "REPO_ROOT", w["root"])
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: _uni(pairs_path=w["pairs"], pairs_digest=w["digest"]))
    rule = _rule(tmp_path)
    reg = _write(tmp_path, _reg(tmp_path, rule, recipe=V2_RECIPE, narrative_inputs=dict(w["entry"])))
    try:
        R.registration_probe(None, path=reg, rule_path=rule)
        lines = []
    except R.RegistrationRejected as exc:
        lines = [x for x in exc.problems if x.startswith("narrative_inputs")]
    assert calls == [1]
    assert _codes(lines) == (set() if sealed else {"pairs_unverified"})


@pytest.mark.parametrize("where", ["저장소 뿌리", "앱 묶음 뿌리"])
def test_결합_적재_전_점검은_받은_뿌리로_실제_대조를_부른다(tmp_path, monkeypatch, sealed_files, where):
    from vlm.uni_config import load_uni_config, main_run_problems

    w = _joint_world(tmp_path / "repo", sealed_files)
    _sealed(monkeypatch, True)
    calls = _spy_ni(monkeypatch)
    monkeypatch.setattr("vlm.budget.expected_tokens", lambda per_epoch, **_k: per_epoch)
    cfg = dataclasses.replace(load_uni_config(), pairs_path=w["pairs"], pairs_digest=w["digest"])
    rule = _rule(tmp_path)
    reg = _write(tmp_path, _reg(tmp_path, rule, recipe=V2_RECIPE, narrative_inputs=dict(w["entry"])))
    kw = {"model_id": cfg.model_id, "model_revision": cfg.model_revision, "pairs_path": cfg.pairs_path,
          "pairs_digest": cfg.pairs_digest, "chat_template_kwargs": cfg.chat_template_kwargs, "num_rounds": cfg.num_rounds,
          "local_epochs": cfg.local_epochs, "total_epochs": cfg.total_epochs, "seed_index": 1,
          "seed_value": int(cfg.seeds[0]), "snapshot_digest": cfg.snapshot_digest, "prompt_path": cfg.prompt_path,
          "coord_space": cfg.coord_space, "output_paths": []}
    root = w["root"] if where == "저장소 뿌리" else _app_bundle(tmp_path, w)
    got = main_run_problems(cfg, registration_path=reg, repo=root, recipe=V2_RECIPE, **kw)
    assert calls == [1]
    assert _codes([x for x in got if x.startswith("narrative_inputs")]) == (set() if where == "저장소 뿌리" else {"code_bytes"})


@pytest.mark.parametrize("where", ["저장소 뿌리", "앱 묶음 뿌리"])
def test_결합_연합_서버는_실행_설정의_뿌리로_실제_대조를_부른다(tmp_path, monkeypatch, sealed_files, where):
    import vlm.uni_config as UC

    w = _joint_world(tmp_path / "repo", sealed_files)
    _sealed(monkeypatch, True)
    calls = _spy_ni(monkeypatch)
    real = UC.load_uni_config()
    monkeypatch.setattr(UC, "load_uni_config",
                        lambda path=None: dataclasses.replace(real, pairs_path=w["pairs"], pairs_digest=w["digest"]))
    root = w["root"] if where == "저장소 뿌리" else _app_bundle(tmp_path, w)
    reg = _registration(tmp_path, V2_RECIPE)                     # 실제 등록 바이트 — 레시피만 신판, 계약 칸을 덧붙인다
    entry = f"\nnarrative_inputs:\n  version: 1\n  sha256: {w['entry']['sha256']}\n"
    reg.write_bytes(reg.read_bytes().rstrip(b"\n") + entry.encode("utf-8"))
    get = _main_get(tmp_path, **{"uni-repo-root": root.as_posix(), "uni-budget-fraction": "", "uni-target-recipe": V2_RECIPE,
                                 "uni-prompt": V3_PROMPT, "uni-pairs": w["pairs"]})
    with pytest.raises(UniFedRejected) as exc:                  # 합성 실행 설정의 다른 칸은 어긋난다 — 이 시험이 보는 것은 계약 줄이다
        _run(tmp_path, get, registration_path=reg)
    assert exc.value.code == "main_config_mismatch" and calls == [1]
    got = [x.strip() for x in str(exc.value).split(" · ") if x.strip().startswith("narrative_inputs")]
    assert _codes(got) == (set() if where == "저장소 뿌리" else {"code_bytes"})


# ---------------------------------------------------------------- 판 6 연결 2 판 — 계약 실물 대조를 PIN 대조 앞에(§4-3 · R15 의 호출 자리)
_ROOT = Path(__file__).resolve().parents[1]


def test_v6_link_contract_files_checked_before_pin(tmp_path, monkeypatch):
    """`repo` 를 주면 영수증 커밋의 계약 바이트를 실물 파일(표 raw · 로더 · 생성 묶음)과 먼저 맞댄다 — 꼴이 틀리면 `contract_form`, 파일이
    없으면 `code_bytes`, 그 줄은 PIN 대조(`v6_contract_problems`) 앞에서 끝난다. `repo` 가 없으면 옛 동작 그대로다."""
    files = _receipt_files(tmp_path)
    cell = _adapter(tmp_path)
    calls = []
    monkeypatch.setattr(NR, "v6_contract_problems", lambda *a, **k: calls.append(1) or [], raising=False)
    assert NI.v6_problems(_reg6("a" * 64), read_at_receipt=files.__getitem__, adapter_dir=cell, adapter_meta={}) == []
    assert calls == [1]
    got = NI.v6_problems(_reg6("a" * 64), read_at_receipt=files.__getitem__, adapter_dir=cell, adapter_meta={}, repo=tmp_path)
    assert got and got[0].startswith("narrative_inputs: contract_form:") and calls == [1]            # 합성 계약은 꼴에서 멈춘다
    real = (_ROOT / NR.INPUTS_CONTRACT_REL).read_bytes()
    files2 = {**files, NI.contract_rel(1): real}
    got = NI.v6_problems(_reg6("a" * 64), read_at_receipt=files2.__getitem__, adapter_dir=cell, adapter_meta={}, repo=tmp_path)
    assert got and got[0].startswith("narrative_inputs: code_bytes:") and calls == [1]               # 실물이 없는 저장소
    assert NI.v6_problems(_reg6("a" * 64), read_at_receipt=files2.__getitem__, adapter_dir=cell, adapter_meta={}, repo=_ROOT) == []
    assert calls == [1, 1]                                                                            # 실물이 맞으면 PIN 대조로
