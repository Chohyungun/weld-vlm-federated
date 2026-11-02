"""학습 전 계약 묶음 · PAIRS_META 신원 · 판 6 ↔ 계약 ↔ 학습 착수 고정(07 4판 §17 · §18, 인터페이스 문서 07e §2 ~ §4).

기대(사유 코드의 정확한 집합)는 구현 전에 고정한 사례 파일(`evaluation/vectors/narrative_inputs_cases.v1.json`)에서만 읽는다. 시험은 사례
파일의 합성 세계(파일 내용이 모두 합성 — 허용치 수치 없음)를 임시 폴더에 짓고 사례마다 바꿀 것(ops)을 얹는다. 스냅샷 digest 는 세계의
이름표(`snapshot_label`)의 sha256 으로 짓는다. 모델 · GPU · 이미지 열기 0.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation import narrative_registration as NR
from tests.test_prereg_v6 import v6

REPO = Path(__file__).resolve().parents[1]
CASES_REL = "evaluation/vectors/narrative_inputs_cases.v1.json"
CASES_SHA256 = "4d06a076fc52fed0f96408d0b7923eed4dfe12e3ac3d46bed18af15b09f764b5"
"""구현 전 고정 커밋(91e83979)의 바이트 — 기대를 바꾸려면 새 판의 파일을 내고 기대 변경 표에 적는다."""
CASES = json.loads((REPO / CASES_REL).read_bytes().decode("utf-8"))
W = CASES["world"]
TABLE_V1 = CASES["world_pairs_meta_v1"]["limits_table"]["path"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def text_sha(key: str) -> str:
    return sha(W["texts"][key].encode("utf-8"))


def snap(label_key: str = "snapshot_label") -> str:
    return sha(W["pairs"][label_key].encode("utf-8"))


def rows_sha(rows: list[dict]) -> str:
    return sha("".join(f"{r['path']}\t{r['sha256']}\n" for r in rows).encode("utf-8"))


def write_world(root: Path) -> None:
    for rel, text in W["files"].items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text.encode("utf-8"))


def write_text(root: Path, rel: str, key: str) -> None:
    (root / rel).write_bytes(W["texts"][key].encode("utf-8"))


def world_meta(root: Path, table: str | None = None) -> dict:
    table = table or W["limits_rel"]
    t = {"path": table, "sha256": NR.raw_sha256(root / table)}
    return {**W["pairs_meta_extra"], "limits_table": {**t, "status": W["limits_status"]},
            "identity": {"contract_version": NR.INPUTS_CONTRACT_VERSION, "limits": dict(t),
                         "limits_loader": NR.bundle_block(root, NR.LIMITS_LOADER_FILES),
                         "generator": NR.bundle_block(root, NR.GENERATOR_FILES_V1), "conditions": {"in_table": True}},
            "text_contract": NR.TEXT_CONTRACT_V2}


def world_meta_v1(root: Path) -> dict:
    t = {"path": TABLE_V1, "sha256": NR.raw_sha256(root / TABLE_V1)}
    return {**W["pairs_meta_v1_extra"], "limits_table": {**t, "status": W["limits_status"]}}


def dump_meta(meta: dict) -> bytes:
    return (json.dumps(meta, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def walk(obj, path: list):
    for key in path[:-1]:
        obj = obj[key]
    return obj, path[-1]


def value_of(op: dict, root: Path):
    if "value" in op:
        return copy.deepcopy(op["value"])
    src = op["from"]
    if "text_sha" in src:
        return text_sha(src["text_sha"])
    if "lf_sha" in src:
        return NR.lf_sha256(root / src["lf_sha"])
    if "raw_sha" in src:
        return NR.raw_sha256(root / src["raw_sha"])
    raise AssertionError(f"모르는 값의 출처 {src}")


def apply_ops(obj, ops: list[dict], root: Path):
    """사례 파일의 `ops_meaning` 그대로."""
    for op in ops:
        if "set" in op:
            parent, key = walk(obj, op["set"])
            parent[key] = value_of(op, root)
        elif "del" in op:
            parent, key = walk(obj, op["del"])
            del parent[key]
        elif "copy" in op:
            src, skey = walk(obj, op["copy"])
            dst, dkey = walk(obj, op["to"])
            dst[dkey] = copy.deepcopy(src[skey])
        elif "reverse" in op:
            parent, key = walk(obj, op["reverse"])
            parent[key] = list(reversed(parent[key]))
        elif "duplicate_first" in op:
            parent, key = walk(obj, op["duplicate_first"])
            parent[key] = [parent[key][0], *parent[key]]
        elif "upper" in op:
            parent, key = walk(obj, op["upper"])
            parent[key] = parent[key].upper()
        elif "drop_first_file" in op:
            parent, key = walk(obj, op["drop_first_file"])
            rows = parent[key]["files"][1:]
            parent[key] = {**parent[key], "files": rows, "sha256": rows_sha(rows)}
        else:
            raise AssertionError(f"모르는 op {op}")
    return obj


def base_verified(meta_raw: bytes) -> NR.VerifiedPairs:
    return NR.VerifiedPairs(path=W["pairs"]["path"], snapshot_digest=snap(), snapshot_problems=(), sealed_status="expected",
                            pairs_meta_raw=meta_raw, snapshot_pairs_meta_sha256=sha(meta_raw), config_pairs_digest=snap(),
                            token_table_pairs_digest=snap())


def build(case: dict, root: Path) -> dict:
    """사례 파일 `world` 의 차례(07e §2-6) — ① files → ② PAIRS_META · meta_ops → ③ mid_files → ④ 계약 · contract_ops → ⑤ 학습 등록의
    sha → ⑥ VerifiedPairs · verified_ops → ⑦ post_files."""
    write_world(root)
    meta = world_meta_v1(root) if case.get("meta_variant") == "v1" else world_meta(root, case.get("meta_table"))
    apply_ops(meta, case.get("meta_ops", []), root)
    meta_raw = case["meta_raw_text"].encode("utf-8") if "meta_raw_text" in case else dump_meta(meta)
    for f in case.get("mid_files", []):
        write_text(root, f["path"], f["text"])
    contract = NR.narrative_inputs_block(root, limits_rel=case.get("contract_table", W["limits_rel"]),
                                         pairs_path=W["pairs"]["path"], snapshot_digest=snap(), pairs_meta_raw=meta_raw)
    original = NR.dump_narrative_inputs(contract)
    apply_ops(contract, case.get("contract_ops", []), root)
    if "contract_raw_text" in case:
        raw = case["contract_raw_text"].encode("utf-8")
    elif case.get("contract_noncanonical"):
        raw = (json.dumps(contract, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    else:
        raw = NR.dump_narrative_inputs(contract)
    verified = base_verified(meta_raw)
    if case.get("verified_from_variant") == "v1":
        v1_raw = dump_meta(world_meta_v1(root))
        d1 = snap("snapshot_label_v1")
        verified = replace(verified, path=W["pairs"]["path_v1"], snapshot_digest=d1, pairs_meta_raw=v1_raw,
                           snapshot_pairs_meta_sha256=sha(v1_raw), config_pairs_digest=d1, token_table_pairs_digest=d1)
    for op in case.get("verified_ops", []):
        v = value_of(op, root)
        verified = replace(verified, **{op["set"]: tuple(v) if isinstance(v, list) else v})
    for f in case.get("post_files", []):
        write_text(root, f["path"], f["text"])
    return {"raw": raw, "registered": sha(original) if case.get("keep_registered") else sha(raw), "verified": verified,
            "meta": meta, "meta_raw": meta_raw, "contract": contract}


def codes(problems: list[str]) -> list[str]:
    assert all(isinstance(p, str) and ": " in p for p in problems), problems
    return sorted({p.split(":", 1)[0] for p in problems})


def test_사례_파일은_구현_전에_고정한_바이트다() -> None:
    assert sha((REPO / CASES_REL).read_bytes()) == CASES_SHA256
    assert CASES["kind"] == "narrative_inputs_cases" and CASES["version"] == 1
    assert set(CASES["reason_codes"]) <= set(NR.REASON_CODES)
    assert CASES["loader_files"] == list(NR.LIMITS_LOADER_FILES) and CASES["generator_files"] == list(NR.GENERATOR_FILES_V1)


def test_합성_세계의_PAIRS_META_는_사례_파일의_객체와_같다(tmp_path: Path) -> None:
    """두 구현 대조 — 사례 빌더가 낸 메타와 이 시험이 평가 쪽 묶음 함수로 지은 메타가 같다."""
    write_world(tmp_path)
    assert world_meta(tmp_path) == CASES["world_pairs_meta"]
    assert world_meta_v1(tmp_path) == CASES["world_pairs_meta_v1"]


@pytest.mark.parametrize("case", CASES["contract_cases"], ids=lambda c: c["id"])
def test_학습_전_계약_대조는_사례의_사유_코드를_낸다(case: dict, tmp_path: Path) -> None:
    b = build(case, tmp_path)
    got = NR.narrative_inputs_problems(b["raw"], registered_sha256=b["registered"], repo=tmp_path, pairs=b["verified"])
    assert codes(got) == case["expect"], got


@pytest.mark.parametrize("case", CASES["identity_cases"], ids=lambda c: c["id"])
def test_PAIRS_META_신원_대조는_사례의_사유_코드를_낸다(case: dict, tmp_path: Path) -> None:
    b = build(case, tmp_path)
    got = NR.pairs_identity_problems(b["meta_raw"], b["contract"])
    assert codes(got) == case["expect"], got
    if case.get("joint_i02"):                       # 생성 쪽 시험 20 과 같은 입력 — 메타 객체가 사례 파일에 그대로 있다
        assert b["meta"] == case["pairs_meta"]
        assert codes(NR.pairs_identity_problems(dump_meta(case["pairs_meta"]), b["contract"])) == case["expect"]


def build_v6(case: dict, root: Path) -> tuple:
    write_world(root)
    meta_raw = dump_meta(world_meta(root))

    def contract_now() -> tuple[dict, bytes]:
        c = NR.narrative_inputs_block(root, limits_rel=W["limits_rel"], pairs_path=W["pairs"]["path"],
                                      snapshot_digest=snap(), pairs_meta_raw=meta_raw)
        return c, NR.dump_narrative_inputs(c)

    contract, raw = contract_now()
    pin = {"schema": "synthetic_pin", "common": {"registration": {NR.INPUTS_REGISTRATION_KEY: {
        "version": NR.INPUTS_CONTRACT_VERSION, "sha256": sha(raw)}}}}
    if "swap_contract" in case:                     # 학습 PIN 은 처음 계약, 판 6 · 넘긴 계약은 바뀐 트리로 다시 지은 계약
        write_text(root, case["swap_contract"]["path"], case["swap_contract"]["text"])
        contract, raw = contract_now()
    reg = v6()
    reg = replace(reg, narrative=replace(reg.narrative, limits_sha256=contract["limits"]["sha256"],
                                         limits_loader_sha256=contract["limits_loader"]["sha256"],
                                         generator_sha256=contract["generator"]["sha256"], inputs_contract_sha256=sha(raw)),
                  clause_axis=replace(reg.clause_axis, limits_sha256=contract["limits"]["sha256"]))
    for op in case.get("v6_ops", []):
        v = value_of(op, root)
        if op["set"] == ["version"]:
            reg = replace(reg, version=v)
        else:
            area, name = op["set"]
            reg = replace(reg, **{area: replace(getattr(reg, area), **{name: v})})
    apply_ops(pin, case.get("pin_ops", []), root)
    pin_raw = case["pin_raw_text"].encode("utf-8") if "pin_raw_text" in case else \
        json.dumps(pin, sort_keys=True, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
    if "contract_raw_text" in case:
        raw = case["contract_raw_text"].encode("utf-8")
    adapters = {"other": [text_sha("other")] * 5, "empty": []}.get(case.get("adapters"), [sha(pin_raw)] * 5)
    return reg, raw, pin_raw, adapters


@pytest.mark.parametrize("case", CASES["v6_cases"], ids=lambda c: c["id"])
def test_판_6_계약_PIN_대조는_사례의_사유_코드를_낸다(case: dict, tmp_path: Path) -> None:
    reg, raw, pin_raw, adapters = build_v6(case, tmp_path)
    got = NR.v6_contract_problems(reg, raw, pin_raw=pin_raw, adapter_pin_sha256=adapters)
    assert codes(got) == case["expect"], got


def test_묶음_블록의_sha_는_로더_묶음_해시와_같다() -> None:
    """두 구현 대조의 이음새 — 같은 파일 목록이면 블록의 sha 가 `limits_loader_bundle_sha256` 과 같다(08 시험 12(가))."""
    block = NR.bundle_block(REPO, NR.LIMITS_LOADER_FILES)
    assert block["rule"] == NR.BUNDLE_RULE and [r["path"] for r in block["files"]] == sorted(NR.LIMITS_LOADER_FILES)
    assert block["sha256"] == NR.limits_loader_bundle_sha256(REPO) == rows_sha(block["files"])
    assert NR.bundle_block(REPO, tuple(reversed(NR.LIMITS_LOADER_FILES))) == block


def test_계약_작성기는_정규형을_내고_다시_적어도_같다(tmp_path: Path) -> None:
    write_world(tmp_path)
    meta_raw = dump_meta(world_meta(tmp_path))
    c = NR.narrative_inputs_block(tmp_path, limits_rel=W["limits_rel"], pairs_path=W["pairs"]["path"], snapshot_digest=snap(),
                                  pairs_meta_raw=meta_raw)
    raw = NR.dump_narrative_inputs(c)
    assert raw.endswith(b"\n") and b"\r" not in raw and NR.dump_narrative_inputs(json.loads(raw)) == raw
    assert c["pairs"]["pairs_meta_sha256"] == sha(meta_raw) and c["conditions"] == {"in_table": True}
    again = NR.narrative_inputs_block(tmp_path, limits_rel=W["limits_rel"], pairs_path=W["pairs"]["path"],
                                      snapshot_digest=snap(), pairs_meta_raw=meta_raw)
    assert NR.dump_narrative_inputs(again) == raw                         # 같은 바이트에서 같은 계약


def test_서술_블록_작성기는_계약_파일이_있을_때만_계약_sha_를_적는다(tmp_path: Path) -> None:
    contract = tmp_path / "narrative_inputs.json"
    contract.write_bytes(b'{"synthetic": true}\n')
    assert NR.narrative_block(REPO).inputs_contract_sha256 is None
    assert NR.narrative_block(REPO, inputs_contract_path=contract).inputs_contract_sha256 == sha(contract.read_bytes())
    reg = replace(v6(), narrative=NR.narrative_block(REPO), clause_axis=NR.clause_axis_block(REPO))
    assert "narrative.inputs_contract_sha256" in reg.missing()             # 빈 칸 — 등록이 완결되지 않는다
    assert reg.clause_axis.multi_clause == "primary_one_auxiliary_breakdown"


def test_채점_진입점은_계약_파일을_주면_판_6_칸과_맞댄다(tmp_path: Path) -> None:
    contract = tmp_path / "narrative_inputs.json"
    contract.write_bytes(b'{"synthetic": true}\n')
    full = NR.narrative_block(REPO, confirmation_path=contract, checker_vectors_path=contract, inputs_contract_path=contract)
    reg = replace(v6(), narrative=full, clause_axis=NR.clause_axis_block(REPO))
    assert NR.identity_problems(reg, inputs_contract_path=contract) == []
    other = tmp_path / "other.json"
    other.write_bytes(b'{"synthetic": false}\n')
    assert [p for p in NR.identity_problems(reg, inputs_contract_path=other)
            if p.startswith("narrative.inputs_contract_sha256")]


# ════════════════════════════════════════════════════════════ 판 6 연결 2 판 — B4 실물 해시(§3-1 · R12 ~ R14)

from tests import v6_link_world as VW


def _contract(root: Path) -> tuple[dict, bytes]:
    write_world(root)
    meta_raw = dump_meta(world_meta(root))
    c = NR.narrative_inputs_block(root, limits_rel=W["limits_rel"], pairs_path=W["pairs"]["path"], snapshot_digest=snap(),
                                  pairs_meta_raw=meta_raw)
    return c, NR.dump_narrative_inputs(c)


def _rebundle(contract: dict, name: str, rows: list[dict]) -> bytes:
    c = copy.deepcopy(contract)
    c[name] = {**c[name], "files": rows, "sha256": rows_sha(rows)}
    return NR.dump_narrative_inputs(c)


def test_v6_link_single_hash_not_bundle(tmp_path: Path) -> None:
    """R12 — 신판 등록의 generator_sha256 에 세 파일 가운데 하나의 LF sha 를 적으면 `hash_rule`(묶음과 단일 해시를 함께 받지 않는다).
    계약 쪽 묶음 칸에 한 파일 sha 를 적어도 같은 코드다."""
    root = tmp_path / "a"
    VW.copy_pinned(root)
    reg = VW.regime2_registration(root)
    bundle = NR.generator_bundle_identity(root)
    assert reg.narrative.generator_sha256 == bundle["sha256"] and bundle["rule"] == NR.BUNDLE_RULE
    assert [r["path"] for r in bundle["files"]] == sorted(NR.GENERATOR_FILES_V1) and bundle["sha256"] == rows_sha(bundle["files"])
    for row in bundle["files"]:
        bad = replace(reg, narrative=replace(reg.narrative, generator_sha256=row["sha256"]))
        got = NR.identity_problems(bad, repo=root, profile=NR.PROFILE_REGIME2)
        assert codes(got) == ["hash_rule"], got
    other = replace(reg, narrative=replace(reg.narrative, generator_sha256="e" * 64))
    assert codes(NR.identity_problems(other, repo=root, profile=NR.PROFILE_REGIME2)) == ["narrative.generator_sha256"]
    c, raw = _contract(tmp_path / "b")
    assert NR.contract_code_problems(raw, repo=tmp_path / "b") == []
    single = copy.deepcopy(c)
    single["generator"]["sha256"] = single["generator"]["files"][0]["sha256"]
    assert codes(NR.contract_code_problems(NR.dump_narrative_inputs(single), repo=tmp_path / "b")) == ["hash_rule"]


def test_v6_link_bundle_actual_files(tmp_path: Path) -> None:
    """R13 — 목록의 이름 변경 · 중복 · 대체는 문자열 계약이 맞아도 `contract_form`/`code_bytes`, 실제 파일의 변경 · 누락은 `code_bytes`.
    꼴 오류는 바이트 단계로 가지 않는다."""
    c, raw = _contract(tmp_path)
    assert NR.contract_code_problems(raw, repo=tmp_path) == []
    rows = c["generator"]["files"]
    renamed = sorted([{**rows[0], "path": "corpus/generate/make_pairs_main_v2.py"}, *rows[1:]], key=lambda r: r["path"])
    assert codes(NR.contract_code_problems(_rebundle(c, "generator", renamed), repo=tmp_path)) == ["code_bytes"]
    assert "contract_form" in codes(NR.contract_code_problems(_rebundle(c, "generator", [rows[0], *rows]), repo=tmp_path))
    swapped = sorted([rows[0], dict(c["limits_loader"]["files"][0]), rows[2]], key=lambda r: r["path"])
    assert "code_bytes" in codes(NR.contract_code_problems(_rebundle(c, "generator", swapped), repo=tmp_path))
    assert codes(NR.contract_code_problems(b"{}\n", repo=tmp_path)) == ["conditions", "contract_form"]
    assert codes(NR.contract_code_problems(b"[]\n", repo=tmp_path)) == ["contract_form"]
    with (tmp_path / NR.GENERATOR_FILES_V1[0]).open("ab") as fh:
        fh.write(b"\n# x\n")
    got = NR.contract_code_problems(raw, repo=tmp_path)
    assert codes(got) == ["code_bytes"] and any(NR.GENERATOR_FILES_V1[0] in p for p in got)
    (tmp_path / NR.GENERATOR_FILES_V1[0]).unlink()
    assert codes(NR.contract_code_problems(raw, repo=tmp_path)) == ["code_bytes"]


def test_v6_link_hash_rules_distinct(tmp_path: Path) -> None:
    """R14 — CRLF 표를 LF sha 로 적으면 `hash_rule`, 묶음을 JSON sha 로 적으면 `contract_form`, 정렬 · 구분자 · 끝 LF 변경은 정규형 밖.
    코드 파일의 CRLF ↔ LF 는 같은 LF sha(양성), 표의 raw 차이는 거부."""
    c, raw = _contract(tmp_path)
    table = tmp_path / W["limits_rel"]
    lf = table.read_bytes()
    table.write_bytes(lf.replace(b"\n", b"\r\n"))
    as_lf = copy.deepcopy(c)
    as_lf["limits"]["sha256"] = NR.lf_sha256(table)
    assert codes(NR.contract_code_problems(NR.dump_narrative_inputs(as_lf), repo=tmp_path)) == ["hash_rule"]
    as_raw = copy.deepcopy(c)
    as_raw["limits"]["sha256"] = NR.raw_sha256(table)
    assert NR.contract_code_problems(NR.dump_narrative_inputs(as_raw), repo=tmp_path) == []       # 표는 raw 규칙 — CRLF 표도 raw 로
    table.write_bytes(lf)
    assert codes(NR.contract_code_problems(NR.dump_narrative_inputs(as_raw), repo=tmp_path)) == ["code_bytes"]
    as_json = copy.deepcopy(c)
    as_json["generator"]["sha256"] = sha(json.dumps(as_json["generator"]["files"], sort_keys=True).encode("utf-8"))
    assert codes(NR.contract_code_problems(NR.dump_narrative_inputs(as_json), repo=tmp_path)) == ["contract_form"]
    rev = copy.deepcopy(c)
    rev["generator"]["files"] = list(reversed(rev["generator"]["files"]))
    assert "contract_form" in codes(NR.contract_code_problems(NR.dump_narrative_inputs(rev), repo=tmp_path))
    assert codes(NR.contract_code_problems(raw.rstrip(b"\n"), repo=tmp_path)) == ["contract_form"]
    assert codes(NR.contract_code_problems(raw.replace(b"\n", b"\r\n"), repo=tmp_path)) == ["contract_form"]
    assert codes(NR.contract_code_problems(raw.replace(b"\t", b"  ").replace(b" \"", b"\t\""), repo=tmp_path)) == ["contract_form"]
    gen0 = tmp_path / NR.GENERATOR_FILES_V1[0]
    gen0.write_bytes(gen0.read_bytes().replace(b"\n", b"\r\n"))
    assert NR.contract_code_problems(raw, repo=tmp_path) == []                                   # 코드 CRLF ↔ LF 등가
