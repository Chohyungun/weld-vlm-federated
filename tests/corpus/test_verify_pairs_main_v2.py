"""2 판 빌드의 산출물 검사 · 봉인 관문의 판 해석(C 08 2 판 §5-1 · §7 의 시험 11 · 15 · 20).

산출물 검사는 메타의 `limits_table.path` 로 판 등록부에서 판을 찾고, 그 판의 표 · 서술로 레코드를 다시 조립해 맞댄다. 2 판 메타는 신원
블록(`identity`)과 문장 계약 판(`text_contract`)이 실제 파일에서 다시 낸 값과 같아야 하고, 1 판 메타에는 그 칸이 없어야 한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from corpus.generate import make_pairs_main as MM
from corpus.generate import make_pairs_pilot as M
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules import limits_loader
from corpus.rules.skeleton_gen import load_defect_lexicon
from corpus.validate import seal_pairs_main as S
from corpus.validate import verify_pairs_main as V
from data.manifest_io import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, load_snapshot, write_snapshot
from tests.corpus.test_verify_pairs_main import ANNS, CAPS, ROWS

REPO = Path(__file__).resolve().parents[2]
NAMES = defect_names()
ED2 = MM.EDITIONS["v2"]
TABLE_V2 = limits_loader.load_limits(str(ED2.limits_csv), pilot=True)


def _build(tmp_path, edition="v2"):
    ed = MM.EDITIONS[edition]
    table = limits_loader.load_limits(str(ed.limits_csv), pilot=True)
    snap, out = tmp_path / "snap", tmp_path / "build"
    digest = write_snapshot(snap, pd.DataFrame(ROWS, columns=list(MANIFEST_COLUMNS)),
                            pd.DataFrame(ANNS, columns=list(ANNOTATION_COLUMNS)), CAPS)
    snapshot = load_snapshot(snap)
    for rel in snapshot.manifest["rel_path"]:
        f = tmp_path / str(rel).replace(chr(92), "/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    res = M.build_pairs(snapshot.manifest, snapshot.annotations, table, NAMES, load_defect_lexicon(),
                        root=tmp_path, check_paths=True, edition=ed.version)
    meta = MM.build_meta(res, snapshot.manifest, manifest_digest=digest, date="2026-10-24", edition=edition)
    M.write_outputs(out, res, asset="d4_pairs_test", version=ed.version, date="2026-10-24",
                    limits_csv=ed.limits_csv, manifest_csv=snap / "manifest.csv", meta=meta)
    return out, snap, tmp_path


@pytest.fixture()
def built_v2(tmp_path):
    return _build(tmp_path)


def _edit_meta(out: Path, fn) -> None:
    p = out / "PAIRS_META.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    fn(meta)
    M.write_json(p, meta)
    M.snapshot_digest(out, V.MEMBERS)               # 계약서를 다시 써서 메타의 내용 결함만 남긴다


def test_2판_빌드는_2판으로_읽혀_전_축이_맞는다(built_v2):
    out, snap, root = built_v2
    rep = V.verify(out, snap, root)
    assert rep["failures"] == {} and rep["coverage"] == "full" and rep["edition"] == "v2"
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    assert meta["text_contract"] == "v2" and meta["identity"]["conditions"] == {"in_table": True}
    assert meta["identity"]["limits"] == {k: meta["limits_table"][k] for k in ("path", "sha256")}
    assert meta["limits_table"]["status"] == MM.LIMITS_STATUS_V2 and meta["asset"].endswith("_v2")
    keys = list(meta)
    assert keys.index("identity") == keys.index("limits_table") + 1 == keys.index("text_contract") - 1


def test_1판_빌드의_메타_키는_그대로이고_신원_칸이_없다(tmp_path):
    out, snap, root = _build(tmp_path, "v1")
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    assert "identity" not in meta and "text_contract" not in meta
    assert meta["limits_table"]["path"] == "corpus/rules/limits_v0_pilot.csv"
    assert V.verify(out, snap, root)["failures"] == {}


def _kinds(out, snap, root) -> set[str]:
    return set(V.verify(out, snap, root)["failures"])


@pytest.mark.parametrize("label, fn, kind", [
    ("status 없음", lambda m: m["limits_table"].pop("status"), "H_meta_limits_status_missing"),
    ("문장 계약 판", lambda m: m.update(text_contract="v1"), "H_meta_text_contract"),
    ("신원 칸 빠짐", lambda m: m["identity"].pop("conditions"), "H_meta_identity_form"),
    ("신원 모르는 칸", lambda m: m["identity"].update(extra=1), "H_meta_identity_form"),
    ("계약 판 참거짓", lambda m: m["identity"].update(contract_version=True), "H_meta_identity_version"),
    ("표 경로 어긋남", lambda m: m["identity"]["limits"].update(path="corpus/rules/limits_v2_pilot_copy.csv"),
     "H_meta_identity_limits"),
    ("표 두 칸 밖 칸", lambda m: m["identity"].update(limits=dict(m["limits_table"])), "H_meta_identity_limits"),
    ("묶음 목록", lambda m: m["identity"]["generator"]["files"].pop(0), "H_meta_identity_bundle"),
    ("묶음 해시", lambda m: m["identity"]["limits_loader"]["files"][0].update(sha256="0" * 64), "H_meta_identity_bundle"),
    ("조건 표지", lambda m: m["identity"].update(conditions={"in_table": False}), "H_meta_identity_conditions"),
])
def test_2판_메타의_신원이_어긋나면_그_사유다(built_v2, label, fn, kind):
    out, snap, root = built_v2
    _edit_meta(out, fn)
    assert kind in _kinds(out, snap, root), label


def test_1판_메타에_2판_신원_칸이_있으면_거부한다(tmp_path):
    out, snap, root = _build(tmp_path, "v1")
    _edit_meta(out, lambda m: m.update(text_contract="v2"))
    assert "H_meta_identity_unexpected" in _kinds(out, snap, root)


def test_판_등록부에_없는_표를_가리키면_거부한다(built_v2):
    out, snap, root = built_v2
    _edit_meta(out, lambda m: m["limits_table"].update(path="corpus/rules/limits_v9_pilot.csv"))
    assert "H_meta_edition_unknown" in _kinds(out, snap, root)


def test_2판_레코드에_1판_표를_가리키는_메타는_레코드부터_어긋난다(built_v2):
    """판이 어긋난 빌드 — 메타만 1 판으로 바꾸면 그 판의 표 · 서술로 다시 조립한 레코드와 다르다."""
    out, snap, root = built_v2

    def to_v1(m):
        m.pop("identity")
        m.pop("text_contract")
        m["limits_table"] = {"path": MM.EDITIONS["v1"].limits_rel,
                             "sha256": hashlib.sha256(MM.EDITIONS["v1"].limits_csv.read_bytes()).hexdigest(),
                             "status": MM.LIMITS_STATUS}
    _edit_meta(out, to_v1)
    kinds = _kinds(out, snap, root)
    assert {"E_record_differs_from_source", "A_gate_recheck", "F_counts_snapshot_hash"} <= kinds


def test_봉인_관문은_빌드_메타의_판으로_독립_대조기에_표를_넘긴다(built_v2):
    out, _snap, _root = built_v2
    assert S.limits_for_build(out) == ("v2", ED2.limits_csv, None)
    _edit_meta(out, lambda m: m["limits_table"].update(path="corpus/rules/limits_v9_pilot.csv"))
    name, path, why = S.limits_for_build(out)
    assert (name, path) == (None, None) and "판 등록부에 없다" in why


# ---------------------------------------------------------------- 시험 20 — I-02 공동 사례(평가 쪽 사례와 같은 입력)

def _cases() -> dict | None:
    rel = "evaluation/vectors/narrative_inputs_cases.v1.json"
    p = REPO / rel
    raw = p.read_bytes() if p.is_file() else None
    if raw is None:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"91e83979:{rel}"], capture_output=True, check=False)
        raw = out.stdout if out.returncode == 0 else None
    if raw is None:
        return None
    assert hashlib.sha256(raw).hexdigest() == "4d06a076fc52fed0f96408d0b7923eed4dfe12e3ac3d46bed18af15b09f764b5"
    return json.loads(raw)


def test_I02_공동_사례는_평가_쪽과_같은_입력에서_같은_판정이다():
    """`limits_table` 의 기존 `status` 를 보존한다. `identity.limits` 는 `{path, sha256}` 의 닫힌 객체이며, 기존 `limits_table` 에서
    그 두 필드만 추출한 값과 같아야 한다 — 생성 쪽 판정(`identity_limits_problem`)이 사례의 기대(받음 · 거부)와 같다."""
    doc = _cases()
    if doc is None:
        pytest.skip("평가 쪽 사례 파일을 읽지 못했다")
    joint = [c for c in doc["identity_cases"] if c.get("joint_i02")]
    assert len(joint) == 4
    for case in joint:
        got = V.identity_limits_problem(copy.deepcopy(case["pairs_meta"]))
        assert (got is None) == (case["expect"] == []), case["id"]
