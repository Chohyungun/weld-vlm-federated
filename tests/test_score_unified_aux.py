"""진입점의 2 차 보조 지표 연결(`scripts/probe/score_unified.py::_aux_scoring`) — 엄격 표본 모드의 합성 세계(`test_score_unified_strict.build`).

지키는 것.

1. 판 5 등록의 산출은 연결 전 진입점(커밋 `BEFORE` 의 blob)과 바이트가 같다(임시 경로만 갈라 맞댄다).
2. 판 6 등록 + 평가 등록 `auxiliary` 의 이름이면 서술 · 조항 축 블록을 싣고, 그 밖의 보고 칸은 판 5 와 같다.
3. 생성 줄에 서술을 덧붙여도 M1 쪽 칸의 바이트가 같다(03 2판 §5a) — 서술 블록만 달라진다.
4. 판 6 의 신원(검사기 · 목록 · 추출기 …)이 지금 코드나 검사 인자(`--aux-train-extractor` · `--aux-prompt` · `--aux-vectors`)의
   파일과 다르면 보조 지표만 빼고(`aux_identity` 에 사유) M1 산출은 그대로 낸다(지시 1023c 과제 2).
5. 판 5 등록이면 진입점이 보조 모듈을 불러오지도 않는다(203 §2).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.probe.score_unified as SU
from evaluation import narrative_registration as NR
from evaluation import strict_narrative as SN
from evaluation.narrative_conformance import VECTOR_REL
from evaluation.prereg_unified import (
    CLAUSE_AXIS_METRICS,
    REGISTRATION_VERSION,
    REGISTRATION_VERSION_V6,
)
from tests.test_prereg_v6 import axis_spec, narrative_spec
from tests.test_score_unified_strict import NOW, _gold_rows, argv, build, out_path
from tests.test_strict_narrative import old_module

REPO = Path(__file__).resolve().parents[1]
ALL_NAMES = (*SN.NARRATIVE_NAMES, *CLAUSE_AXIS_METRICS)
SUFFIX = "\n두께 정보가 없어 합부 판정은 내리지 않는다."
NEW_KEYS = ("narrative", "clause_axis")


def fake_view(root, ids, *, receipt):
    return pd.DataFrame(_gold_rows(list(ids)), columns=["image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px",
                                                        "bbox_y2_px"])


@pytest.fixture()
def views(monkeypatch):
    monkeypatch.setattr(SU, "EVAL_VIEW", fake_view)
    return monkeypatch


def to_v6(r, **narrative_over) -> None:
    act = NR.actual_identity(REPO)
    r.version = REGISTRATION_VERSION_V6
    r.narrative = narrative_spec(**{**act, **narrative_over})
    r.clause_axis = axis_spec(limits_sha256=act["limits_sha256"])


def with_names(doc) -> None:
    doc["auxiliary"] = [*doc["auxiliary"], *ALL_NAMES]


def world(tmp: Path, name: str, **kw):
    d = tmp / name
    d.mkdir()
    return build(d, **kw)


def run(mod, w, extra=()) -> int:
    """판 6 연결 뒤 — 이 세계의 판 6 은 구판(v3.1) 블록이라 보존 기준 커밋의 같은 바이트여야 옛 경로로 읽힌다. 그 커밋을 이 세계의 등록
    커밋으로 바꿔 끼운다(역사 blob 대조만 임시 저장소로 — 판 6 연결 2 판 §2-1). 옛 모듈(연결 전)에는 그 상수가 없어 그대로다."""
    from evaluation import eval_registration as ER

    prev = ER.HISTORY_BASE_COMMIT
    ER.HISTORY_BASE_COMMIT = w["head"]
    try:
        return mod.main([*argv(w), *extra], _seam=mod.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)
    finally:
        ER.HISTORY_BASE_COMMIT = prev


def body(w) -> dict:
    return json.loads(out_path(w).read_text(encoding="utf-8"))


def norm(raw: bytes, root: Path) -> bytes:
    """산출에 실린 임시 경로를 한 이름으로 — 두 세계의 다른 점은 자리뿐이어야 한다."""
    s = raw.decode("utf-8")
    for form in (str(root), json.dumps(str(root))[1:-1], root.as_posix()):
        s = s.replace(form, "<world>")
    return s.encode("utf-8")


def test_판_5_산출은_연결_전_진입점과_바이트가_같다(tmp_path: Path, views, capsys) -> None:
    for var in ("GIT_AUTHOR_DATE", "GIT_COMMITTER_DATE"):    # 두 세계의 커밋 시각을 고정 — 초가 바뀌면 head 해시가 갈린다
        views.setenv(var, "2026-09-21T09:00:00+09:00")
    old = old_module(tmp_path, "scripts/probe/score_unified.py", "score_unified_before_aux")
    views.setattr(old, "EVAL_VIEW", fake_view)
    w_new, w_old = world(tmp_path, "new"), world(tmp_path, "old")
    assert run(SU, w_new) == SU.EXIT_OK, capsys.readouterr().err
    assert run(old, w_old) == old.EXIT_OK, capsys.readouterr().err
    a = norm(out_path(w_new).read_bytes(), tmp_path / "new")
    b = norm(out_path(w_old).read_bytes(), tmp_path / "old")
    assert a == b
    assert not any(k in body(w_new)["report"] for k in NEW_KEYS)


def test_판_6_은_두_블록을_싣고_그_밖의_보고_칸은_판_5_와_같다(tmp_path: Path, views, capsys) -> None:
    w5 = world(tmp_path, "v5", doc_edit=with_names)
    w6 = world(tmp_path, "v6", doc_edit=with_names, reg_edit=to_v6)
    assert run(SU, w5) == SU.EXIT_OK and run(SU, w6) == SU.EXIT_OK, capsys.readouterr().err
    r5, r6 = body(w5)["report"], body(w6)["report"]
    assert set(r6) - set(r5) == set(NEW_KEYS)
    for key in r5:
        if key != "not_computed":
            assert r6[key] == r5[key], key
    assert set(r5["not_computed"]) - set(r6["not_computed"]) == set(ALL_NAMES)
    nar = r6["narrative"]
    assert set(nar["models"]) == {f"{t}_s1" for t in ("uni_central", "uni_fed", "uni_local_C1", "uni_local_C2",
                                                     "uni_local_C3")}
    assert nar["checker"]["checker_sha256"] == NR.actual_identity(REPO)["checker_sha256"]
    assert nar["models"]["uni_fed_s1"]["state.missing"]["unweighted"]["value"] == 1.0          # 서술이 없는 생성 줄
    assert set(nar["paired"]) == {f"uni_fed_s1 − {t}_s1" for t in ("uni_central", "uni_local_C1", "uni_local_C2",
                                                                     "uni_local_C3")}
    paired = r6["clause_axis"]["paired"]                           # 모델 − 대조선의 대응 차(같은 W · 지시 1023d)
    assert set(paired) == {f"{t}_s1 − 대조선" for t in ("uni_central", "uni_fed", "uni_local_C1", "uni_local_C2", "uni_local_C3")} | {
        f"uni_fed_s1 − {t}_s1" for t in ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3")}   # 서술 지표의 짝도
    assert all(set(v) == {"design", "unweighted"} for p in paired.values() for v in p.values())
    assert r6["clause_axis"]["draws"]["shared_with_m1"] is True and "같은 W" in r6["clause_axis"]["interval"]
    ca = r6["clause_axis"]["models"]["uni_fed_s1"]
    assert set(ca) == {"design", "unweighted"}                    # 대표 = 설계 가중 · 부표 = 무가중(04 §7-10 ①)
    for view in ca.values():
        assert view["ungrounded_rate"]["value"] == 1.0 and view["n_citations"] == ca["unweighted"]["n_citations"]
    assert ca["unweighted"]["ungrounded_rate"]["nonexistent"] == ca["unweighted"]["n_citations"]


def test_생성_줄에_서술을_덧붙여도_M1_쪽_칸의_바이트가_같다(tmp_path: Path, views, capsys) -> None:
    plain = world(tmp_path, "plain", doc_edit=with_names, reg_edit=to_v6)
    told = world(tmp_path, "told", doc_edit=with_names, reg_edit=to_v6, text_suffix=SUFFIX)
    assert run(SU, plain) == SU.EXIT_OK and run(SU, told) == SU.EXIT_OK, capsys.readouterr().err
    rp, rt = body(plain)["report"], body(told)["report"]
    for key in rp:
        if key != "narrative":
            assert json.dumps(rt[key], sort_keys=True) == json.dumps(rp[key], sort_keys=True), key
    fed = rt["narrative"]["models"]["uni_fed_s1"]
    assert fed["state.complete"]["unweighted"]["value"] == 1.0
    assert fed["reservation.r_dim"]["unweighted"]["state"] in ("finite", "해당 없음")
    assert rt["narrative"] != rp["narrative"]


@pytest.mark.parametrize("field", ["checker_sha256", "lists_sha256", "extractor_sha256"])
def test_판_6_신원이_지금_코드와_다르면_보조_지표만_빼고_사유를_남긴다(tmp_path: Path, views, capsys, field: str) -> None:
    w5 = world(tmp_path, "v5", doc_edit=with_names)
    w = world(tmp_path, "bad", doc_edit=with_names, reg_edit=lambda r: to_v6(r, **{field: "0" * 64}))
    assert run(SU, w5) == SU.EXIT_OK and run(SU, w) == SU.EXIT_OK
    err = capsys.readouterr().err
    assert "보조 지표를 내지 않는다" in err and f"narrative.{field}" in err
    r5, rb = body(w5)["report"], body(w)["report"]
    assert set(rb) - set(r5) == {"aux_identity"} and not any(k in rb for k in NEW_KEYS)
    for key in r5:
        if key != "not_computed":
            assert rb[key] == r5[key], key                                   # M1 · 대조선 · 회복률 칸은 그대로다
    ident = rb["aux_identity"]
    assert ident["status"] == "refused" and any(p.startswith(f"narrative.{field}") for p in ident["problems"])
    assert ident["checks"] == {"code_tree": "checked", "train_extractor": "not_given", "prompt": "not_given",
                               "vectors": "not_given"}
    assert {n: rb["not_computed"][n] for n in ALL_NAMES} == dict.fromkeys(ALL_NAMES, SN.REFUSED)


TRAIN = REPO / NR.TRAIN_EXTRACTOR_REL


def train_ok(r) -> None:
    to_v6(r, train_extractor_sha256=NR.lf_sha256(TRAIN))


def test_검사_인자의_파일이_등록과_같으면_내고_무엇을_맞댔는지_싣는다(tmp_path: Path, views, capsys) -> None:
    w = world(tmp_path, "ok", doc_edit=with_names, reg_edit=train_ok)
    assert run(SU, w, ["--aux-train-extractor", str(TRAIN), "--aux-vectors", str(REPO / VECTOR_REL)]) == SU.EXIT_OK, \
        capsys.readouterr().err
    rep = body(w)["report"]
    assert "aux_identity" not in rep and all(k in rep for k in NEW_KEYS)
    assert rep["narrative"]["identity_checks"] == {"code_tree": "checked", "train_extractor": "checked",
                                                   "prompt": "not_given", "vectors": "checked"}


def given_file(kind: str) -> bytes:
    if kind == "train_sha":
        return b'EXTRACTOR_ID = "vlm_narrative_v1"\n'
    if kind == "train_id":                                    # 등록 sha 는 이 사본에 맞추고 식별자만 바꾼다
        src = TRAIN.read_bytes().replace(b'EXTRACTOR_ID = "vlm_narrative_v1"', b'EXTRACTOR_ID = "vlm_narrative_v9"')
        assert src != TRAIN.read_bytes()
        return src
    if kind == "prompt":
        return "가상 프롬프트 — 등록과 다르다\n".encode()
    return (REPO / VECTOR_REL).read_bytes() + b" "


@pytest.mark.parametrize("kind,flag,field", [
    ("train_sha", "--aux-train-extractor", "narrative.train_extractor_sha256"),
    ("train_id", "--aux-train-extractor", "narrative.train_extractor_id"),
    ("prompt", "--aux-prompt", "generation.prompt_sha256"),
    ("vectors", "--aux-vectors", "narrative.extract_vectors_sha256"),
], ids=["train_sha", "train_id", "prompt", "vectors"])
def test_검사_인자의_파일이_등록과_다르면_보조_지표를_내지_않는다(tmp_path: Path, views, capsys, kind, flag, field) -> None:
    """학습 쪽 추출기(식별자 · LF sha256) · 프롬프트 · 확인 파일(정본 추출 벡터의 실제 바이트) — 하나라도 다르면 그 칸의 사유를 싣는다."""
    path = tmp_path / "given.bin"
    path.write_bytes(given_file(kind))

    def reg_edit(r) -> None:
        to_v6(r, train_extractor_sha256=NR.lf_sha256(path if kind == "train_id" else TRAIN))

    w = world(tmp_path, "bad", doc_edit=with_names, reg_edit=reg_edit)
    assert run(SU, w, [flag, str(path)]) == SU.EXIT_OK, capsys.readouterr().err
    ident = body(w)["report"]["aux_identity"]
    assert [p for p in ident["problems"] if p.startswith(field)], ident["problems"]
    assert ident["checks"][{"--aux-train-extractor": "train_extractor", "--aux-prompt": "prompt",
                            "--aux-vectors": "vectors"}[flag]] == "checked"


def test_판_5_는_보조_모듈을_불러오지_않는다(monkeypatch) -> None:
    """203 §2 — 판 5 등록이면 판 검사 뒤의 새 모듈 import 에 닿지 않는다. 새 모듈을 막아 두고 판 5 는 None, 판 6 은 import 에서 멈춘다."""
    for name in ("evaluation.strict_narrative", "evaluation.narrative_checker", "evaluation.narrative_registration",
                 "evaluation.clause_axis", "corpus.rules.limits_loader"):
        parent, _, child = name.rpartition(".")
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delattr(importlib.import_module(parent), child, raising=False)
    names = {"auxiliary": list(ALL_NAMES)}
    v5 = SimpleNamespace(reg=SimpleNamespace(version=REGISTRATION_VERSION), eval_doc=names, args=argparse.Namespace())
    assert SU._aux_scoring(v5, [], {}, [], {}, {}, [], []) is None
    v6 = SimpleNamespace(reg=SimpleNamespace(version=REGISTRATION_VERSION_V6), eval_doc=names, args=argparse.Namespace())
    with pytest.raises(ImportError):
        SU._aux_scoring(v6, [], {}, [], {}, {}, [], [])


def test_이름이_없으면_판_6_이어도_내지_않는다(tmp_path: Path, views, capsys) -> None:
    w = world(tmp_path, "noname", reg_edit=to_v6)
    assert run(SU, w) == SU.EXIT_OK, capsys.readouterr().err
    assert not any(k in body(w)["report"] for k in NEW_KEYS)


def test_고친_등록은_v6_의_빈_칸이_없다() -> None:
    from tests.test_prereg_unified import filled

    r = replace(filled(purpose="main"))
    to_v6(r)
    assert r.narrative is not None and not [m for m in r.missing() if m.startswith(("narrative", "clause_axis"))]


# ---------------------------------------------------------------- 판 6 연결 2 판 — 실제 검사기 선택 · 완료 표지(§3-2 · §5 · R9 · R19)
from evaluation import narrative_checker_v4 as NC4
from tests.test_regime2_end_to_end import (
    AUX_FILES,
    _aux,
    _exported,
    _no_cuda,
    approved,
    e2e,
)

FIXTURES = (e2e, approved, _no_cuda)
"""가져온 fixture — pytest 는 모듈 이름 공간의 fixture 를 이름으로 모은다."""


def test_v6_link_runtime_checker_matches_profile(e2e, tmp_path, monkeypatch):
    """R9 — 등록은 v4 인데 적재 모듈의 상수가 핀과 다른 대역이면 `profile_runtime` 이 `AuxRefused.problems` 에 들고 M1 은 보존된다
    (보조 지표만 뺀다). 신판이면 실제 검사기 · 표 · 신원이 같은 프로필이다."""
    e = e2e
    _reg, rp, folder = _exported(e, tmp_path)
    aux, _verified, rejected = _aux(e, rp, folder, monkeypatch, **AUX_FILES)
    assert rejected == {} and isinstance(aux, SN.AuxScoring)
    assert aux.checker_identity["checker_id"] == NR.CHECKER_PINS[1].checker_id and type(aux.checker).__module__ == NR.CHECKER_MODULES["regime2_v4"]
    monkeypatch.setattr(NC4, "LIMITS_SHA256", "0" * 64)
    bad, _verified, rejected = _aux(e, rp, folder, monkeypatch, **AUX_FILES)
    assert rejected == {} and isinstance(bad, SN.AuxRefused)
    assert bad.problems and bad.problems[0].startswith("profile_runtime:")
    assert bad.computed_names() == [] and set(bad.not_computed_reasons()) == set(ALL_NAMES)


def test_v6_link_aux_required_for_completion(e2e, tmp_path, monkeypatch):
    """R19 — 작성 단계의 보조 이름은 `aux_names`(도출 함수), 계산 단계는 `aux_not_computed` · `aux_identity_refused` · `aux_incomplete` 로
    `partial` 이 되고 M1 은 보존된다. 신판의 보조 입력은 다섯 이름을 원한다."""
    from evaluation.eval_registration import AUXILIARY_V6, regime2_eval_problems

    e = e2e
    _reg, rp, folder = _exported(e, tmp_path)
    aux, _v, _r = _aux(e, rp, folder, monkeypatch, **AUX_FILES)
    assert set(AUXILIARY_V6) <= set(aux.computed_names()) and set(AUXILIARY_V6) == set(ALL_NAMES)
    cells = [("uni_local_C1", 1)]
    report = {"narrative": {"models": {"uni_local_C1_s1": {}}, "paired": {"x": 1}},
              "clause_axis": {"models": {"uni_local_C1_s1": {}}, "paired": {"x": 1}}}
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=aux, report=report)["complete"]
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=None, report=report)["reasons"] == ["aux_not_computed"]
    altered = tmp_path / "narrative_altered.py"
    altered.write_bytes(AUX_FILES["train_extractor"].read_bytes() + b"\n# x\n")
    refused, _v, _r = _aux(e, rp, folder, monkeypatch, **{**AUX_FILES, "train_extractor": altered})
    assert isinstance(refused, SN.AuxRefused)
    assert SU.v6_completion(profile_id="regime2_v4", required_cells=cells, verified_cells=cells, aux=refused, report=report)["reasons"] == ["aux_identity_refused"]
    doc = json.loads(json.dumps(e.doc))
    doc["auxiliary"] = doc["auxiliary"][:-1]
    import yaml

    from vlm.registration import registration_from_bytes

    inputs = {"source_doc": yaml.safe_load(e.source_raw), "base_doc": yaml.safe_load(e.cfg_path.read_bytes()),
              "train_doc": registration_from_bytes(e.reg_path.read_bytes()), "prompt_raw": (e.w.repo / "vlm/prompts/unified_v3_1_basis.txt").read_bytes()}
    assert {p.split(":", 1)[0] for p in regime2_eval_problems(doc, **inputs)} == {"aux_names"}
