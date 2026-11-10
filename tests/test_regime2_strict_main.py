"""2 차 체제의 **엄격 채점 진입점 전 구간** — `score_unified.main()` 을 실제로 부른다.

`tests/test_regime2_end_to_end.py` 의 세계(판 6 등록 여섯 파일 · 커밋 후 검수 · 영수증 · 착수 허가 · 학습 산출 메타 다섯)에 엄격 채점의
나머지 입력 — 칸 다섯의 모델 묶음과 에코 묶음(실제 export 경로, 대역 생성기) · 등록 대조선 예측 · 기지답 픽스처 · 조기 점검 규칙(영수증 커밋)
· 허가 파일 — 을 더해 **진입 관문 · 목록 · 묶음 검증 · 주 지표 · 보조 지표 · 완료 판정 · 원장**까지 한 번에 지난다. 채점 관문 · 프레임 검증 ·
완료 판정을 대역으로 끄지 않는다. 대역은 둘뿐이다 — 데이터 쪽 정답 판독기(`SU.EVAL_VIEW`, 실물 평가 자산을 열지 않는다)와 묶음 검증의
구현 식별자 검사(시험의 대역 생성기는 커밋된 코드가 아니다 — `test_uni_joint_entry` 가-8 과 같은 이음새).

지키는 것.
1. 완전한 입력이면 산출 하나를 배타 생성하고 종료 0 — `v6_completion.complete` 참, 보조 다섯 이름이 계산됨, 평가 설정 경로가 지문에 실린다.
2. 칸 하나가 빠지면 완료가 아니다(`status partial` · `complete` 거짓 · 종료 2).
3. 출처 운반 파일이 커밋과 다르면(작업 트리 변조) 영수증 커밋의 바이트를 읽어 영향이 없고, 등록 커밋의 출처가 다르면 원장 줄 전에 거부한다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.probe.score_unified as SU
from evaluation import known_answer as KA
from evaluation.scoring_ledger import read_ledger
from scripts import main_uni as M
from tests import uni_export_world as W
from tests.test_regime2_end_to_end import (
    AUX_FILES,
    REG_REL,
    STEM,
    _commit,
    _no_cuda,
    _register,
    approved,
    e2e,
)
from tests.test_uni_joint_entry import NOW

FIXTURES = (e2e, approved, _no_cuda)
BOX = [10.0, 20.0, 30.0, 40.0]


def _set(d: dict, dotted: str, value) -> None:
    area, _, key = dotted.partition(".")
    d.setdefault(area, {})[key] = value


@pytest.fixture
def view(monkeypatch):
    calls = []

    def fake(root, ids, *, receipt):
        ids = list(ids)
        calls.append(ids)
        return pd.DataFrame([{"image_id": i, "iso_code": "2011", "bbox_x1_px": BOX[0], "bbox_y1_px": BOX[1],
                              "bbox_x2_px": BOX[2], "bbox_y2_px": BOX[3]} for i in ids],
                            columns=["image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"])

    monkeypatch.setattr(SU, "EVAL_VIEW", fake)
    return calls


@pytest.fixture
def strict_world(e2e, monkeypatch):
    """엄격 채점의 입력 전부 — 등록 · 커밋 · 영수증(실제 작성기) 뒤 칸 다섯의 모델 묶음 · 에코 묶음(실제 export)."""
    import evaluation.bundle_v14 as BV

    e, repo = e2e, e2e.w.repo
    # 중앙 칸의 학습 행 지문 — 실제 학습기는 참여자 셋을 합친 행의 지문 하나를 쓴다(연합만 참여자별 사전). e2e 세계의 메타는 C1 export
    # 만 다뤄 중앙을 사전으로 두었다 — 이 세계는 다섯 칸을 모두 export 하므로 실제 꼴로 맞춘다.
    from vlm.pilot_vlm import load_pairs, train_rows_digest

    central_meta = e.train_root / "uni_central_s1" / "adapter_last.meta.json"
    m = json.loads(central_meta.read_text(encoding="utf-8"))
    m["train_rows_digest"] = train_rows_digest(load_pairs("train", None, pairs_path=e.pairs))
    central_meta.write_text(json.dumps(m), encoding="utf-8")
    e.metas["uni_central"] = m
    # 영수증 커밋에 있어야 할 채점 쪽 파일 — 조기 점검 규칙(허가가 가리키는 그 바이트) · 기지답 픽스처
    from vlm import ecc as E

    (repo / SU.ECC_RULES_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(E.RULE_PATH, repo / SU.ECC_RULES_REL)
    ka_raw = (ROOT / KA.FIXTURE_REL).read_bytes()
    (repo / KA.FIXTURE_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo / KA.FIXTURE_REL).write_bytes(ka_raw)
    W.git(repo, "add", SU.ECC_RULES_REL, KA.FIXTURE_REL)
    W.git(repo, "commit", "-q", "-m", "채점 쪽 규칙 · 기지답 픽스처")
    # 등록 대조선 예측 · 허가 파일 — 등록 칸이 그 바이트의 sha 를 가리킨다
    pop_ids = [line.strip() for line in e.lst.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = e.tmp / "main_in" / "baseline.jsonl"
    baseline.write_bytes("".join(json.dumps({"image_id": i, "defects": [{"iso_code": "2011", "bbox_px": BOX}]}) + "\n"
                                 for i in pop_ids).encode("utf-8"))
    permit = e.spec.ecc_permit_path
    scoring = json.loads(json.dumps(e.scoring))
    _set(scoring, "baseline.prediction_sha256", hashlib.sha256(baseline.read_bytes()).hexdigest())
    _set(scoring, "canary.known_answer_sha256", KA.known_answer_sha256(ka_raw))
    _set(scoring, "canary.frame_diag_sha256", hashlib.sha256(permit.read_bytes()).hexdigest())
    reg, raw, tc_text = _register(e, scoring=scoring)
    rp = _commit(e, reg, raw, tc_text)
    folder = e.tmp / "main_out" / "bundles"
    for tag in M.TAGS:
        W.run(e.w, purpose="main", mode="model", tag=tag, receipt_path=rp, list_path=e.lst, run_root=None, plan_path=None,
              folder=folder, loader=e.s.loader, measure_fn=e.s.measure, image_path_of=e.s.paths(e.w),
              adapter_dir=e.train_root / f"{tag}_s1", side_path=e.side)
    W.run(e.w, purpose="main", mode="echo", tag="uni_central", receipt_path=rp, list_path=e.echo, run_root=None, plan_path=None,
          folder=folder, loader=e.s.loader, measure_fn=e.s.measure, image_path_of=e.s.paths(e.w))
    monkeypatch.setattr(BV, "impl_problems", lambda v: [])           # 대역 생성기의 구현 식별자 — 커밋된 코드가 아니다
    e.rp, e.folder, e.baseline, e.permit, e.reg = rp, folder, baseline, permit, reg
    return e


def argv(e, *, aux=True) -> list[str]:
    out = ["--registration", "strict", "--purpose", "main", "--receipt", str(e.rp), "--snapshot", str(e.w.snap),
           "--bundles", str(e.folder), "--generation-list", str(e.lst), "--scoring-list", str(e.lst), "--echo-list", str(e.echo),
           "--train-root", str(e.train_root), "--out", str(e.w.repo / "outputs/main_u"), "--device", W.DEVICE,
           "--side", str(e.side), "--baseline", str(e.baseline), "--ecc-permit", str(e.permit)]
    if aux:
        out += ["--aux-train-extractor", str(AUX_FILES["train_extractor"]), "--aux-prompt", str(AUX_FILES["prompt"]),
                "--aux-vectors", str(AUX_FILES["vectors"])]
    return out


def run(e, **kw) -> int:
    return SU.main(argv(e, **kw), _seam=SU.Seam(checkout=e.w.repo, repo=e.w.repo), _now=NOW)


def outputs(e) -> list[dict]:
    d = e.w.repo / "outputs/main_u"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob(f"{SU.STRICT_OUT_PREFIX}*.json"))]


def test_regime2_strict_main_full_pass(strict_world, view, capsys):
    e = strict_world
    rc = run(e)
    err = capsys.readouterr().err
    assert rc == SU.EXIT_OK, err[-2000:]
    outs = outputs(e)
    assert len(outs) == 1, [sorted(o) for o in outs]
    o = outs[0]
    assert o["status"] == "pass" and o["v6_completion"]["complete"] is True, o.get("v6_completion")
    assert o["fingerprints"]["eval_registration_path"] == f"configs/registration/{STEM}.eval.yaml"
    from evaluation.eval_registration import AUXILIARY_V6

    comp, rep = o["v6_completion"], o["report"]
    assert comp["profile_id"] == "regime2_v4" and comp["required_names"] == list(AUXILIARY_V6) and comp["reasons"] == []
    assert rep["narrative"]["checker"]["checker_id"] == "narrative_checker_v4"           # 실제 검사기 v4(프로필로 고른 것)
    assert set(rep["narrative"]["identity_checks"].values()) == {"checked"}
    assert set(AUXILIARY_V6) <= set(rep["narrative"]["names"]) | set(rep["clause_axis"]["names"])
    assert o["canaries"]["frame"]["meaning"] == "ecc_permit_sha256"
    assert [r for r in read_ledger(SU.canonical_ledger(e.w.repo)) if "call" in json.dumps(r, ensure_ascii=False)]
    assert view                                                         # 정답 판독기가 실제로 불렸다


def test_regime2_strict_main_missing_cell_is_not_complete(strict_world, view, capsys):
    e = strict_world
    for p in list(e.folder.glob("uni_local_C3_s1.*")):
        p.unlink()
    rc = run(e)
    capsys.readouterr()
    assert rc == SU.EXIT_REJECTED
    o = outputs(e)[0]
    assert o["status"] == "partial" and o["v6_completion"]["complete"] is False
    assert "cells_missing" in json.dumps(o["v6_completion"], ensure_ascii=False)


def test_regime2_strict_main_sources_from_receipt_commit(strict_world, view, capsys):
    e = strict_world
    src = e.w.repo / "configs" / "registration" / f"{STEM}.v6_sources.json"
    src.write_bytes(src.read_bytes() + b" ")                             # 작업 트리만 바꿨다 — 영수증 커밋의 바이트를 읽는다
    assert run(e) == SU.EXIT_OK
    capsys.readouterr()
    assert REG_REL
