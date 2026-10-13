"""조기 좌표 점검(`early_coord_check`)의 export 가 평가 쪽 영수증 · 등록 · 분할 표를 지나 **끝까지** 간다 — 1013o 의 1(Critical).

점검 목적이 등록 모듈(`prereg_unified`)에 없어 영수증을 읽는 단계(`read_receipt`)에서 끊겼다(§119 I-07). 합성 세계는 C 의
`tests/uni_export_world.py` 도우미로 짜되 **학습 분할**이다 — 점검 목록은 봉인 페어의 학습 행이다. 점검 step 파일은 `vlm.ecc` 의
파일 꼴로 직접 쓴다(학습 루프를 돌리지 않는다). 모델 · GPU 없음. 본채점은 여전히 이 목적을 받지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import uni_export_world as W  # noqa: E402

ECC_RUN = "ecc-20261013T000000"
STEP = 2


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_ecc_world(tmp: Path) -> SimpleNamespace:
    import pandas as pd
    from PIL import Image

    from data.manifest_io import write_snapshot
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from evaluation.prereg_unified import dump_registration
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.ecc import PURPOSE, append_ledger, step_paths
    from vlm.exposure import append_row
    from vlm.pilot_vlm import train_rows_digest
    from vlm.train_cell import CENTRAL_TAG

    ids = list(W.IDS)
    img = tmp / "img"
    img.mkdir(parents=True)
    images = {}
    for i, iid in enumerate(ids):
        Image.new("L", (1280, 720), i * 20).save(img / f"{iid}.png")
        images[iid] = img / f"{iid}.png"
    eval_img = img / f"{W.EVAL_ID}.png"
    Image.new("L", (1280, 720), 222).save(eval_img)
    repo = tmp / "repo"
    repo.mkdir()
    W.git(repo, "init", "-q", "-b", "main")
    snap = tmp / "snap"
    man = [W._man(i, "train", _sha(images[i])) for i in ids] + [W._man(W.EVAL_ID, "eval", _sha(eval_img))]
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([W._ann(i) for i in ids]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    # 페어 — 점검 실행의 학습 행은 본실험 중앙과 같은 "학습 분할 전체" 다
    rows = [{"image_id": i, "split": "train", "client": "C1", "image_path": images[i].as_posix()} for i in ids]
    pairs = tmp / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rows_digest = train_rows_digest(rows)
    (repo / "configs").mkdir()
    (repo / "configs/base.yaml").write_text(
        "fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n"
        f"  snapshot_digest: {digest}\n  uni_processor_kwargs: {{max_pixels: {W.MAX_PIXELS}}}\n"
        "  uni_model: {id: Qwen/Qwen3.5-4B, revision: " + "a" * 40 + "}\n"
        f"  uni_prompt_path: {W.PROMPT_REL}\n  uni_coord_space: ABS_ORIG\n"
        "  uni_chat_template_kwargs: {enable_thinking: false}\n  uni_batch_size: 1\n  uni_max_new_tokens: null\n"
        f"  uni_pairs: {{path: {pairs.as_posix()}, digest: {_sha(pairs)}}}\n",
        encoding="utf-8")
    W.git(repo, "add", "-A")
    W.git(repo, "commit", "-q", "-m", "설정")
    head = W.git(repo, "rev-parse", "HEAD")

    parent = repo / "outputs" / "early_check"
    root = parent / ECC_RUN
    (root / "lists").mkdir(parents=True)
    raw = canonical_bytes(ids)
    elist = validate_eval_list(raw)
    gen_list = root / "lists" / "gen.txt"
    gen_list.write_bytes(raw)
    plan = root / "plan.yaml"
    plan.write_text("kind: early_coord_check\nversion: 1\ngen_limit: {max_new_tokens: 512}\n", encoding="utf-8")
    exposure = tmp / "exposure" / "uni_exposure_ledger.jsonl"
    append_row(exposure, {"event": "planned", "run_id": ECC_RUN, "purpose": PURPOSE, "tag": None, "seed_index": 1,
                          "mode": None, "list_kind": "gen", "list_path": gen_list.as_posix(),
                          "list_file_sha256": elist.file_sha256, "list_set_sha256": elist.set_sha256, "n": len(ids),
                          "snapshot_digest": digest, "commit": head, "at": "2026-10-13T09:00:00.000000+09:00"})
    code = tmp / "code"
    code.mkdir()
    W.git(code, "init", "-q", "-b", "main")
    (code / "m.py").write_text("x = 1\n", encoding="utf-8")
    W.git(code, "add", "-A")
    W.git(code, "commit", "-q", "-m", "코드")

    w = SimpleNamespace(tmp=tmp, repo=repo, head=head, snap=snap, digest=digest, parent=parent, root=root,
                        lists={"model": gen_list}, elist=elist, plan=plan, exposure=exposure, images=images, code=code)

    def edit(r):
        g = r.generation
        g.purpose, g.list_split = PURPOSE, "train"
        g.train_rows_digest, g.plan_sha256 = rows_digest, _sha(plan)
    w.reg = W.registration(w, edit=edit)
    # 영수증 — 종류 early_coord_check. 리허설처럼 로컬(점검 루트 아래)이다
    d = root / "registration"
    d.mkdir()
    raw_reg = dump_registration(w.reg)
    name = "early_coord_check-20261013-1"
    (d / f"{name}.json").write_bytes(raw_reg)
    (d / "train_config.json").write_bytes(W.train_config_text().encode("utf-8"))
    receipt = {"generation_sha256": w.reg.generation_sha256(), "scoring_sha256": w.reg.scoring_sha256(),
               "registered_at": "2026-10-13T09:00:00+09:00", "main_commit": head, "kind": PURPOSE,
               "registration_path": (d / f"{name}.json").relative_to(repo).as_posix(),
               "registration_file_sha256": hashlib.sha256(raw_reg).hexdigest()}
    w.receipt = d / f"{name}.receipt.json"
    w.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    # 점검 step 파일 — vlm.ecc 의 꼴(어댑터 · 소비 행 · meta · 원장). 학습 루프 없음
    out = root / "train" / f"{CENTRAL_TAG}_s1"
    out.mkdir(parents=True)
    paths = step_paths(out, STEP)
    np.savez(paths["adapter"], **{"a.lora_A.weight": np.full((2, 2), 3.0, dtype=np.float32)})
    paths["consumed"].write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    meta = {"purpose": PURPOSE, "step": STEP, "registered_steps": [STEP, 4],
            "identity": {"cell": CENTRAL_TAG, "run_id": ECC_RUN, "seed_index": 1,    # 시드 번호(학습 쪽 계약 — 사전 점검이 맞댄다)
                         "processor_config_sha256": W.MEASURED["processor_config_sha256"],
                         "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")), "input_adapter_digest": W.hx("init")},
            "adapter": {"file": paths["adapter"].name, "sha256": _sha(paths["adapter"])},
            "consumed": {"file": paths["consumed"].name, "sha256": _sha(paths["consumed"]), "rows": len(ids)},
            "train_rows_digest": rows_digest, "train_config": W.train_config_text(), "pin_sha256": W.hx("pin")}
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    append_ledger(out, {"at": 1.0, "event": "step", "step": STEP, "epoch": 0, "cursor": len(ids), "optimizer_steps": STEP,
                        "supervised_tokens": 10, "consumed_rows": len(ids), "consumed_sha256": meta["consumed"]["sha256"],
                        "adapter_sha256": meta["adapter"]["sha256"], "cum_wall_s": 1.0})
    w.adapter = out
    return w


def _clock():
    """합성 시계 — 영수증 시각(10-13 09:00) 뒤여야 한다(C 세계의 기본 시계는 10-01 이다)."""
    from datetime import datetime

    return W.Clock(start=datetime(2026, 10, 13, 12, 0, 0, tzinfo=W.KST))


@pytest.fixture
def ecc(tmp_path):
    return build_ecc_world(tmp_path)


def _preflight(ecc, **over):
    """`run_export` 가 사전 점검에 넘기는 것과 같은 인자로 `export_preflight` 를 직접 부른다."""
    from vlm.export_preflight import export_preflight
    from vlm.export_run import APPROVED_GENERATOR_LOADERS, APPROVED_MEASURES, parser_sha256
    from vlm.seams import check_seam
    from vlm.train_cell import CENTRAL_TAG

    purpose = "early_coord_check"
    impl_ids = {"export_measure": check_seam(W.measure, APPROVED_MEASURES, seam="export_measure", purpose=purpose,
                                             standin_allowed=True),
                "export_generator": check_seam(W.Loader(), APPROVED_GENERATOR_LOADERS["model"], seam="export_generator",
                                               purpose=purpose, standin_allowed=True)}
    args = dict(purpose=purpose, mode="model", tag=CENTRAL_TAG, seed_index=1, receipt_path=ecc.receipt,
                list_path=ecc.lists["model"], snapshot_root=ecc.snap, adapter_dir=ecc.adapter, plan_path=ecc.plan,
                run_root=ecc.root, measure_env=W.measure, impl_ids=impl_ids, parser_sha256=parser_sha256(), repo=ecc.repo,
                code_repo=ecc.code, exposure_ledger=ecc.exposure, now=_clock()(), check_step=STEP)
    args.update(over)
    return export_preflight(**args)


def test_점검_사전_점검이_영수증_등록_분할_표를_지나_끝까지_간다(ecc):
    """1013o 의 1 — `export_preflight(purpose="early_coord_check", check_step=…)` 가 영수증 종류 · 등록 목적 · 분할 train ·
    계획 · 노출 원장 · 점검 step 파일까지 다 지나 도장 · 곁 파일 값을 낸다."""
    pre = _preflight(ecc)
    assert pre.stamp_fields["purpose"] == "early_coord_check" and pre.stamp_fields["check_step"] == STEP
    assert pre.sidecar_values["scored_adapter_step"] == STEP
    assert pre.registration.generation.purpose == "early_coord_check" and pre.receipt.kind == "early_coord_check"
    assert list(pre.eval_list.ids) == list(W.IDS)


def test_점검_사전_점검은_step_을_빼면_받지_않는다(ecc):
    from vlm.export_run import ExportRefused

    with pytest.raises(ExportRefused) as exc:
        _preflight(ecc, check_step=None)
    assert exc.value.code == "check_step_scope"


@pytest.mark.xfail(strict=True, reason="학습 쪽 vlm/export_writer.py 의 STAMP_KEYS 에 check_step 이 없다 — 같은 쪽 "
                                       "_training_check_step 이 도장에 넣는 키다(학습 쪽 두 파일의 불일치, 1013o 보고). "
                                       "키가 더해지면 이 표지를 뗀다")
def test_점검_export_전체가_묶음을_봉인한다(ecc):
    from vlm.train_cell import CENTRAL_TAG

    res = W.run(ecc, purpose="early_coord_check", tag=CENTRAL_TAG, adapter_dir=ecc.adapter, check_step=STEP,
                folder=ecc.root / "export", standin_allowed=True, clock=_clock())
    assert res.sealed and res.n_written == len(W.IDS)
    assert res.meta["purpose"] == "early_coord_check"


def test_점검_목적의_영수증은_등록_모듈이_받고_본채점은_거부한다():
    from evaluation import early_coord_check as E
    from evaluation.entry_gate import EntryRejected, check_mode
    from evaluation.prereg_unified import PURPOSES, RECEIPT_KINDS, SPLIT_OF_PURPOSE

    assert "early_coord_check" in PURPOSES and "early_coord_check" in RECEIPT_KINDS
    assert SPLIT_OF_PURPOSE["early_coord_check"] == "train"
    with pytest.raises(EntryRejected):
        check_mode("probe", "early_coord_check")
    with pytest.raises(EntryRejected):
        check_mode("strict", "early_coord_check")
    with pytest.raises(E.EarlyCheckRejected):
        E.refuse_for_scoring({"purpose": E.KIND})


def test_점검_export_는_리허설_영수증을_받지_않는다(ecc):
    from vlm.export_run import ExportRefused
    from vlm.train_cell import CENTRAL_TAG

    rc = json.loads(ecc.receipt.read_text(encoding="utf-8"))
    rc["kind"] = "rehearsal"
    ecc.receipt.write_text(json.dumps(rc), encoding="utf-8")
    with pytest.raises(ExportRefused) as exc:
        W.run(ecc, purpose="early_coord_check", tag=CENTRAL_TAG, adapter_dir=ecc.adapter, check_step=STEP,
              folder=ecc.root / "export", standin_allowed=True, clock=_clock())
    assert exc.value.code in ("entry_RECEIPT_KIND", "entry_RECEIPT")
