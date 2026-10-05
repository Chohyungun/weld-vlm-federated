"""통합형 본채점 진입점 — **진입점을 부른다**(진입점 미니스펙 3판 1-3 · 1-6 · 5-1 · 5-4).

합성 세계 하나를 시험마다 새로 짓는다 — `git init` 한 저장소(본체 체크아웃 자리) · 합성 스냅샷 · 리허설 등록과 영수증 · 목록 셋 ·
모델과 에코 묶음(묶음 검증 시험의 정상 도우미) · 학습 원장과 어댑터 meta. 진입점은 **시험의 이음새**(`Seam`)로 그 저장소를 받는다.
동결 자산 · 실제 `outputs/` 를 건드리지 않는다.

지키는 것.

1. 정상 리허설은 종료 0 · 산출물 둘(배타 생성) · 정본 원장의 `call` 한 줄 · 산출물에 지표 값이 없다.
2. 진입 전 거부는 종료 3 이고 **정답 읽기 · 묶음 검증이 한 번도 불리지 않는다**(가-9).
3. 모드와 목적 · 출력 루트 · 영수증 종류 · 커밋된 영수증의 이음새 · 진단 경로.
4. 엄격 경로는 `require_receipt` 를 **글자 상수 두 영역**으로 부른다(정적).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import scripts.probe.score_unified as SU
from data.manifest_io import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, write_snapshot
from evaluation.coord_check import coord_fixture_digest
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.prereg_unified import CoordCfgRecord, Receipt, UnifiedRegistration, dump_registration
from evaluation.scoring_ledger import read_ledger
from tests.test_bundle_v14 import IDS, IMPL_IDS, IMPL_KEYS, hx, impl, make, train_config_text
from tests.test_scoring_layer import gen_row

REPO = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=KST)
CLASSES = ("100", "2011", "301", "401")
REG_REL = "outputs/rehearsal_u/r1/registration/rehearsal-20261001-1.json"
EVAL_ID, TRAIN_ID = "aihub000009", "aihub000008"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          capture_output=True, check=True).stdout.decode("utf-8").strip()


def _man(image_id: str, split: str) -> dict:
    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{image_id}.jpg", "sha256": hx(image_id),
        "width_px": 1280, "height_px": 720, "modality": "RT", "material": "ST", "has_defect": True, "n_defects": 1,
        "defect_types": "porosity", "iso_codes": "2011", "src_labels_raw": "기공", "label_type": "polygon",
        "has_localization": True, "phash_hex": "0" * 16, "group_id": f"g{image_id}", "group_size": 1,
        "strata_key": "ST|porosity", "split": split, "client": "C1" if split != "eval" else "",
        "eval_subset": "", "thickness_mm": None, "thickness_source": "unavailable", "px_per_mm": None,
        "scale_source": "unavailable", "quality_level": "", "ingest_version": "t", "label_map_version": 1, "notes": "",
    }


def _ann(image_id: str) -> dict:
    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#0", "image_id": image_id, "src_label_raw": "기공", "defect_type": "porosity",
        "iso_code": "2011", "polygon_json": "[]", "bbox_x1_px": 10, "bbox_y1_px": 20, "bbox_x2_px": 30,
        "bbox_y2_px": 40, "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


def _ledger_bytes() -> bytes:
    rows = [["run_id", "seed", "cell", "split_hash", "client_id", "round", "n_train_samples", "metric_name",
             "metric_value", "bytes_up", "bytes_down", "wall_time"]]
    for e, steps in ((1, 50.0), (2, 100.0)):
        rows.append(["run_c1", "20260828", "uni_local", "d", "C1", str(e), "10", "optimizer_steps", str(steps), "0", "0", "1"])
    rows.append(["run_c1", "20260828", "uni_local", "d", "C1", "100", "10", "final_adapter_step", "100.0", "0", "0", "1"])
    return ("\r\n".join(",".join(r) for r in rows) + "\r\n").encode("utf-8")


def build_world(tmp: Path, *, purpose: str = "rehearsal", reg_edit=None, meta_extra=None, model_rows=None) -> dict:
    """합성 세계. `reg_edit(reg)` 로 등록을, `meta_extra` 로 곁 파일을 흔든다."""
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    snap = tmp / "snap"
    man = [_man(i, "val") for i in IDS] + [_man(EVAL_ID, "eval"), _man(TRAIN_ID, "train")]
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([_ann(i) for i in IDS]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    (repo / "configs").mkdir()
    (repo / "configs/base.yaml").write_text(
        f"fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n  snapshot_digest: {digest}\n"
        f"  uni_processor_kwargs: {{max_pixels: {1280 * 704}}}\n", encoding="utf-8")
    fixtures = REPO / "vlm/coords_fixtures/golden_fixtures.json"
    (repo / "vlm/coords_fixtures").mkdir(parents=True)
    (repo / "vlm/coords_fixtures/golden_fixtures.json").write_bytes(fixtures.read_bytes())
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "설정")
    head = git(repo, "rev-parse", "HEAD")

    root = repo / "outputs/rehearsal_u/r1"
    lst = canonical_bytes(IDS)
    elist = validate_eval_list(lst)
    for name in ("generation.list", "scoring.list", "echo.list"):
        (root / "lists").mkdir(parents=True, exist_ok=True)
        (root / "lists" / name).write_bytes(lst)

    r = UnifiedRegistration()
    g = r.generation
    g.purpose, g.list_split, g.snapshot_digest = purpose, "val", digest
    g.eval_list_file_sha256 = g.echo_list_file_sha256 = elist.file_sha256
    g.eval_list_set_sha256 = g.echo_list_set_sha256 = elist.set_sha256
    g.prompt_sha256, g.chat_template_kwargs, g.gen_prefix_sha256 = hx("prompt"), {"add_generation_prompt": True}, hx("prefix")
    g.max_new_tokens, g.decoding, g.batch_size, g.padding_side = 256, {"do_sample": False}, 1, "left"
    g.processor_config_sha256, g.processor_min_pixels, g.processor_max_pixels = hx("proc"), 256, 1280 * 704
    g.patch_size, g.merge_size, g.coord_space, g.coord_cfg_hash = 16, 2, "ABS_ORIG", hx("coord")
    g.base_model_id, g.base_model_revision, g.init_adapter_digest = "local/qwen3.5-4b", "frozen", hx("init")
    g.train_config_sha256 = hashlib.sha256(train_config_text().encode("utf-8")).hexdigest()
    g.budget_n, g.budget_r, g.budget_e, g.transformers_version = 30, 3, 10, "4.57.0"
    g.impl_ids = IMPL_KEYS
    r.canary.coord_cfg = CoordCfgRecord(accepted=(hx("coord"),), frozen_value=hx("coord"), frozen_at="2026-10-01",
                                        current_value=hx("coord"), change_scope="first_registration",
                                        equivalence_evidence=("동결 export 없음 — 첫 등록",))
    r.canary.coord_fixture_sha256 = coord_fixture_digest(fixtures.read_bytes())
    r.metric.class_codes, r.metric.coupling_rule = CLASSES, "decoupled_v2"
    r.seeds = {"1": 20260828}
    if reg_edit:
        reg_edit(r)
    raw = dump_registration(r)
    (repo / REG_REL).parent.mkdir(parents=True, exist_ok=True)
    (repo / REG_REL).write_bytes(raw)
    receipt = {"generation_sha256": r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": purpose,
               "registration_path": REG_REL, "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    (root / "registration/receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    ledger = root / "train/uni_local_C1_s1/train_ledger.csv"
    ledger.parent.mkdir(parents=True)
    ledger.write_bytes(_ledger_bytes())
    (ledger.parent / "adapter_last.meta.json").write_text(
        json.dumps({"adapter_sha256": hx("adapter"), "adapter_step": 100}), encoding="utf-8")
    led_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    common = {"generation_sha256": r.generation_sha256(), "snapshot_digest": digest, "list_split": "val",
              "batch_size": 1, **(meta_extra or {})}
    bundles = root / "bundles"
    make(bundles, rows=model_rows or [gen_row(i) for i in IDS], purpose=purpose,
         meta_over=common | {"train_run_id": "run_c1", "train_ledger_sha256": led_sha,
                             "train_ledger_path": ledger.relative_to(repo).as_posix()},
         start_over={"generation_sha256": r.generation_sha256(), "train_run_id": "run_c1",
                     "train_ledger_sha256": led_sha})
    model_attempts = (bundles / "attempts.jsonl").read_bytes()
    make(bundles, rows=[gen_row(i) for i in IDS], purpose=purpose, echo=True, meta_over=common,
         start_over={"generation_sha256": r.generation_sha256()})
    (bundles / "attempts.jsonl").write_bytes(model_attempts + (bundles / "attempts.jsonl").read_bytes())
    return {"repo": repo, "root": root, "snap": snap, "reg": r, "head": head}


def argv(w: dict, *, registration="probe", purpose="rehearsal", out=None, receipt=None) -> list[str]:
    root = w["root"]
    return ["--registration", registration, "--purpose", purpose,
            "--receipt", str(receipt or root / "registration/receipt.json"), "--snapshot", str(w["snap"]),
            "--bundles", str(root / "bundles"), "--generation-list", str(root / "lists/generation.list"),
            "--scoring-list", str(root / "lists/scoring.list"), "--echo-list", str(root / "lists/echo.list"),
            "--train-root", str(root / "train"), "--out", str(out or root), "--device", "cpu:0"]


def run(w: dict, **kw) -> int:
    return SU.main(argv(w, **kw), _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)


def _numbers(obj) -> list:
    if isinstance(obj, bool) or obj is None:
        return []
    if isinstance(obj, (int, float)):
        return [obj]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers(v)]
    return []


# ---------------------------------------------------------------- 1 정상 리허설

def test_정상_리허설은_종료_0_이고_산출물_둘과_원장_한_줄을_남긴다(tmp_path) -> None:
    w = build_world(tmp_path)
    assert run(w) == SU.EXIT_OK
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    gates = json.loads((w["root"] / SU.OUT_GATES).read_text(encoding="utf-8"))
    assert score["run_kind"] == "rehearsal" and score["registration_mode"] == "probe"
    assert score["bundles"]["rejected"] == {} and len(score["bundles"]["verified"]) == 2
    assert score["canaries"]["echo"][0]["status"] == "pass"
    assert all(x["status"] == "pass" for x in score["canaries"]["literal"])
    assert gates["gates"]["③"]["status"] == "pass" and gates["gates"]["⑤"]["status"] == "pass"
    assert gates["adversarial_probe"]["status"] == "pass"
    [(sha, call)] = read_ledger(w["repo"] / "outputs/main_u/scoring_ledger.jsonl")
    assert call["kind"] == "call" and call["purpose"] == "rehearsal" and score["call_sha256"] == sha
    assert call["generation_sha256"] == w["reg"].generation_sha256()
    assert {b["mode"] for b in call["bundles"]} == {"model", "echo"}
    assert all(len(b["generations_content_sha256"]) == 64 for b in call["bundles"])
    assert call["scoring_sha256"] == "미완", "리허설의 채점 쪽은 비어 있다 — 빈 값으로 두지 않고 '미완' 이라 적는다"


def test_리허설_산출물에는_지표_값이_없다(tmp_path) -> None:
    w = build_world(tmp_path)
    run(w)
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert all(s["completed"] for s in score["metric_smoke"])
    assert _numbers(score["metric_smoke"]) == []


def test_같은_자리에_다시_채점하면_덮지_않고_멈춘다(tmp_path) -> None:
    w = build_world(tmp_path)
    run(w)
    with pytest.raises(FileExistsError):
        run(w)
    assert len(read_ledger(w["repo"] / "outputs/main_u/scoring_ledger.jsonl")) == 2, "실패한 호출도 원장에 남는다"


def test_리허설은_고의_중단_기록을_산출에_옮긴다(tmp_path) -> None:
    w = build_world(tmp_path, meta_extra={"fault_events": [{"kind": "kill"}]})
    assert run(w) == SU.EXIT_OK, "리허설은 고의 중단 기록을 옮길 뿐 거부하지 않는다"
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert list(score["bundles"]["fault_events"].values()) == [[{"kind": "kill"}], [{"kind": "kill"}]]


def test_승인_구현이_아니면_대역_실행이고_통과를_내지_않는다(tmp_path) -> None:
    """3판 1-1-다 · 07번 §30-4 — 리허설은 거부하지 않고 상태를 '대역 실행' 으로 적는다. 종료 2."""
    w = build_world(tmp_path, meta_extra={"impl_ids": {**IMPL_IDS, "export_generator": impl(
        "export_generator", "load_generator", approved=False)}})
    assert run(w) == SU.EXIT_REJECTED
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert score["status"] == "stand_in" and score["bundles"]["rejected"] == {}
    assert len(score["bundles"]["stand_in"]) == 2


def test_등록값과_곁_파일이_어긋나면_그_칸이_거부되고_종료_2(tmp_path) -> None:
    w = build_world(tmp_path, meta_extra={"max_new_tokens": 512})
    assert run(w) == SU.EXIT_REJECTED
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert all("registration_item_mismatch" in v for v in score["bundles"]["rejected"].values())


@pytest.mark.parametrize(("edit", "why"), [
    (lambda r: {**r, "extra_key": 1}, "분류 밖 키"),
    (lambda r: {k: v for k, v in r.items() if k != "n_bad_items_dropped"}, "빠진 내용 키"),
])
def test_내용_해시를_낼_수_없는_묶음은_채점하지_않는다(tmp_path, edit, why: str) -> None:
    """검토(10-02 병합 뒤) 2 — 원시 해시가 맞아도 분류 밖 키 · 빠진 내용 키의 묶음은 검증 · 채점으로 가지 않는다(07번 §32-2)."""
    rows = [gen_row(i) for i in IDS]
    rows[0] = edit(rows[0])
    w = build_world(tmp_path, model_rows=rows)
    assert run(w) == SU.EXIT_REJECTED
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert score["bundles"]["rejected"] == {"uni_local_C1_s1": ["content_key"]}, why
    assert [vb for vb in score["bundles"]["verified"] if not vb.endswith(".echo")] == []
    [(_, call)] = read_ledger(w["repo"] / "outputs/main_u/scoring_ledger.jsonl")
    model = next(b for b in call["bundles"] if b["mode"] == "model")
    assert model["generations_content_sha256"] is None and model["content_key_error"]


def test_리허설_산출에_축_배율_카나리아의_블록이_있다(tmp_path) -> None:
    """학습 쪽 재확인 6 — 쓰는 쪽 판정기는 `canaries.axis_ratio` 가 없으면 호출되지 않은 것으로 본다(판정 불가도 호출이다).
    묶음마다 상태(문자열)와 표본 수(정수) — 비의 값은 싣지 않는다."""
    w = build_world(tmp_path)
    assert run(w) == SU.EXIT_OK
    score = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))
    [ax] = score["canaries"]["axis_ratio"]
    assert ax["bundle"] == "uni_local_C1_s1" and isinstance(ax["status"], str)
    assert (ax["status"], ax["reason"]) == ("판정 불가", "등록 칸 없음"), "리허설 세계의 등록에는 카나리아 칸이 없다"
    assert ax["n_pairs"] == len(IDS) and all(type(v) is int for k, v in ax.items() if k.startswith("n_"))
    assert not any(isinstance(v, float) for v in ax.values())


def test_축_배율_등록_칸이_있으면_상태를_낸다(tmp_path) -> None:
    def edit(r):
        r.canary.axis_ratio_threshold, r.canary.axis_ratio_min_pairs = 0.01, 1
    w = build_world(tmp_path, reg_edit=edit)
    assert run(w) == SU.EXIT_OK
    [ax] = json.loads((w["root"] / SU.OUT_SCORE).read_text(encoding="utf-8"))["canaries"]["axis_ratio"]
    assert (ax["status"], ax["reason"], ax["n_pairs"]) == ("통과", "", len(IDS))


def test_call_줄의_원장_잠금_시간_초과도_종료_3(tmp_path, monkeypatch, capsys) -> None:
    """재확인 13 — `call` 줄을 쓰다 잠금을 기다리는 한도를 넘으면 처리되지 않은 예외가 아니라 종료 3 이다."""
    import threading

    from evaluation import scoring_ledger as SL
    w = build_world(tmp_path)
    ledger = w["repo"] / "outputs/main_u/scoring_ledger.jsonl"
    monkeypatch.setattr(SL, "LOCK_TIMEOUT_S", 0.3)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with SL.ledger_lock(ledger):
            held.set()
            release.wait(60)
    t = threading.Thread(target=hold)
    t.start()
    try:
        assert held.wait(60)
        capsys.readouterr()
        assert run(w) == SU.EXIT_ENTRY
    finally:
        release.set()
        t.join(60)
    assert "원장 잠금" in capsys.readouterr().err
    assert read_ledger(ledger) == [], "줄을 쓰지 못했다 — 시도로 세지 않는다"


# ---------------------------------------------------------------- 2 · 3 진입 전 거부

@pytest.fixture
def spy(monkeypatch):
    """정답 읽기 · 묶음 검증 · 이미지의 호출 기록(가-9). 진입 전 거부에서 셋 다 0 이어야 한다."""
    calls = {"gold": 0, "verify": 0}

    def gold(*a, **k):
        calls["gold"] += 1
        raise AssertionError("진입 전 거부인데 정답을 읽었다")

    def verify(*a, **k):
        calls["verify"] += 1
        raise AssertionError("진입 전 거부인데 묶음을 열었다")
    monkeypatch.setattr(SU, "read_annotations_view", gold)
    monkeypatch.setattr(SU, "verify_bundle", verify)
    return calls


@pytest.mark.parametrize(("kw", "why"), [
    ({"registration": "probe", "purpose": "main"}, "탐색 채점에 본실험 목적(가 — 판정 09 의 I-8)"),
    ({"registration": "strict", "purpose": "rehearsal"}, "엄격은 main 만"),
    ({"registration": "strict", "purpose": "main"}, "엄격 + 리허설 영수증(가-6)"),
    ({"registration": "probe", "purpose": "frame_diag"}, "리허설 영수증으로 진단 목적"),
])
def test_모드_목적_영수증이_한_줄로_서지_않으면_종료_3(tmp_path, spy, kw: dict, why: str) -> None:
    w = build_world(tmp_path)
    out = w["repo"] / "outputs/main_u" if kw["registration"] == "strict" else None
    assert run(w, out=out, **kw) == SU.EXIT_ENTRY, why
    assert spy == {"gold": 0, "verify": 0}


@pytest.mark.parametrize("out", ["outputs/main_u", "outputs/main_d/x", "outputs/rehearsal_u", "elsewhere"])
def test_탐색_채점의_출력_루트(tmp_path, spy, out: str) -> None:
    w = build_world(tmp_path)
    assert run(w, out=w["repo"] / out) == SU.EXIT_ENTRY
    assert spy == {"gold": 0, "verify": 0}


def test_엄격의_출력_루트는_main_u_뿐이다(tmp_path, spy) -> None:
    """가-4′ — 엄격 + 비본실험 부모 아래의 출력 루트."""
    w = build_world(tmp_path)
    assert run(w, registration="strict", purpose="main", out=w["root"]) == SU.EXIT_ENTRY


def test_리허설_등록이_eval_기준이면_종료_3(tmp_path, spy) -> None:
    """가-2 — 등록의 기준 분할이 eval 이면 무효다."""
    w = build_world(tmp_path, reg_edit=lambda r: setattr(r.generation, "list_split", "eval"))
    assert run(w) == SU.EXIT_ENTRY
    assert spy == {"gold": 0, "verify": 0}


def test_묶음_폴더가_없으면_원장_줄_전에_종료_3(tmp_path, spy, capsys) -> None:
    """결정 04 의 4절 12 — 처리되지 않은 예외가 아니라 사유 코드로 거부한다. 시도가 아니므로 원장 줄도 없다."""
    w = build_world(tmp_path)
    import shutil
    shutil.rmtree(w["root"] / "bundles")
    assert run(w) == SU.EXIT_ENTRY
    assert "[BUNDLES_MISSING]" in capsys.readouterr().err
    assert not (w["repo"] / "outputs/main_u/scoring_ledger.jsonl").exists()
    assert spy == {"gold": 0, "verify": 0}


def test_등록_거부도_사유_코드를_싣는다(tmp_path, capsys) -> None:
    """결정 04 의 4절 13 — 등록 객체의 관계 위반도 다른 거부와 같은 `[코드]` 꼴이다."""
    w = build_world(tmp_path, reg_edit=lambda r: setattr(r.generation, "list_split", "eval"))
    assert run(w) == SU.EXIT_ENTRY
    assert "[REGISTRATION_INVALID]" in capsys.readouterr().err


def test_등록_파일의_해시가_영수증과_다르면_종료_3(tmp_path, spy) -> None:
    w = build_world(tmp_path)
    p = w["repo"] / REG_REL
    p.write_bytes(p.read_bytes().replace(b'"left"', b'"right"'))
    assert run(w) == SU.EXIT_ENTRY
    assert spy == {"gold": 0, "verify": 0}


def test_등록의_동결본이_닻과_다르면_종료_3(tmp_path, spy) -> None:
    w = build_world(tmp_path, reg_edit=lambda r: setattr(r.generation, "snapshot_digest", "e" * 64))
    assert run(w) == SU.EXIT_ENTRY


def test_목록에_평가_id_가_섞이면_정답을_읽기_전에_종료_3(tmp_path, monkeypatch) -> None:
    """목록 해시를 맞춘 채 eval id 를 넣는다 — 분할 대조가 정답 뷰와 묶음 검증 전에 막는다."""
    ids = [*IDS, EVAL_ID]
    lst = canonical_bytes(ids)
    el = validate_eval_list(lst)

    def edit(r):
        g = r.generation
        g.eval_list_file_sha256 = g.echo_list_file_sha256 = el.file_sha256
        g.eval_list_set_sha256 = g.echo_list_set_sha256 = el.set_sha256
    w = build_world(tmp_path, reg_edit=edit)
    for name in ("generation.list", "scoring.list", "echo.list"):
        (w["root"] / "lists" / name).write_bytes(lst)
    calls = {"gold": 0, "verify": 0}
    monkeypatch.setattr(SU, "read_annotations_view", lambda *a, **k: calls.__setitem__("gold", calls["gold"] + 1))
    monkeypatch.setattr(SU, "verify_bundle", lambda *a, **k: calls.__setitem__("verify", calls["verify"] + 1))
    assert run(w) == SU.EXIT_ENTRY
    assert calls == {"gold": 0, "verify": 0}


def test_커밋된_영수증으로는_이음새를_쓸_수_없다(tmp_path, spy, capsys, monkeypatch) -> None:
    """본체의 루트 커밋(본줄기의 조상)을 적은 진단 영수증 — 이음새를 거부한다(07번 §32-1).

    검토(10-02 병합 뒤) 3 — 이 시험이 **그 검사에 닿는지** 본다. 종료 3 만 보면 앞의 관문이 먼저 끝내도 지난다.
    `guard_seam` 이 불렸는지와 그 사유 코드를 단언한다."""
    w = build_world(tmp_path)
    here = Path(__file__).resolve().parent
    root_commit = subprocess.run(["git", "-C", str(here), "rev-list", "--max-parents=0", "refs/heads/main"],
                                 capture_output=True, check=True).stdout.decode().split()[-1]
    rc = json.loads((w["root"] / "registration/receipt.json").read_text(encoding="utf-8"))
    p = w["root"] / "registration/committed.json"
    p.write_text(json.dumps(rc | {"kind": "frame_diag", "main_commit": root_commit}), encoding="utf-8")
    calls = []
    real = SU.guard_seam

    def counted(receipt):
        calls.append(receipt.kind)
        return real(receipt)
    monkeypatch.setattr(SU, "guard_seam", counted)
    code = SU.main(argv(w, purpose="frame_diag", receipt=p), _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)
    assert code == SU.EXIT_ENTRY
    assert calls == ["frame_diag"], "이음새 검사에 닿았다"
    assert "[SEAM_ON_COMMITTED_RECEIPT]" in capsys.readouterr().err


# ---------------------------------------------------------------- 4 정적

def _calls(name: str) -> list[ast.Call]:
    tree = ast.parse(Path(SU.__file__).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def test_엄격은_require_receipt_를_글자_상수_두_영역으로_부른다() -> None:
    """3판 1-3 의 6 — `side=` 가 변수면 구문 트리로 영역을 읽지 못한다(검토 10 의 3-1)."""
    sides = []
    for c in _calls("require_receipt"):
        kw = {k.arg: k.value for k in c.keywords}
        assert isinstance(kw.get("side"), ast.Constant), "side 는 글자 상수여야 한다"
        sides.append(kw["side"].value)
    assert {"generation", "scoring"} <= set(sides)


def test_진입점은_verify_bundle_을_부르고_명령줄에_원장_경로_인자가_없다() -> None:
    assert _calls("verify_bundle")
    opts = {a for act in SU._parser()._actions for a in act.option_strings}
    assert not {o for o in opts if "ledger" in o or "repo" in o or "checkout" in o}


def test_진입점은_금지한_읽기를_가져오지_않는다() -> None:
    """3판 1-7 의 금지 표 — 진입점 모듈이 가져오는 이름에 없다."""
    tree = ast.parse(Path(SU.__file__).read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    banned = {"load_snapshot", "read_annotations", "read_absorb", "join_defects", "read_manifest", "read_tiles",
              "load_absorbed", "export_for_training"}
    assert not names & banned
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"absorb_image", "absorb_defect", "absorb_region"}


def test_엄격의_평가_정답_뷰는_데이터_쪽_판독기에_묶여_있다() -> None:
    """판정 10 의 1절 — 이름과 조건을 먼저 적고, 들어오면 잇는다. 영수증은 **파일 경로**로 준다(판독기가 바이트로 기록한다)."""
    import inspect

    from data import manifest_view
    assert SU.EVAL_VIEW is manifest_view.read_annotations_view_eval
    assert SU.EVAL_VIEW_READER == "data.manifest_view.read_annotations_view_eval"
    params = inspect.signature(SU.EVAL_VIEW).parameters
    assert params["receipt"].kind is inspect.Parameter.KEYWORD_ONLY
    assert len(SU.EVAL_VIEW_CONDITIONS) == 3
    assert not any("정답 뷰" in x or "구간 규약" in x for x in SU.STRICT_PENDING), "들어온 것은 비워 둔 자리에서 뺀다"


def test_진입점은_평가_id_를_검사_없는_뷰로_읽지_않는다() -> None:
    """검사 없는 뷰(`read_annotations_view`)를 부르는 자리는 정답 함수 하나다. 그 함수에 주는 목록은 에코 목록과 리허설의 생성 목록 —
    둘 다 `val` 이고 분할 대조(`check_list_split`)가 정답 읽기 전에 막는다. 엄격 경로는 단계 5(에코)까지만 간다."""
    tree = ast.parse(Path(SU.__file__).read_text(encoding="utf-8"))
    callers = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "read_annotations_view"
                       for n in ast.walk(f))]
    assert callers == ["_gold"]


# ---------------------------------------------------------------- 에코 전용 — 학습 전 관문(시도로 세지 않는다)

LEDGER_REL = "outputs/main_u/scoring_ledger.jsonl"


def run_echo(w: dict, **kw) -> int:
    return SU.main(argv(w, **kw) + ["--echo-only"], _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)


def _echo_body(w: dict) -> dict:
    return json.loads((w["root"] / SU.OUT_ECHO_GATE).read_text(encoding="utf-8"))


def _drop_model_bundle(w: dict) -> dict[Path, bytes]:
    """학습 전의 묶음 폴더 — 에코 묶음만 있다. 지운 파일의 바이트를 돌려준다(모델 묶음이 붙은 뒤를 되살리는 시험)."""
    from evaluation.bundle_v14 import BundleFiles
    kept: dict[Path, bytes] = {}
    for p in SU._bundle_files(w["root"] / "bundles"):
        if not SU._NAME.match(p.name)["echo"]:
            f = BundleFiles(p)
            for q in (f.generations, f.sidecar, f.start_stamp, f.tokens):
                if q.exists():
                    kept[q] = q.read_bytes()
                    q.unlink()
    return kept


def test_에코_전용은_학습_전_묶음으로_통과하고_원장에_아무것도_쓰지_않는다(tmp_path) -> None:
    w = build_world(tmp_path)
    _drop_model_bundle(w)
    assert run_echo(w) == SU.EXIT_OK
    assert not (w["repo"] / LEDGER_REL).exists(), "에코 전용 호출은 원장에 줄을 쓰지 않는다 — 시도가 아니다"
    body = _echo_body(w)
    assert body["kind"] == "echo_gate" and body["status"] == "pass" and body["reasons"] == []
    assert body["bundles"]["verified"] == ["uni_local_C1_s1.echo"] and body["bundles"]["rejected"] == {}
    assert body["canaries"]["echo"][0]["status"] == "pass"
    assert body["fingerprints"]["generation_sha256"] == w["reg"].generation_sha256()
    assert not (w["root"] / SU.OUT_SCORE).exists() and not (w["root"] / SU.OUT_GATES).exists()


def test_에코_전용은_모델_묶음이_있어도_에코만_본다(tmp_path, capsys) -> None:
    w = build_world(tmp_path)
    assert run_echo(w) == SU.EXIT_OK
    body = _echo_body(w)
    assert body["bundles"] == {"verified": ["uni_local_C1_s1.echo"], "rejected": {}}, "모델 묶음은 산출에 싣지 않는다"
    assert "열지 않은 모델 묶음: uni_local_C1_s1.generations.jsonl" in capsys.readouterr().err
    assert not (w["repo"] / LEDGER_REL).exists()


def test_에코_통과_뒤_모델_묶음이_붙어도_같은_재개는_받는다(tmp_path, capsys) -> None:
    """학습 전에 에코만으로 통과 → 모델 export → 판정 전에 중단 → 학습 쪽이 에코 관문부터 다시 부른다. 에코 입력이 그대로면 받는다."""
    w = build_world(tmp_path)
    kept = _drop_model_bundle(w)
    assert run_echo(w) == SU.EXIT_OK
    before = (w["root"] / SU.OUT_ECHO_GATE).read_bytes()
    for q, raw in kept.items():
        q.write_bytes(raw)
    capsys.readouterr()
    assert run_echo(w) == SU.EXIT_OK
    assert (w["root"] / SU.OUT_ECHO_GATE).read_bytes() == before
    assert not (w["repo"] / LEDGER_REL).exists()


def _receipt_edit(w: dict, **kv) -> None:
    rp = w["root"] / "registration/receipt.json"
    rp.write_text(json.dumps(json.loads(rp.read_text(encoding="utf-8")) | kv), encoding="utf-8")


def _registration_edit(w: dict) -> None:
    p = w["repo"] / REG_REL
    p.write_bytes(p.read_bytes().replace(b'"left"', b'"right"'))


@pytest.mark.parametrize(("case", "code"), [
    ("out_main_u", "[OUT_ROOT]"),
    ("out_elsewhere", "[OUT_ROOT]"),
    ("receipt_generation", "generation 지문"),
    ("registration_file", "[REGISTRATION_FILE_HASH]"),
    ("receipt_scoring", "scoring 지문"),
    ("anchor", "[SNAPSHOT_NOT_ANCHOR]"),
])
def test_에코_전용도_영수증_등록_닻_출력_루트가_어긋나면_진입_전_종료_3(tmp_path, spy, capsys, case: str, code: str) -> None:
    """에코 전용 호출의 0 · 1 단계 — 일반 호출과 같은 검사가 **이 분기에서도** 걸린다. 검사를 지우면 이 반례가 지난다."""
    edit = {"anchor": lambda r: setattr(r.generation, "snapshot_digest", "e" * 64)}.get(case)
    w = build_world(tmp_path, reg_edit=edit)
    _drop_model_bundle(w)
    out = {"out_main_u": w["repo"] / "outputs/main_u", "out_elsewhere": w["repo"] / "elsewhere"}.get(case)
    if case == "receipt_generation":
        _receipt_edit(w, generation_sha256="a" * 64)
    elif case == "receipt_scoring":
        _receipt_edit(w, scoring_sha256="b" * 64)
    elif case == "registration_file":
        _registration_edit(w)
    capsys.readouterr()
    assert run_echo(w, out=out) == SU.EXIT_ENTRY
    assert code in capsys.readouterr().err
    assert spy == {"gold": 0, "verify": 0}
    assert not (w["repo"] / LEDGER_REL).exists()
    assert not list(w["repo"].rglob("echo_gate.json")), "거부하면 산출을 쓰지 않는다"


@pytest.mark.parametrize(("registration", "purpose"), [("strict", "main"), ("probe", "main")])
def test_에코_전용은_본실험_목적을_진입_전에_거부한다(tmp_path, spy, capsys, registration: str, purpose: str) -> None:
    w = build_world(tmp_path)
    out = w["repo"] / "outputs/main_u" if registration == "strict" else None
    assert run_echo(w, registration=registration, purpose=purpose, out=out) == SU.EXIT_ENTRY
    assert "[ECHO_ONLY_PURPOSE]" in capsys.readouterr().err
    assert spy == {"gold": 0, "verify": 0}
    assert not (w["repo"] / LEDGER_REL).exists()
    assert not (w["root"] / SU.OUT_ECHO_GATE).exists()


def test_에코_전용은_묶음이_거부되면_사유를_싣고_종료_2(tmp_path) -> None:
    w = build_world(tmp_path, meta_extra={"max_new_tokens": 512})
    _drop_model_bundle(w)
    assert run_echo(w) == SU.EXIT_REJECTED
    body = _echo_body(w)
    assert body["status"] == "fail" and body["reasons"] == ["echo_bundle_rejected"]
    assert "registration_item_mismatch" in body["bundles"]["rejected"]["uni_local_C1_s1.echo"]
    assert not (w["repo"] / LEDGER_REL).exists()


def test_에코_전용은_에코_묶음이_없으면_사유를_싣는다(tmp_path) -> None:
    w = build_world(tmp_path)
    for p in (w["root"] / "bundles").glob("*.echo.*"):
        p.unlink()
    assert run_echo(w) == SU.EXIT_REJECTED
    assert _echo_body(w)["reasons"] == ["no_echo_bundle"]


def test_에코_전용은_같은_입력의_재개를_받고_다른_산출은_덮지_않는다(tmp_path, capsys) -> None:
    w = build_world(tmp_path)
    _drop_model_bundle(w)
    assert run_echo(w) == SU.EXIT_OK
    before = (w["root"] / SU.OUT_ECHO_GATE).read_bytes()
    assert run_echo(w) == SU.EXIT_OK, "같은 입력의 재개는 같은 바이트라 받는다"
    assert (w["root"] / SU.OUT_ECHO_GATE).read_bytes() == before
    (w["root"] / SU.OUT_ECHO_GATE).write_bytes(before.replace(b'"pass"', b'"PASS"', 1))
    capsys.readouterr()
    assert run_echo(w) == SU.EXIT_ENTRY
    assert "[ECHO_GATE_OUT_EXISTS]" in capsys.readouterr().err


def test_에코_전용은_원장_자리를_건드리지_않는다(tmp_path) -> None:
    """정상 채점이 원장에 한 줄을 쓴 뒤에도 에코 전용 호출은 원장 바이트를 바꾸지 않는다."""
    w = build_world(tmp_path)
    assert run(w) == SU.EXIT_OK
    led = w["repo"] / LEDGER_REL
    before = led.read_bytes()
    assert run_echo(w) == SU.EXIT_OK
    assert led.read_bytes() == before


# ---------------------------------------------------------------- 산출 경로 — `--out` 밖으로 이어지는 링크(공유 회신 §57)

def _dir_link(link: Path, target: Path) -> None:
    """디렉터리 링크 — 기호 링크가 막히면(윈도 권한) 정션. **시험이 만든 링크는 시험이 뗀다**(`_unlink_dir`)."""
    import os
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def _unlink_dir(link: Path) -> None:
    """링크만 뗀다 — 대상을 지우지 않는다(정션은 `rmdir` 가 링크만 없앤다)."""
    import os
    try:
        os.unlink(link)
    except OSError:
        os.rmdir(link)


@pytest.mark.parametrize("echo_only", [True, False])
def test_score_가_밖으로_이어지면_원장_줄_전에_종료_3이고_밖에_쓰지_않는다(tmp_path, capsys, echo_only: bool) -> None:
    w = build_world(tmp_path)
    if echo_only:
        _drop_model_bundle(w)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = w["root"] / "score"
    _dir_link(link, outside)
    try:
        capsys.readouterr()
        assert (run_echo(w) if echo_only else run(w)) == SU.EXIT_ENTRY
        assert "[OUT_PATH_LINK]" in capsys.readouterr().err
        assert list(outside.iterdir()) == [], "허용 영역 밖에 산출이 생기지 않는다"
        assert not (w["repo"] / LEDGER_REL).exists(), "원장 줄 전에 멈춘다 — 시도가 아니다"
    finally:
        _unlink_dir(link)
    assert outside.is_dir(), "링크를 떼도 대상은 남는다"


def test_에코_산출이_하드_링크면_읽지_않고_종료_3(tmp_path, capsys) -> None:
    """같은 바이트를 비교하려는 읽기도 밖의 파일을 읽는 일이다 — 하드 링크면 읽기 전에 거부한다."""
    import os
    w = build_world(tmp_path)
    _drop_model_bundle(w)
    assert run_echo(w) == SU.EXIT_OK
    gate = w["root"] / SU.OUT_ECHO_GATE
    outside = tmp_path / "outside.json"
    outside.write_bytes(gate.read_bytes())
    gate.unlink()
    os.link(outside, gate)
    before = outside.read_bytes()
    capsys.readouterr()
    assert run_echo(w) == SU.EXIT_ENTRY
    assert "[OUT_FILE_HARDLINK]" in capsys.readouterr().err
    assert outside.read_bytes() == before and not (w["repo"] / LEDGER_REL).exists()


def test_에코_산출_파일이_밖을_가리키는_기호_링크면_종료_3(tmp_path, capsys) -> None:
    import os
    w = build_world(tmp_path)
    _drop_model_bundle(w)
    (w["root"] / "score").mkdir(parents=True)
    outside = tmp_path / "outside.json"
    try:
        os.symlink(outside, w["root"] / SU.OUT_ECHO_GATE)
    except OSError:
        pytest.skip("이 환경에서는 파일 기호 링크를 만들 수 없다(권한)")
    capsys.readouterr()
    assert run_echo(w) == SU.EXIT_ENTRY
    assert "[OUT_PATH_LINK]" in capsys.readouterr().err
    assert not outside.exists() and not (w["repo"] / LEDGER_REL).exists()


@pytest.mark.parametrize("echo_only", [True, False])
def test_검사_뒤에_score_가_밖으로_이어져도_쓰기_직전에_멈춘다(tmp_path, capsys, monkeypatch, echo_only: bool) -> None:
    """0 단계 검사와 쓰기 사이에 링크가 생긴 경우 — 쓰기 직전의 검사가 막는다. 일반 리허설은 이미 원장 줄을 쓴 뒤라 시도 하나다."""
    w = build_world(tmp_path)
    if echo_only:
        _drop_model_bundle(w)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = w["root"] / "score"
    original = SU._read_lists

    def read_lists_then_link(ctx):
        res = original(ctx)
        _dir_link(link, outside)
        return res
    monkeypatch.setattr(SU, "_read_lists", read_lists_then_link)
    try:
        capsys.readouterr()
        assert (run_echo(w) if echo_only else run(w)) == SU.EXIT_ENTRY
        assert "[OUT_PATH_LINK]" in capsys.readouterr().err
        assert list(outside.iterdir()) == []
        led = w["repo"] / LEDGER_REL
        if echo_only:
            assert not led.exists()
        else:
            assert [r["kind"] for _, r in read_ledger(led)] == ["call"]
    finally:
        if os.path.lexists(link):
            _unlink_dir(link)
