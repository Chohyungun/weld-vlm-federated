"""조기 좌표 점검 체크포인트 → 내보내기 사슬(외부 검토 §137 C-01, 지시 1014b) — **실제 작성자**의 step 파일로 끝까지.

C-01 의 두 차단 — (1) 실제 작성자(`vlm.pilot_vlm` 의 `UniResumeIdentity`)는 시드 번호(`seed_index`)를 쓰지 않는데 사전 점검이 그 칸을
요구했다, (2) 사전 점검이 도장에 넣는 `check_step` 을 writer 가 모르는 키로 거부했다. 앞의 시험들은 합성 meta 에 `seed_index` 를
손으로 넣어 (1)을 보지 못했고 (2)는 strict xfail 이었다.

- **좁은 시험**은 실제 점검 학습(`tests.test_ecc_train._export_world` — 모델만 대역)이 쓴 체크포인트로 사전 점검의 점검 분기와
  writer 를 직접 부른다. 등록의 시드 값 7 · 번호 1 · step 2 는 그 세계의 칸 설정에서 오는 상수이고, 검사 대상인 체크포인트에서
  거꾸로 읽지 않는다.
- **한 흐름 시험**은 실제 점검 학습(`run_ecc_train`)이 쓴 step 파일 → 평가 쪽 등록 · 영수증 · 스냅숏 · 노출 원장 · 계획 →
  공개 사전 점검(`export_preflight`) → `run_export` → writer 봉인까지 간다. 등록값은 그 실행이 실제로 낸 값(학습 설정 원문 ·
  초기 어댑터 · 예산)과 같은 세계에서 낸 실측(프로세서 설정은 학습 쪽과 같은 함수로 같은 대역 프로세서에서)으로 채운다.
  공식 CLI(`scripts/export_uni.py`)의 계획된 멈춤 → 재호출 → 이어 받기(I-04 와 함께)도 본다.

대역은 모델 · 생성기 · 착수 고정의 코드 · 환경 · 공통 칸뿐이다(그 밖의 검증은 실물). 설정 대조(`config_check`)는 이 세계의 저장소
설정이 본실험 예산 칸을 갖지 않아 이음새로 넘긴다 — export 사슬의 시험이지 착수 고정 · 설정 대조의 시험이 아니다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import uni_export_world as W
from tests.uni_fakes import CountingLoader
from vlm import ecc as E
from vlm import pilot_vlm
from vlm.train_cell import CENTRAL_TAG

ECC_RUN = "ecc-20261013T000000"
SEED = 7                                                                # 등록의 시드표 1 번(이 세계의 상수)


@pytest.fixture(autouse=True)
def _small_world(monkeypatch):
    """GPU 를 쓰지 않고 누적 창을 2 로 줄인다(신원의 `grad_accum` 도 같은 값을 읽는다)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 2)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


# ================================================================ 좁은 시험 — 실제 작성자의 체크포인트
def _actual(tmp: Path):
    """실제 점검 학습이 쓴 step 2 체크포인트와 사전 점검 점검 분기의 인자(등록 시드표 {"1": 7} · 번호 1)."""
    from tests.test_ecc_train import _export_world

    w, kw = _export_world(tmp)
    meta = json.loads(E.step_paths(w.out, 2)["meta"].read_text(encoding="utf-8"))
    ident = meta["identity"]
    assert "seed_index" not in ident                                    # 실제 작성자 — 칸을 덧붙이지 않았다
    assert (ident["seed"], ident["round_idx"], ident["client_idx"]) == (SEED, 0, 0)
    assert kw["reg"].seeds == {"1": SEED} and kw["seed_index"] == 1
    return w, kw


def _writer(folder: Path, stamp: dict, *, seed_index=1, mode="model", purpose="early_coord_check"):
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from vlm.export_writer import ExportWriter

    return ExportWriter(folder, tag=CENTRAL_TAG, seed_index=seed_index, mode=mode,
                        eval_list=validate_eval_list(canonical_bytes(["aihub000001"])),
                        stamp_fields={"purpose": purpose, "export_commit": "a" * 40, "parser_sha256": "b" * 64,
                                      "generation_sha256": "c" * 64, **stamp},
                        device="cpu:0", clock=W.Clock())


def _finish(writer) -> None:
    writer.begin()
    writer.open_generations()
    writer.write({"image_id": "aihub000001", "text": "{}"})
    writer.end()


def _synthetic_stamp(step=2) -> dict:
    return {"train_run_id": "synthetic-ecc", "train_ledger_sha256": "d" * 64, "scored_adapter_sha256": "e" * 64,
            "check_step": step}


def test_실제_작성자의_체크포인트가_시드_번호_없이_사전_점검과_writer_를_지나_봉인된다(tmp_path):
    from vlm.export_preflight import _training_check_step

    w, kw = _actual(tmp_path)
    before = _tree(w.out)
    _, stamp, side = _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert stamp["check_step"] == 2 and type(stamp["check_step"]) is int
    writer = _writer(tmp_path / "export", stamp)
    _finish(writer)
    meta = writer.seal(side)
    stored = json.loads(writer.p_stamp.read_text(encoding="utf-8"))
    assert stored["check_step"] == 2 and type(stored["check_step"]) is int
    assert meta["scored_adapter_step"] == 2 and type(meta["scored_adapter_step"]) is int
    assert _tree(w.out) == before                                       # 원 체크포인트 · 원장을 바꾸지 않는다


@pytest.mark.parametrize("index", [True, False, 0, -1, 1.0, "1", None])
def test_시드_번호는_bool_아닌_양의_정수다(index):
    from vlm.export_preflight import ExportRefused, _ecc_seed_value

    with pytest.raises(ExportRefused) as exc:
        _ecc_seed_value(SimpleNamespace(seeds={"1": SEED}), index)
    assert exc.value.code == "seed_index"


@pytest.mark.parametrize("seeds", [None, {}, {"1": True}, {"1": -1}, {"1": "7"}, {"1": 7.0}, {"2": 108},
                                   {"1": 7, "2": 7},                    # 값이 겹치면 한 체크포인트가 두 번호로 나간다
                                   {"1": 7, "2": "8"}])                 # 고르지 않은 번호의 값도 등록 계약대로
def test_등록의_시드표_값은_서로_다른_0_이상의_정수여야_고른다(seeds):
    from vlm.export_preflight import ExportRefused, _ecc_seed_value

    with pytest.raises(ExportRefused) as exc:
        _ecc_seed_value(SimpleNamespace(seeds=seeds), 1)
    assert exc.value.code == "seed_index"


def test_등록의_실제_시드_0_은_받는다():
    from vlm.export_preflight import _ecc_seed_value

    assert _ecc_seed_value(SimpleNamespace(seeds={"1": 0}), 1) == 0


@pytest.mark.parametrize("updates", [
    {"seed": 8, "seed_index": 1},                       # 맞는 번호 칸이 틀린 실제 시드를 덮지 못한다
    {"seed": True}, {"seed": 7.0}, {"seed": "7"},
    {"round_idx": 1}, {"round_idx": False}, {"client_idx": False},
    {"seed": 108, "client_idx": 1},                     # base 7 · 참여자 1 = 108 — 라운드 0 · 참여자 0 이 아니다
    {"seed_index": 2}, {"seed_index": True},            # 옛 번호 칸이 있으면 같아야 한다
])
def test_신원의_시드_라운드_참여자가_어긋나면_받지_않는다(tmp_path, updates):
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _actual(tmp_path)
    path = E.step_paths(w.out, 2)["meta"]
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["identity"].update(updates)
    path.write_text(json.dumps(meta), encoding="utf-8")                 # 이 음성 사례에서만 손으로 바꾼다
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert exc.value.code == "adapter_identity"


def test_시드는_번호로_고른_값과_맞대지_시드표_어딘가에_있는지로_보지_않는다(tmp_path):
    """번호를 실제 시드에서 역산하지 않는다 — 실제 시드 7 이 시드표의 **다른 번호**에 있어도, 고른 번호의 값과 다르면 거부한다
    ('값이 시드표 어딘가에 있으면 받는다' 는 변이를 죽인다). 짝 — 고른 번호의 값이 맞으면 지난다."""
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _actual(tmp_path)
    kw["reg"].seeds = {"1": SEED + 1000, "2": SEED}                     # seed_index 는 1
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert exc.value.code == "adapter_identity" and "seed" in str(exc.value)
    kw["reg"].seeds = {"1": SEED, "2": SEED + 1000}
    _, stamp, _ = _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert stamp["check_step"] == 2


def test_시드_겹침은_참여자_검사_하나로도_막힌다(tmp_path):
    """등록 base 108 · 라운드 0 · 참여자 0 과 base 7 · 참여자 1 의 실제 시드가 같다(7 + 101 = 108). 시드 값은 같게 두고 참여자만
    어긋나게 해 `client_idx` 검사 하나가 막는 것을 고정한다(시드 검사가 대신 막지 않게)."""
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _actual(tmp_path)
    kw["reg"].seeds = {"1": 108}
    path = E.step_paths(w.out, 2)["meta"]
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["identity"].update({"seed": 108, "client_idx": 1})
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert exc.value.code == "adapter_identity" and "client_idx" in str(exc.value)


def test_조기_점검의_목적_문자열은_writer_와_점검_모듈_export_입구가_같다():
    """writer 는 점검 모듈을 가져오지 않는다 — 셋이 갈리면 모든 점검 export 가 `check_step_scope` 로 막힌다."""
    from vlm import export_writer
    from vlm.export_run import PURPOSES

    assert export_writer.ECC_PURPOSE == E.PURPOSE == "early_coord_check" and E.PURPOSE in PURPOSES


def test_등록의_시드_값만_바뀌어도_받지_않는다(tmp_path):
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _actual(tmp_path)
    before = _tree(w.out)
    kw["reg"].seeds = {"1": SEED + 1}
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert exc.value.code == "adapter_identity" and _tree(w.out) == before


@pytest.mark.parametrize("step", [True, False, 0, -1, 2.0, 2.5, "2"])
def test_요청한_step_은_파일을_찾기_전에_bool_아닌_양의_정수여야_한다(tmp_path, step):
    from vlm.export_preflight import ExportRefused, _training_check_step

    with pytest.raises(ExportRefused) as exc:
        _training_check_step(tmp_path / "없다", step=step, tag=CENTRAL_TAG, measured={}, coord_hash="", cfg=None,
                             reg=SimpleNamespace(seeds={"1": SEED}), run_root=tmp_path, seed_index=1)
    assert exc.value.code == "check_step"


@pytest.mark.parametrize("key, value", [
    ("step", 2.0), ("step", 2.5), ("step", "2"), ("step", True), ("step", 1),
    ("registered_steps", [2.0]), ("registered_steps", ["2"]), ("registered_steps", [True, 2]),
    ("registered_steps", [0, 2]), ("registered_steps", []), ("registered_steps", [1]), ("registered_steps", None),
])
def test_meta_의_step_과_등록_step_목록도_bool_아닌_양의_정수다(tmp_path, key, value):
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _actual(tmp_path)
    path = E.step_paths(w.out, 2)["meta"]
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta[key] = value
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    assert exc.value.code == "check_step"


@pytest.mark.parametrize("step", [None, True, 0, -1, 2.0, "2"])
def test_writer_는_조기_점검_도장의_step_이_양의_정수가_아니면_폴더를_만들기_전에_거부한다(tmp_path, step):
    from vlm.export_writer import ExportRefused

    folder = tmp_path / "writer"
    with pytest.raises(ExportRefused) as exc:
        _writer(folder, _synthetic_stamp(step))
    assert exc.value.code == "check_step" and not folder.exists()


@pytest.mark.parametrize("purpose, mode", [("rehearsal", "model"), ("main", "model"), ("early_coord_check", "echo")])
def test_writer_는_조기_점검_모델_모드_밖의_도장에_check_step_이_있으면_거부한다(tmp_path, purpose, mode):
    from vlm.export_writer import ExportRefused

    stamp = _synthetic_stamp(2)
    if mode == "echo":
        stamp = {"check_step": 2}                                       # 에코 도장에는 학습 출처 칸이 없다
    with pytest.raises(ExportRefused) as exc:
        _writer(tmp_path / "writer", stamp, mode=mode, purpose=purpose)
    assert exc.value.code == "check_step_scope"


@pytest.mark.parametrize("index", [True, "1", 0])
def test_writer_는_조기_점검_모델_export_의_시드_번호를_형_변환하지_않는다(tmp_path, index):
    from vlm.export_writer import ExportRefused

    with pytest.raises(ExportRefused) as exc:
        _writer(tmp_path / "writer", _synthetic_stamp(2), seed_index=index)
    assert exc.value.code == "seed_index"


def test_같은_어댑터라도_다른_step_으로는_이어_받지_않는다(tmp_path):
    from vlm.export_writer import ExportRefused

    first = _writer(tmp_path / "writer", _synthetic_stamp(2))
    first.begin()
    first.end()                                                         # 0 줄 · 미완 묶음
    before = _tree(first.folder)
    with pytest.raises(ExportRefused) as exc:
        _writer(first.folder, _synthetic_stamp(4)).begin()
    assert exc.value.code == "stamp_mismatch" and "check_step" in str(exc.value)
    assert _tree(first.folder) == before


def test_남은_도장의_step_꼴이_틀리면_이어_받지_않는다(tmp_path):
    """같은 값처럼 보이는 형(2.0 · True)을 도장에서 받아 주지 않는다 — 남은 도장을 꼴로 다시 본다."""
    from vlm.export_writer import ExportRefused, _sha

    first = _writer(tmp_path / "writer", _synthetic_stamp(2))
    first.begin()
    first.end()
    stamp = json.loads(first.p_stamp.read_text(encoding="utf-8"))
    stamp["check_step"] = 2.0
    raw = json.dumps(stamp).encode("utf-8")
    old_sha = first.stamp_sha
    first.p_stamp.write_bytes(raw)                                      # 이 음성 사례에서만 — 시도 기록의 도장 지문도 맞춘다
    first.p_att.write_bytes(first.p_att.read_bytes().replace(old_sha.encode("ascii"), _sha(raw).encode("ascii")))  # 바이트로(CR 없이)
    with pytest.raises(ExportRefused) as exc:
        _writer(first.folder, _synthetic_stamp(2)).begin()
    assert exc.value.code == "stamp_mismatch" and "점검 칸" in str(exc.value)


def test_곁_파일의_채점_step_은_도장의_값에서_낸다(tmp_path):
    """사전 점검의 곁 값이 없어도 writer 가 도장의 `check_step` 으로 채점 step 을 싣는다 — 도출 경로만 따로 고정한다."""
    writer = _writer(tmp_path / "writer", _synthetic_stamp(2))
    _finish(writer)
    meta = writer.seal({}, {})
    assert meta["scored_adapter_step"] == 2 and type(meta["scored_adapter_step"]) is int


@pytest.mark.parametrize("value", [True, 1, 2.0, "2", None])
@pytest.mark.parametrize("where", ["values", "measured"])
def test_곁_파일의_채점_step_이_도장과_다르거나_형이_틀리면_봉인하지_않는다(tmp_path, value, where):
    from vlm.export_writer import ExportRefused

    writer = _writer(tmp_path / "writer", _synthetic_stamp(2))
    _finish(writer)
    values = {"scored_adapter_step": value} if where == "values" else {}
    measured = {"scored_adapter_step": value} if where == "measured" else {}
    with pytest.raises(ExportRefused) as exc:
        writer.seal(values, measured)
    assert exc.value.code == "sidecar_clash" and not writer.p_side.exists()


# ================================================================ 한 흐름 — 실제 작성자 → 등록 · 영수증 → 공개 사전 점검 → writer
class _Measure:
    """실측 이음새 — 학습 쪽과 **같은 함수**(`processor_config_sha256`)로 이 세계의 프로세서(대역)를 연 값을 낸다. 나머지 칸은 공용 세계의 값."""

    def __init__(self, processor_kwargs):
        from tests.uni_fakes import FakeProcessor

        self.processor_config_sha256 = pilot_vlm.processor_config_sha256(FakeProcessor(**dict(processor_kwargs)))

    def __call__(self, spec):
        return {**W.MEASURED, "processor_config_sha256": self.processor_config_sha256}


def _chain_world(tmp: Path) -> SimpleNamespace:
    import pandas as pd
    from PIL import Image

    from data.manifest_io import write_snapshot
    from evaluation.actuals import impl_key
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from evaluation.prereg_unified import dump_registration
    from vlm.exposure import append_row
    from vlm.init_adapter import build_initial_adapter
    from vlm.pilot_vlm import load_pairs, train_rows_digest
    from vlm.pin import build_pin, init_probe, write_pin
    from vlm.seams import impl_id
    from vlm.train_cell import UniRunSpec, cell_dir

    ids = list(W.IDS)
    img = tmp / "img"
    img.mkdir(parents=True)
    images = {}
    for i, iid in enumerate(ids):
        Image.new("RGB", (1280, 720), (i * 40, 30, 60)).save(img / f"{iid}.png")
        images[iid] = img / f"{iid}.png"
    eval_img = img / f"{W.EVAL_ID}.png"
    Image.new("RGB", (1280, 720), (222, 0, 0)).save(eval_img)
    repo = tmp / "repo"
    repo.mkdir()
    W.git(repo, "init", "-q", "-b", "main")
    snap = tmp / "snap"
    man = [W._man(i, "train", _sha(images[i])) for i in ids] + [W._man(W.EVAL_ID, "eval", _sha(eval_img))]
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([W._ann(i) for i in ids]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    # 페어 — 학습이 실제로 읽는 꼴(골격 포함). 점검 실행의 학습 행은 본실험 중앙과 같은 "학습 분할 전체" 다
    rows = [{"image_id": iid, "split": "train", "client": "C1", "image_path": images[iid].as_posix(),
             "skeleton": {"defects": [{"type": "2011", "bbox_px": [10, 20, 30 + k, 40]}], "verdict": "불합격",
                          "clauses": []}} for k, iid in enumerate(ids)]
    pairs = tmp / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    processor_kwargs = {"max_pixels": W.MAX_PIXELS}
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
    elist = validate_eval_list(canonical_bytes(ids))
    gen_list = root / "lists" / "gen.txt"
    gen_list.write_bytes(canonical_bytes(ids))
    plan = root / "plan.yaml"
    plan.write_text("kind: early_coord_check\nversion: 1\ngen_limit: {max_new_tokens: 512}\n", encoding="utf-8")
    exposure = tmp / "exposure" / "uni_exposure_ledger.jsonl"
    append_row(exposure, {"event": "planned", "run_id": ECC_RUN, "purpose": E.PURPOSE, "tag": None, "seed_index": 1,
                          "mode": None, "list_kind": "gen", "list_path": gen_list.as_posix(),
                          "list_file_sha256": elist.file_sha256, "list_set_sha256": elist.set_sha256, "n": len(ids),
                          "snapshot_digest": digest, "commit": head, "at": "2026-10-13T09:00:00.000000+09:00"})
    code = tmp / "code"
    code.mkdir()
    W.git(code, "init", "-q", "-b", "main")
    (code / "m.py").write_text("x = 1\n", encoding="utf-8")
    W.git(code, "add", "-A")
    W.git(code, "commit", "-q", "-m", "코드")

    # ---- 실제 점검 학습(모델만 대역) — step 2 에서 멈춘다
    spec = UniRunSpec(purpose=E.PURPOSE, run_stamp=f"ecc_s1_{ECC_RUN}", seed_index=1, seed_value=SEED,
                      snapshot_digest=digest, model_id="Qwen/Qwen3.5-4B", model_revision="a" * 40,
                      pairs_path=str(pairs), pairs_digest=_sha(pairs), chat_template_kwargs={"enable_thinking": False},
                      num_rounds=1, local_epochs=2, total_epochs=2, train_root=root / "train",
                      init_adapter_path=root / "init" / "initial.npz", run_root=root, non_main_parent=parent,
                      standin_allowed=True, prompt_path=W.PROMPT_REL, processor_kwargs=dict(processor_kwargs),
                      expected_supervised_tokens=None)
    trainer = CountingLoader()
    build_initial_adapter(model_id=spec.model_id, seed=SEED, cache_path=spec.init_adapter_path,
                          revision=spec.model_revision, purpose=E.PURPOSE, model_loader=trainer, standin_allowed=True)
    probes = {"code": lambda: {"commit": "c" * 40, "dirty": []},
              "env": lambda: {"torch": "t", "cuda": None, "gpu_name": None},
              "common": lambda: {"uni_block_sha256": "u" * 64, "prompt_sha256": "q" * 64},
              "init_adapter": lambda: init_probe(spec.init_adapter_path)}
    pin_path = root / "PIN.json"
    write_pin(pin_path, build_pin(init_adapter_path=spec.init_adapter_path, probes=probes))
    res = E.run_ecc_train(spec, steps=(2, 4), registered_steps=(2, 4), pin_path=pin_path, model_loader=trainer,
                          pin_probes=probes, config_check=lambda s: [])
    assert res["exit_code"] == E.STEP_EXIT and res["step"] == 2
    cdir = cell_dir(spec, CENTRAL_TAG)

    # ---- 평가 쪽 등록 · 영수증 — 값은 위 실행이 실제로 낸 것과 이 세계의 실측
    meta = json.loads(E.step_paths(cdir, 2)["meta"].read_text(encoding="utf-8"))
    tc_text = meta["train_config"]
    tc = json.loads(tc_text)
    measure, loader = _Measure(processor_kwargs), W.Loader()
    w = SimpleNamespace(tmp=tmp, repo=repo, head=head, snap=snap, digest=digest, parent=parent, root=root,
                        lists={"model": gen_list}, elist=elist, plan=plan, exposure=exposure, images=images, code=code,
                        adapter=cdir, spec=spec, measure=measure, loader=loader, trainer=trainer)

    def edit(r):
        g = r.generation
        g.purpose, g.list_split = E.PURPOSE, "train"
        g.train_rows_digest = train_rows_digest(load_pairs("train", None, pairs_path=pairs))
        g.plan_sha256 = _sha(plan)
        g.processor_config_sha256 = measure.processor_config_sha256
        g.init_adapter_digest = init_probe(spec.init_adapter_path)["init_adapter_digest"]
        g.train_config_sha256 = hashlib.sha256(tc_text.encode("utf-8")).hexdigest()
        g.budget_n, g.budget_r, g.budget_e = tc["total_epochs"], tc["num_rounds"], tc["local_epochs"]
        g.impl_ids = tuple(sorted(impl_key(impl_id(fn, seam=s, approved=False))
                                  for fn, s in ((measure, "export_measure"), (loader, "export_generator"))))
        r.seeds = {"1": SEED, "2": SEED + 1000}                         # 2 번은 다른 실제 시드 — 번호로 고르는지 본다
    w.reg = W.registration(w, edit=edit)
    d = root / "registration"
    d.mkdir()
    raw_reg = dump_registration(w.reg)
    name = "early_coord_check-20261013-1"
    (d / f"{name}.json").write_bytes(raw_reg)
    (d / "train_config.json").write_bytes(tc_text.encode("utf-8"))
    receipt = {"generation_sha256": w.reg.generation_sha256(), "scoring_sha256": w.reg.scoring_sha256(),
               "registered_at": "2026-10-13T09:00:00+09:00", "main_commit": head, "kind": E.PURPOSE,
               "registration_path": (d / f"{name}.json").relative_to(repo).as_posix(),
               "registration_file_sha256": hashlib.sha256(raw_reg).hexdigest()}
    w.receipt = d / f"{name}.receipt.json"
    w.receipt.write_text(json.dumps(receipt), encoding="utf-8")
    return w


def _clock():
    from datetime import datetime

    return W.Clock(start=datetime(2026, 10, 13, 12, 0, 0, tzinfo=W.KST))


@pytest.fixture
def chain(tmp_path):
    return _chain_world(tmp_path)


def _checkpoint_bytes(w) -> dict[str, str]:
    """원 체크포인트와 원장 — export 가 바꾸면 안 되는 것(비용 저널은 공식 CLI 가 덧붙이므로 뺀다)."""
    return {k: v for k, v in _tree(w.adapter).items() if not k.startswith(E.BUDGET_FILE)}


def test_한_흐름_실제_점검_체크포인트가_공개_사전_점검과_run_export_를_지나_봉인된다(chain):
    before = _checkpoint_bytes(chain)
    ident = json.loads(E.step_paths(chain.adapter, 2)["meta"].read_text(encoding="utf-8"))["identity"]
    assert "seed_index" not in ident and (ident["seed"], ident["round_idx"], ident["client_idx"]) == (SEED, 0, 0)
    res = W.run(chain, purpose="early_coord_check", tag=CENTRAL_TAG, adapter_dir=chain.adapter, check_step=2,
                folder=chain.root / "export", standin_allowed=True, clock=_clock(), measure_fn=chain.measure,
                loader=chain.loader)
    assert res.sealed and res.n_written == len(W.IDS) and chain.loader.calls == 1
    assert res.meta["purpose"] == "early_coord_check" and res.meta["scored_adapter_step"] == 2
    stamp = json.loads((chain.root / "export" / f"{CENTRAL_TAG}_s1.export_start.json").read_text(encoding="utf-8"))
    assert stamp["check_step"] == 2 and stamp["train_ledger_sha256"] == _sha(chain.adapter / E.LEDGER_FILE)
    used = Path(chain.loader.cfg.adapter_path)                          # 생성기가 읽는 어댑터 = 사전 점검이 검사한 그 파일
    assert used == E.step_paths(chain.adapter, 2)["adapter"] and _sha(used) == stamp["scored_adapter_sha256"]
    assert _checkpoint_bytes(chain) == before


def test_한_흐름_다른_시드_번호로는_같은_체크포인트를_내보내지_않는다(chain):
    """등록의 시드표 2 번은 다른 실제 시드(1007)다 — 시드 1(실제 시드 7)로 학습한 체크포인트를 2 번으로 내보내지 않는다. 시드표에
    있는 번호라 번호 검사가 아니라 신원 대조(`adapter_identity`)에서 멈추고, 생성기를 올리지 않는다."""
    from vlm.export_run import ExportRefused

    with pytest.raises(ExportRefused) as exc:
        W.run(chain, purpose="early_coord_check", tag=CENTRAL_TAG, adapter_dir=chain.adapter, check_step=2,
              folder=chain.root / "export", standin_allowed=True, clock=_clock(), measure_fn=chain.measure,
              loader=chain.loader, seed_index=2)
    assert exc.value.code == "adapter_identity" and "seed" in str(exc.value) and chain.loader.calls == 0


def test_한_흐름_공식_CLI_의_계획된_멈춤_뒤_재호출이_비용_검사를_지나_이어_받아_봉인한다(chain, capsys):
    """§137 C-01 과 I-04 를 함께 — 공식 CLI 는 부를 때마다 비용 저널에 한 줄을 덧붙이지만 원장 지문은 그대로라 writer 가 이어 받는다."""
    import scripts.export_uni as X

    folder = chain.root / "export_cli"
    argv = ["--purpose", "early_coord_check", "--check-step", "2", "--mode", "model", "--tag", CENTRAL_TAG,
            "--seed-index", "1", "--receipt", str(chain.receipt), "--list", str(chain.lists["model"]),
            "--snapshot", str(chain.snap), "--folder", str(folder), "--run-root", str(chain.root),
            "--plan", str(chain.plan), "--adapter-dir", str(chain.adapter), "--device", "cpu:0",
            "--spent-hours", "1", "--next-hours", "1"]
    clock = _clock()
    seams = {"measure_env": chain.measure, "load_generator": chain.loader, "image_path_of": chain.images,
             "repo": chain.repo, "code_repo": chain.code, "exposure_ledger": chain.exposure,
             "non_main_parent": chain.parent, "clock": clock, "standin_allowed": True}
    before = _checkpoint_bytes(chain)
    assert X.main([*argv, "--stop-after-lines", "2"], **seams) == X.EXIT_OK
    assert "stopped attempt=1 reason=first_pass n_written=2" in capsys.readouterr().out
    assert X.main(argv, **seams) == X.EXIT_OK
    assert "sealed attempt=2 reason=resume_after_interruption n_written=2" in capsys.readouterr().out
    side = json.loads((folder / f"{CENTRAL_TAG}_s1.export_meta.json").read_text(encoding="utf-8"))
    assert side["scored_adapter_step"] == 2 and side["attempts"] == 2
    starts = [json.loads(ln) for ln in (folder / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [r for r in starts if r["event"] == "start"]
    assert len(starts) == 2 and len({r["start_stamp_sha256"] for r in starts}) == 1   # 같은 도장을 이어 받았다
    exports = [r for r in E.budget_rows(chain.adapter) if r["detail"]["what"] == "export"]
    assert [r["event"] for r in exports] == ["budget_ok", "budget_ok"]  # 두 호출 모두 비용 가드를 지났다
    assert {r["source"] for r in exports} == {E.BUDGET_FILE}           # 원장이 아니라 비용 저널에
    assert _checkpoint_bytes(chain) == before                           # 원장 · 체크포인트 바이트는 그대로
