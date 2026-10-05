"""C2 · C1 읽기 검토의 일곱 항목을 고친 자리 — 항목마다 음성 사례를 둔다.

1. 본실험 목적의 필수 설정이 비었거나 설정 파일과 다르면 **적재 전에** 거부한다(파일럿 기본값 · 옛 실행 설정으로 내려가지 않는다).
2. 프로세서 설정이 전달 경로와 재개 신원 · 멱등 신원에 든다.
3. 멱등 판정과 원장 복구가 `identity` 와 `identity_sha256` 을 함께 맞댄다.
4. 초기 어댑터 proof 가 캐시의 배열과 결속된다.
5. 재개 로더는 찢어진 파일만 건너뛴다. 저장 실패는 `after_ckpt` 로 이어지지 않는다. 저장은 fsync 하고 기존 파일을 덮지 않는다.
6. 감독 토큰의 기대값을 저장 전에 맞댄다.

모델 없이 돈다(`tests/uni_fakes.py`, CPU). 7번(시험 표현)은 `tests/test_vlm_chat_template.py` 와 `tests/test_uni_train_cell.py` 에 있다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_uni_train_cell import PLAN, _die_at, _Die, _init, _meta, _rows, _spec, _tree_bytes, reh  # noqa: E402
from tests.uni_fakes import CountingLoader, make_rows  # noqa: E402
from vlm.train_cell import (LOCAL_TAGS, CellRejected, cell_dir, close_ledger,  # noqa: E402
                            read_ledger_rows, run_uni_local_cell)

PROMPT = "vlm/prompts/unified_v2_absorig.txt"


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch

    was = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    assert was or not torch.cuda.is_initialized(), "이 시험이 CUDA 를 초기화했다"


@pytest.fixture
def world(tmp_path):
    loader = CountingLoader()
    spec = _spec(tmp_path)
    _init(spec, loader)
    return SimpleNamespace(tmp=tmp_path, spec=spec, loader=loader, rows=_rows(tmp_path))


def pairs_snapshot(tmp: Path) -> str:
    """임시 페어 파일과 그 계약(`SNAPSHOT.sha256`)을 쓰고 digest 를 돌려준다 — 계약의 산식 그대로."""
    import hashlib

    pairs = tmp / "pairs.jsonl"
    if not pairs.exists():
        pairs.write_text('{"image_id": "x"}' + "\n", encoding="utf-8")
    h = hashlib.sha256(pairs.read_bytes()).hexdigest()
    digest = hashlib.sha256(h.encode()).hexdigest()
    (tmp / "SNAPSHOT.sha256").write_text(f"{h}  pairs.jsonl" + "\n" + f"# snapshot_digest {digest}" + "\n",
                                         encoding="utf-8")
    return digest


def _main_yaml(tmp: Path, **over) -> Path:
    fixed = {"snapshot_digest": "s" * 64,
             "uni_train_budget": {"num_rounds": 2, "local_epochs": 1, "total_epochs": 2},
             "uni_model": {"id": "tiny", "revision": "r1"},
             "uni_pairs": {"path": (tmp / "pairs.jsonl").as_posix(), "digest": pairs_snapshot(tmp)},
             "uni_prompt_sha256": None, "uni_chat_template_kwargs": {"enable_thinking": False},
             "uni_max_new_tokens": None, "uni_batch_size": 1,
             "uni_processor_kwargs": {}, "uni_expected_supervised_tokens": {"C1": 1, "C2": 1, "C3": 1},
             "uni_prompt_path": PROMPT, "uni_coord_space": "ABS_ORIG"}
    fixed.update(over)
    import yaml

    p = tmp / "base_main.yaml"
    p.write_text(yaml.safe_dump({"fixed_before_main_runs": fixed, "experiment": {"seeds": [7, 8, 9]}},
                                allow_unicode=True), encoding="utf-8")
    return p


def _main_spec(tmp: Path, **over):
    kw = dict(purpose="main", plan_sha256=None, prompt_path=PROMPT, processor_kwargs={}, run_root=None,
              expected_supervised_tokens={"C1": 1, "C2": 1, "C3": 1}, config_path=_main_yaml(tmp),
              pairs_digest=pairs_snapshot(tmp))
    kw.update(over)
    return _spec(tmp, **kw)


# ================================================================ 1. 본실험의 필수 설정
def test_1_연합_키가_비면_본실험은_받지_않고_리허설은_받는다():
    from fl.uni_run_config import MAIN_REQUIRED, UniMainIncomplete, client_run_cfg, down_config

    with pytest.raises(UniMainIncomplete) as exc:
        client_run_cfg(down_config(lambda k, d: d))
    for k in MAIN_REQUIRED:
        assert k in str(exc.value)
    got = client_run_cfg(down_config(lambda k, d: {"purpose": "rehearsal", "plan": "x"}.get(k, d)))
    assert got["model_id"] is None                      # 리허설은 파일럿 기본값으로 내려갈 수 있다


@pytest.mark.parametrize("field", ["model_revision", "prompt_path", "processor_kwargs",
                                   "expected_supervised_tokens"])
def test_1_칸의_본실험_필수_칸이_비면_적재_전에_멈춘다(tmp_path, field):
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path, **{field: None}))
    assert exc.value.code == "main_incomplete" and field in str(exc.value)


@pytest.mark.parametrize("over, why", [
    ({"num_rounds": 1, "local_epochs": 2}, "budget"),
    ({"seed_value": 8}, "seed_value"),
    ({"model_revision": "r2"}, "model_revision"),
    ({"pairs_digest": "q" * 64}, "pairs_digest"),
    ({"snapshot_digest": "t" * 64}, "snapshot_digest"),
    ({"processor_kwargs": {"max_pixels": 1}}, "processor_kwargs"),
    ({"expected_supervised_tokens": {"C1": 2, "C2": 1, "C3": 1}}, "expected_supervised_tokens"),
])
def test_1_본실험_값이_설정_파일과_다르면_적재_전에_멈춘다(tmp_path, over, why):
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path, **over))
    assert exc.value.code == "main_config_mismatch" and why in str(exc.value)


def test_1_설정과_같은_본실험은_가드를_지나_초기_어댑터에서_멈춘다(tmp_path):
    """모델을 올리지 않고 확인한다 — 가드를 다 지난 뒤 첫 입력(공통 초기 어댑터)이 없어서 멈춘다."""
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path))
    assert exc.value.code == "init_adapter_missing"


def test_1_저장소의_설정으로는_본실험이_시작하지_않는다(tmp_path):
    """저장소 설정의 **지금 값**에 기대지 않는다 — 다른 쪽이 키를 채울 때마다 깨지지 않게, 설정을 읽어 실제로 빈 칸과 맞댄다.
    빈 칸이 있으면 그 이름들이 정확히 까닭으로 나오고, 다 채워졌으면 값 대조(시험 실행의 값 ≠ 저장소의 값)로 멈춘다."""
    import re

    from vlm.uni_config import UniConfig, load_uni_config

    cfg = load_uni_config(None)
    empty = {n for n in UniConfig.NEEDS["train"] if getattr(cfg, n) in (None, (), "")}
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path, config_path=None))
    assert exc.value.code == "main_config_mismatch"
    named = set(re.findall(r"([A-Za-z_]+) 이 비어 있다", str(exc.value)))
    assert named == empty, (named, empty)


def test_1_설정에_키가_없으면_그_이름이_까닭으로_나온다(tmp_path):
    """보고의 "키가 없으면 본실험이 시작하지 못한다" 가 코드의 동작이다 — 저장소 설정이 아니라 두 키를 뺀 합성 설정으로 본다."""
    import yaml

    doc = yaml.safe_load(_main_yaml(tmp_path).read_text(encoding="utf-8"))
    for key in ("uni_processor_kwargs", "uni_expected_supervised_tokens"):
        del doc["fixed_before_main_runs"][key]
    p = tmp_path / "base_missing_two.yaml"          # `_main_spec` 이 `base_main.yaml` 을 다시 쓰므로 다른 이름에 둔다
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path, config_path=p))
    assert exc.value.code == "main_config_mismatch"
    for name in ("processor_kwargs", "expected_supervised_tokens"):
        assert f"{name} 이 비어 있다" in str(exc.value)


def test_1_프로세서_설정이_등록값과_다르면_연합_서버도_멈춘다(tmp_path):
    from fl.uni_fed import UniFedRejected, UniFedRun

    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=_fed_get(tmp_path, **{"uni-processor-kwargs": '{"max_pixels": 7}'}), num_rounds=2,
                  total_epochs=2, out_dir=tmp_path / "fl" / "uni_fed", run_id="x", base_seed=7,
                  split_hash="s" * 64, local_epochs=1, client_tags=["C1", "C2", "C3"],
                  config_path=_main_yaml(tmp_path))
    assert exc.value.code == "main_config_mismatch" and "processor_kwargs" in str(exc.value)


def test_1_본실험_출력이_비본실험_부모_아래면_멈춘다(tmp_path):
    under = ROOT / "outputs" / "rehearsal_u" / "reh-x" / "train"      # 경로만 쓴다 — 파일을 만들지 않는다
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(spec=_main_spec(tmp_path, train_root=under))
    assert exc.value.code == "main_config_mismatch" and "비본실험 부모" in str(exc.value)
    assert not under.exists()


def _fed_get(tmp: Path, **over):
    vals = {"uni-model": "tiny", "uni-model-revision": "r1", "uni-pairs": (tmp / "pairs.jsonl").as_posix(),
            "uni-pairs-digest": pairs_snapshot(tmp), "uni-prompt": PROMPT,
            "uni-chat-template-kwargs": '{"enable_thinking": false}', "uni-coord-space": "ABS_ORIG",
            "uni-seed-index": "1", "uni-processor-kwargs": "{}",
            "uni-expected-supervised-tokens": '{"C1": 1, "C2": 1, "C3": 1}', "purpose": "main"}
    vals.update(over)
    return lambda k, d: vals.get(k, d)


def test_1_연합_서버도_본실험_설정을_적재_전에_맞댄다(tmp_path):
    from fl.uni_fed import UniFedRejected, UniFedRun

    kw = dict(out_dir=tmp_path / "fl" / "uni_fed", run_id="x", base_seed=7, split_hash="s" * 64,
              local_epochs=1, client_tags=["C1", "C2", "C3"], config_path=_main_yaml(tmp_path))
    UniFedRun(get=_fed_get(tmp_path), num_rounds=2, total_epochs=2, **kw)         # 같으면 선다
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=_fed_get(tmp_path), num_rounds=3, total_epochs=3, **kw)     # 파일럿 기본값 R=3 같은 옛 값
    assert exc.value.code == "main_config_mismatch" and "budget" in str(exc.value)
    with pytest.raises(UniFedRejected) as exc:
        UniFedRun(get=_fed_get(tmp_path, **{"uni-model-revision": "r0"}), num_rounds=2, total_epochs=2, **kw)
    assert "model_revision" in str(exc.value)


# ================================================================ 2. 프로세서 설정
def test_2_프로세서_설정이_바뀌면_완료_산출물을_건너뛰지_않는다(world):
    run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    n = world.loader.n
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=_spec(world.tmp, processor_kwargs={"max_pixels": 1234}),
                           model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_mismatch" and "processor_kwargs" in str(exc.value)
    assert world.loader.n == n


def test_2_프로세서_해시만_다르면_재개_신원이_적용_전에_거부한다(world):
    """실행 신원의 **적재 전 블록**(프로세서 설정 포함)은 같고 **프로세서가 낸 설정만** 다르다. 재개 신원은 전체 신원
    해시 안에 들므로 그 해시도 다르다 — 이 시험은 재개 신원의 필드 대조(`latest_resume`)가 전체 해시 대조(`CellRejected`)보다
    **먼저** 잡는 것을 본다. 그 필드를 재개 신원에서 빼면 두 신원이 같아져 이 시험이 떨어진다(검수 14번 M-5 로 문장을 고쳤다)."""
    from tests.uni_fakes import fake_loader

    def other_processor(*a, **kw):
        model, proc = fake_loader(*a, **kw)
        proc.image_processor.cfg["patch_size"] = 14
        return model, proc

    spec = world.spec
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows,
                           hooks={"after_ckpt": _die_at(0)})
    resume = Path(spec.resume_root)
    before = _tree_bytes(resume)
    with pytest.raises(ValueError) as exc:
        run_uni_local_cell(["C1"], spec=spec, model_loader=CountingLoader(other_processor),
                           rows_by_client=world.rows)
    assert "processor_config_sha256" in str(exc.value) and not isinstance(exc.value, CellRejected)
    assert _tree_bytes(resume) == before


# ================================================================ 3. identity 와 그 해시
def test_3_meta_의_신원_해시가_깨졌거나_없으면_건너뛰지_않는다(world):
    run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    meta_p = cell_dir(world.spec, LOCAL_TAGS["C1"]) / "adapter_last.meta.json"
    good = meta_p.read_bytes()
    m = json.loads(good)
    m["identity_sha256"] = "0" * 64
    meta_p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_sha_broken"
    del m["identity_sha256"]
    meta_p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_missing"
    meta_p.write_bytes(good)
    assert run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader,
                              rows_by_client=world.rows)["C1"].status == "skipped"


def _die_before_end_row(world, monkeypatch):
    """저장 뒤 · 끝 행 전에 멈춘 칸을 만든다. `monkeypatch.undo()` 는 CUDA 차단까지 풀므로 쓰지 않는다."""
    from fl.atomic_log import AtomicLog

    real = AtomicLog.close

    def boom(self, **kw):
        raise _Die("끝 행 직전")

    monkeypatch.setattr(AtomicLog, "close", boom)
    with pytest.raises(_Die):
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    monkeypatch.setattr(AtomicLog, "close", real)


def _resign(meta_p: Path, m: dict) -> None:
    """meta 의 신원을 고친 뒤 해시를 다시 매긴다 — 기록의 온전함 검사를 지나게 해 내용 검사만 떨어뜨린다."""
    from vlm.pilot_vlm import canonical_json, sha256_text

    m["identity_sha256"] = sha256_text(canonical_json(m["identity"]))
    meta_p.write_text(json.dumps(m), encoding="utf-8")


def test_3_meta_의_재개_신원이_훼손되면_건너뛰지_않는다(world):
    run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    meta_p = cell_dir(world.spec, LOCAL_TAGS["C1"]) / "adapter_last.meta.json"
    good = meta_p.read_bytes()
    n = world.loader.n
    m = json.loads(good)
    m["identity"]["resume"]["prompt_sha256"] = "0" * 64          # 해시는 그대로 — 기록이 깨졌다
    meta_p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_sha_broken"
    _resign(meta_p, m)                                           # 해시까지 맞춰도 재개 신원이 요청과 다르다
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_mismatch" and "prompt_sha256" in str(exc.value)
    del m["identity"]["resume"]
    _resign(meta_p, m)
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "resume_identity_missing"
    assert world.loader.n == n
    meta_p.write_bytes(good)


def test_3_목적을_바꾼_요청은_원장을_닫지_못한다(world, monkeypatch):
    _die_before_end_row(world, monkeypatch)
    led = cell_dir(world.spec, LOCAL_TAGS["C1"]) / "train_ledger.csv"
    before = led.read_bytes()
    # 목적을 바꾸면 루트의 종류(`reh-*` ↔ `fdiag-*`)가 먼저 막는다 — 신원 검사까지 가지 않는다(2판 §1-4 의 순서 4).
    other = _spec(world.tmp, purpose="frame_diag", standin_allowed=True)
    with pytest.raises(CellRejected) as exc:
        close_ledger(other, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader)
    assert exc.value.code == "run_root" and "fdiag" in str(exc.value)
    with pytest.raises(CellRejected):
        close_ledger(world.spec, LOCAL_TAGS["C1"], rows=world.rows["C1"][:2], model_loader=world.loader)
    assert led.read_bytes() == before
    assert close_ledger(world.spec, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader) == 2


def test_3_모델_판을_바꾼_요청도_원장을_닫지_못한다(world, monkeypatch):
    """판이 맞는 초기 어댑터 캐시를 따로 두어 캐시 대조를 지나게 한다 — **복구의 신원 검사**가 막는다."""
    _die_before_end_row(world, monkeypatch)
    led = cell_dir(world.spec, LOCAL_TAGS["C1"]) / "train_ledger.csv"
    before = led.read_bytes()
    other = _spec(world.tmp, model_revision="r2", init_adapter_path=reh(world.tmp) / "init_r2" / "initial.npz")
    _init(other, world.loader)
    with pytest.raises(CellRejected) as exc:
        close_ledger(other, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader)
    assert exc.value.code == "identity_mismatch" and "model_revision" in str(exc.value)
    assert led.read_bytes() == before


# ================================================================ 4. 초기 어댑터 proof 결속
def test_4_배열_파일만_바뀐_캐시는_옛_proof_로_지나지_않는다(world):
    from vlm.init_adapter import InitCacheRejected, build_initial_adapter

    cache = world.spec.init_adapter_path
    with np.load(cache) as z:
        keys = list(z.files)
        arrays = [z[k] for k in keys]
    np.savez(cache, **{k: a + 1.0 for k, a in zip(keys, arrays)})     # 같은 키 · 모양, 다른 값
    with pytest.raises(InitCacheRejected, match="파일이 바뀌었다"):
        build_initial_adapter(model_id="tiny", seed=7, cache_path=cache, revision="r1", purpose="rehearsal",
                              model_loader=world.loader)
    with pytest.raises(InitCacheRejected):
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)


def test_4_본실험은_증빙이나_구현_식별자_필드가_빈_proof_를_받지_않는다(tmp_path):
    from vlm.init_adapter import InitCacheRejected, adapter_proof, build_initial_adapter, init_adapter_digest

    cache = tmp_path / "initial.npz"
    arr = [np.ones(3, np.float32)]
    np.savez(cache, k=arr[0])
    impl = {"seam": "model_loader", "approved": True, "module": "vlm.pilot_vlm", "qualname": "_load_model",
            "source_path": "vlm/pilot_vlm.py", "blob_sha1": "b" * 40}
    full = {"model_id": "M", "seed": 1, "revision": None, "purpose": "main", "impl_ids": [impl],
            "init_adapter_digest": init_adapter_digest(arr), **adapter_proof(arr, ["k"])}
    proof_p = cache.with_suffix(".proof.json")
    for broken in ({**full, "impl_ids": [{**impl, "blob_sha1": ""}]},
                   {k: v for k, v in full.items() if k != "init_adapter_digest"},
                   {k: v for k, v in full.items() if k != "tensor_digest"},
                   {**full, "keys_digest": "0" * 64}):
        proof_p.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(InitCacheRejected):
            build_initial_adapter(model_id="M", seed=1, cache_path=cache, purpose="main")
    proof_p.write_text(json.dumps(full), encoding="utf-8")
    assert build_initial_adapter(model_id="M", seed=1, cache_path=cache, purpose="main")[1] == ["k"]


# ================================================================ 5. 재개 로더 · 저장 실패
def _state_dir(tmp: Path) -> tuple[Path, object]:
    from detection.resume import ResumeCheckpointer
    from vlm.resume_uni import UniResumeIdentity

    import torch

    class _T:
        def __init__(self):
            self.model = torch.nn.Linear(2, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.epoch, self.start_epoch, self.scaler, self.train_loader = 0, 0, None, None

    ident = UniResumeIdentity(run_id="r", round_idx=0, client_idx=0, seed=1, total_epochs=2, local_epochs=2,
                              model="m", data="d", cell="uni_local_C1", processor_config_sha256="p")
    d = tmp / "res"
    ResumeCheckpointer(d, identity=ident).save(_T())
    return d, ident


def test_5_형식이_다른_재개_파일은_건너뛰지_않고_멈춘다(tmp_path):
    import torch

    from detection.resume import ResumeRejected, latest_resume

    d, ident = _state_dir(tmp_path)
    good = d / "resume_ep0000.pt"
    torch.save({"format": "other/9", "identity": {}}, d / "resume_ep0001.pt")
    before = _tree_bytes(d)
    with pytest.raises(ResumeRejected, match="형식"):
        latest_resume(d, identity=ident)
    assert _tree_bytes(d) == before and good.exists()


def test_5_신원을_해석할_수_없는_재개_파일은_멈춘다(tmp_path):
    from detection.resume import ResumeIdentity, ResumeRejected, latest_resume

    d, ident = _state_dir(tmp_path)
    # 통합형 신원으로 쓴 파일을 검출 신원으로 읽으면 모르는 키가 있다 — 물러나 처음부터 학습하지 않는다
    with pytest.raises(ResumeRejected, match="신원을 해석할 수 없다"):
        latest_resume(d, identity_cls=ResumeIdentity)


def test_5_찢어진_최신본만_건너뛴다(tmp_path):
    from detection.resume import latest_resume

    d, ident = _state_dir(tmp_path)
    (d / "resume_ep0001.pt").write_bytes(b"PK\x03\x04 torn")
    assert latest_resume(d, identity=ident).epoch_done == 0


def test_5_칸은_형식이_다른_재개_파일_앞에서_처음부터_학습하지_않는다(world):
    import torch

    from detection.resume import ResumeRejected

    rdir = Path(world.spec.resume_root) / f"{LOCAL_TAGS['C1']}_s1"
    rdir.mkdir(parents=True)
    torch.save({"format": "weld-fl-resume/0"}, rdir / "resume_ep0000.pt")
    before = _tree_bytes(rdir)
    with pytest.raises(ResumeRejected):
        run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    assert _tree_bytes(rdir) == before                    # 기존 체크포인트를 덮지 않았다


def test_5_저장이_실패한_epoch_에서는_after_ckpt_가_불리지_않는다(world, monkeypatch):
    from detection.resume import ResumeCheckpointer

    real = ResumeCheckpointer.save
    calls: list[int] = []

    def flaky(self, trainer):
        if int(trainer.epoch) == 0:
            raise OSError("디스크 가득")
        return real(self, trainer)

    monkeypatch.setattr(ResumeCheckpointer, "save", flaky)
    res = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows,
                             hooks={"after_ckpt": lambda ep, ck: calls.append(ep)})
    assert calls == [1]
    assert _meta(res["C1"].cell_dir)["metrics"]["resume_save_failures"] == 1


def test_5_저장은_fsync_뒤에_교체하고_같은_이름을_덮지_않는다(tmp_path, monkeypatch):
    import os

    import detection.resume as rs

    order: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(rs.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(rs.os, "replace", lambda a, b: (order.append(f"replace:{Path(b).name}"), real_replace(a, b))[1])
    d, ident = _state_dir(tmp_path)
    assert order == ["fsync", "replace:resume_ep0000.pt"]
    first = (d / "resume_ep0000.pt").read_bytes()
    _state_dir(tmp_path)                                    # 같은 epoch 를 다시 저장
    assert (d / "resume_ep0000.pt").read_bytes() == first   # 옛 파일은 그 자리 그대로
    assert sorted(p.name for p in d.glob("resume_ep*.pt")) == ["resume_ep0000.pt", "resume_ep0000.r001.pt"]
    from detection.resume import latest_resume

    assert latest_resume(d, identity=ident).path.name == "resume_ep0000.r001.pt"   # 새것이 먼저 읽힌다


def test_5_같은_이름_저장이_실패해도_옛_정상본이_재개_후보로_남는다(tmp_path, monkeypatch):
    """결정표 8번 — 기존 체크포인트는 지우지 않을 뿐 아니라 **후보에서도 빠지지 않는다.** 교체 도중 죽거나 교체가
    실패한 경우를 흉내 낸다. 앞 판은 옛 파일을 `.replaced-k` 로 먼저 옮겨 이 경우 후보가 비었다."""
    import os

    import detection.resume as rs
    from detection.resume import latest_resume

    d, ident = _state_dir(tmp_path)
    real_replace = os.replace

    def fail_on_new(a, b):
        if Path(a).suffix == ".tmp":
            raise OSError("교체 실패")
        return real_replace(a, b)

    monkeypatch.setattr(rs.os, "replace", fail_on_new)
    with pytest.raises(OSError):
        _state_dir(tmp_path)
    monkeypatch.setattr(rs.os, "replace", real_replace)
    state = latest_resume(d, identity=ident)
    assert state is not None and state.path.name == "resume_ep0000.pt"


def test_5_정리는_이번_프로세스가_저장한_파일만_한다(tmp_path):
    from detection.resume import ResumeCheckpointer

    import torch

    class _T:
        def __init__(self, ep):
            self.model = torch.nn.Linear(2, 2)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            self.epoch, self.start_epoch, self.scaler, self.train_loader = ep, 0, None, None

    d, ident = _state_dir(tmp_path)                          # 앞 프로세스의 ep0000
    ck = ResumeCheckpointer(d, identity=ident, keep=1)
    for ep in (1, 2, 3):
        ck.save(_T(ep))
    assert sorted(p.name for p in d.glob("resume_ep*.pt")) == ["resume_ep0000.pt", "resume_ep0003.pt"]


# ================================================================ 6. 감독 토큰의 기대값
def test_6_같은_폴더에서_기대값만_바꾸면_건너뛰지_않는다(world, monkeypatch):
    probe = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    per_epoch = _meta(probe["C1"].cell_dir)["supervised_tokens"] // world.spec.total_epochs
    spec = _spec(world.tmp, train_root=reh(world.tmp) / "train_t", resume_root=reh(world.tmp) / "resume_t",
                 expected_supervised_tokens={"C1": per_epoch, "C2": 1, "C3": 1})
    run_uni_local_cell(["C1"], spec=spec, model_loader=world.loader, rows_by_client=world.rows)
    n = world.loader.n
    changed = _spec(world.tmp, train_root=reh(world.tmp) / "train_t", resume_root=reh(world.tmp) / "resume_t",
                    expected_supervised_tokens={"C1": per_epoch + 1, "C2": 1, "C3": 1})
    with pytest.raises(CellRejected) as exc:
        run_uni_local_cell(["C1"], spec=changed, model_loader=world.loader, rows_by_client=world.rows)
    assert exc.value.code == "identity_mismatch" and "expected_supervised_tokens" in str(exc.value)
    assert world.loader.n == n


def test_6_원장_닫기도_감독_토큰을_맞댄다(world, monkeypatch):
    probe = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    per_epoch = _meta(probe["C1"].cell_dir)["supervised_tokens"] // world.spec.total_epochs
    spec = _spec(world.tmp, train_root=reh(world.tmp) / "train_c", resume_root=reh(world.tmp) / "resume_c",
                 expected_supervised_tokens={"C1": per_epoch, "C2": 1, "C3": 1})
    from types import SimpleNamespace as NS

    _die_before_end_row(NS(spec=spec, loader=world.loader, rows=world.rows), monkeypatch)
    d = cell_dir(spec, LOCAL_TAGS["C1"])
    led = d / "train_ledger.csv"
    before = led.read_bytes()
    changed = _spec(world.tmp, train_root=reh(world.tmp) / "train_c", resume_root=reh(world.tmp) / "resume_c",
                    expected_supervised_tokens={"C1": per_epoch + 1, "C2": 1, "C3": 1})
    with pytest.raises(CellRejected) as exc:
        close_ledger(changed, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader)
    assert exc.value.code == "identity_mismatch"
    meta_p = d / "adapter_last.meta.json"
    good = meta_p.read_bytes()
    m = json.loads(good)
    m["supervised_tokens"] += 1                               # 신원 밖의 실측값만 어긋난 기록
    meta_p.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(CellRejected) as exc:
        close_ledger(spec, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader)
    assert exc.value.code == "close_tokens"
    assert led.read_bytes() == before
    meta_p.write_bytes(good)
    assert close_ledger(spec, LOCAL_TAGS["C1"], rows=world.rows["C1"], model_loader=world.loader) == 2


def test_6_감독_토큰이_기대값과_다르면_저장하지_않는다(world):
    probe = run_uni_local_cell(["C1"], spec=world.spec, model_loader=world.loader, rows_by_client=world.rows)
    per_epoch = _meta(probe["C1"].cell_dir)["supervised_tokens"] // world.spec.total_epochs
    good = _spec(world.tmp, train_root=reh(world.tmp) / "train_ok", resume_root=reh(world.tmp) / "resume_ok",
                 expected_supervised_tokens={"C1": per_epoch, "C2": 1, "C3": 1})
    ok = run_uni_local_cell(["C1"], spec=good, model_loader=world.loader, rows_by_client=world.rows)
    assert _meta(ok["C1"].cell_dir)["supervised_tokens_expected"] == per_epoch * world.spec.total_epochs
    bad = _spec(world.tmp, train_root=reh(world.tmp) / "train_bad", resume_root=reh(world.tmp) / "resume_bad",
                expected_supervised_tokens={"C1": per_epoch + 1, "C2": 1, "C3": 1})
    with pytest.raises(RuntimeError, match="감독 토큰"):
        run_uni_local_cell(["C1"], spec=bad, model_loader=world.loader, rows_by_client=world.rows)
    d = cell_dir(bad, LOCAL_TAGS["C1"])
    assert not (d / "adapter_last.npz").exists()
    assert read_ledger_rows(d / "train_ledger.csv")[-1]["metric_name"] != "final_adapter_step"
