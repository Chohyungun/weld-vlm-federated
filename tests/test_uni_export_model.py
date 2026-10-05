"""모델 생성기의 실제 구현(`vlm/export_model.py`) — 합성 대역 모델로 본다. 실물 검증은 GPU 창의 몫이다.

대역은 `tests/uni_fakes.py` 의 작은 peft 모델(어댑터 상태 사전이 실제 peft 함수로 오간다)에 정해진 토큰을 내는 `generate` 를 얹은 것과,
학습 대역과 같은 렌더(`FakeProcessor`)에 토크나이저 · 패치 크기를 더한 프로세서다. HF 캐시에 기대지 않는다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests import uni_export_world as W
from tests.uni_fakes import FakeProcessor, fake_loader
from vlm.export_preflight import GenerationConfig

TARGET = '{"defects": [{"iso_code": "2011", "bbox_2d": [10, 20, 40, 55]}], "verdict": "합격", "cited_clauses": []}'
EOS = 2


class Tok:
    eos_token_id = EOS

    def __init__(self):
        self.padding_side = "right"
        self.vocab: dict[str, int] = {}

    def ids(self, s: str) -> list[int]:
        return [self.vocab.setdefault(ch, 10 + len(self.vocab)) for ch in s]

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv[i] for i in ids if not (skip_special_tokens and i == EOS))


class Proc(FakeProcessor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.image_processor.patch_size = 16
        self.tokenizer = Tok()


class GenModel:
    """peft 모델을 감싸 정해진 토큰을 내는 `generate` 를 더한다 — 나머지 속성은 peft 모델로 넘긴다."""

    def __init__(self, m, tok: Tok, script: str, *, eos: bool = True, short: bool = False):
        self.m, self.tok, self.script, self.eos, self.short = m, tok, script, eos, short
        self.generation_config = SimpleNamespace(eos_token_id=EOS)
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.m, name)

    def eval(self):
        self.m.eval()
        return self

    def generate(self, input_ids, max_new_tokens, **kw):
        self.calls.append({"max_new_tokens": max_new_tokens, **{k: v for k, v in kw.items() if k != "image_grid_thw"}})
        ids = self.tok.ids(self.script)[:max_new_tokens]
        if self.short:
            ids = ids[: max(1, max_new_tokens // 2)]
        elif self.eos and len(ids) < max_new_tokens:
            ids = ids + [EOS]
        V = max(ids + [EOS]) + 1
        scores = tuple(torch.nn.functional.one_hot(torch.tensor([i]), V).float() * 30.0 for i in ids)
        return SimpleNamespace(sequences=torch.cat([input_ids, torch.tensor([ids])], dim=1), scores=scores)


class Loader:
    def __init__(self, **kw):
        self.kw, self.n, self.model = kw, 0, None

    def __call__(self, model_id, *, init_seed, revision=None, processor_kwargs=None):
        self.n += 1
        m, _ = fake_loader(model_id, init_seed=7, revision=revision, processor_kwargs=processor_kwargs)
        proc = Proc(**dict(processor_kwargs or {}))
        self.model = GenModel(m, proc.tokenizer, self.kw.get("script", TARGET), eos=self.kw.get("eos", True),
                              short=self.kw.get("short", False))
        return self.model, proc


def adapter_npz(path: Path, *, extra: bool = False) -> list[np.ndarray]:
    """대역 모델의 어댑터 키로 값을 바꾼 파일 — 학습이 낸 파일과 같은 꼴(키 순서가 계약)."""
    from peft import get_peft_model_state_dict

    from detection import serialize

    m, _ = fake_loader("x", init_seed=7)
    sd = get_peft_model_state_dict(m)
    keys = serialize.canonical_keys(sd)
    arrays = [a + 0.25 for a in serialize.state_dict_to_ndarrays(sd, keys)]
    blob = dict(zip(keys, arrays))
    if extra:
        blob["mystery.lora_A.weight"] = np.zeros((1, 1), dtype=np.float32)
    np.savez(path, **blob)
    return arrays


def cfg(tmp: Path, **over) -> GenerationConfig:
    kw = {"mode": "model", "tag": "uni_local_C1", "model_id": "local/tiny", "model_revision": "r1",
          "processor_kwargs": {"max_pixels": 1 << 20}, "prompt_text": "좌표를 적어라",
          "chat_template_kwargs": {"enable_thinking": False}, "max_new_tokens": 512,
          "decoding": {"do_sample": False, "num_beams": 1}, "batch_size": 1, "padding_side": "left",
          "coord_space": "ABS_ORIG", "pairs_path": None, "adapter_path": tmp / "adapter_last.npz",
          "purpose": "rehearsal"}
    kw.update(over)
    return GenerationConfig(**kw)


@pytest.fixture
def img():
    from PIL import Image

    return Image.new("L", (1280, 720), 40)


def test_학습_어댑터를_주입하고_greedy_한_번으로_생성문과_멈춤과_격자를_낸다(tmp_path, img):
    from peft import get_peft_model_state_dict

    from detection import serialize
    from vlm.export_model import load_model_generator

    arrays = adapter_npz(tmp_path / "adapter_last.npz")
    loader = Loader()
    g = load_model_generator(cfg(tmp_path), model_loader=loader)
    sd = get_peft_model_state_dict(loader.model.m)
    got = serialize.state_dict_to_ndarrays(sd, serialize.canonical_keys(sd))
    assert all(np.array_equal(a, b) for a, b in zip(arrays, got))                  # 주입이 먹었다
    out = g(img, "aihub000001")
    assert out.text == TARGET and out.gen_stop == "eos" and out.n_new_tokens == len(TARGET) + 1
    assert (out.grid_thw, out.model_input_wh) == ((1, 45, 80), (1280, 720))
    assert len(out.token_ids) == len(out.token_logprobs) == out.n_new_tokens and max(out.token_logprobs) <= 0.0
    assert out.latency_ms is not None and out.latency_ms >= 0.0
    call, = loader.model.calls
    assert call["do_sample"] is False and call["num_beams"] == 1 and call["max_new_tokens"] == 512
    assert loader.model.tok.padding_side == "left"                                  # 등록된 채움 방향


def test_한도에_닿으면_length_이고_종료_토큰도_한도도_아니면_stop_undetermined_를_적는다(tmp_path, img):
    from vlm.export_model import load_model_generator

    adapter_npz(tmp_path / "adapter_last.npz")
    out = load_model_generator(cfg(tmp_path, max_new_tokens=8), model_loader=Loader())(img, "a")
    assert out.gen_stop == "length" and out.n_new_tokens == 8
    out = load_model_generator(cfg(tmp_path, max_new_tokens=len(TARGET)), model_loader=Loader())(img, "a")
    assert out.gen_stop == "length"                                                 # 한도에 닿고 종료 토큰이 없다
    out = load_model_generator(cfg(tmp_path, max_new_tokens=40), model_loader=Loader(short=True, eos=False))(img, "a")
    assert out.gen_stop == "stop_undetermined" and out.n_new_tokens == 20         # 멈추지 않고 적는다(외부 검토 11)


@pytest.mark.parametrize(("over", "msg"), [
    ({"decoding": {"do_sample": True}}, "greedy"),
    ({"decoding": {"do_sample": False, "num_beams": 4}}, "greedy"),
    ({"batch_size": 2}, "배치"),
    ({"mode": "echo"}, "모델 모드"),
    ({"adapter_path": None}, "어댑터 경로"),
])
def test_greedy_한_번_배치_1_이_아니면_모델을_올리기_전에_거부한다(tmp_path, over, msg):
    from vlm.export_model import load_model_generator

    adapter_npz(tmp_path / "adapter_last.npz")
    loader = Loader()
    with pytest.raises(ValueError, match=msg):
        load_model_generator(cfg(tmp_path, **over), model_loader=loader)
    assert loader.n == 0


def test_어댑터_키가_모델과_다르면_생성하지_않는다(tmp_path):
    from vlm.export_model import load_model_generator

    adapter_npz(tmp_path / "adapter_last.npz", extra=True)
    with pytest.raises(RuntimeError, match="키가 모델의 어댑터 키와 다르다"):
        load_model_generator(cfg(tmp_path), model_loader=Loader())


def test_두_모드가_지나는_적재_함수가_모델_모드에서_이_구현으로_간다(tmp_path, img):
    from vlm import export_run as XR
    from vlm.export_generator import load_generator, model_generator_available

    adapter_npz(tmp_path / "adapter_last.npz")
    out = load_generator(cfg(tmp_path), model_loader=Loader())(img, "a")
    assert out.text == TARGET
    assert model_generator_available() and XR.APPROVED_GENERATOR_LOADERS["model"] == (load_generator,)


def test_실제_모델_생성기로_쓴_리허설_묶음을_평가_쪽_검증기가_받는다(tmp_path):
    """합성 세계의 어댑터를 대역 모델의 키로 바꿔 쓴다 — meta 의 어댑터 sha 도 따라 바꾼다."""
    from vlm.export_generator import load_generator
    from vlm.train_cell import _sha256_file

    w = W.build(tmp_path)
    adapter_npz(w.adapter / "adapter_last.npz")
    mp = w.adapter / "adapter_last.meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["adapter_sha256"] = _sha256_file(w.adapter / "adapter_last.npz")
    mp.write_text(json.dumps(meta), encoding="utf-8")
    fake = Loader()
    res = W.run(w, loader=lambda c: load_generator(c, model_loader=fake), with_tokens=True)
    assert res.sealed and res.n_written == len(W.IDS) and res.generator_loaded and fake.n == 1
    rows = [json.loads(x) for x in (w.root / "export" / "uni_local_C1_s1.generations.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert all(r["gen_stop"] == "eos" and r["parse_error"] is None for r in rows)
    assert W.verify(w).mode == "model"


def test_주입이_먹지_않으면_생성하지_않는다(tmp_path, monkeypatch):
    """peft 의 상태 적용이 조용히 아무것도 하지 않는 경우 — 주입 뒤 다시 읽어 파일과 맞대는 검사가 잡는다."""
    import peft

    from vlm.export_model import load_model_generator

    adapter_npz(tmp_path / "adapter_last.npz")
    monkeypatch.setattr(peft, "set_peft_model_state_dict", lambda model, sd: None)
    with pytest.raises(RuntimeError, match="주입이 먹지 않았다"):
        load_model_generator(cfg(tmp_path), model_loader=Loader())



# ================================================================ 안쪽 구현의 승인(외부 검토 10)
def test_본실험은_대역_모델_적재기를_부르기_전에_거부한다(tmp_path):
    from vlm.export_model import load_model_generator
    from vlm.seams import SeamRejected

    adapter_npz(tmp_path / "adapter_last.npz")
    loader = Loader()
    with pytest.raises(SeamRejected):
        load_model_generator(cfg(tmp_path, purpose="main"), model_loader=loader)
    assert loader.n == 0


def _spy(calls):
    def f(*a, **k):
        calls.append(1)
        raise AssertionError("대역이 불렸다")
    return f


@pytest.mark.parametrize("which", ["model_generator", "echo_generator", "model_loader"])
def test_본실험은_바깥_함수를_둔_채_안쪽만_바꿔_끼운_대역을_부르기_전에_거부한다(tmp_path, monkeypatch, which):
    """바깥 적재 함수(승인된 객체)는 그대로 두고 안쪽 속성만 바꾼다 — 이음새 단계에서 막혀 대역 · 실측 · 이미지 호출이 0 이다."""
    import vlm.export_echo as E
    import vlm.export_model as M
    import vlm.pilot_vlm as PV
    from vlm import export_run as XR
    from vlm.export_generator import load_generator
    from vlm.seams import SeamRejected

    w = W.build(tmp_path)
    calls: list = []
    target = {"model_generator": (M, "load_model_generator"), "echo_generator": (E, "load_echo_generator"),
              "model_loader": (PV, "_load_model")}[which]
    monkeypatch.setattr(*target, _spy(calls))
    measured: list = []
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (lambda spec: measured.append(spec),) + XR.APPROVED_MEASURES)
    with pytest.raises(SeamRejected):
        W.run(w, purpose="main", run_root=None, plan_path=None, folder=tmp_path / "main_out",
              loader=load_generator, measure_fn=XR.APPROVED_MEASURES[0])
    assert calls == [] and measured == [] and not (tmp_path / "main_out").exists()
    with pytest.raises(SeamRejected):                      # 부르는 순간에도 다시 본다
        load_generator(cfg(tmp_path, purpose="main"))
    assert calls == []


def test_리허설은_안쪽_대역을_받되_곁_파일에_승인되지_않은_식별자로_적는다(tmp_path, monkeypatch):
    import vlm.export_model as M
    from vlm.export_generator import load_generator

    w = W.build(tmp_path)
    adapter_npz(w.adapter / "adapter_last.npz")
    from vlm.train_cell import _sha256_file

    mp = w.adapter / "adapter_last.meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["adapter_sha256"] = _sha256_file(w.adapter / "adapter_last.npz")
    mp.write_text(json.dumps(meta), encoding="utf-8")
    fake = Loader()
    real = M.load_model_generator
    monkeypatch.setattr(M, "load_model_generator", lambda c, **k: real(c, model_loader=fake))
    res = W.run(w, loader=load_generator)
    assert res.sealed
    ids = res.meta["impl_ids"]
    assert ids["export_generator_model"]["approved"] is False and ids["export_generator_echo"]["approved"] is True
    # 대역 생성기가 부른 적재기는 알 수 없다 — 기본 적재기를 승인으로 적지 않는다(재확인 추가 2)
    assert ids["model_loader"]["approved"] is False
    assert ids["model_loader"]["qualname"] == ids["export_generator_model"]["qualname"]


# ================================================================ 미정 멈춤(외부 검토 11)
def test_멈춤을_정하지_못한_줄을_적고_끝까지_쓰고_평가_쪽_검증기가_그_묶음을_거부한다(tmp_path):
    from evaluation.reject_v14 import BundleRejected, RejectCode
    from vlm.export_generator import load_generator
    from vlm.train_cell import _sha256_file

    w = W.build(tmp_path)
    adapter_npz(w.adapter / "adapter_last.npz")
    mp = w.adapter / "adapter_last.meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["adapter_sha256"] = _sha256_file(w.adapter / "adapter_last.npz")
    mp.write_text(json.dumps(meta), encoding="utf-8")
    fake = Loader(short=True, eos=False)
    res = W.run(w, loader=lambda c: load_generator(c, model_loader=fake))
    assert res.sealed and res.n_written == len(W.IDS)                         # 멈추지 않고 끝까지 썼다
    rows = [json.loads(x) for x in (w.root / "export" / "uni_local_C1_s1.generations.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert {r["gen_stop"] for r in rows} == {"stop_undetermined"}
    with pytest.raises(BundleRejected) as exc:
        W.verify(w)
    assert RejectCode.STOP_UNDETERMINED in exc.value.codes()



# ================================================================ 외부 검토 재확인(10-04)의 추가 둘
def test_진단의_커밋된_영수증은_안쪽만_대역이어도_대역을_부르기_전에_거부한다(tmp_path, monkeypatch):
    """바깥 적재 함수 · 실측은 승인, 안쪽 모델 적재기만 대역, 진단 · `standin_allowed=True` — 영수증 가드가 안쪽 식별자를 합친 뒤에 돈다(추가 1)."""
    import subprocess

    import vlm.pilot_vlm as PV
    from vlm import export_run as XR
    from vlm.export_generator import load_generator
    from vlm.export_writer import ExportRefused

    w = W.build(tmp_path)
    root_commit = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "refs/heads/main"],
                                 capture_output=True, check=True).stdout.decode().split()[-1]
    rc = json.loads(w.receipt.read_text(encoding="utf-8"))
    committed = w.root / "registration" / "fdiag-committed.receipt.json"
    committed.write_text(json.dumps(rc | {"kind": "frame_diag", "main_commit": root_commit}), encoding="utf-8")
    calls: list = []
    monkeypatch.setattr(PV, "_load_model", _spy(calls))
    measured: list = []
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (lambda spec: measured.append(spec),) + XR.APPROVED_MEASURES)
    with pytest.raises(ExportRefused) as exc:
        W.run(w, purpose="frame_diag", receipt_path=committed, loader=load_generator,
              measure_fn=XR.APPROVED_MEASURES[0], standin_allowed=True)
    assert exc.value.code == "seam_on_committed_receipt"
    assert calls == [] and measured == []


def test_생성기가_부른_적재기가_기록과_다르면_쓰지_않는다(tmp_path, monkeypatch):
    """이음새 단계는 기본 적재기(승인)를 적었는데 생성기가 다른 적재기를 불렀다 — 곁 파일이 출처를 잘못 적지 않게 멈춘다(추가 2)."""
    import vlm.pilot_vlm as PV
    from vlm.export_generator import load_generator
    from vlm.export_writer import ExportRefused
    from vlm.seams import impl_id
    from vlm.train_cell import _sha256_file

    w = W.build(tmp_path)
    adapter_npz(w.adapter / "adapter_last.npz")
    mp = w.adapter / "adapter_last.meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["adapter_sha256"] = _sha256_file(w.adapter / "adapter_last.npz")
    mp.write_text(json.dumps(meta), encoding="utf-8")
    fake = Loader()
    monkeypatch.setattr(PV, "resolve_model_loader",
                        lambda model_loader=None, **k: (fake, [impl_id(fake, seam="model_loader", approved=False)]))
    with pytest.raises(ExportRefused) as exc:
        W.run(w, loader=load_generator)
    assert exc.value.code == "impl_drift"
    assert fake.n == 1 and not (w.root / "export" / "uni_local_C1_s1.export_meta.json").exists()
