"""통합형 경로 전달(미니스펙 12번 C2) — 모델 · 페어 · 프롬프트 · 템플릿 · 좌표 규약을 설정에서 받아 넘긴다.

## 무엇을 막는가

- **서버와 클라이언트가 다른 모델로 학습하는 것.** 서버는 설정의 모델로 초기 어댑터를 만들고 클라이언트는
  모듈 상수로 학습했다(12번 §2-1). 이제 둘 다 `fl/uni_run_config.py` 의 한 목록으로 같은 키를 주고받는다.
- **템플릿 인자의 원본이 둘이 되는 것.** 학습 렌더 · 생성 접두 · 접두 지문이 같은 인자 객체를 받는다.
- **정하지 않은 설정 값으로 학습이 시작되는 것.** `null` 인 키가 있으면 그 이름을 모아 멈춘다.

모델과 GPU 없이 돈다. 학습 루프 자체는 여기서 돌리지 않는다 — 값이 루프에 **닿는 자리**까지를 본다.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fl.uni_run_config import (PURPOSES, UNI_RUN_DEFAULTS, UNI_RUN_KEYS,  # noqa: E402
                               client_run_cfg, down_config)
from vlm.uni_config import UniConfigIncomplete, load_uni_config, uni_config_from  # noqa: E402


# ================================================================ 설정 읽기
def _doc(**over) -> dict:
    fixed = {
        "snapshot_digest": "a" * 64,
        "uni_train_budget": {"num_rounds": 2, "local_epochs": 1, "total_epochs": 2},
        "uni_model": {"id": "Qwen/Qwen3.5-4B", "revision": "r" * 40},
        "uni_pairs": {"path": "data/processed/p/pairs.jsonl", "digest": "b" * 64},
        "uni_chat_template_kwargs": {"enable_thinking": False},
        "uni_max_new_tokens": None,
        "uni_batch_size": 1,
        "uni_processor_kwargs": {"max_pixels": 921600},
        "uni_expected_supervised_tokens": {"C1": 10, "C2": 20, "C3": 30},
        "uni_prompt_path": "vlm/prompts/unified_v2_absorig.txt",
        "uni_coord_space": "ABS_ORIG",
    }
    fixed.update(over.pop("fixed", {}))
    return {"fixed_before_main_runs": fixed, "experiment": {"seeds": over.pop("seeds", [1, 2, 3])}}


def test_채운_설정은_학습에_쓸_수_있다():
    cfg = uni_config_from(_doc()).require("train")
    assert (cfg.num_rounds, cfg.local_epochs, cfg.total_epochs) == (2, 1, 2)
    assert cfg.seeds == (1, 2, 3)
    assert cfg.chat_template_kwargs == {"enable_thinking": False}
    assert (cfg.prompt_path, cfg.coord_space) == ("vlm/prompts/unified_v2_absorig.txt", "ABS_ORIG")


def _all_empty_doc() -> dict:
    """학습 단계의 필수 키를 **모두 비운** 합성 설정 — 저장소 설정의 지금 값에 묶이지 않는다."""
    fixed = {"snapshot_digest": None, "uni_train_budget": {"num_rounds": None, "local_epochs": None, "total_epochs": None},
             "uni_model": {"id": None, "revision": None}, "uni_pairs": {"path": None, "digest": None},
             "uni_chat_template_kwargs": None, "uni_processor_kwargs": None, "uni_expected_supervised_tokens": None,
             "uni_prompt_path": None, "uni_coord_space": None}
    return {"fixed_before_main_runs": fixed, "experiment": {"seeds": []}}


def test_빈_필수_키는_전부_이름으로_낸다():
    from vlm.uni_config import UniConfig

    with pytest.raises(UniConfigIncomplete) as exc:
        uni_config_from(_all_empty_doc()).require("train")
    text = " ".join(exc.value.problems)
    for name in UniConfig.NEEDS["train"]:
        assert name in text, name


def test_저장소의_설정은_빈_키만_이름으로_낸다():
    """`configs/base.yaml` 의 **지금 값**과 무관한 규칙을 본다 — 값이 있는 키는 열거되지 않고 빈 키만 열거된다.
    데이터 쪽이 키를 채울 때마다 이 시험을 고치지 않는다(게이트 13회차 실패 — 채워진 두 키를 빈 키로 기대했다)."""
    from vlm.uni_config import UniConfig

    cfg = load_uni_config(ROOT / "configs/base.yaml")
    assert not hasattr(cfg, "prompt_sha256")   # 두지 않는 키다 — 등록 단계가 프롬프트 문자열에서 낸다(12번 정정)
    empty = [n for n in UniConfig.NEEDS["train"] if getattr(cfg, n) in (None, (), "")]
    filled = [n for n in UniConfig.NEEDS["train"] if n not in empty]
    problems = cfg.problems("train")
    named = {n for n in UniConfig.NEEDS["train"] if any(p.startswith(f"{n} 이 비어 있다") for p in problems)}
    assert named == set(empty) and not named & set(filled)
    if empty:
        with pytest.raises(UniConfigIncomplete):
            cfg.require("train")


@pytest.mark.parametrize("over, why", [
    ({"fixed": {"uni_train_budget": {"num_rounds": 2, "local_epochs": 2, "total_epochs": 3}}}, "total_epochs"),
    ({"fixed": {"uni_train_budget": {"num_rounds": None, "local_epochs": 1, "total_epochs": 2}}}, "num_rounds"),
    ({"fixed": {"uni_pairs": {"path": None, "digest": None}}}, "pairs_path"),
    ({"seeds": []}, "seeds"),
])
def test_학습에_필요한_칸이_비거나_어긋나면_멈춘다(over, why):
    with pytest.raises(UniConfigIncomplete) as exc:
        uni_config_from(_doc(**over)).require("train")
    assert any(why in p for p in exc.value.problems)


@pytest.mark.parametrize("over", [
    {"fixed": {"uni_train_budget": {"num_rounds": True, "local_epochs": 1, "total_epochs": 1}}},
    {"fixed": {"uni_batch_size": "1"}},
    {"fixed": {"uni_chat_template_kwargs": "enable_thinking=False"}},
    {"seeds": [1, 1]},
    {"seeds": [1, "2"]},
    {"fixed": {"uni_model": {"id": "", "revision": None}}},
    {"fixed": {"uni_prompt_sha256": 123}},
])
def test_꼴이_틀린_값은_읽을_때_멈춘다(over):
    with pytest.raises(UniConfigIncomplete):
        uni_config_from(_doc(**over))


# ================================================================ 프롬프트 · 페어 · 타깃
def test_프롬프트_해시는_모델이_받는_문자열의_UTF8_이다(tmp_path):
    from vlm.pilot_vlm import load_prompt

    lf, crlf = tmp_path / "lf.txt", tmp_path / "crlf.txt"
    lf.write_bytes("한 줄\n둘\n".encode("utf-8"))
    crlf.write_bytes("한 줄\r\n둘\r\n".encode("utf-8"))
    text, sha = load_prompt(lf)
    assert sha == hashlib.sha256(text.encode("utf-8")).hexdigest()
    # 줄끝이 달라도 모델이 받는 문자열이 같으면 같은 해시다 — 파일 바이트 비교는 작업 트리 청결 검사의 몫이다.
    assert load_prompt(crlf) == (text, sha)


def test_페어는_준_경로에서_읽는다(tmp_path):
    from vlm.pilot_vlm import load_pairs

    p = tmp_path / "pairs.jsonl"
    rows = [{"split": "train", "client": "C1", "id": 1}, {"split": "val", "client": "C1", "id": 2},
            {"split": "train", "client": "C2", "id": 3}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [r["id"] for r in load_pairs("train", "C1", pairs_path=p)] == [1]
    assert [r["id"] for r in load_pairs("train", pairs_path=p)] == [1, 3]
    with pytest.raises(ValueError):
        load_pairs("eval", pairs_path=p)


def test_타깃의_좌표_규약을_인자로_받는다():
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.pilot_vlm import build_target

    row = {"skeleton": {"defects": [{"type": "2011", "bbox_px": [10, 20, 40, 55]}]}}
    g = ImageGeom(orig_w=1280, orig_h=720)
    assert build_target(row, g) == build_target(row, g, CoordCfg(coord_space="ABS_ORIG"))
    assert json.loads(build_target(row, g))["defects"][0]["bbox_2d"] == [10, 20, 40, 55]


# ================================================================ 접두 지문
class _Proc:
    def __init__(self, ids):
        self.ids = ids
        self.kwargs = []

    def apply_chat_template(self, msgs, **kw):
        self.kwargs.append(kw)
        return {"input_ids": self.ids}


def test_생성_접두_해시는_구분자_고정_JSON_의_UTF8_이다():
    from vlm.pilot_vlm import gen_prefix_sha256

    want = hashlib.sha256(b"[1,22,333]").hexdigest()
    assert gen_prefix_sha256(_Proc([1, 22, 333]), "p") == want
    # 배치 차원이 있어도 같은 값이다 — 한 겹을 벗긴다.
    assert gen_prefix_sha256(_Proc([[1, 22, 333]]), "p") == want


def test_생성_접두_해시는_한_줄이_아니거나_정수가_아니면_멈춘다():
    from vlm.pilot_vlm import gen_prefix_sha256

    with pytest.raises(ValueError):
        gen_prefix_sha256(_Proc([[1], [2]]), "p")
    with pytest.raises(TypeError):
        gen_prefix_sha256(_Proc([1, 2.0]), "p")
    with pytest.raises(TypeError):
        gen_prefix_sha256(_Proc([True, 2]), "p")


def test_접두_계산이_템플릿_인자를_그대로_넘긴다():
    from vlm.pilot_vlm import gen_prefix_digest, gen_prefix_sha256

    class _T:
        def __init__(self, v):
            self.v = v

        def tolist(self):
            return self.v

    proc = _Proc([_T([1, 2])])
    gen_prefix_digest(proc, {"enable_thinking": False, "x": 1})
    gen_prefix_sha256(_Proc([1]), "p", {"enable_thinking": False, "x": 1})
    assert proc.kwargs[0]["enable_thinking"] is False and proc.kwargs[0]["x"] == 1


def test_학습_루프가_받은_값을_신원과_인코딩에_쓴다():
    """루프는 GPU 없이 돌 수 없어 소스의 자리를 본다 — 신원의 `data`·`template_mode`·접두 지문과
    `_encode` 가 **인자에서 온 값**을 쓴다. 모듈 상수로 되돌리면 떨어진다."""
    import ast

    src = (ROOT / "vlm/pilot_vlm.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "train_rounds")
    body = ast.unparse(fn)
    for piece in ("data=str(pairs_path or PAIRS_PATH)", "template_mode=template_mode_of(kwargs)",
                  "prefix_digest = gen_prefix_digest(proc, kwargs)", "gen_prefix_digest=prefix_digest",
                  "chat_template_kwargs=kwargs, coord_cfg=cfg_coord",
                  "revision=model_revision, processor_kwargs=processor_kwargs",
                  "load_prompt(prompt_path)"):
        assert piece in body, piece
    assert "PROMPT_PATH.read_text" not in body


# ================================================================ 연합 키
def test_연합_키의_선언이_목록과_기본값까지_같다():
    decl = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    for k, d in UNI_RUN_DEFAULTS.items():
        assert k in decl, f"pyproject 가 {k} 를 선언하지 않았다 — flwr run 이 덮어쓰기를 거부한다"
        assert decl[k] == d, f"{k} 의 선언 기본값 {decl[k]!r} ≠ 목록 {d!r}"
    assert set(PURPOSES) == {"main", "rehearsal", "frame_diag"}


def test_서버와_인프로세스_서버가_같은_통합형_키를_내려보낸다(tmp_path):
    from fl import pilot_sim, server_app

    sent_srv = server_app._cell_train_config("uni_fed", {"uni-model": "M", "purpose": "rehearsal"}, tmp_path)
    sent_sim = pilot_sim._cell_train_config(
        "uni_fed", {"client_tags": ["C1", "C2", "C3"], "uni_model": "M", "purpose": "rehearsal"}, tmp_path)
    for sent in (sent_srv, sent_sim):
        assert set(UNI_RUN_KEYS) <= set(sent)
        assert sent["uni-model"] == "M" and sent["purpose"] == "rehearsal"
        assert sent["uni-pairs"] == ""        # 지정하지 않은 값은 빈 문자열(파일럿 기본값)


def test_클라이언트는_서버가_보낸_키를_읽고_빠지면_거부한다():
    # 비워 둔 키를 파일럿 기본값으로 내리는 것은 리허설 · 진단 목적일 때만이다(본실험은 tests/test_uni_review_fixes.py).
    sent = down_config(lambda k, d: {"uni-model": "M", "uni-pairs": "p.jsonl", "purpose": "rehearsal",
                                     "uni-chat-template-kwargs": '{"enable_thinking": false}',
                                     "uni-coord-space": "ABS_ORIG"}.get(k, d))
    got = client_run_cfg(sent)
    assert got["model_id"] == "M" and got["pairs_path"] == "p.jsonl"
    assert got["chat_template_kwargs"] == {"enable_thinking": False}
    assert got["purpose"] == "rehearsal" and got["model_revision"] is None
    del sent["uni-prompt"]
    with pytest.raises(ValueError, match="보내지 않았다"):
        client_run_cfg(sent)


@pytest.mark.parametrize("bad", [{"purpose": "mian"}, {"purpose": ""},
                                 {"uni-chat-template-kwargs": "{"},
                                 {"uni-chat-template-kwargs": "[1]"}])
def test_목적과_템플릿_인자의_꼴이_틀리면_보내기_전에_멈춘다(bad):
    with pytest.raises(ValueError):
        down_config(lambda k, d: bad.get(k, d))


def test_클라이언트_앱이_통합형_키를_읽는다():
    import ast

    src = (ROOT / "fl/client_app.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "uni_client_round")
    assert '"uni": client_run_cfg(cfg)' in ast.unparse(fn).replace("'", '"')
    train = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train")
    assert "uni_client_round(cfg" in ast.unparse(train)


def test_연합_클라이언트가_받은_값을_페어와_학습에_넘긴다(monkeypatch):
    import vlm.pilot_vlm as pv
    from fl import client_vlm

    seen = {}

    def fake_pairs(split, client=None, *, pairs_path=None):
        seen["pairs_path"] = pairs_path
        return [{"id": 1, "image_id": "i1"}]

    def fake_train(**kw):
        seen.update(kw)
        m = {"supervised_tokens": 10, "epochs_ran": 1, "optimizer_steps": 1, "resumed_from_epoch": None,
             "param_l2": 1.0, "payload_bytes": 4, "seed": 3, "lr": 1e-4, "peak_vram_gb": 0.0,
             "optimizer": "AdamW", "init_proof": {"l2": 1.0}, "injected_proof": None,
             # 서버가 연합 칸 meta 를 만드는 값(`fl/uni_fed.client_strings`)
             "model_id": "M", "processor_config_sha256": "p", "config_blocks": {}, "impl_ids": [],
             "prompt_sha256": "q", "template_mode": "enable_thinking=False", "coord_cfg_hash": "c",
             "target_contract_sha256": "t", "train_input_grid_observed": {"[1, 2, 4]": 1}}
        return [np.zeros(1, np.float32)], ["k"], m, {}

    monkeypatch.setattr(pv, "load_pairs", fake_pairs)
    monkeypatch.setattr(pv, "train_rounds", fake_train)
    uni = client_run_cfg(down_config(lambda k, d: {"uni-model": "M", "uni-model-revision": "R",
                                                   "uni-pairs": "p.jsonl", "uni-prompt": "q.txt",
                                                   "uni-chat-template-kwargs": '{"enable_thinking": false}',
                                                   "uni-coord-space": "ABS_ORIG", "purpose": "rehearsal",
                                                   "uni-pairs-digest": "d" * 64,
                                                   "uni-processor-kwargs": '{"max_pixels": 1000}'}.get(k, d)))
    client_vlm.run_client_round(adapter_in=None, canonical_keys=["k"], round_idx=0, client_idx=0,
                                cfg={"client_tag": "C1", "local_epochs": 1, "num_rounds": 1,
                                     "base_seed": 5, "uni": uni})
    assert seen["pairs_path"] == "p.jsonl" and seen["model_id"] == "M" and seen["model_revision"] == "R"
    assert seen["prompt_path"] == "q.txt" and seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["coord_cfg"].coord_space == "ABS_ORIG"
    # 프로세서 설정 · 페어 지문 · 참여자가 학습(과 그 재개 신원)에 닿는다(외부 검토 2)
    assert seen["processor_kwargs"] == {"max_pixels": 1000}
    assert seen["pairs_digest"] == "d" * 64 and seen["client_tag"] == "C1" and seen["purpose"] == "rehearsal"


def test_서버의_초기_어댑터가_통합형_모델과_판으로_만들어진다(monkeypatch, tmp_path):
    import torch

    import vlm.init_adapter as ia
    from fl import server_app

    seen = {}
    seen_purpose: list[str] = []

    def fake_build(*, model_id=None, seed, cache_path=None, revision=None, purpose="main"):
        seen.update(model_id=model_id, revision=revision)
        seen_purpose.append(purpose)
        a = [np.zeros(2, np.float32)]
        return a, ["k"], {"k": torch.zeros(2)}

    monkeypatch.setattr(ia, "build_initial_adapter", fake_build)
    server_app._load_initial("uni_fed", {"uni-model": "M4", "uni-model-revision": "rev", "model": "yolo11s.pt",
                                         "base-seed": 1, "project": str(tmp_path)})
    assert seen == {"model_id": "M4", "revision": "rev"}
    server_app._load_initial("uni_fed", {"model": "M08", "base-seed": 1, "project": str(tmp_path)})
    assert seen == {"model_id": "M08", "revision": None}   # 통합형 키가 비면 종전대로
    assert seen_purpose == ["main", "main"]                 # 목적 키가 없으면 가장 엄한 본실험 규칙


def test_초기_어댑터_캐시는_모델_판이_다르면_받지_않는다(tmp_path):
    from vlm.init_adapter import build_initial_adapter

    cache = tmp_path / "initial.npz"
    np.savez(cache, k=np.zeros(2, np.float32))
    cache.with_suffix(".proof.json").write_text(
        json.dumps({"model_id": "M", "seed": 1, "revision": "old"}), encoding="utf-8")
    kw = {"purpose": "rehearsal"}     # 본실험의 proof 규칙은 tests/test_uni_train_cell.py 가 본다
    with pytest.raises(RuntimeError, match="신원이 다르다"):
        build_initial_adapter(model_id="M", seed=1, cache_path=cache, revision="new", **kw)
    arrays, keys, _ = build_initial_adapter(model_id="M", seed=1, cache_path=cache, revision="old", **kw)
    assert keys == ["k"]
    # 판을 적지 않던 옛 proof 는 판 없는 요청과만 맞는다
    cache.with_suffix(".proof.json").write_text(json.dumps({"model_id": "M", "seed": 1}), encoding="utf-8")
    build_initial_adapter(model_id="M", seed=1, cache_path=cache, **kw)
    with pytest.raises(RuntimeError):
        build_initial_adapter(model_id="M", seed=1, cache_path=cache, revision="new", **kw)


def test_좌표_규약의_값_공간은_좌표_모듈_하나다():
    """설정 읽기가 좌표 모듈과 다른 목록을 들면 규약이 하나 늘 때 설정이 그 값을 거부한다(`configs/base.yaml` 의 주석도 이 목록을 가리킨다)."""
    from vlm import coords, uni_config

    assert uni_config.COORD_SPACES is coords.COORD_SPACES
