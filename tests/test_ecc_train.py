"""조기 좌표 점검(early_coord_check)의 학습 쪽 — 11번 미니스펙 §7 의 T1 ~ T14 가운데 학습 쪽 몫.

모델 없이 돈다. `train_rounds` 를 **실제로** 부르되 모델 적재기만 합성 대역(`tests/uni_fakes.py`)이다. 대역은 점검 목적에서
`standin_allowed=True` 일 때만 받는다(본 점검은 대역을 받지 않는다 — 이음새 시험). 누적 창을 2 로 줄여(`GRAD_ACCUM`) 행 12 개 ·
epoch 2 개 · 최적화 12 스텝의 작은 세계에서 점검 step `(2, 4, 8)` 을 본다 — 8 은 둘째 epoch 의 중간이다.

평가 쪽은 **실물**(`evaluation.early_coord_check` 3판, `41647a3` 을 병합으로 받았다)을 부른다 — 규칙은 커밋된 규칙 파일에서
미정 칸(순차 규칙 · 허용 시도 수 1 · 판별력 영수증 · 상한)만 채운 바이트, 통과 파일은 평가 쪽 `build_output` 이 합성 대상 다섯 장으로
만든 것이다. 실물 평가 자산 · 봉인 자산을 열지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detection.round_runner import derive_seed
from evaluation import early_coord_check as D
from evaluation.eval_list import canonical_bytes, validate_eval_list
from tests.uni_fakes import CountingLoader, make_rows
from vlm import ecc as E
from vlm import pilot_vlm
from vlm.init_adapter import build_initial_adapter
from vlm.pin import PinMismatch, build_pin, check_pin, init_probe, read_pin, write_pin
from vlm.train_cell import (
    CENTRAL_TAG,
    CellRejected,
    UniRunSpec,
    cell_dir,
    main_start_gate,
)

ECC_ROOT = "ecc-20261013T000000"
STEPS = (2, 4, 8)


@pytest.fixture(autouse=True)
def _small_world(monkeypatch):
    """GPU 를 쓰지 않고 누적 창을 2 로 줄인다(신원의 `grad_accum` 도 같은 값을 읽는다)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 2)


#: 평가 쪽 실물에 넘길 합성 대상 — 다섯 장(학습 분할 합성 id).
TGT = validate_eval_list(canonical_bytes([f"synth:t{i:03d}" for i in range(5)]))
LIMITS = {"max_hours": 30, "max_cost_krw": 90000}


def _rules_raw(**over) -> bytes:
    """커밋된 평가 쪽 규칙 파일(판 · 문턱 · 검사기 지문 · 단가는 그대로)에 미정 칸만 채운 바이트."""
    b = yaml.safe_load(D.RULES_PATH.read_bytes().decode("utf-8"))
    b.update({"sequential_rule": "single", "max_attempts": 1, "discrimination": {"path": "x.json", "sha256": "0" * 64},
              "target_list": {"sha256": TGT.file_sha256, "n": len(TGT)}})
    b["limits"] = {**b["limits"], **LIMITS}
    b.update(over)
    return yaml.safe_dump(b, allow_unicode=True, sort_keys=False).encode("utf-8")


def _rule_file(tmp: Path, **over) -> Path:
    q = tmp / "early_coord_check.yaml"
    q.write_bytes(_rules_raw(**over))
    return q


def _permit_bytes(tmp: Path, rules, expected: dict, *, attempt: int = 1) -> bytes:
    """평가 쪽 `build_output` 으로 만든 통과 파일 — 네 검사 산출 파일은 `tmp/mf/` 에 실제로 쓰고 그 지문을 싣는다."""
    from vlm.ecc import _repo_lf_sha256

    d = tmp / "mf"
    d.mkdir(exist_ok=True)
    files = {}
    for n in D.MODEL_FREE_CHECKS:
        f = d / f"{n}.json"
        f.write_text(json.dumps({"name": n}), encoding="utf-8")
        files[n] = {"path": str(f), "sha256": _repo_lf_sha256(str(f))}
    mf = [{"name": n, "status": D.PASS, "n": len(TGT), "n_bad": 0, "n_ids": len(TGT), "ids_set_sha256": TGT.set_sha256}
          for n in D.MODEL_FREE_CHECKS]
    steps = [{"status": D.PASS, "reason": "", "n_images": len(TGT), "ids_set_sha256": TGT.set_sha256}]
    ev = {"lineage": {"snapshot": "s", "pairs": "p", "r0": "r"}, "model_free_files": files,
          "step_outputs": [{"step": 100, "raw_sha256": "g", "parsed_sha256": "q", "consumed_sha256": "c"}],
          "attempt": attempt, "step_adapter_digests": ["w100"], "processor_cfg_sha256": "pc",
          "model_input_wh": [1280, 704], **expected}
    return D.canonical(D.build_output(rules=rules, target=TGT, model_free=mf, steps=steps, evidence=ev))


_REAL_EVAL_SIDE = E._eval_side


def _spec(tmp: Path, root: str = ECC_ROOT, **over) -> UniRunSpec:
    r = tmp / root
    kw = {"purpose": E.PURPOSE, "run_stamp": f"ecc_s1_{root}", "seed_index": 1, "seed_value": 7,
          "snapshot_digest": "s" * 64, "model_id": "tiny", "model_revision": "r1", "pairs_path": str(tmp / "pairs.jsonl"),
          "pairs_digest": "p" * 64, "chat_template_kwargs": {"enable_thinking": False}, "num_rounds": 1,
          "local_epochs": 2, "total_epochs": 2, "train_root": r / "train", "init_adapter_path": r / "init" / "initial.npz",
          "run_root": r, "non_main_parent": tmp, "standin_allowed": True, "prompt_path": None, "processor_kwargs": {},
          "expected_supervised_tokens": {"C1": 1, "C2": 1, "C3": 1}}
    kw.update(over)
    return UniRunSpec(**kw)


def _rows(tmp: Path) -> list[dict]:
    img = tmp / "img"
    return make_rows(img, "C1", 7) + make_rows(img, "C2", 4) + make_rows(img, "C3", 1)


def _probes(spec: UniRunSpec, **over):
    p = {"code": lambda: {"commit": "c" * 40, "dirty": []},
         "env": lambda: {"torch": "t", "cuda": None, "gpu_name": None},
         "common": lambda: {"uni_block_sha256": "u" * 64, "prompt_sha256": "q" * 64},
         "init_adapter": lambda: init_probe(spec.init_adapter_path)}
    p.update(over)
    return p


def _world(tmp: Path, root: str = ECC_ROOT, **over) -> SimpleNamespace:
    loader = CountingLoader()
    spec = _spec(tmp, root, **over)
    build_initial_adapter(model_id=spec.model_id, seed=spec.seed_value, cache_path=spec.init_adapter_path,
                          revision=spec.model_revision, purpose=E.PURPOSE, model_loader=loader, standin_allowed=True)
    pin = tmp / root / "PIN.json"
    write_pin(pin, build_pin(init_adapter_path=spec.init_adapter_path, probes=_probes(spec)))
    return SimpleNamespace(tmp=tmp, spec=spec, loader=loader, rows=_rows(tmp), pin=pin, out=cell_dir(spec, CENTRAL_TAG))


def _run(w: SimpleNamespace, *, resume_step=None, stop=True, **kw):
    kw.setdefault("pin_probes", _probes(w.spec))
    kw.setdefault("config_check", lambda s: [])
    return E.run_ecc_train(w.spec, steps=STEPS, registered_steps=STEPS, pin_path=w.pin, resume_step=resume_step,
                           model_loader=w.loader, rows=w.rows, stop=stop, **kw)


def _meta(w: SimpleNamespace, step: int) -> dict:
    return json.loads(E.step_paths(w.out, step)["meta"].read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): _sha(p) for p in sorted(d.rglob("*")) if p.is_file()}


# ================================================================ T1 · T11 · T12
def test_T1_점검_step_에서_멈추고_파일_넷을_쓰고_그_뒤의_최적화가_없다(tmp_path):
    w = _world(tmp_path)
    res = _run(w)
    assert res["exit_code"] == E.STEP_EXIT == 4 and res["step"] == 2
    paths = E.step_paths(w.out, 2)
    assert all(p.is_file() for p in paths.values())
    clock = (w.out / E.CLOCK_FILE).read_text(encoding="utf-8").splitlines()
    assert [ln.split(",")[0] for ln in clock[1:]] == ["1", "2"]         # step 2 뒤의 최적화가 없다
    m = _meta(w, 2)
    assert m["purpose"] == E.PURPOSE and m["step"] == 2 and m["registered_steps"] == list(STEPS)
    assert _sha(paths["adapter"]) == m["adapter"]["sha256"] and _sha(paths["state"]) == m["state"]["sha256"]
    assert isinstance(m["train_config"], str) and json.loads(m["train_config"])["grad_accum"] == 2
    events = [r["event"] for r in E.ledger_rows(w.out)]
    assert events == ["start", "step", "stop"]


def test_T11_소비_행은_순열의_앞_G_곱하기_s_행이고_epoch_안에서_겹치지_않는다(tmp_path):
    w = _world(tmp_path)
    _run(w, stop=False)
    seed = derive_seed(w.spec.seed_value, 0, 0)
    order0 = E.epoch_order(len(w.rows), seed, 0)
    got2 = E.step_paths(w.out, 2)["consumed"].read_text(encoding="utf-8").splitlines()
    assert got2 == [w.rows[i]["image_id"] for i in order0[:2 * 2]] and len(set(got2)) == len(got2)
    got8 = E.step_paths(w.out, 8)["consumed"].read_text(encoding="utf-8").splitlines()
    order1 = E.epoch_order(len(w.rows), seed, 1)
    assert got8 == [w.rows[i]["image_id"] for i in order0] + [w.rows[i]["image_id"] for i in order1[:2 * 8 - 12]]
    assert len(set(got8[12:])) == len(got8[12:])                        # 둘째 epoch 안에서도 겹치지 않는다


def test_T12_점검_실행의_학습_행과_순열이_본실험_중앙과_같다(tmp_path):
    from vlm.train_cell import _run_identity

    w = _world(tmp_path)
    _run(w)
    m = _meta(w, 2)
    main = _spec(tmp_path, purpose="main", run_stamp="main_s1", run_root=None, non_main_parent=None)
    ident = _run_identity(main, CENTRAL_TAG, client_idx=0, rows_digest=pilot_vlm.train_rows_digest(w.rows),
                          n_rows=len(w.rows), prompt_sha256="x", init_digest="y")
    assert m["train_rows_digest"] == ident["train_rows_digest"]
    assert m["derived_seed"] == ident["derived_seed"] == derive_seed(w.spec.seed_value, 0, 0)
    assert m["perm_sha256"] == E.perm_sha256(E.epoch_order(len(w.rows), ident["derived_seed"], 0))


# ================================================================ T2 · T13
def test_T2_멈춤_지정이_있어도_학습_궤적이_같다(tmp_path):
    """관찰(파일만 쓰고 잇는) 점검 실행의 끝 어댑터가 멈춤 지정 없이 같은 행 · 시드로 돈 학습과 같다."""
    w = _world(tmp_path)
    res = _run(w, stop=False)
    assert res["exit_code"] == 0 and res["optimizer_steps"] == 12
    init = np.load(w.spec.init_adapter_path)
    _, _, m_plain, _ = pilot_vlm.train_rounds(
        rows=w.rows, epochs=2, round_idx=0, client_idx=0, base_seed=w.spec.seed_value,
        adapter_in=[init[k] for k in init.files], adapter_keys=list(init.files), model_id="tiny", run_id="x",
        init_seed=w.spec.seed_value, num_rounds=1, model_revision="r1", model_loader=CountingLoader(),
        purpose="rehearsal", standin_allowed=True, tag=CENTRAL_TAG,
        chat_template_kwargs={"enable_thinking": False}, processor_kwargs={})
    arrays_ecc = res["metrics"]
    # 끝 어댑터의 노름 · 학습량 · 마지막 학습률이 소수점까지 같다 — 콜백이 계산에 끼어들지 않았다(CPU 대역의 결정론 범위).
    for key in ("optimizer_steps", "supervised_tokens", "param_l2", "lr"):
        assert m_plain[key] == arrays_ecc[key], key


def _resumed_chain(tmp: Path, root: str) -> SimpleNamespace:
    w = _world(tmp, root)
    assert _run(w)["step"] == 2
    assert _run(w, resume_step=2)["step"] == 4
    assert _run(w, resume_step=4)["step"] == 8
    return w


def test_T13_epoch_중간에서_멈췄다_이은_실행의_노출_원장_학습량_비용이_끊지_않은_실행과_같다(tmp_path):
    whole = _world(tmp_path, "ecc-20261013T000001")
    _run(whole, stop=False)
    chain = _resumed_chain(tmp_path, "ecc-20261013T000002")
    for s in STEPS:
        a, b = _meta(whole, s), _meta(chain, s)
        assert E.step_paths(whole.out, s)["consumed"].read_bytes() == E.step_paths(chain.out, s)["consumed"].read_bytes()
        for key in ("optimizer_steps", "supervised_tokens", "epoch", "cursor", "perm_sha256"):
            assert a[key] == b[key], (s, key)
        assert a["consumed"]["sha256"] == b["consumed"]["sha256"]
    cums = [_meta(chain, s)["cum_wall_s"] for s in STEPS]
    assert cums == sorted(cums) and cums[0] > 0                         # 비용 누적은 재개로 처음으로 돌아가지 않는다
    events = [r["event"] for r in E.ledger_rows(chain.out)]
    assert events.count("resume") == 2 and events.count("step") == 3


def test_T13_순열_지문이_다르면_재개를_적용하지_않는다(tmp_path):
    import torch

    w = _world(tmp_path)
    _run(w)
    paths = E.step_paths(w.out, 2)
    state = torch.load(paths["state"], weights_only=False)
    state["perm_sha256"] = "0" * 64
    paths["state"].unlink()
    torch.save(state, paths["state"])
    meta = _meta(w, 2)
    meta["perm_sha256"] = "0" * 64
    meta["state"]["sha256"] = _sha(paths["state"])
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(E.EccRefused) as exc:
        _run(w, resume_step=2)
    assert exc.value.code == "perm_mismatch"
    assert not E.step_paths(w.out, 4)["meta"].exists()


def test_T13_재개_파일의_바이트가_meta_와_다르면_적용_전에_거부한다(tmp_path):
    w = _world(tmp_path)
    _run(w)
    p = E.step_paths(w.out, 2)["adapter"]
    p.write_bytes(p.read_bytes() + b"x")
    loads = w.loader.n
    with pytest.raises(E.EccRefused) as exc:
        _run(w, resume_step=2)
    assert exc.value.code == "resume_file" and w.loader.n == loads + 1   # 초기 어댑터 읽기 뒤 · 학습 상태 적용 전


# ================================================================ T3 · 이음새
def test_T3_다른_목적은_점검_step_을_받지_않고_점검_목적은_그것_없이_돌지_않는다(tmp_path):
    loader = CountingLoader()
    cs = E.CheckStop(steps=STEPS, out_dir=tmp_path / "o")
    for purpose in ("rehearsal", "frame_diag"):
        with pytest.raises(ValueError, match="early_coord_check 목적에서만"):
            pilot_vlm.train_rounds(rows=[], epochs=1, round_idx=0, client_idx=0, base_seed=1, purpose=purpose,
                                   standin_allowed=True, model_loader=loader, check_stop=cs)
    with pytest.raises(ValueError, match="check_stop"):
        pilot_vlm.train_rounds(rows=[], epochs=1, round_idx=0, client_idx=0, base_seed=1, purpose=E.PURPOSE,
                               standin_allowed=True, model_loader=loader)
    with pytest.raises(ValueError, match="resume_dir"):
        pilot_vlm.train_rounds(rows=[], epochs=1, round_idx=0, client_idx=0, base_seed=1, purpose=E.PURPOSE,
                               standin_allowed=True, model_loader=loader, check_stop=cs, resume_dir=str(tmp_path))
    assert loader.n == 0                                                # 모두 모델을 올리기 전에 멈췄다


def test_T3_점검_목적은_대역을_허용하지_않으면_부르기_전에_거부하고_연합은_받지_않는다(tmp_path):
    from fl.uni_run_config import client_run_cfg, down_config
    from vlm.seams import SeamRejected

    loader = CountingLoader()
    with pytest.raises(SeamRejected):
        pilot_vlm.resolve_model_loader(loader, purpose=E.PURPOSE, standin_allowed=False)
    cfg = down_config(lambda k, d: d)
    cfg["purpose"] = E.PURPOSE
    with pytest.raises(ValueError, match="연합은 이 목적을 받지 않는다"):
        client_run_cfg(cfg)
    assert loader.n == 0


def test_T3_본실험_실행기는_점검_인자를_받지_않는다(tmp_path, capsys):
    import scripts.main_uni as M
    from tests.test_main_uni import _cfg_yaml

    p = _cfg_yaml(tmp_path)
    for argv in (["local", "--seed", "1", "--resume-step", "2"], ["central", "--seed", "1", "--ecc-root", "x"],
                 ["fed", "--seed", "1", "--spent-hours", "1"]):
        assert M.main(argv, out=tmp_path / "out", config_path=p) == 2
        assert "ecc_args_on_main" in capsys.readouterr().err
    assert M.main(["ecc", "--seed", "1"], out=tmp_path / "out", config_path=p) == 2
    assert "ecc_root_missing" in capsys.readouterr().err
    root = ["--ecc-root", "outputs/early_check/ecc-20261013T000000"]
    for argv, code in ((["ecc", "--seed", "1", *root, "--spent-hours", "0"], "ecc_budget_args"),
                       (["ecc", "--seed", "1", *root, "--spent-hours", "0", "--next-hours", "1"], "ecc_not_prepared"),
                       (["ecc", "--seed", "1", *root, "--spent-hours", "0", "--next-hours", "1", "--permit", "x"],
                        "ecc_args_on_run"),
                       (["ecc-prepare", "--seed", "1", *root, "--max-hours", "3"], "ecc_args_on_prepare"),
                       (["ecc-accept", "--seed", "1", *root], "ecc_args_on_accept"),
                       (["ecc-accept", "--seed", "1", *root, "--permit", "x", "--max-hours", "3"], "ecc_args_on_accept"),
                       (["central", "--seed", "1", "--permit", "x"], "ecc_args_on_main")):
        assert M.main(argv, out=tmp_path / "out", config_path=p) == 2
        assert code in capsys.readouterr().err, argv
    assert not (tmp_path / "out").exists()


# ================================================================ T7 · 착수 고정
def test_T7_착수_고정_칸_하나를_바꾸면_시작과_재개가_거부된다(tmp_path):
    w = _world(tmp_path)
    for part in ("code", "env", "common"):
        bad = _probes(w.spec, **{part: lambda: {"changed": True}})
        with pytest.raises(E.EccRefused) as exc:
            _run(w, pin_probes=bad)
        assert exc.value.code == "pin_mismatch" and part in str(exc.value)
    assert w.loader.n == 1                                              # 초기 어댑터를 만든 한 번뿐 — 학습은 없다
    _run(w)                                                             # 같은 고정이면 시작한다
    dirty = _probes(w.spec, code=lambda: {"commit": "c" * 40, "dirty": ["추적 밖: x.py"]})
    with pytest.raises(E.EccRefused) as exc:
        _run(w, resume_step=2, pin_probes=dirty)
    assert exc.value.code == "pin_mismatch"


def test_T7_초기_어댑터가_바뀌면_착수_고정이_어긋난다(tmp_path):
    w = _world(tmp_path)
    arr = dict(np.load(w.spec.init_adapter_path))
    k = min(arr)
    arr[k] = arr[k] + 1.0
    np.savez(w.spec.init_adapter_path, **arr)
    with pytest.raises(PinMismatch) as exc:
        check_pin(w.pin, build_pin(init_adapter_path=w.spec.init_adapter_path, probes=_probes(w.spec)))
    assert "init_adapter.init_adapter_digest" in exc.value.fields


def test_T7_착수_고정은_코드가_커밋과_다르면_쓰지_않고_덮지_않는다(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(FileExistsError):
        write_pin(w.pin, build_pin(init_adapter_path=w.spec.init_adapter_path, probes=_probes(w.spec)))
    dirty = build_pin(init_adapter_path=w.spec.init_adapter_path,
                      probes=_probes(w.spec, code=lambda: {"commit": "c", "dirty": ["커밋과 다르다: a.py"]}))
    with pytest.raises(PinMismatch):
        write_pin(tmp_path / "other" / "PIN.json", dirty)


# ================================================================ 점검 실행의 가드
def test_새_점검_실행은_step_이_있는_루트에서_시작하지_않고_재개는_등록_순서를_지킨다(tmp_path):
    w = _world(tmp_path)
    _run(w)
    with pytest.raises(E.EccRefused) as exc:
        _run(w)
    assert exc.value.code == "ecc_run_exists"
    with pytest.raises(E.EccRefused) as exc:
        _run(w, resume_step=4)
    assert exc.value.code == "resume_order"
    with pytest.raises(E.EccRefused) as exc:
        _run(w, resume_step=8)
    assert exc.value.code == "resume_step"


def test_루트의_기본_부모는_목적마다_다르고_부를_때_읽는다(monkeypatch, tmp_path):
    from vlm import run_root as R

    ecc = R.REPO_ROOT / "outputs/early_check" / ECC_ROOT
    reh = R.REPO_ROOT / "outputs/rehearsal_u" / "reh-20261013T000000"
    assert R.run_root_problems("early_coord_check", ecc, []) == []
    assert R.run_root_problems("rehearsal", reh, []) == []
    assert any("바로 아래가 아니다" in s for s in R.run_root_problems("early_coord_check", reh.parent / ECC_ROOT, []))
    monkeypatch.setattr(R, "NON_MAIN_PARENT", tmp_path)                  # 리허설 부모 바꾸기가 먹고 점검 부모는 그대로다
    assert R.run_root_problems("rehearsal", tmp_path / "reh-20261013T000000", []) == []
    assert R.run_root_problems("early_coord_check", ecc, []) == []


def test_점검_step_목록이_규칙_파일과_다르거나_루트가_규칙_밖이면_시작하지_않는다(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(E.EccRefused) as exc:
        E.run_ecc_train(w.spec, steps=STEPS, registered_steps=(2, 4), pin_path=w.pin, model_loader=w.loader,
                        config_check=lambda s: [], rows=w.rows, pin_probes=_probes(w.spec))
    assert exc.value.code == "steps_not_registered"
    bad = _world(tmp_path, "reh-20261013T000000")
    with pytest.raises(E.EccRefused) as exc:
        _run(bad)
    assert exc.value.code == "run_root"
    with pytest.raises(E.EccRefused) as exc:
        _run(w, config_check=lambda s: ["model_revision 'r1' ≠ 설정 'r2'"])
    assert exc.value.code == "config_mismatch"


def test_본_점검은_학습_행_목록을_받지_않는다(tmp_path):
    w = _world(tmp_path)
    strict = _spec(tmp_path, standin_allowed=False)
    with pytest.raises(E.EccRefused) as exc:
        E.run_ecc_train(strict, steps=STEPS, registered_steps=STEPS, pin_path=w.pin, model_loader=None,
                        config_check=lambda s: [], rows=w.rows, pin_probes=_probes(strict))
    assert exc.value.code == "rows_override"
    assert not E.step_paths(w.out, 2)["meta"].exists()


def test_규칙은_평가_쪽_load_rules_require_ready_로_읽고_그것이_없거나_거부하면_시작하지_않는다(tmp_path, monkeypatch):
    p = _rule_file(tmp_path)
    got = E.load_check_rules(p)
    assert (got["steps"], got["max_attempts"]) == ((100, 200, 400), 1)
    assert got["rules_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() == got["rules"].raw_sha256
    assert (got["limits"]["max_hours"], got["limits"]["max_cost_krw"]) == (30, 90000)
    with pytest.raises(E.EccRefused) as exc:
        E.load_check_rules(D.RULES_PATH)                                # 커밋된 규칙 파일 — 순차 규칙 · 시도 수 · 상한이 아직 null
    assert exc.value.code == "ecc_rules_rejected"
    with pytest.raises(E.EccRefused) as exc:
        E.load_check_rules(tmp_path / "없다.yaml")
    assert exc.value.code == "ecc_rules_missing"
    monkeypatch.setattr(E, "EVAL_SIDE", "evaluation.no_such_module_for_test")
    with pytest.raises(E.EccRefused) as exc:
        E.load_check_rules(p)
    assert exc.value.code == "ecc_eval_side_missing"


def test_평가_쪽_실물의_서명과_대조값이_학습_쪽_호출과_같다():
    import inspect

    from vlm.train_cell import PERMIT_EXPECTED_KEYS

    assert list(inspect.signature(D.verify_permit).parameters) == ["raw", "expected_sha256", "rules", "expected",
                                                                   "file_sha256_of"]
    assert list(inspect.signature(D.effective_limits).parameters) == ["rules", "max_hours", "max_cost_krw"]
    assert list(inspect.signature(D.load_rules).parameters) == ["raw"]
    assert tuple(PERMIT_EXPECTED_KEYS) == tuple(D.PERMIT_EXPECTED_KEYS)          # 3판 — 그 밖의 칸은 거부된다
    assert set(D.PERMIT_EXPECTED_KEYS) | {"permit_sha256"} <= set(E.PERMIT_KEY_FIELDS)    # 키 기록이 대조값 넷을 싣는다
    assert issubclass(D.EarlyCheckRejected, ValueError)
    assert len({D.PASS, D.FAIL, D.INDETERMINATE, D.CONTINUE}) == 4      # action_for 가 넷을 서로 다른 동작으로 옮긴다


def test_상한과_단가는_규칙_파일에서_오고_실행_인자는_좁히기만_한다(tmp_path):
    rules = E.load_check_rules(_rule_file(tmp_path))["rules"]
    lim = E.effective_limits(rules)
    base = yaml.safe_load(D.RULES_PATH.read_bytes().decode("utf-8"))["limits"]
    assert (lim["max_hours"], lim["max_cost_krw"]) == (30.0, 90000.0)
    assert lim["cost_krw_of_hour"] == pytest.approx(base["price_krw_per_hour"] * base["safety_factor"])
    assert E.effective_limits(rules, max_hours=5, max_won=10000)["max_hours"] == 5.0
    for kw in ({"max_hours": 31}, {"max_won": 90001}):
        with pytest.raises(E.EccRefused) as exc:
            E.effective_limits(rules, **kw)
        assert exc.value.code == "ecc_budget_widen"


# ================================================================ T9 · T10
def test_T9_상한을_넘을_예상이면_시작_재개하지_않고_세_수를_원장에_적는다(tmp_path):
    w = _world(tmp_path)
    budget = {"spent_hours": 9.0, "next_hours": 2.0, "max_hours": 10.0, "max_won": None, "won_per_hour": 2500}
    with pytest.raises(E.EccRefused) as exc:
        _run(w, budget=budget)
    assert exc.value.code == "ecc_budget_exceeded" and w.loader.n == 1
    rows = E.ledger_rows(w.out)
    assert rows[-1]["event"] == "ecc_budget_exceeded"
    detail = json.loads(rows[-1]["detail"])
    assert (detail["spent_hours"], detail["next_hours"], detail["max_hours"]) == (9.0, 2.0, 10.0)
    ok = {**budget, "spent_hours": 1.0}
    assert _run(w, budget=ok)["step"] == 2
    won = {**budget, "spent_hours": 1.0, "max_hours": None, "max_won": 5000}
    with pytest.raises(E.EccRefused):
        _run(w, resume_step=2, budget=won)                              # 3 h × 2,500 원 > 5,000 원
    assert not E.step_paths(w.out, 4)["meta"].exists()
    with pytest.raises(E.EccRefused) as exc:                            # 쓴 시간이 점검 원장의 누적 처리 시간보다 작다(20b m-3)
        _run(w, resume_step=2, budget={**budget, "spent_hours": 0.0, "max_hours": 100.0})
    assert exc.value.code == "ecc_spent_below_ledger"
    with pytest.raises(E.EccRefused) as exc:                            # 단가가 없다(20b I-2 — 기본값을 두지 않는다)
        _run(w, resume_step=2, budget={**budget, "spent_hours": 1.0, "max_hours": 100.0, "won_per_hour": None})
    assert exc.value.code == "ecc_rate_missing"


def test_T10_평가_쪽_결정별_다음_동작():
    F = D
    assert E.action_for({"outcome": F.PASS, "step_index": 0}, step=2, steps=STEPS) == "end_pass"
    assert E.action_for({"outcome": F.CONTINUE, "step_index": 0}, step=2, steps=STEPS) == "continue"
    assert E.action_for({"outcome": F.CONTINUE, "step_index": 1}, step=4, steps=STEPS) == "continue"  # 두 step 연속의 보류
    assert E.action_for({"outcome": F.INDETERMINATE, "step_index": 2}, step=8, steps=STEPS) == "stop_undecidable"
    assert E.action_for({"outcome": F.FAIL, "step_index": 1}, step=4, steps=STEPS) == "stop_fail"
    for bad, step in (({"outcome": F.CONTINUE, "step_index": 2}, 8),       # 마지막 step 뒤에 이을 step 이 없다
                      ({"outcome": F.PASS, "step_index": 1}, 2),           # 결정의 step 자리가 다르다
                      ({"outcome": F.PASS}, 3),                            # 등록 밖 step
                      ({"outcome": "pass"}, 2)):                           # 평가 쪽 결정 문자열이 아니다
        with pytest.raises(E.EccRefused):
            E.action_for(bad, step=step, steps=STEPS)


def test_시작한_시도는_준비_기록으로_세고_착수_고정이_다르면_세지_않는다(tmp_path):
    for name, pin in (("ecc-20261013T000001", "a" * 64), ("ecc-20261013T000002", "a" * 64),
                      ("ecc-20261013T000003", "b" * 64), ("reh-20261013T000004", "a" * 64)):
        (tmp_path / name).mkdir()
        (tmp_path / name / E.PREPARE_FILE).write_text(json.dumps({"pin_sha256": pin}), encoding="utf-8")
    (tmp_path / "ecc-20261013T000005").mkdir()                         # 준비 기록이 없는 루트
    assert E.attempts_started(tmp_path, "a" * 64) == 2
    assert E.attempts_started(tmp_path, "b" * 64) == 1
    assert E.attempts_started(tmp_path / "없다", "a" * 64) == 0


def test_시도_기록은_칸이_다_있어야_하고_덧붙인다(tmp_path):
    row = {"run_id": "ecc-1", "pin_sha256": "a" * 64, "last_step": 4, "hours": 1.5, "won": 3750, "reason": "fail"}
    E.record_attempt(tmp_path, row)
    E.record_attempt(tmp_path, {**row, "run_id": "ecc-2"})
    lines = (tmp_path / E.ATTEMPTS_FILE).read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["run_id"] for x in lines] == ["ecc-1", "ecc-2"]
    with pytest.raises(E.EccRefused):
        E.record_attempt(tmp_path, {"run_id": "x"})


# ================================================================ T8 · T14 — 본실험 칸의 문
def _gate_world(tmp: Path, *, accept: bool = True):
    """점검 통과를 받은 시드 폴더 — 평가 쪽 `build_output` 의 통과 파일을 `accept_permit` 으로 받아 사본과 키 기록을 쓴다."""
    w = _world(tmp)
    seed_dir = tmp / "seed1"
    main = _spec(tmp, purpose="main", run_stamp="main_s1", run_root=None, non_main_parent=None, standin_allowed=False,
                 pin_path=w.pin, ecc_permit_path=seed_dir / "ecc_permit.json")
    rule = _rule_file(tmp)
    rules = E.load_check_rules(rule)["rules"]
    digest = init_probe(main.init_adapter_path)["init_adapter_digest"]
    _, pin_sha = read_pin(w.pin)
    expected = {"start_adapter_digest": digest, "pin_sha256": pin_sha, "check_run_id": "uni_central-ecc", "seed": 1}
    src = tmp / "early_coord_check.json"
    src.write_bytes(_permit_bytes(tmp, rules, expected))
    key = None
    if accept:
        key = E.accept_permit(src, rules=rules, expected=expected, seed_dir=seed_dir, permit_name="ecc_permit.json",
                              attempts_parent=tmp / "ecc_parent",
                              attempt_row={"run_id": "uni_central-ecc", "pin_sha256": pin_sha, "last_step": 100,
                                           "hours": 0.5, "won": 1500},
                              extra={"attempt": 1, "ecc_root": ECC_ROOT})
    return SimpleNamespace(w=w, main=main, digest=digest, rule=rule, rules=rules, expected=expected, src=src,
                           seed_dir=seed_dir, key=key, pin_sha=pin_sha)


def _gate(g, spec=None, **kw):
    spec = spec or g.main
    return main_start_gate(spec, init_digest=kw.pop("init_digest", g.digest), pin_probes=_probes(spec),
                           rule_path=kw.pop("rule_path", g.rule), **kw)


def test_T14_점검_통과를_받으면_키_기록이_통과_파일_밖에_서고_본_칸은_실물_verify_permit_을_지난다(tmp_path):
    from vlm.train_cell import PERMIT_EXPECTED_KEYS

    g = _gate_world(tmp_path)
    assert g.key["permit_sha256"] == _sha(g.src) == _sha(g.seed_dir / "ecc_permit.json")
    assert {k: g.key[k] for k in PERMIT_EXPECTED_KEYS} == g.expected
    (row,) = [json.loads(x) for x in (tmp_path / "ecc_parent" / E.ATTEMPTS_FILE).read_text(encoding="utf-8").splitlines()]
    assert (row["reason"], row["permit_sha256"]) == ("pass", g.key["permit_sha256"])
    gate = _gate(g)
    assert gate["ecc_permit_sha256"] == g.key["permit_sha256"] and gate["permit_expected"] == g.expected
    assert tuple(gate["permit_expected"]) == PERMIT_EXPECTED_KEYS
    with pytest.raises(E.EccRefused) as exc:                            # 한 시드에 한 번 — 덮지 않는다
        E.accept_permit(g.src, rules=g.rules, expected=g.expected, seed_dir=g.seed_dir, permit_name="ecc_permit.json",
                        attempts_parent=tmp_path / "ecc_parent", attempt_row=dict(row), extra={"attempt": 1, "ecc_root": "x"})
    assert exc.value.code == "ecc_permit_exists"


@pytest.mark.parametrize("case", ["permit_bytes", "key_run_id", "key_pin", "key_missing", "check_file", "rules_changed",
                                  "permit_missing", "eval_side"])
def test_T14_본_칸의_착수_허가_음성(tmp_path, case, monkeypatch):
    g = _gate_world(tmp_path)
    kp = g.seed_dir / E.PERMIT_KEY_FILE
    key = json.loads(kp.read_text(encoding="utf-8"))
    kw, spec = {}, None
    if case == "permit_bytes":
        cp = g.seed_dir / "ecc_permit.json"
        cp.write_bytes(cp.read_bytes() + b" ")
        want = ("ecc_permit_rejected", "PERMIT_SHA")
    elif case == "key_run_id":
        kp.write_text(json.dumps({**key, "check_run_id": "다른 점검 실행"}), encoding="utf-8")
        want = ("ecc_permit_rejected", "PERMIT_MISMATCH")
    elif case == "key_pin":
        kp.write_text(json.dumps({**key, "pin_sha256": "f" * 64}), encoding="utf-8")
        want = ("ecc_permit_key_mismatch", "pin_sha256")
    elif case == "key_missing":
        kp.unlink()
        want = ("ecc_permit_key_missing", "")
    elif case == "check_file":
        (tmp_path / "mf" / "gold_echo.json").write_text("{}", encoding="utf-8")
        want = ("ecc_permit_rejected", "PERMIT_MISMATCH")
    elif case == "rules_changed":
        kw["rule_path"] = _rule_file(tmp_path, gen_limit=256)
        want = ("ecc_permit_rejected", "PERMIT_MISMATCH")
    elif case == "permit_missing":
        spec = _spec(tmp_path, purpose="main", run_stamp="main_s1", run_root=None, non_main_parent=None,
                     pin_path=g.w.pin, ecc_permit_path=tmp_path / "없다" / "ecc_permit.json")
        want = ("ecc_permit_missing", "")
    else:
        monkeypatch.setattr(E, "EVAL_SIDE", "evaluation.no_such_module_for_test")
        want = ("ecc_eval_side_missing", "")
    with pytest.raises(CellRejected) as exc:
        _gate(g, spec, **kw)
    assert exc.value.code == want[0] and want[1] in str(exc.value)


def test_T14_점검_실행_쪽_값이_통과_파일과_다르면_받지_않고_아무것도_쓰지_않는다(tmp_path):
    g = _gate_world(tmp_path, accept=False)
    with pytest.raises(E.EccRefused) as exc:
        E.accept_permit(g.src, rules=g.rules, expected={**g.expected, "seed": 2}, seed_dir=g.seed_dir,
                        permit_name="ecc_permit.json", attempts_parent=tmp_path / "ecc_parent",
                        attempt_row={"run_id": "r", "pin_sha256": g.pin_sha, "last_step": 100, "hours": 0.5, "won": 1500},
                        extra={"attempt": 1, "ecc_root": ECC_ROOT})
    assert exc.value.code == "ecc_permit_rejected"
    assert not g.seed_dir.exists() and not (tmp_path / "ecc_parent").exists()


def test_T14_착수_고정이_다르거나_출발_어댑터가_다르면_허가를_보기_전에_거부한다(tmp_path):
    g = _gate_world(tmp_path, accept=False)                             # 허가 파일 · 키가 없어도 착수 고정이 먼저 막는다
    with pytest.raises(CellRejected) as exc:
        main_start_gate(g.main, init_digest=g.digest, rule_path=g.rule,
                        pin_probes=_probes(g.main, env=lambda: {"gpu_name": "다른 장치"}))
    assert exc.value.code == "pin_mismatch"
    with pytest.raises(CellRejected) as exc:
        _gate(g, init_digest="d" * 64)
    assert exc.value.code == "pin_mismatch"


def test_T8_본실험_칸은_착수_고정과_허가의_자리_없이_가드를_지나지_않는다(tmp_path, monkeypatch):
    import vlm.uni_config as U
    from vlm.train_cell import _preflight

    monkeypatch.setattr(U, "main_run_problems", lambda *a, **k: [])
    monkeypatch.setattr(U, "load_uni_config", lambda p=None: SimpleNamespace())

    def spec(**over):
        return _spec(tmp_path, purpose="main", run_stamp="main_s1", run_root=None, non_main_parent=None,
                     standin_allowed=False, prompt_path=str(pilot_vlm.PROMPT_PATH), **over)

    for over, code in (({"pin_path": None, "ecc_permit_path": tmp_path / "p.json"}, "main_pin_missing"),
                       ({"pin_path": tmp_path / "PIN.json", "ecc_permit_path": None}, "main_permit_missing")):
        with pytest.raises(CellRejected) as exc:
            _preflight(spec(**over), model_loader=None, rows_given=False, hooks=None)
        assert exc.value.code == code
    _preflight(spec(pin_path=tmp_path / "PIN.json", ecc_permit_path=tmp_path / "p.json"),
               model_loader=None, rows_given=False, hooks=None)       # 자리가 있으면 지나고, 값은 칸의 문이 본다


def test_T8_통과_뒤_본_중앙은_점검_실행의_파일을_읽지_않는다(tmp_path):
    """점검 실행의 루트는 본실험 출력 금지 부모 아래이고, 본실험 칸의 학습 호출에는 점검 멈춤 · 점검 재개가 없다."""
    import inspect

    from vlm import train_cell as TC
    from vlm.uni_config import NON_MAIN_PARENTS

    assert Path("outputs/early_check") in NON_MAIN_PARENTS
    src = inspect.getsource(TC._run_cell)
    assert "check_stop" not in src and "resume_from" not in src
    w = _world(tmp_path)
    _run(w)
    before = _tree(tmp_path / ECC_ROOT)
    loader = CountingLoader()
    init = np.load(w.spec.init_adapter_path)
    _, _, m, _ = pilot_vlm.train_rounds(
        rows=w.rows, epochs=1, round_idx=0, client_idx=0, base_seed=7, adapter_in=[init[k] for k in init.files],
        adapter_keys=list(init.files), model_id="tiny", run_id="main-like", init_seed=7, model_revision="r1",
        model_loader=loader, purpose="rehearsal", standin_allowed=True, tag=CENTRAL_TAG,
        chat_template_kwargs={"enable_thinking": False}, processor_kwargs={})
    assert m["resume_identity"]["input_adapter_digest"] == init_probe(w.spec.init_adapter_path)["init_adapter_digest"]
    assert _tree(tmp_path / ECC_ROOT) == before                         # 점검 실행의 파일을 건드리지 않았다


# ================================================================ T6 · 점검 export 의 학습 쪽 출처(별도 분기)
def _export_world(tmp: Path):
    from tests.uni_fakes import write_pairs

    w = _world(tmp)
    write_pairs(Path(w.spec.pairs_path), w.rows)
    assert _run(w)["step"] == 2
    m = _meta(w, 2)
    ident = m["identity"]
    kw = {"measured": {"processor_config_sha256": ident["processor_config_sha256"]}, "coord_hash": ident["coord_cfg_hash"],
          "cfg": SimpleNamespace(pairs_path=w.spec.pairs_path),
          "reg": SimpleNamespace(generation=SimpleNamespace(train_rows_digest=None)), "run_root": tmp / ECC_ROOT}
    return w, kw


def test_T6_점검_export_는_step_파일을_meta_원장과_맞대고_도장에_step_을_싣는다(tmp_path):
    from vlm.export_preflight import _training_check_step

    w, kw = _export_world(tmp_path)
    meta, stamp, side = _training_check_step(w.out, step=2, tag=CENTRAL_TAG, **kw)
    m = _meta(w, 2)
    assert stamp["check_step"] == 2 and stamp["scored_adapter_sha256"] == m["adapter"]["sha256"]
    assert side["scored_adapter_step"] == 2 and side["train_config"] == m["train_config"]
    assert meta["init_adapter_digest"] == init_probe(w.spec.init_adapter_path)["init_adapter_digest"]


@pytest.mark.parametrize("case", ["adapter_bytes", "ledger_line", "tag", "step", "rows", "root"])
def test_T6_점검_export_는_어긋난_step_파일을_거부한다(tmp_path, case):
    from vlm.export_preflight import ExportRefused, _training_check_step

    w, kw = _export_world(tmp_path)
    tag, step = CENTRAL_TAG, 2
    if case == "adapter_bytes":
        a = E.step_paths(w.out, 2)["adapter"]
        a.write_bytes(a.read_bytes() + b"x")
        want = "adapter_sha"
    elif case == "ledger_line":
        led = w.out / E.LEDGER_FILE
        led.write_text("".join(ln for ln in led.read_text(encoding="utf-8").splitlines(keepends=True)
                               if ",step,2," not in ln), encoding="utf-8")
        want = "check_ledger"
    elif case == "tag":
        tag, want = "uni_local_C1", "check_tag"
    elif case == "step":
        step, want = 4, "check_step_missing"
    elif case == "rows":
        from tests.uni_fakes import write_pairs

        write_pairs(Path(w.spec.pairs_path), w.rows[:-1])
        want = "train_rows"
    else:
        kw["run_root"], want = tmp_path / "ecc-20991231T000000", "run_root"
    with pytest.raises(ExportRefused) as exc:
        _training_check_step(w.out, step=step, tag=tag, **kw)
    assert exc.value.code == want


def test_T6_본실험_학습_출처는_점검_step_폴더를_받지_않는다(tmp_path):
    from vlm.export_preflight import ExportRefused, _training

    w, kw = _export_world(tmp_path)
    with pytest.raises(ExportRefused) as exc:
        _training(w.out, tag=CENTRAL_TAG, seed_index=1, purpose="main", reg=kw["reg"], measured=kw["measured"],
                  coord_hash=kw["coord_hash"], cfg=kw["cfg"], run_root=None)
    assert exc.value.code == "adapter_missing"                          # step 어댑터만 있고 끝 어댑터 · 닫힌 원장이 없다
