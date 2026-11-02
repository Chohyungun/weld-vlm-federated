"""진단 하네스 04 · 05 의 외부 어댑터 체제 검증(지시 1023a — 외부 검토 200 M-02). 모델 · GPU 없이.

`--adapter` 로 준 어댑터가 **어떤 레시피 · 계약 · 프롬프트로 학습됐는지** 곁의 학습 메타(칸 meta · 하네스 결과 · 초기 어댑터 proof)에서
읽어 실행의 체제(`resolve_regime`)와 맞댄다. 맞는 어댑터는 지나고, 어긋난 어댑터 · 메타 없는 어댑터 · 메타와 결속되지 않은 어댑터는
**모델 · 프로세서를 올리기 전에** 멈춘다. 의도한 교차 탐침은 `--cross-regime` 으로만 받고 산출 이름 · 기록에 교차를 박는다.

모델 적재 함수(04 의 프로세서 적재 · 05 의 적재기 해석)는 대역으로 막는다 — 대역이 불리면 '검사를 지나 적재 자리에 닿았다' 는 뜻이다.
실제 적재는 하지 않는다. 대역 실행 끝까지의 기록(04 결과의 `adapter_regime` · 실행 폴더 이름)은 02g 드라이런이 본다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.probe_fakes import write_probe_snapshot
from vlm import narrative as N
from vlm import pilot_vlm as PV
from vlm import registration as R
from vlm.init_adapter import init_adapter_digest

V1, V2 = "json_only", "json_line_then_basis"
HDIR = "docs/dev_log/2026-10-19-본실험복구"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H04 = _load("h04_adapter_regime_test", f"{HDIR}/04_암기시험_C.py")
H05 = _load("h05_adapter_regime_test", f"{HDIR}/05_손실분해_C.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _regime(recipe: str) -> dict:
    return {"recipe": recipe, "target_contract_sha256": PV.target_contract_sha256(recipe),
            "prompt_sha256": R.regime_prompt_sha256(recipe)}


def _npz(p: Path, *, b_value: float = 0.0, seed: int = 0) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    np.savez(p, **{"m.q.lora_A.weight": rng.standard_normal((2, 4)).astype(np.float32),
                   "m.q.lora_B.weight": np.full((4, 2), b_value, dtype=np.float32)})
    return p


def _tc_text(recipe: str, tc_over: dict | None = None) -> str:
    """학습 설정 원문(합성) — 계약 · 프롬프트 칸과 덮을 칸. 덮는 값이 None 이면 그 칸을 뺀다."""
    tc = {"target_contract_sha256": PV.target_contract_sha256(recipe), "prompt_sha256": R.regime_prompt_sha256(recipe),
          **(tc_over or {})}
    return json.dumps({k: v for k, v in tc.items() if v is not None}, ensure_ascii=False)


def _cell(d: Path, recipe: str, *, tc_over: dict | None = None, **over) -> Path:
    """본 칸 산출의 꼴 — `adapter_last.npz` + `adapter_last.meta.json`(학습 설정 원문에 계약 · 프롬프트, 원문의 UTF-8 sha256 은
    생산자 `vlm.train_cell` 과 같은 셈)."""
    p = _npz(d / "adapter_last.npz", b_value=0.5)
    text = _tc_text(recipe, tc_over)
    meta = {"adapter_sha256": _sha(p), "train_config": text,
            "train_config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), **over}
    (d / "adapter_last.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return p


def _harness(d: Path, recipe: str, *, stub: bool, train_over: dict | None = None, regime_over: dict | None = None) -> Path:
    """이 하네스의 학습 산출 꼴 — `adapter.npz` + `result.json`(학습이 실제로 쓴 계약 · 프롬프트, 결과의 `regime` 칸). 덮는 값이
    None 이면 그 칸을 뺀다."""
    p = _npz(d / "adapter.npz", b_value=0.25)
    train = {"adapter_sha256": _sha(p), "target_contract_sha256": PV.target_contract_sha256(recipe),
             "prompt_sha256": R.regime_prompt_sha256(recipe), **(train_over or {})}
    regime = {**_regime(recipe), **(regime_over or {})}
    res = {"run_name": d.name, "stub": stub, "regime": {k: v for k, v in regime.items() if v is not None},
           "train": {k: v for k, v in train.items() if v is not None}}
    (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    return p


def _initial(d: Path, *, b_value: float = 0.0) -> Path:
    """공통 초기 어댑터 꼴 — `initial.npz`(lora_B = 0) + `initial.proof.json`(digest)."""
    p = _npz(d / "initial.npz", b_value=b_value)
    with np.load(p) as z:
        arrays = [z[k] for k in z.files]
    (d / "initial.proof.json").write_text(json.dumps({"init_adapter_digest": init_adapter_digest(arrays)}), encoding="utf-8")
    return p


# ================================================================ 곁 메타의 판독(`adapter_regime`)
def test_칸_산출은_meta_의_학습_설정_원문에서_체제를_읽는다(tmp_path):
    for recipe in (V1, V2):
        p = _cell(tmp_path / recipe, recipe)
        got = H04.adapter_regime(p, stub=False)
        assert (got["kind"], got["recipe"], got["meta_file"]) == ("cell", recipe, "adapter_last.meta.json")
        assert (got["target_contract_sha256"], got["prompt_sha256"]) == (_regime(recipe)["target_contract_sha256"],
                                                                        R.regime_prompt_sha256(recipe))
        meta = json.loads((p.parent / "adapter_last.meta.json").read_text(encoding="utf-8"))
        assert got["adapter_sha256"] == _sha(p)
        assert got["train_config_sha256"] == hashlib.sha256(meta["train_config"].encode("utf-8")).hexdigest()
    # 다른 시드 · 예산으로 학습한 어댑터도 받는다 — 원문과 그 해시만 맞으면 된다(지금 등록과 같을 필요는 없다)
    other = H04.adapter_regime(_cell(tmp_path / "other", V1, tc_over={"seed": 999, "budget_fraction": 0.125}), stub=False)
    assert (other["kind"], other["recipe"]) == ("cell", V1)


@pytest.mark.parametrize("line", ["target_contract_sha256", "prompt_sha256"])
def test_칸_meta_의_학습_설정_원문을_고치고_옛_해시를_두면_교차로도_받지_않는다(tmp_path, line):
    """외부 검토 204 권고 1 — 곁 meta 가 주장하는 원문 해시와 원문 자체를 맞댄다. 어댑터 · 해시는 그대로 두고 원문의 계약(또는
    프롬프트) 줄만 2 차 값으로 고친 meta 는 읽는 자리에서 멈춘다 — 교차 허용 전이다."""
    p = _cell(tmp_path / "c", V1)
    meta_p = p.parent / "adapter_last.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    old_sha = meta["train_config_sha256"]
    meta["train_config"] = _tc_text(V1, {line: _regime(V2)[line]})               # 원문만 바뀐다
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    assert meta["train_config_sha256"] == old_sha
    with pytest.raises(SystemExit, match="train_config_sha256"):
        H04.adapter_regime(p, stub=False)
    meta["train_config_sha256"] = "T" * 64                                        # 꼴이 아닌 해시도
    meta_p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="train_config_sha256"):
        H04.adapter_regime(p, stub=False)


@pytest.mark.parametrize(("kind", "field", "value", "needle"), [
    ("cell", "prompt_sha256", None, "프롬프트 sha256"),                    # 프롬프트 없는 원문
    ("cell", "prompt_sha256", "A" * 64, "프롬프트 sha256"),                # 대문자 — 꼴이 아니다
    ("cell", "prompt_sha256", "a" * 63, "프롬프트 sha256"),                # 자리 수
    ("cell", "target_contract_sha256", None, "타깃 계약 sha256"),
    ("cell", "target_contract_sha256", "zz", "타깃 계약 sha256"),
    ("cell", "target_contract_sha256", "0" * 64, "아는 레시피"),            # 꼴은 맞지만 모르는 계약
    ("harness", "prompt_sha256", None, "프롬프트 sha256"),
    ("harness", "prompt_sha256", "g" * 64, "프롬프트 sha256"),
    ("harness", "target_contract_sha256", None, "타깃 계약 sha256"),
])
def test_필수_항목이_없거나_꼴이_아닌_메타는_교차_플래그로도_받지_않는다(tmp_path, kind, field, value, needle):
    """외부 검토 204 권고 2 — 원래 체제의 필수 항목(계약 · 프롬프트 sha256 의 꼴, 계약에서 되찾는 레시피)을 교차 허용 전에 본다."""
    if kind == "cell":
        p = _cell(tmp_path / "x", V1, tc_over={field: value})                     # 원문과 해시는 서로 맞다 — 칸만 빠졌다 · 틀렸다
    else:
        p = _harness(tmp_path / "x", V1, stub=False, train_over={field: value}, regime_over={field: value})
    with pytest.raises(SystemExit, match=needle):
        H04.adapter_regime(p, stub=False)
    with pytest.raises(SystemExit, match=needle):                                 # `--cross-regime` 에서도 판독에서 멈춘다
        H04.check_adapter_regime(H04.adapter_regime(p, stub=False), _regime(V2), cross=True)


def test_하네스_결과의_체제_칸이_학습_칸과_다르면_받지_않는다(tmp_path):
    p = _harness(tmp_path / "h", V1, stub=False, regime_over=_regime(V2))           # 학습 칸은 1 차 · 체제 칸은 2 차
    with pytest.raises(SystemExit, match="regime"):
        H04.adapter_regime(p, stub=False)
    ok = H04.adapter_regime(_harness(tmp_path / "ok", V1, stub=False), stub=False)
    got = H04.check_adapter_regime(ok, _regime(V2), cross=True)                     # 온전한 메타 + 교차 — 받고 표시한다
    assert (got["cross"], got["matches_regime"], got["recipe"]) == (True, False, V1)


def test_하네스_산출은_result_의_학습_칸에서_체제를_읽고_대역_실제를_섞지_않는다(tmp_path):
    p = _harness(tmp_path / "h", V1, stub=True)
    got = H04.adapter_regime(p, stub=True)
    assert (got["kind"], got["recipe"], got["stub"], got["meta_file"]) == ("harness", V1, True, "result.json")
    with pytest.raises(SystemExit, match="대역 실행의 어댑터"):
        H04.adapter_regime(p, stub=False)


def test_생성_전_중단된_하네스도_학습완료_메타로_읽는다(tmp_path):
    p = _harness(tmp_path / "h", V2, stub=True)
    result = p.with_name("result.json")
    doc = json.loads(result.read_bytes())
    doc.update(kind="probe_train_done", schema=1)
    p.with_name("TRAIN_DONE.json").write_text(json.dumps(doc), encoding="utf-8")
    result.unlink()
    got = H04.adapter_regime(p, stub=True)
    assert got["meta_file"] == "TRAIN_DONE.json" and got["recipe"] == V2


def test_초기_어댑터는_proof_와_lora_B_가_0_인지로_체제_없는_어댑터다(tmp_path):
    got = H04.adapter_regime(_initial(tmp_path / "i"), stub=False)
    assert (got["kind"], got["recipe"], got["target_contract_sha256"]) == ("initial", None, None)
    with pytest.raises(SystemExit, match="lora_B"):                                   # 학습한 어댑터를 초기 어댑터로 받지 않는다
        H04.adapter_regime(_initial(tmp_path / "trained", b_value=0.1), stub=False)
    bad = _initial(tmp_path / "bad")
    (bad.parent / "initial.proof.json").write_text(json.dumps({"init_adapter_digest": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="proof"):
        H04.adapter_regime(bad, stub=False)


@pytest.mark.parametrize("case", ["no_meta", "cell_unbound", "cell_no_train_config", "cell_unknown_contract",
                                  "harness_unbound", "missing_file"])
def test_메타가_없거나_어댑터와_결속되지_않으면_받지_않는다(tmp_path, case):
    if case == "no_meta":
        p = _npz(tmp_path / "x" / "adapter_last.npz")
        needle = "학습 메타"
    elif case == "cell_unbound":
        p = _cell(tmp_path / "c", V1)
        p.write_bytes(p.read_bytes() + b"\0")                                       # 곁 meta 는 그대로 · 바이트만 다르다
        needle = "adapter_sha256"
    elif case == "cell_no_train_config":
        p = _cell(tmp_path / "c", V1, train_config=None)
        needle = "학습 설정 원문"
    elif case == "cell_unknown_contract":
        p = _cell(tmp_path / "c", V1, tc_over={"target_contract_sha256": "0" * 64})
        needle = "아는 레시피"
    elif case == "harness_unbound":
        p = _harness(tmp_path / "h", V1, stub=False)
        p.write_bytes(p.read_bytes() + b"\0")
        needle = "train.adapter_sha256"
    else:
        p = tmp_path / "none" / "adapter.npz"
        needle = "없다"
    with pytest.raises(SystemExit, match=needle):
        H04.adapter_regime(p, stub=False)


# ================================================================ 체제 대조(`check_adapter_regime`)
def test_어댑터의_체제가_실행과_같으면_받고_다르면_교차_플래그로만_받아_기록에_박는다(tmp_path):
    v1 = H04.adapter_regime(_cell(tmp_path / "v1", V1), stub=False)
    v2 = H04.adapter_regime(_cell(tmp_path / "v2", V2), stub=False)
    ok = H04.check_adapter_regime(v1, _regime(V1), cross=False)
    assert (ok["matches_regime"], ok["cross"]) == (True, False) and "note" not in ok
    for info, regime in ((v1, _regime(V2)), (v2, _regime(V1))):                     # 새 체제 + 1 차 어댑터, 그 반대
        with pytest.raises(SystemExit, match="--cross-regime"):
            H04.check_adapter_regime(info, regime, cross=False)
        got = H04.check_adapter_regime(info, regime, cross=True)
        assert (got["matches_regime"], got["cross"]) == (False, True) and "정상 재현이 아니다" in got["note"]
    init = H04.adapter_regime(_initial(tmp_path / "i"), stub=False)
    for recipe in (V1, V2):                                                        # 초기 어댑터는 두 체제에서 받는다
        got = H04.check_adapter_regime(init, _regime(recipe), cross=False)
        assert (got["matches_regime"], got["cross"]) == (None, False)


def test_05_의_어댑터_계획은_교차일_때만_결과_파일_이름에_교차를_박는다(tmp_path):
    v1, init = _cell(tmp_path / "v1", V1), _initial(tmp_path / "i")
    out = tmp_path / "loss.json"
    adapters, checked, got = H05.plan_adapters(H04, [f"a={v1}", f"init={init}"], tmp_path, _regime(V1), cross=False, out=out)
    assert got == out and set(adapters) == {"a", "init"}
    assert (checked["a"]["cross"], checked["init"]["kind"]) == (False, "initial")
    with pytest.raises(SystemExit, match="--cross-regime"):
        H05.plan_adapters(H04, [f"a={v1}"], tmp_path, _regime(V2), cross=False, out=out)
    _a, checked, got = H05.plan_adapters(H04, [f"a={v1}", f"init={init}"], tmp_path, _regime(V2), cross=True, out=out)
    assert got.name == "loss.cross-regime.json" and checked["a"]["cross"] and not checked["init"]["cross"]
    with pytest.raises(SystemExit, match="교차가 아닌"):                             # 교차가 없는데 플래그
        H05.plan_adapters(H04, [f"a={v1}", f"init={init}"], tmp_path, _regime(V1), cross=True, out=out)
    with pytest.raises(SystemExit, match="학습 메타"):
        H05.plan_adapters(H04, [f"x={_npz(tmp_path / 'bare' / 'adapter.npz')}"], tmp_path, _regime(V1), cross=False, out=out)


# ================================================================ 실행 순서 — 모델 · 프로세서를 올리기 전에
class Loaded(Exception):
    """대역 적재 — 불리면 검사를 지나 적재 자리에 닿았다는 뜻이다(실제 적재는 하지 않는다)."""


@pytest.fixture
def harness_env(monkeypatch):
    """하네스가 바꾸는 작업 폴더 · 경로 · 환경을 시험 뒤에 되돌린다. 실제 적재 함수(프로세서 · 모델)는 불리면 `Loaded` 를 내고 그
    이름을 `calls` 에 적는다 — 거부 시험은 `calls == []`(적재 호출 0)를 본다."""
    import transformers

    calls: list[str] = []

    def stop(name):
        def f(*a, **k):
            calls.append(name)
            raise Loaded
        return f

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", stop("AutoProcessor.from_pretrained"))
    monkeypatch.setattr(PV, "_load_model", stop("_load_model"))
    return calls


@pytest.fixture
def no_load(harness_env, monkeypatch):
    """04 의 프로세서 적재 · 05 의 적재기 해석 · 학습 적재기를 대역으로 막는다(`harness_env` 의 `calls` 를 돌려준다)."""
    def stop(*a, **k):
        harness_env.append("resolve_model_loader")
        raise Loaded

    monkeypatch.setattr(PV, "resolve_model_loader", stop)
    return harness_env


def _r2_args(tmp: Path) -> list[str]:
    raw, _where = N.canonical_vector_bytes()
    vec = tmp / "canonical.json"
    vec.write_bytes(raw)
    pairs, digest = write_probe_snapshot(tmp / "synthetic_pairs")
    return ["--recipe", V2, "--prompt-file", "vlm/prompts/unified_v3_1_basis.txt", "--max-new-tokens", "1024",
            "--narrative-vectors", str(vec), "--pairs", str(pairs), "--pairs-digest", digest]


def test_04_실제_실행은_어댑터의_체제를_프로세서_전에_맞대고_맞으면_적재_자리로_간다(tmp_path, no_load):
    out = tmp_path / "out"
    pairs, digest = write_probe_snapshot(tmp_path / "v1_pairs")
    pair_args = ["--pairs", str(pairs), "--pairs-digest", digest]
    base = ["--stage", "2", "--out-root", str(out)]
    with pytest.raises(Loaded):                                                     # 1 차 체제 + 1 차 칸 어댑터
        H04.main([*base, *pair_args, "--run-name", "ok", "--adapter", str(_cell(tmp_path / "v1", V1))])
    with pytest.raises(Loaded):                                                     # 2 차 체제 + 2 차 하네스 어댑터
        H04.main([*base, "--run-name", "ok2", "--adapter", str(_harness(tmp_path / "h2", V2, stub=False)),
                  *_r2_args(tmp_path)])
    with pytest.raises(Loaded):                                                     # 초기 어댑터 — 두 체제
        H04.main([*base, *pair_args, "--run-name", "init", "--adapter", str(_initial(tmp_path / "i"))])
    assert not out.exists()


@pytest.mark.parametrize("case", ["new_regime_old_adapter", "old_regime_new_adapter", "no_meta", "cross_flag_without_cross",
                                  "cross_flag_without_adapter"])
def test_04_어긋난_어댑터_메타_없는_어댑터는_프로세서_전에_멈추고_실행_폴더를_만들지_않는다(tmp_path, no_load, case):
    out = tmp_path / "out"
    base = ["--stage", "2", "--run-name", "g", "--out-root", str(out)]
    extra = {"new_regime_old_adapter": ["--adapter", str(_cell(tmp_path / "v1", V1)), *_r2_args(tmp_path)],
             "old_regime_new_adapter": ["--adapter", str(_cell(tmp_path / "v2", V2))],
             "no_meta": ["--adapter", str(_npz(tmp_path / "bare" / "adapter_last.npz"))],
             "cross_flag_without_cross": ["--adapter", str(_cell(tmp_path / "v1", V1)), "--cross-regime"],
             "cross_flag_without_adapter": ["--cross-regime"]}[case]
    with pytest.raises(SystemExit) as exc:                                          # `Loaded` 가 아니다 — 적재 자리 전
        H04.main([*base, *extra])
    assert not isinstance(exc.value, Loaded) and exc.value.code not in (0, None)
    assert not out.exists()


def test_04_교차_탐침은_교차_꼬리를_단_실행_폴더_이름으로_돈다(tmp_path, no_load):
    out = tmp_path / "out"
    base = ["--stage", "2", "--run-name", "g", "--out-root", str(out), "--adapter", str(_cell(tmp_path / "v1", V1)),
            "--cross-regime", *_r2_args(tmp_path)]
    (out / "g").mkdir(parents=True)                                                 # 꼬리 없는 이름은 정상 재현의 자리 — 겹치지 않는다
    with pytest.raises(Loaded):
        H04.main(base)
    (out / ("g" + H04.CROSS_TAG)).mkdir()
    with pytest.raises(SystemExit, match=r"g\.cross-regime"):                       # 같은 교차 이름은 다시 돌지 않는다
        H04.main(base)


def test_05_실제_실행은_어댑터마다_체제를_적재기_전에_맞댄다(tmp_path, no_load):
    out = tmp_path / "loss.json"
    pairs, digest = write_probe_snapshot(tmp_path / "v1_pairs")
    pair_args = ["--pairs", str(pairs), "--pairs-digest", digest]
    ok = ["--adapter", f"a={_cell(tmp_path / 'v1', V1)}", "--adapter", f"init={_initial(tmp_path / 'i')}"]
    with pytest.raises(Loaded):
        H05.main([str(tmp_path), "--out", str(out), *pair_args, *ok])
    for extra in (["--adapter", f"a={_cell(tmp_path / 'v2', V2)}"],                  # 1 차 체제 + 2 차 어댑터
                  ["--recipe", V2, "--prompt-file", "vlm/prompts/unified_v3_1_basis.txt",
                   "--adapter", f"a={_cell(tmp_path / 'v1', V1)}"],                  # 2 차 체제 + 1 차 어댑터
                  ["--adapter", f"x={_npz(tmp_path / 'bare' / 'adapter.npz')}"]):     # 메타 없음
        with pytest.raises(SystemExit) as exc:
            H05.main([str(tmp_path), "--out", str(out), *pair_args, *extra])
        assert not isinstance(exc.value, Loaded)
    with pytest.raises(SystemExit, match="조용히 버리지 않는다"):                       # 대역 · 정렬만은 --adapter 를 받지 않는다
        H05.main(["--stub", "--out", str(out), *ok])
    assert not out.exists()


# ================================================================ 두 진입점 · 05 실제 경로(지시 1023b — 외부 검토 204 의 작은 보완)
def test_두_진입점_모두_어댑터_바이트가_곁_메타와_다르면_적재_호출_0_으로_멈춘다(tmp_path, no_load):
    """곁 메타는 그대로 두고 어댑터 바이트만 바꾼다(결속 불일치) — 04 · 05 의 실제 실행 진입점이 모두 적재 전에 멈춘다."""
    for which in ("cell", "harness"):
        d = tmp_path / which
        p = _cell(d, V1) if which == "cell" else _harness(d, V1, stub=False)
        p.write_bytes(p.read_bytes() + b"\0")
        with pytest.raises(SystemExit, match="adapter_sha256"):
            H04.main(["--stage", "2", "--run-name", f"g_{which}", "--out-root", str(tmp_path / "out"), "--adapter", str(p)])
        with pytest.raises(SystemExit, match="adapter_sha256"):
            H05.main([str(tmp_path), "--out", str(tmp_path / f"loss_{which}.json"), "--adapter", f"a={p}"])
    assert no_load == [] and not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("loss_*.json"))


def test_05_정렬만_대역_실행은_adapter_와_교차_플래그를_받지_않는다_적재_호출_0(tmp_path, no_load):
    p = _cell(tmp_path / "v1", V1)
    for mode in ("--align-only", "--stub"):
        for extra in (["--adapter", f"a={p}"], ["--cross-regime"]):
            with pytest.raises(SystemExit, match="조용히 버리지 않는다"):
                H05.main([mode, "--out", str(tmp_path / "loss.json"), *extra])
    assert no_load == [] and not (tmp_path / "loss.json").exists()


def _fake_cell(d: Path, recipe: str) -> Path:
    """대역 모델(`tests.uni_fakes.fake_loader`)의 어댑터 키 · 꼴로 된 칸 산출 — 05 실제 경로가 대역 모델에 주입할 수 있다.
    lora_B 를 0 에서 옮겨 학습한 어댑터의 꼴로 둔다."""
    from peft import get_peft_model_state_dict

    from detection import serialize
    from tests.uni_fakes import fake_loader

    model, _proc = fake_loader(None, init_seed=0)
    sd = get_peft_model_state_dict(model)
    keys = serialize.canonical_keys(sd)
    arrays = [a + np.float32(0.01) for a in serialize.state_dict_to_ndarrays(sd, keys)]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "adapter_last.npz"
    np.savez(p, **dict(zip(keys, arrays)))
    text = _tc_text(recipe)
    meta = {"adapter_sha256": _sha(p), "train_config": text,
            "train_config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    (d / "adapter_last.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return p


def test_05_실제_경로의_교차_탐침은_결과_파일_이름과_기록에_교차를_박는다(tmp_path, harness_env, monkeypatch):
    """05 실제 경로를 대역 모델 · 대역 프로세서로 끝까지 돈다 — 적재기 해석만 대역이다(실제 프로세서 · 모델 적재 0). 1 차 칸 어댑터를
    2 차 체제에서 받으면: 플래그 없이는 적재 전에 멈추고, `--cross-regime` 이면 결과 파일 이름에 `.cross-regime` 이 붙고 기록의
    `adapter_regime` 에 교차가 박힌다."""
    from tests.uni_fakes import fake_loader, make_rows

    rows = [{**r, "width_px": 64, "height_px": 32, "target_text": f"{r['image_id']} 의 서술이다."}
            for c, n in (("C1", 2), ("C3", 1)) for r in make_rows(tmp_path / "img", c, n)]
    pairs, digest = write_probe_snapshot(tmp_path / "pairs", rows)
    rows_file = tmp_path / "rows.txt"
    rows_file.write_text("".join(r["image_id"] + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setattr(H05, "PAIRS", str(pairs))
    seen: list[str] = []

    def loader(model_id, *, init_seed, revision=None, processor_kwargs=None):
        seen.append(str(model_id))
        model, _p = fake_loader(None, init_seed=init_seed)
        proc = H05.CharProcessor()
        proc.tokenizer.decode = lambda ids: f"<{int(ids[0])}>"
        return model, proc

    monkeypatch.setattr(PV, "resolve_model_loader", lambda *a, **k: (loader, [{"id": "test-double", "approved": False}]))
    out = tmp_path / "loss.json"
    args = [str(tmp_path), "--out", str(out), "--rows-file", str(rows_file), "--recipe", V2,
            "--prompt-file", "vlm/prompts/unified_v3_1_basis.txt", "--adapter", f"a={_fake_cell(tmp_path / 'v1', V1)}",
            "--pairs", str(pairs), "--pairs-digest", digest]
    with pytest.raises(SystemExit, match="--cross-regime"):                         # 플래그 없이는 적재 전에 멈춘다
        H05.main(args)
    assert seen == [] and not out.exists()
    assert H05.main([*args, "--cross-regime"]) == 0
    tagged = tmp_path / ("loss" + H04.CROSS_TAG + ".json")
    assert tagged.is_file() and not out.exists() and len(seen) == 1 and harness_env == []
    res = json.loads(tagged.read_text(encoding="utf-8"))
    assert res["pairs_input"]["snapshot_digest"] == digest
    ar = res["adapter_regime"]["a"]
    assert (ar["kind"], ar["recipe"], ar["matches_regime"], ar["cross"]) == ("cell", V1, False, True)
    assert "정상 재현이 아니다" in ar["note"] and res["regime"]["recipe"] == V2
    assert res["adapters"]["a"]["checks"]["all_aligned"] and res["adapters"]["a"]["checks"]["n_rows"] == len(rows)
