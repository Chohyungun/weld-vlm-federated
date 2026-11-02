"""학습량 민감도 실행(1017b) — 본실험과 분리된 등록 벌 · 식별자 · 출력 뿌리. 모델 · GPU 없이.

본다: 등록 벌이 본실험 등록과 `budget_fraction` 한 칸만 다른 바이트인지, 뿌리의 실행 묶음(`RUN_PROFILE.json`)이 등록 벌을 묶는지,
그 뿌리의 칸 명세 · 실행 스탬프 · 착수 고정 `common` · 연합 실행 설정이 그 벌에서 오고 본실험 뿌리를 가리키지 않는지, 본실험과 신원이
다른지, 묶음이 없는 뿌리는 지금과 같은지(기본값 회귀 0), 명령줄이 섞인 인자를 거부하는지.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import main_uni as M

MAIN_REG = ROOT / "configs" / "train_registration.yaml"
SENS_REG = ROOT / "configs" / "train_registration_sens025.yaml"


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    """저장소 설정 · 등록 그대로. 예산 몫의 기대 토큰만 대역(비율을 곱한다) — 봉인 페어 · 토큰 표 없이 돈다(그 셈은
    `tests/test_budget_fraction.py` 가 본다)."""
    import vlm.budget as B

    monkeypatch.setattr(B, "expected_tokens",
                        lambda per, *, fraction, **kw: None if per is None else {c: int(v * fraction) for c, v in per.items()})
    from tests.registration_history import historical_base, legacy_default

    legacy_default(monkeypatch, tmp_path)
    return M.load_main_config(historical_base())


def test_등록_벌은_본실험_등록과_budget_fraction_한_칸만_다른_바이트다():
    from tests.registration_history import historical_registration

    original = historical_registration()
    a, b = original.split(b"\n"), SENS_REG.read_bytes().split(b"\n")
    assert len(a) == len(b)
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    assert diff == [(b"budget_fraction: 0.125", b"budget_fraction: 0.25")]
    ra, rb = yaml.safe_load(original), yaml.safe_load(SENS_REG.read_text(encoding="utf-8"))
    assert {k for k in ra if ra[k] != rb[k]} == {"budget_fraction"}


def test_실행_묶음은_뿌리에_등록_벌을_한_번_묶고_다른_벌_본실험_뿌리를_거부한다(tmp_path):
    root = tmp_path / "sens_u025"
    assert M.bind_profile(root, SENS_REG) == root.resolve()
    rec = json.loads((root / M.PROFILE_FILE).read_text(encoding="utf-8"))
    assert rec == {"train_registration": "configs/train_registration_sens025.yaml", "run_name": "sens_u025",
                   "train_registration_sha256": hashlib.sha256(SENS_REG.read_bytes()).hexdigest()}
    assert M.run_profile(root) == {"train_registration": ROOT / rec["train_registration"], "run_name": "sens_u025"}
    M.bind_profile(root, SENS_REG)                                                      # 같은 벌 — 그대로
    assert M.run_profile(tmp_path / "plain") == {"train_registration": None, "run_name": "main"}   # 묶음 없음 = 본실험

    for args, why in (((root, MAIN_REG), "sens_registration"), ((M.OUT, SENS_REG), "sens_root_is_main"),
                      ((M.OUT / "seed1" / "x", SENS_REG), "sens_root_is_main"), ((tmp_path / "main", SENS_REG), "sens_root_name")):
        with pytest.raises(SystemExit) as exc:
            M.bind_profile(*args)
        assert exc.value.code == 2, why
    rec["run_name"] = "other"
    (root / M.PROFILE_FILE).write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.bind_profile(root, SENS_REG)                                                  # run_profile_mismatch
    rec["run_name"], rec["train_registration_sha256"] = "sens_u025", "0" * 64
    (root / M.PROFILE_FILE).write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(SystemExit):
        M.run_profile(root)                                                             # run_profile_changed


def test_민감도_뿌리의_명세는_그_벌에서_오고_본실험_뿌리를_가리키지_않으며_신원이_다르다(tmp_path, cfg):
    from vlm.train_cell import _run_identity

    root = M.bind_profile(tmp_path / "sens_u025", SENS_REG)
    sens = M.spec_for(cfg, 1, out=root)
    main = M.spec_for(cfg, 1, out=tmp_path / "main_like")                              # 묶음 없음 — 지금과 같다
    assert (sens.budget_fraction, sens.run_stamp, sens.train_registration_path) == (0.25, "sens_u025_s1", SENS_REG)
    assert (main.budget_fraction, main.run_stamp, main.train_registration_path) == (0.125, "main_s1", None)
    assert sens.expected_supervised_tokens != main.expected_supervised_tokens
    main_root = M.OUT.resolve()
    for p in (sens.train_root, sens.init_adapter_path, sens.resume_root, sens.pin_path, sens.ecc_permit_path):
        assert Path(p).resolve().is_relative_to(root) and not Path(p).resolve().is_relative_to(main_root), p
    a = _run_identity(main, "uni_central", client_idx=0, rows_digest="r", n_rows=1, prompt_sha256="p", init_digest="i")
    b = _run_identity(sens, "uni_central", client_idx=0, rows_digest="r", n_rows=1, prompt_sha256="p", init_digest="i")
    assert a != b and (a["budget_fraction"], b["budget_fraction"]) == (0.125, 0.25) and a["run_id"] != b["run_id"]
    from flwr.common.config import parse_config_args

    got = parse_config_args([M.fed_run_config(cfg, 1, out=root)])
    assert got["run-stamp"] == "sens_u025_s1" and got["uni-budget-fraction"] == "0.25"
    for k in ("project", "uni-train-root"):
        assert Path(got[k]).resolve().is_relative_to(root) and not Path(got[k]).resolve().is_relative_to(main_root)


def test_착수_고정의_common_은_명세의_등록_벌을_묶는다(tmp_path, monkeypatch):
    from vlm import pilot_vlm

    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 32)                    # 기록 등록(누적 32 시절)과 맞대는 시험 — 10-31 누적 8 변경 뒤에도 그 시절 코드값으로
    from vlm.pin import build_pin

    from tests.registration_history import historical_base, legacy_default

    legacy = legacy_default(monkeypatch, tmp_path)
    probes = {"code": lambda: {"commit": "c"}, "env": dict, "init_adapter": lambda: {"init_adapter_digest": "d"}}
    sens = build_pin(init_adapter_path=tmp_path / "x.npz", probes=probes, config_path=historical_base(),
                     registration_path=SENS_REG)["common"]
    main = build_pin(init_adapter_path=tmp_path / "x.npz", probes=probes, config_path=historical_base())["common"]
    assert sens["registration"]["budget_fraction"] == 0.25 and main["registration"]["budget_fraction"] == 0.125
    assert sens["registration_sha256"] == hashlib.sha256(SENS_REG.read_bytes()).hexdigest() != main["registration_sha256"]
    assert main["registration_sha256"] == hashlib.sha256(legacy.read_bytes()).hexdigest()


def test_구판_민감도_등록을_현재_설정에_섞으면_거부한다():
    from vlm.registration import load_registration, registration_problems
    from vlm.uni_config import load_uni_config

    bad = registration_problems(load_registration(SENS_REG)[0], uni=load_uni_config())
    assert any("프롬프트" in p for p in bad) and any("eval_gen_limit" in p for p in bad)


def test_시작_문은_명세의_등록_벌로_착수_고정을_다시_낸다(tmp_path, monkeypatch, cfg):
    import vlm.pin as P
    from vlm.train_cell import CellRejected, main_start_gate

    root = M.bind_profile(tmp_path / "sens_u025", SENS_REG)
    spec = M.spec_for(cfg, 1, out=root)
    spec.pin_path.parent.mkdir(parents=True, exist_ok=True)
    spec.pin_path.write_text("{}", encoding="utf-8")
    seen = {}

    def spy(**kw):
        seen.update(kw)
        raise P.PinMismatch(["code"], "시험이 여기서 멈춘다")

    monkeypatch.setattr(P, "build_pin", spy)
    with pytest.raises(CellRejected):
        main_start_gate(spec, init_digest="d")
    assert seen["registration_path"] == SENS_REG


def test_명령줄의_민감도_실행은_본실험_뿌리를_읽지도_쓰지도_않는다(tmp_path, monkeypatch, cfg):
    """실행기의 기본 뿌리를 지켜보는 빈 자리로 두고 `initadapter` 를 민감도 인자로 끝까지 돌린다(모델 적재 · 가드만 대역)."""
    import numpy as np

    import vlm.init_adapter as IA

    seen = {}

    def fake_build(**kw):
        seen.update(kw)
        return [np.zeros(2, np.float32)], ["k"], None

    monkeypatch.setattr(M, "_guarded", lambda spec: None)
    monkeypatch.setattr(IA, "build_initial_adapter", fake_build)
    watched, root = tmp_path / "main_u", tmp_path / "sens_u025"
    argv = ["initadapter", "--seed", "1", "--train-registration", str(SENS_REG), "--out-root", str(root)]
    assert M.main(argv, out=watched) == 0
    assert not watched.exists()
    assert Path(seen["cache_path"]).resolve().is_relative_to(root.resolve())
    rec = [json.loads(ln) for ln in (root / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["stage"] for r in rec] == ["initadapter"]
    assert M.main(argv[:3], out=watched) == 0 and watched.is_dir()                      # 인자 없음 — 기본 뿌리(회귀 0)
    assert not (watched / M.PROFILE_FILE).exists()


def test_명령줄은_섞인_민감도_인자를_거부하고_status_는_민감도_뿌리를_본다(tmp_path, capsys):
    root = tmp_path / "sens_u025"
    assert M.main(["pin", "--seed", "1", "--out-root", str(root)]) == 2
    assert "[sens_args]" in capsys.readouterr().err
    assert M.main(["ecc", "--seed", "1", "--train-registration", str(SENS_REG), "--out-root", str(root)]) == 2
    assert "[sens_cmd]" in capsys.readouterr().err
    assert M.main(["status", "--train-registration", str(SENS_REG), "--out-root", str(M.OUT)]) == 2
    assert "[sens_root_is_main]" in capsys.readouterr().err
    assert M.main(["status", "--train-registration", str(SENS_REG), "--out-root", str(root)]) == 0
    assert (root / M.PROFILE_FILE).is_file()
