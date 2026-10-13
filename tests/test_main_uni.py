"""통합형 본실험 실행기(`scripts/main_uni.py`, 12번 C3) — 모델 없이 도는 부분.

학습 단계는 모델을 올리므로 여기서는 **올리기 전**을 본다 — 저장소 설정이 10-13 결정값을 싣는 것, 그 설정으로도 공통 초기
어댑터 · 착수 고정 없이는 학습 명령이 모델을 올리기 전에 멈추는 것, 설정이 비면 거부하는 것, 설정에서 칸 설정을 만드는 것, 연합 run-config 가 flwr 의 파서와 선언된 키를 지나는 것, 고의 중단 변수의 거부, 연합 칸의 상태별 처리,
시드별 체인의 순서, 상태 격자. 모델 적재 함수는 불리면 떨어지게 막는다.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vlm.train_cell as _tc  # noqa: E402
from scripts import main_uni as M  # noqa: E402
from tests.test_uni_review_fixes import pairs_snapshot  # noqa: E402

#: 칸 학습의 실물 — 저장소 설정 시험이 되돌려 쓴다(모델 적재 함수는 `_no_model` 이 막은 채로 둔다)
_REAL_LOCAL, _REAL_CENTRAL = _tc.run_uni_local_cell, _tc.run_uni_central_cell
#: 의사결정로그 2026-10-13 13:5x「본실험 등록 조건 확정」 — N = E = R = 1 · 평가 생성 한도 L = 512. 데이터 쪽이
#: configs/base.yaml 에 적었다(wt/A 8318844). 시험은 그 결정을 고정한다 — 설정을 바꾸면 이 시험이 먼저 안다.
DECIDED_BUDGET = (1, 1, 1)
DECIDED_GEN_LIMIT = 512


@pytest.fixture(autouse=True)
def _no_model(monkeypatch):
    """모델 적재 · 칸 학습이 불리면 떨어진다 — 이 시험들은 올리기 전만 본다."""
    import vlm.init_adapter as ia
    import vlm.train_cell as tc

    def boom(*a, **k):
        raise AssertionError("모델 적재나 학습이 불렸다")

    monkeypatch.setattr(ia, "build_initial_adapter", boom)
    monkeypatch.setattr(tc, "run_uni_local_cell", boom)
    monkeypatch.setattr(tc, "run_uni_central_cell", boom)
    for k in list(__import__("os").environ):
        if k.upper() in M.FAULT_VARS:
            monkeypatch.delenv(k)


def _cfg_yaml(tmp: Path, **over) -> Path:
    import yaml

    fixed = {"snapshot_digest": "s" * 64,
             "uni_train_budget": {"num_rounds": 2, "local_epochs": 1, "total_epochs": 2},
             "uni_model": {"id": "Qwen/Qwen3.5-4B", "revision": "a" * 40},
             "uni_pairs": {"path": (tmp / "pairs.jsonl").as_posix(), "digest": pairs_snapshot(tmp)},
             "uni_prompt_sha256": None, "uni_chat_template_kwargs": {"enable_thinking": False},
             "uni_max_new_tokens": None, "uni_batch_size": 1,
             "uni_processor_kwargs": {"max_pixels": 921600},
             "uni_expected_supervised_tokens": {"C1": 10, "C2": 20, "C3": 30},
             "uni_prompt_path": "vlm/prompts/unified_v2_absorig.txt", "uni_coord_space": "ABS_ORIG"}
    fixed.update(over)
    p = tmp / "base.yaml"
    p.write_text(yaml.safe_dump({"fixed_before_main_runs": fixed, "experiment": {"seeds": [11, 12, 13]}},
                                allow_unicode=True), encoding="utf-8")
    return p


def test_저장소의_설정은_10_13_결정값을_싣는다():
    cfg = M.load_main_config()
    assert (cfg.num_rounds, cfg.local_epochs, cfg.total_epochs) == DECIDED_BUDGET
    assert cfg.total_epochs == cfg.num_rounds * cfg.local_epochs
    assert cfg.max_new_tokens == DECIDED_GEN_LIMIT


@pytest.fixture(scope="module")
def repo_pairs():
    """저장소 설정의 학습 명령은 모델을 올리기 전에 봉인 페어 계약을 해시로 맞댄다 — 이 트리에 없으면 아래 시험은 설 수 없다."""
    from vlm.uni_config import load_uni_config, pairs_snapshot_problems

    cfg = load_uni_config()
    bad = pairs_snapshot_problems(ROOT / cfg.pairs_path, cfg.pairs_digest)
    if bad:
        pytest.skip(f"봉인 페어 계약이 이 트리에 없거나 다르다(정션 · 본체 밖 사본) — {bad[0]}")


def _record_build(monkeypatch) -> list[dict]:
    """초기 어댑터 만들기를 기록만 하는 대역으로 — 모델을 올리지 않고 파일도 쓰지 않는다."""
    import numpy as np

    import vlm.init_adapter as ia

    calls: list[dict] = []
    monkeypatch.setattr(ia, "build_initial_adapter",
                        lambda **kw: calls.append(kw) or ([np.zeros(2, dtype=np.float32)], ["a"], {}))
    return calls


@pytest.mark.parametrize("cmd, reason", [("local", "init_adapter_missing"), ("central", "init_adapter_missing"),
                                         ("fed", "init_adapter_missing"), ("all", "init_adapter_missing"),
                                         ("close-ledger", "close_inputs_missing")])
def test_저장소의_설정으로도_초기_어댑터_없이는_학습_명령이_모델을_올리기_전에_멈춘다(cmd, reason, tmp_path, capsys,
                                                                       monkeypatch, repo_pairs):
    monkeypatch.chdir(ROOT)                                            # 설정의 상대 경로(페어 · 프롬프트)는 저장소 기준이다
    monkeypatch.setattr(_tc, "run_uni_local_cell", _REAL_LOCAL)        # 칸의 실물 가드까지 — 모델 적재 함수는 막힌 채
    monkeypatch.setattr(_tc, "run_uni_central_cell", _REAL_CENTRAL)
    built = _record_build(monkeypatch) if cmd == "all" else None       # all 은 initadapter 를 지난다(대역)
    argv = [cmd, "--seed", "1"] + (["--tag", "uni_local_C1"] if cmd == "close-ledger" else [])
    assert M.main(argv, out=tmp_path / "out") == 2
    assert f"[{reason}]" in capsys.readouterr().err
    assert built is None or len(built) == 1
    out = tmp_path / "out"
    assert not any(q.suffix in (".npz", ".csv") for q in out.rglob("*")) if out.exists() else True


def test_저장소의_설정으로_initadapter_는_설정의_모델_판과_시드로_공통_초기_어댑터를_만든다(tmp_path, monkeypatch, repo_pairs):
    """착수 고정보다 먼저 도는 유일한 학습 명령이다(착수 고정이 초기 어댑터의 digest 를 묶는다). 모델은 대역이다."""
    monkeypatch.chdir(ROOT)
    calls = _record_build(monkeypatch)
    assert M.main(["initadapter", "--seed", "1"], out=tmp_path / "out") == 0
    cfg = M.load_main_config()
    (kw,) = calls
    assert (kw["model_id"], kw["revision"], kw["seed"], kw["purpose"]) == \
        (cfg.model_id, cfg.model_revision, int(cfg.seeds[0]), "main")
    assert Path(kw["cache_path"]) == M.seed_dir(tmp_path / "out", 1) / "fl" / "uni_fed" / "initial.npz"


def test_설정에서_칸_설정을_만든다(tmp_path):
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    spec = M.spec_for(cfg, 2, out=tmp_path / "out", config_path=p)
    assert (spec.purpose, spec.seed_index, spec.seed_value, spec.run_stamp) == ("main", 2, 12, "main_s2")
    assert (spec.num_rounds, spec.local_epochs, spec.total_epochs) == (2, 1, 2)
    assert spec.model_revision == "a" * 40 and spec.processor_kwargs == {"max_pixels": 921600}
    assert spec.expected_supervised_tokens == {"C1": 10, "C2": 20, "C3": 30}
    assert spec.init_adapter_path == tmp_path / "out" / "seed2" / "fl" / "uni_fed" / "initial.npz"
    with pytest.raises(SystemExit):
        M.spec_for(cfg, 4, out=tmp_path / "out")          # 등록 시드표 밖


def test_저장소_설정의_프롬프트_파일과_좌표_규약을_실행기가_그대로_읽는다(tmp_path):
    """`configs/base.yaml` 에 값이 있는 두 키 — 실행기의 칸 설정과 연합 run-config 가 그 값을 싣는다.
    나머지 키는 아직 비어 있어 저장소 설정으로는 칸 설정을 만들 수 없으므로, 두 값만 저장소 설정에서 옮긴다."""
    import yaml

    from vlm.coords import COORD_SPACES
    from vlm.uni_config import load_uni_config

    real = yaml.safe_load((ROOT / "configs/base.yaml").read_text(encoding="utf-8"))["fixed_before_main_runs"]
    prompt, coord = real["uni_prompt_path"], real["uni_coord_space"]
    got = load_uni_config(ROOT / "configs/base.yaml")
    assert (got.prompt_path, got.coord_space) == (prompt, coord)        # 읽는 쪽이 저장소의 키를 읽는다
    assert (ROOT / prompt).is_file() and coord in COORD_SPACES
    p = _cfg_yaml(tmp_path, uni_prompt_path=prompt, uni_coord_space=coord)
    cfg = M.load_main_config(p)
    spec = M.spec_for(cfg, 1, out=tmp_path / "out", config_path=p)
    assert (spec.prompt_path, spec.coord_space) == (prompt, coord)
    s = M.fed_run_config(cfg, 1, out=tmp_path / "out")
    assert f'uni-prompt="{prompt}"' in s and f'uni-coord-space="{coord}"' in s


def test_연합_run_config_가_flwr_파서와_선언된_키와_받는_쪽을_지난다(tmp_path):
    from flwr.common.config import parse_config_args

    from fl.uni_run_config import client_run_cfg

    cfg = M.load_main_config(_cfg_yaml(tmp_path))
    s = M.fed_run_config(cfg, 1, out=tmp_path / "out")
    got = parse_config_args([s])
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"]
    assert set(got) <= set(declared), sorted(set(got) - set(declared))
    assert got["cell"] == "uni_fed" and got["purpose"] == "main" and got["base-seed"] == 11
    assert json.loads(got["uni-chat-template-kwargs"]) == {"enable_thinking": False}
    assert json.loads(got["uni-processor-kwargs"]) == {"max_pixels": 921600}
    uni = client_run_cfg({**{k: v for k, v in declared.items()}, **got})   # 본실험 필수 키가 모두 찼다
    assert uni["model_revision"] == "a" * 40 and uni["seed_index"] == 1
    assert uni["expected_supervised_tokens"] == {"C1": 10, "C2": 20, "C3": 30}


def test_고의_중단_변수가_보이면_설정을_읽기_전에_거부한다(tmp_path, monkeypatch, capsys):
    """설정 파일이 **없는** 경로를 준다 — 설정을 먼저 읽으면 다른 사유로 떨어진다. 이름의 대소문자는 Windows 의 환경이
    대문자로 저장하므로 여기서 시험되지 않는다(사전을 넘기는 시험이 따로 본다)."""
    monkeypatch.setenv("WELD_REHEARSAL_FAULT", "train:after_ckpt:uni_central:ep=0")
    assert M.main(["local", "--seed", "1"], out=tmp_path / "out", config_path=tmp_path / "없는_설정.yaml") == 2
    assert "[fault_env]" in capsys.readouterr().err


def _led(path: Path, *, closed: bool, run_id="uni_fed_s11_main_s1") -> None:
    from fl.atomic_log import AtomicLog

    log = AtomicLog(path, run_id=run_id, seed=11, cell="uni_fed", split_hash="s" * 64)
    log.log_round(round_idx=0, client_id="server", n_train_samples=0, metrics={"global_l2": 1.0})
    if closed:
        log.close(client_id="server", step=2)


def test_연합_칸의_상태별로_사유와_함께_멈춘다(tmp_path):
    """상태마다 **사유 코드**를 맞댄다 — `SystemExit` 만 보면 다른 까닭으로 멈춰도 지나간다(검수 14번 M-3 가).
    닫힌 산출물을 되짚어 건너뛰는 경우는 `tests/test_uni_review14.py` 가 실제 산출물로 본다."""
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    out = tmp_path / "out"
    init = M.seed_dir(out, 1) / "fl" / "uni_fed"
    init.mkdir(parents=True)
    fed = M.fed_dir(out, 1)                               # 연합 산출은 학습 루트 아래 uni_fed_s1(2판 §13-4)
    fed.mkdir(parents=True)
    calls: list[str] = []
    runner = lambda s, log: calls.append(s)             # noqa: E731

    def reason() -> str:
        with pytest.raises(SystemExit) as exc:
            M.stage_fed(cfg, 1, out=out, config_path=p, runner=runner)
        return exc.value.reason

    assert reason() == "init_adapter_missing"            # 아무것도 없다
    (init / "initial.npz").write_bytes(b"x")
    _led(fed / "atomic_log.csv", closed=False)
    assert reason() == "fed_ledger_not_fresh"            # 행은 있는데 어댑터가 없다
    (fed / "atomic_log.csv").unlink()
    (fed / "adapter_last.npz").write_bytes(b"x")
    _led(fed / "atomic_log.csv", closed=False)
    assert reason() == "ledger_open"                     # 어댑터는 있는데 열렸다
    (fed / "atomic_log.csv").unlink()
    _led(fed / "atomic_log.csv", closed=True)
    (fed / "adapter_last.meta.json").write_text(json.dumps({"train_run_id": "uni_fed_s11_main_s1"}),
                                                encoding="utf-8")
    assert reason() == "init_adapter_rejected"           # 닫혔지만 되짚을 초기 어댑터가 본실험 캐시가 아니다
    assert calls == []


def test_연합은_착수_고정과_허가가_없으면_실행기를_부르기_전에_멈춘다(tmp_path, capsys):
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    out = tmp_path / "out"
    init = M.seed_dir(out, 1) / "fl" / "uni_fed"
    init.mkdir(parents=True)
    import numpy as np

    np.savez(init / "initial.npz", a=np.zeros(2, dtype=np.float32))
    M.fed_dir(out, 1).mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        M.stage_fed(cfg, 1, out=out, config_path=p, runner=lambda s, log: pytest.fail("실행기를 불렀다"))
    assert exc.value.reason == "pin_missing"


def test_연합은_새로_돌고_끝나면_닫힌_원장을_확인한다(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_start_gate", lambda spec: {})       # 시작 문은 위 시험과 tests/test_ecc_train.py 가 본다
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    out = tmp_path / "out"
    init = M.seed_dir(out, 1) / "fl" / "uni_fed"
    init.mkdir(parents=True)
    (init / "initial.npz").write_bytes(b"x")
    fed = M.fed_dir(out, 1)
    fed.mkdir(parents=True)

    def fake_flwr(run_config: str, log: Path) -> None:
        (fed / "adapter_last.npz").write_bytes(b"a")
        _led(fed / "atomic_log.csv", closed=True)

    assert M.stage_fed(cfg, 1, out=out, config_path=p, runner=fake_flwr) == "trained"
    (fed / "adapter_last.npz").unlink()
    (fed / "atomic_log.csv").unlink()
    with pytest.raises(SystemExit):                     # 돌았는데 닫히지 않았다
        M.stage_fed(cfg, 1, out=out, config_path=p, runner=lambda s, log: None)


def test_설정과_다른_요청은_초기_어댑터를_만들기_전에_멈춘다(tmp_path, capsys):
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    (tmp_path / "o").mkdir()
    other = _cfg_yaml(tmp_path / "o", uni_train_budget={"num_rounds": 1, "local_epochs": 2, "total_epochs": 2})
    with pytest.raises(SystemExit):
        M.stage_initadapter(cfg, 1, out=tmp_path / "out", config_path=other)
    assert "main_config_mismatch" in capsys.readouterr().err


def test_시드마다_다섯_모델을_끝내고_다음_시드로_간다(tmp_path):
    cfg = M.load_main_config(_cfg_yaml(tmp_path))
    seen: list[tuple[int, str]] = []
    stages = {name: (lambda name: lambda c, n, **kw: seen.append((n, name)))(name)
              for name in ("initadapter", "local", "central", "fed")}
    M.cmd_all(cfg, [1, 2], out=tmp_path / "out", stages=stages)
    assert seen == [(n, s) for n in (1, 2) for s in ("initadapter", "local", "central", "fed")]


def test_상태_격자(tmp_path, capsys):
    out = tmp_path / "out"
    sd = M.seed_dir(out, 1)
    c1 = sd / "uni_local_C1_s1"
    c1.mkdir(parents=True)
    from fl.atomic_log import AtomicLog

    log = AtomicLog(c1 / "train_ledger.csv", run_id="r", seed=11, cell="uni_local", split_hash="s")
    log.log_round(round_idx=0, client_id="C1", n_train_samples=1, metrics={"epochs_ran": 1.0})
    grid = M.cmd_status(out, _cfg_yaml(tmp_path))
    assert grid[1]["uni_local_C1"] == "partial" and grid[1]["uni_central"] == "none"
    (c1 / "adapter_last.npz").write_bytes(b"x")
    assert M.status_grid(out, 3)[1]["uni_local_C1"] == "open"
    log.close(client_id="C1", step=1)
    assert M.status_grid(out, 3)[1]["uni_local_C1"] == "closed"


def test_close_ledger_는_로컬_중앙_칸만_받는다(tmp_path, capsys):
    assert M.main(["close-ledger", "--seed", "1", "--tag", "uni_fed"], out=tmp_path / "out",
                  config_path=_cfg_yaml(tmp_path)) == 2
    assert "tag" in capsys.readouterr().err


@pytest.mark.parametrize("case, why", [("no_contract", "계약이 없다"), ("edited", "실물과 다르다"),
                                       ("digest", "digest")])
def test_페어_스냅샷이_설정과_맞지_않으면_적재_전에_멈춘다(tmp_path, capsys, case, why):
    p = _cfg_yaml(tmp_path)
    cfg = M.load_main_config(p)
    if case == "no_contract":
        (tmp_path / "SNAPSHOT.sha256").unlink()
    elif case == "edited":
        (tmp_path / "pairs.jsonl").write_text('{"image_id": "y"}' + "\n", encoding="utf-8")   # 계약 뒤에 바뀐 페어
    else:
        import yaml

        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        doc["fixed_before_main_runs"]["uni_pairs"]["digest"] = "d" * 64
        p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
        cfg = M.load_main_config(p)
    with pytest.raises(SystemExit):
        M.stage_initadapter(cfg, 1, out=tmp_path / "out", config_path=p)
    err = capsys.readouterr().err
    assert "main_config_mismatch" in err and why in err
