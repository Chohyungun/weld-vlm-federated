"""통합형 본실험 실행기 — 시드별 다섯 모델 체인(미니스펙 12번 §1-3).

한 시드의 다섯 모델(로컬 C1 · C2 · C3 · 중앙 · 연합)을 끝낸 뒤 다음 시드로 간다. **시드 목록 · 학습량 · 모델 판 · 페어 ·
프롬프트 경로 · 좌표 규약 · 프로세서 설정 · 감독 토큰 기대값은 `configs/base.yaml` 에서 읽는다** — 상수로 두지 않는다.
설정이 비어 있으면(`null`) 어느 학습 명령도 시작하지 않는다. 목적은 언제나 `main` 이다(리허설 · 진단은 오케스트레이터의 몫이다).
**저장소 루트에서 띄운다** — 설정의 상대 경로(페어 · 프롬프트)는 저장소 루트 기준이라, 다른 폴더에서 띄우면 거부한다.

## 명령

    python -X utf8 scripts/main_uni.py status
    python -X utf8 scripts/main_uni.py initadapter --seed 1
    python -X utf8 scripts/main_uni.py local --seed 1 [--client C1]
    python -X utf8 scripts/main_uni.py central --seed 1
    python -X utf8 scripts/main_uni.py fed --seed 1
    python -X utf8 scripts/main_uni.py all [--seed 1]
    python -X utf8 scripts/main_uni.py close-ledger --seed 1 --tag uni_local_C1
    python -X utf8 scripts/main_uni.py pin --seed 1
    python -X utf8 scripts/main_uni.py ecc-prepare --seed 1 --ecc-root outputs/early_check/ecc-YYYYMMDDTHHMMSS --target-list <대상 목록>
    python -X utf8 scripts/main_uni.py gate-check --seed 1
    python -X utf8 scripts/main_uni.py ecc --seed 1 --ecc-root outputs/early_check/ecc-… --spent-hours H --next-hours H \
        [--max-hours H] [--max-won W] [--resume-step S]
    python -X utf8 scripts/main_uni.py ecc-accept --seed 1 --ecc-root outputs/early_check/ecc-… --permit <평가 쪽 통과 파일>

- **착수 고정 · 착수 허가(11번 §4 · §10, 평가 쪽 10번 §14-3)** — 칸을 열기 전에 `pin` 이 그 시드의 `PIN.json` 을 쓴다(코드 · 환경 ·
  공통 설정 · 초기 어댑터). 로컬 · 중앙 · 연합은 **시작 · 재개마다** 그것을 지금 값과 맞대고, 조기 좌표 점검의 통과 파일
  (`ecc_permit.json` — 평가 쪽 `early_coord_check.json` 을 그 자리에 둔다)을 평가 쪽 `verify_permit` 으로 대조한다. 어느 하나라도
  없거나 어긋나면 시작하지 않는다.
- **조기 좌표 점검** — `ecc-prepare` 가 그 시드의 공통 초기 어댑터를 점검 루트로 옮기며 착수 고정의 digest 와 맞대고(규칙 파일의
  허용 시도 수를 넘으면 새 루트를 만들지 않는다), `ecc` 가 본실험 중앙 칸 자체를 다음 점검 step 까지 학습해 멈춘다(종료 4).
  step 목록 · 상한 · 단가(`limits`)는 평가 쪽 규칙 파일(`configs/early_coord_check.yaml`)을 평가 쪽 `load_rules` ·
  `effective_limits` 로 읽어 쓴다. 실행 인자 `--max-hours` · `--max-won` 은 규칙의 상한을 좁히기만 하고, 단가는 실행 인자로 받지
  않는다(금액 = 시간 × 단가 × 계수). 평가 쪽 결정이
  `이어 간다` 면 `--resume-step` 으로 다음 step 까지 잇는다. 통과면 `ecc-accept` 가 평가 쪽 통과 파일을 평가 쪽 검증으로 받아
  시드 폴더에 사본과 착수 허가의 키 기록(`ECC_PERMIT_KEY.json`)을 쓴다 — 본실험 칸은 그 키를 통과 파일 밖에서 읽는다.
  점검 단계의 진행 기록은 점검 루트의 `progress.jsonl` 에 남긴다(본실험 진행 기록과 섞지 않는다).
  `ecc-prepare` 는 평가 쪽 대상 목록(`--target-list`, 규칙 파일의 `target_list` 와 sha · n 이 같아야 한다)으로 노출 원장에
  `planned` 줄(run_id = 점검 루트 이름 · purpose early_coord_check · list_kind gen)을 덧붙이고, 그 시드의 `PIN.json` 을 점검
  루트로 복사한다(같은 바이트 — 점검 실행기는 루트 안의 사본만 받는다). `gate-check --seed n` 은 본 칸의 시작 문
  (`main_start_gate`)만 돌려 착수 고정 · 허가 이식 묶음이 그 인스턴스에서 서는지 본다 — 모델을 올리지 않는다.

- **멱등** — 칸마다 어댑터 파일이 있으면 meta 의 신원을 먼저 맞댄다(`vlm/train_cell.py`). 같고 원장이 닫혔으면 건너뛰고,
  열렸으면 `close-ledger` 를 안내하며 멈추고, 다르면 멈춘다. 연합도 같은 재료(신원 · 신원 해시 · 초기 어댑터 digest ·
  어댑터 파일 sha · 원장 규칙 ①~④)로 되짚는다(`fl.uni_fed.check_fed_prior`). 연합은 같은 원장에 이어 쓰지 않는다 —
  원장에 행이 있는데 어댑터가 없으면 멈춘다. 스탬프가 시드마다 고정이라 이 실행기로 새 스탬프를 줄 수 없다 — 그 시드의
  `uni_fed_s<n>/` 를 지우지 말고 옆으로 옮긴 뒤 R 라운드를 처음부터 다시 돈다. 연합의 `close-ledger` 는 없다(재개가 없다).
- **거부는 모델 적재 전이다.** 설정 · 페어 스냅샷 · 출력 경로 · 고의 중단 변수의 검사가 모두 앞선다.
- export(C8)는 이 실행기에 없다.

산출: 저장소 루트의 `outputs/main_u/seed<n>/` — 칸 폴더 `<tag>_s<n>/`(연합도 `uni_fed_s<n>/` — 리허설 2판 §13-4), 공통 초기 어댑터 `fl/uni_fed/initial.npz`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlm.fault import FAULT_VAR, NONCE_VAR

#: 저장소 루트 기준이다 — 작업 폴더에 기대지 않는다.
OUT = REPO_ROOT / "outputs" / "main_u"
TAGS = ("uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_central", "uni_fed")
FAULT_VARS = (FAULT_VAR, NONCE_VAR)
CLIENT_TAGS = ("C1", "C2", "C3")
#: 시드 폴더 안의 착수 고정 · 착수 허가(조기 좌표 점검의 통과 파일)의 자리.
PIN_FILE = "PIN.json"
PERMIT_FILE = "ecc_permit.json"


class MainRefused(SystemExit):
    """시작하지 않는다. 종료 코드 2 · 사유를 stderr 에 한 줄로 낸다."""

    def __init__(self, code: str, message: str):
        self.reason = code
        print(f"[main_uni] 거부 [{code}] {message}", file=sys.stderr, flush=True)
        super().__init__(2)


# ---------------------------------------------------------------- 설정과 가드
def guard_env(env: dict[str, str] | None = None) -> None:
    """본실험은 고의 중단 변수가 하나라도 보이면 무엇보다 먼저 거부한다(리허설 2판 §1-4 의 순서 2).
    Windows 의 환경 이름은 대소문자를 가리지 않으므로 대문자로 맞춰 본다."""
    names = {k.upper() for k in (os.environ if env is None else env)}
    hit = [v for v in FAULT_VARS if v in names]
    if hit:
        raise MainRefused("fault_env", f"본실험은 고의 중단 변수를 받지 않는다: {hit}")


def guard_cwd() -> None:
    """저장소 루트에서만 돈다 — 설정의 상대 경로(페어 · 프롬프트)를 여는 곳이 작업 폴더라, 다른 폴더에서 띄우면
    같은 이름의 다른 파일을 읽는다."""
    if Path.cwd().resolve() != REPO_ROOT.resolve():
        raise MainRefused("cwd", f"작업 폴더 {Path.cwd()} 가 저장소 루트 {REPO_ROOT} 가 아니다 — 루트에서 띄운다")


def load_main_config(path: Path | None = None):
    """설정을 읽고 학습에 쓸 수 있는지 본다. 비어 있으면 이름을 모아 거부한다."""
    from vlm.uni_config import UniConfigIncomplete, load_uni_config

    try:
        return load_uni_config(path).require("train")
    except UniConfigIncomplete as exc:
        raise MainRefused("config_incomplete", str(exc)) from None


def seed_dir(out: Path, n: int) -> Path:
    return Path(out) / f"seed{n}"


def fed_dir(out: Path, n: int) -> Path:
    """연합 칸의 출력 — 로컬 · 중앙과 같은 꼴로 학습 루트(시드 폴더) 아래 `uni_fed_s<n>/`(리허설 2판 §13-4). 서버는 `uni-train-root` 로 받는다."""
    return seed_dir(out, n) / f"uni_fed_s{n}"


def spec_for(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None):
    """시드 번호 `n`(1 부터)의 로컬 · 중앙 칸 설정. 값은 모두 설정에서 온다."""
    from vlm.train_cell import UniRunSpec

    if not 1 <= n <= len(cfg.seeds):
        raise MainRefused("seed_index", f"시드 번호 {n} 가 등록 시드표 1..{len(cfg.seeds)} 밖이다")
    sd = seed_dir(out, n)
    return UniRunSpec(
        purpose="main", run_stamp=f"main_s{n}", seed_index=n, seed_value=int(cfg.seeds[n - 1]),
        snapshot_digest=str(cfg.snapshot_digest), model_id=str(cfg.model_id), model_revision=cfg.model_revision,
        pairs_path=str(cfg.pairs_path), pairs_digest=str(cfg.pairs_digest),
        chat_template_kwargs=dict(cfg.chat_template_kwargs), num_rounds=int(cfg.num_rounds),
        local_epochs=int(cfg.local_epochs), total_epochs=int(cfg.total_epochs),
        train_root=sd, init_adapter_path=sd / "fl" / "uni_fed" / "initial.npz", resume_root=sd / "_resume",
        prompt_path=str(cfg.prompt_path), coord_space=str(cfg.coord_space),
        processor_kwargs=dict(cfg.processor_kwargs),
        expected_supervised_tokens=dict(cfg.expected_supervised_tokens), config_path=config_path,
        pin_path=sd / PIN_FILE, ecc_permit_path=sd / PERMIT_FILE,
    )


def _progress(out: Path, n: int, stage: str, **kw) -> None:
    p = Path(out) / "progress.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"seed": n, "stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **kw},
                            ensure_ascii=False) + "\n")


def _guarded(spec) -> None:
    """모델 적재 전 가드 — 칸 모듈의 것을 그대로 부른다. 거부는 사유 코드와 함께 종료 2 다."""
    from vlm.train_cell import CellRejected, check_before_load

    try:
        check_before_load(spec)
    except CellRejected as exc:
        raise MainRefused(exc.code, str(exc)) from None


# ---------------------------------------------------------------- 단계
def stage_initadapter(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None) -> str:
    """공통 초기 어댑터. 다섯 칸이 같은 파일에서 출발한다. 캐시가 있으면 proof 규칙(본실험)으로 읽기만 한다."""
    from vlm.init_adapter import build_initial_adapter, init_adapter_digest

    spec = spec_for(cfg, n, out=out, config_path=config_path)
    _guarded(spec)
    arrays, keys, _ = build_initial_adapter(model_id=spec.model_id, seed=spec.seed_value,
                                            cache_path=spec.init_adapter_path, revision=spec.model_revision,
                                            purpose="main")
    digest = init_adapter_digest(arrays)
    _progress(out, n, "initadapter", init_adapter_digest=digest, n_tensors=len(keys))
    return digest


def _cell_call(fn: Callable, stage: str, n: int, out: Path) -> Any:
    from vlm.train_cell import CellRejected

    try:
        res = fn()
    except CellRejected as exc:
        raise MainRefused(exc.code, str(exc)) from None
    _progress(out, n, stage, status={k: v.status for k, v in res.items()} if isinstance(res, dict) else res.status)
    return res


def stage_local(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None, client: str | None = None):
    """로컬 칸 — 참여자를 주면 그 칸 하나만(인스턴스마다 칸 하나, 26번 확정판), 없으면 세 참여자를 차례로."""
    from vlm.train_cell import CLIENT_ORDER, run_uni_local_cell

    spec = spec_for(cfg, n, out=out, config_path=config_path)
    clients = CLIENT_ORDER if client is None else (client,)
    return _cell_call(lambda: run_uni_local_cell(clients, spec=spec), "local", n, out)


def stage_central(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None):
    from vlm.train_cell import run_uni_central_cell

    spec = spec_for(cfg, n, out=out, config_path=config_path)
    return _cell_call(lambda: run_uni_central_cell(spec=spec), "central", n, out)


def _toml_str(s: str) -> str:
    """run-config 의 문자열 값. JSON 처럼 큰따옴표가 든 값은 작은따옴표(리터럴)로 싼다."""
    if "'" in s:
        raise ValueError(f"작은따옴표가 든 값은 run-config 로 넘기지 않는다: {s!r}")
    if '"' in s or "\\" in s:
        return f"'{s}'"
    return f'"{s}"'


def fed_run_config(cfg: Any, n: int, *, out: Path = OUT) -> str:
    """`flwr run --run-config` 문자열. 키는 모두 `pyproject.toml` 에 선언된 것이다(시험이 맞댄다)."""
    from vlm.pilot_vlm import canonical_json

    spec = spec_for(cfg, n, out=out)
    sd = seed_dir(out, n)
    items: dict[str, Any] = {
        "cell": _toml_str("uni_fed"),
        "num-server-rounds": spec.num_rounds, "local-epochs": spec.local_epochs, "total-epochs": spec.total_epochs,
        "num-clients": len(CLIENT_TAGS), "base-seed": spec.seed_value,
        "run-stamp": _toml_str(spec.run_stamp), "split-hash": _toml_str(spec.snapshot_digest),
        "model": _toml_str(spec.model_id), "project": _toml_str(Path(os.path.abspath(sd)).as_posix()),
        "client-tags": _toml_str(",".join(CLIENT_TAGS)), "resume-root": _toml_str(""),
        "uni-model": _toml_str(spec.model_id), "uni-model-revision": _toml_str(str(spec.model_revision)),
        "uni-pairs": _toml_str(Path(spec.pairs_path).as_posix()), "uni-pairs-digest": _toml_str(spec.pairs_digest),
        "uni-prompt": _toml_str(Path(spec.prompt_path).as_posix()),
        "uni-chat-template-kwargs": _toml_str(canonical_json(dict(spec.chat_template_kwargs))),
        "uni-coord-space": _toml_str(spec.coord_space), "uni-seed-index": _toml_str(str(n)),
        "uni-processor-kwargs": _toml_str(canonical_json(dict(spec.processor_kwargs))),
        "uni-expected-supervised-tokens": _toml_str(canonical_json(dict(spec.expected_supervised_tokens))),
        "purpose": _toml_str("main"), "plan": _toml_str(""), "rehearsal-root": _toml_str(""),
        "uni-train-root": _toml_str(Path(os.path.abspath(sd)).as_posix()),
    }
    return " ".join(f"{k}={v}" for k, v in items.items())


def fed_state(out_dir: Path) -> str:
    """연합 칸의 상태 — `closed` · `open`(어댑터는 있는데 끝 행 없음) · `partial`(원장에 행이 있는데 어댑터 없음) · `none`."""
    from fl.atomic_log import END_METRIC, last_data_row

    npz, led = out_dir / "adapter_last.npz", out_dir / "atomic_log.csv"
    last = last_data_row(led) if led.exists() else None
    if npz.exists():
        return "closed" if last is not None and last["metric_name"] == END_METRIC else "open"
    return "partial" if last is not None else "none"


def stage_fed(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None,
              runner: Callable[[str, Path], None] | None = None) -> str:
    """통합형 연합. 서버(`fl/server_app.py`)가 본실험 설정을 다시 맞대고 초기 어댑터를 같은 파일에서 읽는다."""
    spec = spec_for(cfg, n, out=out, config_path=config_path)
    _guarded(spec)
    out_dir = fed_dir(out, n)
    state = fed_state(out_dir)
    if state == "closed":
        _check_fed_prior(spec, out_dir)
        _progress(out, n, "fed", status="skipped")
        return "skipped"
    if state == "open":
        raise MainRefused("ledger_open", f"연합 어댑터는 있는데 원장이 닫히지 않았다: {out_dir} — 원인을 먼저 본다")
    if state == "partial":
        raise MainRefused("fed_ledger_not_fresh",
                          f"연합 원장에 행이 있는데 어댑터가 없다: {out_dir} — 이 폴더를 지우지 말고 옆으로 옮긴 뒤 "
                          "R 라운드를 처음부터 다시 돈다(스탬프는 시드마다 고정이고 연합은 재개가 없다)")
    if not spec.init_adapter_path.exists():
        raise MainRefused("init_adapter_missing", f"공통 초기 어댑터가 없다: {spec.init_adapter_path} — initadapter 먼저")
    _start_gate(spec)
    if runner is None:
        from scripts.main_det import _flwr_run as runner          # 검출과 같은 flwr 실행 함수(2판 §1-3 의 연합 행)
    runner(fed_run_config(cfg, n, out=out), seed_dir(out, n) / "flwr_uni_fed.log")
    if fed_state(out_dir) != "closed":
        raise MainRefused("fed_not_closed", f"연합이 끝났는데 원장이 닫히지 않았다: {out_dir}")
    _progress(out, n, "fed", status="trained")
    return "trained"


def fed_expected_identity(spec: Any, init_digest: str) -> dict[str, Any]:
    """이 시드의 연합 칸 meta 가 가져야 할 `identity` — **설정에서** 낸다. 서버는 같은 함수를 실행 설정에서 부른다."""
    from fl.atomic_log import new_run_id
    from fl.uni_fed import fed_identity
    from vlm.pilot_vlm import load_prompt

    _, prompt_sha = load_prompt(spec.prompt_path)
    return fed_identity(
        run_id=new_run_id("uni_fed", spec.seed_value, spec.run_stamp), seed_index=spec.seed_index,
        seed_value=spec.seed_value, purpose="main", num_rounds=spec.num_rounds, local_epochs=spec.local_epochs,
        total_epochs=spec.total_epochs, model_id=spec.model_id, model_revision=spec.model_revision,
        pairs_digest=spec.pairs_digest, prompt_sha256=prompt_sha, chat_template_kwargs=dict(spec.chat_template_kwargs),
        coord_space=spec.coord_space, init_digest=init_digest, snapshot_digest=spec.snapshot_digest, plan_sha256=None,
        client_tags=CLIENT_TAGS, processor_kwargs=spec.processor_kwargs,
        expected_supervised_tokens=spec.expected_supervised_tokens)


def _start_gate(spec: Any) -> dict[str, str]:
    """연합의 시작 문 — 로컬 · 중앙과 같은 함수(`vlm.train_cell.main_start_gate`)로 착수 고정과 착수 허가를 맞댄다."""
    from vlm.pin import init_probe
    from vlm.train_cell import CellRejected, main_start_gate

    try:
        return main_start_gate(spec, init_digest=init_probe(spec.init_adapter_path)["init_adapter_digest"])
    except CellRejected as exc:
        raise MainRefused(exc.code, str(exc)) from None


def stage_pin(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None) -> str:
    """그 시드의 착수 고정을 쓴다 — 공통 초기 어댑터가 있어야 하고, 코드가 커밋과 다르면 쓰지 않는다. 덮지 않는다."""
    from vlm.pin import PinMismatch, build_pin, write_pin

    spec = spec_for(cfg, n, out=out, config_path=config_path)
    _guarded(spec)
    if not spec.init_adapter_path.exists():
        raise MainRefused("init_adapter_missing", f"공통 초기 어댑터가 없다: {spec.init_adapter_path} — initadapter 먼저")
    try:
        sha = write_pin(spec.pin_path, build_pin(init_adapter_path=spec.init_adapter_path, config_path=config_path))
    except (PinMismatch, FileExistsError) as exc:
        raise MainRefused("pin_write", str(exc)) from None
    _progress(out, n, "pin", pin_sha256=sha)
    return sha


def spec_for_ecc(cfg: Any, n: int, root: Path, *, config_path: Path | None = None,
                 non_main_parent: Path | None = None):
    """조기 좌표 점검의 칸 설정 — 본실험 중앙과 같은 값, 출력 · 초기 어댑터만 점검 루트 아래다(11번 §1 · §3-1)."""
    from dataclasses import replace

    from vlm.ecc import PURPOSE

    base = spec_for(cfg, n, config_path=config_path)
    root = Path(root)
    return replace(base, purpose=PURPOSE, run_stamp=f"ecc_s{n}_{root.name}", train_root=root / "train",
                   init_adapter_path=root / "init" / "initial.npz", resume_root=None, run_root=root,
                   non_main_parent=non_main_parent, pin_path=None, ecc_permit_path=None)


def stage_ecc_prepare(cfg: Any, n: int, root: Path, *, target_list: Path, out: Path = OUT, config_path: Path | None = None,
                      non_main_parent: Path | None = None, rule_path: Path | None = None,
                      exposure_ledger: Path | None = None) -> str:
    """점검 루트를 만들고 그 시드의 공통 초기 어댑터(· proof)와 착수 고정을 옮긴다. 옮긴 digest · sha 가 착수 고정의 것과 같아야 한다.

    평가 쪽 대상 목록(`target_list` — 규칙 파일의 `target_list.sha256` · `n` 과 같아야 한다)을 노출 원장에 `planned` 줄로 적는다 —
    점검 export 는 그 줄이 있어야 시작한다(`vlm/export_preflight` 8, 리허설 2판 §5-4). `exposure_ledger` 는 시험의 이음새다.
    """
    import shutil

    from evaluation.eval_list import validate_eval_list
    from vlm.ecc import PREPARE_FILE, PURPOSE, EccRefused, attempts_started, load_check_rules
    from vlm.exposure import append_row, default_ledger, write_list_copy
    from vlm.pin import init_probe, read_pin
    from vlm.run_root import run_root_problems

    main_spec = spec_for(cfg, n, out=out, config_path=config_path)
    root = Path(root)
    bad = run_root_problems(PURPOSE, root, [root], parent=non_main_parent)
    if bad:
        raise MainRefused("run_root", " · ".join(bad))
    if root.exists():
        raise MainRefused("ecc_root_exists", f"점검 루트가 이미 있다 — 새 점검 실행은 새 루트로: {root}")
    if not Path(main_spec.pin_path).is_file():
        raise MainRefused("pin_missing", f"그 시드의 착수 고정이 없다 — pin 먼저: {main_spec.pin_path}")
    pin, pin_sha = read_pin(main_spec.pin_path)
    try:
        rules = load_check_rules(rule_path)
    except EccRefused as exc:
        raise MainRefused(exc.code, str(exc)) from None
    used = attempts_started(root.parent, pin_sha)
    if used >= rules["max_attempts"]:
        raise MainRefused("ecc_attempts_exhausted",
                          f"이 시드의 점검 시도 {used} 번이 허용 시도 수 {rules['max_attempts']} 에 닿았다 — 새 루트를 만들지 않는다")
    tl = Path(target_list)
    if not tl.is_file():
        raise MainRefused("ecc_target_list", f"평가 쪽 대상 목록이 없다: {tl.name}")
    raw = tl.read_bytes()
    try:
        el = validate_eval_list(raw, where=tl.name)
    except Exception as exc:  # noqa: BLE001 — 목록 꼴의 거부 사유를 그대로 싣는다(평가 쪽 꼴)
        raise MainRefused("ecc_target_list", f"대상 목록의 꼴이 아니다: {exc}") from None
    want_sha, want_n = rules["rules"].target_sha256, int(rules["rules"].target_n)
    if el.file_sha256 != want_sha or len(el) != want_n:
        raise MainRefused("ecc_target_list", f"대상 목록(sha {el.file_sha256[:12]}… · {len(el)} 장)이 규칙 파일의 "
                                             f"target_list(sha {want_sha[:12]}… · {want_n} 장)와 다르다")
    src = Path(main_spec.init_adapter_path)
    want = pin["init_adapter"]["init_adapter_digest"]
    if init_probe(src)["init_adapter_digest"] != want:
        raise MainRefused("pin_mismatch", "그 시드의 공통 초기 어댑터가 착수 고정의 것과 다르다")
    ecc = spec_for_ecc(cfg, n, root, config_path=config_path, non_main_parent=non_main_parent)
    dst = Path(ecc.init_adapter_path)
    dst.parent.mkdir(parents=True)
    shutil.copyfile(src, dst)
    if src.with_suffix(".proof.json").is_file():
        shutil.copyfile(src.with_suffix(".proof.json"), dst.with_suffix(".proof.json"))
    got = init_probe(dst)["init_adapter_digest"]
    if got != want:
        raise MainRefused("pin_mismatch", "옮긴 초기 어댑터의 digest 가 착수 고정과 다르다")
    pin_copy = root / PIN_FILE                                       # §119 I-06 — 점검 실행기는 루트 안의 사본만 받는다
    shutil.copyfile(main_spec.pin_path, pin_copy)
    if read_pin(pin_copy)[1] != pin_sha:
        raise MainRefused("pin_mismatch", "옮긴 착수 고정의 sha256 이 원본과 다르다")
    ledger = Path(exposure_ledger) if exposure_ledger is not None else default_ledger()
    at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        copy = write_list_copy(ledger, root.name, "gen", raw)
        append_row(ledger, {"event": "planned", "run_id": root.name, "purpose": PURPOSE, "tag": None, "seed_index": n,
                            "mode": None, "list_kind": "gen", "list_path": copy.as_posix(),
                            "list_file_sha256": el.file_sha256, "list_set_sha256": el.set_sha256, "n": len(el),
                            "snapshot_digest": main_spec.snapshot_digest, "commit": pin["code"]["commit"], "at": at})
    except Exception as exc:  # noqa: BLE001 — 노출 원장의 거부 사유(ExposureError 등)를 그대로 싣는다
        raise MainRefused("exposure_ledger", f"노출 원장에 planned 줄을 적지 못했다: {exc}") from None
    rec = {"seed_index": n, "pin_sha256": pin_sha, "init_adapter_digest": got, "rules_sha256": rules["rules_sha256"],
           "steps": list(rules["steps"]), "target_list_sha256": el.file_sha256, "target_n": len(el),
           "exposure_ledger": ledger.as_posix(), "attempt": used + 1, "init_source": src.name, "time": at}
    (root / PREPARE_FILE).write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    _progress(root, n, "ecc_prepare", ecc_root=root.name, pin_sha256=pin_sha)
    return got


def stage_ecc(cfg: Any, n: int, root: Path, *, spent_hours: float, next_hours: float,
              max_hours: float | None = None, max_won: float | None = None, resume_step: int | None = None,
              out: Path = OUT, config_path: Path | None = None, rule_path: Path | None = None,
              non_main_parent: Path | None = None, **seams: Any) -> dict[str, Any]:
    """점검 학습을 다음 점검 step 까지 — 평가 쪽 규칙의 step 목록 · 상한과 단가(`limits` — 평가 쪽 `effective_limits`, 실행 인자는
    좁히기만)로. 점검 step 도달이면 `exit_code` 4. 단가를 코드 기본값 · 실행 인자로 두지 않는다(20b 의 I-2).

    규칙의 지문이 루트를 준비할 때(`ecc-prepare`)와 다르면 시작 · 재개하지 않는다 — 한 시도 안에서 규칙이 바뀌지 않는다.
    """
    from vlm.ecc import PREPARE_FILE, EccRefused, effective_limits, load_check_rules, run_ecc_train
    from vlm.pin import read_pin
    from vlm.train_cell import CellRejected

    main_spec = spec_for(cfg, n, out=out, config_path=config_path)
    spec = spec_for_ecc(cfg, n, root, config_path=config_path, non_main_parent=non_main_parent)
    prep = Path(root) / PREPARE_FILE
    if not prep.is_file():
        raise MainRefused("ecc_not_prepared", f"점검 루트의 준비 기록이 없다 — ecc-prepare 먼저: {Path(root).name}")
    rec = json.loads(prep.read_text(encoding="utf-8"))
    pin_copy = Path(root) / PIN_FILE
    if not pin_copy.is_file() or read_pin(pin_copy)[1] != read_pin(main_spec.pin_path)[1]:
        raise MainRefused("pin_mismatch", "점검 루트의 착수 고정 사본이 없거나 시드 폴더의 원본과 다르다 — ecc-prepare 가 옮긴다")
    try:
        rules = load_check_rules(rule_path)
        if rec.get("rules_sha256") != rules["rules_sha256"]:
            raise MainRefused("ecc_rules_changed", "규칙 파일이 이 루트를 준비할 때와 다르다 — 새 시도는 새 루트로")
        lim = effective_limits(rules["rules"], max_hours=max_hours, max_won=max_won)
        budget = {"spent_hours": spent_hours, "next_hours": next_hours, "max_hours": lim["max_hours"],
                  "max_won": lim["max_cost_krw"], "won_per_hour": lim["cost_krw_of_hour"],
                  "price_detail": {k: lim[k] for k in ("price_krw_per_hour", "safety_factor", "price_source")}}
        res = run_ecc_train(spec, steps=rules["steps"], registered_steps=tuple(int(s) for s in rec.get("steps") or ()),
                            pin_path=pin_copy, resume_step=resume_step, budget=budget, **seams)
    except (EccRefused, CellRejected) as exc:
        raise MainRefused(exc.code, str(exc)) from None
    _progress(Path(root), n, "ecc", ecc_root=Path(root).name, exit_code=res["exit_code"], step=res.get("step"))
    return res


def stage_ecc_accept(cfg: Any, n: int, root: Path, permit: Path, *, out: Path = OUT, config_path: Path | None = None,
                     rule_path: Path | None = None, non_main_parent: Path | None = None) -> dict[str, Any]:
    """점검 통과 뒤 — 평가 쪽 통과 파일을 평가 쪽 `verify_permit` 으로 받아 시드 폴더에 사본 · 착수 허가의 키 기록을 쓴다.

    대조값(평가 쪽 3판의 넷)은 **점검 실행 쪽에서** 낸다: 출발 어댑터(점검 루트의 초기 어댑터 digest), 착수 고정(시드의
    `PIN.json`), 점검 실행 신원(`cell_run_id(점검 칸, 중앙)` — step meta 의 신원과 같아야 한다), 시드 번호. 시도 기록의 시간은
    점검 원장의 누적 처리 시간, 금액은 그 시간 × 규칙의 단가 × 계수(평가 쪽 `effective_limits`)다.
    """
    from vlm.ecc import (
        PREPARE_FILE,
        EccRefused,
        accept_permit,
        cum_wall_s,
        effective_limits,
        ledger_rows,
        load_check_rules,
    )
    from vlm.pin import init_probe, read_pin
    from vlm.train_cell import CENTRAL_TAG, cell_dir, cell_run_id

    root = Path(root)
    main_spec = spec_for(cfg, n, out=out, config_path=config_path)
    ecc = spec_for_ecc(cfg, n, root, config_path=config_path, non_main_parent=non_main_parent)
    prep = root / PREPARE_FILE
    if not prep.is_file():
        raise MainRefused("ecc_not_prepared", f"점검 루트의 준비 기록이 없다: {root.name}")
    if not Path(permit).is_file():
        raise MainRefused("ecc_permit_missing", f"평가 쪽 통과 파일이 없다: {Path(permit).name}")
    rec = json.loads(prep.read_text(encoding="utf-8"))
    _, pin_sha = read_pin(main_spec.pin_path)
    if rec.get("pin_sha256") != pin_sha or rec.get("seed_index") != n:
        raise MainRefused("pin_mismatch", "점검 루트의 준비 기록이 이 시드의 착수 고정과 다르다")
    cdir = cell_dir(ecc, CENTRAL_TAG)
    metas = [json.loads(m.read_text(encoding="utf-8")) for m in sorted(cdir.glob("step*.meta.json"))]
    if not metas:
        raise MainRefused("ecc_no_steps", "점검 실행에 step 기록이 없다 — 통과를 받을 수 없다")
    run_id = cell_run_id(ecc, CENTRAL_TAG)
    if {m["identity"].get("run_id") for m in metas} != {run_id}:
        raise MainRefused("ecc_identity", "점검 step 들의 실행 신원이 이 점검 루트의 것과 다르다")
    hours = cum_wall_s(cdir) / 3600                                 # 학습 루프의 누적 처리 시간(적재 · export 생성 제외)
    budget_rows = [json.loads(r["detail"]) for r in ledger_rows(cdir) if r.get("event") in ("budget_ok", "ecc_budget_exceeded")]
    spent = budget_rows[-1].get("spent_hours") if budget_rows else None   # 운영자가 마지막에 준 인스턴스 시간
    try:
        rules = load_check_rules(rule_path)
        if rec.get("rules_sha256") != rules["rules_sha256"]:
            raise MainRefused("ecc_rules_changed", "규칙 파일이 이 루트를 준비할 때와 다르다")
        rate = effective_limits(rules["rules"])["cost_krw_of_hour"]
        expected = {"start_adapter_digest": init_probe(ecc.init_adapter_path)["init_adapter_digest"],
                    "pin_sha256": pin_sha, "check_run_id": run_id, "seed": n}
        key = accept_permit(Path(permit), rules=rules["rules"], expected=expected,
                            seed_dir=Path(main_spec.pin_path).parent, permit_name=Path(main_spec.ecc_permit_path).name,
                            attempts_parent=root.parent,
                            attempt_row={"run_id": run_id, "pin_sha256": pin_sha, "last_step": max(m["step"] for m in metas),
                                         "hours": round(hours, 4), "won": round(hours * float(rate)),
                                         "train_loop_hours": round(hours, 4), "spent_hours_last_budget": spent,
                                         "won_basis": "train_loop_hours × cost_krw_of_hour"},
                            extra={"attempt": rec.get("attempt"), "ecc_root": root.name})
    except EccRefused as exc:
        raise MainRefused(exc.code, str(exc)) from None
    _progress(root, n, "ecc_accept", ecc_root=root.name, permit_sha256=key["permit_sha256"])
    return key


def stage_gate_check(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None) -> dict[str, Any]:
    """본 칸의 시작 문만 — 착수 고정을 지금 값과 맞대고 허가 이식 묶음(통과 파일 · 키 기록 · 네 검사 산출 파일)을 대조한다.
    모델을 올리지 않는다. 둘째 ~ 다섯째 인스턴스에서 칸을 켜기 전에 돌린다(26번 §6)."""
    spec = spec_for(cfg, n, out=out, config_path=config_path)
    if not spec.init_adapter_path.exists():
        raise MainRefused("init_adapter_missing", f"공통 초기 어댑터가 없다: {spec.init_adapter_path}")
    gate = _start_gate(spec)
    _progress(out, n, "gate_check", pin_sha256=gate["pin_sha256"], ecc_permit_sha256=gate["ecc_permit_sha256"])
    print(json.dumps({"seed": n, **{k: gate[k] for k in ("pin_sha256", "ecc_permit_sha256")}}, ensure_ascii=False))
    return gate


def _check_fed_prior(spec: Any, out_dir: Path) -> None:
    """닫힌 연합 산출물을 건너뛰기 전에 **로컬 · 중앙과 같은 재료로** 되짚는다(검수 14번 I-1). 모델을 올리지 않는다."""
    from fl.atomic_log import new_run_id
    from fl.uni_fed import UniFedRejected, check_fed_prior
    from vlm.init_adapter import InitCacheRejected, init_adapter_digest, read_initial_cache

    if not spec.init_adapter_path.exists():
        raise MainRefused("init_adapter_missing", f"공통 초기 어댑터가 없다: {spec.init_adapter_path} — 앞 산출물을 되짚을 수 없다")
    try:
        arrays, _ = read_initial_cache(spec.init_adapter_path, model_id=spec.model_id, seed=spec.seed_value,
                                       revision=spec.model_revision, purpose="main")
    except (InitCacheRejected, RuntimeError, ValueError) as exc:
        raise MainRefused("init_adapter_rejected", str(exc)) from None
    try:
        check_fed_prior(out_dir, identity=fed_expected_identity(spec, init_adapter_digest(arrays)),
                        run_id=new_run_id("uni_fed", spec.seed_value, spec.run_stamp),
                        num_rounds=spec.num_rounds, num_clients=len(CLIENT_TAGS))
    except UniFedRejected as exc:
        raise MainRefused(exc.code, str(exc)) from None


def cmd_all(cfg: Any, seeds: list[int], *, out: Path = OUT, config_path: Path | None = None,
            stages: dict[str, Callable] | None = None) -> None:
    """시드마다 initadapter → local → central → fed. 한 시드의 다섯 모델을 끝낸 뒤 다음 시드로 간다."""
    st = stages or {"initadapter": stage_initadapter, "local": stage_local,
                    "central": stage_central, "fed": stage_fed}
    for n in seeds:
        for name in ("initadapter", "local", "central", "fed"):
            print(f"\n########## 시드 {n} · {name} ##########", flush=True)
            st[name](cfg, n, out=out, config_path=config_path)
        _progress(out, n, "chain_done")


def cell_state(d: Path) -> str:
    """로컬 · 중앙 칸의 상태 — `closed` · `open` · `partial` · `none`."""
    from fl.atomic_log import END_METRIC, last_data_row

    npz, led = d / "adapter_last.npz", d / "train_ledger.csv"
    last = last_data_row(led) if led.exists() else None
    if npz.exists():
        return "closed" if last is not None and last["metric_name"] == END_METRIC else "open"
    return "partial" if last is not None else "none"


def status_grid(out: Path, n_seeds: int) -> dict[int, dict[str, str]]:
    grid: dict[int, dict[str, str]] = {}
    for n in range(1, n_seeds + 1):
        sd = seed_dir(out, n)
        row = {t: cell_state(sd / f"{t}_s{n}") for t in TAGS[:4]}
        row["uni_fed"] = fed_state(fed_dir(out, n))
        grid[n] = row
    return grid


def cmd_status(out: Path = OUT, config_path: Path | None = None) -> dict[int, dict[str, str]]:
    """시드 × 칸 격자. 설정이 비어 있어도 돈다(시드 수는 등록 시드표, 없으면 3)."""
    from vlm.uni_config import load_uni_config

    try:
        n_seeds = len(load_uni_config(config_path).seeds) or 3
    except Exception:  # noqa: BLE001 - 상태 보기는 설정이 깨져도 돈다
        n_seeds = 3
    grid = status_grid(out, n_seeds)
    mark = {"closed": "O", "open": "열림", "partial": "잔해", "none": "·"}
    for n, row in grid.items():
        print(f"시드 {n}: " + "  ".join(f"{t} {mark[s]}" for t, s in row.items()))
    return grid


def main(argv: list[str] | None = None, *, out: Path = OUT, config_path: Path | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["status", "initadapter", "local", "central", "fed", "all", "close-ledger",
                                    "pin", "ecc-prepare", "ecc", "ecc-accept", "gate-check"])
    ap.add_argument("--ecc-root", type=Path, default=None, help="ecc-prepare · ecc 의 점검 루트(outputs/early_check/ecc-…)")
    ap.add_argument("--resume-step", type=int, default=None, help="ecc — 판정 불가 뒤 이 점검 step 에서 잇는다")
    ap.add_argument("--spent-hours", type=float, default=None, help="ecc — 인스턴스가 켜져 있던 시간(상한 대조)")
    ap.add_argument("--next-hours", type=float, default=None, help="ecc — 다음 일의 예상 시간(상한 대조)")
    ap.add_argument("--max-hours", type=float, default=None, help="ecc — 점검 시도의 GPU 시간 상한")
    ap.add_argument("--max-won", type=float, default=None, help="ecc — 점검 시도의 금액 상한(원)")
    ap.add_argument("--permit", type=Path, default=None, help="ecc-accept — 평가 쪽 통과 파일(early_coord_check.json)")
    ap.add_argument("--target-list", type=Path, default=None, help="ecc-prepare — 평가 쪽 대상 목록(규칙 파일의 target_list 와 같아야 한다)")
    ap.add_argument("--seed", type=int, default=None, help="시드 번호(1 부터)")
    ap.add_argument("--tag", default=None, help="close-ledger 의 칸 태그")
    ap.add_argument("--client", default=None, choices=list(CLIENT_TAGS), help="local — 이 참여자의 칸 하나만")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        cmd_status(out, config_path)
        return 0
    try:
        guard_env()
        guard_cwd()
        cfg = load_main_config(config_path)
        if a.client is not None and a.cmd != "local":
            raise MainRefused("client_on_other", "--client 는 local 에서만 받는다")
        if a.cmd == "all":
            cmd_all(cfg, [a.seed] if a.seed else list(range(1, len(cfg.seeds) + 1)), out=out,
                    config_path=config_path)
            return 0
        if a.seed is None:
            raise MainRefused("seed_missing", "--seed 가 필요하다")
        ecc_args = (a.ecc_root, a.resume_step, a.spent_hours, a.next_hours, a.max_hours, a.max_won, a.permit,
                    a.target_list)
        if a.cmd not in ("ecc", "ecc-prepare", "ecc-accept") and any(v is not None for v in ecc_args):
            raise MainRefused("ecc_args_on_main", "점검 인자(--ecc-root · --resume-step · …)는 ecc · ecc-prepare · ecc-accept 에서만 받는다")
        if a.cmd == "gate-check":
            stage_gate_check(cfg, a.seed, out=out, config_path=config_path)
            return 0
        if a.cmd in ("ecc", "ecc-prepare", "ecc-accept"):
            if a.ecc_root is None:
                raise MainRefused("ecc_root_missing", "--ecc-root 가 필요하다")
            if a.cmd == "ecc-prepare":
                if a.target_list is None or any(v is not None for v in ecc_args[1:-1]):
                    raise MainRefused("ecc_args_on_prepare", "ecc-prepare 는 --ecc-root · --target-list 만 받는다")
                stage_ecc_prepare(cfg, a.seed, a.ecc_root, target_list=a.target_list, out=out, config_path=config_path)
                return 0
            if a.target_list is not None:
                raise MainRefused("ecc_args_on_run", "--target-list 는 ecc-prepare 에서만 받는다")
            if a.cmd == "ecc-accept":
                if a.permit is None or any(v is not None for v in ecc_args[1:-2]):
                    raise MainRefused("ecc_args_on_accept", "ecc-accept 는 --ecc-root · --permit 만 받는다")
                stage_ecc_accept(cfg, a.seed, a.ecc_root, a.permit, out=out, config_path=config_path)
                return 0
            if a.permit is not None:
                raise MainRefused("ecc_args_on_run", "ecc 는 --permit 을 받지 않는다")
            if a.spent_hours is None or a.next_hours is None:
                raise MainRefused("ecc_budget_args", "ecc 는 --spent-hours · --next-hours 가 있어야 한다(11번 §5)")
            res = stage_ecc(cfg, a.seed, a.ecc_root, spent_hours=a.spent_hours, next_hours=a.next_hours,
                            max_hours=a.max_hours, max_won=a.max_won,
                            resume_step=a.resume_step, out=out, config_path=config_path)
            return int(res["exit_code"])
        if a.cmd == "pin":
            stage_pin(cfg, a.seed, out=out, config_path=config_path)
            return 0
        if a.cmd == "close-ledger":
            from vlm.train_cell import CellRejected, close_ledger

            if a.tag not in TAGS[:4]:
                raise MainRefused("tag", f"close-ledger 는 로컬 · 중앙 칸만 받는다: {a.tag!r}")
            try:
                close_ledger(spec_for(cfg, a.seed, out=out, config_path=config_path), a.tag)
            except CellRejected as exc:
                raise MainRefused(exc.code, str(exc)) from None
            return 0
        if a.cmd == "local":
            stage_local(cfg, a.seed, out=out, config_path=config_path, client=a.client)
            return 0
        {"initadapter": stage_initadapter, "central": stage_central,
         "fed": stage_fed}[a.cmd](cfg, a.seed, out=out, config_path=config_path)
        return 0
    except MainRefused as exc:
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
