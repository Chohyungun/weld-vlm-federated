"""통합형 본실험 실행기 — 시드별 다섯 모델 체인(미니스펙 12번 §1-3).

한 시드의 다섯 모델(로컬 C1 · C2 · C3 · 중앙 · 연합)을 끝낸 뒤 다음 시드로 간다. **시드 목록 · 학습량 · 모델 판 · 페어 ·
프롬프트 경로 · 좌표 규약 · 프로세서 설정 · 감독 토큰 기대값은 `configs/base.yaml` 에서 읽는다** — 상수로 두지 않는다.
설정이 비어 있으면(`null`) 어느 학습 명령도 시작하지 않는다. 목적은 언제나 `main` 이다(리허설 · 진단은 오케스트레이터의 몫이다).
**저장소 루트에서 띄운다** — 설정의 상대 경로(페어 · 프롬프트)는 저장소 루트 기준이라, 다른 폴더에서 띄우면 거부한다.

## 명령

    python -X utf8 scripts/main_uni.py status
    python -X utf8 scripts/main_uni.py initadapter --seed 1
    python -X utf8 scripts/main_uni.py local --seed 1
    python -X utf8 scripts/main_uni.py central --seed 1
    python -X utf8 scripts/main_uni.py fed --seed 1
    python -X utf8 scripts/main_uni.py all [--seed 1]
    python -X utf8 scripts/main_uni.py close-ledger --seed 1 --tag uni_local_C1

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


def stage_local(cfg: Any, n: int, *, out: Path = OUT, config_path: Path | None = None):
    from vlm.train_cell import run_uni_local_cell

    spec = spec_for(cfg, n, out=out, config_path=config_path)
    return _cell_call(lambda: run_uni_local_cell(spec=spec), "local", n, out)


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
    ap.add_argument("cmd", choices=["status", "initadapter", "local", "central", "fed", "all", "close-ledger"])
    ap.add_argument("--seed", type=int, default=None, help="시드 번호(1 부터)")
    ap.add_argument("--tag", default=None, help="close-ledger 의 칸 태그")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        cmd_status(out, config_path)
        return 0
    try:
        guard_env()
        guard_cwd()
        cfg = load_main_config(config_path)
        if a.cmd == "all":
            cmd_all(cfg, [a.seed] if a.seed else list(range(1, len(cfg.seeds) + 1)), out=out,
                    config_path=config_path)
            return 0
        if a.seed is None:
            raise MainRefused("seed_missing", "--seed 가 필요하다")
        if a.cmd == "close-ledger":
            from vlm.train_cell import CellRejected, close_ledger

            if a.tag not in TAGS[:4]:
                raise MainRefused("tag", f"close-ledger 는 로컬 · 중앙 칸만 받는다: {a.tag!r}")
            try:
                close_ledger(spec_for(cfg, a.seed, out=out, config_path=config_path), a.tag)
            except CellRejected as exc:
                raise MainRefused(exc.code, str(exc)) from None
            return 0
        {"initadapter": stage_initadapter, "local": stage_local, "central": stage_central,
         "fed": stage_fed}[a.cmd](cfg, a.seed, out=out, config_path=config_path)
        return 0
    except MainRefused as exc:
        return int(exc.code)


if __name__ == "__main__":
    raise SystemExit(main())
