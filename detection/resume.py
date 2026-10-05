"""재개 전용 체크포인트 — 검출 학습이 죽었을 때 0에서 다시 시작하지 않기 위한 경로.

## 이것은 `best` 체크포인트가 아니다

프로젝트 규칙은 **`best` 금지 · `last` 채점**이다. 그 규칙이 막는 것은 *선택*이다 —
여러 시점 중 val 이 좋은 하나를 고르면 그것이 암묵적 조기 종료이기 때문이다.

여기서 만드는 파일은 **고를 수 있는 후보가 아니다.**

- 저장 시점은 **매 epoch 경계**이고 val 지표를 보지 않는다. 애초에 검증을 돌지 않는다.
- 읽는 쪽은 **재개 경로 하나뿐**이다. 채점기·내보내기·집계는 이 경로를 모른다.
- 재개는 **같은 라운드·같은 클라이언트·같은 시드**로만 허용한다. 신원이 하나라도
  다르면 거부한다. 다른 실행의 상태를 이어받아 `R × E = N` 이 조용히 깨지는 경로를 막는다.
- 학습이 정상 완주하면 이 파일들은 **버린다.** 산출물은 `export_weights()` 의 raw 가중치다.

즉 `last` 를 만들어 가는 도중의 스냅샷이지 `last` 의 경쟁자가 아니다.

## 왜 필요한가

파일럿 22~25 GPU시간에서 외부 중지가 2회 났다. 기저율 **11~18 GPU시간당 1회**다.
본실험은 400~1,700 GPU시간이고, `FIXED_OVERRIDES["save"]=False` 에 `save_model()` 이
no-op 이라 ②③ 은 100 epoch 단일 런이다. 재개 경로가 없으면 완주 확률이 사실상 0이다.

분리 실행(`Start-Process` detach)은 기본으로 쓰되 유일한 방어로 두지 않는다 —
무중단 근거가 5시간 두 번뿐이라 검정력이 없다.

## 무엇을 담는가

옵티마이저 상태와 RNG 를 함께 담지 않으면 재개가 "이어 하기"가 아니라 "다시 하기"가 된다.
SGD momentum 버퍼가 0에서 다시 쌓이고 증강 난수열이 갈라지면, 재개한 런과 무중단으로
완주한 런이 다른 궤적을 그린다. 그러면 재개는 사고 대응이 아니라 **오염원**이다.

| 항목 | 이유 |
|---|---|
| raw 가중치 (정본 키 순서) | EMA 가 아니다. FedAvg 가 평균하는 것과 같은 텐서다 |
| 옵티마이저 state_dict | momentum 버퍼. 없으면 재개 직후 수 epoch 이 다른 궤적 |
| AMP scaler state | loss scale 이 초기값으로 되돌아가면 첫 스텝들이 스킵될 수 있다 |
| epoch 카운터 | `start_epoch` 복원 → 스케줄러·warmup·close_mosaic 위치가 따라온다 |
| 누적 스텝 수 | 회계 매트릭스의 `R × E = N` 감사가 프로세스 경계를 넘어 성립해야 한다 |
| RNG (python·numpy·torch·cuda) | 증강 난수열 |

**epoch 경계에서만 저장한다.** epoch 중간에서는 DataLoader 워커의 순회 위치를 복원할
방법이 없어서, 중간 재개는 "같은 epoch 의 앞부분을 두 번 보는" 오염이 된다.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from detection import serialize

__all__ = [
    "FORMAT",
    "ResumeState",
    "ResumeCheckpointer",
    "ResumeRejected",
    "latest_resume",
    "clear_resume",
]

FORMAT = "weld-fl-resume/1"

#: 재개 디렉터리 안의 파일명. `best`·`last` 라는 이름을 의도적으로 쓰지 않는다 —
#: 채점 경로가 실수로 집어 갈 수 있는 이름을 만들지 않는다.
_PREFIX = "resume_ep"
_SUFFIX = ".pt"


@dataclass(frozen=True)
class ResumeIdentity:
    """이 상태가 어느 실행의 것인지. 하나라도 다르면 재개를 거부한다.

    모델 **구조**의 동일성은 여기서 보지 않는다 — `apply_resume` 의 `assert_compatible`
    이 키·모양·dtype 계열까지 대조하므로 중복이고, 정본 키는 모델이 만들어진 뒤에야
    알 수 있어 신원 필드로 쓰면 저장 시점과 대조 시점이 어긋난다.
    여기 있는 것은 **학습량 등가를 좌우하는 값들**과 **정책의 판**이다.
    """

    run_id: str
    round_idx: int
    client_idx: int
    seed: int
    total_epochs: int
    local_epochs: int
    model: str
    data: str
    loader_reseed_per_epoch: bool = False
    loader_seed: int | None = None
    profile: str = ""
    #: 아래 셋은 **학습 정책의 판**이다. 값이 아니라 판을 적는다 — 같은 seed·같은 예산이어도
    #: 정책이 다르면 다른 프로토콜이고, 이어 가면 회계는 맞는데 궤적이 섞인다.
    #:
    #: 기본값이 빈 문자열인 이유는 뒤호환이다. `_load_one` 이 `ResumeIdentity(**dict)` 로
    #: 되살리므로 이 필드가 없던 체크포인트는 `""` 가 되고, 값을 채우는 새 실행과 **자동으로
    #: 어긋나 거부된다**(`mismatch` 가 전 필드를 본다). 검출 쪽은 세 값을 비우므로 영향이 없다.
    shuffle_policy: str = ""
    template_mode: str = ""
    gen_prefix_digest: str = ""

    def mismatch(self, other: "ResumeIdentity") -> list[str]:
        return [
            f"{f}: 체크포인트 {getattr(self, f)!r} ≠ 현재 {getattr(other, f)!r}"
            for f in self.__dataclass_fields__
            if getattr(self, f) != getattr(other, f)
        ]


class ResumeRejected(ValueError):
    """읽을 수는 있는데 이어 받을 수 없는 재개 파일이다 — 형식이 다르거나 신원을 해석할 수 없다.

    **건너뛰고 직전 파일로 물러나지 않는다.** 물러나면 모든 후보가 걸러진 뒤 처음부터 학습하게 되고, 새 실행의
    저장이 그 폴더의 기존 체크포인트 이름을 덮는다. 찢어진 파일(끝까지 쓰이지 않아 `torch.load` 가 못 읽는 파일)만
    건너뛴다.
    """


@dataclass
class ResumeState:
    """읽어 들인 재개 상태."""

    path: Path
    identity: ResumeIdentity
    epoch_done: int
    epochs_ran_in_round: int
    optimizer_steps: int
    wall_s: float
    payload: dict[str, Any]

    @property
    def next_epoch(self) -> int:
        """재개해서 처음 돌 전역 epoch 번호."""
        return self.epoch_done + 1


def _rng_snapshot() -> dict[str, Any]:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _rng_restore(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda"])
        except (RuntimeError, ValueError):
            # 장치 수가 달라진 경우. 재개 자체를 막을 사유는 아니지만 조용히 넘기지 않는다.
            print("[resume] CUDA RNG 복원 실패 — 장치 구성이 저장 시점과 다르다", flush=True)


class ResumeCheckpointer:
    """`on_fit_epoch_end` 콜백. epoch 경계마다 재개 상태를 덤프한다.

    Args:
        out_dir: 재개 전용 디렉터리. **학습 산출물 디렉터리와 분리한다** — 채점·내보내기가
            훑는 경로에 재개 파일을 두지 않는다.
        identity: 이 실행의 신원. 재개 시 대조한다.
        canonical_keys: 정본 키 순서. `None` 이면 **첫 저장 때 모델에서 유도한다** —
            로컬·중앙 칸은 서버 파라미터를 받지 않아 호출 시점에 키를 모른다.
        step_counter: `optimizer_steps` 를 읽을 객체(`.n` 속성). 프로세스 경계를 넘어
            누적 스텝을 잇기 위해 함께 저장한다.
        resumed_epochs: 이번 프로세스가 시작되기 전에 이 라운드에서 이미 돈 epoch 수.
        resumed_steps: 마찬가지로 이미 밟은 옵티마이저 스텝 수.
        keep: 남겨 둘 최신 체크포인트 개수. 2 이상을 권한다 — 최신본이 찢어졌을 때
            직전 것으로 물러날 수 있어야 한다.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        identity: ResumeIdentity,
        canonical_keys: Sequence[str] | None = None,
        step_counter: Any = None,
        resumed_epochs: int = 0,
        resumed_steps: int = 0,
        keep: int = 2,
        state_dict_fn: Any = None,
    ) -> None:
        #: 무엇을 저장할지 고르는 훅. 기본은 모델 전체 state_dict(검출). 통합형은 어댑터만
        #: 교환·저장하므로 `get_peft_model_state_dict` 를 준다. 기본값을 그대로 쓰면 4bit
        #: 양자화된 베이스 가중치까지 통째로 덤프하게 된다.
        self.state_dict_fn = state_dict_fn or (
            lambda tr: getattr(tr.model, "module", tr.model).state_dict()
        )
        self.out_dir = Path(out_dir)
        self.identity = identity
        self.canonical_keys = list(canonical_keys) if canonical_keys is not None else None
        self.step_counter = step_counter
        self.resumed_epochs = int(resumed_epochs)
        self.resumed_steps = int(resumed_steps)
        self.keep = max(1, int(keep))
        self.saved: list[Path] = []
        self.n_saves = 0
        self.last_error: str | None = None
        #: 호출자가 저장 직전에 채워 넣는 추가 필드. 학습 루프마다 이어야 할 누적값이
        #: 다르다(통합형은 감독 토큰 수). 예약 키를 덮어쓰지는 못한다.
        self.extra: dict[str, Any] = {}

    # -- 콜백 --------------------------------------------------------------
    def __call__(self, trainer: Any) -> None:
        try:
            self.save(trainer)
        except Exception as exc:  # noqa: BLE001 - 체크포인트 실패로 학습을 죽이지 않는다
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[resume] 저장 실패 (학습은 계속한다): {self.last_error}", flush=True)

    def save(self, trainer: Any) -> Path:
        import torch

        sd = self.state_dict_fn(trainer)
        if self.canonical_keys is None:
            self.canonical_keys = serialize.canonical_keys(sd)
        arrays = serialize.state_dict_to_ndarrays(sd, self.canonical_keys)
        ran_now = int(trainer.epoch) - int(trainer.start_epoch) + 1
        steps_now = int(getattr(self.step_counter, "n", 0) or 0)

        payload = {
            "format": FORMAT,
            "purpose": "재개 전용. 채점·선택 대상이 아니다 (detection/resume.py 문서 참조)",
            "identity": vars(self.identity),
            "epoch_done": int(trainer.epoch),
            "epochs_ran_in_round": self.resumed_epochs + ran_now,
            "optimizer_steps": self.resumed_steps + steps_now,
            "wall_s": float(getattr(trainer, "_resume_wall_s", 0.0)),
            "canonical_keys": self.canonical_keys,
            "weights": arrays,
            "optimizer": trainer.optimizer.state_dict(),
            "scaler": (trainer.scaler.state_dict()
                       if getattr(trainer, "scaler", None) is not None else None),
            "rng": _rng_snapshot(),
            "loader_generator": loader_generator_state(trainer),
        }
        # 검출의 배치 수와 optimizer 갱신 횟수는 다른 카운터다. 후자는
        # trainer에서 재개 이전 누적값을 복원한 뒤 계속 센다.
        if hasattr(trainer, "n_optimizer_updates"):
            payload["optimizer_updates"] = int(trainer.n_optimizer_updates)
        # 8-3: 실제 적용 수. **미상이면 키를 쓰지 않는다** — 0 을 쓰면 다음 재개가 그것을
        # 확정값으로 읽어 "전 스텝 스킵" 처럼 보인다(`round_runner` 의 applied_known).
        if getattr(trainer, "applied_known", True) and hasattr(trainer, "n_optimizer_updates_applied"):
            payload["optimizer_updates_applied"] = int(trainer.n_optimizer_updates_applied)
        if hasattr(trainer, "_resume_lr_trace"):
            payload["lr_trace"] = list(trainer._resume_lr_trace.trace)
        for k, v in self.extra.items():
            if k in payload:
                raise ValueError(f"예약된 체크포인트 키를 덮어쓸 수 없다: {k!r}")
            payload[k] = v

        self.out_dir.mkdir(parents=True, exist_ok=True)
        final = self.out_dir / f"{_PREFIX}{int(trainer.epoch):04d}{_SUFFIX}"
        tmp = final.with_suffix(".tmp")
        # 원자적 교체. 찢어진 쓰기는 .tmp 에 남고 최신 정상본은 그대로 살아 있다.
        torch.save(payload, tmp)
        # 바이트가 디스크에 닿은 뒤에 교체한다. 저장 형식은 바꾸지 않는다 — `torch.save` 가 쓴 파일을 다시 열어
        # fsync 만 한다(판정 05 의 14).
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        if final.exists():
            # 같은 이름이 이미 있으면 **그 파일을 건드리지 않고** 새 이름으로 저장한다 — 기존 체크포인트는 지우지도
            # 옮기지도 않고 재개 후보에서도 빠지지 않는다(결정표 8번). 찢어져 건너뛴 최신본 자리에 다시 저장할 때가
            # 이 경우다. 새 이름 `resume_epNNNN.rKKK.pt` 는 같은 epoch 의 옛 이름보다 뒤로 정렬되어 먼저 읽히고,
            # 교체 한 번(`os.replace(tmp, …)`)이 실패하거나 그 사이에 죽어도 옛 파일은 후보로 남는다.
            k = 1
            while final.with_name(f"{final.stem}.r{k:03d}{_SUFFIX}").exists():
                k += 1
            final = final.with_name(f"{final.stem}.r{k:03d}{_SUFFIX}")
        os.replace(tmp, final)
        self.n_saves += 1
        self.saved.append(final)
        self._prune()
        return final

    def _prune(self) -> None:
        """이 체크포인터가 **이번 프로세스에서 저장한** 파일만 `keep` 개로 줄인다.

        폴더의 다른 파일(앞 프로세스의 체크포인트 · 찢어진 파일 · 같은 이름을 피해 새 이름으로 남긴 옛 파일)은
        지우지 않는다 — 기존 체크포인트를 지우지 않는다(결정표 8번).
        """
        for old in self.saved[: -self.keep]:
            try:
                old.unlink()
            except OSError:
                pass
        self.saved = self.saved[-self.keep:]


def loader_generator_state(trainer: Any) -> Any | None:
    """학습 로더의 `torch.Generator` 상태. 없으면 `None`.

    Ultralytics `build_dataloader` 는 이 생성기를 **상수 `6148914691236517205 + RANK`** 로
    시드한다 — `args.seed` 와 무관하다. epoch 별 셔플 순열이 여기서 나오므로, 새 프로세스는
    항상 epoch 0 의 순열부터 다시 뽑는다. 재개하면서 이 상태를 잇지 않으면 재개한 런은
    중단 없이 완주한 런과 **다른 데이터 순서**를 보게 되고, 그러면 재개가 사고 대응이
    아니라 오염원이 된다.
    """
    loader = getattr(trainer, "train_loader", None)
    gen = getattr(loader, "generator", None)
    return gen.get_state() if gen is not None else None


def restore_loader_generator(trainer: Any, state: Any) -> bool:
    """생성기 상태를 되돌리고 반복자를 다시 만든다. 되돌렸으면 `True`.

    반복자를 새로 만들어야 하는 이유: 로더는 생성 시점에 **이미 첫 순열을 뽑아** 프리페치를
    시작한다. 상태만 바꾸면 그 순열은 그대로 남는다. `reset()` 이 워커를 접고 반복자를
    다시 만들면서 되돌린 상태로 순열을 뽑는다.

    ## 무엇을 보장하고 무엇을 보장하지 않는가

    **보장한다** — 재개한 런은 **이미 쓴 순열을 다시 보지 않는다.** 되돌리지 않으면 새
    프로세스의 로더가 epoch 0 의 순열부터 다시 뽑으므로, 재개 지점부터 앞 epoch 들의
    데이터 순서를 그대로 반복한다.

    **보장하지 않는다** — 중단 없이 완주한 런과 **같은 순열은 아니다.** PyTorch 의
    `DataLoader` 반복자는 생성될 때 순열보다 먼저 워커용 base seed 를 한 번 뽑는데,
    무중단 런은 그 시점에 그 draw 를 하지 않는다. 한 draw 만큼 어긋나고, 되감을 방법이
    없다. 워커 RNG 도 마찬가지로 새 base seed 에서 갈라진다.

    그래서 **재개한 런은 시드만으로 재현되지 않는다.** 재개 사실과 지점을 기록해야 한다.
    갈라짐의 크기는 `scripts/verify_resume.py` 가 재실행 대조군과 함께 실측한다.
    """
    loader = getattr(trainer, "train_loader", None)
    gen = getattr(loader, "generator", None)
    if gen is None or state is None:
        return False
    gen.set_state(state)
    loader.reset()
    return True


class _Torn(Exception):
    """끝까지 쓰이지 않은 파일 — `torch.load` 가 바이트를 풀지 못했다."""


def _is_torn(exc: BaseException) -> bool:
    """`torch.load` 가 낸 예외가 **바이트가 덜 쓰인 파일**의 것인지. 모듈 부재 같은 해석 실패는 아니다."""
    import pickle
    import zipfile

    if isinstance(exc, (EOFError, pickle.UnpicklingError, zipfile.BadZipFile)):
        return True
    msg = str(exc)
    return isinstance(exc, RuntimeError) and any(
        s in msg for s in ("PytorchStreamReader", "zip archive", "central directory", "file read failed"))


def _load_one(path: Path, identity_cls: type | None = None) -> ResumeState:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - 찢어짐만 가려내고 나머지는 그대로 올린다
        if _is_torn(exc):
            raise _Torn(f"{type(exc).__name__}: {exc}") from exc
        raise ResumeRejected(f"재개 파일을 해석할 수 없다({type(exc).__name__}: {exc}): {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        got = payload.get("format") if isinstance(payload, dict) else type(payload).__name__
        raise ResumeRejected(f"알 수 없는 체크포인트 형식: {got!r} ({path})")
    cls = identity_cls or ResumeIdentity
    try:
        ident = cls(**payload["identity"])
    except (KeyError, TypeError) as exc:
        raise ResumeRejected(f"재개 파일의 신원을 해석할 수 없다({type(exc).__name__}: {exc}): {path}") from exc
    return ResumeState(
        path=path,
        identity=ident,
        epoch_done=int(payload["epoch_done"]),
        epochs_ran_in_round=int(payload["epochs_ran_in_round"]),
        optimizer_steps=int(payload["optimizer_steps"]),
        wall_s=float(payload.get("wall_s", 0.0)),
        payload=payload,
    )


def latest_resume(out_dir: str | Path, *, identity: ResumeIdentity | None = None,
                  identity_cls: type | None = None) -> ResumeState | None:
    """가장 최신의 온전한 재개 상태. 없으면 `None`.

    최신본이 **찢어졌으면**(끝까지 쓰이지 않아 `torch.load` 가 못 읽으면) 그 다음으로 물러난다. 물러난 사실은
    표준출력에 남긴다 — 조용히 옛 상태로 재개하면 몇 epoch 이 유실된 채 진행된다. **읽히는데 형식이 다르거나
    신원을 해석할 수 없는 파일은 `ResumeRejected` 로 멈춘다** — 물러나 처음부터 학습하면 그 폴더의 기존
    체크포인트가 새 저장에 덮인다.

    `identity` 를 주면 신원 대조까지 한다. 어긋나면 `ValueError` 다 — **다른 실행의 상태로
    이어 가는 것을 허용하지 않는다.** 이어 갔다면 `R × E = N` 이 회계에는 맞는데 실제
    학습량은 다른 상태가 되고, 그 어긋남은 지표에 흔적을 남기지 않는다.

    `identity_cls` 는 신원을 되살릴 클래스다. 없으면 `identity` 의 클래스, 그것도 없으면 `ResumeIdentity` 다 —
    통합형은 필드가 더 많은 자기 신원(`vlm/resume_uni.py`)을 쓴다.
    """
    d = Path(out_dir)
    if not d.is_dir():
        return None
    cls = identity_cls or (type(identity) if identity is not None else None)
    for path in sorted(d.glob(f"{_PREFIX}*{_SUFFIX}"), reverse=True):
        try:
            state = _load_one(path, cls)
        except _Torn as exc:
            print(f"[resume] {path.name} 찢어진 파일({exc}) — 직전 것으로 물러난다", flush=True)
            continue
        if identity is not None:
            bad = state.identity.mismatch(identity)
            if bad:
                raise ValueError(
                    "재개 체크포인트의 신원이 현재 실행과 다르다. 이어 가면 학습량 등가가 "
                    "조용히 깨진다:\n  " + "\n  ".join(bad)
                )
        return state
    return None


def clear_resume(out_dir: str | Path) -> int:
    """정상 완주 후 재개 파일을 지운다. 지운 개수를 돌려준다.

    남겨 두면 다음 실행이 "이미 끝난 라운드"를 재개 상태로 오인할 여지가 생기고,
    무엇보다 채점 대상이 아닌 가중치 파일이 산출물 트리에 남는다.
    """
    d = Path(out_dir)
    n = 0
    if not d.is_dir():
        return 0
    for p in list(d.glob(f"{_PREFIX}*{_SUFFIX}")) + list(d.glob(f"{_PREFIX}*.tmp")):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


def apply_resume(
    trainer: Any,
    state: ResumeState,
    *,
    state_dict_fn: Any = None,
    load_state_dict_fn: Any = None,
) -> None:
    """`_setup_train` 이 끝난 트레이너에 재개 상태를 밀어 넣는다.

    호출 순서가 중요하다. 이 함수는 **가중치 주입(서버 파라미터) 뒤, 스케줄러 재설정
    앞**에서 불려야 한다. 재개 상태의 가중치가 서버 파라미터를 덮어써야 맞다 —
    라운드 중간에서 죽은 것이므로 그 시점의 가중치가 정본이다.

    로더 생성기 복원은 여기서 하지 않는다. mosaic 종료 재적용이 반복자를 다시 만들기
    때문에, 그보다 앞에서 되돌리면 순열이 한 번 더 소비된다. `restore_loader_generator`
    를 `_setup_train` 의 **맨 끝**에서 따로 부른다.

    Args:
        state_dict_fn: 기준 state_dict 를 읽는 훅. 통합형은 어댑터만 교환하므로
            `get_peft_model_state_dict` 를 준다. 기본은 모델 전체 state_dict.
        load_state_dict_fn: 되돌린 state_dict 를 밀어 넣는 훅. 통합형은
            `set_peft_model_state_dict` 다. 기본은 `model.load_state_dict(strict=True)`.
    """
    import torch

    model = getattr(trainer.model, "module", trainer.model)
    ref = state_dict_fn(trainer) if state_dict_fn else model.state_dict()
    keys = list(state.payload["canonical_keys"])
    arrays = list(state.payload["weights"])
    serialize.assert_compatible(arrays, keys, ref)
    sd = serialize.ndarrays_to_state_dict(arrays, keys, ref)
    if load_state_dict_fn:
        load_state_dict_fn(model, sd)
    else:
        model.load_state_dict(sd, strict=True)

    trainer.optimizer.load_state_dict(state.payload["optimizer"])
    if state.payload.get("scaler") is not None and getattr(trainer, "scaler", None) is not None:
        trainer.scaler.load_state_dict(state.payload["scaler"])
    _rng_restore(state.payload["rng"])

    trainer.start_epoch = state.next_epoch
    # AMP scaler 와 옵티마이저를 CPU 에서 읽었으므로 텐서를 장치로 옮긴다.
    for st in trainer.optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(trainer.device)
