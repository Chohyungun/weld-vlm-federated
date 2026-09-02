"""검출 3칸 공통 진입점 — `train_round`.

로컬·중앙·연합 세 칸이 **모두 이 함수 하나를 통과한다.** 로컬과 중앙은 `R=1, E=N`인
퇴화 케이스일 뿐이다. 칸마다 다른 학습 루프를 타면 "구조 내 동일" 원칙이 문면으로만
남고, 세 칸의 차이가 학습 방식 때문인지 코드 경로 때문인지 구분할 수 없게 된다.

무상태로 설계했다. 호출마다 트레이너를 새로 만들고 라운드 간 상태를 파이썬 객체에
걸지 않는다. 연합 클라이언트는 라운드마다 새 프로세스일 수 있으므로, 상태를 들고 있는
설계는 시뮬레이션에서만 우연히 동작한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from detection import serialize

# `FedDetectionTrainer` 는 `train_round` 안에서 지연 import 한다. 이 모듈의 `FIXED_OVERRIDES`·
# `derive_seed` 는 ultralytics 없이도 의미가 있는 순수 데이터·함수이고, 실제로 검출 학습을
# 돌리지 않는 쪽(회계 감사, 설정 검증)이 이것들만 쓰려고 무거운 optional 의존성을 끌 이유가 없다.

__all__ = [
    "RoundResult",
    "train_round",
    "derive_seed",
    "FIXED_OVERRIDES",
    "FIXED_PILOT",
    "FIXED_PROBE_NONDET",
    "PROFILES",
]


#: 5칸 공통 고정값. 여기 있는 키는 호출자가 덮어쓸 수 없다(§`train_round` 참조).
#: 값 자체는 파일럿에서 확정하며 확정본은 configs 에 둔다 — 여기 있는 것은 규칙이 되는 항목이다.
FIXED_OVERRIDES: dict[str, Any] = {
    # 조기 종료 금지 — R×E=N 학습량 등가가 논문의 공정성 주장이다.
    # patience 를 epoch 수 이상으로 두고, 그것과 별개로 stopper 자체를 스텁으로 교체한다.
    "patience": 10000,
    # optimizer='auto' 는 명시한 lr0·momentum 을 버리고 AdamW 로 갈아치운다(경고 한 줄만 남긴다).
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "cos_lr": True,
    "warmup_epochs": 3.0,
    "close_mosaic": 10,
    "batch": 32,
    "imgsz": 640,
    "deterministic": True,
    # 저장·검증은 래퍼가 no-op 으로 막지만, 인자 수준에서도 꺼 두어 의도를 드러낸다.
    "save": False,
    "val": False,
    "plots": False,
    # mlflow 가 설치돼 있으면 Ultralytics 가 자동 연동한다. 로깅 경로는 서버 시점 하나로 통일한다.
    "mlflow": False,
}


#: 파일럿 프로파일. **본실험과 값이 다르지만 파일럿 안에서는 다섯 칸이 전부 같은 값을 쓴다.**
#: 공통 고정이 금지하는 것은 "칸마다 다른 값"이지 "실험마다 다른 값"이 아니다. 프로파일을
#: 통째로 교체하게 만들어서, 칸별로 몰래 다른 값을 넣는 경로만 계속 막는다.
#:
#: 파일럿에서도 풀지 않는 것: patience(조기 종료 금지), optimizer 명시 고정,
#: mlflow=False, save/val=False. 규칙을 풀면 파일럿이 검증하는 대상이 본실험과 달라진다.
FIXED_PILOT: dict[str, Any] = {
    **{k: v for k, v in FIXED_OVERRIDES.items()},
    "batch": 2,       # 16GB 에서 416px 기준. grad accum 으로 유효 배치를 맞춘다
    "imgsz": 416,
    "close_mosaic": 1,  # E=2 라 10 이면 한 번도 발화하지 않는다
}

#: **계측 전용.** `deterministic=False` 하나만 다르다. 결정성 비용이 얼마인지는 실측 없이는
#: 알 수 없고, 그 값은 GPU 부하 판단의 입력이다. 그렇다고 `extra_overrides` 로 공통 고정을
#: 뚫는 구멍을 내면 칸별로 몰래 값을 바꾸는 경로가 함께 열린다 — 프로파일을 통째로 바꾸는
#: 기존 방식을 그대로 쓰고, **이름으로 산출물에 표시가 남게** 한다.
#:
#: **이 프로파일로 만든 가중치는 실험 산출물이 아니다.** 다섯 칸 중 어디에도 쓰지 않는다.
FIXED_PROBE_NONDET: dict[str, Any] = {**FIXED_OVERRIDES, "deterministic": False}

#: 이름으로 프로파일을 고른다. 호출자가 dict 을 직접 조립하지 못하게 한다.
PROFILES: dict[str, dict[str, Any]] = {
    "main": FIXED_OVERRIDES,
    "pilot": FIXED_PILOT,
    "probe_nondet": FIXED_PROBE_NONDET,
}


@dataclass
class RoundResult:
    """한 라운드(또는 로컬·중앙 칸의 전체 학습) 산출물."""

    ndarrays: list[np.ndarray]
    num_examples: int
    round_idx: int
    client_idx: int
    epochs_ran: int
    seed: int
    optimizer_steps: int
    param_l2_norm: float
    payload_bytes: int
    injection_digest: list[float] = field(default_factory=list)
    effective_optimizer: dict[str, Any] = field(default_factory=dict)
    lr_trace: list[tuple[int, float]] = field(default_factory=list)
    stopper_calls: list[tuple[int, float | None]] = field(default_factory=list)
    #: 실제로 끼워진 stopper 의 클래스 이름. 스텁 교체가 실패하면 여기가 달라진다 —
    #: 회계 검사 (4)가 이 값으로 실패할 수 있게 됐다(74번 감사 P9 정정).
    stopper_class: str = ""
    #: stopper 가 참을 돌려준 횟수. 스텁에서 읽은 실측이며 상수가 아니다.
    stopper_true_count: int | None = None
    budget_fired_at: int | None = None
    peak_vram_gb: float = 0.0
    #: 실제 `optimizer.step()` 횟수. `optimizer_steps`(배치 수)와 다르다 —
    #: Ultralytics 가 `nbs=64` 기준으로 누적하기 때문이다(숨은 기본값 #10).
    optimizer_updates: int = 0
    #: 재개해서 이어 간 실행인가. 이어 간 런은 무중단 런과 다른 궤적을 그리므로
    #: 회계에 남겨야 한다(`detection/resume.py` 참조).
    resumed_from_epoch: int | None = None


def derive_seed(base_seed: int, round_idx: int, client_idx: int) -> int:
    """라운드·클라이언트마다 다르면서 재현 가능한 시드.

    고정 시드를 그대로 쓰면 50라운드가 같은 셔플 순서를 반복한다. 클라이언트마다 데이터
    크기가 달라 반복 주기도 달라지므로, 그 자체가 칸 간 숨은 비대칭이 된다.
    `base_seed` 3세트는 불변이고 파생 공식과 함께 configs 에 기록한다.
    """
    return int(base_seed) + 10007 * int(round_idx) + 101 * int(client_idx)


class _LRTrace:
    """epoch별 학습률 기록. 연합 칸과 중앙집중 칸의 궤적 일치를 검증하는 근거다."""

    def __init__(self) -> None:
        self.trace: list[tuple[int, float]] = []

    def __call__(self, trainer: Any) -> None:
        lrs = getattr(trainer, "lr", None) or {}
        first = next(iter(lrs.values()), float("nan"))
        self.trace.append((int(trainer.epoch), float(first)))


class _StepCounter:
    """옵티마이저 스텝 수. 총 갱신 횟수를 논문에 실측치로 싣는다."""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, trainer: Any) -> None:
        self.n += 1


def train_round(
    *,
    data_yaml: str | Path,
    model: str,
    total_epochs: int,
    local_epochs: int,
    round_idx: int = 0,
    client_idx: int = 0,
    base_seed: int = 0,
    num_examples: int = 0,
    weights_in: Sequence[np.ndarray] | None = None,
    canonical_keys: Sequence[str] | None = None,
    project: str | Path | None = None,
    profile: str = "main",
    extra_overrides: dict[str, Any] | None = None,
    callbacks: dict[str, Any] | None = None,
    resume_dir: str | Path | None = None,
    run_id: str = "",
    clear_resume_on_success: bool = True,
    loader_reseed_per_epoch: bool = False,
) -> RoundResult:
    """라운드 하나를 실행하고 raw 가중치를 돌려준다.

    Args:
        total_epochs: **전역 예산 N.** 라운드 길이가 아니다. warmup 과 cosine 스케줄이
            이 값을 기준으로 계산되므로, 여기에 E를 넣으면 라운드마다 스케줄이 리셋되어
            중앙집중 칸과 학습률 궤적이 달라진다.
        local_epochs: 이번 라운드에 돌 epoch 수 E. 로컬·중앙 칸은 `total_epochs`와 같다.
        num_examples: FedAvg 가중치로 쓰이는 클라이언트 표본 수. 호출자가 매니페스트에서
            센 값을 넘긴다.
        project: 산출물 디렉터리. **절대경로를 준다** — 생략하면 Ultralytics 전역 설정의
            `runs_dir` 로 떨어져 저장소 루트가 오염된다.
        profile: `"main"` 또는 `"pilot"`. 프로파일 안에서는 다섯 칸이 같은 값을 쓴다.
        callbacks: 계측 전용 훅 `{이벤트명: 콜러블}`. 계측은 학습을 바꾸지 않으므로
            공통 고정 검사 대상이 아니다. 학습 경로에 영향을 주는 콜백을 여기로 넣지 마라.
        resume_dir: 재개 전용 체크포인트 디렉터리. 주면 **epoch 경계마다 덤프하고, 같은
            신원의 상태가 이미 있으면 거기서 이어 간다.** `best` 금지 규칙과 충돌하지
            않는다 — 고를 수 있는 후보가 아니라 `last` 를 만들어 가는 도중의 상태이고,
            읽는 쪽이 이 재개 경로 하나뿐이다(`detection/resume.py` 참조).
            **학습 산출물(`project`)과 다른 경로를 줘라.** 채점·내보내기가 훑는 트리에
            재개 가중치를 두지 않는다.
        run_id: 재개 신원의 일부. 같은 라운드·클라이언트라도 다른 실행이면 이어 가지
            않게 하려면 여기에 실행 식별자를 준다.
        clear_resume_on_success: 정상 완주 시 재개 파일을 지운다. 끄면 다음 실행이 이미
            끝난 라운드를 재개 상태로 오인할 수 있다.
        loader_reseed_per_epoch: epoch 진입마다 로더 셔플 생성기를 `f(seed, epoch)` 으로
            다시 시드한다. **다섯 칸 공통 고정 항목(데이터 순서·증강 난수열)을 바꾸므로
            켜려면 다섯 칸 전부에 켜고 첫 런 착수 전에 확정해야 한다.** 켜면 시드가 실제로
            데이터 순서를 통제하고(숨은 기본값 #9) 재개가 정확해진다.

    Raises:
        ValueError: 5칸 공통 고정 항목을 덮어쓰려 하면 실패한다.
    """
    # 공통 고정 위반은 ultralytics 를 로드하기 **전에** 잡는다. 설정 오류를 알아내는 데
    # optional 의존성이 필요할 이유가 없고, 그래야 이 검사가 어느 환경에서든 돈다.
    if profile not in PROFILES:
        raise ValueError(f"알 수 없는 프로파일: {profile!r}. 허용: {sorted(PROFILES)}")
    fixed = PROFILES[profile]
    overrides: dict[str, Any] = dict(fixed)
    if extra_overrides:
        clashes = sorted(set(extra_overrides) & set(fixed))
        if clashes:
            raise ValueError(
                f"5칸 공통 고정 항목은 칸별로 덮어쓸 수 없다: {clashes}. "
                "값을 바꾸려면 전 칸에 동시에 적용해야 한다."
            )
        overrides.update(extra_overrides)

    seed = derive_seed(base_seed, round_idx, client_idx)
    overrides.update(
        {
            "model": model,
            "data": str(data_yaml),
            "epochs": int(total_epochs),
            "seed": seed,
        }
    )
    if project is not None:
        overrides["project"] = str(Path(project).resolve())
        overrides["name"] = f"r{round_idx:03d}_c{client_idx}"
        overrides["exist_ok"] = True

    from detection.fed_trainer import FedDetectionTrainer  # 지연 import (detection extra)

    # `mlflow` 는 train 인자가 아니라 Ultralytics 전역 settings 키다. 트레이너에 넘기면
    # 알 수 없는 키로 거부되며 CLI 도움말을 뿜고 죽는다(실측). 정책 선언은 FIXED 에 두고
    # 적용은 settings 로 한다 — 자동 MLflow 연동 차단이라는 의도는 동일하다.
    mlflow_off = overrides.pop("mlflow", None)
    if mlflow_off is False:
        from ultralytics import settings as _ul_settings
        _ul_settings.update({"mlflow": False})

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 재개 상태는 트레이너를 만들기 **전에** 읽는다. 신원이 어긋나면 학습을 시작조차
    # 하지 않는 것이 맞다 — 시작한 뒤에 거부하면 GPU 시간만 버린다.
    resume_state = None
    identity = None
    if resume_dir is not None:
        from detection.resume import ResumeCheckpointer, ResumeIdentity, clear_resume, latest_resume

        identity = ResumeIdentity(
            run_id=str(run_id),
            round_idx=int(round_idx),
            client_idx=int(client_idx),
            seed=int(seed),
            total_epochs=int(total_epochs),
            local_epochs=int(local_epochs),
            model=str(model),
            data=str(Path(data_yaml).resolve()),
        )
        resume_state = latest_resume(resume_dir, identity=identity)

    trainer = FedDetectionTrainer(
        overrides=overrides,
        weights_in=weights_in,
        canonical_keys=canonical_keys,
        round_idx=round_idx,
        local_epochs=local_epochs,
        resume_state=resume_state,
        loader_reseed_per_epoch=loader_reseed_per_epoch,
    )
    lr_trace = _LRTrace()
    steps = _StepCounter()
    trainer.add_callback("on_fit_epoch_end", lr_trace)
    trainer.add_callback("on_train_batch_end", steps)
    for event, fn in (callbacks or {}).items():
        trainer.add_callback(event, fn)

    resumed_steps = int(getattr(resume_state, "optimizer_steps", 0) or 0)
    checkpointer = None
    if resume_dir is not None:
        checkpointer = ResumeCheckpointer(
            resume_dir,
            identity=identity,
            canonical_keys=canonical_keys,
            step_counter=steps,
            resumed_epochs=int(getattr(resume_state, "epochs_ran_in_round", 0) or 0),
            resumed_steps=resumed_steps,
        )
        # 예산 콜백보다 **뒤에** 등록한다. 예산이 `stop` 을 켠 마지막 epoch 에서도
        # 체크포인트가 남아야, 그 직후 죽었을 때 라운드를 다시 돌지 않는다.
        trainer.add_callback("on_fit_epoch_end", checkpointer)

    trainer.train()

    keys = list(canonical_keys) if canonical_keys is not None else serialize.canonical_keys(
        trainer._unwrapped_model().state_dict()
    )
    trainer._canonical_keys = keys
    out = trainer.export_weights()

    peak_vram = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    )
    budget = trainer.budget
    if resume_dir is not None and clear_resume_on_success:
        # 정상 완주. 재개 파일은 산출물이 아니므로 남기지 않는다.
        clear_resume(resume_dir)
    return RoundResult(
        ndarrays=out,
        num_examples=int(num_examples),
        round_idx=int(round_idx),
        client_idx=int(client_idx),
        epochs_ran=int(budget.epochs_ran) if budget else int(total_epochs),
        seed=seed,
        # 재개했다면 이전 프로세스의 스텝을 잇는다. 잇지 않으면 회계 매트릭스의
        # R×E=N 감사가 재개한 라운드에서만 미달로 보인다 — 실제로는 밟은 스텝인데.
        optimizer_steps=resumed_steps + steps.n,
        param_l2_norm=serialize.params_l2_norm(out),
        payload_bytes=serialize.payload_nbytes(out),
        injection_digest=list(trainer.injection_digest),
        effective_optimizer=trainer.effective_optimizer(),
        lr_trace=list(lr_trace.trace),
        stopper_calls=list(getattr(trainer.stopper, "calls", [])),
        stopper_class=type(trainer.stopper).__name__ if getattr(trainer, "stopper", None) else "",
        stopper_true_count=getattr(trainer.stopper, "true_count", None),
        budget_fired_at=budget.fired_at_epoch if budget else None,
        peak_vram_gb=peak_vram,
        optimizer_updates=int(getattr(trainer, "n_optimizer_updates", 0)),
        resumed_from_epoch=(resume_state.next_epoch if resume_state is not None else None),
    )
