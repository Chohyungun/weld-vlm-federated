"""연합학습용 검출 트레이너 — Ultralytics `DetectionTrainer` 래퍼 (지정 함정 구간 #1).

## 왜 래핑하는가

Ultralytics는 자체 저장·재개·조기 종료·EMA 로직을 갖고 있고, 그 기본 동작 몇 개가 이
연구의 공정성 주장과 정면으로 어긋난다. 선언만 해 두면 프레임워크가 조용히 다르게
동작하므로, 어긋나는 지점을 코드로 막는다.

| 기본 동작 | 어긋나는 규칙 | 처방 |
|---|---|---|
| `final_eval()`이 `best.pt`를 로드해 검증 | last 채점 (best 선택은 암묵적 조기 종료) | no-op |
| `EarlyStopping(patience)` | 조기 종료 금지 | patience를 epoch 이상 + stopper 스텁 |
| `optimizer='auto'`가 `lr0`·`momentum`을 버리고 AdamW로 교체 | 5칸 공통 고정의 '최적화' | `SGD` 명시 + 실사용 기록 |
| `save_model()`이 EMA 가중치를 저장 | FedAvg는 raw 가중치를 평균한다 | EMA 비활성 + no-op |
| mlflow 자동 연동, 전역 `runs_dir` 재지향 | 로깅 단일 경로, 저장소 위생 | `mlflow=False` + `project` 절대경로 |
| 검증 로더를 무조건 만들고 `workers×2`(cpu 캡→12) 워커를 생성 즉시 spawn·프리페치 | 소비자 없는 로더가 호스트 커밋 ≈13.7 GB·기동 I/O 를 먹는다(14번 §B) | val 로더만 `workers=0` (`get_dataloader`) |

## 라운드 경계를 어떻게 만드는가

`epochs`는 라운드 길이가 아니라 **전역 예산 N**(=100)으로 고정한다. 그래야 warmup과
cosine 스케줄이 중앙집중 학습과 같은 궤적을 그린다. 라운드 r은 `start_epoch = r × E`에서
시작해 로컬 E epoch만 돌고 멈춘다. 멈추는 방식은 조기 종료가 아니라 **예산 도달 정지**이며,
둘을 로그에서 구분할 수 있게 남긴다.

로컬·중앙 칸도 `R=1, E=100`인 퇴화 케이스로 이 트레이너를 통과시킨다. 세 칸이 서로 다른
학습 루프를 타면 "구조 내 동일" 원칙이 문면으로만 남는다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    from ultralytics.models.yolo.detect import DetectionTrainer
except ModuleNotFoundError as exc:  # pragma: no cover - 설치 환경에서는 도달하지 않는다
    raise ModuleNotFoundError(
        "detection extra 가 설치되지 않았다. `uv sync --extra detection` 으로 ultralytics 를 "
        "설치해야 이 모듈을 쓸 수 있다. 검출 학습을 돌리지 않는 환경(채점기·좌표 모듈 등)은 "
        "이 모듈을 import 하지 않아도 된다."
    ) from exc

from detection import serialize

__all__ = ["AppliedStepCounter", "FedDetectionTrainer", "LoaderReseed", "NoEarlyStopping",
           "RoundBudget", "attach_applied_counter"]


class AppliedStepCounter:
    """`optimizer.step()` 이 **실제로 호출된** 횟수. 트레이너를 참조하지 않는 작은 상자다.

    트레이너를 직접 참조하면 optimizer → 훅 → 트레이너 → optimizer 순환이 생겨 라운드마다
    트레이너를 새로 만드는 ④ 에서 가중치 한 벌이 gc 패스까지 더 상주한다(정적 검토 Minor).
    """

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


def attach_applied_counter(optimizer: Any, counter: AppliedStepCounter) -> bool:
    """`optimizer.step` 뒤에 불리는 **공개 훅**을 걸어 실제 적용 수를 센다(34번 §3-1 8-3).

    AMP 의 `torch.amp.GradScaler.step` 은 inf/NaN 기울기를 찾으면 `optimizer.step()` 을 아예
    부르지 않는다(torch 규약). 그래서 트레이너의 `optimizer_step` 호출 수는 **시도 수**이고,
    적용 수는 `step` 이 실제로 불린 횟수다.

    `optimizer.step` 을 감싸지 않는 이유: torch 의 `LRScheduler.__init__` 이 같은 자리를
    `step_fn.__func__` 로 감싸므로, 우리가 먼저 감싸면 그 뒤 스케줄러 생성이 AttributeError 로
    죽는다(실측). 공개 훅은 순서와 무관하고 `optimizer.step` 을 건드리지 않으므로 학습 산술과도
    무관하다 — 바이트 동일은 `tests/test_update_counters.py` 가 고정한다.

    같은 카운터가 이미 걸린 옵티마이저면 아무것도 하지 않고 False. **옵티마이저가 새로
    만들어지면**(상류 OOM 자동 축소가 `_build_train_pipeline` 에서 그렇게 한다) 다음 호출에서
    다시 걸리고, 카운터는 그대로라 계수가 이어진다.
    """
    if getattr(optimizer, "_fed_applied_counter", None) is counter:
        return False

    def _post(_opt: Any, _args: Any, _kwargs: Any) -> None:
        counter.n += 1

    optimizer.register_step_post_hook(_post)
    optimizer._fed_applied_counter = counter
    return True


class NoEarlyStopping:
    """`EarlyStopping` 자리에 끼우는 스텁. 절대 중단시키지 않고 호출 이력만 남긴다.

    `patience`를 크게 잡는 것만으로도 조기 종료는 막히지만, 그것은 "설정이 그러하다"는
    간접 증거일 뿐이다. 이 스텁은 **중단이 구조적으로 불가능함**을 보장하면서, 동시에
    호출 이력을 남겨 사후 감사에서 "조기 종료가 걸리지 않았다"를 증명하게 해 준다.

    `possible_stop` 속성이 반드시 있어야 한다 — 학습 루프의 검증 게이트
    (`self.args.val or final_epoch or self.stopper.possible_stop or self.stop`)가 이 값을
    참조하므로, 없으면 AttributeError로 학습이 죽는다.
    """

    def __init__(self) -> None:
        self.possible_stop: bool = False
        self.best_fitness = 0.0
        self.calls: list[tuple[int, float | None]] = []
        #: 참을 돌려준 횟수. 이 스텁에서는 0 이 유지되지만 **회계는 이 값을 읽는다** —
        #: 이전 판은 회계 쪽에 0 을 리터럴로 박아 검사가 공허했다(74번 감사 P9).
        #: 값이 어디서 오는지가 요점이다. 진짜 stopper 가 끼워지면 `stopper_class` 가
        #: 그것을 잡는다.
        self.true_count: int = 0

    def __call__(self, epoch: int, fitness: Any = None) -> bool:
        self.calls.append((int(epoch), None if fitness is None else float(fitness)))
        stop = False
        self.true_count += int(stop)
        return stop


class RoundBudget:
    """로컬 예산(E epoch)에 도달하면 학습 루프를 멈추는 콜백.

    `on_fit_epoch_end` 직후에 `if self.stop: break`가 있으므로 즉시 종료된다.
    조기 종료와 구분하기 위해 발화 시점을 따로 기록한다 — 사후 감사에서 `stop`이 켜진
    이유가 예산인지 조기 종료인지 대조하는 근거다.
    """

    def __init__(self, local_epochs: int, resumed_epochs: int = 0) -> None:
        self.local_epochs = int(local_epochs)
        #: 재개 전에 이 라운드에서 이미 돈 epoch 수. 예산은 라운드 전체를 세야 하므로
        #: 이 값을 더하지 않으면 재개한 라운드가 E epoch 을 **다시** 돌아 학습량 등가가 깨진다.
        self.resumed_epochs = int(resumed_epochs)
        self.fired_at_epoch: int | None = None
        self.epochs_ran: int = int(resumed_epochs)

    def __call__(self, trainer: Any) -> None:
        self.epochs_ran = self.resumed_epochs + (trainer.epoch - trainer.start_epoch + 1)
        if self.epochs_ran >= self.local_epochs:
            trainer.stop = True
            if self.fired_at_epoch is None:
                self.fired_at_epoch = int(trainer.epoch)


class LoaderReseed:
    """epoch 진입마다 학습 로더의 셔플 생성기를 `f(seed, epoch)` 으로 다시 시드한다.

    ## 이것이 고치는 두 가지

    1. **시드가 데이터 순서를 통제하게 된다.** Ultralytics 는 로더 생성기를 상수
       `6148914691236517205 + RANK` 로 시드하므로, 기본 상태에서는 `args.seed` 가 셔플
       순열에도 워커 증강 난수열에도 닿지 않는다(숨은 기본값 #9). 시드 3세트가 무엇을
       흔드는지가 헤드 초기화로 좁아진다.
    2. 로더 상태를 `(seed, epoch)`에서 다시 계산한다. 데이터 순서와 워커 시드를
       재현하지만, 저장하지 않는 누적 gradient 등 때문에 학습 궤적의 동등성은 보장하지 않는다.

    ## 대가

    `reset()` 이 워커를 접었다 다시 띄운다. Windows spawn 이라 epoch 당 비용이 붙는다.
    실측치는 70번에 있다.

    ## 왜 기본값이 아닌가

    데이터 순서와 증강 난수열은 **다섯 칸 공통 고정 항목**이다. 켜면 다섯 칸 전부에 켜야
    하고 첫 런 착수 전에 확정해야 한다. 트랙이 단독으로 바꿀 수 있는 값이 아니다.
    """

    MIX = 6364136223846793005
    _M64 = (1 << 64) - 1

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)
        self.reseeds: list[tuple[int, int]] = []

    @classmethod
    def _splitmix64(cls, x: int) -> int:
        """확산 함수. `base*MIX + epoch` 을 그대로 쓰면 인접 epoch 의 시드가 **1만 차이난다.**

        난수 생성기가 그 차이를 알아서 흩어 주기를 기대하지 않는다 — 기대해야 할 이유가
        없고, 어긋나도 지표에 흔적이 남지 않는 종류의 문제다. 여기서 흩어 놓는다.
        """
        m = cls._M64
        z = (x + 0x9E3779B97F4A7C15) & m
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & m
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & m
        return (z ^ (z >> 31)) & m

    def seed_for(self, epoch: int) -> int:
        return self._splitmix64((self.base_seed * self.MIX + int(epoch)) & self._M64) % (2**63 - 1)

    def __call__(self, trainer: Any) -> None:
        loader = getattr(trainer, "train_loader", None)
        gen = getattr(loader, "generator", None)
        if gen is None:
            return
        s = self.seed_for(trainer.epoch)
        gen.manual_seed(s)
        loader.reset()
        self.reseeds.append((int(trainer.epoch), s))


class FedDetectionTrainer(DetectionTrainer):
    """라운드 단위로 호출되는 검출 트레이너.

    내부 접촉점은 아래 여섯으로 한정한다. 늘리면 Ultralytics 버전 변동에 그만큼 취약해진다.
      1. `_setup_train`  — 가중치 주입, 전역 epoch 오프셋, 재개 상태 적용, mosaic 재적용,
                           EMA off, stopper 교체
      2. `validate`      — no-op
      3. `final_eval`    — no-op (stock은 best.pt를 로드한다)
      4. `save_model`    — no-op (stock은 EMA 가중치를 저장한다)
      5. `optimizer_step` — 갱신 **시도** 수 계측(적용 수는 optimizer.step 공개 훅, 학습 동작 무변경)
      6. `get_dataloader` — 검증 로더만 `workers=0` (소비되지 않는 로더의 워커 12개 제거, 14번 §B-3)
    """

    def __init__(
        self,
        cfg: Any = None,
        overrides: dict | None = None,
        _callbacks: dict | None = None,
        *,
        weights_in: Sequence[np.ndarray] | None = None,
        canonical_keys: Sequence[str] | None = None,
        round_idx: int = 0,
        local_epochs: int | None = None,
        resume_state: Any = None,
        loader_reseed_per_epoch: bool = False,
        loader_seed: int | None = None,
    ) -> None:
        self._weights_in = weights_in
        self._canonical_keys = list(canonical_keys) if canonical_keys is not None else None
        self._round_idx = int(round_idx)
        self._local_epochs = local_epochs
        #: `detection.resume.ResumeState` 또는 None. 있으면 `_setup_train` 에서 적용한다.
        self._resume_state = resume_state
        self._loader_reseed_per_epoch = bool(loader_reseed_per_epoch)
        self._loader_seed = loader_seed
        self.injection_digest: list[float] = []
        self.budget: RoundBudget | None = None
        self.loader_reseed: LoaderReseed | None = None
        #: `optimizer_step` 호출(= 갱신 **시도**) 횟수. 배치 수와 다르다 — `optimizer_step` 주석 참조.
        #: 8-3(34번 §3-1): 이름은 재개 payload 키·세 시드 meta·원자 로그 지표와의 뒤호환을 위해
        #: 그대로 두고 의미는 **시도 수**다. 실제 적용 수는 `n_optimizer_updates_applied`.
        self.n_optimizer_updates: int = 0
        #: 실제 적용 수의 재개분 baseline(이전 프로세스 몫). `round_runner` 가 넣는다.
        self.applied_base: int = 0
        #: 이번 프로세스에서 실제로 적용된 횟수를 세는 상자(훅이 올린다).
        self._applied = AppliedStepCounter()
        #: 검증 DataLoader 의 실효 워커 수 증빙. None = 아직 안 만듦(또는 미계측), 0 = 접촉점 6 적용.
        #: "설정이 아니라 실제로 돈 값"을 남긴다 — `RoundResult.val_loader_workers` 로 나간다.
        self.val_loader_workers: int | None = None

        if cfg is None:
            from ultralytics.cfg import get_cfg  # 지역 import — 모듈 로드를 가볍게 유지
            cfg = get_cfg()
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        if self._local_epochs is not None:
            resumed = int(getattr(self._resume_state, "epochs_ran_in_round", 0) or 0)
            if resumed >= self._local_epochs:
                raise ValueError("라운드 예산이 이미 완료된 체크포인트다. 추가 학습은 허용하지 않는다.")
            self.budget = RoundBudget(self._local_epochs, resumed_epochs=resumed)
            # 인스턴스 로컬 등록. 전역 default_callbacks 를 건드리면 라운드마다 콜백이 쌓인다.
            self.add_callback("on_fit_epoch_end", self.budget)

        if self._loader_reseed_per_epoch:
            # `on_train_epoch_start` 는 학습 루프에서 `enumerate(self.train_loader)` 와
            # close_mosaic 재적용보다 **앞**에 발화한다. 그래서 여기서 다시 시드하면
            # 그 epoch 이 쓰는 순열이 전부 새 시드에서 나온다.
            self.loader_reseed = LoaderReseed(
                int(self.args.seed) if self._loader_seed is None else int(self._loader_seed)
            )
            self.add_callback("on_train_epoch_start", self.loader_reseed)

    # -- 접촉점 1 -----------------------------------------------------------
    def _setup_train(self) -> None:
        super()._setup_train()

        # (a) 서버 가중치 주입. strict=True — 키가 하나라도 어긋나면 즉시 실패한다.
        if self._weights_in is not None:
            if self._canonical_keys is None:
                raise ValueError("weights_in을 주려면 canonical_keys도 함께 줘야 한다.")
            target = self._unwrapped_model()
            ref = target.state_dict()
            serialize.assert_compatible(self._weights_in, self._canonical_keys, ref)
            sd = serialize.ndarrays_to_state_dict(self._weights_in, self._canonical_keys, ref)
            target.load_state_dict(sd, strict=True)
            # (b) 주입 증빙. 서버가 보낸 값과 대조해 주입이 무력화되지 않았음을 확인한다.
            after = serialize.state_dict_to_ndarrays(target.state_dict(), self._canonical_keys)
            self.injection_digest = serialize.tensor_digest(after)

        # (c) 전역 epoch 오프셋. epochs는 전역 예산 N이고 start_epoch만 라운드마다 움직인다.
        if self._local_epochs is not None:
            self.start_epoch = self._round_idx * int(self._local_epochs)

        # (c') 재개 상태 적용. 라운드 중간에서 죽은 것이므로 그 시점의 가중치·옵티마이저·
        #      RNG 가 정본이고, 서버 파라미터(a)를 덮어쓰는 것이 맞다. `start_epoch` 도
        #      여기서 다시 움직이므로 (d)(e)보다 반드시 앞이어야 한다.
        if self._resume_state is not None:
            from detection.resume import apply_resume

            apply_resume(self, self._resume_state)
            print(f"[resume] epoch {self._resume_state.epoch_done} 까지 완료된 상태에서 "
                  f"epoch {self.start_epoch} 부터 이어 간다 "
                  f"({self._resume_state.path.name})", flush=True)

        # (d) 스케줄러 재개. super()가 start_epoch=0 기준으로 이미 설정했으므로 다시 맞춘다.
        #     이 한 줄이 연합 칸의 학습률 궤적을 중앙집중 칸과 같게 만든다.
        self.scheduler.last_epoch = self.start_epoch - 1

        # (e) mosaic 종료 재적용. 학습 루프의 조건이 `epoch == epochs - close_mosaic` 등호라
        #     라운드 재진입으로 그 시점을 건너뛰면 영영 발화하지 않는다.
        if self.start_epoch >= self.epochs - self.args.close_mosaic:
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        # (f) EMA 비활성. FedAvg가 평균하는 것은 raw 가중치다.
        if self.ema is not None:
            self.ema.enabled = False

        # (g) 조기 종료를 구조적으로 불가능하게 만든다.
        self.stopper = NoEarlyStopping()
        self.stop = False

        # (h) 로더 셔플 순열 복원. (e)의 mosaic 재적용이 반복자를 다시 만들므로 반드시
        #     그 뒤여야 한다. 이 한 줄이 "재개한 런"과 "중단 없이 완주한 런"의 데이터
        #     순서를 같게 만든다.
        #     `loader_reseed_per_epoch` 가 켜져 있으면 복원하지 않는다 — epoch 진입에서
        #     `(seed, epoch)` 으로 다시 시드하므로 되돌린 상태가 곧 덮인다. 그쪽이 더 정확하다
        #     (되돌리기는 base seed draw 한 번만큼 어긋난다).
        if self._resume_state is not None and not self._loader_reseed_per_epoch:
            from detection.resume import restore_loader_generator

            ok = restore_loader_generator(self, self._resume_state.payload.get("loader_generator"))
            if not ok:
                print("[resume] 로더 셔플 상태를 복원하지 못했다 — 데이터 순서가 갈라진다",
                      flush=True)

    # -- 접촉점 2·3·4 -------------------------------------------------------
    def validate(self) -> tuple[dict, None]:
        """no-op. 클라이언트는 검증하지 않는다.

        `val=False`로 두어도 전역 마지막 epoch에서 `final_epoch`이 참이 되어 검증 게이트가
        열리고, 예산 정지로 `self.stop`이 켜져도 마찬가지다. 라운드별 지표는 서버가 val로
        산출하므로 여기서 도는 검증은 시간만 쓴다.
        """
        # Ultralytics는 이 반환값으로 self.metrics를 교체한다. 빈 dict를 반환하면
        # 마지막 epoch만 CSV 열 수가 줄어 LR이 검증 지표 열에 들어간다.
        return dict.fromkeys(getattr(self, "metrics", {}), float("nan")), None

    def save_metrics(self, metrics: dict) -> None:
        """검증 미실행을 NaN으로 표시하고 학습 손실·LR은 원래 값으로 기록한다."""
        recorded = {
            key: (float("nan") if key.startswith(("val/", "metrics/")) else value)
            for key, value in metrics.items()
        }
        super().save_metrics(recorded)

    def final_eval(self) -> None:
        """no-op. stock은 `best.pt`를 로드해 검증한다 — best 선택 경로 자체를 제거한다."""
        return None

    @property
    def n_optimizer_updates_applied(self) -> int:
        """실제로 `optimizer.step()` 이 호출된 총 횟수(재개분 + 이번 프로세스분).

        GradScaler 가 inf/NaN 에서 건너뛴 시도는 빠진다. 시도 수는 `n_optimizer_updates`.
        """
        return int(self.applied_base) + int(self._applied.n)

    # -- 접촉점 5 -------------------------------------------------------------
    def optimizer_step(self) -> None:
        """실제 갱신 횟수를 센다. 학습 동작은 그대로 둔다.

        `on_train_batch_end` 콜백이 세는 것은 **배치 수**이지 갱신 횟수가 아니다.
        Ultralytics 는 `nbs=64` 기준으로 `accumulate = round(nbs / batch)` 만큼 누적하므로
        batch 32 면 2배치에 1회, batch 2 면 32배치에 1회 갱신한다. 논문에 싣는 "총 갱신
        횟수"는 이쪽이다.

        이 값이 따로 필요한 두 번째 이유가 있다. **연합 칸은 라운드마다 트레이너를 새로
        만들고, 누적 카운터(`last_opt_step`)는 `_do_train` 의 지역 변수라 라운드 경계에서
        `-1` 로 리셋된다.** 그래서 연합은 라운드 첫 배치에서 즉시 갱신하고, 라운드 끝에
        남은 누적분은 갱신 없이 버려진다. 중앙집중 칸에는 그 경계가 없다. 크기는
        `누적수 − 1` 배치/라운드 이하이지만, **선언에 없는 칸 간 비대칭이므로 수치로
        남긴다.**

        8-3(34번 §3-1): 여기서 세는 값은 갱신 **시도 수**다. 상류는 `scaler.step(optimizer)` 를
        부르고 GradScaler 는 inf/NaN 기울기에서 `optimizer.step()` 을 건너뛰므로, 호출 수와
        적용 수가 다르다. 적용 수는 공개 훅(`attach_applied_counter`)이 센다 — 매 호출에서
        멱등하게 다시 걸어, 상류가 옵티마이저를 새로 만들어도 계수가 끊기지 않는다.
        **논문의 "총 갱신 횟수" 는 적용 수(`optimizer_updates_applied`)를 쓰고, 그 값이 없는
        실행(세 시드 본실험·④)은 시도 수임을 각주로 밝힌다.**
        """
        attach_applied_counter(self.optimizer, self._applied)
        super().optimizer_step()
        self.n_optimizer_updates += 1

    def save_model(self) -> bool:
        """no-op. stock 체크포인트는 EMA 가중치를 담는다. 저장은 서버가 맡는다.

        `False`를 돌려주어 `on_model_save` 콜백도 돌지 않게 한다.
        """
        return False

    # -- 접촉점 6 -------------------------------------------------------------
    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """검증 로더만 `workers=0` 으로 만든다. 학습 로더는 stock 그대로다 (14번 §B-3, 15번 G3).

        stock 은 `_build_train_pipeline` 이 검증 로더를 **무조건** 만들고(`trainer.py:291`),
        `workers*2`(=16, `os.cpu_count()` 캡으로 실효 12)·`batch*2` 로 `InfiniteDataLoader` 를
        생성하는 순간 워커 spawn + `prefetch_factor=4` 프리페치가 시작된다(`build.py:75·376`).
        그런데 이 래퍼는 `validate`·`final_eval` 이 no-op 이라 그 로더를 **한 번도 소비하지
        않는다**(`trainer.py` 의 `test_loader` 참조는 대입 L291 과 close L646 뿐). 워커 12개 =
        호스트 커밋 ≈ 13.7 GB 와 클라이언트 기동마다 검증 이미지 3,072장 프리페치 I/O 가 순수
        낭비였고, 09-07 06:12 의 SuperLink 하트비트 만료도 그 I/O 폭주 직후였다(15번 §1).

        등가: 데이터셋 생성(`build_dataset`)은 stock 그대로 둔다 — `get_img_files` →
        `check_file_speeds` 가 전역 `random` 을 1회 소비하므로(`base.py:184`, `data/utils.py:86`)
        생략하면 주 프로세스 RNG 궤적이 갈라진다. 로더의 `_base_seed` 는 로더 자신의 generator
        에서만 뽑히고(torch `dataloader.py`), 학습 로더는 검증 로더보다 먼저 생성·반복자가 확정된다
        (`trainer.py:281`). 따라서 학습 배치열·워커 시드·모델/옵티마이저 상태는 B 전후 bit 동일해야
        하며, 판정은 재실행 라운드 1 오라클(`global_r001.npz` sha256 `5c0378a1…`)이 한다.
        """
        if mode != "val":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        from ultralytics.data import build_dataloader
        from ultralytics.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(rank):  # detect/train.py:91-92 와 동일
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        loader = build_dataloader(
            dataset,
            batch=batch_size,
            workers=0,  # stock: self.args.workers * 2
            shuffle=False,  # stock: mode == "train"
            rank=rank,
            drop_last=False,  # stock: self.args.compile and mode == "train"
            device=self.device,
        )
        nw = getattr(loader, "num_workers", None)
        if nw != 0:
            raise RuntimeError(
                f"검증 로더 워커 {nw} != 0 — 상류 build_dataloader 배선이 바뀌었다(B 전제 위반, 14번 §B-3)"
            )
        self.val_loader_workers = int(nw)
        return loader

    # -- 보조 ---------------------------------------------------------------
    def _unwrapped_model(self):
        """DDP 래핑을 벗긴 실제 `nn.Module`."""
        return getattr(self.model, "module", self.model)

    def export_weights(self) -> list[np.ndarray]:
        """학습 후 raw 가중치를 정본 키 순서로 내보낸다."""
        if self._canonical_keys is None:
            raise ValueError("canonical_keys가 없다.")
        return serialize.state_dict_to_ndarrays(
            self._unwrapped_model().state_dict(), self._canonical_keys
        )

    def effective_optimizer(self) -> dict[str, Any]:
        """실제로 쓰인 옵티마이저와 학습률.

        `optimizer='auto'`는 명시한 `lr0`·`momentum`을 버리고 AdamW로 갈아치우면서 경고
        한 줄만 남긴다. 설정에 무엇을 적었는지가 아니라 **무엇이 실제로 돌았는지**를
        회계 매트릭스에 남겨야 5칸 공통 고정이 지켜졌음을 사후에 증명할 수 있다.
        """
        opt = self.optimizer
        groups = getattr(opt, "param_groups", [])
        first = groups[0] if groups else {}
        return {
            "optimizer": type(opt).__name__,
            "lr": float(first.get("lr", float("nan"))),
            "momentum": float(first.get("momentum", first.get("betas", [float("nan")])[0]))
            if ("momentum" in first or "betas" in first)
            else float("nan"),
            "arg_optimizer": str(self.args.optimizer),
            "arg_lr0": float(self.args.lr0),
            "arg_momentum": float(self.args.momentum),
        }
