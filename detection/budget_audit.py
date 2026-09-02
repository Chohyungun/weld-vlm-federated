"""클라이언트 × 라운드 회계 매트릭스 — 학습량 등가의 실측 증명.

## 왜 머지 차단 조건인가

Flower `FedAvg`의 기본값 `accept_failures=True`는 클라이언트 하나가 라운드 중 실패해도
나머지로 집계를 진행한다. 그러면 그 라운드의 실효 학습량이 줄어드는데 **지표에는 아무
흔적이 남지 않는다.** `R × E = N` 등가는 이 연구의 공정성 주장 전체를 떠받치는 전제이므로,
조용히 깨지는 이 경로를 막아야 한다.

막는 방법은 두 겹이다. 서버는 `accept_failures=False`로 실패를 라운드 중단으로 만들고,
이 모듈은 **셀의 부재 자체를 실패로 정의**한다. 전자만 있으면 "실패가 없었다"를 신뢰에
의존하게 되지만, 후자가 있으면 R × 클라이언트 수 전 셀이 채워졌는지를 사후에 센다.

## 조기 종료 부재는 무엇으로 증명되는가 (74번 감사 P9 정정)

이전 판은 `stopper_true_count=0` 을 **리터럴로 박아 놓고** 그 상수를 검사했다. 어떤
경우에도 통과하는 검사였다. 공허한 검사를 통과 근거로 남겨 두는 것이 검사가 없는 것보다
나쁘다 — 회계표에 초록불이 하나 늘지만 정보량은 0 이고, 읽는 사람은 무언가 확인됐다고
믿는다.

지금은 셋으로 나눠 적는다.

1. **`stopper_class`** — 스텁(`NoEarlyStopping`)이 실제로 끼워졌는가. 교체가 실패하면
   Ultralytics 의 진짜 `EarlyStopping` 이 남으므로 여기서 잡힌다. **실패 가능한 검사다.**
2. **`stopper_true_count`** — 스텁이 남긴 호출 이력에서 참 판정 횟수. 계측이 없는 경로
   (통합형은 Ultralytics 를 쓰지 않아 stopper 자체가 없다)에서는 `None` 이고, 그 경우
   **통과가 아니라 "이 셀의 근거는 다른 곳"으로 보고서에 적힌다.**
3. **대체 증거** — 조기 종료 부재의 실질 증거는 회계가 아니라 `results.csv` 행 수와
   `optimizer_steps` 다. 검사 (2)(3) 의 `epochs_ran == E`·`sum == N` 이 그 회계 쪽
   대응물이며, 이 둘이 실제로 값을 하는 검사다.

## 실측값과 재구성값을 구분한다

`value_source` 가 그 표식이다. ⑦ 재개 경로는 회계 매트릭스가 인메모리라 중단 시 앞
라운드 셀이 사라지고, 원자 로그에서 되살린다. 되살린 값은 실측이 아니다 — 로그에 없는
필드(`epochs_ran` 등)는 상수로 채워진다. 표식이 없으면 산출물을 읽는 사람이 둘을
구분할 수 없다.

## 실사용 optimizer·lr 을 왜 기록하는가

Ultralytics의 `optimizer='auto'`는 명시한 `lr0`·`momentum`을 **경고 한 줄만 남기고 버린
뒤 AdamW로 갈아치운다.** 설정 파일에 SGD라고 적혀 있어도 실제로는 다른 옵티마이저가 돌 수
있다는 뜻이고, 그 순간 5칸 공통 고정의 '최적화' 항목이 깨진다. 설정에 무엇을 적었는지가
아니라 무엇이 실제로 돌았는지를 남겨야 사후에 증명이 된다.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

__all__ = ["AccountingCell", "AccountingMatrix", "AuditReport"]

#: 조기 종료를 구조적으로 낼 수 없는 stopper 구현. 여기 없는 클래스가 끼워져 있으면
#: 스텁 교체가 실패한 것이고, 그것은 회계 실패다.
_NO_STOP_CLASSES = frozenset({"NoEarlyStopping"})

_CSV_COLUMNS = [
    "round_idx",
    "client_idx",
    "participated",
    "epochs_ran",
    "optimizer_steps",
    "num_examples",
    "seed",
    "param_l2_norm",
    "payload_bytes",
    # 실사용 최적화 설정 — 설정값이 아니라 실제로 돌아간 값이다
    "optimizer",
    "lr",
    "momentum",
    "arg_optimizer",
    "arg_lr0",
    "arg_momentum",
    "budget_fired_at",
    # 조기 종료 계측 3종. `stopper_true_count` 가 빈칸이면 "0 회 관측"이 아니라
    # **"이 경로에는 stopper 계측이 없다"** 는 뜻이다. 74번 P9 정정.
    "stopper_class",
    "stopper_true_count",
    "stopper_calls",
    # 이 행의 값이 실측인가 재구성인가. 섞이면 산출물을 인용할 수 없다.
    "value_source",
    # 배치 수(`optimizer_steps`)와 실제 갱신 횟수는 다르다 — Ultralytics 가 nbs=64 기준으로
    # 누적한다(숨은 기본값 #10). 논문의 "총 갱신 횟수"는 아래 컬럼이다.
    "optimizer_updates",
    # 재개해서 이어 간 칸인가. 이어 간 런은 무중단 런과 다른 궤적을 그린다.
    "resumed_from_epoch",
    # FedAvg 가 실제로 쓴 가중과 그 단위. 총괄 판정 2(2026-09-02)로 통합형은 감독 토큰
    # 총합이 됐고, 검출은 표본 수다. **어느 단위로 잰 값인지 산출물이 말해야** RQ3 을
    # 해석할 수 있다 — 단위가 바뀌면 C3 비중이 1.52배 움직인다.
    "fedavg_weight",
    "fedavg_weight_unit",
    "supervised_tokens",
]


@dataclass
class AccountingCell:
    """(라운드, 클라이언트) 한 칸의 실측 기록."""

    round_idx: int
    client_idx: int
    epochs_ran: int
    optimizer_steps: int
    num_examples: int
    seed: int
    param_l2_norm: float = 0.0
    payload_bytes: int = 0
    optimizer: str = ""
    lr: float = float("nan")
    momentum: float = float("nan")
    arg_optimizer: str = ""
    arg_lr0: float = float("nan")
    arg_momentum: float = float("nan")
    budget_fired_at: int | None = None
    #: 실제로 끼워진 stopper 의 클래스 이름. 빈 문자열이면 계측이 없다는 뜻이다.
    stopper_class: str = ""
    #: 스텁이 참을 돌려준 횟수. **`None` 은 "0 회"가 아니라 "계측 없음"이다.**
    #: 기본값을 0 으로 두면 아무도 재지 않은 셀이 조용히 통과한다(74번 P9).
    stopper_true_count: int | None = None
    #: stopper 가 호출된 횟수. epoch 수와 맞아야 학습 루프가 그 게이트를 실제로 지났다.
    stopper_calls: int | None = None
    optimizer_updates: int = 0
    resumed_from_epoch: int | None = None
    participated: bool = True
    #: "measured" | "reconstructed". 재구성 셀은 감사 보고서에 따로 센다.
    value_source: str = "measured"
    #: FedAvg 가 실제로 쓴 가중. **`None` 은 "0" 이 아니라 "미기록" 이다** — 0 을 기본값으로
    #: 두면 아무도 기록하지 않은 셀이 "가중 0" 으로 읽히고, 그것이 P9 와 같은 종류의 혼동이다.
    fedavg_weight: float | None = None
    #: "num_examples"(검출) | "supervised_tokens"(통합형, 총괄 판정 2). 빈 문자열은 미기록.
    fedavg_weight_unit: str = ""
    #: 감독 토큰 총합. 통합형에서는 가중 그 자체이고, 검출에서는 0 이다.
    supervised_tokens: int = 0

    @classmethod
    def from_round_result(cls, result: Any) -> "AccountingCell":
        eff = dict(getattr(result, "effective_optimizer", {}) or {})
        return cls(
            round_idx=result.round_idx,
            client_idx=result.client_idx,
            epochs_ran=result.epochs_ran,
            optimizer_steps=result.optimizer_steps,
            num_examples=result.num_examples,
            seed=result.seed,
            param_l2_norm=result.param_l2_norm,
            payload_bytes=result.payload_bytes,
            optimizer=str(eff.get("optimizer", "")),
            lr=float(eff.get("lr", float("nan"))),
            momentum=float(eff.get("momentum", float("nan"))),
            arg_optimizer=str(eff.get("arg_optimizer", "")),
            arg_lr0=float(eff.get("arg_lr0", float("nan"))),
            arg_momentum=float(eff.get("arg_momentum", float("nan"))),
            budget_fired_at=result.budget_fired_at,
            # 실물을 읽는다. 이전 판은 여기에 0 을 박아 검사 (4)를 공허하게 만들었다.
            stopper_class=str(getattr(result, "stopper_class", "") or ""),
            stopper_true_count=getattr(result, "stopper_true_count", None),
            stopper_calls=(len(result.stopper_calls)
                           if getattr(result, "stopper_calls", None) is not None else None),
            optimizer_updates=int(getattr(result, "optimizer_updates", 0) or 0),
            resumed_from_epoch=getattr(result, "resumed_from_epoch", None),
            fedavg_weight=float(result.num_examples),
            fedavg_weight_unit="num_examples",
        )


@dataclass
class AuditReport:
    """감사 결과. `ok`가 거짓이면 그 run 은 무효이며 채점하지 않는다."""

    ok: bool
    failures: list[str] = field(default_factory=list)
    total_epochs_by_client: dict[int, int] = field(default_factory=dict)
    total_optimizer_steps: int = 0
    #: 실제 갱신 횟수 합. 배치 수와 다르다(숨은 기본값 #10).
    total_optimizer_updates: int = 0
    #: 재개해서 이어 간 셀 목록. **실패가 아니다** — 재개는 정당한 복구 수단이다.
    #: 다만 이어 간 런은 무중단 런과 다른 궤적을 그리므로 보고서에 드러나 있어야 한다.
    resumed_cells: list[list[int]] = field(default_factory=list)
    #: 값이 실측이 아니라 재구성인 셀. 이것도 실패가 아니지만 인용 전에 알아야 한다.
    reconstructed_cells: list[list[int]] = field(default_factory=list)
    #: **통과도 실패도 아닌 것.** 검사가 원리적으로 적용되지 않은 항목을 여기 적는다.
    #: 이 목록이 비어 있지 않다면 그 셀의 근거는 회계가 아니라 다른 산출물에 있다.
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class AccountingMatrix:
    """라운드 × 클라이언트 셀을 모아 학습량 등가를 실측 검증한다."""

    def __init__(self, num_rounds: int, client_ids: Iterable[int], local_epochs: int, total_epochs: int) -> None:
        self.num_rounds = int(num_rounds)
        self.client_ids = sorted(int(c) for c in client_ids)
        self.local_epochs = int(local_epochs)
        self.total_epochs = int(total_epochs)
        self.cells: dict[tuple[int, int], AccountingCell] = {}

    def record(self, cell: AccountingCell) -> None:
        key = (cell.round_idx, cell.client_idx)
        if key in self.cells:
            raise ValueError(f"이미 기록된 셀이다: 라운드 {key[0]}, 클라이언트 {key[1]}")
        self.cells[key] = cell

    def record_result(self, result: Any) -> None:
        self.record(AccountingCell.from_round_result(result))

    # -- 검증 ---------------------------------------------------------------
    def audit(self) -> AuditReport:
        failures: list[str] = []
        notes: list[str] = []

        # (1) 셀의 부재 자체가 실패다 — 실패한 클라이언트는 로그에 아예 없을 수 있다.
        missing = [
            (r, c)
            for r in range(self.num_rounds)
            for c in self.client_ids
            if (r, c) not in self.cells
        ]
        if missing:
            failures.append(
                f"빈 셀 {len(missing)}개 — 라운드 중 이탈한 클라이언트가 있다: {missing[:6]}"
            )

        # (2) 각 셀이 예산만큼 돌았는가
        for (r, c), cell in sorted(self.cells.items()):
            if cell.epochs_ran != self.local_epochs:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: epochs_ran={cell.epochs_ran} != E={self.local_epochs}"
                )

        # (3) 클라이언트별 총 노출이 전역 예산 N 과 같은가 (R × E = N)
        totals: dict[int, int] = {c: 0 for c in self.client_ids}
        for (_, c), cell in self.cells.items():
            totals[c] = totals.get(c, 0) + cell.epochs_ran
        for c, t in sorted(totals.items()):
            if t != self.total_epochs:
                failures.append(f"클라이언트 {c}: 총 epoch {t} != N={self.total_epochs}")

        # (4) 조기 종료가 실제로 걸리지 않았는가 — 74번 P9 정정본.
        #
        #     이전 판은 리터럴 0 을 읽어 **어떤 경우에도 통과했다.** 지금은 실패할 수
        #     있는 것만 검사하고, 검사가 적용되지 않는 셀은 통과시키는 대신 `notes` 에
        #     적어 근거가 다른 곳에 있음을 산출물에 남긴다.
        uninstrumented: list[tuple[int, int]] = []
        for (r, c), cell in sorted(self.cells.items()):
            if cell.stopper_true_count:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: 조기 종료 판정이 {cell.stopper_true_count}회 발생"
                )
            if cell.stopper_class and cell.stopper_class not in _NO_STOP_CLASSES:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: stopper 가 스텁이 아니다 "
                    f"({cell.stopper_class}) — 조기 종료가 구조적으로 가능한 상태다"
                )
            # **재개한 셀은 이 프로세스가 돈 epoch 만 stopper 를 지난다.** `epochs_ran` 은
            # 라운드 누적치라 그대로 비교하면 재개 런이 무조건 실패한다 — §4-6 게이트
            # 실행에서 실제로 그렇게 났다(33 epoch 누적 대 이번 프로세스 2회).
            ran_here = cell.epochs_ran - int(cell.resumed_from_epoch or 0)
            if cell.stopper_calls is not None and cell.stopper_calls < ran_here:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: stopper 호출 {cell.stopper_calls}회 < "
                    f"이 프로세스가 돈 epoch {ran_here}회 — 학습 루프가 그 게이트를 "
                    f"다 지나지 않았다 (누적 epochs_ran={cell.epochs_ran}, "
                    f"resumed_from={cell.resumed_from_epoch})"
                )
            if cell.stopper_true_count is None and not cell.stopper_class:
                uninstrumented.append((r, c))
        if uninstrumented:
            notes.append(
                f"조기 종료 계측이 없는 셀 {len(uninstrumented)}개 {uninstrumented[:6]} — "
                "이 셀들에 대해 검사 (4)는 통과가 아니라 **미적용**이다. 조기 종료 부재의 "
                "증거는 results.csv 행 수와 optimizer_steps(및 검사 (2)(3)의 "
                "epochs_ran==E · 합계==N)에 있다."
            )

        # (4'') 가중 단위 일관성 — 같은 run 안에서 단위가 갈리면 집계가 두 목적함수를
        #      섞은 것이다. 총괄 판정 2 가 통합형을 감독 토큰으로 옮겼으므로 **칸 안에서**
        #      단위가 하나인지 여기서 지킨다.
        units = {c.fedavg_weight_unit for c in self.cells.values() if c.fedavg_weight_unit}
        if len(units) > 1:
            failures.append(f"FedAvg 가중 단위가 셀마다 다르다: {sorted(units)}")
        zero_w = sorted([r, c] for (r, c), cell in self.cells.items()
                        if cell.fedavg_weight is not None and cell.fedavg_weight <= 0)
        if zero_w:
            failures.append(
                f"가중이 0 이하인 셀 {len(zero_w)}개 {zero_w[:6]} — 그 클라이언트의 학습이 "
                "집계에 반영되지 않았다는 뜻이다"
            )
        # (4''') 단위 일치 — 회계가 단위를 거짓말하지 못하게 한다. 85번 ① 에서 실제
        #        전송 가중은 페어 수인데 weight-unit 은 supervised_tokens 로 남았다.
        #        단위가 가리키는 값과 기록된 가중이 다르면 그 자체가 실패다.
        for (r, c), cell in sorted(self.cells.items()):
            if cell.fedavg_weight is None or not cell.fedavg_weight_unit:
                continue
            expect = {"supervised_tokens": float(cell.supervised_tokens),
                      "num_examples": float(cell.num_examples)}.get(cell.fedavg_weight_unit)
            if expect is None:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: 알 수 없는 가중 단위 "
                    f"{cell.fedavg_weight_unit!r}"
                )
            elif abs(cell.fedavg_weight - expect) > 1e-6:
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: 가중 단위가 거짓말한다 — unit="
                    f"{cell.fedavg_weight_unit} 이면 가중이 {expect} 여야 하는데 "
                    f"{cell.fedavg_weight} 가 전송됐다 (85번 ① 형태)"
                )

        unweighted = sorted([r, c] for (r, c), cell in self.cells.items()
                            if cell.fedavg_weight is None)
        if unweighted:
            # 통과도 실패도 아니다. 연합 칸이 아니면(로컬·중앙) 가중 자체가 없다.
            notes.append(
                f"FedAvg 가중이 기록되지 않은 셀 {len(unweighted)}개 {unweighted[:6]} — "
                "연합 칸이 아니거나 클라이언트가 단위를 싣지 않았다. **0 으로 읽지 마라.**"
            )

        # (4') 재구성 값 표식
        recon = sorted([r, c] for (r, c), cell in self.cells.items()
                       if cell.value_source != "measured")
        if recon:
            notes.append(
                f"실측이 아닌 재구성 셀 {len(recon)}개 {recon[:6]} — 원자 로그에 없는 필드는 "
                "상수로 채워졌다. 이 행의 epochs_ran·optimizer·lr 을 실측으로 인용하지 마라."
            )

        # (5) 최적화 설정이 실제로 고정됐는가 — 'auto' 교체를 여기서 잡는다
        opts = {cell.optimizer for cell in self.cells.values() if cell.optimizer}
        if len(opts) > 1:
            failures.append(f"실사용 optimizer 가 셀마다 다르다: {sorted(opts)}")
        for (r, c), cell in sorted(self.cells.items()):
            if cell.arg_optimizer and cell.arg_optimizer.lower() == "auto":
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: optimizer='auto' — 명시한 lr0·momentum 이 버려진다"
                )
            if cell.optimizer and cell.arg_optimizer and not cell.optimizer.lower().startswith(
                cell.arg_optimizer.lower()[:3]
            ):
                failures.append(
                    f"라운드 {r} 클라이언트 {c}: 설정 optimizer={cell.arg_optimizer} 인데 "
                    f"실사용은 {cell.optimizer} 다"
                )

        return AuditReport(
            ok=not failures,
            failures=failures,
            total_epochs_by_client=totals,
            total_optimizer_steps=sum(cell.optimizer_steps for cell in self.cells.values()),
            total_optimizer_updates=sum(
                cell.optimizer_updates for cell in self.cells.values()
            ),
            resumed_cells=sorted(
                [r, c] for (r, c), cell in self.cells.items()
                if cell.resumed_from_epoch is not None
            ),
            reconstructed_cells=recon,
            notes=notes,
        )

    # -- 산출물 -------------------------------------------------------------
    def to_csv(self, path: str | Path) -> Path:
        """회계 매트릭스를 CSV 로 남긴다. 정식 산출물이며 머지 차단 조건이다."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            w.writeheader()
            for _, cell in sorted(self.cells.items()):
                row = asdict(cell)
                w.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})
        return p

    def to_json(self, path: str | Path) -> Path:
        """감사 결과 요약. MLflow 에 함께 올린다."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report = self.audit()
        payload = {
            "num_rounds": self.num_rounds,
            "client_ids": self.client_ids,
            "local_epochs": self.local_epochs,
            "total_epochs": self.total_epochs,
            "audit": report.as_dict(),
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p
