"""클라이언트 × 라운드 회계 매트릭스 — 학습량 등가의 실측 증명.

## 왜 머지 차단 조건인가

Flower `FedAvg`의 기본값 `accept_failures=True`는 클라이언트 하나가 라운드 중 실패해도
나머지로 집계를 진행한다. 그러면 그 라운드의 실효 학습량이 줄어드는데 **지표에는 아무
흔적이 남지 않는다.** `R × E = N` 등가는 이 연구의 공정성 주장 전체를 떠받치는 전제이므로,
조용히 깨지는 이 경로를 막아야 한다.

막는 방법은 두 겹이다. 서버는 `accept_failures=False`로 실패를 라운드 중단으로 만들고,
이 모듈은 **셀의 부재 자체를 실패로 정의**한다. 전자만 있으면 "실패가 없었다"를 신뢰에
의존하게 되지만, 후자가 있으면 R × 클라이언트 수 전 셀이 채워졌는지를 사후에 센다.

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
    "stopper_true_count",
    # 배치 수(`optimizer_steps`)와 실제 갱신 횟수는 다르다 — Ultralytics 가 nbs=64 기준으로
    # 누적한다(숨은 기본값 #10). 논문의 "총 갱신 횟수"는 아래 컬럼이다.
    "optimizer_updates",
    # 재개해서 이어 간 칸인가. 이어 간 런은 무중단 런과 다른 궤적을 그린다.
    "resumed_from_epoch",
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
    stopper_true_count: int = 0
    optimizer_updates: int = 0
    resumed_from_epoch: int | None = None
    participated: bool = True

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
            stopper_true_count=0,  # 스텁 stopper 는 True 를 돌려줄 수 없다
            optimizer_updates=int(getattr(result, "optimizer_updates", 0) or 0),
            resumed_from_epoch=getattr(result, "resumed_from_epoch", None),
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

        # (4) 조기 종료가 실제로 걸리지 않았는가
        for (r, c), cell in sorted(self.cells.items()):
            if cell.stopper_true_count:
                failures.append(f"라운드 {r} 클라이언트 {c}: 조기 종료 판정이 {cell.stopper_true_count}회 발생")

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
