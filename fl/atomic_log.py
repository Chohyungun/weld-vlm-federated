"""원자 로그 — 라운드별 실측을 한 줄씩 남긴다.

RQ3(참여 이득)은 라운드별 궤적에서 나온다. 궤적을 후처리로 뽑으려면 라운드마다 아래를
남겨야 하고, **지금 남기지 않으면 나중에 전 실험을 다시 돌려야 한다.**

```
run_id, seed, cell, split_hash, client_id, round,
n_train_samples, metric_name, metric_value,
bytes_up, bytes_down, wall_time
```

## 왜 long format 인가

한 줄에 지표 하나만 담는다. 라운드마다 컬럼이 늘어나는 wide format 은 지표가 추가될 때마다
스키마가 바뀌고, 칸마다 산출되는 지표가 달라 빈 칸이 생긴다. long format 은 어느 칸이
어떤 지표를 냈는지가 행의 유무로 드러나므로, 누락을 사후에 셀 수 있다.

## 학습 중에 지표를 만들지 않는다

이 로그가 남기는 것은 **학습 과정의 실측**(표본 수·통신량·소요 시간·파라미터 norm)이지
성능 지표가 아니다. 성능은 학습이 끝난 뒤 저장된 체크포인트를 단일 채점기로 일괄 채점해
얻는다. 학습 중에 성능을 보면 조기 종료 유혹이 생기고, 그 순간 `R × E = N` 이 무의미해진다.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = ["AtomicRecord", "AtomicLog", "FIELDS", "new_run_id"]

#: 열 순서는 51번 지표 설계의 스키마를 그대로 따른다. 순서를 바꾸지 않는다.
FIELDS = (
    "run_id",
    "seed",
    "cell",
    "split_hash",
    "client_id",
    "round",
    "n_train_samples",
    "metric_name",
    "metric_value",
    "bytes_up",
    "bytes_down",
    "wall_time",
)


def new_run_id(cell: str, seed: int, stamp: str) -> str:
    """`{cell}_{seed}_{stamp}` 형식. stamp 는 호출자가 넘긴다.

    시각을 이 함수가 직접 읽지 않는 이유는 같은 run 의 여러 프로세스(서버·클라이언트)가
    같은 `run_id` 를 써야 하기 때문이다. 각자 시각을 읽으면 run 이 쪼개진다.
    """
    return f"{cell}_s{int(seed)}_{stamp}"


@dataclass
class AtomicRecord:
    """원자 로그 한 줄. 지표 하나가 한 줄이다."""

    run_id: str
    seed: int
    cell: str
    split_hash: str
    client_id: int | str
    round: int
    n_train_samples: int
    metric_name: str
    metric_value: float
    bytes_up: int = 0
    bytes_down: int = 0
    wall_time: float = 0.0

    def as_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in FIELDS}


class AtomicLog:
    """CSV 한 파일에 append 한다.

    라운드마다 flush 하는 이유는 학습이 중간에 죽어도 그때까지의 실측이 남아야 하기
    때문이다. 파일럿의 산출물은 "어디서 끊겼는가"이고, 끊긴 지점의 직전 라운드 기록이
    없으면 그 답을 못 얻는다.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        seed: int,
        cell: str,
        split_hash: str,
    ) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.seed = int(seed)
        self.cell = cell
        self.split_hash = split_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        if exists:
            with self.path.open("r", encoding="utf-8", newline="") as fh:
                head = fh.readline().strip().split(",")
            if head != list(FIELDS):
                raise ValueError(
                    f"기존 로그의 열 구성이 다르다: {head}. 스키마가 바뀌면 이전 실험과 "
                    "합칠 수 없으므로 새 파일로 시작해야 한다."
                )
            return
        with self.path.open("w", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    # -- 기록 --------------------------------------------------------------
    def write(self, records: Iterable[AtomicRecord]) -> int:
        rows = [r.as_row() for r in records]
        if not rows:
            return 0
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writerows(rows)
            fh.flush()
            os.fsync(fh.fileno())
        return len(rows)

    def log_round(
        self,
        *,
        round_idx: int,
        client_id: int | str,
        n_train_samples: int,
        metrics: dict[str, float],
        bytes_up: int = 0,
        bytes_down: int = 0,
        wall_time: float = 0.0,
    ) -> int:
        """지표 dict 을 줄 단위로 펼쳐 기록한다."""
        recs = [
            AtomicRecord(
                run_id=self.run_id,
                seed=self.seed,
                cell=self.cell,
                split_hash=self.split_hash,
                client_id=client_id,
                round=int(round_idx),
                n_train_samples=int(n_train_samples),
                metric_name=str(name),
                metric_value=float(value),
                bytes_up=int(bytes_up),
                bytes_down=int(bytes_down),
                wall_time=float(wall_time),
            )
            for name, value in metrics.items()
        ]
        return self.write(recs)

    # -- 검증 --------------------------------------------------------------
    def read_rows(self) -> list[dict[str, str]]:
        with self.path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def audit_rounds(self, expected_rounds: int, expected_clients: Sequence[int | str]) -> list[str]:
        """라운드 × 클라이언트 누락을 센다.

        회계 매트릭스가 학습량 등가를 검사한다면, 이쪽은 **궤적이 끊긴 자리**를 찾는다.
        RQ3 은 라운드별 궤적에서 나오므로 중간이 비면 곡선을 그릴 수 없다.
        """
        seen = {(int(r["round"]), r["client_id"]) for r in self.read_rows()}
        missing = [
            (rd, str(c))
            for rd in range(expected_rounds)
            for c in expected_clients
            if (rd, str(c)) not in seen
        ]
        if not missing:
            return []
        return [f"원자 로그 결측 {len(missing)}건 (라운드, 클라이언트): {missing[:8]}"]


@dataclass
class RoundTimer:
    """벽시계 측정. `wall_time` 열의 입력이다."""

    started: float = field(default_factory=time.perf_counter)

    def lap(self) -> float:
        now = time.perf_counter()
        elapsed = now - self.started
        self.started = now
        return elapsed
