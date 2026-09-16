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

__all__ = ["AtomicRecord", "AtomicLog", "FIELDS", "IDENTITY_FIELDS", "new_run_id", "policy_stamp",
           "POLICY_SUFFIX_RESEED", "LedgerIdentityMismatch", "read_ledger_identity",
           "assert_ledger_compatible"]

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


#: loader 재시드 정책(F01 opt-in)이 켜진 실행의 run_id 접미. **꺼진 실행은 접미가 없다** —
#: 기존 세 시드의 run_id(`sep_fed_s20260828_main_s1` 꼴)는 한 글자도 바뀌지 않는다(뒤호환).
POLICY_SUFFIX_RESEED = "_rs1"


def policy_stamp(stamp: str, loader_reseed_per_epoch: bool = False) -> str:
    """실행 식별자(stamp)에 난수 정책을 새긴다(34번 §3-1 8-4).

    같은 seed·같은 출력 경로에 정책만 다른 실행이 겹치면 `run_id` 가 달라져 원장 신원 검사
    (`AtomicLog` 의 열기 거부)에 걸리고, 사후에도 run_id 만으로 정책을 구분할 수 있다.
    정책이 꺼진 기본 실행은 stamp 를 그대로 돌려준다.
    """
    if type(loader_reseed_per_epoch) is not bool:
        raise ValueError("loader_reseed_per_epoch 는 bool 이어야 한다")
    return f"{stamp}{POLICY_SUFFIX_RESEED}" if loader_reseed_per_epoch else stamp


#: 원장의 소유자를 정하는 필드. 이 넷이 같으면 같은 실행의 원장이다.
IDENTITY_FIELDS = ("run_id", "seed", "cell", "split_hash")


class LedgerIdentityMismatch(ValueError):
    """기존 원장의 신원(run_id·seed·cell·split_hash)과 새 실행이 다르다 — 그 위에 쓰지 않는다.

    **`ValueError` 를 상속하는 것이 의도다.** flwr 1.33 의 서버 실행 경로는 두 런타임 모두
    `except Exception` 으로만 받는다(`superlink/runtime/run_serverapp.py`,
    `simulation/run_simulation.py`). `SystemExit` 로 던지면 SuperLink 는 run 상태 FAILED·사유
    푸시(`flwr_exit`)를 건너뛰고, 인프로세스 시뮬레이션은 서버 **스레드**의 SystemExit 를
    threading 이 조용히 무시해 `success=True` 로 끝난다 — 거부했는데 성공으로 보인다.
    그래서 이 예외를 그대로 전파한다(34번 §3-1 8-4 의 "SystemExit" 문구는 이 근거로 정정).
    """


def read_ledger_identity(path: str | Path) -> dict[str, str] | None:
    """원장 첫 데이터 행의 신원 4 필드. 헤더만 있거나 파일이 없으면 None.

    원장은 append 전용이라 첫 행이 곧 그 파일의 소유자다. 헤더가 스키마와 다르면 ValueError.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if head != list(FIELDS):
            raise ValueError(f"기존 로그의 열 구성이 다르다: {head}")
        for row in reader:
            if not row or all(not c for c in row):
                continue
            # 첫 `write` 도중 죽으면 첫 데이터 행이 잘려 있을 수 있다. 신원 4 열이 온전하면
            # 읽고, 그마저 잘렸으면 **소유자를 모른다**고 크게 실패한다 — KeyError 로 죽거나
            # 빈 신원으로 통과시키면 남의 원장에 append 하는 바로 그 경로가 열린다.
            if len(row) > len(FIELDS):
                raise ValueError(f"{p}: 원장 행이 스키마보다 길다({len(row)} 열)")
            rec = dict(zip(FIELDS, row))
            if any(not rec.get(k) for k in ("run_id", "seed", "cell")):
                raise ValueError(
                    f"{p}: 원장 첫 데이터 행이 손상돼 신원을 읽을 수 없다: {row[:4]}. "
                    "이 원장이 잔해라면 실패 디렉터리로 옮긴 뒤 다시."
                )
            return {k: str(rec.get(k, "")) for k in IDENTITY_FIELDS}
    return None


def assert_ledger_compatible(
    path: str | Path, *, run_id: str, seed: int, cell: str, split_hash: str
) -> dict[str, str] | None:
    """기존 원장이 있으면 신원을 대조하고, 하나라도 다르면 `LedgerIdentityMismatch`.

    같은 run_id 의 재기동(같은 seed·cell·split_hash)은 통과한다 — 원장은 그 실행의 것이다.
    빈 원장(헤더만)·부재는 통과. 외부 검토자 권고(26번 §8-4): 경로 문자열 규칙이 아니라
    **기존 run manifest(원장 첫 행)와 대조**해 거부한다.
    """
    found = read_ledger_identity(path)
    if found is None:
        return None
    mine = dict(zip(IDENTITY_FIELDS,
                    (str(run_id), str(int(seed)), str(cell), str(split_hash))))
    diff = {k: (found[k], mine[k]) for k in mine if found[k] != mine[k]}
    if diff:
        raise LedgerIdentityMismatch(
            f"다른 실행의 원장 위에 쓰지 않는다: {Path(path)} — 기존≠새 {diff}. "
            "다른 정책·시드·분할의 실행은 별도 출력 경로(또는 다른 run-stamp)로 돌려라. "
            "이 원장이 미완주 실패 잔해라면 **산출물 디렉터리 전체**를 실패 경로로 옮긴 뒤 "
            "다시 띄운다 — 원장만 치우면 완결된 가중치·회계 위에 새 실행이 덮어쓴다."
        )
    return found


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
        # 원장 신원 검사(34번 §3-1 8-4): 같은 경로에 **다른 실행**의 행이 있으면 열지 않는다.
        # 09-05 형 혼입("다른 시도의 회계")을 append 전에 막는다. 같은 run_id 의 재기동은 통과.
        assert_ledger_compatible(self.path, run_id=run_id, seed=seed, cell=cell, split_hash=split_hash)
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
