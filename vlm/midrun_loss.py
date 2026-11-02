"""학습 중 손실 기록 — 본학습 칸은 손실을 에폭 끝에 한 번 남기므로(R = E = N = 1 이면 학습이 끝날 때까지 값이 없다) 학습 도중의 창 손실을 따로 적는다.

엘리스 전체 학습의 멈춤 지점 (나) "손실이 줄지 않으면 멈춘다" 의 입력이다(의사결정로그 10-31 c0c418f2, 미니스펙
`_workspace/2026-11-01-main2b/MINISPEC_midrun_loss.md`). A3 대체 승인의 동일성 파일(`vlm/pilot_vlm.py` 등)을 바꾸지 않으려고
학습 고리가 아니라 손실 함수(`vlm.loss_norm.supervised_ce_sum`)의 관찰자로 붙는다 — 값을 바꾸지 않고 읽기만 한다.

환경 변수 `WELD_MIDRUN_LOSS_DIR` 가 있을 때만 켠다(없으면 아무 일도 없다 — 지금 동작 그대로). `WELD_MIDRUN_LOSS_EVERY` 는 한 줄에
묶는 손실 호출 수(마이크로 배치 수, 기본 800 = 누적 8 에서 100 걸음). 줄마다 그 창의 감독 토큰 가중 평균 교차 엔트로피
`Σ ce / Σ 감독 토큰` 을 적는다. 학습 계산 · 난수 · 순서에는 끼어들지 않는다.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

ENV_DIR = "WELD_MIDRUN_LOSS_DIR"
ENV_EVERY = "WELD_MIDRUN_LOSS_EVERY"
DEFAULT_EVERY = 800


class WindowLossRecorder:
    """손실 호출을 `every` 개씩 묶어 창 손실 한 줄을 JSONL 로 덧붙인다. 끝에 남은 호출은 `close` 가 마지막 줄로 쓴다."""

    def __init__(self, path: Path, *, every: int, tag: str):
        if int(every) < 1:
            raise ValueError(f"{ENV_EVERY} 는 1 이상의 정수다: {every!r}")
        self.path, self.every, self.tag = Path(path), int(every), str(tag)
        self.calls = 0
        self.cum_tokens = 0
        self._ce = 0.0
        self._tok = 0
        self._win_calls = 0
        self._t0 = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, ce, n_tokens: int) -> None:
        self._ce += float(ce)          # 떼어 낸 0 차원 텐서 — 값만 읽는다
        self._tok += int(n_tokens)
        self._win_calls += 1
        self.calls += 1
        if self._win_calls >= self.every:
            self._flush(final=False)

    def _flush(self, *, final: bool) -> None:
        if self._win_calls == 0:
            return
        self.cum_tokens += self._tok
        row = {"tag": self.tag, "calls": self.calls, "window_calls": self._win_calls, "window_tokens": self._tok,
               "window_ce": (self._ce / self._tok) if self._tok else None, "cum_tokens": self.cum_tokens,
               "wall_s": round(time.monotonic() - self._t0, 3), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "final": final}
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._ce, self._tok, self._win_calls = 0.0, 0, 0

    def close(self) -> None:
        self._flush(final=True)


def _safe(tag: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(tag)) or "run"


@contextlib.contextmanager
def observe(tag: str) -> Iterator[WindowLossRecorder | None]:
    """`WELD_MIDRUN_LOSS_DIR` 가 있으면 이 블록 동안 손실 관찰자를 건다. 없으면 `None` 을 내고 아무것도 하지 않는다."""
    d = os.environ.get(ENV_DIR)
    if not d:
        yield None
        return
    from vlm import loss_norm

    every = int(os.environ.get(ENV_EVERY) or DEFAULT_EVERY)
    rec = WindowLossRecorder(Path(d) / f"midrun_loss_{_safe(tag)}_{os.getpid()}.jsonl", every=every, tag=tag)
    loss_norm._OBSERVERS.append(rec)
    try:
        yield rec
    finally:
        loss_norm._OBSERVERS.remove(rec)
        rec.close()
