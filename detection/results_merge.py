"""Ultralytics `results.csv` 조각을 **읽기 전용**으로 합친다(34번 §3-1 8-1).

재개한 런은 `results.csv` 가 프로세스마다 갈린다. 상류 `BaseTrainer.__init__` 은 `args.resume`
가 거짓이면 같은 save_dir 의 `results.csv` 를 지우므로(`engine/trainer.py:199-201`), 우리 재개
(상류 resume 을 쓰지 않는다)는 이전 프로세스의 epoch 별 loss 행을 잃었다. 그래서 `train_round`
가 재개 직전에 옛 파일을 `results.before_resume_<epoch_done>__<run_id>__<시각>.csv` 로 옮겨 둔다.

이 모듈은 그 조각들을 **파생 테이블(메모리 행 목록)** 로만 합친다 — 어떤 파일도 쓰거나
지우거나 옮기지 않는다(원본 CSV 무수정 원칙, 감사 F03 권고).

## 구판 짧은 행을 오른쪽으로 패딩하면 안 된다 (F03)

F03 의 실제 결함은 "열이 줄었다"가 아니라 **가운데 열이 빠졌다**는 것이다. 수정 전 코드의
`validate()` 는 빈 dict 을 돌려줬고, 상류 `save_metrics` 는 받은 키만 쓰므로 마지막 epoch 행이

    epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,lr/pg0,lr/pg1,lr/pg2   (8 열)

로 남았다 — `metrics/*`·`val/*` 7 열이 통째로 빠지고 LR 3 개가 그 자리로 당겨진 모양이다.
동결 산출물이 실제로 이렇다(`outputs/main_c/seed1/sep_central/r000_c0/results.csv` 101 행 중
1 행이 8 열, 뒤 3 값 = 0.000102443 ×3 = LR). 그래서 **오른쪽 NaN 패딩은 금지**다: 그러면 LR 이
`metrics/precision(B)`·`metrics/recall(B)`·`metrics/mAP50(B)` 값으로 들어앉아, 재지도 않은 성능
지표를 실측값처럼 싣게 된다. 아래 `_remap_short_row` 는 앞 블록(epoch·time·train/*)과 뒤
블록(lr/*)으로 갈라 붙이고 가운데(metrics/*·val/*)를 NaN 으로 둔다. 길이가 그 규칙으로도
설명되지 않으면 **추측하지 않고 거부**한다.

## 그 밖의 규칙

- epoch 키가 겹치면 **뒤 파일(재개 뒤)** 을 채택한다.
- 손상 행(부분 기록된 마지막 줄)은 버리고 **버린 사실을 보고한다** — 침묵 스킵은 금지.
  버리지 않으면 "뒤 파일 승" 과 결합해 정상 행을 전 열 NaN 으로 덮는다.
- 헤더가 없거나(0 바이트) 열 구성이 다른 조각은 그 조각만 건너뛰고 나머지를 합친다.
- 열 이름이 중복된 헤더는 거부한다(행 dict 이 조용히 줄어든다).
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = ["EPOCH", "MergeReport", "before_resume_files", "merge_results", "read_results",
           "sanitize_run_id", "fragment_name"]

EPOCH = "epoch"
#: `results.before_resume_<epoch_done>__<run_id>__<시각>.csv`. `<epoch_done>` 은 상류의 **0-기반**
#: `trainer.epoch` 이고 CSV 의 `epoch` 열은 `self.epoch + 1` 이다(1-기반) — 이름의 숫자가 파일
#: 안의 마지막 epoch 보다 1 작다. 구분자를 `__` 로 둔 이유는 run_id 자체가 `_` 를 포함하기
#: 때문이다(`main_s1_rs1`).
_FRAGMENT = re.compile(r"^results\.before_resume_(\d+)__(.*)__(\d{8}-\d{6}(?:-\d+)?)\.csv$")
_SAFE = re.compile(r"[^A-Za-z0-9.=+-]+")


def sanitize_run_id(run_id: str) -> str:
    """파일 이름에 넣을 수 있게 다듬는다. 빈 값은 `norun`."""
    cleaned = _SAFE.sub("-", str(run_id)).strip("-")
    return cleaned or "norun"


def fragment_name(epoch_done: int, run_id: str, stamp: str) -> str:
    return f"results.before_resume_{int(epoch_done)}__{sanitize_run_id(run_id)}__{stamp}.csv"


@dataclass
class MergeReport:
    """병합이 무엇을 고쳐 읽고 무엇을 버렸는지. 파생 테이블과 함께 보고한다."""

    remapped: list[tuple[str, int]] = field(default_factory=list)      # (파일, epoch) 구판 짧은 행
    dropped_rows: list[tuple[str, int, str]] = field(default_factory=list)   # (파일, 줄번호, 사유)
    skipped_files: list[tuple[str, str]] = field(default_factory=list)      # (파일, 사유)

    def as_dict(self) -> dict:
        return {"remapped": [list(x) for x in self.remapped],
                "dropped_rows": [list(x) for x in self.dropped_rows],
                "skipped_files": [list(x) for x in self.skipped_files]}

    def summary(self) -> str:
        return (f"구판 행 재사상 {len(self.remapped)} · 버린 행 {len(self.dropped_rows)} · "
                f"건너뛴 조각 {len(self.skipped_files)}")


def _blocks(header: Sequence[str]) -> tuple[int, int] | None:
    """(앞 블록 길이, 뒤 블록 시작) — 가운데가 빠진 구판 행을 되맞추는 좌표.

    앞 블록 = 첫 `metrics/`·`val/` 열 앞까지, 뒤 블록 = 첫 `lr/` 열부터 끝까지.
    그 구조가 아니면 None(=되맞출 수 없음).
    """
    mids = [i for i, h in enumerate(header) if h.startswith(("metrics/", "val/"))]
    lrs = [i for i, h in enumerate(header) if h.startswith("lr/")]
    if not mids or not lrs or lrs[0] <= mids[0]:
        return None
    return mids[0], lrs[0]


def _remap_short_row(header: Sequence[str], row: Sequence[str]) -> tuple[list[str], bool] | None:
    """구판 짧은 행을 15 열 좌표로 되맞춘다. 규칙으로 설명되지 않으면 None."""
    pos = _blocks(header)
    if pos is None:
        return None
    front, lr_start = pos
    tail = len(header) - lr_start
    if len(row) != front + tail:
        return None
    out = [""] * len(header)
    out[:front] = list(row[:front])
    out[lr_start:] = list(row[front:])
    return out, True


def _to_values(header: Sequence[str], cells: Sequence[str], *, path: Path,
               line_no: int) -> dict[str, float]:
    row: dict[str, float] = {}
    for k, v in zip(header, cells):
        if k == EPOCH:
            try:
                row[k] = int(float(v))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: epoch 값을 읽을 수 없다: {v!r}") from exc
        else:
            row[k] = float(v) if v != "" else math.nan
    return row


def read_results(path: str | Path, report: MergeReport | None = None
                 ) -> tuple[list[str], list[dict[str, float]]]:
    """조각 하나를 읽는다 → (헤더, 행 목록). `epoch` 은 int, 나머지는 float(빈 칸은 NaN).

    구판 짧은 행은 좌표를 되맞추고(`report.remapped`), 손상 행은 버린다(`report.dropped_rows`).
    헤더가 깨진 파일은 `ValueError` — 여러 조각을 합칠 때는 `merge_results` 가 그 조각만
    건너뛰고 사유를 보고한다.
    """
    p = Path(path)
    # BOM 을 허용한다 — 사람이 Excel 로 열었다 저장한 파일이 온다.
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader, [])]
        if not header or header[0] != EPOCH:
            raise ValueError(f"{p}: results.csv 헤더가 아니다: {header[:3]}")
        if len(set(header)) != len(header):
            raise ValueError(f"{p}: 헤더에 중복 열 이름이 있다: {header}")
        rows: list[dict[str, float]] = []
        for line_no, raw in enumerate(reader, start=2):
            if not raw or all(not c.strip() for c in raw):
                continue
            cells = [c.strip() for c in raw]
            if len(cells) > len(header):
                if report is not None:
                    report.dropped_rows.append((p.name, line_no, f"열 {len(cells)} > 헤더 {len(header)}"))
                    continue
                raise ValueError(f"{p}:{line_no}: 헤더 {len(header)} 열보다 긴 행({len(cells)} 열)")
            remapped = False
            if len(cells) < len(header):
                fixed = _remap_short_row(header, cells)
                if fixed is None:
                    # 부분 기록된 마지막 줄 — 버린다. 오른쪽 패딩하면 "뒤 파일 승" 과 결합해
                    # 앞 조각의 정상 행을 전 열 NaN 으로 덮는다.
                    if report is not None:
                        report.dropped_rows.append(
                            (p.name, line_no, f"설명되지 않는 열 수 {len(cells)}/{len(header)}"))
                        continue
                    raise ValueError(
                        f"{p}:{line_no}: 열 수 {len(cells)} 가 구판 서명과도 맞지 않는다"
                        f"(헤더 {len(header)}) — 부분 기록된 행으로 보인다")
                cells, remapped = fixed
            row = _to_values(header, cells, path=p, line_no=line_no)
            if remapped and report is not None:
                report.remapped.append((p.name, int(row[EPOCH])))
            rows.append(row)
    return header, rows


def merge_results(paths: Sequence[str | Path], report: MergeReport | None = None,
                  *, quiet: bool = False) -> list[dict[str, float]]:
    """조각들을 epoch 오름차순 한 표로 합친다. 겹치는 epoch 은 **뒤 파일**이 이긴다.

    열 순서는 첫 정상 조각의 헤더를 따르고, 모든 행이 그 열을 전부 가진다(빠진 값은 NaN).
    반환값은 새 dict 목록이다 — 입력 파일은 읽기만 한다.
    읽을 수 없는 조각은 건너뛰고 사유를 `report`(없으면 표준출력)로 알린다 — 침묵 스킵은 금지.
    """
    rep = report if report is not None else MergeReport()
    header: list[str] | None = None
    by_epoch: dict[int, dict[str, float]] = {}
    for path in paths:
        try:
            h, rows = read_results(path, rep)
        except (OSError, ValueError) as exc:
            rep.skipped_files.append((Path(path).name, str(exc)))
            continue
        if header is None:
            header = h
        elif h != header:
            rep.skipped_files.append((Path(path).name, f"열 구성이 첫 조각과 다르다: {h}"))
            continue
        for row in rows:
            by_epoch[int(row[EPOCH])] = row
    if not quiet and report is None and (rep.dropped_rows or rep.skipped_files):
        print(f"[results_merge] {rep.summary()} — {rep.dropped_rows} {rep.skipped_files}", flush=True)
    return [by_epoch[e] for e in sorted(by_epoch)]


def before_resume_files(save_dir: str | Path, *, run_id: str | None = None,
                        report: MergeReport | None = None) -> list[Path]:
    """`train_round` 가 보존한 조각들(재개 순) + 현재 `results.csv`(있으면 마지막).

    `run_id` 를 주면 **그 실행의 조각만** 모은다. 주지 않으면 전부 모으되 신원이 둘 이상
    섞여 있으면 보고한다 — 같은 save_dir 이 재사용될 수 있고(`name=r{round}_c{client}` 고정),
    옛 실행 조각은 `epoch_done` 이 크다는 이유로 새 실행 행보다 뒤에 놓여 표를 오염시킨다.
    """
    d = Path(save_dir)
    want = sanitize_run_id(run_id) if run_id is not None else None
    parts: list[tuple[int, str, Path]] = []
    others: set[str] = set()
    for p in sorted(d.glob("results.before_resume_*.csv")):
        m = _FRAGMENT.match(p.name)
        if m is None:
            if report is not None:
                report.skipped_files.append((p.name, "조각 이름 규약과 다르다"))
            continue
        rid = m.group(2)
        if want is not None and rid != want:
            others.add(rid)
            continue
        others.add(rid)
        parts.append((int(m.group(1)), m.group(3), p))
    if len(others) > 1:
        msg = f"{d}: 조각에 실행 신원이 {sorted(others)} 로 섞여 있다"
        if report is not None:
            report.skipped_files.append((d.name, msg))
        else:
            print(f"[results_merge] {msg}", flush=True)
    ordered = [p for _, _, p in sorted(parts, key=lambda t: (t[0], t[1]))]
    cur = d / "results.csv"
    return ordered + ([cur] if cur.is_file() else [])
