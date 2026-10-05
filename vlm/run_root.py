"""리허설 · 진단 루트의 규칙 — 목적마다 받는 루트의 종류와 "모든 경로가 그 아래" (리허설 2판 §5-1 · §1-4 의 순서 4).

- 비본실험 부모는 `outputs/rehearsal_u/` 다. 그 **바로 아래** 리허설 루트 `reh-YYYYMMDDTHHMMSS` 와 진단 루트
  `fdiag-YYYYMMDDTHHMMSS` 를 둔다. `rehearsal` 은 `reh-*` 만, `frame_diag` 는 `fdiag-*` 만 받는다.
- 판정은 **양쪽을 모두 `resolve()` 한 뒤 `is_relative_to`** 로 한다. `outputs/` 는 정션으로 공유되므로 문자열 접두 비교는
  `rehearsal_u_x/` · `rehearsal_u/../main_u/` 에 속는다.
- 본실험(`main`)의 쪽 — 비본실험 부모 아래를 쓰지도 읽지도 않는다 — 은 `vlm.uni_config.main_run_problems` 가 본다.

비본실험 부모는 이음새처럼 **파이썬 인자로만** 바꾼다 — 합성 시험이 `tmp_path` 를 준다(2판 §5-1 의 시험의 경로).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

__all__ = ["NON_MAIN_PARENT", "ROOT_KIND", "run_root_problems", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[1]
NON_MAIN_PARENT = Path("outputs/rehearsal_u")
#: 목적 → 루트 이름의 꼴.
#: `fullmatch` 로만 쓴다 — `$` 는 끝 개행을, `\d` 는 유니코드 숫자를 받는다.
ROOT_KIND = {"rehearsal": re.compile(r"reh-[0-9]{8}T[0-9]{6}"), "frame_diag": re.compile(r"fdiag-[0-9]{8}T[0-9]{6}")}


def run_root_problems(purpose: str, run_root: str | Path | None, paths: Iterable[str | Path], *,
                      parent: str | Path | None = None) -> list[str]:
    """리허설 · 진단의 루트와 경로가 규칙에 맞는지. 빈 목록이면 맞다. 본실험은 여기서 보지 않는다(빈 목록)."""
    if purpose == "main":
        return []
    if purpose not in ROOT_KIND:
        return [f"모르는 목적: {purpose!r}"]
    if run_root is None:
        return [f"{purpose} 는 루트(--run-root)가 있어야 한다"]
    pp = (REPO_ROOT / NON_MAIN_PARENT if parent is None else Path(parent)).resolve()
    rr = Path(run_root).resolve()
    out: list[str] = []
    if rr.parent != pp:
        out.append(f"루트 {rr} 가 비본실험 부모 {pp} 의 바로 아래가 아니다")
    if not ROOT_KIND[purpose].fullmatch(rr.name):
        out.append(f"{purpose} 는 {ROOT_KIND[purpose].pattern} 꼴의 루트만 받는다: {rr.name}")
    for p in paths:
        q = Path(p).resolve()
        if not (q == rr or q.is_relative_to(rr)):
            out.append(f"경로 {p} 가 루트 {rr} 아래가 아니다")
    return out
