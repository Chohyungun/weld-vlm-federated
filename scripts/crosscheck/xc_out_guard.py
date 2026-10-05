"""독립 구현 스크립트의 출력 경로 보호 — 새 출력만, 저장소의 `_workspace/` 아래에만 쓴다.

출력 폴더를 만들거나 입력을 읽기 **전에** 부른다. 경로는 정션 · 링크를 푼 실제 경로로 맞댄다.

- 출력이 이미 있으면 멈춘다 — 이전 산출 · 비교 대상 · 봉인본을 덮어쓰지 않는다.
- 실제 경로가 `<저장소>/_workspace/` 아래가 아니면 멈춘다 — 본체 자료 · 동결 명부 · 다른 트랙의 산출 · 추적
  경로에는 쓰지 않는다. `_workspace` 자체가 링크이면 멈춘다(그 너머가 어디인지 이 저장소가 정하지 않는다).
- 입력과 같은 경로이거나, 입력이 출력 아래에 있거나, 출력이 입력 아래에 있으면 멈춘다.
- 출력의 조상 가운데 계약서(`SNAPSHOT.sha256`)가 있는 폴더가 있으면 멈춘다(동결 디렉터리).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

CONTRACT_NAME = "SNAPSHOT.sha256"


def _real(p: Path | str) -> str:
    return os.path.normcase(os.path.realpath(p))


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def guard_new_output(out: Path | str, *, repo: Path | str, inputs: Iterable[Path | str] = ()) -> Path:
    """`out` 이 쓸 수 있는 새 경로인지 본다. 아니면 `SystemExit` 로 멈춘다. 실제 경로를 돌려준다."""
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise SystemExit(f"출력이 이미 있다 — 없는 새 경로만 쓴다: {out}")
    ws = Path(repo) / "_workspace"
    if not ws.is_dir():
        raise SystemExit(f"출력 뿌리가 없다: {ws}")
    real_ws = _real(ws)
    if real_ws != os.path.normcase(os.path.abspath(ws)):
        raise SystemExit(f"출력 뿌리가 링크다 — 쓰지 않는다: {ws}")
    real_out = _real(out)
    if real_out == real_ws or not _inside(real_out, real_ws):
        raise SystemExit(f"출력의 실제 경로가 {ws} 아래가 아니다: {real_out}")
    for p in inputs:
        rp = _real(p)
        if _inside(rp, real_out) or _inside(real_out, rp):
            raise SystemExit(f"출력이 입력과 겹친다: {p}")
    for anc in Path(real_out).parents:
        if (anc / CONTRACT_NAME).is_file():
            raise SystemExit(f"출력의 조상이 동결 디렉터리다: {anc}")
    return Path(real_out)


def recheck_inside(out: Path | str, *, repo: Path | str) -> None:
    """출력을 만든 뒤 다시 본다 — 만드는 사이에 경로가 바뀌지 않았는지."""
    if not _inside(_real(out), _real(Path(repo) / "_workspace")):
        raise SystemExit(f"만든 출력의 실제 경로가 _workspace 밖이다: {out}")
