"""동결 디렉터리 쓰기 가드 + 격리 파일 경로 해석. 80번 G11-1.

## 왜 있나

`data/interim/manifest_v1/` 을 만든 파이프라인 스크립트 셋(`run_dedup_v1` ·
`run_hist_match` · `run_border_mask`)은 **`manifest.csv` 를 직접 덮어쓴다.** 동결 전에
쓰인 코드라 스스로 "SNAPSHOT 은 잠그지 않았다"고 출력한다. 지금은 잠겨 있으므로
재실행 한 번이 동결본을 지운다 — 읽기 전용 속성에 걸려 멈추더라도 `PermissionError` 는
무슨 일이 벌어진 건지 말해 주지 않고, 누군가 속성을 풀면 조용히 성공한다.

그래서 **파일 속성이 아니라 계약의 존재로** 막는다. `SNAPSHOT.sha256` 이 있는 디렉터리는
완결된 스냅샷이고, 재파생은 항상 새 경로에 한다(개발규약 1-1·1-6).

## 쓰는 법

    from data.frozen_guard import assert_writable, legacy_path

    assert_writable(V1)                       # 동결됐으면 FrozenDirectoryError
    p = legacy_path("manifest_pre_mask.csv")  # attic/ 을 먼저 본다
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ATTIC_NAME",
    "CONTRACT_NAME",
    "FrozenDirectoryError",
    "assert_writable",
    "is_frozen",
    "legacy_path",
]

#: 스냅샷 계약 파일. 이게 있으면 그 디렉터리는 완결된 동결본이다.
CONTRACT_NAME = "SNAPSHOT.sha256"

#: 격리 하위 디렉터리 이름.
ATTIC_NAME = "attic"

_V1 = Path(__file__).resolve().parents[1] / "data/interim/manifest_v1"


class FrozenDirectoryError(RuntimeError):
    """동결된 스냅샷 디렉터리에 쓰려고 했다."""


def is_frozen(directory: Path) -> bool:
    return (Path(directory) / CONTRACT_NAME).is_file()


def assert_writable(directory: Path, *, what: str = "이 디렉터리") -> None:
    """동결 디렉터리면 멈춘다. 파이프라인 스크립트의 진입에서 부른다.

    메시지에 **무엇을 해야 하는지**를 적는다. "권한 없음"만 던지면 다음 사람이
    읽기 전용 속성을 풀고 다시 돌린다 — 그게 정확히 막으려는 사고다.
    """
    d = Path(directory)
    if not is_frozen(d):
        return
    raise FrozenDirectoryError(
        f"{what}({d})는 동결된 스냅샷이다 — {CONTRACT_NAME} 가 있다.\n"
        "  이 스크립트는 manifest.csv 를 덮어쓰므로 재실행하면 동결본이 사라진다.\n"
        "  재파생이 필요하면 --outdir 로 **새 경로**를 주고, 새 스냅샷으로 잠근 뒤\n"
        "  어느 쪽이 정본인지 docs/의사결정로그.md 에 남겨라.\n"
        "  근거: 개발규약 1-1·1-6, 80번 G11-1. 설명은 data/interim/manifest_v1/README.md."
    )


def legacy_path(name: str, root: Path | None = None) -> Path:
    """격리된 옛 매니페스트·분할 메타의 경로. `attic/` 을 먼저 본다.

    본 디렉터리에서 발견되면 격리가 풀린 것이므로 알린다 — 조용히 쓰면 위생이 되돌아간
    것을 아무도 모른다. 파일이 아예 없으면 사유를 담아 `FileNotFoundError`.
    """
    base = Path(root) if root is not None else _V1
    attic = base / ATTIC_NAME / name
    if attic.is_file():
        return attic
    loose = base / name
    if loose.is_file():
        print(f"  ! {name} 이 격리 디렉터리가 아니라 동결 디렉터리 본체에 있다. "
              f"attic/ 으로 되돌려라 (80번 G11-1)")
        return loose
    raise FileNotFoundError(
        f"{name} 을 {attic} 에서도 {loose} 에서도 찾지 못했다. "
        "격리 목록과 사유는 data/interim/manifest_v1/attic/README.md."
    )
