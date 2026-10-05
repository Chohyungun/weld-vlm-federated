"""`configs/vt_snapshot.yaml` 을 읽는다. 값은 이 파일에만 있고 모듈은 기본값을 두지 않는다."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "vt_snapshot.yaml"


def load_config(path: Path | None = None) -> dict:
    with (path or CONFIG_PATH).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def repo_path(rel: str, root: Path | None = None) -> Path:
    """설정의 저장소 기준 상대 경로를 절대 경로로."""
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"설정의 경로는 저장소 기준 상대 경로여야 하고 '..' 를 쓰지 않는다: {rel}")
    return (root or REPO_ROOT) / p
