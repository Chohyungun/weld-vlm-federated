"""진단 도구 04·04b·05의 페어 입력 대조. 모델 적재·봉인·등록 변경은 하지 않는다.

새 페어는 호출자가 기대하는 스냅샷 digest를 명시한다. 옛 기본 호출만 설정의
동일한 1차 경로·digest로 보완한다. 파일 자체가 주장하는 digest를 기대값으로
채택하지 않는다. 이 검사는 바이트 신원 확인이며 봉인 승인이나 착수 허가가 아니다.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

LEGACY_PAIRS = "data/processed/pairs_main_v1/pairs.jsonl"
_SHA = re.compile(r"[0-9a-f]{64}")


def verify_probe_pairs(pairs_path: str | Path, expected_digest: str | None = None, *,
                       require_digest: bool = False) -> dict:
    """입력과 스냅샷 구성원의 실제 바이트를 대조하거나 SystemExit로 거부한다."""
    from vlm.uni_config import load_uni_config, pairs_snapshot_problems

    path = Path(pairs_path)
    source = "argument"
    if expected_digest is None:
        if require_digest or path.resolve() != Path(LEGACY_PAIRS).resolve():
            raise SystemExit("새 페어 입력은 --pairs-digest를 명시한다 — 스냅샷의 자기 선언으로 대신하지 않는다")
        cfg = load_uni_config()
        if not cfg.pairs_path or Path(cfg.pairs_path).resolve() != path.resolve():
            raise SystemExit("옛 기본 페어와 현재 설정의 경로가 다르다 — --pairs와 --pairs-digest를 명시한다")
        expected_digest = cfg.pairs_digest
        source = "legacy_config"
    if not isinstance(expected_digest, str) or not _SHA.fullmatch(expected_digest):
        raise SystemExit("--pairs-digest는 소문자 16진 SHA256 64자여야 한다")
    snapshot = path.parent / "SNAPSHOT.sha256"
    try:
        raw = snapshot.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        digests = [line.split() for line in lines if line.startswith("# snapshot_digest")]
        if digests != [["#", "snapshot_digest", expected_digest]]:
            raise ValueError("스냅샷 digest 줄이 하나가 아니거나 기대값과 다르다")
        names = set()
        for line in lines:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or not _SHA.fullmatch(parts[0]):
                raise ValueError("스냅샷 구성원 줄의 꼴이 틀리다")
            name = parts[1].strip()
            member = Path(name)
            if (name in names or member.is_absolute() or member.name != name
                    or any(c in name for c in "/\\:") or name in {".", ".."}):
                raise ValueError("스냅샷 구성원 이름이 중복되거나 같은 폴더의 파일 이름이 아니다")
            if (path.parent / member).resolve().parent != path.parent.resolve():
                raise ValueError("스냅샷 구성원이 페어 폴더 밖을 가리킨다")
            names.add(name)
        problems = pairs_snapshot_problems(path, expected_digest)
        if problems:
            raise ValueError("; ".join(problems))
        return {"path": path.as_posix(), "snapshot_digest": expected_digest,
                "snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "pairs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_digest_source": source, "verified": True}
    except (OSError, ValueError) as exc:
        raise SystemExit(f"진단 페어 입력 대조 실패: {exc}") from exc
