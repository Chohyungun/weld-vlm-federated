"""진단 입력 시험용 합성 스냅샷. 연구 데이터와 봉인 명부를 읽거나 바꾸지 않는다."""

import hashlib
import json
from pathlib import Path


def write_probe_snapshot(folder: Path, rows: list[dict] | None = None) -> tuple[Path, str]:
    folder.mkdir(parents=True, exist_ok=True)
    content = {"pairs.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in (rows or [])),
               "discarded.jsonl": "", "counts.json": "{}\n", "PAIRS_META.json": "{}\n"}
    entries = []
    for name, text in content.items():
        raw = text.encode("utf-8")
        (folder / name).write_bytes(raw)
        entries.append((hashlib.sha256(raw).hexdigest(), name))
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    snapshot = "".join(f"{h}  {n}\n" for h, n in entries) + f"\n# snapshot_digest {digest}\n"
    (folder / "SNAPSHOT.sha256").write_bytes(snapshot.encode())
    return folder / "pairs.jsonl", digest
