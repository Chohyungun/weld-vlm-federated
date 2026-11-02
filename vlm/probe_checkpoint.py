"""진단 학습 완료 기록과 외부 저장 확인의 연결. 업로드 자체는 런북 래퍼가 한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_train_done(run: Path, result: dict, *, contract_sha256: str | None) -> dict:
    """학습 메타를 먼저 완전히 쓴 뒤 원자적으로 공개한다. 동일 이름은 덮어쓰지 않는다."""
    dest = run / "TRAIN_DONE.json"
    if dest.exists():
        raise SystemExit("학습 완료 기록이 이미 있다 — 덮어쓰지 않는다")
    adapter_sha = sha(run / "adapter.npz")
    if result["train"]["adapter_sha256"] != adapter_sha:
        raise SystemExit("학습 완료 기록의 adapter_sha256이 저장된 파일과 다르다")
    rec = {"kind": "probe_train_done", "schema": 1, "at": datetime.now(UTC).isoformat(),
           "run_name": result["run_name"], "stub": result["stub"],
           "adapter_sha256": adapter_sha, "contract_sha256": contract_sha256,
           "plan_sha256": sha(run / "PLAN.json"), "rows_sha256": sha(run / "rows.txt"),
           "train": result["train"], "regime": result["regime"], "pairs_input": result["pairs_input"]}
    part = dest.with_suffix(".json.part")
    with part.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, dest)
    return rec


def _snapshot(run: Path, contract_sha256: str) -> tuple[dict, dict]:
    try:
        files = ("TRAIN_DONE.json", "adapter.npz", "PLAN.json", "rows.txt")
        before = {name: sha(run / name) for name in files}
        done = json.loads((run / "TRAIN_DONE.json").read_bytes())
        valid = (isinstance(done, dict) and done.get("kind") == "probe_train_done" and done.get("run_name") == run.name
                 and done.get("contract_sha256") == contract_sha256
                 and done.get("adapter_sha256") == before["adapter.npz"]
                 and done.get("plan_sha256") == before["PLAN.json"] and done.get("rows_sha256") == before["rows.txt"])
    except (OSError, ValueError) as exc:
        raise SystemExit("학습 완료 기록·파일을 읽을 수 없다 — 생성하지 않는다") from exc
    if not valid:
        raise SystemExit("학습 완료 기록이 현재 실행 계약·파일과 다르다 — 생성하지 않는다")
    return done, before


def _verify_saved(run: Path, contract_sha256: str, before: dict) -> dict:
    """즉시 검사와 대기가 공유하는 한 판정. Drive 전송 자체를 실행하거나 재검증하지 않는다."""
    receipt = run / "TRAIN_SAVED_RECEIPT.json"
    try:
        raw = receipt.read_bytes()
        rec = json.loads(raw)
        if not isinstance(rec, dict):
            raise TypeError("객체 아님")
        tar = rec.get("tar")
        want_members = {f"{run.name}/{n}" for n in before} | {"RUN_CONTRACT.json", "ENV_RECEIPT.json"}
        members = rec.get("members")
        names = [m.get("path") for m in members if isinstance(m, dict)] if isinstance(members, list) else []
        valid = (rec.get("kind") == "train_saved" and rec.get("name") == run.name and rec.get("ok") is True
                 and rec.get("contract_sha256") == contract_sha256
                 and rec.get("train_done_sha256") == before["TRAIN_DONE.json"]
                 and rec.get("adapter_sha256") == before["adapter.npz"]
                 and isinstance(tar, dict) and isinstance(tar.get("sha256"), str)
                 and len(tar["sha256"]) == 64 and all(c in "0123456789abcdef" for c in tar["sha256"])
                 and tar.get("sha256") == tar.get("drive_sha256") and type(tar.get("bytes")) is int and tar["bytes"] > 0
                 and isinstance(members, list) and len(members) == len(want_members)
                 and len(names) == len(want_members) and all(isinstance(n, str) for n in names)
                 and set(names) == want_members
                 and all(sha(run / name) == digest for name, digest in before.items()) and receipt.read_bytes() == raw)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit("학습 보존 영수증을 읽을 수 없다 — 생성하지 않는다") from exc
    if not valid:
        raise SystemExit("학습 보존 영수증이 실패했거나 현재 실행·파일과 다르다 — 생성하지 않는다")
    return {"receipt_sha256": hashlib.sha256(raw).hexdigest(), "tar_sha256": tar["sha256"], "ok": True}


def verify_train_saved(run: Path, contract_sha256: str) -> dict:
    """대기·재시도 없이 현재 학습 보존 영수증을 한 번 확인한다."""
    run = Path(run)
    _, before = _snapshot(run, contract_sha256)
    return _verify_saved(run, contract_sha256, before)


def wait_train_saved(run: Path, timeout_s: float, *, contract_sha256: str) -> dict:
    """명시된 시간 안에 같은 실행의 보존 영수증이 생겼을 때만 공유 판정으로 반환한다."""
    if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise SystemExit("학습 보존 대기 시간은 유한한 양수여야 한다")
    run = Path(run)
    _, before = _snapshot(run, contract_sha256)
    until = time.monotonic() + timeout_s
    while True:
        if (run / "TRAIN_SAVED_RECEIPT.json").exists():
            return _verify_saved(run, contract_sha256, before)
        remaining = until - time.monotonic()
        if remaining <= 0:
            raise SystemExit("학습 보존 영수증 대기 시간 초과 — 학습 산출을 남기고 생성하지 않는다")
        time.sleep(min(0.25, remaining))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="학습 보존 영수증의 대기 없는 재대조")
    parser.add_argument("--verify", nargs=2, required=True, metavar=("RUN", "CONTRACT"))
    args = parser.parse_args(argv)
    run, contract = map(Path, args.verify)
    try:
        digest = sha(contract)
        verify_train_saved(run, digest)
        if sha(contract) != digest:
            raise SystemExit("소비 중 계약 변경 — 생성하지 않는다")
    except OSError as exc:
        raise SystemExit("계약/보존 파일을 읽을 수 없다 — 생성하지 않는다") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
