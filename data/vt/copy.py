"""원천 zip 스무 개의 복사와 바이트 대조 (4판 5-2 · 3판 1-2).

    uv run python -X utf8 -m data.vt.copy --drive-root <공유 드라이브의 '1.데이터' 폴더> \\
        --listing <vt_zip_listing.tsv> --listing-sha256 <앞자리> \\
        --label-census <vt_label_census.json> --label-census-sha256 <앞자리>

- zip 을 바이트 구간(설정 `copy.chunk_bytes`)마다 **다시 열어** E: 의 `.part` 에 이어 쓰고, 구간마다 그 앞에서
  지킴(`data.vt.guard.Guard`)을 본다. sha256 · md5 를 이어 계산한다.
- 끝나면 `.part` 를 처음부터 다시 읽어 두 해시를 맞대고, 같을 때만 최종 이름으로 바꾼다.
- 끊긴 `.part` 는 그 바이트를 다시 해시한 뒤 이어 쓴다. 기록에 있는 zip 은 다시 받지 않는다.
- 공유 드라이브의 파일은 읽기만 한다 — 지우지도 옮기지도 않는다.

대조(`verify`)는 셋이다 — 바이트 수(목록 파일), 라벨 zip 의 sha256(라벨 집계 산출), 드라이브 캐시의 서버 md5.
서버 md5 는 **스무 개가 모두 캐시에 잡힐 때만** 증거로 센다(대조군 없는 0 건은 무효).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path

from data.vt.config import REPO_ROOT, load_config, repo_path
from data.vt.guard import Guard, assert_vt_writable

_HEX32 = re.compile(rb"[0-9a-f]{32}")
#: 원천 zip 스무 개 · 라벨 zip 열 개(1판 3-1)
LISTING_N = (20, 10)


class CopyMismatch(RuntimeError):
    """복사본을 다시 읽은 해시 · 크기가 복사 중의 값과 다르다."""


def hash_file(path: Path, read_bytes: int) -> tuple[str, str, int]:
    sha, md5, n = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as fh:
        for b in iter(lambda: fh.read(read_bytes), b""):
            sha.update(b)
            md5.update(b)
            n += len(b)
    return sha.hexdigest(), md5.hexdigest(), n


def _set_aside(part: Path) -> Path:
    """믿을 수 없는 `.part` 를 지우지 않고 옆 이름으로 비켜 둔다."""
    k = 0
    while (bad := part.with_name(f"{part.name}.bad{k}")).exists():
        k += 1
    os.replace(part, bad)
    return bad


def copy_zip(
    src: Path,
    dst: Path,
    *,
    chunk_bytes: int,
    read_bytes: int,
    guard: Guard,
    log: Callable[[dict], None],
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """`src` 를 구간마다 다시 열어 `dst` 로 복사한다. 반환값은 기록 한 줄."""
    if dst.exists():
        raise FileExistsError(f"이미 있다 — 기록을 보고 다시 받지 않는다: {dst}")
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_size = os.path.getsize(src)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    offset = 0
    if part.exists():
        # 이어 쓰기 전에 사본의 접두를 원천의 같은 접두와 맞댄다 — 사본만 다시 해시하면 뒤의 대조가
        # 사본 = 사본 의 동어반복이 된다(끊긴 꼬리가 0 으로 채워진 사본을 받아들인다)
        p_sha, _, offset = hash_file(part, read_bytes)
        ok = offset <= src_size
        if ok:
            s_sha = hashlib.sha256()
            pos = 0
            while pos < offset:                       # 원천도 구간마다 다시 열고 지킴을 본다
                guard.wait()
                with src.open("rb") as fi:
                    fi.seek(pos)
                    n = 0
                    while n < chunk_bytes and pos + n < offset:
                        b = fi.read(min(read_bytes, chunk_bytes - n, offset - pos - n))
                        if not b:
                            raise CopyMismatch(f"원천이 선언한 크기보다 짧다: {src}")
                        s_sha.update(b)
                        n += len(b)
                pos += n
            ok = s_sha.hexdigest() == p_sha
        if not ok:
            bad = _set_aside(part)
            log({"resume_rejected": dst.name, "offset": offset, "moved_to": bad.name})
            offset = 0
        else:
            # 맞으면 해시 상태를 사본의 접두로 다시 세운다(원천 접두와 같은 바이트다)
            with part.open("rb") as fh:
                for b in iter(lambda: fh.read(read_bytes), b""):
                    sha.update(b)
                    md5.update(b)
            log({"resume": dst.name, "offset": offset})
    t0, chunks = clock(), 0
    while offset < src_size:
        before = guard.wait()
        n = 0
        with src.open("rb") as fi, part.open("ab") as fo:
            fi.seek(offset)
            while n < chunk_bytes and offset + n < src_size:
                b = fi.read(min(read_bytes, chunk_bytes - n, src_size - offset - n))
                if not b:
                    raise CopyMismatch(f"원천이 선언한 크기보다 짧다: {src}")
                fo.write(b)
                sha.update(b)
                md5.update(b)
                n += len(b)
            fo.flush()
            os.fsync(fo.fileno())
        offset += n
        chunks += 1
        log({"zip": dst.name, "chunk": chunks, "offset": offset, "of": src_size,
             "c_free_before_gb": before["c_free_gb"], "c_free_after_gb": round(guard.c_free_fn(), 2)})
    h_sha, h_md5, h_n = hash_file(part, read_bytes)
    if (h_sha, h_md5, h_n) != (sha.hexdigest(), md5.hexdigest(), src_size):
        bad = _set_aside(part)                       # 다시 돌려도 이 사본을 이어 받지 않게 비켜 둔다
        raise CopyMismatch(f"다시 읽은 사본이 복사 중의 값과 다르다: {dst.name} → {bad.name}")
    os.replace(part, dst)
    return {"name": dst.name, "bytes": src_size, "sha256": h_sha, "md5": h_md5,
            "chunks": chunks, "seconds": round(clock() - t0, 1)}


# ------------------------------------------------------------------------------------------
# 목록 · 대조
# ------------------------------------------------------------------------------------------
def read_listing(path: Path) -> list[tuple[int, str]]:
    """`<바이트>\\t<드라이브 기준 상대 경로>` 줄들."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            b, rel = line.split("\t", 1)
            out.append((int(b), rel))
    return out


def sha256_of(path: Path) -> str:
    return hash_file(path, 1 << 20)[0]


def check_reference(path: Path, sha_prefix: str) -> None:
    if len(sha_prefix) < 8:
        raise CopyMismatch(f"참조 해시 앞자리는 8 자 이상이다: {sha_prefix!r}")
    got = sha256_of(path).upper()
    if not got.startswith(sha_prefix.upper()):
        raise CopyMismatch(f"참조 파일의 해시가 다르다: {path.name} {got[:16]}… (기대 {sha_prefix})")


def drivefs_server_md5(names: list[str], db: Path) -> dict[str, list[str]]:
    """드라이브 캐시 DB(읽기 전용)에서 이름마다 휴지통이 아닌 항목의 서버 md5 들."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.text_factory = bytes                       # 기본값이면 proto 디코드 오류로 0 건이 된다
    try:
        out: dict[str, list[str]] = {}
        for n in names:
            # local_title 은 TEXT 칸이다 — 바이트로 묶으면 BLOB 과 TEXT 비교라 늘 0 건이 된다(10-04 실측)
            rows = con.execute("SELECT proto FROM items WHERE local_title = ? AND trashed = 0", (n,)).fetchall()
            out[n] = sorted({m.decode() for (proto,) in rows for m in _HEX32.findall(proto or b"")[:1]})
        return out
    finally:
        con.close()


def default_drivefs_db() -> Path | None:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "DriveFS"
    dbs = sorted(base.glob("*/metadata_sqlite_db")) if base.is_dir() else []
    return dbs[0] if len(dbs) == 1 else None


def verify(records: dict[str, dict], listing: list[tuple[int, str]], label_sha: dict[str, str],
           server_md5: dict[str, list[str]] | None, on_disk: dict[str, int] | None = None) -> dict:
    """복사 기록을 세 참조와 맞댄다. 장 단위 값이 아니라 zip 단위 판정만 낸다.

    - 라벨 zip 은 집계 산출의 sha256 과 **전부** 맞대야 한다(이름이 하나라도 안 맞으면 실패).
    - 서버 md5 의 대조군은 **라벨 zip 이 모두 캐시에 하나씩 잡히고 그 md5 가 복사본과 같은 것**이다(4판 5-2).
      대조군이 서면 원천 zip 은 잡힌 것마다 맞대고 수를 싣는다. 서지 않으면 md5 를 세지 않는다.
    - `on_disk` 를 주면 디스크의 파일 크기도 기록과 맞댄다.
    """
    rows, ok = [], True
    label_names = [Path(rel).name for _, rel in listing if Path(rel).name in label_sha]
    control = (server_md5 is not None and len(label_names) == len(label_sha) and all(
        server_md5.get(n, []) == [records[n]["md5"]] for n in label_names if n in records)
        and all(n in records for n in label_names))
    for b, rel in listing:
        n = Path(rel).name
        r = records.get(n)
        row = {"name": n, "copied": r is not None}
        if r is None:
            ok = False
            rows.append(row)
            continue
        row["bytes_ok"] = r["bytes"] == b
        if on_disk is not None:
            row["on_disk_ok"] = on_disk.get(n) == r["bytes"]
        if n in label_sha:
            row["label_sha256_ok"] = r["sha256"] == label_sha[n]
        if control and len(server_md5.get(n, [])) == 1:
            row["server_md5_ok"] = server_md5[n] == [r["md5"]]
        ok &= all(v for k, v in row.items() if k.endswith("_ok"))
        rows.append(row)
    n_label = sum("label_sha256_ok" in r for r in rows)
    # 원천 이미지 zip 은 라벨처럼 따로 선 sha256 이 없다 — 서버 md5 가 유일한 내용 증거다. 없으면 그 사실을 따로 싣고
    # 뒤 단계의 문(`require_copy_verified`)이 읽는다(06b I-6)
    images = [r for r in rows if r["name"] not in label_sha]
    evidence = all(r.get("server_md5_ok") is True for r in images)
    return {"all_ok": ok and len(rows) == len(listing) and n_label == len(label_sha),
            "image_content_evidence_all": evidence,
            "image_zips_with_content_evidence": sum(r.get("server_md5_ok") is True for r in images),
            "n": len(rows), "label_sha256_checked": n_label,
            "server_md5_control": "라벨 zip 이 모두 캐시에 잡히고 md5 가 같다" if control else "대조 불가 — 라벨 zip 대조군이 서지 않았다",
            "server_md5_checked": sum("server_md5_ok" in r for r in rows), "zips": rows}


def load_jsonl_tolerant(path: Path) -> list[dict]:
    """JSONL 을 읽는다. 쓰다 끊겨 **마지막 줄만** 잘렸으면 그 줄을 버린다. 중간 줄이 깨졌으면 멈춘다."""
    if not path.exists():
        return []
    # 바이트로 줄을 가른 뒤 줄마다 푼다 — 여러 바이트 글자의 가운데에서 잘린 꼬리도 그 줄만 버린다(06c N-5)
    lines = [x for x in path.read_bytes().split(b"\n") if x.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise
    return out


def repair_jsonl_tail(path: Path) -> bool:
    """이어 쓰기 전에 잘린 꼬리를 고친다 — 온전한 줄만 임시 이름에 다시 쓰고 바꾼다(06b I-7 ①).

    잘린 조각 뒤에 새 줄을 붙이면 가운데 줄이 깨진다. 고쳤으면 참.
    """
    if not path.exists():
        return False
    raw = path.read_bytes()
    nonempty = [x for x in raw.split(b"\n") if x.strip()]
    if not raw or (raw.endswith(b"\n") and len(load_jsonl_tolerant(path)) == len(nonempty)):
        return False
    good = load_jsonl_tolerant(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return True


def source_sha256(src: Path, *, chunk_bytes: int, read_bytes: int, guard: Guard) -> str:
    """원천 전체의 sha256 — 구간마다 다시 열고 지킴을 본다."""
    h, pos, size = hashlib.sha256(), 0, os.path.getsize(src)
    while pos < size:
        guard.wait()
        with src.open("rb") as fi:
            fi.seek(pos)
            n = 0
            while n < chunk_bytes and pos + n < size:
                b = fi.read(min(read_bytes, chunk_bytes - n, size - pos - n))
                if not b:
                    raise CopyMismatch(f"원천이 선언한 크기보다 짧다: {src}")
                h.update(b)
                n += len(b)
        pos += n
    return h.hexdigest()


def _load_records(path: Path) -> dict[str, dict]:
    return {r["name"]: r for r in load_jsonl_tolerant(path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive-root", type=Path, required=True)
    ap.add_argument("--listing", type=Path, required=True)
    ap.add_argument("--listing-sha256", required=True)
    ap.add_argument("--label-census", type=Path, required=True)
    ap.add_argument("--label-census-sha256", required=True)
    ap.add_argument("--drivefs-db", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    c = cfg["copy"]
    staging = repo_path(cfg["staging_root"])
    raw_dir, lab_dir = repo_path(c["raw_zip_dir"]), repo_path(c["label_zip_dir"])
    rec_dir = staging / "copy"
    roots = [raw_dir, lab_dir, staging]
    for d in (raw_dir, lab_dir, rec_dir):
        assert_vt_writable(d, allowed_roots=roots, repo_root=REPO_ROOT)
        d.mkdir(parents=True, exist_ok=True)
    check_reference(args.listing, args.listing_sha256)
    check_reference(args.label_census, args.label_census_sha256)
    listing = read_listing(args.listing)
    census = json.loads(args.label_census.read_text(encoding="utf-8"))
    label_sha = {z["name"]: z["sha256"] for z in census["inputs"]["vt_label_zips"]}
    if (len(listing), len(label_sha)) != LISTING_N:
        raise CopyMismatch(f"목록 · 라벨 해시의 수가 {LISTING_N} 이 아니다: {len(listing)} · {len(label_sha)}")

    rec_path, log_path = rec_dir / "copy_record.jsonl", rec_dir / "copy_log.jsonl"
    logf = log_path.open("a", encoding="utf-8", newline="\n")

    def log(d: dict) -> None:
        d = {"t": time.strftime("%H:%M:%S"), **d}
        logf.write(json.dumps(d, ensure_ascii=False) + "\n")
        logf.flush()
        print(json.dumps(d, ensure_ascii=False), flush=True)

    repair_jsonl_tail(rec_path)
    records = _load_records(rec_path)
    if not args.verify_only:
        guard = Guard.from_config(cfg, REPO_ROOT, log=log)
        # 라벨 먼저, 그다음 작은 원천부터 — 큰 정상 zip 이 끝에 온다
        order = sorted(listing, key=lambda x: (not Path(x[1]).name.startswith(tuple(c["label_prefixes"])), x[0]))
        for b, rel in order:
            name = Path(rel).name
            if name in records:
                continue
            src = args.drive_root / rel
            if os.path.getsize(src) != b:
                raise CopyMismatch(f"공유 드라이브의 크기가 목록과 다르다: {name}")
            dst = (lab_dir if name.startswith(tuple(c["label_prefixes"])) else raw_dir) / name
            if dst.exists():
                # 최종 이름으로 바꾼 뒤 기록 전에 끊겼다 — 원천을 다시 읽어 같을 때만 기록한다(06b I-6)
                sha, md5, n = hash_file(dst, int(c["read_bytes"]))
                s_sha = source_sha256(src, chunk_bytes=int(c["chunk_bytes"]), read_bytes=int(c["read_bytes"]),
                                      guard=guard)
                if s_sha != sha:
                    raise CopyMismatch(f"기록 없이 남은 최종 사본이 원천과 다르다: {name}")
                r = {"name": name, "bytes": n, "sha256": sha, "md5": md5, "chunks": 0, "seconds": 0.0,
                     "recovered_final": True, "recovered_matched_source": True}
            else:
                r = copy_zip(src, dst, chunk_bytes=int(c["chunk_bytes"]), read_bytes=int(c["read_bytes"]),
                             guard=guard, log=log)
            with rec_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            records[name] = r
            log({"done": name, "bytes": r["bytes"], "seconds": r["seconds"]})

    db = args.drivefs_db or default_drivefs_db()
    server = drivefs_server_md5([Path(rel).name for _, rel in listing], db) if db else None
    disk = {p.name: p.stat().st_size for d in (raw_dir, lab_dir) for p in d.glob("*.zip")}
    res = verify(records, listing, label_sha, server, disk)
    (rec_dir / "copy_verify.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                                             encoding="utf-8", newline="\n")
    log({"verify_all_ok": res["all_ok"], "server_md5_control": res["server_md5_control"]})
    return 0 if res["all_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
