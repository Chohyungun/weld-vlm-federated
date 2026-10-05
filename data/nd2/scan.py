"""기술자 수준 생성기를 짝 집합에 덩어리로 돌린다 — 중간 저장 · 이어 하기 · 메모리 지킴(미니스펙 2판 5절, 게이트 회차 1).

- 덩어리(행 영상 `block` 장)마다 그 앞에서 지킴(`wait`)을 부른다: 여유 메모리가 문턱 아래거나 본체 게이트 락이 있으면
  풀릴 때까지 기다린다. 시작할 때는 `start_min_gb` 이상이어야 한다(아니면 시작하지 않는다).
- 덩어리를 마치면 `blk_#####.csv`(바닥 이상인 짝)와 `blk_#####.json`(생성기별 0.01 칸 분포 · 짝 수 · 시간 · 실행 해시 ·
  CSV 의 sha256)을 임시 이름으로 쓰고 이름을 바꾼다.
- **실행 식별(`RUN.json`)** — 행 · 열 id 목록(순서 포함) · 기술자 바이트 · 바닥 · 덩어리 크기 · P 탐색 설정 · 스레드 · 계산 코드의
  해시를 묶는다. 다시 돌릴 때 같은 폴더의 `RUN.json` 이 다르거나, 실행 식별 없는 덩어리 · 이 실행의 목록 밖 덩어리가 있으면
  **기존 산출을 그대로 두고 멈춘다**(`RunMismatch`). 이어 하기는 실행 해시와 CSV 해시가 맞는 덩어리만 건너뛴다.
- 요약 · 짝 읽기는 그 실행의 덩어리 목록(0 … n−1)만, 실행 해시 · CSV 해시를 확인하며 읽는다(공유 회신 §63-1).
- **판정하지 않는다.** 바닥(`floors`)은 기록 바닥이다 — 후보 문턱은 등록(3.5)에서 정하고, 분포가 있으니 다시 스캔하지 않는다.
- 짝은 `a_id < b_id`(문자열). 뒤집을 때 이동의 부호를 바꾼다(이동은 a 기준 b 의 위치).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from data.nd2 import desc_scan as S

BINS = np.linspace(-1.0, 1.0, 201)
HEAD = ["a_id", "b_id", "g3", "g4", "g4_dx", "g4_dy", "p", "p_dx", "p_dy"]
GENS = ("g3", "g4", "p")


class StartRefused(RuntimeError):
    """시작 조건(여유 메모리)이 서지 않았다."""


class RunMismatch(RuntimeError):
    """같은 자리의 기존 산출이 이 실행과 다르다 — 지우지 않고 멈춘다."""


SCAN_CODE = ("ncc.py", "desc_scan.py", "scan.py")


def _snapshot_code() -> dict[str, bytes]:
    """`data/nd2/*.py` 를 이 모듈을 들여올 때 한 번 읽어 둔다 — 실행 중에 디스크의 파일이 바뀌어도 지문은 실제로 도는 코드를 가리킨다."""
    base = Path(__file__).resolve().parent
    return {f.name: f.read_bytes().replace(b"\r\n", b"\n") for f in sorted(base.glob("*.py"))}


_CODE_AT_IMPORT = _snapshot_code()


def code_version(files: Sequence[str]) -> str:
    """`data/nd2/` 의 파일들(이름 순)의 sha256 — 들여올 때 읽어 둔 내용으로, 줄끝은 LF 로 맞춰 잰다."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.encode() + b"\0" + _CODE_AT_IMPORT[f] + b"\0")
    return h.hexdigest()


def _sha_text(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _sha_arr(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float32).tobytes()).hexdigest()


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_identity(name: str, a_ids: Sequence[str], b_ids: Sequence[str], Da: np.ndarray, Db: np.ndarray, *, within: bool,
                 floors: dict[str, float], block: int, workers: int, do_p: bool, p_min_overlap: float) -> dict:
    ident = {"name": name, "n_a": len(a_ids), "a_ids_sha256": _sha_text(list(a_ids)), "n_b": len(b_ids),
             "b_ids_sha256": _sha_text(list(b_ids)), "Da_sha256": _sha_arr(Da), "Db_sha256": _sha_arr(Db),
             "within": bool(within), "floors": {k: float(floors[k]) for k in sorted(floors)}, "block": int(block),
             "workers": int(workers), "do_p": bool(do_p), "p_min_overlap": float(p_min_overlap),
             "n_blocks": (len(a_ids) + block - 1) // block, "code": code_version(SCAN_CODE)}
    ident["run"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    return ident


def bind_run(d: Path, ident: dict) -> None:
    """`d` 를 이 실행에 묶는다. 다른 실행의 산출이 있으면 지우지 않고 `RunMismatch`."""
    d.mkdir(parents=True, exist_ok=True)
    rid = d / "RUN.json"
    blk = sorted(x.name for x in d.iterdir() if x.name.startswith("blk_") and not x.name.endswith(".tmp"))
    if rid.exists():
        old = json.loads(rid.read_text(encoding="utf-8"))
        if old != ident:
            diff = sorted(k for k in set(old) | set(ident) if old.get(k) != ident.get(k))
            raise RunMismatch(f"{d} 의 기존 산출은 다른 실행이다(다른 항목: {', '.join(diff)}) — 보존하고 멈춘다. 다른 경로를 써라")
    elif blk:
        raise RunMismatch(f"{d} 에 실행 식별 없는 덩어리 {len(blk)} 개가 있다 — 보존하고 멈춘다")
    allowed = {f"blk_{k:05d}.{e}" for k in range(ident["n_blocks"]) for e in ("csv", "json")}
    extra = [b for b in blk if b not in allowed]
    if extra:
        raise RunMismatch(f"{d} 에 이 실행의 목록 밖 덩어리가 있다({extra[:3]}) — 보존하고 멈춘다")
    if not rid.exists():
        _write_atomic(rid, json.dumps(ident, ensure_ascii=False, sort_keys=True))


def _block_done(d: Path, k: int, run: str) -> bool:
    meta_p, csv_p = d / f"blk_{k:05d}.json", d / f"blk_{k:05d}.csv"
    if not meta_p.exists():
        return False
    m = json.loads(meta_p.read_text(encoding="utf-8"))
    if m.get("block") != k:                              # 번호가 다른 메타는 다른 CSV 를 가리킨다(공유 회신 §63 재검토)
        raise RunMismatch(f"{meta_p} 의 block 이 {m.get('block')!r} 다(파일 번호 {k}) — 보존하고 멈춘다")
    if m.get("run") != run or not csv_p.exists() or _sha_file(csv_p) != m.get("csv_sha256"):
        raise RunMismatch(f"{meta_p} 가 이 실행의 것이 아니거나 CSV 해시가 다르다 — 보존하고 멈춘다")
    return True


def _hist(x: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    return np.histogram(np.clip(x, -1.0, 1.0 - 1e-9), bins=BINS)[0].astype(np.int64)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def run_pairs(
    name: str,
    a_ids: Sequence[str],
    b_ids: Sequence[str],
    Da: np.ndarray,
    Db: np.ndarray,
    *,
    within: bool,
    out_dir: Path,
    floors: dict[str, float],
    wait: Callable[[], dict],
    mem_now: Callable[[], float],
    start_min_gb: float,
    block: int = 100,
    workers: int = 2,
    do_p: bool = True,
    p_min_overlap: float = 0.2,
    log: Callable[[str], None] = print,
) -> dict:
    """행 `a_ids` 대 열 `b_ids`. `within` 이면 같은 목록이어야 하고 위 삼각(j > i)만 본다. 반환은 합친 요약."""
    if within and list(a_ids) != list(b_ids):
        raise ValueError("within 은 같은 목록이어야 한다")
    d = out_dir / name
    ident = run_identity(name, a_ids, b_ids, Da, Db, within=within, floors=floors, block=block, workers=workers,
                         do_p=do_p, p_min_overlap=p_min_overlap)
    bind_run(d, ident)
    m0 = mem_now()
    if m0 < start_min_gb:
        raise StartRefused(f"여유 메모리 {m0:.1f} GB < 시작 조건 {start_min_gb} GB")
    Dna, Dnb = S.normalize(Da), S.normalize(Db)
    Fa = S.spectra(Dna, workers)
    Fb = Fa if within else S.spectra(Dnb, workers)
    win = S._Win(min_overlap=p_min_overlap) if do_p else None
    n_blocks = (len(a_ids) + block - 1) // block
    for k in range(n_blocks):
        meta_p = d / f"blk_{k:05d}.json"
        if _block_done(d, k, ident["run"]):
            continue
        g = wait()
        t0 = time.perf_counter()
        rows: list[tuple] = []
        hist = {key: np.zeros(200, dtype=np.int64) for key in GENS}
        n_pairs, t_g, t_p = 0, 0.0, 0.0
        for i in range(k * block, min((k + 1) * block, len(a_ids))):
            j0 = i + 1 if within else 0
            if j0 >= len(b_ids):
                continue
            n = len(b_ids) - j0
            ta = time.perf_counter()
            g4, g4y, g4x = S.g4_peaks(Fa[i], Fb[j0:], workers)
            g3 = Dnb[j0:].reshape(n, -1) @ Dna[i].ravel()
            t_g += time.perf_counter() - ta
            if do_p:
                tb = time.perf_counter()
                p, py, px = S.psearch_batch(Da[i], Db[j0:], win, workers)
                t_p += time.perf_counter() - tb
            else:
                p = np.full(n, np.nan)
                py = px = np.zeros(n, dtype=np.int32)
            n_pairs += n
            for key, v in (("g3", g3), ("g4", g4), ("p", p)):
                hist[key] += _hist(np.asarray(v, dtype=np.float64))
            keep = (g3 >= floors["g3"]) | (g4 >= floors["g4"]) | (np.nan_to_num(p, nan=-9.0) >= floors["p"])
            for jj in np.flatnonzero(keep):
                a, b = a_ids[i], b_ids[j0 + jj]
                if a == b:
                    continue
                sgn = 1
                if a > b:
                    a, b, sgn = b, a, -1
                pv = "" if not np.isfinite(p[jj]) else round(float(p[jj]), 5)
                rows.append((a, b, round(float(g3[jj]), 5), round(float(g4[jj]), 5), sgn * int(g4x[jj]),
                             sgn * int(g4y[jj]), pv, sgn * int(px[jj]), sgn * int(py[jj])))
        rows.sort(key=lambda r: (r[0], r[1]))
        csv_p = d / f"blk_{k:05d}.csv"
        tmp = csv_p.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(HEAD)
            w.writerows(rows)
        os.replace(tmp, csv_p)
        _write_atomic(meta_p, json.dumps({
            "block": k, "rows": [k * block, min((k + 1) * block, len(a_ids))], "n_pairs": n_pairs, "n_kept": len(rows),
            "sec_total": round(time.perf_counter() - t0, 3), "sec_g3g4": round(t_g, 3), "sec_p": round(t_p, 3),
            "mem_at_start_gb": g.get("mem_free_gb"), "hist": {key: v.tolist() for key, v in hist.items()},
            "run": ident["run"], "csv_sha256": _sha_file(csv_p),
        }, ensure_ascii=False))
        log(f"  {name} 덩어리 {k + 1}/{n_blocks} · 짝 {n_pairs:,} · 남김 {len(rows):,} · {time.perf_counter() - t0:.1f}초")
    return summarize(d, ident["run"])


def _run_metas(d: Path, expect_run: str | None = None) -> tuple[dict, list[dict]]:
    """그 실행의 덩어리 목록(0 … n−1)만 — 실행 해시와 CSV 해시를 확인한다. 빠진 덩어리는 건너뛰지 않고 멈춘다.

    `expect_run` 을 주면 **읽기 전에** 디스크의 실행 해시가 지금 기대하는 실행(`run_identity(...)["run"]`)과 같은지 본다 —
    자기 일관성만으로는 옛 입력의 온전한 스캔이 통과한다(공유 회신 §64-2).
    """
    rid = d / "RUN.json"
    if not rid.exists():
        raise RunMismatch(f"{d} 에 실행 식별(RUN.json)이 없다 — 읽지 않는다")
    ident = json.loads(rid.read_text(encoding="utf-8"))
    if expect_run is not None and ident.get("run") != expect_run:
        raise RunMismatch(f"{d} 의 실행 해시가 지금 기대하는 실행과 다르다 — 보존하고 읽지 않는다")
    metas = []
    for k in range(ident["n_blocks"]):
        if not _block_done(d, k, ident["run"]):
            raise RunMismatch(f"{d} 의 덩어리 {k} 가 없다 — 실행이 끝나지 않았다")
        metas.append(json.loads((d / f"blk_{k:05d}.json").read_text(encoding="utf-8")))
    return ident, metas


def summarize(d: Path, expect_run: str | None = None) -> dict:
    ident, metas = _run_metas(d, expect_run)
    hist = {k: (np.sum([m["hist"][k] for m in metas], axis=0).tolist() if metas else [0] * 200) for k in GENS}
    return {"blocks": len(metas), "n_pairs": int(sum(m["n_pairs"] for m in metas)),
            "n_kept": int(sum(m["n_kept"] for m in metas)), "sec_total": round(sum(m["sec_total"] for m in metas), 3),
            "sec_g3g4": round(sum(m["sec_g3g4"] for m in metas), 3), "sec_p": round(sum(m["sec_p"] for m in metas), 3),
            "hist": hist, "run": ident["run"]}


def rate_at(hist: list[int], thr: float) -> float:
    """0.01 칸 분포에서 문턱 이상의 비율(칸 경계에 맞는 문턱만 정확하다)."""
    h = np.asarray(hist)
    k = round((thr + 1.0) * 100)
    return float(h[k:].sum() / h.sum()) if h.sum() else float("nan")


def read_rows(d: Path, expect_run: str | None = None) -> list[dict]:
    """그 실행의 덩어리를 **검증한 경로 그대로**(파일 번호 k) 읽는다 — 메타의 값으로 경로를 다시 고르지 않는다."""
    ident, _ = _run_metas(d, expect_run)
    out = []
    for k in range(ident["n_blocks"]):
        with (d / f"blk_{k:05d}.csv").open(encoding="utf-8", newline="") as fh:
            out += list(csv.DictReader(fh))
    return out
