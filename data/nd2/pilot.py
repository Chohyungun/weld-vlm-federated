"""간선 v2 파일럿 — 평가 100 장 · 학습검증 1,000 장(미니스펙 2판 5절, 게이트 회차 1). **판정하지 않는다.**

    python -m data.nd2.pilot <단계>     단계: sample · metacheck · scan · calib · hops · attach · recut · all

- 실물을 읽는 단계(scan · calib · hops · attach · recut)는 **같은 입력에 대한 성공한 메타 대조**(`metacheck.json` 의 `ok` 와
  입력 해시 · 입력 · 사슬 설정이 지금과 같음)가 있어야 시작한다. 그다음 여유 메모리가 `start_min_gb` 이상이어야 하고, 덩어리마다
  `chunk_min_gb` 아래거나 본체 게이트 락이 있으면 기다린다. 작업자 하나.
- 이어 하기는 **실행 식별이 같을 때만** 한다 — 스캔은 폴더마다 `RUN.json`(`scan.bind_run`), 규칙 입력 · 보정은 `*_RUN.json`,
  표본은 같은 바이트일 때만. 다르면 기존 산출을 그대로 두고 멈춘다(종료 코드 4). 메타 대조 실패는 종료 코드 3(공유 회신 §63).
- **소비 쪽도 대조한다** — 규칙 입력은 스캔을, 홉 · 규칙 입력은 보정을 읽기 전에 지금 설정 · 입력에서 기대하는 실행 식별과 남은 산출의
  식별을 맞춘다(다르면 종료 코드 4). 메타 대조의 재사용 식별에 `configs/base.yaml` 해시를 넣는다. 출력 경로는 `_workspace/` · `outputs/`
  아래만(동결 자리 · 그 밖은 종료 코드 5, 만들기 전에).
- 원천 · 동결 자산은 읽기만 한다. 다시 자른 타일은 메모리에서만 쓴다. 그림을 산출에 싣지 않는다(수 · 해시만).
- `metacheck` 는 그림을 읽지 않는다(매니페스트 · 진행 기록 · 기술자 id 의 대조) — 메모리 조건 없이 돈다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import Counter, OrderedDict, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from data.nd2 import calib as C
from data.nd2 import features as F
from data.nd2 import recut as R
from data.nd2 import scan as SC
from data.vt.guard import Guard, gate_lock_path, mem_free_gb

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "neardup_v2.yaml"
csv.field_size_limit(10_000_000)


# ------------------------------------------------------------------------------------------
# 공통
# ------------------------------------------------------------------------------------------
def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def log(msg: str) -> None:
    print(msg, flush=True)


def make_guard(cfg: dict, out: Path) -> Guard:
    r = cfg["resources"]
    logp = out / "guard_waits.jsonl"

    def _log(d: dict) -> None:
        d = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **d}
        with logp.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        log(f"  기다림: 여유 메모리 {d.get('mem_free_gb')} GB · 게이트 {d.get('gate')}")

    return Guard(c_free_min_gb=0.0, mem_free_min_gb=float(r["chunk_min_gb"]), c_wait_max_s=1e12,
                 wait_step_s=float(r["wait_step_s"]), gate_path=gate_lock_path(ROOT), log=_log, check_c=False)


class MetaCheckFailed(RuntimeError):
    """메타 대조가 실패했거나, 같은 입력에 대한 성공한 대조가 없다."""


INPUT_KEYS = ("manifest", "tiles", "encode_progress", "manifest_v0", "histmatch_reference", "border_mask_calibration",
              "desc", "desc_ids")


#: 메타 대조가 읽는 설정 파일 — 입력과 함께 해시를 묶는다(공유 회신 §63-2 재검토: `base.yaml` 변경을 못 잡았다)
BASE_CFG = Path("configs") / "base.yaml"


def input_hashes(cfg: dict) -> dict[str, str]:
    h = {k: sha256_file(ROOT / cfg["inputs"][k]) for k in INPUT_KEYS}
    h["base_yaml"] = sha256_file(ROOT / BASE_CFG)
    return h


class OutPathForbidden(RuntimeError):
    """출력 경로가 허락된 뿌리(작업 공간 · outputs) 밖이거나 동결 자리다."""


def check_out_path(out: Path) -> Path:
    """파일럿 출력은 `_workspace/` 또는 `outputs/` 아래만(정션을 풀어 비교). `data/` · 동결 자리 · 그 밖은 거부 — 만들기 전에."""
    from data.frozen_guard import is_frozen

    o = Path(out).resolve()
    roots = [(ROOT / "_workspace").resolve(), (ROOT / "outputs").resolve()]
    if not any(o == r or r in o.parents for r in roots):
        raise OutPathForbidden(f"출력 경로 {out} 는 _workspace/ · outputs/ 밖이다 — 만들지 않는다")
    if any(is_frozen(x) for x in (o, *o.parents)):
        raise OutPathForbidden(f"출력 경로 {out} 는 동결 자리(또는 그 안)다 — 만들지 않는다")
    return o


def _with_run(ident: dict) -> dict:
    return {**ident, "run": hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()}


def require_current(run_json: Path, ident: dict, what: str) -> str:
    """소비하기 전에 — 생산 단계가 남긴 실행 식별이 지금 설정 · 입력에서 기대하는 것과 같은가. 다르면 `RunMismatch`."""
    want = _with_run(ident) if "run" not in ident else ident
    if not run_json.exists():
        raise SC.RunMismatch(f"{what}: 실행 식별 {run_json.name} 가 없다 — 생산 단계를 먼저 돌려라")
    have = json.loads(run_json.read_text(encoding="utf-8"))
    if have != want:
        diff = sorted(k for k in set(have) | set(want) if have.get(k) != want.get(k))
        raise SC.RunMismatch(f"{what}: 남은 산출은 지금 설정 · 입력의 실행이 아니다(다른 항목: {', '.join(diff)}) — 소비하지 않는다")
    return want["run"]


def require_metacheck(cfg: dict, out: Path, step: str) -> None:
    """실물 단계의 공통 관문 — 같은 입력 · 같은 설정에 대한 성공한 메타 대조가 있어야 한다(공유 회신 §63-2)."""
    p = out / "metacheck.json"
    if not p.exists():
        raise MetaCheckFailed(f"[{step}] 메타 대조 기록이 없다 — metacheck 를 먼저 돌려라")
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("ok") is not True:
        raise MetaCheckFailed(f"[{step}] 메타 대조가 실패한 상태다: {m.get('failures')}")
    if m.get("inputs_cfg") != cfg["inputs"] or m.get("chain_cfg") != cfg["chain"]:
        raise MetaCheckFailed(f"[{step}] 메타 대조 때와 입력 · 사슬 설정이 다르다 — metacheck 를 다시 돌려라")
    now = input_hashes(cfg)
    if m.get("input_sha256") != now:
        diff = sorted(k for k in now if m.get("input_sha256", {}).get(k) != now[k])
        raise MetaCheckFailed(f"[{step}] 메타 대조 뒤 입력이 바뀌었다({', '.join(diff)}) — metacheck 를 다시 돌려라")


def bind_outputs(run_json: Path, ident: dict, outputs: list[Path]) -> None:
    """산출 파일들을 실행 식별에 묶는다 — 기록이 다르거나, 기록 없이 산출만 있으면 지우지 않고 `RunMismatch`."""
    ident = _with_run(ident)
    if run_json.exists():
        old = json.loads(run_json.read_text(encoding="utf-8"))
        if old != ident:
            diff = sorted(k for k in set(old) | set(ident) if old.get(k) != ident.get(k))
            raise SC.RunMismatch(f"{run_json.name}: 기존 산출은 다른 실행이다(다른 항목: {', '.join(diff)}) — 보존하고 멈춘다")
        return
    have = [o.name for o in outputs if o.exists()]
    if have:
        raise SC.RunMismatch(f"실행 식별 없는 산출 {have} — 보존하고 멈춘다")
    write_json(run_json, ident)


def _sha_rows(rows: list) -> str:
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def require_start(cfg: dict, step: str, out: Path | None = None) -> float:
    if out is not None:
        require_metacheck(cfg, out, step)
    m = mem_free_gb()
    need = float(cfg["resources"]["start_min_gb"])
    if m < need:
        raise SC.StartRefused(f"[{step}] 여유 메모리 {m:.1f} GB < 시작 조건 {need} GB — 시작하지 않는다")
    return m


def read_manifest(cfg: dict) -> dict[str, dict]:
    with (ROOT / cfg["inputs"]["manifest"]).open(encoding="utf-8", newline="") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def read_progress(cfg: dict) -> dict[str, dict]:
    out = {}
    with (ROOT / cfg["inputs"]["encode_progress"]).open(encoding="utf-8") as fh:
        for line in fh:
            o = json.loads(line)
            out[o["image_id"]] = o
    return out


def desc_index(cfg: dict) -> tuple[list[str], dict[str, int], np.ndarray]:
    ids = (ROOT / cfg["inputs"]["desc_ids"]).read_text(encoding="utf-8").split("\n")[:-1]
    D = np.load(ROOT / cfg["inputs"]["desc"], mmap_mode="r")
    if D.shape[0] != len(ids):
        raise RuntimeError(f"기술자 {D.shape[0]} 행 ≠ id {len(ids)} 개")
    return ids, {k: i for i, k in enumerate(ids)}, D


def set_of(split: str) -> str:
    return "E" if split == "eval" else "T"


def rank_key(seed: int, iid: str) -> str:
    return hashlib.sha256(f"{seed}|{iid}".encode()).hexdigest()


def read_sample(out: Path) -> tuple[list[str], list[str]]:
    E, T = [], []
    with (out / "sample.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            (E if r["set"] == "E" else T).append(r["image_id"])
    return E, T


# ------------------------------------------------------------------------------------------
# 단계
# ------------------------------------------------------------------------------------------
def step_sample(cfg: dict, out: Path) -> None:
    """평가 · 학습검증에서 공개 시드의 sha256 순으로 앞에서 n 장. 결정적이다(시험)."""
    p = cfg["pilot"]
    man = read_manifest(cfg)
    ids, _, _ = desc_index(cfg)
    have = set(ids)
    by = defaultdict(list)
    for iid, r in man.items():
        if iid in have:
            by[set_of(r["split"])].append(iid)
    pick = {s: sorted(v, key=lambda i: (rank_key(int(p["seed"]), i), i)) for s, v in by.items()}
    rows = [("E", i) for i in sorted(pick["E"][: int(p["n_eval"])])] + [("T", i) for i in sorted(pick["T"][: int(p["n_trainval"])])]
    text = "set,image_id\n" + "".join(f"{s},{i}\n" for s, i in rows)
    dst = out / "sample.csv"
    if dst.exists():
        if dst.read_text(encoding="utf-8") != text:
            raise SC.RunMismatch(f"{dst} 가 지금 설정의 표본과 다르다 — 덮어쓰지 않고 멈춘다. 다른 출력 경로를 써라")
    else:
        dst.write_text(text, encoding="utf-8", newline="\n")
    log(f"표본: 평가 {int(p['n_eval'])} · 학습검증 {int(p['n_trainval'])} (모집단 평가 {len(by['E']):,} · 학습검증 {len(by['T']):,})")


def step_metacheck(cfg: dict, out: Path) -> dict:
    """그림을 읽지 않는 대조 — 원점 집합이 실제 상자를 품는가, 기술자 id 가 매니페스트와 같은가, 입력 해시."""
    man = read_manifest(cfg)
    prog = read_progress(cfg)
    with (ROOT / cfg["inputs"]["manifest_v0"]).open(encoding="utf-8", newline="") as fh:
        v0 = {r["image_id"]: (int(r["width_px"]), int(r["height_px"])) for r in csv.DictReader(fh)}
    with (ROOT / cfg["inputs"]["tiles"]).open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    ids, _, D = desc_index(cfg)
    ch = cfg["chain"]
    pano = [i for i, o in prog.items() if o["reason"] != "ok"]
    contained, not_contained, y_bad, widths = 0, [], [], []
    for i in sorted(pano):
        w, h = v0[i]
        box = prog[i]["box"]
        xs = R.origins(w)
        widths.append(w)
        if box[0] in xs and box[2] - box[0] == ch["tile_w"] and box[3] - box[1] == ch["tile_h"]:
            contained += 1
        else:
            not_contained.append(i)
        if not (0 <= box[1] and box[3] <= h and box[1] % ch["align"] == 0):
            y_bad.append(i)
    n_orig = sum(len(R.origins(w)) for w in widths)
    split = Counter(set_of(man[i]["split"]) for i in pano)
    orig_by_set = Counter()
    for i, w in zip(sorted(pano), widths, strict=True):
        orig_by_set[set_of(man[i]["split"])] += len(R.origins(w))
    res = {
        "panoramas": len(pano), "panoramas_by_set": dict(split), "provenance": dict(Counter(prov[i] for i in pano)),
        "origin_contains_actual_box": contained, "origin_not_contained": not_contained[:20],
        "n_origin_not_contained": len(not_contained), "y_out_of_range_or_unaligned": len(y_bad),
        "origins_total": n_orig, "origins_by_set": dict(orig_by_set),
        "desc_ids_equal_manifest": sorted(ids) == sorted(man), "desc_rows": int(D.shape[0]),
        "manifest_rel_in_histmatch": sum(1 for i in man if man[i]["rel_path"].startswith("data/interim/tiles_v1_histmatch/")),
        "input_sha256": input_hashes(cfg), "inputs_cfg": cfg["inputs"], "chain_cfg": cfg["chain"],
    }
    bm = json.loads((ROOT / cfg["inputs"]["border_mask_calibration"]).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    enc = base["preprocess"]["tile"]["encode"]
    t = base["preprocess"]["tile"]
    res["chain_settings_match"] = {
        "quality": enc["quality"] == ch["quality"], "mode_L": enc["mode"] == "L",
        "progressive_false": enc["progressive"] is False, "optimize_false": enc["optimize"] is False,
        "tile_size": list(t["tile_size"]) == [ch["tile_w"], ch["tile_h"]], "stride_x": t["stride_x"] == ch["stride_x"],
        "align8": bool(t["align8"]) == (ch["align"] == 8), "fill": bm["fill_value"], "border_frac": bm["border_frac"],
    }
    failures = []
    if not_contained:
        failures.append(f"원점 집합이 실제 상자를 품지 못함 {len(not_contained)}")
    if y_bad:
        failures.append(f"세로 상자 위반 {len(y_bad)}")
    if not res["desc_ids_equal_manifest"]:
        failures.append("기술자 id ≠ 매니페스트 id")
    if res["manifest_rel_in_histmatch"] != len(man):
        failures.append("매니페스트 경로가 정합 타일이 아닌 행이 있다")
    failures += [f"사슬 설정 불일치 {k}" for k, v in res["chain_settings_match"].items() if isinstance(v, bool) and not v]
    res["ok"], res["failures"] = not failures, failures
    write_json(out / "metacheck.json", res)
    log(f"메타 대조: 파노라마 {len(pano):,} · 원점이 실제 상자를 품음 {contained:,} · 못 품음 {len(not_contained)} · 원점 합 {n_orig:,}")
    if failures:
        raise MetaCheckFailed("메타 대조 실패 — " + "; ".join(failures))
    return res


def _subset(D: np.ndarray, pos: dict[str, int], ids: list[str]) -> np.ndarray:
    idx = np.array([pos[i] for i in ids])
    order = np.argsort(idx)
    out = np.empty((len(ids), 36, 64), dtype=np.float32)
    out[order] = D[idx[order]]
    return out


SCAN_SETS = ("EE", "TT", "ET")


def _scan_args(cfg: dict, out: Path) -> dict[str, tuple]:
    """스캔 셋의 (행 id, 열 id, 행 기술자, 열 기술자, within) — 생산(step_scan)과 소비 쪽 대조가 같은 것을 쓴다."""
    E, T = read_sample(out)
    _, pos, D = desc_index(cfg)
    De, Dt = _subset(D, pos, E), _subset(D, pos, T)
    return {"EE": (E, E, De, De, True), "TT": (T, T, Dt, Dt, True), "ET": (E, T, De, Dt, False)}


def _scan_params(cfg: dict) -> dict:
    p, r = cfg["pilot"], cfg["resources"]
    return {"floors": p["record_floors"], "block": int(r["block_rows"]), "workers": int(r["fft_workers"]),
            "do_p": True, "p_min_overlap": float(p["p_min_overlap"])}


def scan_expected(cfg: dict, out: Path) -> dict[str, dict]:
    """지금 표본 · 기술자 · 설정 · 코드로 스캔을 돌리면 남을 실행 식별(`scan.run_identity`)."""
    prm = _scan_params(cfg)
    return {n: SC.run_identity(n, a, b, Da, Db, within=w, **prm) for n, (a, b, Da, Db, w) in _scan_args(cfg, out).items()}


def require_scan_current(cfg: dict, out: Path) -> dict[str, str]:
    """스캔 산출을 소비하기 전에(공유 회신 §63-1 재검토) — 셋마다 `RUN.json` 이 지금 기대와 같아야 한다."""
    return {n: require_current(out / "scan" / n / "RUN.json", ident, f"스캔 {n}") for n, ident in scan_expected(cfg, out).items()}


def step_scan(cfg: dict, out: Path) -> dict:
    """표본 안 짝 — 평가 안(E·E) · 학습검증 안(T·T) · 교차(E·T)에 G3 · G4 · P 탐색. 분포 · 시간 · 바닥 이상 짝."""
    require_start(cfg, "scan", out)
    g = make_guard(cfg, out)
    prm = _scan_params(cfg)
    common = {"out_dir": out / "scan", "floors": prm["floors"], "wait": g.wait, "mem_now": mem_free_gb,
              "start_min_gb": 0.0, "block": prm["block"], "workers": prm["workers"], "do_p": prm["do_p"],
              "p_min_overlap": prm["p_min_overlap"], "log": log}
    res = {n: SC.run_pairs(n, a, b, Da, Db, within=w, **common) for n, (a, b, Da, Db, w) in _scan_args(cfg, out).items()}
    summ = {}
    for k, s in res.items():
        summ[k] = {"n_pairs": s["n_pairs"], "n_kept": s["n_kept"],
                   "us_per_pair_g3g4": round(1e6 * s["sec_g3g4"] / max(s["n_pairs"], 1), 3),
                   "us_per_pair_p": round(1e6 * s["sec_p"] / max(s["n_pairs"], 1), 3),
                   "rate": {f"{gname}>={t}": SC.rate_at(s["hist"][gname], t)
                            for gname, ts in (("g4", (0.35, 0.40, 0.45, 0.50)), ("g3", (0.3, 0.4, 0.5, 0.6, 0.7)),
                                              ("p", (0.6, 0.7, 0.8, 0.9))) for t in ts},
                   "hist": s["hist"]}
    write_json(out / "scan_summary.json", summ)
    log("스캔 요약: " + " · ".join(f"{k} 짝 {v['n_pairs']:,} G4≥0.45 {v['rate']['g4>=0.45']:.5f}" for k, v in summ.items()))
    return summ


def hop_g3_taus(cfg: dict, out: Path) -> tuple[list[float], list[float]]:
    """홉 도달에 쓰는 G3 문턱 후보 — 합성 양성의 출처별 5 백분위(작은 변환)의 최솟값을 0.01 아래로 내린 것.

    원형 교차상관의 이동 0 값이 G3 이므로 늘 G4 ≥ G3 다. G3 문턱이 G4 문턱(0.45) 이상이면 "G4 또는 G3" 는 G4 와 같은 집합이라
    따로 세지 않는다(두 번째 목록으로 돌려준다). 고르는 일이 아니다 — 도달의 위 끝을 재려는 값이다.
    """
    runs = require_calib_current(cfg, out)
    summ = json.loads((out / "calib_summary.json").read_text(encoding="utf-8"))
    if summ.get("runs") != runs:
        raise SC.RunMismatch("calib_summary.json 이 지금 보정 실행의 요약이 아니다 — 소비하지 않는다")
    if summ.get("sources") != calib_sources(out):       # 요약을 낸 원천 CSV 가 그대로인가(공유 회신 §64-2)
        raise SC.RunMismatch("calib_summary.json 을 낸 원천 CSV 가 지금 파일과 다르다 — 소비하지 않는다")
    s = summ["g3_synthetic_positive"]
    tau = float(np.floor(100 * min(v["p5_small"] for v in s.values())) / 100)
    g4 = float(cfg["pilot"]["hop_g4"])
    return ([tau], []) if tau < g4 else ([], [tau])


def hop_sets(h1: set, expand, with_groups) -> dict[str, set]:
    """변형마다 출발 집합을 나눈다(공유 회신 §63-3) — 묶음 제외는 h1 에서, 묶음 포함은 h1 + 묶음 동료에서 넓힌다."""
    h1g = with_groups(h1)
    return {"h1": set(h1), "h1_with_groups": h1g, "h2": set(h1) | expand(set(h1)),
            "h2_with_groups": with_groups(h1g | expand(h1g))}


def step_hops(cfg: dict, out: Path) -> dict:
    """평가 표본에서 학습검증 전체로 1 홉 · 2 홉(G4, 그리고 G4 또는 G3 ≥ 합성 문턱). P 탐색은 넣지 않는다. 보정(calib) 다음에 돈다.

    정의(변형 v 의 생성기 짝을 N_v): h1 = 평가 표본과 짝인 학습검증 영상, h1_with_groups = h1 ∪ 묶음 동료,
    **h2 = h1 ∪ N_v(h1)**(묶음 제외), **h2_with_groups = 묶음(h1_with_groups ∪ N_v(h1_with_groups))**(묶음 포함).
    """
    require_start(cfg, "hops", out)
    p, r = cfg["pilot"], cfg["resources"]
    E, _ = read_sample(out)
    man = read_manifest(cfg)
    ids, pos, D = desc_index(cfg)
    TV = sorted(i for i in ids if man[i]["split"] != "eval")
    Dtv = _subset(D, pos, TV)
    De = _subset(D, pos, E)
    groups = defaultdict(set)
    for i in TV:
        groups[man[i]["group_id"]].add(i)
    g = make_guard(cfg, out)
    taus, same_as_g4 = hop_g3_taus(cfg, out)
    lo_g3 = min(taus) if taus else 9.0
    floors = {"g3": lo_g3, "g4": float(p["hop_g4"]), "p": 9.0}
    common = {"out_dir": out / "hops", "floors": floors, "wait": g.wait, "mem_now": mem_free_gb, "start_min_gb": 0.0,
                  "block": int(r["block_rows"]), "workers": int(r["fft_workers"]), "do_p": False, "log": log}
    h1_run = SC.run_pairs("H1", E, TV, De, Dtv, within=False, **common)["run"]
    h1_rows = SC.read_rows(out / "hops" / "H1", expect_run=h1_run)
    eset = set(E)

    def hit(row, g3t):
        return float(row["g4"]) >= float(p["hop_g4"]) or (g3t is not None and float(row["g3"]) >= g3t)

    def reach1(rows, g3t, seeds):
        s = set()
        for row in rows:
            a, b = row["a_id"], row["b_id"]
            e, t = (a, b) if a in eset else (b, a)
            if e in seeds and hit(row, g3t):
                s.add(t)
        return s

    def with_groups(s):
        out_ = set(s)
        for t in s:
            out_ |= groups[man[t]["group_id"]]
        return out_

    variants = [None, *taus]
    cap = int(p["hop_seed_cap"])
    ordered = sorted(E, key=lambda i: (rank_key(int(p["seed"]), i), i))   # id 순은 시기 · 재질과 얽힌다(함정 12)
    steps = [int(n) for n in p["hop_seed_steps"]]
    # 2 홉은 (변형, 평가 시작 수)마다 그 1 홉 + 묶음 동료가 상한 이하일 때만 잰다 — 작은 시작 수에서라도 2 홉을 얻는다
    h1g_of = {(v, n): with_groups(reach1(h1_rows, v, set(ordered[:n]))) for v in variants for n in steps}
    measured2 = {key for key, s_ in h1g_of.items() if len(s_) <= cap}
    seeds2 = sorted(set().union(*(h1g_of[k] for k in measured2))) if measured2 else []
    sp = out / "hops" / "H2_seeds.txt"
    text = "".join(f"{i}\n" for i in seeds2)
    if sp.exists() and sp.read_text(encoding="utf-8") != text:
        raise SC.RunMismatch("H2 시작 영상이 지난 실행과 다르다 — 보존하고 멈춘다. 다른 출력 경로를 써라")
    sp.write_text(text, encoding="utf-8", newline="\n")
    log(f"2 홉 시작 영상 {len(seeds2):,} 장(상한 {cap:,} 안의 (변형, 시작 수)들의 1 홉 + 묶음 동료)")
    nbr = defaultdict(list)
    if seeds2:
        h2_run = SC.run_pairs("H2", seeds2, TV, _subset(D, pos, seeds2), Dtv, within=False, **common)["run"]
        for row in SC.read_rows(out / "hops" / "H2", expect_run=h2_run):
            nbr[row["a_id"]].append(row)
            nbr[row["b_id"]].append(row)
    seed2_set = set(seeds2)

    def expand(start: set, g3t) -> set:
        """출발 집합의 각 영상과 생성기 짝이 되는 학습검증 영상(출발 집합은 2 홉 스캔의 시작 영상 안에 있어야 한다)."""
        if not start <= seed2_set:
            raise RuntimeError("2 홉 출발 집합이 스캔한 시작 영상 밖에 있다")
        got = set()
        for t in start:
            for row in nbr.get(t, []):
                if hit(row, g3t):
                    got.add(row["b_id"] if row["a_id"] == t else row["a_id"])
        return got
    res = {"trainval": len(TV), "g3_taus": taus, "g3_taus_dropped_as_same_as_g4": same_as_g4,
           "seed_cap": cap, "h2_seeds": len(seeds2), "seed_steps": {}}
    for n in steps:
        seeds = set(ordered[:n])
        per = {}
        for g3t in variants:
            h1 = reach1(h1_rows, g3t, seeds)
            h1g = h1g_of[(g3t, n)]
            row_ = {"h1": len(h1), "h1_frac": round(len(h1) / len(TV), 5), "h1_with_groups": len(h1g),
                    "h1_with_groups_frac": round(len(h1g) / len(TV), 5)}
            if (g3t, n) in measured2:
                hs = hop_sets(h1, lambda s_, v=g3t: expand(s_, v), with_groups)
                h2, h2g = hs["h2"], hs["h2_with_groups"]
                row_ |= {"h2": len(h2), "h2_frac": round(len(h2) / len(TV), 5), "h2_with_groups": len(h2g),
                         "h2_with_groups_frac": round(len(h2g) / len(TV), 5)}
            else:
                row_ |= {"h2": None, "note": f"1 홉 + 묶음 동료 {len(h1g):,} 장이 상한 {cap:,} 을 넘어 2 홉은 재지 않았다"}
            per["g4" if g3t is None else f"g4|g3>={g3t}"] = row_
        res["seed_steps"][str(n)] = per
    write_json(out / "hops.json", res)
    log("홉: " + json.dumps(res["seed_steps"][str(p["hop_seed_steps"][-1])], ensure_ascii=False))
    return res


class _TileCache:
    def __init__(self, man: dict[str, dict], n: int = 128) -> None:
        self.man, self.n, self.d = man, n, OrderedDict()
        self.sec_decode = 0.0

    def get(self, iid: str) -> F.Prepared:
        if iid in self.d:
            self.d.move_to_end(iid)
            return self.d[iid]
        t0 = time.perf_counter()
        with Image.open(ROOT / self.man[iid]["rel_path"]) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32)[R.Y0:R.Y1, R.X0:R.X1]
        v = F.prepare(a)
        self.sec_decode += time.perf_counter() - t0
        self.d[iid] = v
        if len(self.d) > self.n:
            self.d.popitem(last=False)
        return v


def step_attach(cfg: dict, out: Path) -> dict:
    """표본 스캔의 후보 일부에 규칙 입력(두 가설 정합 · 겹친 자리 특징)을 단다 — 짝당 시간을 잰다. 판정하지 않는다."""
    require_start(cfg, "attach", out)
    p, fcfg = cfg["pilot"], cfg["features"]
    man = read_manifest(cfg)
    g = make_guard(cfg, out)
    scan_runs = require_scan_current(cfg, out)          # 소비 전에 — 지금 설정 · 입력의 스캔인가
    rows = []
    for name in SCAN_SETS:
        for row in SC.read_rows(out / "scan" / name, expect_run=scan_runs[name]):
            row["set_pair"] = name
            rows.append(row)
    # 생성기별 층: G4 만 · P 만 · G3 만 · 둘 이상 — 층마다 고르게(정렬은 공개 시드의 해시)
    # 층은 생성기의 겹침으로 나눈다. G4 ≥ G3 이므로 "G3 만" 은 G3 문턱(합성)이 G4 문턱보다 낮을 때만 생긴다.
    # P 의 0.8 은 층을 나누는 값일 뿐 후보 문턱이 아니다.
    taus, _ = hop_g3_taus(cfg, out)

    def stratum(row):
        g4 = float(row["g4"]) >= float(p["hop_g4"])
        pp = row["p"] != "" and float(row["p"]) >= 0.8
        g3 = bool(taus) and float(row["g3"]) >= taus[0]
        if g4 and pp:
            return "g4+p"
        if g4:
            return "g4"
        if pp:
            return "p"
        return "g3" if g3 else "floor"
    by = defaultdict(list)
    for row in rows:
        by[stratum(row)].append(row)
    k = int(p["attach_max_pairs"])
    pick = []
    for s in sorted(by):
        v = sorted(by[s], key=lambda r_: (rank_key(int(p["seed"]), r_["a_id"] + "|" + r_["b_id"]), r_["a_id"], r_["b_id"]))
        pick += v[: max(1, k // len(by))]
    cache = _TileCache(man)
    outp = out / "attach.csv"
    pick_key = [[r_["a_id"], r_["b_id"], r_["set_pair"], r_["g4_dx"], r_["g4_dy"], r_["p_dx"], r_["p_dy"], r_["p"]]
                for r_ in pick]
    meta = json.loads((out / "metacheck.json").read_text(encoding="utf-8"))
    bind_outputs(out / "attach_RUN.json", {
        "pick_sha256": _sha_rows(pick_key), "n_pick": len(pick), "features": fcfg, "hop_g4": p["hop_g4"], "g3_taus": taus,
        "code": SC.code_version(("ncc.py", "features.py")), "input_sha256": meta["input_sha256"],
        "scan_runs": scan_runs}, [outp])
    pick_set = {(r_["a_id"], r_["b_id"]) for r_ in pick}
    done = set()
    if outp.exists():
        with outp.open(encoding="utf-8", newline="") as fh:
            done = {(r_["a_id"], r_["b_id"]) for r_ in csv.DictReader(fh)}
        if not done <= pick_set:
            raise SC.RunMismatch("attach.csv 에 이 실행의 고른 짝 밖의 행이 있다 — 보존하고 멈춘다")
    new = not outp.exists()
    t_all = 0.0
    with outp.open("a", encoding="utf-8", newline="\n") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["a_id", "b_id", "set_pair", "stratum", "reason", "sec", *F.HEAD])
        for n, row in enumerate(pick):
            if (row["a_id"], row["b_id"]) in done:
                continue
            if n % 50 == 0:
                g.wait()
            cells = [(int(row["g4_dx"]), int(row["g4_dy"]))]
            if row["p"] != "":
                cells.append((int(row["p_dx"]), int(row["p_dy"])))
            cells.append((0, 0))
            t0 = time.perf_counter()
            try:
                pa, pb = cache.get(row["a_id"]), cache.get(row["b_id"])
                vals, why = F.features(pa, pb, cells, window_cells=float(fcfg["window_cells"]),
                                       min_overlap=float(fcfg["compute_min_overlap"]), fold_eps=None)
            except (OSError, ValueError) as exc:
                vals, why = None, "decode_failed"
                log(f"  해독 실패 {row['a_id']} {row['b_id']}: {type(exc).__name__}")
            dt = time.perf_counter() - t0
            t_all += dt
            w.writerow([row["a_id"], row["b_id"], row["set_pair"], stratum(row), why, round(dt, 4),
                        *(vals if vals is not None else [""] * len(F.HEAD))])
    with outp.open(encoding="utf-8", newline="") as fh:
        got = list(csv.DictReader(fh))
    secs = np.array([float(r_["sec"]) for r_ in got])
    res = {"pairs": len(got), "by_stratum": dict(Counter(r_["stratum"] for r_ in got)),
           "by_reason": dict(Counter(r_["reason"] for r_ in got)),
           "sec_per_pair_mean": round(float(secs.mean()), 4) if len(secs) else None,
           "sec_per_pair_median": round(float(np.median(secs)), 4) if len(secs) else None,
           "note": "짝마다 두 영상의 해독 · 표현 준비를 포함(캐시 128 장). 생성기 칸 이동 · 접힌 짝 · 영 이동을 가설로"}
    write_json(out / "attach_summary.json", res)
    log(f"규칙 입력: 짝 {len(got)} · 짝당 {res['sec_per_pair_mean']} 초(평균)")
    return res


def _normal_label_names(cfg: dict) -> dict[str, str]:
    """정상 라벨 zip 에서 image_id → 원천 파일명(확장자 제외). id 와 파일명이 다른 10 건 때문에 파일명이 유일한 키다."""
    out = {}
    for zp in sorted((ROOT / cfg["inputs"]["labels"]).glob("*정상*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith(".json"):
                    o = json.loads(z.read(name).decode("utf-8-sig"))
                    if str(o["info"].get("type")) != "RT":
                        continue
                    out[f"aihub71761:{int(o['info']['id'])}"] = Path(str(o["image_data"]["file_name"])).stem
    return out


def _raw_members(cfg: dict) -> dict[str, tuple[Path, str]]:
    idx = {}
    for zp in sorted((ROOT / cfg["inputs"]["raw_zips"]).glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    idx[Path(name).stem] = (zp, name)
    return idx


def _chain_params(cfg: dict) -> tuple[int, int, np.ndarray]:
    ref = np.asarray(json.loads((ROOT / cfg["inputs"]["histmatch_reference"]).read_text(encoding="utf-8"))["histogram"],
                     dtype=np.int64)
    fill = int(json.loads((ROOT / cfg["inputs"]["border_mask_calibration"]).read_text(encoding="utf-8"))["fill_value"])
    return int(cfg["chain"]["quality"]), fill, np.cumsum(ref) / ref.sum()


def _pick_panoramas(cfg: dict, n: int, tv_only: bool, salt: str) -> list[str]:
    man = read_manifest(cfg)
    prog = read_progress(cfg)
    pano = [i for i, o in prog.items() if o["reason"] != "ok" and (not tv_only or man[i]["split"] != "eval")]
    return sorted(pano, key=lambda i: (rank_key(int(cfg["pilot"]["seed"]), salt + i), i))[:n]


def step_recut(cfg: dict, out: Path) -> dict:
    """자기 확인 — 표본 파노라마(공개 시드 50 장)를 원천에서 다시 잘라, 실제 원점의 정합 JPEG sha256 이 매니페스트와 같은가.
    같은 원점의 기술자가 저장 기술자와 같은가. 해독 · 다시 자르기 시간. 다르면 어긋남을 적고 멈춘다(종료 코드 3)."""
    require_start(cfg, "recut", out)
    n = int(cfg["pilot"]["recut_check_n"])
    man = read_manifest(cfg)
    prog = read_progress(cfg)
    _ids, pos, D = desc_index(cfg)
    names = _normal_label_names(cfg)
    members = _raw_members(cfg)
    q, fill, ref_cdf = _chain_params(cfg)
    g = make_guard(cfg, out)
    pick = _pick_panoramas(cfg, n, tv_only=False, salt="recut|")
    rows, bad = [], []
    for k, iid in enumerate(pick):
        if k % 10 == 0:
            g.wait()
        box = prog[iid]["box"]
        zp, mem = members[names[iid]]
        t0 = time.perf_counter()
        with zipfile.ZipFile(zp) as z:
            raw = z.read(mem)
        t1 = time.perf_counter()
        xs, Dr, sha = R.recut(raw, box[1], quality=q, fill=fill, ref_cdf=ref_cdf, want_sha_at=box[0])
        t2 = time.perf_counter()
        d_ok = float(np.abs(Dr[xs.index(box[0])] - D[pos[iid]]).max()) if box[0] in xs else None
        ok = sha == man[iid]["sha256"]
        rows.append({"image_id": iid, "n_origins": len(xs), "sha_equal": ok, "desc_max_abs_diff": d_ok,
                     "sec_read": round(t1 - t0, 4), "sec_recut": round(t2 - t1, 4)})
        if not ok:
            bad.append(iid)
    res = {"n": len(rows), "sha_equal": sum(r_["sha_equal"] for r_ in rows), "mismatch": bad,
           "desc_equal": sum(1 for r_ in rows if r_["desc_max_abs_diff"] == 0.0),
           "desc_max_abs_diff_max": max((r_["desc_max_abs_diff"] or 0.0) for r_ in rows) if rows else None,
           "origins_mean": round(float(np.mean([r_["n_origins"] for r_ in rows])), 3) if rows else None,
           "sec_read_mean": round(float(np.mean([r_["sec_read"] for r_ in rows])), 4) if rows else None,
           "sec_recut_mean": round(float(np.mean([r_["sec_recut"] for r_ in rows])), 4) if rows else None,
           "sec_recut_per_origin": round(sum(r_["sec_recut"] for r_ in rows) / max(1, sum(r_["n_origins"] for r_ in rows)), 4),
           "label_names": len(names), "raw_members": len(members), "rows": rows}
    write_json(out / "recut_check.json", res)
    log(f"모원본 자기 확인: {res['sha_equal']}/{res['n']} 해시 일치 · 기술자 일치 {res['desc_equal']} · "
        f"파노라마당 {res['sec_read_mean']}+{res['sec_recut_mean']} 초")
    if bad:
        log(f"!! 해시 불일치 {len(bad)} 장 — 멈춘다")
        sys.exit(3)
    return res


CALIB_CODE = ("ncc.py", "desc_scan.py", "recut.py", "calib.py")


def calib_plan(cfg: dict, out: Path) -> dict:
    """보정이 고를 영상과 실행 식별 — 생산(step_calib)과 소비 쪽 대조(require_calib_current)가 같은 것을 쓴다."""
    c = cfg["calib"]
    man = read_manifest(cfg)
    with (ROOT / cfg["inputs"]["tiles"]).open(encoding="utf-8", newline="") as fh:
        prov = {r_["image_id"]: r_["provenance"] for r_ in csv.DictReader(fh)}
    seed = int(cfg["pilot"]["seed"])
    meta = json.loads((out / "metacheck.json").read_text(encoding="utf-8"))
    code = SC.code_version(CALIB_CODE)
    by = defaultdict(list)
    for i, r_ in man.items():
        if r_["split"] != "eval":
            by[prov[i]].append(i)
    g3_pick = []
    for pv in sorted(by):
        g3_pick += [(pv, i) for i in sorted(by[pv], key=lambda i: (rank_key(seed, "g3|" + i), i))[: int(c["g3_per_provenance"])]]
    q, fill, ref_cdf = _chain_params(cfg)
    ov_pick = _pick_panoramas(cfg, int(c["overlap_panoramas"]), tv_only=True, salt="overlap|")
    return {
        "g3_pick": g3_pick, "ov_pick": ov_pick, "chain": (q, fill, ref_cdf), "man": man,
        "g3_ident": {"pick_sha256": _sha_rows([list(x) for x in g3_pick]), "n_pick": len(g3_pick),
                     "transforms": list(C.TRANSFORMS), "code": code, "input_sha256": meta["input_sha256"]},
        "ov_ident": {"pick_sha256": _sha_rows(ov_pick), "n_pick": len(ov_pick), "overlaps": list(c["overlaps"]),
                     "p_min_overlap": float(cfg["pilot"]["p_min_overlap"]), "chain": {"quality": q, "fill": fill},
                     "code": code, "input_sha256": meta["input_sha256"]},
    }


def calib_sources(out: Path) -> dict[str, str | None]:
    """보정 요약의 원천 CSV 두 개의 sha256(없으면 None)."""
    return {n: (sha256_file(out / n) if (out / n).exists() else None) for n in ("calib_g3.csv", "calib_overlap.csv")}


def require_calib_current(cfg: dict, out: Path) -> dict[str, str]:
    """보정 산출을 소비하기 전에(공유 회신 §63-1 재검토) — 두 실행 식별이 지금 설정 · 입력에서 기대하는 것과 같아야 한다."""
    plan = calib_plan(cfg, out)
    return {"g3": require_current(out / "calib_g3_RUN.json", plan["g3_ident"], "보정(G3 합성 양성)"),
            "overlap": require_current(out / "calib_overlap_RUN.json", plan["ov_ident"], "보정(겹침 곡선)")}


def step_calib(cfg: dict, out: Path) -> dict:
    """(가) G3 합성 양성(출처마다, 학습검증) · (다) 겹침 곡선(학습검증 파노라마). 분포와 곡선만 — 고르지 않는다."""
    require_start(cfg, "calib", out)
    c = cfg["calib"]
    prog = read_progress(cfg)
    g = make_guard(cfg, out)
    plan = calib_plan(cfg, out)
    man, pick = plan["man"], plan["g3_pick"]
    # (가)
    g3p = out / "calib_g3.csv"
    bind_outputs(out / "calib_g3_RUN.json", plan["g3_ident"], [g3p])
    if not g3p.exists():
        tmp = g3p.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["provenance", "image_id", *C.TRANSFORMS])
            for k, (pv, i) in enumerate(pick):
                if k % 50 == 0:
                    g.wait()
                v = C.g3_positive((ROOT / man[i]["rel_path"]).read_bytes())
                w.writerow([pv, i, *[round(v[t], 5) for t in C.TRANSFORMS]])
        os.replace(tmp, g3p)
    with g3p.open(encoding="utf-8", newline="") as fh:
        g3rows = list(csv.DictReader(fh))
    g3sum = {}
    for pv in sorted({r_["provenance"] for r_ in g3rows}):
        rr = [r_ for r_ in g3rows if r_["provenance"] == pv]
        small = np.array([float(r_[t]) for r_ in rr for t in C.TRANSFORMS if t not in C.LARGE])
        g3sum[pv] = {"n_images": len(rr), "p5_small": round(float(np.percentile(small, 5)), 5),
                     "p1_small": round(float(np.percentile(small, 1)), 5),
                     "per_transform_p5": {t: round(float(np.percentile([float(r_[t]) for r_ in rr], 5)), 5) for t in C.TRANSFORMS}}
    # (다)
    ovp = out / "calib_overlap.csv"
    q, fill, ref_cdf = plan["chain"]
    pick = plan["ov_pick"]
    bind_outputs(out / "calib_overlap_RUN.json", plan["ov_ident"], [ovp])
    if not ovp.exists():
        names = _normal_label_names(cfg)
        members = _raw_members(cfg)
        tmp = ovp.with_suffix(".csv.tmp")
        head = ["image_id", "width", "overlap", "shift_px", "g3", "g4", "g4_dx", "g4_dy", "p", "p_dx", "p_dy", "p_shift_ok"]
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(head)
            for k, iid in enumerate(pick):
                if k % 10 == 0:
                    g.wait()
                zp, mem = members[names[iid]]
                with zipfile.ZipFile(zp) as z:
                    raw = z.read(mem)
                with Image.open(BytesIO(raw)) as im:
                    im_l = im.convert("L")
                for r_ in C.overlap_curve_one(im_l, prog[iid]["box"][1], list(c["overlaps"]), quality=q, fill=fill,
                                              ref_cdf=ref_cdf, p_min_overlap=float(cfg["pilot"]["p_min_overlap"])):
                    w.writerow([iid, im_l.width, *[r_[h] for h in head[2:]]])
        os.replace(tmp, ovp)
    with ovp.open(encoding="utf-8", newline="") as fh:
        ovrows = list(csv.DictReader(fh))
    curve = {}
    for ov in sorted({r_["overlap"] for r_ in ovrows}, key=float, reverse=True):
        rr = [r_ for r_ in ovrows if r_["overlap"] == ov]
        g4 = np.array([float(r_["g4"]) for r_ in rr])
        pv = np.array([float(r_["p"]) if r_["p"] not in ("", "None") else np.nan for r_ in rr])
        ok = np.array([r_["p_shift_ok"] == "True" for r_ in rr])
        curve[ov] = {"n": len(rr), "g4>=0.45": round(float((g4 >= 0.45).mean()), 4),
                     "p_shift_ok": round(float(ok.mean()), 4),
                     **{f"p>={t}&shift_ok": round(float(((np.nan_to_num(pv, nan=-9) >= t) & ok).mean()), 4)
                        for t in (0.6, 0.7, 0.8, 0.9)},
                     "g3_median": round(float(np.median([float(r_["g3"]) for r_ in rr])), 4)}
    res = {"g3_synthetic_positive": g3sum, "overlap_curve": curve, "runs": require_calib_current(cfg, out),
           "sources": calib_sources(out),
           "note": "제안값을 고르지 않는다 — 문턱은 등록(3.5)에서. 학습검증 영상 · 파노라마만 썼다"}
    write_json(out / "calib_summary.json", res)
    log("보정: G3 p5(작은 변환) " + ", ".join(f"{k} {v['p5_small']}" for k, v in g3sum.items()))
    return res


STEPS = {"sample": step_sample, "metacheck": step_metacheck, "scan": step_scan, "calib": step_calib,
         "hops": step_hops, "attach": step_attach, "recut": step_recut}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=[*STEPS, "all"])
    args = ap.parse_args(argv)
    cfg = load_cfg()
    try:
        out = check_out_path(ROOT / cfg["pilot"]["out"])
    except OutPathForbidden as exc:
        log(f"!! {exc}")
        return 5
    out.mkdir(parents=True, exist_ok=True)
    steps = list(STEPS) if args.step == "all" else [args.step]
    for s in steps:
        t0 = time.perf_counter()
        log(f"== {s}")
        try:
            STEPS[s](cfg, out)
        except SC.StartRefused as exc:
            log(f"!! {exc}")
            return 2
        except MetaCheckFailed as exc:
            log(f"!! {exc}")
            return 3
        except SC.RunMismatch as exc:
            log(f"!! {exc}")
            return 4
        log(f"   {s} {time.perf_counter() - t0:.1f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
