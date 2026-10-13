"""평가 층화 표본 — 동결 평가셋에서 묶음(group_id) 단위로 층 비례 표본을 뽑는다. 명세: D 25번 §2(wt/D 051f43a).

입력은 매니페스트의 **다섯 열**(`image_id` · `split` · `group_id` · `material` · `n_defects`)뿐이다(§2-1) —
상자 · 좌표 · 주석 파일 · 영상 · 평가 폴더는 읽지 않는다(불변조건 1-4). 결과는 평가셋(`split == "eval"`) 안의 id 만 담는다.

규칙(정수 산술 — 실수 반올림이 끼지 않는다. 같은 입력이면 입력 행의 순서와 상관없이 같은 목록):
- §2-2 층 = 묶음의 값. `Σ` = 묶음의 결함 수 합, `n` = 묶음의 평가 이미지 수.
  구간 `0`(Σ = 0) · `1`(2Σ < 3n) · `2-3`(2Σ < 7n) · `4+`(그 밖). 층 = `재질|구간`. 묶음 안 재질이 둘이면 거부한다.
- §2-3 배분(묶음 수) `n_h = min(G_h, max(2, ⌊(2·n·G_h + N) / (2·N)⌋))`, 목표 `n ≥ N` 이면 `n_h = G_h`.
- §2-4 순서 = `sha256(UTF-8(SEED + "|" + group_id))` 의 16진 소문자 오름차순, 같으면 `group_id` 오름차순. 앞에서 `n_h` 개.
- §2-5 목록 = 뽑힌 묶음의 평가 이미지 전부, `image_id` 코드포인트 오름차순, `evaluation.eval_list` 정규형.
  곁 파일 = CSV `image_id,stratum,G_h,n_h`(머리줄 + 같은 순서, LF).
- §2-6 같은 시드에서 크기가 커지면 앞의 목록을 품는다(같은 순열의 접두).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

#: 쓰는 매니페스트 열(§2-1) — 이 밖은 읽지 않는다
COLUMNS = ("image_id", "split", "group_id", "material", "n_defects")
SPLIT_EVAL = "eval"
#: 공개 상수 시드(§2-4) — 학습 시드와 무관
SEED = "eval_sample_v1|aihub71761_rt_v1|1f80e98b"
#: 봉인하는 세 크기(§2-8)
SIZES = (2000, 3000, 4000)
#: 명세 문서(D 25번) — 이 구현이 따르는 판
SPEC = {"path": "docs/dev_log/2026-10-07-본실험코퍼스/25_평가표본_대표지표_D.md", "commit": "051f43a",
        "sha256": "f730d886fa9d948ecbb59f9b826c1faed44907d7577cbc515e6ec43ac31cb4ba", "section": "§2"}
SIDE_HEADER = ("image_id", "stratum", "G_h", "n_h")


class SampleRefused(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def group_band(total_defects: int, n_images: int) -> str:
    """묶음 평균 결함 수 구간(§2-2) — 정수 비교만."""
    if n_images <= 0 or total_defects < 0:
        raise SampleRefused("band", "묶음의 이미지 수는 양수, 결함 수 합은 0 이상")
    if total_defects == 0:
        return "0"
    if 2 * total_defects < 3 * n_images:
        return "1"
    if 2 * total_defects < 7 * n_images:
        return "2-3"
    return "4+"


def allocate(groups_in_stratum: int, size: int, population: int) -> int:
    """층의 묶음 수(§2-3) — 정수 산술, 0.5 올림, 최소 2, 층의 묶음 수를 넘지 않는다."""
    if size >= population:
        return groups_in_stratum
    return min(groups_in_stratum, max(2, (2 * size * groups_in_stratum + population) // (2 * population)))


def rank_key(seed: str, group_id: str) -> tuple[str, str]:
    """§2-4 의 순서 열쇠 — (sha256 16진 소문자, group_id)."""
    return hashlib.sha256((seed + "|" + group_id).encode("utf-8")).hexdigest(), group_id


def read_eval_groups(rows: Iterable[Mapping[str, str]]) -> dict[str, dict]:
    """평가셋 행을 묶음으로 모은다 — {group_id: {material, ids, total_defects}}. 꼴이 어긋나면 거부한다."""
    groups: dict[str, dict] = defaultdict(lambda: {"material": None, "ids": [], "total_defects": 0})
    seen: set[str] = set()
    for r in rows:
        missing = [c for c in COLUMNS if c not in r]
        if missing:
            raise SampleRefused("columns", f"매니페스트 열이 없다: {missing}")
        if r["split"] != SPLIT_EVAL:
            continue
        iid, gid, material = r["image_id"], r["group_id"], r["material"]
        if iid in seen:
            raise SampleRefused("duplicate_id", "평가셋 image_id 가 겹친다")
        if not iid or not gid or not material:
            raise SampleRefused("empty_field", "image_id · group_id · material 은 비지 않아야 한다")
        try:
            n = int(r["n_defects"])
        except (TypeError, ValueError) as exc:
            raise SampleRefused("n_defects", "n_defects 가 정수가 아니다") from exc
        if n < 0 or str(n) != r["n_defects"].strip():
            raise SampleRefused("n_defects", "n_defects 는 0 이상의 정수 글자")
        g = groups[gid]
        if g["material"] not in (None, material):
            raise SampleRefused("mixed_material", "한 묶음에 재질이 둘이다")
        g["material"] = material
        g["ids"].append(iid)
        g["total_defects"] += n
        seen.add(iid)
    if not groups:
        raise SampleRefused("empty", "평가셋 행이 없다")
    return dict(groups)


def sample_eval(rows: Iterable[Mapping[str, str]], sizes: Iterable[int] = SIZES, seed: str = SEED) -> dict:
    """크기마다 표본을 뽑는다. 결과: {population, strata, samples: {크기: {ids, side_rows, ...}}}."""
    sizes = list(sizes)
    if not sizes or any(type(s) is not int or s <= 0 for s in sizes):
        raise SampleRefused("size", "크기는 양의 정수")
    groups = read_eval_groups(rows)
    population = sum(len(g["ids"]) for g in groups.values())
    by_stratum: dict[str, list[str]] = defaultdict(list)
    stratum_of: dict[str, str] = {}
    for gid, g in groups.items():
        s = f"{g['material']}|{group_band(g['total_defects'], len(g['ids']))}"
        stratum_of[gid] = s
        by_stratum[s].append(gid)
    order = {s: sorted(gids, key=lambda gid: rank_key(seed, gid)) for s, gids in by_stratum.items()}
    strata = {s: {"G_h": len(gids), "images": sum(len(groups[g]["ids"]) for g in gids)} for s, gids in sorted(by_stratum.items())}
    samples = {}
    for size in sizes:
        n_h = {s: allocate(len(order[s]), size, population) for s in order}
        chosen = {gid: s for s in order for gid in order[s][:n_h[s]]}
        ids = sorted(iid for gid in chosen for iid in groups[gid]["ids"])
        side = [(iid, stratum_of[gid], len(order[stratum_of[gid]]), n_h[stratum_of[gid]])
                for gid in chosen for iid in groups[gid]["ids"]]
        side.sort(key=lambda r: r[0])
        per = {s: {"n_h": n_h[s], "sampled_images": sum(len(groups[g]["ids"]) for g in order[s][:n_h[s]])} for s in sorted(order)}
        samples[size] = {"ids": ids, "side_rows": side, "groups": len(chosen), "images": len(ids), "strata": per}
    return {"seed": seed, "population": {"eval_images": population, "eval_groups": len(groups)}, "strata": strata,
            "samples": samples}


def list_bytes(ids: list[str]) -> bytes:
    """목록 파일 바이트 — 평가 목록 정규형(`evaluation.eval_list.canonical_bytes`, 한 정의)."""
    from evaluation.eval_list import canonical_bytes

    return canonical_bytes(ids)


def side_bytes(side_rows: list[tuple]) -> bytes:
    lines = [",".join(SIDE_HEADER)] + [",".join(str(x) for x in r) for r in side_rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_manifest_columns(manifest_csv: Path) -> list[dict[str, str]]:
    """매니페스트에서 `COLUMNS` 만 꺼낸다(다른 열은 버린다)."""
    with manifest_csv.open(encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (rd.fieldnames or [])]
        if missing:
            raise SampleRefused("columns", f"매니페스트 열이 없다: {missing}")
        return [{c: r[c] for c in COLUMNS} for r in rd]


def write_samples(result: dict, out: Path, snapshot_digest: str) -> dict:
    """새 폴더에 크기마다 목록 · 곁 파일과 SAMPLE.json 을 쓴다(폴더가 있으면 거부 — 덮어쓰지 않는다)."""
    from evaluation.eval_list import set_digest, validate_eval_list

    out.mkdir(parents=True, exist_ok=False)
    meta = {"spec": dict(SPEC), "seed": result["seed"], "manifest_snapshot_digest": snapshot_digest,
            "columns": list(COLUMNS), "population": result["population"], "strata": result["strata"], "sizes": {}}
    for size, smp in result["samples"].items():
        raw = list_bytes(smp["ids"])
        checked = validate_eval_list(raw, where=f"eval_sample_{size}")          # 정규형 검사를 지나야 쓴다
        assert checked.set_sha256 == set_digest(smp["ids"])
        lp, sp = out / f"eval_sample_{size}.txt", out / f"eval_sample_{size}_strata.csv"
        lp.write_bytes(raw)
        sp.write_bytes(side_bytes(smp["side_rows"]))
        meta["sizes"][str(size)] = {"images": smp["images"], "groups": smp["groups"], "strata": smp["strata"],
                                    "list_file": lp.name, "file_sha256": checked.file_sha256, "set_sha256": checked.set_sha256,
                                    "side_file": sp.name, "side_sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    (out / "SAMPLE.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m data.eval_sample", description=__doc__.splitlines()[0])
    ap.add_argument("--manifest-dir", type=Path, required=True, help="봉인 매니페스트 폴더(SNAPSHOT.sha256 · manifest.csv)")
    ap.add_argument("--expect-digest", required=True, help="스냅샷 다이제스트(configs 의 fixed_before_main_runs.snapshot_digest)")
    ap.add_argument("--out", type=Path, required=True, help="새 폴더(있으면 거부) — 목록은 추적 밖에 둔다")
    ap.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    a = ap.parse_args(argv)
    from data.manifest_io import verify_snapshot

    digest = verify_snapshot(a.manifest_dir)
    if digest != a.expect_digest:
        raise SampleRefused("snapshot", f"매니페스트 다이제스트 {digest} 가 기대 {a.expect_digest} 와 다르다")
    result = sample_eval(read_manifest_columns(a.manifest_dir / "manifest.csv"), a.sizes)
    meta = write_samples(result, a.out, digest)
    print(json.dumps({k: {x: v[x] for x in ("images", "groups", "file_sha256", "set_sha256", "side_sha256")}
                      for k, v in meta["sizes"].items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
