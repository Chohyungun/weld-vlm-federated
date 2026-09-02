"""동결 디렉터리 위생 감사 — 정본과 경쟁 파일을 대조한다. 80번 G11-1·E16·E21.

`data/interim/manifest_v1/` 안에 **동결본과 다른 분할을 담은 완전한 매니페스트**가
보호 없이 있었다. 파일명 하나를 잘못 고르면 평가셋 소속이 통째로 뒤집힌다.
격리(`attic/`) 후에도 그 사실이 잊히지 않도록, 무엇이 얼마나 다른지를 **매번 다시 재서**
`attic/README.md` 의 수치를 뒷받침한다.

    uv run python scripts/audit_frozen_dir.py

하는 일 셋.

1. SNAPSHOT 계약 4파일의 sha256 재검증 — 정본이 그대로인지.
2. 계약 밖 파일 목록 — 격리 대상(경쟁 매니페스트·분할 메타)이 본 디렉터리에 남아 있는지.
3. 경쟁 매니페스트가 정본과 **어디서 몇 행 다른지** 실측.

실패하면 종료 코드 1. 정본이 흔들리면 그 자체가 사고다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"
ATTIC = V1 / "attic"

#: 격리 대상. 동결본과 **다른 분할**을 담은 완전한 매니페스트와 그 분할 메타다.
QUARANTINE = (
    "manifest_e3.csv",
    "manifest_pre_e3.csv",
    "manifest_pre_histmatch.csv",
    "manifest_pre_mask.csv",
    "split_meta.json",
    "split_meta_e3.json",
)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_contract(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def compare_manifest(cand: Path, ref: pd.DataFrame) -> dict:
    """경쟁 매니페스트가 정본과 어디서 몇 행 다른지."""
    c = pd.read_csv(cand, dtype=str, keep_default_na=False)
    common = sorted(set(c["image_id"]) & set(ref["image_id"]))
    a = ref.set_index("image_id").loc[common]
    b = c.set_index("image_id").loc[common]
    out = {
        "rows": len(c),
        "rows_frozen": len(ref),
        "only_in_candidate": len(set(c["image_id"]) - set(ref["image_id"])),
        "only_in_frozen": len(set(ref["image_id"]) - set(c["image_id"])),
        "common": len(common),
    }
    for col in ("split", "client"):
        if col in b.columns:
            out[f"{col}_differs"] = int((a[col].astype(str) != b[col].astype(str)).sum())
    if "split" in c.columns:
        out["eval_n"] = int((c["split"] == "eval").sum())
    if "client" in c.columns:
        tr = c[c["split"] != "eval"] if "split" in c.columns else c
        n_c1 = int((tr["client"] == "C1").sum())
        n_cl = int(tr["client"].isin(["C1", "C2"]).sum())
        out["c1_share_of_steel"] = round(n_c1 / n_cl, 4) if n_cl else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=V1 / "frozen_dir_audit.json")
    args = ap.parse_args()

    contract = read_contract(V1 / "SNAPSHOT.sha256")
    print(f"동결 계약 {len(contract)}파일 · {V1}")

    # --- 1. 정본 재검증 ---
    ok = True
    digests = {}
    for name, expect in contract.items():
        got = sha256(V1 / name)
        digests[name] = got
        same = got == expect
        ok &= same
        print(f"  [{'일치' if same else '불일치'}] {name:26s} {got[:16]}…")
    if not ok:
        print("  !! 동결 정본이 바뀌었다. 여기서 멈춘다 — 다른 무엇보다 먼저 볼 일이다")
        return 1

    # --- 2. 계약 밖 파일 ---
    present = {p.name for p in V1.iterdir() if p.is_file()}
    outside = sorted(present - set(contract) - {"SNAPSHOT.sha256"})
    still_loose = [n for n in QUARANTINE if n in present]
    print(f"\n계약 밖 파일 {len(outside)}개 (본 디렉터리)")
    if still_loose:
        print(f"  !! 격리 대상이 아직 본 디렉터리에 있다: {still_loose}")
    else:
        print("  격리 대상 6종 없음 — attic/ 으로 옮겨졌다")

    attic_present = sorted(p.name for p in ATTIC.iterdir() if p.is_file()) if ATTIC.is_dir() else []
    print(f"attic/ {len(attic_present)}개: {attic_present}")

    # --- 3. 경쟁 매니페스트 대조 ---
    ref = pd.read_csv(V1 / "manifest.csv", dtype=str, keep_default_na=False)
    cmp_out = {}
    print("\n경쟁 매니페스트 대조 (정본 대비)")
    for name in QUARANTINE:
        if not name.endswith(".csv"):
            continue
        p = ATTIC / name if (ATTIC / name).exists() else V1 / name
        if not p.exists():
            print(f"  {name:28s} 없음")
            continue
        d = compare_manifest(p, ref)
        cmp_out[name] = d
        print(f"  {name:28s} {d['rows']:,}행 · split 차이 {d.get('split_differs', 0):,} · "
              f"client 차이 {d.get('client_differs', 0):,} · eval {d.get('eval_n', 0):,} · "
              f"C1 비중 {d.get('c1_share_of_steel')}")

    # --- 분할 메타 대조 ---
    caps = yaml.safe_load((V1 / "data_capabilities.yaml").read_text(encoding="utf-8"))
    frozen_split = caps.get("split_meta", {})
    meta_out: dict[str, object] = {"frozen_data_capabilities": frozen_split}
    print(f"\n동결 분할 메타 (정본) {frozen_split}")
    # 갈리는 값은 **`dirichlet` 안**에 있다. 최상위 키만 비교하면 폐기본까지 "일치"로
    # 나온다 — 그러면 이 감사가 잡으라는 것을 못 잡는다(80번 E21).
    fz_dir = frozen_split.get("dirichlet", {})
    for name in ("split_meta.json", "split_meta_e3.json"):
        p = ATTIC / name if (ATTIC / name).exists() else V1 / name
        if not p.exists():
            continue
        j = json.loads(p.read_text(encoding="utf-8"))
        d = j.get("dirichlet", {})
        keep = {k: d.get(k) for k in ("seed_used", "attempts", "c1_share") if k in d}
        agree = all(fz_dir.get(k) == v for k, v in keep.items())
        meta_out[name] = {"dirichlet": keep, "agrees_with_frozen": bool(agree)}
        role = "정본 분할의 유도 원본" if agree else "**폐기값 — 다른 분할**"
        print(f"  {name:22s} dirichlet {keep}  → {role}")

    report = {
        "snapshot_dir": str(V1.relative_to(REPO_ROOT).as_posix()),
        "contract_files": list(contract),
        "contract_verified": bool(ok),
        "digests": digests,
        "quarantined": attic_present,
        "still_loose_in_frozen_dir": still_loose,
        "outside_contract_in_frozen_dir": outside,
        "competing_manifests": cmp_out,
        "split_meta": meta_out,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\n기록: {args.out}")
    return 1 if still_loose else 0


if __name__ == "__main__":
    raise SystemExit(main())
