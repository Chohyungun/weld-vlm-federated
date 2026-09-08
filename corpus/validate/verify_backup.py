"""봉인 자산 드라이브 백업 — 목록 산출과 적재 후 대조. 09-08 총괄 판정 (24번 A-4 ②).

`data/processed/pairs_pilot_v1|v2` 는 AI허브 71761 파생 좌표를 담아 **git 으로 보낼 수
없다** — `origin` 이 공개 GitHub 이고 해외 호스팅이라 레드라인 1(국외 반출 금지) 위반이다.
그런데 `data/processed` 는 정션이라 물리 사본이 `E:\\Fedvlm_for_welding\\data\\processed`
한 곳뿐이고, 그 디스크가 죽으면 논문에 실을 해시의 실물이 사라진다. 그래서 **국내 공유
드라이브 2사본**으로 간다.

**드라이브 접근은 사람이 한다.** 이 스크립트는 복사하지 않는다 — 무엇을 옮길지 목록을
내고, 옮긴 뒤 제대로 갔는지 대조만 한다. 자동 업로드를 넣지 않은 이유는 하나다. 목적지가
국내인지 판단하는 것은 사람이고, 스크립트가 대신 정하면 레드라인 판단이 코드 안으로
숨는다.

## 쓰는 순서

    # 1) 무엇을 옮기는지 본다. 매니페스트가 나온다 (봉인 디렉터리에는 쓰지 않는다).
    uv run python -m corpus.validate.verify_backup plan --out backup_manifest.json

    # 2) 사람이 복사한다. 국내 공유 드라이브로.
    #    디렉터리 이름을 그대로 유지해야 대조가 된다 (pairs_pilot_v1/ …).

    # 3) 적재본을 대조한다. 원본을 읽고 사본을 읽어 sha256 을 직접 비교한다.
    uv run python -m corpus.validate.verify_backup verify --dest "<드라이브 경로>"

대조는 **계약서(SNAPSHOT.sha256)와 사본** 둘 다에 건다. 원본↔사본만 보면 원본이 이미
틀어져 있어도 같이 틀어진 채 통과한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from corpus.generate.frozen_out import CONTRACT_NAME, snapshot_summary, tracked_names

REPO = Path(__file__).resolve().parents[2]

#: 총괄이 드라이브 보관으로 판정한 봉인 자산 전부 (09-08, 의사결정로그 abdafe8).
#: pairs 는 AI허브 파생이라 애초에 git 이 불가였고, cycle_pilot* 는 공개 검수에서
#: 재배포 478건이 걸려 추적 전환이 기각됐다. 그래서 **한 벌로 묶어** 같이 옮긴다.
DEFAULT_DIRS = (
    REPO / "corpus/generate/cycle_pilot",
    REPO / "corpus/generate/cycle_pilot_v2",
    REPO / "data/processed/pairs_pilot_v1",
    REPO / "data/processed/pairs_pilot_v2",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_plan(dirs) -> dict:
    """옮길 파일 목록. 계약 구성원 **전부** + 계약서 자체를 담는다.

    계약서를 빼면 사본만으로는 무엇이 있어야 하는지 알 수 없다 — 사본이 자립해야
    원본 디스크가 죽어도 복구가 된다.

    위험한 것은 추적 밖 구성원뿐이고 추적분은 git·origin 에 이미 있다. 그런데도
    **추적분까지 담는다.** 빼면 사본이 자기 계약서로 스스로를 검증하지 못해, 복구 때마다
    git 에서 나머지를 맞춰 와야 한다. 그 대가가 9 KB 다 — 자립을 산다. 항목마다
    `tracked` 를 달아 어느 것이 실제 위험분인지는 목록에 남긴다.
    """
    items, problems = [], []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            problems.append(f"{d}: 디렉터리가 없다")
            continue
        names, _digest = snapshot_summary(d)
        if not names:
            problems.append(f"{d}: {CONTRACT_NAME} 가 없다 — 봉인본이 아니다")
            continue
        tracked = tracked_names(d) or frozenset()
        for name in [*names, CONTRACT_NAME]:
            p = d / name
            if not p.is_file():
                problems.append(f"{d.name}/{name}: 계약에 있는데 실물이 없다")
                continue
            items.append({
                "dir": d.name,
                "name": name,
                "src": str(p),
                "rel": f"{d.name}/{name}",
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                # git 이 들고 있으면 이 사본이 유일본은 아니다. 실제 위험분은 False 쪽.
                "tracked": name in tracked,
                # 계약서 자체는 자기 해시를 담을 수 없다 (파생물이다).
                "contract_sha256": next(
                    (h for h, n in _contract_entries(d) if n == name), None),
            })
    total = sum(i["bytes"] for i in items)
    at_risk = [i for i in items if not i["tracked"]]
    return {
        "purpose": "봉인 자산 국내 공유 드라이브 백업 (09-08 판정, 의사결정로그 abdafe8)",
        "redline": ("pairs_pilot_* 는 AI허브 71761 파생물이고 cycle_pilot* 는 규정 원문과"
                    " 겹치는 생성문이다 — 국외 리전·해외 호스팅 스토리지 금지"
                    " (규약 2-1·2-5). 목적지가 국내인지는 사람이 확인한다."),
        "n_at_risk": len(at_risk),
        "at_risk_bytes": sum(i["bytes"] for i in at_risk),
        "n_files": len(items),
        "total_bytes": total,
        "total_mb": round(total / 1024 / 1024, 2),
        "dirs": [Path(d).name for d in dirs],
        "items": items,
        "problems": problems,
    }


def _contract_entries(d: Path) -> list[tuple[str, str]]:
    """계약서에 적힌 `(sha256, 이름)`. 계약서 자신은 목록에 없다."""
    out = []
    try:
        text = (d / CONTRACT_NAME).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return out
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#"):
            h, _, n = ln.partition(" ")
            out.append((h.strip(), n.strip()))
    return out


def verify(plan: dict, dest: Path) -> dict:
    """사본 대조. **원본↔사본**과 **계약서↔사본** 을 둘 다 본다."""
    dest = Path(dest)
    rows, n_ok = [], 0
    for it in plan["items"]:
        cp = dest / it["rel"]
        row = {"rel": it["rel"], "copy": str(cp)}
        if not cp.is_file():
            row["verdict"] = "missing"
            rows.append(row)
            continue
        got = sha256_file(cp)
        row["sha256"] = got
        row["matches_source"] = got == it["sha256"]
        # 계약서에 기록이 있는 구성원은 계약서와도 맞아야 한다.
        row["matches_contract"] = (None if it["contract_sha256"] is None
                                   else got == it["contract_sha256"])
        if not row["matches_source"]:
            row["verdict"] = "mismatch"
        elif row["matches_contract"] is False:
            row["verdict"] = "contract_mismatch"
        else:
            row["verdict"] = "ok"
            n_ok += 1
        rows.append(row)
    bad = [r for r in rows if r["verdict"] != "ok"]
    return {"dest": str(dest), "n_files": len(rows), "n_ok": n_ok,
            "n_bad": len(bad), "rows": rows,
            "verdict": "ok" if not bad else "failed"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="봉인 자산 드라이브 백업 목록·대조")
    ap.add_argument("cmd", choices=["plan", "verify"])
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="대상 봉인 디렉터리. 미지정 시 pairs_pilot_v1·v2")
    ap.add_argument("--dest", default=None, help="verify: 적재한 드라이브 경로")
    ap.add_argument("--out", default=None, help="plan: 매니페스트를 쓸 파일")
    ap.add_argument("--manifest", default=None,
                    help="verify: plan 이 낸 매니페스트. 미지정 시 즉석에서 다시 만든다")
    args = ap.parse_args(argv)

    dirs = [Path(x) for x in args.dirs] if args.dirs else list(DEFAULT_DIRS)

    if args.cmd == "plan":
        plan = build_plan(dirs)
        for it in plan["items"]:
            print(f"  {'ㆍ' if it['tracked'] else '★'} {it['rel']:42s}"
                  f" {it['bytes']:>10,} B  {it['sha256'][:16]}…")
        print(f"\n파일 {plan['n_files']}개 / {plan['total_mb']} MB")
        print(f"  ★ git 밖 — 이 사본이 유일본이 된다: {plan['n_at_risk']}개 /"
              f" {plan['at_risk_bytes']:,} B")
        print("  ㆍgit·origin 에도 있음. 사본이 자기 계약서로 자립하도록 함께 담는다")
        print(f"!! {plan['redline']}")
        if plan["problems"]:
            print("\n문제:", *plan["problems"], sep="\n  ", file=sys.stderr)
        if args.out:
            p = Path(args.out)
            with p.open("w", encoding="utf-8", newline="") as fh:
                fh.write(json.dumps(plan, ensure_ascii=False, indent=1) + "\n")
            print(f"매니페스트: {p}")
        return 1 if plan["problems"] else 0

    if not args.dest:
        print("verify 는 --dest 가 필요하다 (적재한 드라이브 경로)", file=sys.stderr)
        return 2
    plan = (json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if args.manifest else build_plan(dirs))
    r = verify(plan, Path(args.dest))
    for row in r["rows"]:
        mark = {"ok": "○", "missing": "✕", "mismatch": "✕",
                "contract_mismatch": "✕"}[row["verdict"]]
        print(f"{mark} {row['rel']:44s} {row['verdict']}")
    print(f"\n{r['n_ok']}/{r['n_files']} 일치 — 판정 {r['verdict']}")
    if r["verdict"] != "ok":
        print("사본이 원본·계약과 다르다. 다시 복사하라.", file=sys.stderr)
    return 0 if r["verdict"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
