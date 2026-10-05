"""봉인 자산 드라이브 백업 — 목록 산출과 적재 후 대조. 09-08 판정 (24번 A-4 ②).

대상은 봉인처 명부(`data.frozen_guard.EXPECTED_SEALED`)에서 실물이 있어야 하는 자리
(`expected`·`restored`)를 파생한다 — 지금은 `data/interim/manifest_v1`(본실험 매니페스트 계약)·
`data/processed/pairs_pilot_v1`(둘 다 AI허브 71761 파생 → 레드라인 1, git 금지)과
`cycle_pilot`·`cycle_pilot_v2` 의 미추적 구성원(유료 표준 스크린에 걸려 추적 전환 보류 —
24번 부록 A-4·09-08 판정)이다. `pairs_pilot_v2` 는 09-11 소실·사본 없음이라 명부가
`lost` 로 들고 목록에서는 뺀다(30번 부록 A-3). `origin` 이 공개 GitHub 이고 해외 호스팅이라
어느 쪽도 **git 으로 보낼 수 없고**, `data/interim`·`data/processed` 는 정션이라 물리 사본이
`<본체 저장소>` 한 곳뿐이다 — 그 디스크가 죽으면 논문에 실을 해시의 실물이
사라진다. 그래서 **국내 공유 드라이브 2사본**으로 간다.

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

from corpus.generate.frozen_out import (
    CONTRACT_NAME,
    PRESENT_STATUSES,
    snapshot_summary,
    tracked_names,
)
from data.frozen_guard import EXPECTED_SEALED

REPO = Path(__file__).resolve().parents[2]

#: 드라이브 보관으로 판정된 봉인 자산 전부 (09-08, 의사결정로그 abdafe8).
#: pairs 는 AI허브 파생이라 애초에 git 이 불가였고, cycle_pilot* 는 공개 검수에서
#: 재배포 478건이 걸려 추적 전환이 기각됐다. 그래서 **한 벌로 묶어** 같이 옮긴다.
#: 09-16 부터 본실험 매니페스트 계약 `data/interim/manifest_v1`(A)도 명부에 들어와 함께
#: 옮긴다 — 계약 구성원 4 + 계약서로 약 47 MB 가 늘었고 그대로 수용됐다.
#:
#: 목록을 여기 다시 적지 않는다 — 봉인처 명부(`data.frozen_guard.EXPECTED_SEALED`)에서
#: **있어야 하는 것**만 파생한다. 소실이 기록된 곳은 옮길 실물이 없으니 빠지고, 그 사실은 명부
#: 대조기(`python -m corpus.generate.frozen_out`)가 매번 알린다. 두 목록이 따로 살면
#: 한쪽만 고쳐진다 — 09-13 에 그 차이로 이 목록이 소실된 v2 를 계속 들고 있었다.

#: 실물이 있어야 하는 상태 — 대조기와 **같은 집합**을 쓴다. `restored`(소실 뒤 동일 바이트
#: 복원)는 검사도 백업도 `expected` 와 같다. 복원된 봉인본은 git 밖 단일 사본으로 돌아온
#: 것이라 오히려 백업이 급하다. 여기서 빠뜨리면 목록이 조용히 줄어든다(41번 I-1).
BACKED_UP_STATUSES = PRESENT_STATUSES


def sealed_dirs(registry: dict[str, dict[str, str]] | None = None,
                root: Path | None = None) -> tuple[Path, ...]:
    """명부에서 백업 대상 봉인처를 고른다. 명부·뿌리를 **부를 때** 읽어 시험이 갈아 끼울 수 있다."""
    reg = EXPECTED_SEALED if registry is None else registry
    base = REPO if root is None else Path(root)
    return tuple(base / rel for rel, e in reg.items()
                 if e.get("status") in BACKED_UP_STATUSES)


SEALED_DIRS = sealed_dirs()

#: 봉인본은 아니지만 **git 밖 단일 사본**이 생기는 곳 (09-15 과제 4).
#: 사람 gold 라벨 `labels_*.jsonl` 과 항목별 층 `*.strata.json` 은 `.gitignore` 로 추적하지
#: 않는다(라벨러 이름·항목별 후보 판정). 백업 목록에 없으면 09-11 과 같은 단일 사본 구조다.
UNSEALED_DIRS = (REPO / "corpus/validate/judge_labels",)

DEFAULT_DIRS = SEALED_DIRS + UNSEALED_DIRS


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
        sealed = bool(names)
        if sealed:
            wanted = [*names, CONTRACT_NAME]
        elif d in UNSEALED_DIRS:
            # 봉인본이 아닌 백업 대상 — 계약이 없으니 **있는 파일 전부**가 목록이다.
            # 대조는 원본↔사본만 된다(계약서↔사본은 없다). 그 사실을 항목에 남긴다.
            wanted = sorted(p.name for p in d.iterdir() if p.is_file())
        else:
            problems.append(f"{d}: {CONTRACT_NAME} 가 없다 — 봉인본이 아니다")
            continue
        tracked = tracked_names(d) or frozenset()
        entries = _contract_entries(d) if sealed else []
        for name in wanted:
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
                "sealed": sealed,
                # 계약서 자체는 자기 해시를 담을 수 없다 (파생물이다). 봉인 아닌 곳은 None.
                "contract_sha256": next((h for h, n in entries if n == name), None),
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
                    help="대상 디렉터리. 미지정 시 봉인처 명부의 expected 전부 + judge_labels")
    ap.add_argument("--dest", default=None, help="verify: 적재한 드라이브 경로")
    ap.add_argument("--out", default=None, help="plan: 매니페스트를 쓸 파일")
    ap.add_argument("--manifest", default=None,
                    help="verify: plan 이 낸 매니페스트. 미지정 시 즉석에서 다시 만든다")
    args = ap.parse_args(argv)

    dirs = [Path(x) for x in args.dirs] if args.dirs else list(DEFAULT_DIRS)

    if args.cmd == "plan":
        plan = build_plan(dirs)
        for it in plan["items"]:
            tag = ("ㆍ" if it["tracked"] else "★") + ("" if it.get("sealed", True) else "○")
            print(f"  {tag:2s} {it['rel']:42s}"
                  f" {it['bytes']:>10,} B  {it['sha256'][:16]}…")
        print(f"\n파일 {plan['n_files']}개 / {plan['total_mb']} MB")
        print(f"  ★ git 밖 — 이 사본이 유일본이 된다: {plan['n_at_risk']}개 /"
              f" {plan['at_risk_bytes']:,} B")
        print("  ㆍgit·origin 에도 있음. 사본이 자기 계약서로 자립하도록 함께 담는다")
        print("  ○ 봉인본이 아니다(계약서 없음) — 원본↔사본만 대조된다")
        lost = [rel for rel, e in EXPECTED_SEALED.items() if e.get("status") == "lost"]
        if lost:
            print(f"  소실 기록 {len(lost)}곳은 옮길 실물이 없어 목록에서 뺐다 — 명부 대조기가 알린다")
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
