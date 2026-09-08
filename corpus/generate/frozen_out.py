"""동결 산출 디렉터리 가드 — corpus 진입점용. 개발규약 1-1·1-6.

## 왜 있나

`run_cycle_corpus` 는 `--out` 기본값이 `corpus/generate/cycle_pilot` 이었고 그 디렉터리는
이미 sha256 으로 봉인돼 있었다. 인자 없이 한 번 돌리면 봉인본이 지워진다. `make_pairs_pilot`
쪽에는 가드가 있긴 했는데 **경로가 v1 인지만 봤다** — 그 사이 `pairs_pilot_v2` 도 봉인됐고
그게 바로 기본값이라, 있는 가드가 실제 위험을 통과시켰다. 이름을 열거하는 가드는 자산이
늘어나는 속도를 못 따라간다. 그래서 **이름이 아니라 계약을 본다.**

계약은 저장소 공통이다 — `SNAPSHOT.sha256` 이 있으면 완결된 스냅샷이고, 재파생은 항상
새 경로에 한다. 계약 판정(`is_frozen`)과 예외형(`FrozenDirectoryError`)은
`data.frozen_guard` 것을 그대로 쓴다. 여기서 다시 정의하면 두 곳이 갈린다.

`assert_writable` 을 그대로 부르지 않는 이유는 하나다. 그 함수 본문이 "이 스크립트는
manifest.csv 를 덮어쓰므로 … `--outdir` 로 새 경로를" 이라고 매니페스트 파이프라인을
지목한다. corpus 사이클에는 manifest.csv 도 `--outdir` 도 없어서, 그 문장을 읽은 사람은
없는 플래그를 찾는다. 안내가 틀린 가드는 다음 사람을 엉뚱한 데로 보낸다 — 그 함수 자신의
docstring 이 경계하는 실패다. **판정과 예외형은 공유하고, 사람이 읽는 문장만 이쪽 것을 쓴다.**

## 우회를 두지 않는 이유

`--force` 는 없다. 규약 1-6 이 "sha256 부여 후 재생성 금지"이므로 봉인본을 제자리에서
덮어쓸 정당한 사유가 없고, 탈출구는 이미 있다 — `--out` 에 새 경로를 준다. 플래그를 두면
막으려던 사고가 플래그 한 글자 뒤로 옮겨갈 뿐이다.

봉인 **자체를 다시 쓰는** 것은 성격이 다르다(자료를 지우지 않고 기록만 갱신한다).
그쪽은 `snapshot_cycle_pilot --reseal` 이 맡고, 지울 것을 화면에 찍은 뒤 사람이 직접
입력해 확인하게 한다 — `confirm_overwrite` 가 그 절차다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

# 계약·예외형의 정본은 `data.frozen_guard` 하나다 — 여기서는 가져다 쓰기만 한다.
from data.frozen_guard import CONTRACT_NAME, FrozenDirectoryError, is_frozen

_REPO = Path(__file__).resolve().parents[2]

__all__ = [
    "CONTRACT_NAME",
    "FrozenDirectoryError",
    "assert_not_frozen",
    "confirm_overwrite",
    "find_sealed",
    "is_frozen",
    "snapshot_summary",
    "tracked_names",
    "verify_contract",
]


def snapshot_summary(directory: Path) -> tuple[list[str], str | None]:
    """봉인된 구성원 이름과 snapshot_digest. 계약 파일이 없으면 `([], None)`.

    깨진 계약 파일 때문에 가드가 죽으면 안 된다 — 가드가 죽으면 호출자는 그냥 쓴다.
    읽기 실패는 조용히 빈 요약으로 떨어뜨리고, 차단 여부는 `is_frozen` 이 정한다.
    """
    p = Path(directory) / CONTRACT_NAME
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except (OSError, ValueError):   # 읽기 실패 · utf-8 아님(UnicodeDecodeError)
        return [], None
    names, digest = [], None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("# snapshot_digest"):
            digest = ln.split()[-1]
        elif ln:
            names.append(ln.split(None, 1)[-1])
    return names, digest


def assert_not_frozen(directory: Path, *, what: str = "출력 디렉터리",
                      flag: str = "--out") -> None:
    """봉인된 디렉터리를 겨눴으면 멈춘다. **인자를 읽자마자** 부른다.

    뒤에서 부르면 GPU 를 몇 시간 쓴 뒤에 막히고, 그때는 이미 중간 산출물이 봉인본 위에
    떨어져 있다. 실제로 09-02 에 `_raw_generated.json` 이 그렇게 지워졌다.
    """
    d = Path(directory)
    if not is_frozen(d):
        return
    names, digest = snapshot_summary(d)
    raise FrozenDirectoryError(
        f"{what}({d})는 봉인된 스냅샷이다 — {CONTRACT_NAME} 가 있다.\n"
        f"  봉인 구성원 {len(names)}개"
        + (f" / snapshot_digest {digest}" if digest else "")
        + "\n"
        f"  이 실행은 같은 이름의 파일을 덮어쓴다. 되돌릴 수 없다.\n"
        f"  재생성이 필요하면 {flag} 에 **새 경로**를 줘라 (관례: `..._v3` 처럼 판본 접미).\n"
        "  새 산출물을 따로 봉인한 뒤 어느 쪽이 정본인지 docs/의사결정로그.md 에 남겨라.\n"
        "  근거: 개발규약 1-6 (corpus·페어·색인은 sha256 부여 후 재생성 금지)."
    )


def confirm_overwrite(directory: Path, *, action: str, flag: str,
                      stream=None) -> None:
    """지울 것을 찍고 사람이 직접 입력해 확인하게 한다. 확인 실패면 `SystemExit`.

    비대화식(파이프·CI·백그라운드)이면 **무조건 거부한다.** 확인을 받을 수 없는 자리에서
    통과시키면 그 플래그는 확인 절차가 아니라 그냥 스위치다.
    """
    d = Path(directory)
    out = stream or sys.stdout
    names, digest = snapshot_summary(d)
    print(f"\n!! {flag} 는 {d} 의 봉인을 {action}한다.", file=out)
    if digest:
        print(f"   현재 snapshot_digest: {digest}", file=out)
    print(f"   기록된 구성원 {len(names)}개:", file=out)
    for n in names:
        print(f"     - {n}", file=out)
    print("   이 기록이 사라지면 논문에 실은 해시와 실물을 대조할 수 없다"
          " (개발규약 1-6).", file=out)

    if not sys.stdin.isatty():
        raise SystemExit(
            f"{flag} 는 확인을 받아야 한다. 비대화식 실행에서는 쓸 수 없다 —"
            " 사람이 직접 터미널에서 돌려라."
        )
    want = d.name
    print(f"   계속하려면 디렉터리 이름을 그대로 입력하라 [{want}]: ", end="", file=out)
    out.flush()
    got = sys.stdin.readline().strip()
    if got != want:
        raise SystemExit(f"입력이 {got!r} 로 {want!r} 와 다르다 — 중단한다.")


# ------------------------------------------------------------------ 계약 대조
#
# 봉인 계약서와 실물이 맞는지 보는 일은 **트리마다 답이 다르다.** 계약 구성원 일부가
# `.gitignore` 로 추적 밖이라(`corpus/generate/cycle_pilot/*.jsonl`), 그 파일들은
# 만든 워크트리에만 있고 main 체크아웃에는 없다. 그래서 "구성원이 실물과 맞는가" 를
# 단순 존재 검사로 쓰면 만든 자리에서만 통과하고 다른 곳에서는 무조건 깨진다 — 실제로
# 09-08 게이트가 그렇게 막혔다.
#
# 가르는 기준은 **git 이 그 이름을 추적하는가** 다.
#   추적분이 없다  → 있어야 할 것이 사라진 것이다. **깨졌다.**
#   미추적분이 없다 → 이 트리에 원본이 없을 뿐이다. 대조를 **건너뛰되 기록한다.**
# 조용히 넘어가면 안 된다. 판정 결과에 건너뛴 이름이 남아야 다음 사람이 "이 트리에서는
# 무엇을 못 봤는지" 를 안다.


def tracked_names(directory: Path) -> frozenset[str] | None:
    """git 이 추적하는 구성원 이름. 판단할 수 없으면 `None` (추측하지 않는다)."""
    d = Path(directory)
    try:
        # git 은 UTF-8 로 쓰는데 `text=True` 는 로캘(win32 cp949)로 읽는다. 오류 문구가
        # 한국어면 읽기 스레드가 UnicodeDecodeError 로 죽는다 — 인코딩을 명시한다.
        r = subprocess.run(["git", "ls-files", "--", str(d)],
                           cwd=str(_REPO), capture_output=True, timeout=60,
                           encoding="utf-8", errors="replace", check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return frozenset(PurePosixPath(x.strip()).name
                     for x in r.stdout.splitlines() if x.strip())


def verify_contract(directory: Path, *,
                    tracked: frozenset[str] | None = None) -> dict:
    """계약서와 실물 대조. 판정은 넷이다.

    * `no_contract`     — 계약 파일이 없다. 봉인본이 아니다.
    * `ok`              — 구성원이 전부 이 트리에 있고 이름이 맞는다. **엄격히 봤다.**
    * `incomplete_tree` — 없는 것이 전부 **추적 밖**이다. 이 트리에 원본이 없을 뿐이다.
    * `broken`          — **추적분**이 없다. 있어야 할 것이 사라졌다.

    `unverified` 에 건너뛴 이름이 남는다 — 이것이 "조용한 통과" 를 막는 장치다.
    """
    d = Path(directory)
    names, digest = snapshot_summary(d)
    out: dict[str, Any] = {"dir": str(d), "digest": digest,
                           "n_members": len(names), "members": names}
    if not names:
        out["verdict"] = "no_contract"
        out["unverified"] = []
        return out

    missing = [n for n in names if not (d / n).is_file()]
    if tracked is None:
        tracked = tracked_names(d)
    if tracked is None:
        # git 을 못 물었다. 없는 것이 추적분인지 판단할 수 없으니 깨졌다고도,
        # 괜찮다고도 하지 않는다.
        out.update(verdict="unverifiable" if missing else "ok",
                   unverified=missing, missing_tracked=[],
                   reason="git ls-files 를 부르지 못해 추적 여부를 판단할 수 없다")
        return out

    out["missing_tracked"] = [n for n in missing if n in tracked]
    out["unverified"] = [n for n in missing if n not in tracked]
    if out["missing_tracked"]:
        out["verdict"] = "broken"
        out["reason"] = ("계약에 있고 git 이 추적하는데 실물이 없다 —"
                         " 봉인본이 훼손됐다")
    elif out["unverified"]:
        out["verdict"] = "incomplete_tree"
        out["reason"] = (f"추적 밖 구성원 {len(out['unverified'])}개가 이 트리에 없다."
                         " 만든 워크트리에만 있는 자산이다 (.gitignore)")
    else:
        out["verdict"] = "ok"
    return out


def find_sealed(*roots: Path) -> list[Path]:
    """계약 파일을 가진 디렉터리. 하위 1단만 훑는다 — 정션 아래를 재귀하면 멈춘다."""
    found: list[Path] = []
    for root in roots or (_REPO / "corpus/generate", _REPO / "data/processed"):
        r = Path(root)
        if not r.is_dir():
            continue
        for d in [r, *(x for x in r.iterdir() if x.is_dir())]:
            if is_frozen(d) and d not in found:
                found.append(d)
    return found


def main(argv: list[str] | None = None) -> int:
    """`uv run python -m corpus.generate.frozen_out [디렉터리…]`

    게이트가 어느 트리에서든 돌려 "여기서 무엇을 못 봤는지" 를 볼 수 있게 한다.
    깨진 것이 있을 때만 실패한다 — 트리에 원본이 없는 것은 실패가 아니다.
    """
    ap = argparse.ArgumentParser(description="봉인 계약서와 실물 대조")
    ap.add_argument("dirs", nargs="*", help="미지정 시 corpus/generate·data/processed 를 훑는다")
    ap.add_argument("--json", action="store_true", help="판정을 JSON 으로 출력")
    args = ap.parse_args(argv)

    targets = [Path(x) for x in args.dirs] if args.dirs else find_sealed()
    results = [verify_contract(t) for t in targets]
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        for r in results:
            mark = {"ok": "○", "incomplete_tree": "△", "broken": "✕",
                    "unverifiable": "?", "no_contract": "-"}[r["verdict"]]
            try:
                shown = Path(r["dir"]).relative_to(_REPO)
            except ValueError:
                shown = Path(r["dir"])
            print(f"{mark} {shown}  구성원 {r['n_members']}개  [{r['verdict']}]")
            if r.get("missing_tracked"):
                print(f"    실물 없음(추적분): {r['missing_tracked']}")
            if r.get("unverified"):
                print(f"    이 트리에서 대조 못 함: {r['unverified']}")
        n_broken = sum(1 for r in results if r["verdict"] == "broken")
        n_skip = sum(len(r.get("unverified") or []) for r in results)
        print(f"\n봉인 {len(results)}곳 · 깨짐 {n_broken} · 대조 못 한 구성원 {n_skip}개")
    return 1 if any(r["verdict"] == "broken" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
