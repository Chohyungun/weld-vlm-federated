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

import sys
from pathlib import Path

# 계약·예외형의 정본은 `data.frozen_guard` 하나다 — 여기서는 가져다 쓰기만 한다.
from data.frozen_guard import CONTRACT_NAME, FrozenDirectoryError, is_frozen

__all__ = [
    "CONTRACT_NAME",
    "FrozenDirectoryError",
    "assert_not_frozen",
    "confirm_overwrite",
    "is_frozen",
    "snapshot_summary",
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
