"""봉인 전 관문 — 검사 **둘을 다 돌리고** 하나라도 통과가 아니면 거부한다. 읽기 전용이다.

검사기가 둘 있어도 부르는 코드가 없으면 방어가 사람의 기억에 얹힌다. 한 번 빠뜨리면 그대로 봉인된다.
이 스크립트가 그 자리를 메운다 — 봉인 절차(16번 §3)의 1·2번은 여기서만 돌린다.

| # | 무엇 | 통과 조건 |
|---|---|---|
| 1 | `corpus.validate.verify_pairs_main` (같은 트랙) | 종료 0 · `ok` · `coverage: full` · `not_checked: []` · `verdict: 일치` |
| 2 | `scripts/probe/pairs_independent.py` (독립 재구현) | `mismatch: {}` · `coverage: full` · `sealable: true` · 종료 0 |

**종료 코드만 보지 않는다.** 2번은 `--pairs` 를 주지 않으면 대조 없이 끝나고, `--allow` 를 주면
불일치를 면제한다. 그래서 여기서는 `--pairs` 를 반드시 주고 `--allow` 를 **주지 않으며**,
돌려받은 보고의 내용까지 다시 본다.

**2번의 분류는 `verdict` 문장이 아니라 기계가 읽는 필드로 한다.** 그 대조기는 범위 제한일 때
`verdict` 에 "대조 범위 제한 — 본 것은 전부 같았다" 를 쓴다. 문장으로 가르면 그 보고가 "일치가
아니다" 라는 이유로 **적발**이 되고, 범위 제한이 있었다는 사실이 보고에서 사라진다(교차 검수 I-1).

종료 코드는 두 검사기와 같은 규약이다 — **0 = 둘 다 통과(봉인 가), 1 = 적발·실행 실패, 2 = 대조 범위 제한.**
**적발과 실행 실패가 범위 제한보다 앞선다.** 한 검사에 적발이 있고 다른 검사에 범위 제한이 있으면 **1** 이다 —
낮은 단계로 요약하면 최종 상태만 읽는 쪽이 적발을 "못 본 축이 있다" 로 분류한다.
그렇다고 범위 제한을 지우지는 않는다: 보고의 `범위제한_있음`·`범위제한_검사` 가 그 사실을 따로 싣는다.
봉인 조건은 0 이다. 파일은 2번의 산출 디렉터리에만 쓴다(미추적 경로).

이 관문이 **덮지 않는 것**: 절차 §3-3 의 두 번째 빌드 바이트 대조. 그것은 빌드를 한 번 더 돌리는 일이라
여기서 하지 않는다. 0 을 "봉인 절차가 끝났다" 로 읽지 않는다.

실행: uv run python -m corpus.validate.seal_pairs_main --build data/processed/pairs_main_v1
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
from pathlib import Path

from corpus.validate.verify_pairs_main import SNAP

REPO = Path(__file__).resolve().parents[2]
INDEPENDENT = REPO / "scripts/probe/pairs_independent.py"
LABEL_MAP = REPO / "configs/label_map.yaml"


#: 절대경로. 윈도(드라이브 문자)와 POSIX(`/`) 둘 다 본다. `/` 는 낱말 뒤나 " / " 같은 구분자에서
#: 걸리지 않게 왼쪽 경계와 뒤 한 글자를 요구한다 — 사유를 잇는 " / " 가 경로로 읽히면 안 된다.
_ABS_PATH = re.compile(r'''[A-Za-z]:[\\/][^\s"'<>|]* | (?<![\w.])/(?=[^\s/])[^\s"'<>|]*''', re.VERBOSE)

def _scrub(text: str) -> str:
    """로그·사유에서 절대경로를 감춘다 — 저장소 안이면 상대경로로, 밖이면 자리표로(개발규약 2-6).

    `_rel()` 로 경로를 감추면서 다른 문구를 그대로 실으면 감춘 뜻이 없다.

    **공백이 든 경로는 뿌리까지만 지운다.** 정규식으로는 경로의 끝을 알 수 없어서 첫 공백에서 멈추고,
    `E:\\…\\some dir\\x.py` 는 `some dir\\x.py` 로, 집 경로는 `<경로> Papers\\x.py` 로 남는다.
    남는 것은 **절대경로가 아니다** — 거기까지가 이 함수의 약속이다.
    """
    def one(m):
        raw = m.group(0)
        try:
            return Path(raw).resolve().relative_to(REPO).as_posix()
        except (ValueError, OSError):
            return "<경로>"
    return _ABS_PATH.sub(one, text)


def _run(argv: list[str]) -> tuple[int, dict | None, str]:
    """검사기를 돌리고 (종료 코드, 표준 출력의 JSON, 꼬리 로그)를 돌려준다."""
    proc = subprocess.run([sys.executable, "-X", "utf8", "-B", *argv], cwd=REPO, check=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout or ""
    doc = None
    start = out.find("{")
    if start >= 0:
        try:
            doc = json.loads(out[start:])
        except ValueError:
            doc = None
    tail = (proc.stderr or "").strip().splitlines()[-3:]
    return proc.returncode, doc, " / ".join(tail)[:300]


def _rel(p: Path) -> str:
    """저장소 기준 상대경로. 밖에 있으면 이름만 — 로컬 절대경로를 보고에 싣지 않는다.

    **먼저 있는 그대로 맞춰 본다.** `resolve()` 는 정션을 따라가 본체 실경로로 바꾸므로
    `outputs/…` 처럼 정션 아래 경로가 이름 한 토막으로 줄어든다(그 아래가 공유 산출 경로다).
    """
    p = Path(p)
    for cand in ((p if p.is_absolute() else REPO / p), p.resolve()):
        try:
            return cand.relative_to(REPO).as_posix()
        except (ValueError, OSError):
            continue
    return p.name


#: 검사 하나의 상태. 세 갈래를 불리언 둘로 표현하면 합성 규칙이 코드에서 보이지 않는다.
PASS, FOUND, BROKEN, LIMITED = "통과", "적발", "실행실패", "범위제한"


def _check(name: str, code: int | None, *, findings=(), limited=(), broken=(),
           summary: dict | None = None, out: str | None = None) -> dict:
    """검사 하나의 결과. **적발·실행 실패가 범위 제한보다 앞선다**(외부 검토 §2).

    사유는 상태가 무엇이든 **전부** 남긴다 — 적발로 판정해도 못 본 축이 있었다는 사실은 지우지 않는다.

    **같은 낱말이 층마다 다른 것을 가리킨다**(이름은 옛 소비자 때문에 그대로 둔다):

    | 자리 | 뜻 |
    |---|---|
    | 검사 행의 `범위제한` | **상태의 별칭** — `상태 == "범위제한"` 일 때만 참. 적발·실행실패면 거짓 |
    | 검사 행의 `미대조_있음` | 못 본 축이 **있었다는 사실** — 상태와 무관 |
    | 최상위 `범위제한_있음`·`범위제한_검사` | 검사 행의 `미대조_있음` 을 센 것이다. **적발·실행실패 검사도 든다** |

    그래서 `범위제한_검사` 에 든 검사를 "범위 제한 상태" 로 읽으면 틀린다. 상태는 `상태` 를 본다.
    """
    state = BROKEN if broken else (FOUND if findings else (LIMITED if limited else PASS))
    row = {"검사": name, "상태": state, "통과": state == PASS,
           # 옛 소비자를 위해 남긴 불리언이다. 판정의 정본은 `상태` 다.
           "범위제한": state == LIMITED,
           # **상태와 별개로** 미대조 축이 있었는지를 남긴다. 상태만 보면 적발인 검사의 범위 제한이
           # 최종 층에서 사라진다 — 판정을 올리는 것과 사실을 버리는 것은 다르다(교차 검수 I-4).
           "미대조_있음": bool(limited),
           # 사유는 보고로 나간다. **여기서 절대경로를 감춘다** — 잡는 자리가 아니라 적는 자리에서 해야
           # 표준오류 꼬리든 예외 문구든 빠짐없이 지난다(개발규약 2-6).
           "종료코드": code, "사유": [_scrub(str(x)) for x in (*broken, *findings, *limited)]}
    if summary is not None:
        row["요약"] = summary
    if out is not None:
        row["산출"] = out
    return row


#: 독립 대조기의 보고에 **반드시** 있어야 하는 필드. 그 모듈은 어느 경로로 끝나든 이 넷을 싣는다
#: (`scripts/probe/pairs_independent.py` 의 머리말과 `main()` 끝의 `sealable` 대입 주석).
#: 없는 보고는 그 모듈의 산출로 받지 않는다. **줄 번호로 가리키지 않는다** — 판본마다 밀린다.
INDEPENDENT_FIELDS = ("mismatch", "coverage", "not_compared", "sealable")


def _independent(rep: dict, code: int | None) -> dict:
    """독립 대조기의 보고를 **기계가 읽는 필드**로 가른다 (교차 검수 I-1).

    `verdict` 는 사람이 읽는 문장이다. 그것으로 가르면 범위 제한 보고("대조 범위 제한 — 본 것은
    전부 같았다")가 "일치가 아니다" 라는 이유로 적발이 되고, `범위제한_있음` 이 사실과 반대로 나간다.

    | 보고 | 종료 | 상태 |
    |---|---|---|
    | `mismatch` 가 비지 않음 (`blocking` 도 남음) | 1 | 적발 |
    | `mismatch` 가 비지 않음 (허용 범주로 `blocking` 을 비움) | 3 | 적발 — 비운 불일치도 불일치다 |
    | `mismatch` 가 비고 `coverage: partial` | 2 | 범위 제한 |
    | `mismatch` 가 비고 `coverage: full` | 0 | 통과 |
    | `coverage: none` | 2 | **실행 실패** — 아래 이유 |
    | 필드가 서로 어긋남 | — | 실행 실패 |
    """
    missing = [k for k in INDEPENDENT_FIELDS if k not in rep]
    if missing:
        return {"broken": [f"보고에 계약 필드가 없다: {missing}"]}
    coverage = rep["coverage"]
    if coverage == "none":
        # `--pairs` 를 주지 않은 실행의 표지다. 관문은 **늘** 주므로, 이 보고가 왔다면 우리가 부른 실행이
        # 아니다. 범위가 좁았던 것이 아니라 **대조를 한 건도 하지 않은 것**이라 범위 제한으로 내리지 않는다.
        return {"broken": [f"대조하지 않은 보고다(coverage {coverage}) — 관문은 --pairs 를 준다"]}
    findings, limited = [], []
    if rep["mismatch"]:
        # `blocking` 이 아니라 `mismatch` 를 본다. 허용 범주로 비운 불일치도 불일치다.
        findings.append(f"불일치 {sorted(rep['mismatch'])}")
    if coverage != "full":
        # **`elif` 가 아니다.** `mismatch` 와 `not_compared` 는 서로 독립이고 그 대조기는 둘을 함께 낸다.
        # 앞서 `elif` 였던 탓에 한 보고가 둘을 들고 오면 범위 제한이 지워졌다(교차 검수 I-4).
        # 상태는 적발이 맞다 — 지워지면 안 되는 것은 기계가 읽으라고 만든 사실이다.
        limited.append(f"대조 못 한 축 {rep['not_compared']}")
    want = (1 if rep.get("blocking") else 3) if rep["mismatch"] else (0 if coverage == "full" else 2)
    broken = []
    if code != want:
        broken.append(f"종료 코드 {code} — 보고는 {want} 를 뜻한다. 어느 쪽도 믿을 수 없다")
    if bool(rep["sealable"]) != (not findings and not limited and not broken):
        broken.append(f"sealable {rep['sealable']} 이 보고의 나머지와 어긋난다")
    return {"findings": findings, "limited": limited, "broken": broken}


def check(build: Path, snap: Path, root: Path, out_dir: Path) -> dict:
    from corpus.generate.make_pairs_pilot import LIMITS_CSV

    results: list[dict] = []

    # ── 1. 같은 트랙의 산출물 검사
    code, rep, err = _run(["-m", "corpus.validate.verify_pairs_main", "--build", str(build),
                           "--snapshot", str(snap), "--root", str(root)])
    if rep is None:
        results.append(_check("verify_pairs_main", code, broken=["보고를 읽지 못했다", err]))
    else:
        findings, limited = [], []
        if rep.get("failures"):
            findings.append(f"적발 {sorted(rep['failures'])}")
        if bool(rep.get("not_checked")) or rep.get("coverage") != "full":
            limited.append(f"대조 못 한 축 {rep.get('not_checked')}")
        if not findings and not limited and (rep.get("verdict") != "일치" or not rep.get("ok")):
            # 적발도 미대조도 없다면서 일치가 아니라고 한다 — 보고가 스스로 어긋난 것이다.
            findings.append(f"판정 {rep.get('verdict')} · ok {rep.get('ok')}")
        want = 1 if findings else (2 if limited else 0)
        broken = ([] if code == want else
                  [f"종료 코드 {code} — 보고는 {want} 를 뜻한다. 어느 쪽도 믿을 수 없다"])
        results.append(_check("verify_pairs_main", code, findings=findings, limited=limited, broken=broken,
                              summary={k: rep.get(k) for k in ("n_records", "n_discarded", "coverage",
                                                               "verdict", "n_image_paths_sampled")}))

    # ── 2. 독립 재구현과의 대조. `--allow` 를 주지 않는다 — 면제하고도 0 이 나온다.
    if not INDEPENDENT.is_file():
        # 돌지 못한 것은 "못 봤다" 가 아니다. 단순 미대조와 같은 칸에 넣으면 구별이 사라진다.
        results.append(_check("pairs_independent", None, broken=[f"{_rel(INDEPENDENT)} 가 없다"]))
    elif out_dir.exists():
        results.append(_check("pairs_independent", None,
                              broken=[f"산출 경로가 이미 있다: {_rel(out_dir)}. 새 이름을 준다 — 지우지 않는다"]))
    else:
        code, rep, err = _run([str(INDEPENDENT), "--manifest-dir", str(snap), "--limits", str(LIMITS_CSV),
                               "--label-map", str(LABEL_MAP), "--pairs", str(build), "--out", str(out_dir)])
        # 대조기가 입력 계약 위반(`InputContractError` — 계약서의 파일별 해시·지문·구성원·줄 꼴이 어긋남)으로
        # 멈추면 `main()` 이 예외를 잡지 않아 표준 출력에 보고가 없고 종료 1 로 끝난다. 그 경우가 **이 갈래**다 —
        # 범위 제한이 아니라 `실행실패` 로 받는다. 입력을 믿을 수 없어 대조가 시작되지 못한 것이다.
        if rep is None:
            results.append(_check("pairs_independent", code, broken=["보고를 읽지 못했다", err]))
        else:
            results.append(_check("pairs_independent", code, **_independent(rep, code),
                                  summary={k: rep.get(k) for k in ("n_expected_keep", "n_expected_discard",
                                                                   "n_got", "verdict", "coverage",
                                                                   "not_compared", "counts_note")},
                                  out=_rel(out_dir)))

    # **적발·실행 실패가 범위 제한보다 앞선다.** `all`·`any` 로 합치면 적발이 범위 제한으로 내려가고,
    # 최종 상태만 읽는 쪽이 종료 코드 2 를 "적발은 없었다" 로 읽는다(외부 검토 §2).
    blocked = [r["검사"] for r in results if r["상태"] in (FOUND, BROKEN)]
    # 상태가 아니라 **사실**로 센다. 적발로 판정된 검사도 미대조 축이 있었으면 여기 든다.
    limited_checks = [r["검사"] for r in results if r["미대조_있음"]]
    # "대조해서 어긋났다" 와 "검사기가 돌지 못했다" 는 종료 코드가 같아도(둘 다 1) 다른 사실이다.
    broken_checks = [r["검사"] for r in results if r["상태"] == BROKEN]
    passed = all(r["상태"] == PASS for r in results)
    return {
        # 최상위 경로도 감추는 함수를 지난다. `--snapshot` 기본값이 절대경로라 그대로 실렸다(m-7).
        "build": _rel(build), "snapshot": _rel(snap),
        "검사": results,
        "봉인_가능": passed,
        "판정": "통과" if passed else ("불통과" if blocked else "대조 범위 제한"),
        "불통과_검사": blocked,
        "실행실패_검사": broken_checks,
        # 적발로 판정해도 **범위 제한이 있었다는 사실은 지우지 않는다.**
        "범위제한_있음": bool(limited_checks),
        "범위제한_검사": limited_checks,
        # 0 을 "절차가 끝났다" 로 읽지 않게 이 관문이 덮지 않는 것을 산출물이 스스로 말한다.
        "덮지_않는_것": ["절차 §3-3 두 번째 빌드 바이트 대조", "타일 이미지의 내용 해시",
                     "허용치 표의 사람 이중 검수", "골격의 사실성(사람 gold)"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, type=Path, help="검사할 빌드 디렉터리")
    ap.add_argument("--snapshot", type=Path, default=SNAP, help="빌드가 나온 매니페스트 스냅샷")
    ap.add_argument("--root", type=Path, default=REPO, help="`image_path` 를 푸는 기준")
    ap.add_argument("--out", type=Path, default=None,
                    help="독립 대조의 산출 경로(미추적·새 경로). 생략하면 outputs 아래에 실행 시각으로 만든다")
    args = ap.parse_args(argv)
    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or (REPO / "outputs/seal_pairs_main" / f"{args.build.name}_{stamp}")
    report = check(args.build, args.snapshot, args.root, out_dir)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    if report["봉인_가능"]:
        return 0
    # 적발·실행 실패가 있으면 1 이다. 범위 제한만 남았을 때에만 2 로 내려간다.
    return 1 if report["판정"] == "불통과" else 2


if __name__ == "__main__":
    raise SystemExit(main())
