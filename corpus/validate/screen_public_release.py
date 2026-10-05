"""공개 저장소 전환 전 전수 검수 — 09-08 판정 (24번 부록 A-4 ①).

`corpus/generate/cycle_pilot*/` 의 미추적 산출물을 git 추적으로 돌리기 전에, 그 내용이
**공개 저장소로 나가도 되는지**를 한 건도 빠짐없이 본다. `origin` 은 공개 GitHub
(`Chohyungun/weld-vlm-federated`)이고 해외 호스팅이다.

## 무엇을 보는가

판정이 지목한 것은 `PAID_STD` 스크린이다. 그런데 저장소 자체 정책이 그보다 넓다 —
`.gitignore:54` 가 규정 원문 덤프를 막으며 이렇게 적었다.

> 무료 공개 문서라도 '열람·이용'과 '재배포'는 다른 권리다.
> 저장소가 공개 전환될 수 있으므로 원문·표 덤프는 추적하지 않는다.

즉 KR-RULES-P2·IACS47 이 규약 2-5 의 **무료 공개 허용 목록**에 있어도, 그 원문을 길게
그대로 실어 공개 저장소에 올리는 것은 이미 저장소가 스스로 금지한 행위다. QA 축은
규정 구절을 재료로 쓰므로 여기가 실제 위험이다. 그래서 차단 사유를 넷 둔다.

| 사유 | 무엇 | 왜 차단인가 |
|---|---|---|
| `paid_std` | ISO 5817·10042·10675 · AWS Welding Handbook 식별자 | 유료 표준 전재 (규약 2-5) |
| `verbatim_source` | 규정 원문과 40자 이상 연속 일치 | 재배포 (.gitignore:54) |
| `aihub_id` | `aihub#####:` 형태 식별자 | 국외 반출 금지 (레드라인 1) |
| `local_path` | 로컬 절대경로 | 개인 환경 노출 (규약 2-6) |

`PAID_STD` 는 `run_cycle_corpus` 정본을 가져다 쓴다 — 같은 성격의 정규식을 여기 다시
두지 않는다(80번 G2-6).

## 판정

**한 건이라도 걸리면 전환하지 않는다** (지시 사항). 종료코드 1.

**대상은 미추적 산출물이다 — 새 clone 의 기본 상태에서는 `unverified`(미검사, 종료코드 2)가 정상이다.**
검사 대상(`cycle_pilot*` 의 미추적 구성원)도 재배포 판정의 원천 문서도 git 밖이라 clone 에는 없다.
그때 나오는 `no_target_dir`·`no_source_docs` 는 고장이 아니라 "여기서는 검사할 수 없다" 는 사실 표시다
(09-19 판정). 실제 검사는 그 산출물이 있는 트리에서 돌린다.

실행:
  uv run python -m corpus.validate.screen_public_release
  uv run python -m corpus.validate.screen_public_release --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from corpus.generate.frozen_out import snapshot_summary, tracked_names

# 유료 표준 식별자 스크린의 정본. 여기서 다시 정의하지 않는다.
from corpus.generate.run_cycle_corpus import PAID_STD

REPO = Path(__file__).resolve().parents[2]

#: 검수 대상 봉인처. 추적 밖 구성원만 본다 — 이미 추적 중인 것은 이미 공개됐다.
TARGET_DIRS = (
    REPO / "corpus/generate/cycle_pilot",
    REPO / "corpus/generate/cycle_pilot_v2",
)

#: 재배포 판정의 원천. 이 문서들과 길게 겹치면 원문을 실어 나른 것이다.
SOURCE_DOCS = (
    REPO / "corpus/parse/survey/KR-RULES-P2/KR-RULES-P2_p316-336.md",
    REPO / "corpus/parse/survey/IACS47/IACS47_full.md",
)

#: 연속 일치 길이 한계. 40자면 우연이 아니라 인용이다 (한국어 기준 약 두 문장 조각).
SHINGLE = 40

AIHUB_ID = re.compile(r"aihub\d+\s*:")

#: 로컬 절대경로. **폴더 이름 목록에 기대지 않는다** (09-16 추기). 옛 규칙은 드라이브 다음
#: 첫 폴더가 `Users`·`Program Files`·이 저장소 이름일 때만 잡아서, 목록 밖 폴더·공유 드라이브·
#: 이스케이프가 한 겹 남은 경로(문자열에 JSON 이 다시 들어 있는 경우)를 놓쳤다.
#:
#: 1. 드라이브 — 문자 하나 + `:` + 구분자 1~2개(`\` · `\\` · `/`) + 폴더 이름의 첫 글자.
#:    드라이브 문자 앞이 영문자·숫자면 잡지 않는다: `https://` 의 `s:`, YAML 문자열의
#:    `documents:\n` 같은 것. 시각(`12:30`)·규격 표기(`5817:2014`)는 콜론 앞이 숫자라 애초에 안 걸린다.
#: 2. UNC — `\\호스트\공유`. 앞이 역슬래시·단어 문자가 아닌 곳에서 역슬래시 2개(한 겹 더 이스케이프된
#:    4개 포함) + 두 글자 이상 호스트 + 구분자 + 공유 이름. 정규식 조각(`\d+\s`)은 호스트 자리에
#:    기호가 와서 안 걸린다.
#: 3. POSIX 사용자 폴더 — `/home/…`·`/Users/…` (옛 갈래 그대로).
LOCAL_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]{1,2}[^\s\\/:*?\"<>|][^\s:*?\"<>|]*"
    r"|(?<![\\\w])\\\\(?:\\\\)?[\w.$-]{2,}\\{1,2}[\w$][^\s\\/:*?\"<>|]*"
    r"|/(?:home|Users)/[A-Za-z]"
)

#: 차단 사유. 하나라도 나오면 전환하지 않는다.
BLOCKING = ("paid_std", "verbatim_source", "aihub_id", "local_path")

#: 기본 제외 — **판정받은 두 스냅샷 디렉터리뿐이다.** 09-18 판정이 "합성" 으로 본 것은 이 둘이고
#: (해시·지각 해시·경로가 전부 생성기 산물, 접두 형식만 같고 값은 실물이 아니다), 판정은 본 범위에만 선다.
#: `data/mock/` 아래의 다른 경로(안내문 포함)와 나중에 생기는 자료는 **검사 대상이다** — 경로 이름이 mock
#: 이라는 이유만으로 새 자료까지 자동 승인하지 않는다(09-21 추기).
#: 기본 대상에는 없다. **이 경로를 대상으로 줬을 때만** 작동한다 — 차단 규칙 자체는 그대로다.
EXCLUDED_PREFIXES = ("data/mock/mock_aihub_v1", "data/mock/mock_riawelc_v1")
EXCLUDE_REASON = ("합성 데이터 — 09-18 판정(해시·지각 해시·경로가 전부 생성기 산물이고 출처 접두의 형식만 실물과 같다). "
                  "판정 범위는 두 스냅샷 data/mock/mock_aihub_v1 · data/mock/mock_riawelc_v1 뿐이고, "
                  "그 밖의 data/mock/** 은 판정 밖이라 검사한다")

#: 판정 셋. `clear` 는 **실제로 검사한 문자열이 있을 때만**이다.
VERDICT_NOTE = {
    "blocked": "차단 사유가 나왔다 — 전환하지 않는다",
    "clear": "검사한 문자열에서 차단 사유가 나오지 않았다",
    "unverified": "검사를 못 했다 — 위험이 없다는 뜻이 아니다",
}

#: 종료 코드. 게시용 검사에서 필수 대상 누락은 통과할 수 없다.
EXIT_CODE = {"clear": 0, "blocked": 1, "unverified": 2}


def normalize(text: str) -> str:
    """공백만 무너뜨린다. 표 조판이 달라도 같은 문장은 같은 문장이다."""
    return re.sub(r"\s+", " ", text).strip()


def doc_shingles(path: Path, n: int = SHINGLE) -> set[str]:
    """원천 문서 **하나**의 n자 연속 조각. 정규화한 뒤 n자 미만이면 빈 집합이다.

    길이는 정규화(공백 무너뜨리기) **뒤에** 잰다 — 공백으로 부풀린 파일이 긴 문서로 읽히면 안 된다.
    """
    t = normalize(path.read_text(encoding="utf-8", errors="replace"))
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def source_shingles(docs=None, n: int = SHINGLE) -> set[str]:
    """원천 문서의 n자 연속 조각. 없으면 빈 집합 — 그 경우 판정에 남긴다.

    기본값을 인자에 박지 않는다 — 그러면 import 시점에 묶여서 대상을 바꿀 수 없고,
    검수기를 검수하는 시험이 실제 대상을 건드리게 된다.
    """
    out: set[str] = set()
    for p in (SOURCE_DOCS if docs is None else docs):
        if p.is_file():
            out |= doc_shingles(p, n)
    return out


def iter_texts(path: Path):
    """파일에서 사람이 읽는 문자열만 뽑는다. `(레코드 키, 문자열)` 을 흘린다."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for i, line in enumerate(raw.splitlines()):
            if line.strip():
                yield from _walk(json.loads(line), f"{path.name}#{i}")
    else:
        yield from _walk(json.loads(raw), path.name)


def _walk(obj, where: str):
    if isinstance(obj, str):
        yield where, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{where}[{i}]")


def screen_text(text: str, shingles: set[str]) -> list[tuple[str, str]]:
    """이 문자열의 차단 사유. `(사유, 증거)` 목록 — 비면 깨끗하다."""
    hits: list[tuple[str, str]] = []
    m = PAID_STD.search(text)
    if m:
        hits.append(("paid_std", m.group(0)))
    m = AIHUB_ID.search(text)
    if m:
        hits.append(("aihub_id", m.group(0)))
    m = LOCAL_PATH.search(text)
    if m:
        hits.append(("local_path", m.group(0)))
    if shingles:
        t = normalize(text)
        for i in range(max(0, len(t) - SHINGLE + 1)):
            frag = t[i:i + SHINGLE]
            if frag in shingles:
                hits.append(("verbatim_source", frag))
                break
    return hits


def untracked_members(directory: Path) -> list[str]:
    """계약 구성원 중 추적 밖인 것 = 이번에 새로 공개될 파일."""
    names, _ = snapshot_summary(directory)
    tracked = tracked_names(directory)
    if tracked is None:
        raise SystemExit(f"{directory}: git 추적 여부를 판단할 수 없다 — 검수를 멈춘다")
    return [n for n in names if n not in tracked]


def _shown(p: Path) -> str:
    """저장소 밖 경로도 그대로 보여준다 — 여기서 죽으면 검수 자체가 안 돈다."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def _excluded_by_prefix(d: Path) -> bool:
    """기본 제외 접두 아래인가. 대상으로 **줬을 때만** 본다 — 기본 대상에는 이 경로가 없다."""
    rel = _shown(d).replace("\\", "/")
    return any(rel == pre or rel.startswith(pre + "/") for pre in EXCLUDED_PREFIXES)


def screen(dirs=None, *, excludes: dict[str, str] | None = None) -> dict:
    """대상별 실재·구성원·검사량을 함께 낸다.

    **`clear` 는 실제로 검사한 문자열이 있을 때만이다.** 옛 판은 대상 디렉터리가 없으면 조용히
    건너뛰어, 한 글자도 안 본 실행이 적발 0·`clear` 로 나왔다(검토 §20-5). 이제 그런 실행은
    `unverified` 다 — **"검사를 못 했다" 이지 "위험 없음" 이 아니다.**
    """
    dirs = TARGET_DIRS if dirs is None else dirs
    excludes = excludes or {}
    # 원천은 **하나씩** 본다. 합집합의 조각 수와 파일 존재만 보면, 원천 하나가 비어 있어도
    # 다른 원천의 조각에 가려 정상 검사로 읽힌다(검토 §23-3).
    sources: list[dict] = []
    shingles: set[str] = set()
    for p in SOURCE_DOCS:
        exists = p.is_file()
        own = doc_shingles(p) if exists else set()
        shingles |= own
        sources.append({"path": _shown(p), "exists": exists, "n_shingles": len(own)})
    report: dict = {
        "shingle_len": SHINGLE,
        "n_source_shingles": len(shingles),
        "source_docs_present": [s["path"] for s in sources if s["exists"]],
        "sources": sources,
        "targets": [],
        "files": [],
        "unverified_reasons": [],
    }

    def _unverified(reason: str, what: str) -> None:
        report["unverified_reasons"].append({"reason": reason, "what": what})

    if not shingles:
        # 비교할 조각이 하나도 없으면 재배포 판정을 못 한 것이다. 통과로 적지 않는다.
        report["warning"] = ("원천 문서가 이 트리에 없어 verbatim_source 를 판정하지 못했다"
                             " — 이 상태의 통과는 근거가 아니다")
    n_present = len(report["source_docs_present"])
    if n_present < len(SOURCE_DOCS) or not SOURCE_DOCS:
        _unverified("no_source_docs", f"{len(SOURCE_DOCS)}개 중 {n_present}개만 있다")
    for s in sources:
        if s["exists"] and not s["n_shingles"]:
            # 파일은 있는데 비교 조각을 못 만든다 — 비었거나, 정규화 뒤 SHINGLE 자 미만이다.
            _unverified("source_without_shingles", s["path"])

    for d in dirs:
        d = Path(d)
        shown = _shown(d)
        t: dict = {"path": shown, "exists": d.is_dir(), "excluded": False,
                   "exclude_reason": None, "n_members": 0, "n_untracked": 0,
                   "missing_members": []}
        reason = excludes.get(d.name) or excludes.get(shown.replace("\\", "/"))
        if reason is None and _excluded_by_prefix(d):
            reason = EXCLUDE_REASON
        if reason is not None:
            t["excluded"], t["exclude_reason"] = True, reason
            report["targets"].append(t)
            continue
        if not d.is_dir():
            _unverified("no_target_dir", shown)
            report["targets"].append(t)
            continue
        names, _digest = snapshot_summary(d)
        t["n_members"] = len(names)
        members = untracked_members(d)
        t["n_untracked"] = len(members)
        if not names:
            _unverified("missing_members", f"{shown}: 계약서가 없어 구성원을 알 수 없다")
        for name in members:
            p = d / name
            entry = {"path": _shown(p), "exists": p.is_file(),
                     "n_texts": 0, "hits": []}
            if p.is_file():
                for where, text in iter_texts(p):
                    entry["n_texts"] += 1
                    for reason_, evidence in screen_text(text, shingles):
                        entry["hits"].append({"reason": reason_, "where": where,
                                              "evidence": evidence[:120]})
            else:
                t["missing_members"].append(name)
            report["files"].append(entry)
        if t["missing_members"]:
            _unverified("missing_members", f"{shown}: {t['missing_members']}")
        report["targets"].append(t)

    report["n_targets"] = len(report["targets"])
    report["n_targets_missing"] = sum(1 for t in report["targets"]
                                      if not t["exists"] and not t["excluded"])
    report["n_targets_excluded"] = sum(1 for t in report["targets"] if t["excluded"])
    report["n_members_missing"] = sum(len(t["missing_members"]) for t in report["targets"])
    report["n_files_examined"] = sum(1 for f in report["files"] if f["exists"])
    report["n_texts_examined"] = sum(f["n_texts"] for f in report["files"])
    report["n_hits"] = sum(len(f["hits"]) for f in report["files"])
    report["n_blocking"] = sum(1 for f in report["files"] for h in f["hits"]
                               if h["reason"] in BLOCKING)
    if report["n_texts_examined"] == 0:
        _unverified("nothing_examined", "검사한 문자열이 0개다")
    if report["n_blocking"]:
        report["verdict"] = "blocked"                 # 적발이 먼저다
    elif report["unverified_reasons"]:
        report["verdict"] = "unverified"
    else:
        report["verdict"] = "clear"
    report["verdict_note"] = VERDICT_NOTE[report["verdict"]]
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="공개 저장소 전환 전 전수 검수",
        epilog="대상은 미추적 산출물이다 — 새 clone 의 기본 상태에서는 unverified(미검사, 종료 2)가"
               " 정상이고, 그것은 '여기서는 검사할 수 없다' 는 사실 표시다.")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], metavar="이름=사유",
                    help="대상을 명시적으로 뺀다. 사유는 결과에 남는다 (반복 가능)")
    args = ap.parse_args(argv)

    excludes: dict[str, str] = {}
    for spec in args.exclude:
        name, _, reason = spec.partition("=")
        if not name or not reason:
            ap.error(f"--exclude 는 '이름=사유' 꼴이어야 한다: {spec!r}")
        excludes[name] = reason

    r = screen(excludes=excludes)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return EXIT_CODE[r["verdict"]]

    print(f"원천 조각 {r['n_source_shingles']:,}개 ({SHINGLE}자) "
          f"/ 원천 문서 {len(r['source_docs_present'])}개")
    if r.get("warning"):
        print(f"!! {r['warning']}")
    for s in r["sources"]:
        if not s["exists"]:
            print(f"✕ 원천 {s['path']}  이 트리에 없다 — 재배포 판정 못 함")
        elif not s["n_shingles"]:
            print(f"✕ 원천 {s['path']}  비교 조각 0개(비었거나 정규화 뒤 {SHINGLE}자 미만)"
                  " — 재배포 판정 못 함")
    for t in r["targets"]:
        if t["excluded"]:
            print(f"- {t['path']}  제외 — {t['exclude_reason']}")
        elif not t["exists"]:
            print(f"✕ {t['path']}  이 트리에 없다 — 검사 못 함")
        elif t["missing_members"]:
            print(f"✕ {t['path']}  구성원 {len(t['missing_members'])}개 없다 — 검사 못 함")
    for f in r["files"]:
        mark = "✕" if f["hits"] else "○"
        print(f"{mark} {f['path']}  문자열 {f['n_texts']:,}개  적발 {len(f['hits'])}건")
        for h in f["hits"][:5]:
            print(f"    [{h['reason']}] {h['where']}: {h['evidence']!r}")
        if len(f["hits"]) > 5:
            print(f"    … 외 {len(f['hits']) - 5}건")
    print(f"\n검사한 파일 {r['n_files_examined']}개 · 문자열 {r['n_texts_examined']:,}개"
          f"  (대상 {r['n_targets']}곳 · 없음 {r['n_targets_missing']} · 제외 {r['n_targets_excluded']}"
          f" · 구성원 없음 {r['n_members_missing']})")
    print(f"판정: {r['verdict']} — {r['verdict_note']}  (차단 사유 {r['n_blocking']}건)")
    if r["verdict"] == "blocked":
        print("전환하지 마라 — 적발 내용을 보고한다 (09-08 판정).", file=sys.stderr)
    if r["verdict"] == "unverified":
        for u in r["unverified_reasons"]:
            print(f"    [{u['reason']}] {u['what']}", file=sys.stderr)
        print("검사를 못 했다 — 이 상태로 게시하지 마라. 위험이 없다는 뜻이 아니다.", file=sys.stderr)
    return EXIT_CODE[r["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
