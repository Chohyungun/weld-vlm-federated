"""공개 저장소 전환 전 전수 검수 — 09-08 총괄 판정 (24번 부록 A-4 ①).

`corpus/generate/cycle_pilot*/` 의 미추적 산출물을 git 추적으로 돌리기 전에, 그 내용이
**공개 저장소로 나가도 되는지**를 한 건도 빠짐없이 본다. `origin` 은 공개 GitHub
(`Chohyungun/weld-vlm-federated`)이고 해외 호스팅이다.

## 무엇을 보는가

총괄이 지목한 것은 `PAID_STD` 스크린이다. 그런데 저장소 자체 정책이 그보다 넓다 —
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

**한 건이라도 걸리면 전환하지 않는다** (총괄 지시). 종료코드 1.

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
LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|Fedvlm|Program Files)|/(?:home|Users)/[A-Za-z]")

#: 차단 사유. 하나라도 나오면 전환하지 않는다.
BLOCKING = ("paid_std", "verbatim_source", "aihub_id", "local_path")


def normalize(text: str) -> str:
    """공백만 무너뜨린다. 표 조판이 달라도 같은 문장은 같은 문장이다."""
    return re.sub(r"\s+", " ", text).strip()


def source_shingles(docs=None, n: int = SHINGLE) -> set[str]:
    """원천 문서의 n자 연속 조각. 없으면 빈 집합 — 그 경우 판정에 남긴다.

    기본값을 인자에 박지 않는다 — 그러면 import 시점에 묶여서 대상을 바꿀 수 없고,
    검수기를 검수하는 시험이 실제 대상을 건드리게 된다.
    """
    out: set[str] = set()
    for p in (SOURCE_DOCS if docs is None else docs):
        if not p.is_file():
            continue
        t = normalize(p.read_text(encoding="utf-8", errors="replace"))
        out.update(t[i:i + n] for i in range(len(t) - n + 1))
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


def screen(dirs=None) -> dict:
    dirs = TARGET_DIRS if dirs is None else dirs
    shingles = source_shingles()
    report: dict = {
        "shingle_len": SHINGLE,
        "n_source_shingles": len(shingles),
        "source_docs_present": [_shown(p) for p in SOURCE_DOCS if p.is_file()],
        "files": [],
    }
    if not shingles:
        # 원천이 없으면 재배포 판정을 못 한 것이다. 통과로 적지 않는다.
        report["warning"] = ("원천 문서가 워크트리에 없어 verbatim_source 를 판정하지 못했다"
                             " — 이 상태의 통과는 근거가 아니다")
    for d in dirs:
        if not d.is_dir():
            continue
        for name in untracked_members(d):
            p = d / name
            entry = {"path": _shown(p), "exists": p.is_file(),
                     "n_texts": 0, "hits": []}
            if p.is_file():
                for where, text in iter_texts(p):
                    entry["n_texts"] += 1
                    for reason, evidence in screen_text(text, shingles):
                        entry["hits"].append({"reason": reason, "where": where,
                                              "evidence": evidence[:120]})
            report["files"].append(entry)
    report["n_hits"] = sum(len(f["hits"]) for f in report["files"])
    report["n_blocking"] = sum(1 for f in report["files"] for h in f["hits"]
                               if h["reason"] in BLOCKING)
    report["verdict"] = "blocked" if report["n_blocking"] else "clear"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="공개 저장소 전환 전 전수 검수")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    r = screen()
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return 1 if r["verdict"] == "blocked" else 0

    print(f"원천 조각 {r['n_source_shingles']:,}개 ({SHINGLE}자) "
          f"/ 원천 문서 {len(r['source_docs_present'])}개")
    if r.get("warning"):
        print(f"!! {r['warning']}")
    for f in r["files"]:
        mark = "✕" if f["hits"] else "○"
        print(f"{mark} {f['path']}  문자열 {f['n_texts']:,}개  적발 {len(f['hits'])}건")
        for h in f["hits"][:5]:
            print(f"    [{h['reason']}] {h['where']}: {h['evidence']!r}")
        if len(f["hits"]) > 5:
            print(f"    … 외 {len(f['hits']) - 5}건")
    print(f"\n판정: {r['verdict']}  (차단 사유 {r['n_blocking']}건)")
    if r["verdict"] == "blocked":
        print("전환하지 마라 — 총괄에 보고한다 (09-08 판정).", file=sys.stderr)
    return 1 if r["verdict"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
