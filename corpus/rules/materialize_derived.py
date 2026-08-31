"""corpus/derived/ 파생물 실체화 — 재실행 가능한 재파생 스크립트.

`limits.csv` 한 파일에서 세 소비처가 파생된다(개발규약 1-7). 이 스크립트는 그중
D 소비분 둘을 실체화한다:

  corpus/derived/chunk_meta.jsonl   RAG 청크 메타 필터 축 (rag/retrieve.py 의 Chunk)
  corpus/derived/gold_clauses.csv   채점 정답 조항 목록 (evaluation/gold.py)

규칙:
- 입력은 `limits_loader` 공식 로더 경유. G0 통과분만 파생된다.
- 파생물은 수기 수정 금지. 원천이 바뀌면 이 스크립트로 재파생한다.
  원천이 바뀌었는데 파생물이 낡은 채로 남는 것이 최악이라, 산출물마다 원천 CSV 의
  sha256 을 박는다. 소비처는 로드 시 이 해시를 자기가 본 원천과 대조할 수 있다.
- 산출은 결정론이다. 같은 원천이면 같은 바이트가 나온다(타임스탬프를 넣지 않는 이유).
- gold_clauses.csv 는 평가 자산(D6)이다. 채점기·검색 결정 절차 외 어떤 학습 단계에도
  투입하지 않는다(개발규약 1-4). 파일 헤더 주석에 같은 경고를 박는다.

파일 형식 규약 (소비처와 공유):
- chunk_meta.jsonl: 첫 줄은 `{"_meta": {...}}` 헤더 레코드(원천 경로·sha256·건수·pilot).
  데이터 행은 둘째 줄부터이며 `derive_chunk_meta()` 항목 그대로다. 소비처는 `_meta` 키가
  있는 행을 건너뛴다.
- gold_clauses.csv: `#` 로 시작하는 줄은 주석이다. pandas 는 `comment="#"` 로 읽는다.

실행: uv run python -m corpus.rules.materialize_derived [--csv 경로] [--check]
  --check 는 파일을 쓰지 않고 기존 파생물이 현재 원천과 일치하는지만 검사한다.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"
OUT_DIR = REPO / "corpus/derived"

GOLD_WARNING = (
    "gold_clauses.csv 는 평가 자산(D6)이다. 채점기와 검색(임베딩) 결정 절차 외 어떤"
    " 학습 단계에도 투입하지 않는다. 누출은 논문 철회 사유다(개발규약 1-4)."
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def render_chunk_meta(chunks, src: Path, src_sha: str, pilot: bool) -> str:
    head = {"_meta": {
        "source_csv": src.name,
        "source_sha256": src_sha,
        "n_chunks": len(chunks),
        "pilot": pilot,
        "regenerate": "uv run python -m corpus.rules.materialize_derived",
        "note": "파생물 — 수기 수정 금지. 원천이 바뀌면 재파생한다.",
    }}
    lines = [json.dumps(head, ensure_ascii=False)]
    lines += [json.dumps(c, ensure_ascii=False, default=str) for c in chunks]
    return "\n".join(lines) + "\n"


def render_gold(rows, src: Path, src_sha: str, pilot: bool) -> str:
    import csv as _csv
    buf = io.StringIO()
    buf.write(f"# {GOLD_WARNING}\n")
    buf.write(f"# 원천: {src.name} sha256={src_sha}\n")
    buf.write(f"# pilot={pilot} · 파생물 — 수기 수정 금지."
              f" 재파생: uv run python -m corpus.rules.materialize_derived\n")
    fields = list(rows[0].keys())
    w = _csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 기존 파생물과 현재 원천의 일치만 검사")
    args = ap.parse_args()

    src = Path(args.csv)
    # v0-pilot 은 sources.yaml·조항 목록이 없어 pilot 모드로만 로드된다. 본 CSV 로
    # 전환되면 파일명이 limits.csv 가 되고 그때는 전체 G0 를 요구한다.
    pilot = "pilot" in src.name
    src_sha = sha256_file(src)

    from corpus.rules import limit_eval, limits_loader
    table = limits_loader.load_limits(str(src), pilot=pilot)

    chunks = limit_eval.derive_chunk_meta(table)
    gold = limit_eval.derive_gold_clauses(table)
    if not chunks or not gold:
        print("파생 결과가 비었다 — 원천을 확인하라", file=sys.stderr)
        return 2

    rendered = {
        OUT_DIR / "chunk_meta.jsonl": render_chunk_meta(chunks, src, src_sha, pilot),
        OUT_DIR / "gold_clauses.csv": render_gold(gold, src, src_sha, pilot),
    }

    if args.check:
        stale = []
        for path, content in rendered.items():
            if not path.exists():
                stale.append(f"{path.name}: 없음")
            elif path.read_text(encoding="utf-8") != content:
                stale.append(f"{path.name}: 원천과 불일치 (재파생 필요)")
        if stale:
            print("낡은 파생물:", *stale, sep="\n  ")
            return 1
        print("파생물이 현재 원천과 일치한다.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    changed = []
    for path, content in rendered.items():
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != content:
            path.write_text(content, encoding="utf-8")
            changed.append(path.name)

    print(f"원천 {src.name} sha256={src_sha[:12]}…")
    print(f"청크 {len(chunks)}건 / gold 조항 {len(gold)}건"
          f" / 갱신 {changed if changed else '없음(동일)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
