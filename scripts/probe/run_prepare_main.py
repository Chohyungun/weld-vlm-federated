"""실데이터 작업 명부 — `corpus.production.prepare.prepare_run` 을 직접 부른다(CLI 인자 반영 전, 지시 A 1008f).

CPU 만 쓴다. GPU · 생성 · 서버 반입은 하지 않는다. 경로는 모두 인자로 받는다(저장소에 로컬 경로를 남기지 않는다).
`--repo-root` 는 본 체크아웃이다 — prepare 의 게이트 검사가 그 뿌리의 `.git`(디렉터리)에서 본체 락을 보고, 매니페스트 · 영상의
상대 경로도 그 뿌리에서 푼다. 끝나면 출력 뿌리에 묶음 지문(`<run_id>_bundle.sha256`)과 요약(`<run_id>_summary.json`, 수와 지문만 —
id 없음)을 쓴다. 서버로 보낼 묶음은 `<run_id>/tasks.jsonl` · `<run_id>/RUN.json` · `<run_id>_config.json`(config 사본)이다.

    python -X utf8 -B -m scripts.probe.run_prepare_main --repo-root <본 체크아웃> --pairs data/processed/pairs_main_v1/pairs.jsonl \\
        --bank corpus/parse/passage_bank_v3.jsonl --bank-text <본문 은행> --config configs/corpus_production.json \\
        --out-root outputs/corpus_production --run-id <이름>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-root", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path, help="포인터 은행(추적)")
    ap.add_argument("--bank-text", type=Path, default=None, help="본문 은행(미추적)")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--rights", type=Path, default=None)
    ap.add_argument("--review", type=Path, default=None)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)

    from corpus.production.common import read_json
    from corpus.production.prepare import PrepareRefused, prepare_run

    config = read_json(args.config)
    run_dir = args.out_root / args.run_id
    started = time.monotonic()
    try:
        prepare_run(args.pairs, args.bank, config, run_dir, source_text_path=args.bank_text, source_registry_path=args.registry,
                    source_rights_path=args.rights, source_review_path=args.review, repo_root=args.repo_root)
    except PrepareRefused as exc:
        print(f"거부: {exc.code}", file=sys.stderr)
        return exc.exit_code
    elapsed = time.monotonic() - started

    config_copy = args.out_root / f"{args.run_id}_config.json"
    shutil.copyfile(args.config, config_copy)
    bundle = {"tasks.jsonl": _sha(run_dir / "tasks.jsonl"), "RUN.json": _sha(run_dir / "RUN.json"),
              config_copy.name: _sha(config_copy)}
    (args.out_root / f"{args.run_id}_bundle.sha256").write_text(
        "".join(f"{v}  {k}\n" for k, v in bundle.items()), encoding="utf-8", newline="\n")
    run = read_json(run_dir / "RUN.json")
    summary = {"run_id": args.run_id, "elapsed_seconds": round(elapsed, 1), "bundle_sha256": bundle,
               "identity_sha256": run["identity_sha256"], "config_sha256": run["config_sha256"],
               "tasks_sha256": run["tasks_sha256"], "identity_basis": run["identity_basis"], "counts": run["counts"]}
    (args.out_root / f"{args.run_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("run_id", "elapsed_seconds", "identity_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
