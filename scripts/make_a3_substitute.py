"""A3 대체 승인서를 짓는다(D7, 미니스펙 `_workspace/2026-10-31-main2/MINISPEC_a3_substitute.md` §6).

증거 넷을 한 폴더에 복사하고(있으면 같은 바이트일 때만 둔다), 승인서 객체를 지금 HEAD 로 지은 뒤 **같은 검증 함수**
(`evaluation.ecc_substitute.verify_substitute`)로 0 건을 확인하고서야 승인서를 배타 생성한다. 덮지 않는다.

    .venv/Scripts/python.exe -X utf8 -B scripts/make_a3_substitute.py --decision "<사용자 결정 문구>" \
        --decision-log-section "<설계 결정 기록 절 제목>" --decided-at 2026-10-31T09:00:00+09:00 [--out-dir outputs/main2/ecc_substitute]

증거의 기본 경로는 A3 재시도 회수본 · 양식 · 로컬 묶음 탐침이다(인자로 바꿀 수 있다). 설계 결정 기록 절은 HEAD 에 커밋돼 있어야 한다.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import ecc_substitute as S

A3_COMMIT = "0656cdb690c7c39101bab0257196a4f74fa826d9"
EVIDENCE = {
    "a3_check": "outputs/recovery_probe/colab_a2a3_20261027r2_a3_01/A3_CHECK_258.json",
    "a3_result": "outputs/recovery_probe/colab_a2a3_20261027r2_a3_01/extracted/return_a3/a3-dad1769a-10281334/result.json",
    "a3_gate_form": "_workspace/cto/colab/a2a3-20261027-r2/A3_GATE_FORM_a3-dad1769a-r2.md",
    "batch_equivalence": "outputs/probe_batchgen_20261029/batch_equivalence.json",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decision", required=True)
    ap.add_argument("--decision-log-section", required=True)
    ap.add_argument("--decided-at", required=True)
    ap.add_argument("--a3-commit", default=A3_COMMIT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/main2/ecc_substitute")
    for k, v in EVIDENCE.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=Path, default=ROOT / v)
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    approval = out / "ecc_a3_substitute.json"
    if approval.exists():
        print(f"승인서가 이미 있다 — 덮지 않는다: {approval}", file=sys.stderr)
        return 2
    src = {k: Path(getattr(a, k)) for k in S.EVIDENCE_KINDS}
    missing = [str(p) for p in src.values() if not p.is_file()]
    if missing:
        print(f"증거 파일이 없다: {missing}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    placed = {}
    for k, p in src.items():
        dst = out / p.name
        if dst.exists():
            if dst.read_bytes() != p.read_bytes():
                print(f"승인서 폴더에 같은 이름의 다른 파일이 있다 — 덮지 않는다: {dst}", file=sys.stderr)
                return 2
        else:
            shutil.copyfile(p, dst)
        placed[k] = dst
    doc = S.build_approval(repo=ROOT, rev="HEAD", a3_commit=a.a3_commit, evidence=placed, decision=a.decision,
                           decision_log_section=a.decision_log_section, decided_at=a.decided_at)
    raw = S.dump_approval(doc)
    try:
        summary = S.verify_substitute(raw, evidence_dir=out, repo=ROOT, rev="HEAD")
    except S.SubstituteRefused as exc:
        print(f"[{exc.code}] 승인서를 쓰지 않는다 — " + " · ".join(exc.problems), file=sys.stderr)
        return 2
    with open(approval, "xb") as fh:
        fh.write(raw)
    print(f"wrote {approval} sha256 {hashlib.sha256(raw).hexdigest()}")
    print(f"a3_values {summary['a3_values']} · batch_size {summary['batch_size']} · evidence {[e['kind'] for e in summary['evidence']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
