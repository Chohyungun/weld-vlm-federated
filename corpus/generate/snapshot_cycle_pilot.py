"""corpus 사이클 파일럿 산출물 스냅샷 + 근거 보존 — 규약 1-6 이행 (74번 감사 P5).

두 가지가 빠져 있었다.

1. `corpus/generate/cycle_pilot/` 산출물에 sha256 스냅샷이 없다. 규약 1-6 은 corpus·페어·
   색인에 sha256 을 부여하고 재생성을 금지한다.
2. 통과율 0.94(QA) · 0.345(판정추론)를 뒷받침하는 **항목 레코드가 `.gitignore` 로 추적
   제외**라 main 에 없다. 워크트리를 정리하면 근거가 소실된다.

용량 때문에 원본 jsonl 을 추적할 수는 없다(약 700KB, 생성문 전문 포함). 그래서 원본은
그대로 두고 **항목 단위 축약본**을 추적한다. 축약본은 항목마다 판정에 쓰인 플래그와
생성문의 sha256 을 담으므로, 나중에 원본이 나오면 한 건씩 대조할 수 있고 원본이
없어도 통과율은 축약본만으로 재계산된다.

산출 (전부 추적 대상):
  cycle_pilot/SNAPSHOT.sha256        파일별 sha256 + 결합 다이제스트 (원본 jsonl 포함)
  cycle_pilot/EVIDENCE.jsonl         항목 축약 레코드 (생성문은 sha256 만)
  cycle_pilot/EVIDENCE_SUMMARY.json  축약본에서 재계산한 통과율 + 보고서 대조 결과

재계산값이 보고서(`cycle_corpus_report.json`·`_phi_report.json`)와 어긋나면 **시끄럽게
실패한다.** 조용히 지나가면 스냅샷이 틀린 수치를 고정해 버린다.

실행: uv run python -m corpus.generate.snapshot_cycle_pilot [--check]
  --check 는 쓰지 않고 기존 스냅샷·축약본이 현재 산출물과 일치하는지만 본다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "corpus/generate/cycle_pilot"

SNAPSHOT = "SNAPSHOT.sha256"
EVIDENCE = "EVIDENCE.jsonl"
SUMMARY = "EVIDENCE_SUMMARY.json"

# 스냅샷 대상. 순서를 고정한다 — 결합 다이제스트가 순서에 의존한다.
MEMBERS = (
    "_raw_generated.json",
    "_phi_discarded.jsonl",
    "_phi_reasoning.jsonl",
    "_phi_report.json",
    "cycle_corpus_report.json",
    "discarded.jsonl",
    "judge_agreement.json",
    "qa_accepted.jsonl",
    "reasoning_accepted.jsonl",
)

# 항목 축약본을 뜨는 파일과 그 배출 단계. 판정기 두 벌(deepseek 정본 · phi 대조)을
# 구분해 담는다 — judge_agreement.json 의 일치도가 이 둘의 대조값이다.
ITEM_FILES = (
    ("reasoning_accepted.jsonl", "deepseek", "accepted"),
    ("discarded.jsonl", "deepseek", "discarded"),
    ("_phi_reasoning.jsonl", "phi", "accepted"),
    ("_phi_discarded.jsonl", "phi", "discarded"),
    ("qa_accepted.jsonl", "-", "accepted"),
)

REASON_CAP = 120   # 기각 사유는 분포 집계용이라 앞부분이면 충분하다


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def snapshot_digest(entries: list[tuple[str, str]]) -> str:
    return hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()


def render_snapshot(entries: list[tuple[str, str]]) -> str:
    body = "\n".join(f"{h}  {name}" for h, name in entries)
    return body + f"\n# snapshot_digest {snapshot_digest(entries)}\n"


def _kind(rec: dict) -> str:
    """판정추론(axis 축)과 QA(질문·답 축)를 가른다 — discarded.jsonl 이 둘을 섞어 담는다."""
    return "reasoning" if "axis" in rec else "qa"


def build_evidence(out_dir: Path) -> list[dict]:
    """항목 축약 레코드. 생성문 자체는 담지 않고 sha256 만 담는다."""
    items: list[dict] = []
    for fname, judge, stage in ITEM_FILES:
        p = out_dir / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            kind = _kind(r)
            rec = {
                "kind": kind,
                "judge": judge,
                "stage": stage,
                "source_file": fname,
                "sample_id": r.get("sample_id"),
                "stage0_pass": r.get("stage0_pass"),
                "stage0_reasons": r.get("stage0_reasons") or [],
                "text_sha256": sha256_bytes((r.get("text") or "").encode("utf-8")),
            }
            if kind == "reasoning":
                sk = r.get("skeleton") or {}
                rec["axis"] = r.get("axis")
                rec["clause_id"] = sk.get("clause_id")
                rec["rule_id"] = sk.get("rule_id")
                rec["judge_pass"] = r.get("judge_pass")
                rec["judge_parse_ok"] = r.get("judge_parse_ok")
                if stage == "discarded":
                    rec["judge_reason"] = (r.get("judge_reason") or "")[:REASON_CAP]
            else:
                rec["aspect"] = r.get("aspect")
                rec["doc"] = r.get("doc")
                rec["passage_id"] = r.get("passage_id")
            items.append(rec)
    items.sort(key=lambda r: (r["judge"], r["kind"], r["stage"], str(r["sample_id"])))
    return items


def recompute(items: list[dict]) -> dict:
    """축약본만으로 통과율을 재계산한다. 원본 jsonl 없이도 성립해야 한다."""
    out: dict = {}
    for judge in sorted({r["judge"] for r in items if r["kind"] == "reasoning"}):
        rs = [r for r in items if r["kind"] == "reasoning" and r["judge"] == judge]
        s0_pass = [r for r in rs if r["stage0_pass"]]
        s0_fail = [r for r in rs if not r["stage0_pass"]]
        judged = [r for r in s0_pass if r["judge_pass"] is not None]
        j_pass = [r for r in judged if r["judge_pass"]]
        reasons: Counter = Counter()
        for r in s0_fail:
            reasons.update(r["stage0_reasons"])
        out[f"reasoning:{judge}"] = {
            "n_in": len(rs),
            "stage0": {
                "n_pass": len(s0_pass),
                "n_fail": len(s0_fail),
                "pass_rate": round(len(s0_pass) / len(rs), 4) if rs else None,
                "fail_reasons": dict(sorted(reasons.items())),
            },
            "stage2_judge": {
                "n_in": len(judged),
                "n_pass": len(j_pass),
                "n_fail": len(judged) - len(j_pass),
                "pass_rate": round(len(j_pass) / len(judged), 4) if judged else None,
                "n_format_violation": sum(1 for r in judged
                                          if r.get("judge_parse_ok") is False),
            },
            "n_accepted": len(j_pass),
            "end_to_end_rate": round(len(j_pass) / len(rs), 4) if rs else None,
            "axis_split": dict(sorted(Counter(r["axis"] for r in rs).items())),
        }

    qa = [r for r in items if r["kind"] == "qa"]
    # QA 는 판정기를 거치지 않는다. 두 판정기 실행에 같은 QA 폐기분이 각각 실려 있어
    # (단계, sample_id) 로 중복을 제거한다.
    seen: dict[str, dict] = {}
    for r in qa:
        seen.setdefault(f"{r['stage']}|{r['sample_id']}", r)
    uniq = list(seen.values())
    acc = [r for r in uniq if r["stage"] == "accepted"]
    dis = [r for r in uniq if r["stage"] == "discarded"]
    qa_reasons: Counter = Counter()
    for r in dis:
        qa_reasons.update(r["stage0_reasons"])
    out["qa"] = {
        "n_in": len(uniq),
        "n_accepted": len(acc),
        "n_discarded": len(dis),
        "pass_rate": round(len(acc) / len(uniq), 4) if uniq else None,
        "fail_reasons": dict(sorted(qa_reasons.items())),
    }
    return out


def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def crosscheck(recomputed: dict, out_dir: Path) -> list[str]:
    """보고서에 실린 수치와 축약본 재계산값을 대조한다. 어긋나면 사유 문자열을 낸다."""
    problems: list[str] = []
    pairs = (
        ("cycle_corpus_report.json", "reasoning:deepseek"),
        ("_phi_report.json", "reasoning:phi"),
    )
    checked = (
        ("stage0", "n_pass"),
        ("stage0", "n_fail"),
        ("stage2_judge", "n_pass"),
        ("stage2_judge", "n_fail"),
        ("n_accepted",),
    )
    for fname, key in pairs:
        p = out_dir / fname
        if not p.exists() or key not in recomputed:
            problems.append(f"{fname}: 대조 대상 없음")
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        got = recomputed[key]
        for path in checked:
            want = _dig(rep, ("reasoning",) + path)
            cur = _dig(got, path)
            if want != cur:
                problems.append(
                    f"{fname} reasoning.{'.'.join(path)}: 보고서 {want} ≠ 재계산 {cur}")
        want_qa = _dig(rep, ("qa", "n_accepted"))
        if want_qa != recomputed["qa"]["n_accepted"]:
            problems.append(f"{fname} qa.n_accepted: 보고서 {want_qa}"
                            f" ≠ 재계산 {recomputed['qa']['n_accepted']}")
    return problems


def render_evidence(items: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True)
                     for r in items) + "\n"


def render_summary(recomputed: dict, entries: list[tuple[str, str]],
                   problems: list[str], n_items: int) -> str:
    doc = {
        "_meta": {
            "purpose": "cycle_pilot 통과율의 근거 보존 (규약 1-6 · 74번 감사 P5)",
            "note": "원본 jsonl 은 용량 때문에 미추적. 항목 축약본 EVIDENCE.jsonl 과"
                    " 이 요약이 근거다. 원본이 있으면 SNAPSHOT.sha256 으로 대조한다.",
            "regenerate": "uv run python -m corpus.generate.snapshot_cycle_pilot",
            "n_evidence_items": n_items,
            "snapshot_digest": snapshot_digest(entries),
        },
        "recomputed": recomputed,
        "crosscheck_vs_reports": "일치" if not problems else problems,
    }
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="쓰지 않고 기존 스냅샷·축약본과 현재 산출물의 일치만 검사")
    args = ap.parse_args()

    missing = [n for n in MEMBERS if not (OUT_DIR / n).exists()]
    if missing:
        print(f"스냅샷 대상이 없다: {missing}", file=sys.stderr)
        print("원본은 드라이브 보관분이다 — 워크트리에 복원한 뒤 실행하라", file=sys.stderr)
        return 2

    entries = [(sha256_file(OUT_DIR / n), n) for n in MEMBERS]
    items = build_evidence(OUT_DIR)
    recomputed = recompute(items)
    problems = crosscheck(recomputed, OUT_DIR)
    if problems:
        # 조용히 지나가면 스냅샷이 틀린 수치를 고정한다.
        print("보고서 대조 불일치:", *problems, sep="\n  ", file=sys.stderr)
        return 3

    rendered = {
        OUT_DIR / SNAPSHOT: render_snapshot(entries),
        OUT_DIR / EVIDENCE: render_evidence(items),
        OUT_DIR / SUMMARY: render_summary(recomputed, entries, problems, len(items)),
    }

    if args.check:
        stale = [p.name for p, c in rendered.items()
                 if not p.exists() or p.read_bytes() != c.encode("utf-8")]
        if stale:
            print("갱신 필요:", stale, file=sys.stderr)
            return 1
        print(f"스냅샷·축약본이 현재 산출물과 일치한다 (항목 {len(items)}건)")
        return 0

    for path, content in rendered.items():
        # newline="" 로 열어 win32 CRLF 자동 변환을 막는다 (.gitattributes eol=lf)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(content)

    print(f"스냅샷 {len(entries)}개 파일 / 축약 {len(items)}건")
    print(f"snapshot_digest {snapshot_digest(entries)}")
    for k, v in recomputed.items():
        print(f"  {k}: 통과율 {v.get('end_to_end_rate', v.get('pass_rate'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
