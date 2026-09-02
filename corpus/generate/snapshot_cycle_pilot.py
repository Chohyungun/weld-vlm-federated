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

#: 이 스크립트가 만드는 파생물. 스냅샷 대상에서 뺀다 (자기 해시를 자기가 담을 수 없다).
DERIVED = (SNAPSHOT, EVIDENCE, SUMMARY)

#: 스냅샷에 반드시 있어야 하는 것. 없으면 사이클이 끝나지 않은 것이다.
REQUIRED = ("cycle_corpus_report.json", "discarded.jsonl", "qa_accepted.jsonl",
            "reasoning_accepted.jsonl")


def members(out_dir: Path) -> tuple[str, ...]:
    """스냅샷 대상 파일. **디렉터리에서 찾아 정렬한다.**

    고정 목록으로 두면 실행기가 산출을 하나 더 내도(조치 축 `remedy_accepted.jsonl`)
    스냅샷이 그것을 덮지 않고, 그 사실이 조용히 지나간다. 정렬 순서는 결정론이므로
    결합 다이제스트는 여전히 재현된다.
    """
    return tuple(sorted(p.name for p in out_dir.iterdir()
                        if p.is_file() and p.name not in DERIVED
                        and p.suffix in (".json", ".jsonl")))

# 항목 축약본을 뜨는 파일과 그 배출 단계. 판정기 두 벌(deepseek 정본 · phi 대조)을
# 구분해 담는다 — judge_agreement.json 의 일치도가 이 둘의 대조값이다.
ITEM_FILES = (
    ("reasoning_accepted.jsonl", "deepseek", "accepted"),
    ("remedy_accepted.jsonl", "-", "accepted"),
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
                if "stage1_pass" in r:
                    rec["stage1_pass"] = r["stage1_pass"]
                rec["clause_id"] = sk.get("clause_id")
                rec["rule_id"] = sk.get("rule_id")
                rec["judge_pass"] = r.get("judge_pass")
                rec["judge_parse_ok"] = r.get("judge_parse_ok")
                # 재실행분은 후보별 키(`judge_<id>_*`)를 쓴다 — 후보끼리 덮어쓰지 않아야
                # 같은 파일에서 일치도를 잰다. 축약본도 그대로 담는다.
                for k, v in r.items():
                    if k.startswith("judge_") and k.endswith(("_pass", "_parse_ok",
                                                              "_reason_is_echo")):
                        rec[k] = v
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


def recompute_axes(items: list[dict]) -> dict:
    """재실행분 축약본 → 축별 단계·후보별 통과 수. 원본 jsonl 없이 성립해야 한다."""
    out: dict = {}
    for axis in sorted({r.get("axis") for r in items if r.get("axis")}):
        rs = [r for r in items if r.get("axis") == axis]
        s0 = [r for r in rs if r["stage0_pass"]]
        s1 = [r for r in s0 if r.get("stage1_pass")]
        stages = {"stage0_numeric_lock": {"n_in": len(rs), "n_pass": len(s0)}}
        if any("stage1_pass" in r for r in rs):
            stages["stage1_rule"] = {"n_in": len(s0), "n_pass": len(s1)}
        judges: dict[str, dict] = {}
        cids = {k[len("judge_"):-len("_pass")] for r in rs for k in r
                if k.startswith("judge_") and k.endswith("_pass")}
        for cid in sorted(c for c in cids if c):   # `judge_pass`(v1 키)는 후보가 아니다
            judged = [r for r in s1 if r.get(f"judge_{cid}_pass") is not None]
            j_pass = [r for r in judged if r[f"judge_{cid}_pass"]]
            judges[cid] = {
                "n_in": len(judged), "n_pass": len(j_pass),
                "n_fail": len(judged) - len(j_pass),
                "pass_rate": round(len(j_pass) / len(judged), 4) if judged else None,
                "n_format_violation": sum(1 for r in judged
                                          if r.get(f"judge_{cid}_parse_ok") is False),
                "n_reason_is_echo": sum(1 for r in judged
                                        if r.get(f"judge_{cid}_reason_is_echo")),
            }
        out[f"axis:{axis}"] = {"n_in": len(rs), "stages": stages, "judges": judges}

    qa = [r for r in items if r["kind"] == "qa"]
    seen: dict[str, dict] = {}
    for r in qa:
        seen.setdefault(f"{r['stage']}|{r['sample_id']}", r)
    uniq = list(seen.values())
    acc = [r for r in uniq if r["stage"] == "accepted"]
    out["axis:QA"] = {"n_in": len(uniq),
                      "stages": {"stage1_rule": {"n_in": len(uniq), "n_pass": len(acc)}},
                      "judges": {}}
    out["qa"] = {"n_in": len(uniq), "n_accepted": len(acc),
                 "pass_rate": round(len(acc) / len(uniq), 4) if uniq else None}
    return out

def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def crosscheck(recomputed: dict, out_dir: Path) -> list[str]:
    """보고서에 실린 수치와 축약본 재계산값을 대조한다. 어긋나면 사유 문자열을 낸다.

    보고 스키마가 두 벌이다 — v1 은 `reasoning`/`qa` 평면 구조, 재실행분은 `axes` 아래
    축별 구조(G6-1: 축마다 검증 수준을 말한다). **어느 쪽인지 자동으로 가른다.**
    스키마를 못 알아보고 조용히 통과하면 스냅샷이 틀린 수치를 고정한다.
    """
    rep_path = out_dir / "cycle_corpus_report.json"
    if not rep_path.exists():
        return ["cycle_corpus_report.json: 없음"]
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    if "axes" in rep:
        return _crosscheck_axes(recomputed, rep)
    return _crosscheck_flat(recomputed, out_dir)


def _crosscheck_axes(recomputed: dict, rep: dict) -> list[str]:
    """재실행분 스키마 — 축별 stages + 후보별 stage2."""
    problems: list[str] = []
    for axis, block in rep["axes"].items():
        key = f"axis:{axis}"
        got = recomputed.get(key)
        if got is None:
            problems.append(f"{axis}: 축약본에 대응 축이 없다")
            continue
        if block["n_in"] != got["n_in"]:
            problems.append(f"{axis} n_in: 보고서 {block['n_in']} ≠ 재계산 {got['n_in']}")
        for name, st in block["stages"].items():
            if st.get("status") != "ran":
                continue
            cur = got["stages"].get(name)
            if cur is None:
                problems.append(f"{axis}.{name}: 축약본에 없다")
            elif st["n_pass"] != cur["n_pass"]:
                problems.append(
                    f"{axis}.{name} n_pass: 보고서 {st['n_pass']} ≠ 재계산 {cur['n_pass']}")
        for cid, cand in (block.get("stage2_judges") or {}).items():
            cur = got["judges"].get(cid)
            if cur is None:
                problems.append(f"{axis} 후보 {cid}: 축약본에 없다")
            elif cand["n_pass"] != cur["n_pass"]:
                problems.append(
                    f"{axis} 후보 {cid} n_pass: 보고서 {cand['n_pass']} ≠ 재계산 {cur['n_pass']}")
    return problems


def _crosscheck_flat(recomputed: dict, out_dir: Path) -> list[str]:
    """v1 스키마 — `reasoning`/`qa` 평면 구조. 판정기 두 벌이 보고서 두 개로 갈려 있다."""
    problems: list[str] = []
    checked = (
        ("stage0", "n_pass"), ("stage0", "n_fail"),
        ("stage2_judge", "n_pass"), ("stage2_judge", "n_fail"), ("n_accepted",),
    )
    for fname, key in (("cycle_corpus_report.json", "reasoning:deepseek"),
                       ("_phi_report.json", "reasoning:phi")):
        p = out_dir / fname
        if not p.exists() or key not in recomputed:
            problems.append(f"{fname}: 대조 대상 없음")
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        got = recomputed[key]
        for path in checked:
            want, cur = _dig(rep, ("reasoning",) + path), _dig(got, path)
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
    ap.add_argument("--dir", default=str(OUT_DIR), help="사이클 산출 디렉터리")
    args = ap.parse_args()
    out_dir = Path(args.dir)

    missing = [n for n in REQUIRED if not (out_dir / n).exists()]
    if missing:
        print(f"스냅샷 대상이 없다: {missing}", file=sys.stderr)
        print("원본은 드라이브 보관분이다 — 워크트리에 복원한 뒤 실행하라", file=sys.stderr)
        return 2

    entries = [(sha256_file(out_dir / n), n) for n in members(out_dir)]
    items = build_evidence(out_dir)
    new_schema = "axes" in json.loads(
        (out_dir / "cycle_corpus_report.json").read_text(encoding="utf-8"))
    recomputed = recompute_axes(items) if new_schema else recompute(items)
    problems = crosscheck(recomputed, out_dir)
    if problems:
        # 조용히 지나가면 스냅샷이 틀린 수치를 고정한다.
        print("보고서 대조 불일치:", *problems, sep="\n  ", file=sys.stderr)
        return 3

    rendered = {
        out_dir / SNAPSHOT: render_snapshot(entries),
        out_dir / EVIDENCE: render_evidence(items),
        out_dir / SUMMARY: render_summary(recomputed, entries, problems, len(items)),
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
        rate = v.get("end_to_end_rate", v.get("pass_rate"))
        if rate is None and "stages" in v:
            last = list(v["stages"].values())[-1]
            rate = (round(last["n_pass"] / v["n_in"], 4) if v["n_in"] else None)
        print(f"  {k}: 통과 {rate}"
              + (f" | 후보 {[(c, d['n_pass']) for c, d in v['judges'].items()]}"
                 if v.get("judges") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
