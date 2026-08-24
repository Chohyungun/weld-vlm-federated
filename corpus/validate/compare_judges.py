"""검증기 두 개의 판정을 대조한다.

검증기를 바꾸면 통과율이 얼마나 흔들리는지가 본실험 전에 알아야 할 값이다.
같은 생성분(재생성 없음)에 서로 다른 검증기를 걸어 일치율과 방향을 본다.

실행: uv run python -m corpus.validate.compare_judges
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "corpus/generate/cycle_pilot"


def load(jsonl: Path) -> dict[str, dict]:
    if not jsonl.exists():
        return {}
    out = {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["sample_id"]] = r
    return out


def verdicts(acc: Path, dis: Path) -> dict[str, bool]:
    """채택본과 폐기본을 합쳐 sample_id → 통과 여부."""
    v: dict[str, bool] = {}
    for sid in load(acc):
        v[sid] = True
    for sid, r in load(dis).items():
        if "judge_pass" in r:          # 0단계 폐기가 아니라 검증 단계 폐기만
            v[sid] = bool(r["judge_pass"])
    return v


def main() -> None:
    phi = verdicts(D / "_phi_reasoning.jsonl", D / "_phi_discarded.jsonl")
    new = verdicts(D / "reasoning_accepted.jsonl", D / "discarded.jsonl")
    common = sorted(set(phi) & set(new))

    agree = [s for s in common if phi[s] == new[s]]
    only_phi = [s for s in common if phi[s] and not new[s]]
    only_new = [s for s in common if new[s] and not phi[s]]

    phi_rep = json.loads((D / "_phi_report.json").read_text(encoding="utf-8"))
    new_rep = json.loads((D / "cycle_corpus_report.json").read_text(encoding="utf-8"))

    # 축별 불일치 — 어떤 축에서 흔들리는지 본다
    axis = {}
    for sid in common:
        a = "조치서술" if sid.startswith("remedy") else "조항검색_기준서술"
        cell = axis.setdefault(a, {"n": 0, "agree": 0})
        cell["n"] += 1
        cell["agree"] += int(phi[sid] == new[sid])
    for a, c in axis.items():
        c["agreement"] = round(c["agree"] / c["n"], 4) if c["n"] else None

    out = {
        "대조 대상": {
            "A": {"model": phi_rep["validation"]["model"],
                  "n_pass": phi_rep["reasoning"]["stage2_judge"]["n_pass"],
                  "pass_rate": phi_rep["reasoning"]["stage2_judge"]["pass_rate"]},
            "B": {"model": new_rep["validation"]["model"],
                  "n_pass": new_rep["reasoning"]["stage2_judge"]["n_pass"],
                  "pass_rate": new_rep["reasoning"]["stage2_judge"]["pass_rate"]},
        },
        "n_common": len(common),
        "agreement": round(len(agree) / len(common), 4) if common else None,
        "disagreement": {
            "A만 통과": len(only_phi),
            "B만 통과": len(only_new),
        },
        "pass_rate_shift_pp": round(
            (new_rep["reasoning"]["stage2_judge"]["pass_rate"]
             - phi_rep["reasoning"]["stage2_judge"]["pass_rate"]) * 100, 2),
        "axis_agreement": axis,
        "note": ("같은 생성분에 검증기만 바꿔 걸었다. 생성은 재실행하지 않았다. "
                 "검증기 교체는 실패분 재생성 금지 규칙과 별개다."),
    }
    (D / "judge_agreement.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
