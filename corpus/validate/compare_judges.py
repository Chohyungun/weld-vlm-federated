"""검증기 두 개의 판정을 대조한다.

검증기를 바꾸면 통과율이 얼마나 흔들리는지가 본실험 전에 알아야 할 값이다.
같은 생성분(재생성 없음)에 서로 다른 검증기를 걸어 일치율과 방향을 본다.

이 스크립트는 인자가 하나도 없었고 `judge_agreement.json` 을 **봉인된 `cycle_pilot/` 안에**
썼다. 그 파일은 SNAPSHOT 구성원이라(`7e19ef0a…`), 한 번 돌리면 봉인이 조용히 깨진다.
지금은 읽을 사이클(`--dir`)과 쓸 자리(`--out`)를 따로 받고, 쓸 자리가 봉인돼 있으면 막는다.
봉인된 corpus 위에 새로 만든 분석은 새 자리에 둔다 (규약 1-6).

실행: uv run python -m corpus.validate.compare_judges
        --dir corpus/generate/cycle_pilot --out corpus/validate/judge_agreement_v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# 봉인 계약의 정본은 `data.frozen_guard` 다 — corpus 쪽 문안만 frozen_out 이 감싼다.
from corpus.generate.frozen_out import assert_not_frozen


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="대조할 사이클 산출 디렉터리 (읽기만 한다)")
    ap.add_argument("--out", default=None,
                    help="judge_agreement.json 을 쓸 디렉터리. 미지정 시 --dir 과 같은 자리 —"
                         " 그 자리가 봉인돼 있으면 막힌다")
    args = ap.parse_args()
    D = Path(args.dir)
    out_dir = Path(args.out) if args.out else D
    # 인자를 읽자마자 막는다. `judge_agreement.json` 은 v1 SNAPSHOT 의 구성원이다.
    assert_not_frozen(out_dir, what="--out 대상", flag="--out")

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
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "judge_agreement.json"
    # newline="" 로 열어 win32 CRLF 자동 변환을 막는다 (.gitattributes eol=lf)
    with dst.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("산출:", dst)


if __name__ == "__main__":
    main()
