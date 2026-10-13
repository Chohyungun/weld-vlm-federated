"""무내용 대조선(idq512, H 구성)을 대표 지표 후보의 이미지 축 둘에서 채점한다 — 25번 보고의 대표 지표 결정표.

등록 규칙을 그대로 쓴다(`evaluation.content_free.fit_idq`, 족 `idq512`, 적합 train+val, 평가셋은 채점에만). 새로 고르는 것이 없다.
H 구성의 Macro-F1 이 등록 재구성값 0.9128 과 같은지를 자기 검사로 낸다(`configs/base.yaml` 의
`self_check_macro_f1_reconstructed`).

    python -X utf8 -B -m scripts.probe.content_free_image_axes --out <추적 밖 경로>/content_free_image_axes.json

출력: 집계만(id 없음) — 이미지 수준 결함 유무 F1 · Class-Jaccard(빈 집합끼리 1) · Macro-F1(자기 검사).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "data/interim/manifest_v1"
CLASSES = ("100", "2011", "301", "401")
ALT = {"2012": "2011"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    from evaluation.content_free import FIT_SPLITS, fit_idq, predict_codes
    from evaluation.eval_set import read_gold, read_manifest
    from evaluation.metrics.detection import score_detection

    rows = read_manifest(SNAP)
    fit_ids = [r["image_id"] for r in rows if r["split"] in FIT_SPLITS]
    eval_ids = [r["image_id"] for r in rows if r["split"] == "eval"]
    fit_codes, _ = read_gold(SNAP, set(fit_ids))
    ev_codes, _ = read_gold(SNAP, set(eval_ids))
    norm = lambda cs: {ALT.get(c, c) for c in cs} & set(CLASSES)  # noqa: E731
    fit_gold = {i: norm(fit_codes.get(i, ())) for i in fit_ids}
    gold = {i: norm(ev_codes.get(i, ())) for i in eval_ids}
    rule = fit_idq(512, fit_ids, fit_gold, CLASSES, snapshot=SNAP)
    pred = {i: set(v) for i, v in predict_codes(rule, eval_ids, snapshot=SNAP).items()}
    det = score_detection({i: sorted(v) for i, v in pred.items()}, {i: sorted(v) for i, v in gold.items()}, CLASSES)
    tp = sum(1 for i in eval_ids if pred[i] and gold[i])
    fp = sum(1 for i in eval_ids if pred[i] and not gold[i])
    fn = sum(1 for i in eval_ids if gold[i] and not pred[i])
    cj = sum(1.0 if not (pred[i] | gold[i]) else len(pred[i] & gold[i]) / len(pred[i] | gold[i]) for i in eval_ids) / len(eval_ids)
    empty_cj = sum(1.0 for i in eval_ids if not gold[i]) / len(eval_ids)
    out = {"rule": "idq512 H (fit train+val, score eval12461)", "n_eval": len(eval_ids),
           "macro_f1_self_check": det.macro_f1, "registered_reconstructed": 0.9128,
           "image_presence_f1": 2 * tp / (2 * tp + fp + fn), "class_jaccard": cj,
           "class_jaccard_of_all_empty_prediction": empty_cj}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
