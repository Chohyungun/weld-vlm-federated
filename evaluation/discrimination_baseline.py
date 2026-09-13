"""출처 고정 Δ_AUC에 병기할 ID 구간 빈도 대조선. 이미지는 읽지 않는다."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from data.label_map import load_label_map
from evaluation.content_free import FIT_SPLITS, FREQ, fit_idq, scored
from evaluation.discrimination import CROP, gini
from evaluation.eval_set import ensure_verified, parse_iso_codes, read_manifest
from evaluation.params import CLASS_NAMES
from evaluation.strata import ID_GRANULARITY


def compute_metadata_baseline(
    snapshot: str | Path,
    *,
    image_ids: Iterable[str] | None = None,
    classes: Sequence[str] | None = None,
    k: int = 512,
) -> dict:
    """기존 idq512 규칙을 train+val에 적합하고 지정 eval/N-crop에서 채점한다.

    기본 K=512와 점수 구성은 기존 대조선을 재사용한다. 다른 K는 등록된 사다리의
    민감도 진단에만 사용하며, 평가 결과로 가장 좋은 K를 고르지 않는다.
    ``image_ids``를 받으면 본채점과 같은 모집단인지 검증한다. 이 대조선은 2026-09-09
    감사에서 추가한 사후 진단이며, 기존 지표의 사전등록을 소급 변경하지 않는다.
    """
    if classes is None:
        label_map = load_label_map()
        classes = tuple(label_map.iso_code(name) for name in CLASS_NAMES)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes must be nonempty and unique")
    if k not in ID_GRANULARITY:
        raise ValueError("k must belong to the existing ID_GRANULARITY ladder")
    snapshot = Path(snapshot)
    rows = read_manifest(snapshot)
    with (snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        provenance = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}
    eligible = {
        r["image_id"]: r for r in rows
        if r["split"] == "eval" and provenance.get(r["image_id"]) == CROP
    }
    ids = set(eligible) if image_ids is None else set(image_ids)
    if ids - eligible.keys():
        raise ValueError("metadata baseline population must contain only eval/N-crop images")
    fit_rows = [r for r in rows if r["split"] in FIT_SPLITS]
    fit_ids = [r["image_id"] for r in fit_rows]
    fit_gold = {r["image_id"]: parse_iso_codes(r["iso_codes"]) for r in fit_rows}
    rule = fit_idq(k, fit_ids, fit_gold, classes, snapshot=snapshot)
    values, diagnostic = scored(rule, sorted(ids), FREQ, snapshot=snapshot)
    scores = {i: max(s for _, s in items) for i, items in values.items()}
    defect = sorted(i for i in ids if eligible[i]["has_defect"] == "True")
    normal = sorted(ids - set(defect))
    point = gini(scores, defect, normal)
    return {
        "name": f"idq{k}_max_class_frequency",
        "status": "post_hoc_audit_diagnostic_2026-09-09",
        "fit_population": "train+val",
        "n_fit": len(fit_ids),
        "classes": list(classes),
        "image_pixels_read": False,
        "provenance": CROP,
        "n_defect": len(defect),
        "n_normal": len(normal),
        "delta_auc": point if math.isfinite(point) else None,
        "snapshot_digest": ensure_verified(snapshot),
        "diagnostic": diagnostic,
        "interpretation": "메타데이터만으로 도달한 점추정이다. 상한이나 모델의 정보 사용에 대한 인과 증거가 아니다",
    }
