"""판별력 Δ 의 임계 의존성과 그 임계 독립판 — 승격 판단 재료 (22번 §5 과제 4, 17번 §12-5).

본채점이 산출물에 싣는 Δ 는 **운용점 한 점(conf 0.25)** 에서 계산된다. 발화 정의가
"예측 결함 집합이 비어 있지 않다" 이므로 임계를 낮추면 두 발화율이 함께 1 로 올라가고
차가 0 으로 눌린다. **Δ 를 헤드라인 보조지표로 승격하려면 그 눌림이 어디서 시작되는지,
칸 순위가 임계에 따라 뒤집히는지 알아야 한다** — 22번이 단일 임계 사전등록을 금지한 것과
같은 이유다.

이 스크립트가 내는 것은 둘이다.

1. **곡선** — 등록 격자 전 점에서 Δ 를 다시 낸 값과 그 점에서의 칸 순위.
   격자는 여기서 다시 쓰지 않고 `ScoringParams.conf_sweep` 으로 읽는다(하드코딩 금지).
2. **임계 독립판 `Δ_AUC = 2·AUROC − 1`** — 같은 질문을 임계 없이 묻는다. 지름길은
   여기서도 정의상 정확히 0 이다. 칸 대비는 **같은 재표집에서 짝지어** 낸다.

CI 는 격자 점마다 부트스트랩을 다시 돌리면 14×5×2000 회가 되므로, 곡선용으로는 점추정만
내고 CI 는 임계 독립판과 본채점 운용점(`score_cells_v1.json` `discrimination`)에서 낸다.

GPU 무접촉. 하한 레코드를 읽어 임계로 거를 뿐 추론을 다시 하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path

from evaluation.detect_infer import filter_by_conf
from evaluation.discrimination import CROP, fires, gini, image_score
from evaluation.params import add_common_args, params_from_args
from evaluation.schema import PredictionRecord
from evaluation.stats import cluster_bootstrap

OPERATING_CONF = 0.25
"""표에 함께 싣는 **운용점 예시**. 확증 기준이 아니다(22번 §1-2-2)."""


def _load_raw(path: Path) -> list[PredictionRecord]:
    with path.open(encoding="utf-8") as fh:
        return [PredictionRecord.model_validate_json(line) for line in fh if line.strip()]


def _filtered(recs: list[PredictionRecord], conf: float) -> dict[str, bool]:
    """임계 위 상자만 남기고 발화 여부를 낸다.

    자르는 것은 본채점과 **같은 함수**(`filter_by_conf`), 발화 판정도 본채점과 **같은 함수**
    (`discrimination.fires`)다. 어느 한쪽이라도 여기서 다시 쓰면 곡선이 정의 차이를 재게 된다.
    """
    return {r.image_id: fires(r) for r in filter_by_conf(recs, conf)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("--dest", default=None, help="저장 경로(기본 산출물 폴더 아래)")
    args = ap.parse_args()
    params = params_from_args(args)

    from scripts.probe.score_cells import DET_TAGS, load_population, raw_record_path

    pop = load_population(params)
    # 출처 표는 본채점과 **같은 파일**을 읽는다(진단 전체를 다시 돌리지 않는다).
    with (params.snapshot / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}

    ctx = {
        r["image_id"]: (r["has_defect"] == "True", prov.get(r["image_id"], ""))
        for r in pop.rows
    }
    groups = {r["image_id"]: r["group_id"] for r in pop.rows}
    band = {i for i, (_, p) in ctx.items() if p == CROP}
    defect = {i for i in band if ctx[i][0]}
    normal = band - defect
    print(f"출처 고정 {CROP}: 결함 {len(defect):,} · 정상 {len(normal):,}")

    by_group: dict[str, list[str]] = {}
    for i in sorted(band):
        by_group.setdefault(groups[i], []).append(i)
    units = sorted(by_group)

    def sub(sc: dict[str, float], gs: Sequence[str]) -> float:
        """재표집된 묶음에 속한 이미지만으로 Δ_AUC 를 낸다. 같은 묶음이 여러 번 뽑히면
        그만큼 중복 계산된다(`cluster_bootstrap` 계약)."""
        ids = [i for g in gs for i in by_group[g]]
        return gini(sc, [i for i in ids if i in defect], [i for i in ids if i in normal])

    grid, source = params.conf_sweep, params.conf_sweep_source
    print(f"격자 {len(grid)}점 (출처 {source})")

    curves: dict[str, list[dict]] = {}
    gini_by_cell: dict[str, dict] = {}
    scores: dict[str, dict[str, float]] = {}
    for tag in DET_TAGS:
        recs = _load_raw(raw_record_path(params, tag))
        pts = []
        for conf in grid:
            fired = _filtered(recs, conf)
            pts.append({
                "conf": conf,
                "fire_rate_defect": sum(fired.get(i, False) for i in defect) / len(defect),
                "fire_rate_normal": sum(fired.get(i, False) for i in normal) / len(normal),
                "n_missing_prediction": sum(1 for i in band if i not in fired),
            })
            pts[-1]["delta"] = pts[-1]["fire_rate_defect"] - pts[-1]["fire_rate_normal"]
        curves[tag] = pts
        peak = max(pts, key=lambda p: p["delta"])
        op = next(p for p in pts if p["conf"] == OPERATING_CONF)
        print(f"[{tag}] 하한 {pts[0]['conf']:.2f} Δ {pts[0]['delta']:+.4f} · "
              f"운용 {OPERATING_CONF} Δ {op['delta']:+.4f} · "
              f"최대 {peak['conf']:.2f} Δ {peak['delta']:+.4f}")

        scores[tag] = {r.image_id: image_score(r) for r in recs}
        ci = cluster_bootstrap(
            units, lambda gs, t=tag: sub(scores[t], gs), drop_undefined=True
        )
        gini_by_cell[tag] = ci.as_dict()
        print(f"    임계 독립 Δ_AUC {ci.point:+.4f} [{ci.lo:+.4f}, {ci.hi:+.4f}]")

    # 칸 대비 — **같은 재표집에서 짝지어** 낸다. 칸마다 따로 낸 CI 를 눈으로 겹쳐 보면
    # 묶음이 공통이라는 사실이 빠져 차의 CI 가 실제보다 넓어진다.
    locals_ = [t for t in DET_TAGS if t.startswith("sep_local")]

    def contrast(fn):
        return cluster_bootstrap(
            units,
            lambda gs: fn({t: sub(scores[t], gs) for t in DET_TAGS}),
            drop_undefined=True,
        )

    def _mean_local(v):
        return sum(v[t] for t in locals_) / len(locals_)

    contrasts = {
        "fed_minus_central": contrast(lambda v: v["sep_fed"] - v["sep_central"]),
        "fed_minus_local_mean": contrast(lambda v: v["sep_fed"] - _mean_local(v)),
        "central_minus_local_mean": contrast(lambda v: v["sep_central"] - _mean_local(v)),
    }
    print("임계 독립 축 대비 (묶음 짝지음):")
    for k, v in contrasts.items():
        print(f"  {k:26s} {v.point:+.4f} [{v.lo:+.4f}, {v.hi:+.4f}]")
    den = contrasts["central_minus_local_mean"].point
    num = contrasts["fed_minus_local_mean"].point
    rec = None if den <= 0 else 100.0 * num / den
    print("  회복률 " + ("정의 불가(분모 ≤ 0)" if rec is None
                       else f"{rec:+.1f}% (분모 {den:+.4f})"))

    order = {c: sorted(DET_TAGS, key=lambda x: -next(
        p["delta"] for p in curves[x] if p["conf"] == c)) for c in grid}
    stable = len({tuple(v) for v in order.values()}) == 1
    print(f"칸 순위: 격자 전 점에서 {'동일' if stable else '뒤집힘 있음'}")
    for conf in grid:
        print(f"  {conf:.2f}: {' > '.join(order[conf])}")

    dest = Path(args.dest) if args.dest else params.out / "discrimination_sweep_v1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "provenance": CROP,
        "n_defect": len(defect),
        "n_normal": len(normal),
        "n_groups": len(units),
        "grid": list(grid),
        "grid_source": source,
        "operating_conf": OPERATING_CONF,
        "computed_from": "하한(export floor) 레코드 + 임계 필터 — 재추론 없음",
        "ci_note": "곡선은 점추정만. 운용점 CI 는 score_cells_v1.json 의 discrimination 블록",
        "threshold_free": {
            "definition": "2·AUROC − 1 (출처 고정 구간, 결함 대 정상, 동점 1/2)",
            "baseline": "0 = 순위 정보 없음. 지름길은 상수 점수라 정의상 정확히 0",
            "ci": "묶음 클러스터 부트스트랩 (칸 대비는 같은 재표집에서 짝지음)",
            "by_cell": gini_by_cell,
            "contrasts": {k: v.as_dict() for k, v in contrasts.items()},
            "recovery_pct": rec,
            "recovery_note": (
                "분모(중앙집중 − 로컬평균)가 작아 비율이 민감하다. 점추정 단독이 아니라 "
                "대비 CI 와 함께 읽어야 한다"
            ),
        },
        "rank_stable_across_grid": stable,
        "rank_by_conf": {f"{c:.2f}": order[c] for c in grid},
        "curves": curves,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
