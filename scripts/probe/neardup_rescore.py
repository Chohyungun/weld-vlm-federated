"""근사 중복 제외 뒤 사전실험 세 시드를 **재추론 없이** 다시 채점한다. 07번 미니스펙 §18.

    .venv/Scripts/python.exe -X utf8 -B scripts/probe/neardup_rescore.py \
        --exclude <제외 목록> --dest outputs/main_d/seed3set --version v1

## 무엇을 재나

"제외하면 수치가 얼마나 움직이는가" 를 숫자로 낸다. **채택 여부의 결정이 아니라 결정에 필요한 증거**다.

재추론이 필요 없다. 예측은 이미 있고, 바뀌는 것은 **어느 이미지를 세느냐** 뿐이다.
COCO 매칭 캐시는 이미지마다의 검출·정답을 들고 있어서 **제외된 이미지에 중복도 0 을 주면** 그대로 줄어든 모집단의 값이 된다.

## 왜 이 재채점이 필요한가

회복률은 `(연합 − 로컬평균) / (중앙 − 로컬평균)` 이다. 학습에서 본 그림이 평가에 남아 있으면
**분자와 분모 양쪽에서 노출이 높은 쪽이 유리하다.** 노출은 칸마다 다르다 —
train 에 상대가 있는 eval 이 중앙·연합은 3,905장(31.3 %)인데 C3 는 529장(4.2 %)이다.
로컬 세 모델의 평균이 분모에 들어가므로 이 비대칭이 회복률을 어느 쪽으로 미는지는 **세어 봐야 안다.**

## 비교할 수 있는 것과 없는 것

제외는 유병률을 바꾼다(결함 56.9 % → 39.4 %). 정상 이미지의 비중이 커지면 **모델이 같아도 정밀도가 내려간다.**

| | 제외 전후 직접 비교 |
|---|---|
| `map_50` · `map_50_95` | **하지 않는다.** 정밀도를 적분한 값이라 유병률에 직접 걸린다 |
| 회복률 | 두 벌을 낸다. 비율이라 덜 움직이지만 **두 모집단의 두 수**다 |
| 계수(TP·FP·FN)·장수·유병률·잔존율 | 비교한다. 세는 값이다 |

두 벌을 나란히 싣되 **차이를 "제외의 효과" 라고 부르지 않는다.** 모집단 변화와 누출 제거가 섞여 있고 여기서 가르지 못한다.

## 산출물

`neardup_rescore_{version}.json` 을 **새 이름으로** 쓴다. v3 산출물은 읽기만 한다.
목록이 확정 전이면 `list_status` 가 `provisional` 이고 목록 해시가 함께 박힌다 — 확정되면 다시 돌린다.

## 구간은 어느 계약으로 냈나 — 산출물이 스스로 말한다

이 스크립트의 구간은 **사전실험(분리형)의 옛 등록**(`evaluation.prereg.RECOVERY_CI_REGISTRATION`)을 따르는
동결 함수(`evaluation.recovery_ci.percentile_ci`)로 낸다 — 분모가 0 이하인 추첨은 버리고 세며, 백분위는
방식 인자 없이 `np.percentile` 을 부른다. **통합형 본실험의 등록 규약으로 낸 값이 아니다.** 통합형의 구간 규약
(보간 방식·무한 처리·구간 부재 표현)은 아직 고르지 않았다. 그래서 산출물의 `interval_convention` 이
옛 등록 블록과 신뢰수준·방식·미정의 처리의 이름을 통째로 싣고 그 문장을 적는다.

`--control` 이 내는 대조 분포의 2.5·97.5 백분위는 **신뢰구간이 아니다** — 무작위 제외 추첨의 참고 분포다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.adapters import read_records
from evaluation.cells import load_population
from evaluation.params import ScoringParams
from evaluation.probes.source_probe import load_provenance
from evaluation.prereg import RECOVERY_CI_ALPHA, RECOVERY_CI_INTERVAL, RECOVERY_CI_REGISTRATION
from evaluation.provenance import _file_sha256, write_new_text
from evaluation.recovery_ci import (
    _block_repeat_index,
    build_cache,
    identity_check,
    percentile_ci,
    recovery_from,
    weighted_map,
)
from evaluation.stats import BOOTSTRAP_N, BOOTSTRAP_SEED
from scripts.probe.score_cells import raw_record_path

TAGS = ("sep_central", "sep_local_C1", "sep_local_C2", "sep_local_C3", "sep_fed")
LOCALS = ("sep_local_C1", "sep_local_C2", "sep_local_C3")
CENTRAL, FED = "sep_central", "sep_fed"

EXPOSURE = {
    "sep_central": {"n": 3905, "frac": 0.313},
    "sep_fed": {"n": 3905, "frac": 0.313},
    "sep_local_C1": {"n": 2977, "frac": 0.239},
    "sep_local_C2": {"n": 1488, "frac": 0.119},
    "sep_local_C3": {"n": 529, "frac": 0.042},
}
"""train 에 상대가 있는 eval 장수 — 그 칸의 학습 자료에 같은 그림이 있다는 뜻이다.

중앙·연합은 학습 풀 전체를 보므로 같은 값이다. **이 비대칭이 이 재채점의 동기다.**
"""

EXPOSURE_NOTE = (
    "train 에 상대가 있는 eval 장수. 이 스크립트에 적어 둔 고정 배경 수치이고 이 실행이 "
    "계산한 값이 아니다 — 다른 제외 목록으로 돌리면 맞지 않는다. 노출을 나눈 값을 "
    "'이미지 한 장당 손실' 이라고 부르지 않는다(한 장씩 빼 본 것이 아니다)")
"""산출물의 `exposure_note`. 누가 준 수치인지가 아니라 **무엇인지**를 적는다."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CONTROL_PERCENTILES = (2.5, 97.5)
"""대조 추첨 분포에서 적는 두 백분위. **신뢰구간이 아니다.**"""


def _percentile_default() -> str:
    """실행하는 numpy 의 `np.percentile` 기본 방식. 글자로 적어 두지 않는다 — 판이 바뀌면 조용히 거짓이 된다."""
    import inspect
    return str(inspect.signature(np.percentile).parameters["method"].default)


def interval_convention() -> dict:
    """이 산출물의 구간이 따른 계약. **값 옆에서 출처와 지위를 읽게** 옛 등록 블록을 통째로 싣는다."""
    return {
        "registration": RECOVERY_CI_REGISTRATION,
        "alpha": RECOVERY_CI_ALPHA,
        "interval": RECOVERY_CI_INTERVAL,
        "percentile_method": ("np.percentile 기본값(방식 인자 없음) — 실행한 numpy "
                              f"{np.__version__} 의 기본값은 {_percentile_default()}"),
        "undefined_denominator": ("drop_undefined — 분모가 0 이하인 추첨의 회복률을 nan 으로 두고 "
                                  "구간 계산에서 버리며 n_undefined 로 센다"),
        "point_estimate_undefined": ("시드 하나라도 분모가 0 이하면 시드 평균 점추정은 null 이다 — "
                                     "구간처럼 버리지 않는다"),
        "not_unified_registration": ("통합형 등록 규약으로 낸 값이 아니다. 사전실험(분리형) 옛 등록의 "
                                     "구간 함수를 그대로 썼고, 통합형의 구간 규약은 아직 고르지 않았다"),
        "control_percentiles": {
            "levels": list(CONTROL_PERCENTILES),
            "what_this_is_not": "신뢰구간이 아니다 — 무작위 제외 추첨의 참고 분포에서 잰 백분위다",
        },
    }


def finite_or_none(x) -> float | None:
    """유한하면 수, 아니면 `None`. 표준 JSON 은 NaN 을 담지 못한다(`allow_nan=False` 가 멈춘다)."""
    return float(x) if x is not None and np.isfinite(x) else None


def mean_or_none(values) -> float | None:
    """시드 평균. **하나라도 미정의면 `None`** — 버리고 평균하면 다른 추정량이 된다."""
    vals = [finite_or_none(v) for v in values]
    return None if any(v is None for v in vals) or not vals else float(np.mean(vals))


def is_defect(value: str) -> bool:
    """매니페스트의 결함 여부. **`True`·`False` 두 글자만 받는다** — 표기가 바뀌면 전부 정상으로 세지 않고 멈춘다."""
    if value == "True":
        return True
    if value == "False":
        return False
    raise SystemExit(f"has_defect 가 True/False 가 아니다: {value!r}")


def cells_signature(members: dict, n_drop: dict) -> str:
    """대조 칸의 서명. 칸 키·칸 안 순서·제외 수가 같아야 같은 값이다 — 대조 재현이 이 값과 맞댄다."""
    return hashlib.sha256(
        "\n".join(f"{k[0]}|{k[1]}|{n_drop[k]}|" + ",".join(members[k])
                   for k in sorted(members)).encode("utf-8")).hexdigest()


SCORER_FILES: tuple[str, ...] = (
    "scripts/probe/neardup_rescore.py",
    "scripts/probe/score_cells.py",
    "evaluation/recovery_ci.py",
    "evaluation/cells.py",
    "evaluation/adapters.py",
    "evaluation/eval_set.py",
    "evaluation/probes/source_probe.py",
)
"""이 값을 낸 코드. **원점수 두 개의 항등만으로 입력 동일성을 대신할 수 없다** —
같은 mAP 를 내면서 대조군이나 가중이 달라질 수 있다."""


def code_provenance() -> dict:
    """채점 경로 파일들의 sha256. 줄끝을 LF 로 맞춘 뒤 센다 — 체크아웃 설정에 흔들리지 않게."""
    return {f: _file_sha256(REPO / f) for f in SCORER_FILES}


def seed_inputs(root: Path, n: int, artifact: str, params) -> dict:
    """이 시드가 읽은 것들의 해시. 산출물만 보고 입력 판본을 되짚을 수 있어야 한다."""
    out = {"artifact": sha256_of(root / f"seed{n}" / artifact),
           "raw_records": {t: sha256_of(raw_record_path(params, t)) for t in TAGS}}
    for name in ("manifest.csv", "tiles.csv"):
        path = params.snapshot / name
        if path.exists():
            out[name] = sha256_of(path)
    return out


def load_exclusion(path: Path) -> tuple[frozenset[str], dict]:
    raw = path.read_bytes()
    ids = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path.name}: 제외 목록에 중복이 있다")
    return frozenset(ids), {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest(),
                            "n": len(ids)}


def seed_caches(root: Path, n: int, artifact: str, log) -> tuple[dict, dict, list[str], dict]:
    """시드 하나의 캐시 다섯과 항등 검사. **채점 기준을 바꾸지 않는다** — 산출물은 읽기만 한다."""
    art = json.loads((root / f"seed{n}" / artifact).read_text(encoding="utf-8"))
    p = art["params"]
    params = ScoringParams(snapshot=Path(p["snapshot"]), pilot=Path(p["pilot"]),
                           out=root / f"seed{n}", seed=int(p["seed"]), profile=p["profile"])
    pop = load_population(params)
    gold = {i: pop.gold_boxes.get(i, []) for i in pop.eval_ids}
    caches, checks = {}, {}
    for tag in TAGS:
        # **하한 tier 의 원시 레코드**다(운용점 0.25 가 아니라 export 하한 0.01).
        # mAP 는 임계 독립 지표라 곡선 전 구간이 필요하고, CI 생성기도 같은 파일을 읽는다.
        src = raw_record_path(params, tag)
        recs = read_records(src.read_text(encoding="utf-8").splitlines())
        pred = {r.image_id: [(d.iso_code, tuple(d.bbox_px), d.score or 0.0)
                             for d in r.defects if d.bbox_px] for r in recs}
        cache = build_cache(pred, gold, pop.classes)
        if cache is None:
            raise SystemExit(f"{tag}: GT 박스 0 — 캐시를 만들 수 없다")
        exp = art["threshold_independent"]["per_tag"][tag]
        checks[tag] = identity_check(cache, exp["map_50"], exp["map_50_95"])
        if not checks[tag]["passed"]:
            raise SystemExit(f"시드 {n} {tag}: 항등 검사 실패 — 캐시가 v3 값을 재현하지 못한다")
        caches[tag] = cache
        log(f"  [{tag}] 항등 통과 · map_50 {checks[tag]['map_50']:.6f}")
    ids = next(iter(caches.values())).image_ids
    groups = {r["image_id"]: r["group_id"] for r in pop.rows}
    return caches, checks, ids, {"params": params, "pop": pop, "groups": groups}


def recovery_of(values: dict[str, float]) -> tuple[float, float]:
    return recovery_from(values[CENTRAL], values[FED], [values[t] for t in LOCALS])


def population_facts(pop, excluded: frozenset[str]) -> dict:
    """유병률과 층별 잔존율. **세는 값이라 제외 전후를 비교한다.**"""
    rows = pop.rows
    keep = [r for r in rows if r["image_id"] not in excluded]

    def split(rs):
        d = sum(1 for r in rs if is_defect(r["has_defect"]))
        return {"n": len(rs), "n_defect": d, "n_normal": len(rs) - d,
                "prevalence": (d / len(rs)) if rs else None}

    by_stratum = {}
    before = Counter(r["strata_key"] for r in rows)
    after = Counter(r["strata_key"] for r in keep)
    for key in sorted(before):
        by_stratum[key] = {"before": before[key], "after": after[key],
                           "survival": after[key] / before[key] if before[key] else None}
    share_before = {k: v / len(rows) for k, v in before.items()}
    share_after = {k: (after[k] / len(keep)) if keep else 0.0 for k in before}
    for key, cell in by_stratum.items():
        s_b, s_a = share_before[key], share_after[key]
        cell["share_before"] = s_b
        cell["share_after"] = s_a
        cell["share_ratio"] = (s_a / s_b) if s_b else None
    return {"before": split(rows), "after": split(keep),
            "n_excluded_in_population": len(rows) - len(keep),
            "by_stratum": by_stratum}


def material_facts(pop, excluded: frozenset[str]) -> dict:
    out = {}
    for mat in sorted({r["material"] for r in pop.rows}):
        rs = [r for r in pop.rows if r["material"] == mat]
        kept = [r for r in rs if r["image_id"] not in excluded]
        out[mat] = {"before": len(rs), "after": len(kept),
                    "survival": len(kept) / len(rs) if rs else None}
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="outputs/main_d")
    ap.add_argument("--exclude", required=True, type=Path, help="제외할 image_id 목록(한 줄에 하나)")
    ap.add_argument("--list-status", default="provisional", choices=("provisional", "final"))
    ap.add_argument("--artifact", default="score_cells_v3.json")
    ap.add_argument("--dest", default="outputs/main_d/seed3set")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--resamples", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--no-ci", action="store_true",
                    help=("재표집 구간을 내지 않는다. --control 을 함께 주면 대조 추첨 분포의 "
                          "2.5·97.5 백분위는 그대로 나간다 — 신뢰구간이 아닌 참고 분포다"))
    ap.add_argument("--allow-empty-exclusion", action="store_true",
                    help="빈 제외 목록을 허용한다(항등 대조 목적일 때만)")
    ap.add_argument("--control", type=int, default=0,
                    help="층·출처를 맞춘 무작위 제외를 N 번 뽑아 구성 변화만의 몫을 잰다")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root = REPO / args.root
    dest = REPO / args.dest
    out_path = dest / f"neardup_rescore_{args.version}.json"
    if out_path.exists():
        raise SystemExit(f"이미 있다: {out_path} — 새 판으로 낸다(옛 산출물을 덮지 않는다)")

    excluded, list_meta = load_exclusion(args.exclude)
    seeds = [int(x) for x in args.seeds.split(",")]
    t0 = time.perf_counter()
    print(f"제외 목록 {list_meta['n']}장 · {args.list_status} · sha {list_meta['sha256'][:16]}…")

    per_seed: dict[int, dict] = {}
    curves: dict[int, dict] = {}
    control: dict[int, dict] = {}
    masks: dict[int, np.ndarray] = {}
    meta_by_seed: dict[int, dict] = {}
    ref_ids: list[str] = []
    ref_gmap = np.empty(0, dtype=np.int64)
    ref_ctl: dict = {}
    for n in seeds:
        print(f"시드 {n}")
        caches, checks, ids, extra = seed_caches(root, n, args.artifact, print)
        # 목록에 평가 모집단 밖 ID 가 있으면 **아무것도 제외하지 않고 지나간다.** 목록 장수와
        # 실제 적용 수가 조용히 갈라지므로 여기서 멈춘다. 빈 목록도 실수로 보고 막는다.
        defect_of = {r["image_id"]: is_defect(r["has_defect"]) for r in extra["pop"].rows}
        outside = sorted(excluded - set(ids))
        if outside:
            raise SystemExit(
                f"제외 목록에 평가 모집단 밖 ID 가 {len(outside)}개 있다 — 예: {outside[:3]}")
        if not excluded and not args.allow_empty_exclusion:
            raise SystemExit("제외 목록이 비었다 — 항등 대조가 목적이면 --allow-empty-exclusion 을 준다")
        mask = np.array([0 if i in excluded else 1 for i in ids], dtype=np.int64)
        ones = np.ones(len(ids), dtype=np.int64)
        before = {t: weighted_map(caches[t], ones)[0] for t in TAGS}
        after = {t: weighted_map(caches[t], mask)[0] for t in TAGS}
        r_b, d_b = recovery_of(before)
        r_a, d_a = recovery_of(after)
        per_seed[n] = {
            "map_50_before": before, "map_50_after": after,
            "recall_floor_before": {t: floor_recall(caches[t], ones) for t in TAGS},
            "recall_floor_after": {t: floor_recall(caches[t], mask) for t in TAGS},
            "recovery_before": finite_or_none(r_b), "denominator_before": d_b,
            "recovery_after": finite_or_none(r_a), "denominator_after": d_a,
            "identity_check": {t: checks[t]["passed"] for t in TAGS},
            "n_images_before": len(ids), "n_images_after": int(mask.sum()),
            "n_excluded_applied": int(len(ids) - mask.sum()),
            "n_support_classes_before": n_support_classes(caches[CENTRAL], ones),
            "n_support_classes_after": n_support_classes(caches[CENTRAL], mask),
            "inputs": seed_inputs(root, n, args.artifact, extra["params"]),
        }
        masks[n], meta_by_seed[n] = mask, extra
        print(f"  회복률 {r_b * 100:+.2f}% → {r_a * 100:+.2f}%  (분모 {d_b:.4f} → {d_a:.4f})")

        gmap, n_groups = group_map(ids, extra["groups"])
        if seeds.index(n) == 0:
            ref_ids, ref_gmap, ref_ng = list(ids), gmap, n_groups
        elif list(ids) != ref_ids or not np.array_equal(gmap, ref_gmap):
            # 같은 추첨을 쓰려면 좌표계가 같아야 한다. 다르면 조용히 어긋난 구간이 나온다.
            raise SystemExit(f"시드 {n}: 평가 모집단·묶음 사상이 시드 {seeds[0]} 과 다르다")
        if args.control:
            members, n_drop, ctl_meta = control_cells(extra["pop"], excluded,
                                                      extra["params"].snapshot)
            if seeds.index(n) == 0:
                ref_ctl = ctl_meta
            elif ctl_meta["signature"] != ref_ctl["signature"]:
                # 서명이 다르면 "b 번째 추첨이 세 시드에서 같은 장을 뺀다" 가 성립하지 않는다.
                raise SystemExit(f"시드 {n}: 대조 칸 서명이 시드 {seeds[0]} 과 다르다")
            # 씨앗을 시드마다 같게 둔다 — b 번째 추첨이 세 시드에서 **같은 장을 뺀다.**
            # 그래야 시드 평균 회복률의 귀무분포도 b 번째끼리 묶어 낼 수 있다.
            is_defect = np.array(
                [defect_of.get(i, False) for i in ids], dtype=bool)
            control[n] = control_draws(caches, ids, members, n_drop, args.control,
                                       BOOTSTRAP_SEED, print, f"시드 {n}",
                                       is_defect=is_defect)
        if not args.no_ci:
            curves[n] = seed_draw_curves(caches, mask, gmap, n_groups, args.resamples,
                                         print, f"시드 {n}")
        # 캐시는 시드 하나분만 든다 — 셋을 한꺼번에 들면 메모리가 크다.
        caches.clear()
        if n != seeds[0]:
            meta_by_seed[n].pop("pop", None)   # 대조 추첨까지 끝난 뒤에 놓는다

    summary = summarize_recovery(per_seed, seeds)

    pop0 = meta_by_seed[seeds[0]]["pop"]
    pop_facts = population_facts(pop0, excluded)

    ctl = {"computed": False, "reason": "--control 0"}
    if args.control:
        first = seeds[0]
        ctl = control_summary(control, per_seed, seeds, args.control, {
            "prevalence": pop_facts["after"]["prevalence"],
            "n_kept": per_seed[first]["n_images_after"],
            "n_support_classes": per_seed[first]["n_support_classes_after"],
        })
        ctl["cells"] = ref_ctl["cells"]
        ctl["cells_signature"] = ref_ctl["signature"]
        ctl["rng_seed"] = BOOTSTRAP_SEED
        ctl["percentiles"] = interval_convention()["control_percentiles"]

    ci = {"computed": False, "reason": "--no-ci"}
    if not args.no_ci:
        ci = combine_curves(curves, seeds, args.resamples)
        ci["n_groups"] = int(ref_ng)

    payload = {
        "artifact_version": args.version,
        "kind": "neardup_rescore",
        "list_status": args.list_status,
        "caveat": (
            "제외 목록은 확정 전이다. 확정되면 다시 돌린다."
            if args.list_status == "provisional" else "확정 목록으로 돌렸다."),
        "exclusion_list": list_meta,
        "source_artifact": args.artifact,
        "comparability": {
            "직접 비교하지 않는다": ["map_50", "map_50_95", "정밀도", "F1", "AUC-PR"],
            "각 모집단의 기술통계로 본다": ["계수(TP·FP·FN)", "장수", "유병률", "층별 잔존율",
                                          "하한 tier 재현율"],
            "두 벌을 낸다": ["회복률", "분모", "신뢰구간"],
            "이유": ("제외가 모집단 구성을 바꾼다. 값이 어느 쪽으로 움직일지는 뺀 영상의 "
                     "정답 박스 수·난이도·오탐 분포에 달려 있어 유병률 하나로 정해지지 않는다"),
            "주의": ("두 값의 차를 '제외의 효과'라 부르지 않는다. 같은 예측을 다른 평가 모집단에서 "
                     "집계한 결과이고, 근사 중복 후보 제거와 모집단 구성 변화가 함께 있다"),
            "재현율 주의": ("재현율도 구성에서 자유롭지 않다. 쉬운 정답이 있는 영상을 빼면 재학습 없이 "
                            "내려간다. 전후 차이를 검출력 변화로 읽지 않는다"),
        },
        "exposure_train_side": EXPOSURE,
        "exposure_note": EXPOSURE_NOTE,
        "interval_convention": interval_convention(),
        "code_provenance": code_provenance(),
        "per_seed": per_seed,
        "summary": summary,
        "ci": ci,
        "composition_control": ctl,
        "population": pop_facts,
        "material": material_facts(pop0, excluded),
        "limits": [
            ("제외 목록은 근사 중복 **후보**다. 오탐·미탐 검수가 끝난 확정 목록이 아니므로 "
             "확정 중복 수의 하한이라고 부르지 않는다. 목록이 바뀌면 새 판으로 다시 낸다"),
            ("재표집 단위가 묶음이다. eval 안쪽에 남은 근사 중복 쌍을 잇는 연결 성분으로 올리면 "
             "의존성이 반영되는데, 구간 폭이 반드시 넓어진다는 보장은 아니다 — 쌍 목록이 오면 실측한다"),
            ("CI 는 **고정된 세 학습 시드에 조건부인** 평가 묶음 부트스트랩이다. 학습 시드를 "
             "다시 표집한 구간이 아니다. 시드 평균 점추정과 시드 하나의 구간을 같은 추정량처럼 두지 않는다"),
            ("대조군이 맞춘 것은 칸별 제외 장수뿐이다. 실제 결함 유병률·클래스 조합·정답 박스 수·"
             "난이도는 맞추지 않았다 — 남은 차이는 composition_control.residual_composition 에 있다"),
            "제외 전 값도 같은 평가셋에서 나온 값이라 누출을 안고 있다. 두 값 모두 오염의 정도가 다를 뿐이다",
        ],
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    dest.mkdir(parents=True, exist_ok=True)
    # NaN 토큰은 표준 JSON 이 아니다 — 읽는 쪽이 조용히 실패하거나 0 으로 읽는다.
    write_new_text(out_path,
                   json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"\n{out_path.relative_to(REPO)}")
    print(f"회복률 평균 {_pct(summary['recovery_mean_before'])} → {_pct(summary['recovery_mean_after'])}")
    return 0


def _pct(x: float | None) -> str:
    return "미정의" if x is None else f"{x * 100:+.2f}%"


def summarize_recovery(per_seed: dict[int, dict], seeds) -> dict:
    """시드 평균 회복률. **시드 하나라도 미정의면 평균도 `None`** 이고 까닭과 수를 적는다.

    앞 판은 `np.mean` 이 NaN 을 내고 표준 JSON 쓰기(`allow_nan=False`)에서 계산을 다 한 뒤에 멈췄다.
    같은 산출 안에서 구간은 미정의 추첨을 조용히 버리는데 점추정은 죽었다 — 두 처리를 이름으로 가른다.
    """
    out = {}
    for arm in ("before", "after"):
        vals = [per_seed[n][f"recovery_{arm}"] for n in seeds]
        out[f"recovery_mean_{arm}"] = mean_or_none(vals)
        out[f"recovery_by_seed_{arm}"] = dict(zip(map(str, seeds), vals, strict=True))
        out[f"n_seeds_undefined_{arm}"] = sum(1 for v in vals if v is None)
    out["undefined_rule"] = interval_convention()["point_estimate_undefined"]
    return out


def group_map(ids: Sequence[str], groups: dict[str, str]) -> tuple[np.ndarray, int]:
    """이미지를 묶음 번호로 사상한다. 재표집 단위가 묶음이라 이 사상이 추첨의 좌표계다."""
    names = sorted({groups[i] for i in ids})
    idx = {g: k for k, g in enumerate(names)}
    return np.asarray([idx[groups[i]] for i in ids], dtype=np.int64), len(names)


def seed_draw_curves(caches, mask, gmap, n_groups, n_resamples, log, label) -> dict:
    """시드 하나의 재표집 곡선. **난수를 시드마다 같은 자리에서 다시 튼다.**

    세 시드가 같은 평가셋을 보므로 추첨도 같아야 한다 — 추첨이 갈리면 평가셋 불확실성이
    시드 평균에서 √3 만큼 깎여 구간이 좁아진다. 같은 씨앗으로 되틀면 세 시드가 b 번째
    추첨에서 **같은 중복도**를 쓴다. 대신 캐시는 한 시드분만 들고 있으면 된다.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    p = np.full(n_groups, 1 / n_groups)
    out = {"before": {"r": [], "d": []}, "after": {"r": [], "d": []}}
    t0 = time.perf_counter()
    for b in range(n_resamples):
        drawn = rng.multinomial(n_groups, p)[gmap]
        # 추첨은 **한 번**이다. 같은 중복도를 두 팔이 나눠 쓴다 — 짝지어야 차이의 구간이 선다.
        for arm, w in (("before", drawn), ("after", drawn * mask)):
            m = {t: weighted_map(caches[t], w)[0] for t in TAGS}
            r, d = recovery_of(m)
            out[arm]["r"].append(r)
            out[arm]["d"].append(d)
        if (b + 1) % 100 == 0:
            log(f"  [{label}] 재표집 {b + 1}/{n_resamples} ({time.perf_counter() - t0:.0f}s)")
    return out


def floor_recall(cache, weights: np.ndarray) -> dict:
    """하한 tier 에서의 **재현율** — 제외 전후를 직접 비교해도 되는 값이다.

    분모가 정답 박스 수뿐이라 유병률이 바뀌어도 뜻이 흔들리지 않는다. 정밀도 계열(mAP·F1·
    AUC-PR)은 오탐이 분모에 들어가 모집단 구성에 끌려가지만, 재현율은 '있는 결함 중 몇 개를
    찾았는가' 하나만 잰다. 그래서 하락이 보이면 그것은 구성이 아니라 검출력의 변화다.

    IoU 0.5 한 층만 본다(`dt_match[0]`). 하한 tier 라 점수 문턱이 사실상 없으므로
    이 값은 그 모델이 **도달 가능한 최대 재현율**이다.
    """
    weights = np.asarray(weights, dtype=np.int64)
    per_class, tp_all, gt_all = {}, 0, 0
    for k, c in enumerate(cache.per_class):
        idx = _block_repeat_index(c.start, c.length, weights)
        npig = int((c.npig * weights).sum())
        hit = int(((c.dt_match[0, idx] > 0) & ~c.dt_ignore[0, idx]).sum())
        per_class[cache.classes[k]] = (hit / npig) if npig else None
        tp_all += hit
        gt_all += npig
    vals = [v for v in per_class.values() if v is not None]
    return {"per_class": per_class,
            "macro": float(np.mean(vals)) if vals else None,
            "micro": (tp_all / gt_all) if gt_all else None,
            "n_gt": gt_all}


def control_cells(pop, excluded: frozenset[str], snapshot: Path) -> tuple[dict, dict, dict]:
    """대조 추첨의 칸을 (층 × 출처) 로 잡는다.

    층만 맞추면 **함정 11** 이 샌다 — 결함은 거의 전부 1280x720 크롭이고 정상은 파노라마
    타일이라 출처가 난이도의 큰 부분을 설명한다. 근사 중복은 크롭 쪽에 몰려 있으므로,
    출처를 맞추지 않은 대조는 '중복을 뺐더니 떨어졌다' 와 '쉬운 크롭을 뺐더니 떨어졌다' 를
    가르지 못한다.

    출처는 매니페스트에 없다 — 동결 스냅샷의 `tiles.csv` 가 정본이다(`source_probe` 와 같은
    출처를 쓴다). 매니페스트의 `source` 열은 데이터셋 id(`aihub71761`) 라 여기서는 상수다.
    """
    prov = load_provenance(snapshot / "tiles.csv")
    members: dict[tuple[str, str], list[str]] = {}
    n_drop: dict[tuple[str, str], int] = {}
    missing = 0
    for r in pop.rows:
        origin = prov.get(r["image_id"])
        if origin is None:
            missing += 1
            origin = "unknown"
        key = (r["strata_key"], origin)
        members.setdefault(key, []).append(r["image_id"])
        n_drop.setdefault(key, 0)
        if r["image_id"] in excluded:
            n_drop[key] += 1
    if missing:
        raise SystemExit(f"출처 미상 {missing}장 — tiles.csv 가 평가 모집단을 덮지 못한다")
    cells = {f"{k[0]}|{k[1]}": {"n": len(v), "n_dropped": n_drop[k]}
             for k, v in sorted(members.items())}
    # **칸 서명** — 시드마다 같은 추첨을 쓴다는 주석이 성립하려면 칸 키·칸 안 순서·제외 수가
    # 시드 사이에서 같아야 한다. ID·묶음 사상이 같아도 원천 행 순서가 달라지면 깨진다.
    return members, n_drop, {"cells": cells, "signature": cells_signature(members, n_drop)}


def n_support_classes(cache, w: np.ndarray) -> int:
    """이 가중에서 **GT 가 남아 있는 클래스 수**.

    `weighted_map` 은 GT 0 인 클래스를 격자에서 빼고 평균한다. 즉 고정 K 개로 나누는
    macro AP 가 아니다 — 추첨이나 모집단에 따라 평균에 들어가는 클래스가 달라질 수 있다.
    "고정 support 로 비교했다"고 쓰지 않기 위해 그 수를 센다.
    """
    return sum(1 for c in cache.per_class if int((c.npig * w).sum()) > 0)


def control_draws(caches, ids, members, n_drop, n_draws, rng_seed, log, label,
                  *, is_defect: np.ndarray) -> dict:
    """같은 묶음 대표층·같은 전처리 출처에서 같은 장수를 무작위로 제외한 **참고 분포**.

    **무엇을 맞췄고 무엇을 안 맞췄는지.** 맞춘 것은 `strata_key`(묶음 대표 클래스)와
    `tiles.csv` 의 출처별 **제외 장수** 둘뿐이다. `strata_key` 는 이미지별 실제 라벨이 아니라
    묶음 구성원 라벨의 대표라, 같은 칸 안에 정상과 결함이 섞여 있다. 따라서 이 추첨은
    **실제 결함 유병률·다중 클래스 구성·정답 박스 수·난이도를 맞추지 않는다.**

    그래서 추첨마다 남은 구성(유병률·장수·지지량 클래스 수)을 함께 기록한다. 실제 제외와의
    차이를 곧바로 "중복 제거의 몫" 이라 부르지 않기 위한 재료다 — 맞춘 공변량 밖의 차이가
    얼마나 남았는지 눈에 보여야 한다.
    """
    rng = np.random.default_rng(rng_seed)
    pos = {i: k for k, i in enumerate(ids)}
    out = {t: [] for t in TAGS}
    out["recovery"], out["denominator"] = [], []
    out["prevalence"], out["n_kept"], out["n_support_classes"] = [], [], []
    t0 = time.perf_counter()
    for b in range(n_draws):
        w = np.ones(len(ids), dtype=np.int64)
        for key, k in n_drop.items():
            if k == 0:
                continue
            pool = members[key]
            for j in rng.choice(len(pool), size=k, replace=False):
                w[pos[pool[j]]] = 0
        m = {t: weighted_map(caches[t], w)[0] for t in TAGS}
        r, d = recovery_of(m)
        for t in TAGS:
            out[t].append(m[t])
        out["recovery"].append(r)
        out["denominator"].append(d)
        keep = w > 0
        n_keep = int(keep.sum())
        out["n_kept"].append(n_keep)
        out["prevalence"].append(float(is_defect[keep].sum()) / n_keep if n_keep else None)
        out["n_support_classes"].append(n_support_classes(caches[CENTRAL], w))
        if (b + 1) % 10 == 0:
            log(f"  [{label}] 대조 추첨 {b + 1}/{n_draws} ({time.perf_counter() - t0:.0f}s)")
    return out


def _against_control(arr, actual) -> dict:
    """대조 분포 하나와 실제값 하나를 맞댄다. **미정의를 세고, 순위에서 뺀다.**

    미정의(NaN)가 섞이면 평균·분위수가 통째로 NaN 이 되고, 비교식에서 NaN 은 언제나 거짓이라
    백분위가 `0.0` 이라는 **숫자**로 보인다. "가장 낮았다" 와 "잴 수 없었다" 가 같은 값으로
    나가는 것이라 반드시 갈라 둔다.
    """
    a = np.asarray(arr, dtype=float)
    ok = np.isfinite(a)
    val = a[ok]
    out = {"n_draws": int(a.size), "n_defined": int(ok.sum()),
           "n_undefined": int((~ok).sum()),
           "actual": (float(actual) if np.isfinite(actual) else None)}
    if val.size == 0 or out["actual"] is None:
        out.update({"control_mean": None, "control_lo": None, "control_hi": None,
                    "percentile_of_actual": None, "gap_vs_control_mean": None,
                    "reason": "유효 추첨 0" if val.size == 0 else "실제값 미정의"})
        return out
    mean = float(np.mean(val))
    out.update({"control_mean": mean,
                "control_lo": float(np.percentile(val, CONTROL_PERCENTILES[0])),
                "control_hi": float(np.percentile(val, CONTROL_PERCENTILES[1])),
                "percentile_of_actual": float((val < out["actual"]).mean() * 100),
                "gap_vs_control_mean": out["actual"] - mean})
    return out


def _spread(arr) -> dict:
    """구성 지표의 퍼짐. 실제값과 맞대는 것이 아니라 **분포 자체**를 적는다."""
    a = np.asarray([v for v in arr if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(np.mean(a)),
            "lo": float(np.percentile(a, CONTROL_PERCENTILES[0])),
            "hi": float(np.percentile(a, CONTROL_PERCENTILES[1])),
            "min": float(a.min()), "max": float(a.max())}


def control_summary(control: dict[int, dict], per_seed: dict[int, dict], seeds, n_draws,
                    actual_facts: dict) -> dict:
    """대조 분포와 실제 제외를 맞댄다.

    **읽는 법.** 이것은 정한 공변량(묶음 대표층 × 전처리 출처)만 맞춘 무작위 제외와의 차이다.
    분포 밖이라고 곧바로 "중복 제거의 몫" 이 아니고, 분포 안이라고 "구성 변화로 전부 설명됨" 도
    아니다. 맞추지 않은 축(실제 결함 라벨·다중 클래스·정답 박스 수·난이도)의 차이가 남는다.
    그 남은 차이의 크기를 `residual_composition` 에 적는다.
    """
    out = {"computed": True, "n_draws": n_draws,
           "matched_on": ["strata_key(묶음 대표 클래스)", "tiles.csv 의 출처(N-crop·N-tile·N-band)"],
           "matched_quantity": "칸별 제외 장수",
           "not_matched": ["이미지별 실제 결함 여부", "다중 클래스 조합", "정답 박스 수",
                           "묶음 크기·묶음 상관", "층 안의 난이도"],
           "reading": ("정한 공변량을 맞춘 무작위 제외와의 차이다. 인과 분해가 아니며 "
                       "분포 밖을 중복 제거의 몫으로, 분포 안을 구성 변화로 전부 설명됨으로 읽지 않는다"),
           "per_tag": {}}
    for t in [*TAGS, "recovery", "denominator"]:
        rows = {}
        for n in seeds:
            actual = per_seed[n]["map_50_after"][t] if t in TAGS else per_seed[n][f"{t}_after"]
            rows[str(n)] = _against_control(control[n][t], actual)
        (out["per_tag"] if t in TAGS else out)[t] = rows

    # 헤드라인은 시드 평균이다 — 그 자리의 참고 분포도 같은 추첨으로 낸다.
    # 어느 시드든 미정의인 추첨은 평균에서도 미정의다(CI 쪽 규칙과 같게 둔다).
    stacked = np.vstack([np.asarray(control[n]["recovery"], dtype=float) for n in seeds])
    mean_curve = np.where(np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan)
    out["recovery_mean_over_seeds"] = _against_control(
        mean_curve, float(np.mean([per_seed[n]["recovery_after"] for n in seeds])))

    # **맞추지 않은 축이 얼마나 남았나.** 실제 제외 뒤의 값과 대조 추첨 분포를 나란히 둔다.
    first = seeds[0]
    out["residual_composition"] = {
        "note": ("맞춘 것은 칸별 제외 장수뿐이다. 아래가 대조 추첨에서 실제로 남은 구성이며, "
                 "실제 제외의 값과 다르면 그만큼이 맞추지 못한 차이다"),
        "prevalence": {"actual": actual_facts.get("prevalence"),
                       "control": _spread(control[first]["prevalence"])},
        "n_kept": {"actual": actual_facts.get("n_kept"),
                   "control": _spread(control[first]["n_kept"])},
        "n_support_classes": {"actual": actual_facts.get("n_support_classes"),
                              "control": _spread(control[first]["n_support_classes"]),
                              "note": ("GT 가 남은 클래스 수. weighted_map 은 GT 0 클래스를 평균에서 "
                                       "빼므로 고정 support macro AP 가 아니다")},
    }
    return out


def combine_curves(curves: dict[int, dict], seeds, n_resamples) -> dict:
    """시드별 곡선을 b 번째끼리 묶어 평균한다 — 같은 추첨이므로 짝이 맞는다."""
    out = {"computed": True, "n_resamples": n_resamples, "rng_seed": BOOTSTRAP_SEED,
           "cluster_unit": "group", "alpha": RECOVERY_CI_ALPHA,
           "convention": "interval_convention 을 본다 — 통합형 등록 규약으로 낸 값이 아니다",
           "note": "같은 추첨을 두 모집단·세 시드에 함께 적용했다 — 그래서 차이의 구간도 낸다"}
    rbar = {}
    for arm in ("before", "after"):
        stacked = np.vstack([np.asarray(curves[n][arm]["r"]) for n in seeds])
        rbar[arm] = np.where(np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan)
        out[arm] = {"recovery_mean": percentile_ci(rbar[arm], RECOVERY_CI_ALPHA),
                    "denominator": {str(n): percentile_ci(np.asarray(curves[n][arm]["d"]),
                                                          RECOVERY_CI_ALPHA)
                                    for n in seeds},
                    "recovery_by_seed": {str(n): percentile_ci(np.asarray(curves[n][arm]["r"]),
                                                               RECOVERY_CI_ALPHA)
                                         for n in seeds}}
    out["delta_recovery_mean"] = percentile_ci(rbar["after"] - rbar["before"], RECOVERY_CI_ALPHA)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
