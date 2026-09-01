"""채점 모집단 선택지 비용표 — 67번 질문 2 의 결정 입력. 71번 과제 6.

세 선택지를 같은 축으로 세운다: **패스 수 · 추론 GPU-일 · P9 검정력 · 자명하한 · 방어력.**

| 축 | 전량 12,461 × 시드3 | 전량 × 시드1 | 층화 부표본 2,000 × 시드1 |

부표본 안의 검정력 손실은 **추정하지 않고 시뮬레이션한다** — 동결 평가셋에서 실제로
묶음 단위 층화 추출을 돌려 N-crop 정상 장수·묶음 수를 센다. 이미지 단위로 뽑으면
불변조건 1-5(중복 묶음 단위 분할)를 어기므로 묶음 단위가 유일한 후보다.

자명하한은 **표본 구성이 바뀌면 값이 바뀐다**(`trivial_bound` 독스트링). 부표본을 택하면
사전등록 상수를 그 표본에서 다시 산출해 병기해야 하므로, 그 값도 여기서 낸다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np

from data.label_map import load_label_map
from evaluation.eval_set import parse_iso_codes, read_manifest
from evaluation.probes.metadata_probe import MetaSample, trivial_bound

FROZEN = Path("data/interim/manifest_v1")
OUT = Path("outputs/pilot_d")
SUBSAMPLE_N = 2000
SUBSAMPLE_SEED = 20260825
P9_MIN_CLUSTERS = 20
CROP = "N-crop"

#: 통합형 이미지 생성 실측(61번 §4: 평가 653장 1.4시간). 4B 배율은 미실측(67번 §4-4).
UNI_SEC_PER_IMAGE = 7.72
#: ⑤ 판정부 실측(71번 과제 4, `judge_cost_probe_v1.json`). 텍스트 전용이라 훨씬 싸다.
#: 검색 적중 이미지에서만 생성이 돌므로 **이미지당** 값에 적중률이 이미 반영돼 있다.
JUDGE_SEC_PER_IMAGE = 2.7432
MODEL_SCALE = (3.0, 5.0)
#: 패스 수 — 67번 §2-4. 분리형 5모델 × 시드 + 통합형 2칸 × 1시드.
PASSES = {"전량×시드3": 17, "전량×시드1": 7, "부표본2000×시드1": 7}
#: 그 패스의 내역. 분리형은 ⑤(텍스트), 통합형은 이미지 생성이다.
PASS_SPLIT = {
    "전량×시드3": {"judge": 15, "uni": 2},
    "전량×시드1": {"judge": 5, "uni": 2},
    "부표본2000×시드1": {"judge": 5, "uni": 2},
}


def samples_of(rows) -> list[MetaSample]:
    return [
        MetaSample(
            image_id=r["image_id"], width_px=int(r["width_px"]),
            height_px=int(r["height_px"]), file_bytes=0, n_channels=1, quant_table_id=0,
            iso_codes=tuple(sorted(parse_iso_codes(r["iso_codes"]))),
        )
        for r in rows
    ]


def group_stratified_subsample(rows, n_target: int, seed: int) -> list[dict]:
    """묶음 단위 층화 추출. `strata_key` 비율을 유지하며 목표 장수에 가장 가깝게 뽑는다.

    이미지 단위로 뽑지 않는 이유는 불변조건 1-5 다 — 같은 용접부 연속 촬영이 표본 안팎으로
    갈리면 부표본이 전량 평가셋의 축소판이 아니게 된다.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    strata: dict[str, list[str]] = defaultdict(list)
    for g, members in by_group.items():
        strata[members[0]["strata_key"]].append(g)

    rng = np.random.default_rng(seed)
    total = len(rows)
    picked: list[dict] = []
    for key in sorted(strata):
        groups = sorted(strata[key])
        stratum_images = sum(len(by_group[g]) for g in groups)
        quota = n_target * stratum_images / total
        order = rng.permutation(len(groups))
        taken = 0
        for i in order:
            g = groups[i]
            if taken >= quota:
                break
            picked.extend(by_group[g])
            taken += len(by_group[g])
    return picked


def p9_precision_sim(rows, prov, *, fp_rate: float, clustered: bool, seed: int) -> dict:
    """검정력 시뮬레이션 — **묶음 수만 다른 두 모집단에서 TOST 가 성립하는가.**

    "묶음 최소선 20 을 넘는다"는 형식 조건일 뿐이고, 실제로 등가를 주장하려면 차의 CI 가
    ±δ 안에 들어가야 한다. 그래서 실제 묶음 구조 위에 **같은 오탐률**을 얹고 CI 폭을 잰다.
    두 출처에 같은 비율을 주므로 참값 차는 0 이고, 성립 여부는 순수하게 정밀도 문제다.

    Args:
        clustered: True 면 묶음 단위로 오탐 여부를 정한다(묶음 내 완전 상관, 보수적).
            False 면 이미지 단위 독립(낙관적). 실제는 둘 사이다.
    """
    from evaluation.probes.cross_source import NormalImage, p9_cross_source

    rng = np.random.default_rng(seed)
    normals = [r for r in rows if r["has_defect"] == "False"]
    if clustered:
        groups = sorted({r["group_id"] for r in normals})
        flag = {g: bool(rng.random() < fp_rate) for g in groups}
        images = [
            NormalImage(r["image_id"], r["group_id"], prov.get(r["image_id"], "?"),
                        flag[r["group_id"]])
            for r in normals
        ]
    else:
        images = [
            NormalImage(r["image_id"], r["group_id"], prov.get(r["image_id"], "?"),
                        bool(rng.random() < fp_rate))
            for r in normals
        ]
    images = [i for i in images if i.source in ("N-crop", "N-tile")]
    rep = p9_cross_source(images)
    return {
        "fp_rate": fp_rate,
        "clustered": clustered,
        "diff_half_width": round(rep.diff.half_width, 4),
        "n_crop_groups": rep.tost.n_clusters_a,
        "equivalent": rep.tost.equivalent,
        "verdict": rep.tost.verdict,
    }


def p9_power(rows, prov) -> dict:
    normals = [r for r in rows if r["has_defect"] == "False"]
    crop = [r for r in normals if prov.get(r["image_id"]) == CROP]
    groups = {r["group_id"] for r in crop}
    return {
        "n_normal": len(normals),
        "n_crop_normal": len(crop),
        "n_crop_groups": len(groups),
        "formal_verdict_possible": len(groups) >= P9_MIN_CLUSTERS,
        "ci_width_scale_vs_full": None,   # 아래에서 채운다
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lm = load_label_map()
    classes = [lm.iso_code(n) for n in
               ("crack", "porosity", "lack_of_fusion", "slag_inclusion")]
    with (FROZEN / "tiles.csv").open(encoding="utf-8", newline="") as fh:
        prov = {r["image_id"]: r["provenance"] for r in csv.DictReader(fh)}

    rows = read_manifest(FROZEN)
    ev = [r for r in rows if r["split"] == "eval"]
    sub = group_stratified_subsample(ev, SUBSAMPLE_N, SUBSAMPLE_SEED)

    full_power = p9_power(ev, prov)
    sub_power = p9_power(sub, prov)
    if full_power["n_crop_groups"] and sub_power["n_crop_groups"]:
        sub_power["ci_width_scale_vs_full"] = round(
            (full_power["n_crop_groups"] / sub_power["n_crop_groups"]) ** 0.5, 2
        )

    bound_full = trivial_bound(samples_of(ev), classes)
    bound_sub = trivial_bound(samples_of(sub), classes)

    def gpu_days(n_passes: int, n_images: int, sec: float) -> dict:
        base = n_passes * n_images * sec / 86400
        return {
            "0.8B": round(base, 2),
            "4B_x3": round(base * MODEL_SCALE[0], 2),
            "4B_x5": round(base * MODEL_SCALE[1], 2),
        }

    options = {
        "전량×시드3": {"n_images": len(ev), "power": full_power, "bound": bound_full},
        "전량×시드1": {"n_images": len(ev), "power": full_power, "bound": bound_full},
        "부표본2000×시드1": {"n_images": len(sub), "power": sub_power, "bound": bound_sub},
    }
    table = {}
    for name, o in options.items():
        n_pass = PASSES[name]
        pw = o["power"]
        split = PASS_SPLIT[name]
        sec_mixed = (
            split["judge"] * JUDGE_SEC_PER_IMAGE + split["uni"] * UNI_SEC_PER_IMAGE
        ) / n_pass
        table[name] = {
            "n_images": o["n_images"],
            "n_passes": n_pass,
            "pass_split": split,
            "gpu_days_inference": gpu_days(n_pass, o["n_images"], UNI_SEC_PER_IMAGE),
            "gpu_days_inference_measured_split": gpu_days(n_pass, o["n_images"], sec_mixed),
            "p9": {
                "n_crop_normal": pw["n_crop_normal"],
                "n_crop_groups": pw["n_crop_groups"],
                "formal_verdict_possible": pw["formal_verdict_possible"],
                "ci_width_scale_vs_full": pw["ci_width_scale_vs_full"],
            },
            "trivial_bound_all_positive": round(o["bound"], 4),
            "bound_delta_vs_full": round(o["bound"] - bound_full, 4),
        }

    # 오탐 배치 한 번으로 결론을 내지 않는다 — 시드 3개의 범위를 함께 낸다.
    sim_seeds = (SUBSAMPLE_SEED, SUBSAMPLE_SEED + 1, SUBSAMPLE_SEED + 2)
    sim = {"전량": [], "부표본2000": []}
    for rate in (0.02, 0.05):
        for clustered in (False, True):
            for pop_name, rows_ in (("전량", ev), ("부표본2000", sub)):
                runs = [
                    p9_precision_sim(rows_, prov, fp_rate=rate, clustered=clustered, seed=s)
                    for s in sim_seeds
                ]
                hw = sorted(r["diff_half_width"] for r in runs)
                sim[pop_name].append({
                    "fp_rate": rate, "clustered": clustered,
                    "n_seeds": len(sim_seeds),
                    "half_width_median": hw[len(hw) // 2],
                    "half_width_range": [hw[0], hw[-1]],
                    "n_equivalent": sum(1 for r in runs if r["equivalent"]),
                    "n_crop_groups": runs[0]["n_crop_groups"],
                })
            a, b = sim["전량"][-1], sim["부표본2000"][-1]
            print(f"  시뮬 p={rate} 상관{'있음' if clustered else '없음'} · "
                  f"전량 반폭 {a['half_width_median']} 등가 {a['n_equivalent']}/3 · "
                  f"부표본 반폭 {b['half_width_median']} 등가 {b['n_equivalent']}/3",
                  flush=True)

    for name in table:
        key = "부표본2000" if name.startswith("부표본") else "전량"
        table[name]["p9_precision_sim"] = sim[key]

    payload = {
        "basis": {
            "uni_seconds_per_image": UNI_SEC_PER_IMAGE,
            "judge_seconds_per_image": JUDGE_SEC_PER_IMAGE,
            "judge_note": (
                "71번 과제 4 실측. **`verdict_mode: clause_only` 를 유지하면 검색 적중이 "
                "0이라 ⑤ 생성 비용도 0이다** — 이 항목은 conditional 승격 시에만 발생한다"
            ),
            "model_scale_4b": list(MODEL_SCALE),
            "passes_rule": "분리형 5모델 × 시드 + 통합형 2칸 × 1시드 (67번 §2-4)",
            "subsample": {"n_target": SUBSAMPLE_N, "seed": SUBSAMPLE_SEED,
                          "unit": "묶음(group_id)", "strata": "strata_key"},
        },
        "eval_full": {
            "n_images": len(ev),
            "n_groups": len({r["group_id"] for r in ev}),
            **full_power,
        },
        "subsample_realized": {
            "n_images": len(sub),
            "n_groups": len({r["group_id"] for r in sub}),
            "share": round(len(sub) / len(ev), 4),
            **sub_power,
        },
        "table": table,
    }
    dest = out / "scoring_population_cost_v1.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"전량 {len(ev)}장 / 묶음 {len({r['group_id'] for r in ev})} · "
          f"N-crop 정상 {full_power['n_crop_normal']}장 / 묶음 {full_power['n_crop_groups']}")
    print(f"부표본 {len(sub)}장 / 묶음 {len({r['group_id'] for r in sub})} · "
          f"N-crop 정상 {sub_power['n_crop_normal']}장 / 묶음 {sub_power['n_crop_groups']} "
          f"→ 형식 판정 {'성립' if sub_power['formal_verdict_possible'] else '불가'}")
    print(f"자명하한 전량 {bound_full:.4f} · 부표본 {bound_sub:.4f}")
    for k, v in table.items():
        m = v["gpu_days_inference_measured_split"]
        print(f"[{k}] 패스 {v['n_passes']} · 단일단가 0.8B {v['gpu_days_inference']['0.8B']} "
              f"/ 4B {v['gpu_days_inference']['4B_x3']}~{v['gpu_days_inference']['4B_x5']} "
              f"· 실측분리 0.8B {m['0.8B']} / 4B {m['4B_x3']}~{m['4B_x5']}")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
