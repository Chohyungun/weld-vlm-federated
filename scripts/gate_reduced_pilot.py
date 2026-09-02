"""§4-6 축소 파일럿 게이트 — 본실험 착수 전 필수 (80번 체크리스트 21항).

## 왜 이 게이트가 있는가

40번 §4-6 이 measure/feasibility 관점에서 남긴 조건이다. **다섯 칸이 전부 자명하한
근처로 붕괴하는 사태를 본실험 전에 잡는다.** 1~2 GPU일(9주 예산의 약 2%)로 그것을
사는 거래다.

79번 C2 가 이 게이트에 대해 적은 것: "필수 게이트가 불합격인지 미실행인지조차 기록이
없다." 파일럿 채점(66번)은 노출비 0.041 = 명세의 1/24 조건에서 나온 값이라 §4-6 의
판정으로 쓸 수 없었다. 그래서 **판정 자체가 없는 상태**였고, 이 스크립트가 그것을 만든다.

## 명세 조건 — 그대로 따른다

| 항목 | 명세(40번 §4-6) | 이 실행 |
|---|---|---|
| 칸 | 분리·중앙 1시드 | `train_round` 단일 런 |
| 표본 | 학습 풀 20% 서브샘플 | 층화 묶음 표집 (아래) |
| epoch | 1/3 | 100 → **33** |
| 프로파일 | 본실험 | `profile="main"` (YOLO11s · 640 · batch 32) |

**`fraction` 을 쓰지 않는다.** Ultralytics 의 `fraction` 은 무작위 표집이 아니라 목록
앞부분을 자른다. 70번 §1-4 에서 실측했듯 그렇게 자르면 배경 이미지 비율이 68.6% 대
전체 49.8% 로 갈린다. 게이트의 통과 기준 1 이 **Macro-F1 대 자명하한** 비교라서 클래스
구성이 틀어지면 판정 자체가 무의미해진다. 그래서 `strata_key`(재질×클래스) 층화로
**묶음 단위** 비례 배분해 뽑는다 — 선택 로직은 `scripts/make_pilot_subset.select_groups`
를 그대로 재사용한다(중복 구현하지 않는다).

## 통과 기준 3개 동시 충족 (판정은 `score` 단계에서)

1. 4결함 Macro-F1 이 **0.2081 을 유의하게 초과** (묶음 단위 클러스터 부트스트랩 CI 하한)
2. **FAR@정상 < 0.5** (정상 이미지에서 결함을 1건 이상 주장한 비율)
3. `|FPR(크롭 출신 정상) − FPR(타일 출신 정상)|` — 이 시점에서는 **판정하지 않고 값만 기록**

미달 시 본실험 착수 중지 + 총괄 판단 요청(40번 §4-9 차선안 분기).

## 사용

    uv run python scripts/gate_reduced_pilot.py view     # 20% 층화 뷰 생성
    uv run python scripts/gate_reduced_pilot.py train    # 33 epoch (장시간 — 분리 실행)
    uv run python scripts/gate_reduced_pilot.py score    # 평가셋 채점 → 판정
    uv run python scripts/gate_reduced_pilot.py status   # 진행 확인

산출: `outputs/gate_c/reduced_pilot/`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

SNAPSHOT_DIR = "data/interim/manifest_v1"
OUT = Path("outputs/gate_c/reduced_pilot").resolve()
VIEW = OUT / "view"

MODEL = "yolo11s.pt"          # 본실험 프로파일
PROFILE = "main"
MAIN_EPOCHS = 100             # 본실험 전역 예산
EPOCH_DIVISOR = 3             # 명세 "epoch 1/3"
EPOCHS = MAIN_EPOCHS // EPOCH_DIVISOR      # 33
TRAIN_FRACTION = 0.20         # 명세 "학습 풀 20%"
BASE_SEED = 20260828          # 파일럿·검출 초기 가중치와 같은 상수 (시드 1세트)
RUN_STAMP = "gate46"


def _snapshot():
    from data.manifest_io import load_snapshot

    sn = load_snapshot(SNAPSHOT_DIR)
    txt = (Path(SNAPSHOT_DIR) / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    digest = next(l.split()[-1] for l in txt.splitlines() if "snapshot_digest" in l)
    return sn, digest


def cmd_view() -> None:
    """학습 풀의 20% 를 층화 묶음 표집해 중앙 뷰를 만든다.

    **스냅샷을 새로 쓰지 않는다.** 동결 매니페스트는 읽기 전용이므로(불변조건 1-1·1-2)
    메모리 안에서 manifest 만 걸러 `build_yolo_view` 에 넘긴다.
    """
    import pandas as pd
    from data.manifest_io import split_view
    from detection.dataset_view import build_yolo_view
    from scripts.make_pilot_subset import select_groups

    sn, digest = _snapshot()
    train = split_view(sn.manifest, "train")
    target = int(round(len(train) * TRAIN_FRACTION))
    print(f"스냅샷 {sn.snapshot_id} digest {digest[:8]}… / 학습 풀 {len(train):,}장 "
          f"→ 목표 {target:,}장 ({TRAIN_FRACTION:.0%})", flush=True)

    picked, ledger = select_groups(train, target=target, seed=BASE_SEED)
    keep = train[train["group_id"].isin(picked)]
    print(f"묶음 {len(picked):,}개 / 이미지 {len(keep):,}장 실제 비율 "
          f"{len(keep)/len(train):.4f}", flush=True)

    # 층화가 실제로 보존됐는지 — 게이트의 통과 기준 1 이 여기 걸린다.
    before = (train["strata_key"].value_counts(normalize=True)).rename("full")
    after = (keep["strata_key"].value_counts(normalize=True)).rename("subset")
    comp = pd.concat([before, after], axis=1).fillna(0.0)
    comp["abs_diff"] = (comp["full"] - comp["subset"]).abs()
    print(comp.round(5).to_string(), flush=True)
    max_drift = float(comp["abs_diff"].max())

    # val 은 전체를 유지한다 — 로깅 전용이고 `val: False` 라 학습 스텝을 만들지 않는다.
    sub = pd.concat([keep, split_view(sn.manifest, "val")], ignore_index=True)
    r = build_yolo_view(replace(sn, manifest=sub), out_dir=VIEW, train_client=None)
    print(f"뷰 train {r.n_images['train']:,}장 박스 {r.n_boxes['train']:,} / "
          f"val {r.n_images['val']:,}장 / 배경 {r.n_background:,} / "
          f"geom 제외 {r.n_geom_invalid} → {r.data_yaml}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "view_meta.json").write_text(json.dumps({
        "명세": {"조건": "학습 풀 20% x epoch 1/3, 본실험 프로파일, 분리·중앙 1시드",
                 "출처": "40번 §4-6"},
        "snapshot_id": sn.snapshot_id, "snapshot_digest": digest,
        "train_pool": int(len(train)), "target": target,
        "groups_picked": len(picked), "images_picked": int(len(keep)),
        "realized_fraction": round(len(keep) / len(train), 6),
        "strata_max_abs_drift": round(max_drift, 6),
        "seed": BASE_SEED,
        "n_images": r.n_images, "n_boxes": r.n_boxes,
        "n_background": r.n_background, "n_geom_invalid": r.n_geom_invalid,
        "층별_회계": json.loads(ledger.to_json(orient="records", force_ascii=False)),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {OUT / 'view_meta.json'}")


def cmd_train() -> None:
    """33 epoch 단일 런. 장시간이므로 분리 실행한다."""
    from detection.init_weights import build_initial_weights
    from detection.round_runner import train_round

    data_yaml = VIEW / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"뷰가 없다. 먼저 `view` 를 돌려라: {data_yaml}")
    meta = json.loads((OUT / "view_meta.json").read_text(encoding="utf-8"))

    # 다섯 칸과 같은 초기 가중치에서 출발한다 — 게이트도 같은 출발선이어야 본실험의
    # 예고편이 된다. 캐시는 게이트 전용 경로에 둔다(다섯 칸 산출물을 건드리지 않는다).
    arrays, keys, _ = build_initial_weights(
        pretrained=MODEL, nc=4, seed=BASE_SEED, cache_path=OUT / "initial.npz")

    t0 = time.perf_counter()
    res = train_round(
        data_yaml=data_yaml, model=MODEL,
        total_epochs=EPOCHS, local_epochs=EPOCHS,   # 단일 런 — 라운드가 곧 전체 예산
        round_idx=0, client_idx=0, base_seed=BASE_SEED,
        num_examples=meta["images_picked"],
        weights_in=arrays, canonical_keys=keys,
        project=OUT / "run", profile=PROFILE,
        resume_dir=OUT / "_resume", run_id=RUN_STAMP,
    )
    wall = time.perf_counter() - t0

    # **가중치를 여기서 저장한다.** `save: False` 는 다섯 칸 공통 고정이라 Ultralytics 가
    # 아무것도 안 남긴다(best 체크포인트 금지의 이행). 채점이 읽는 것은 `RoundResult` 를
    # 떨군 이 npz 다 — 다섯 칸이 전부 같은 규약을 쓴다.
    from detection.train_cell import save_cell_weights

    npz = save_cell_weights(OUT / "weights", "last", res)
    print(f"가중치 저장 → {npz}", flush=True)

    from detection.budget_audit import AccountingCell, AccountingMatrix

    acc = AccountingMatrix(num_rounds=1, client_ids=[0], local_epochs=EPOCHS,
                           total_epochs=EPOCHS)
    acc.record_result(res)
    rep = acc.audit()
    acc.to_csv(OUT / "accounting.csv")
    acc.to_json(OUT / "accounting.json")

    (OUT / "train_result.json").write_text(json.dumps({
        "조건": {"profile": PROFILE, "model": MODEL, "epochs": EPOCHS,
                 "train_images": meta["images_picked"], "seed": BASE_SEED},
        "epochs_ran": res.epochs_ran,
        "optimizer_steps": res.optimizer_steps,
        "optimizer_updates": res.optimizer_updates,
        "peak_vram_gb": round(res.peak_vram_gb, 3),
        "param_l2": res.param_l2_norm,
        "injection_digest": res.injection_digest,
        "effective_optimizer": res.effective_optimizer,
        "stopper_class": res.stopper_class,
        "stopper_true_count": res.stopper_true_count,
        "stopper_calls": len(res.stopper_calls),
        "budget_fired_at": res.budget_fired_at,
        "lr_trace_head": res.lr_trace[:3], "lr_trace_tail": res.lr_trace[-3:],
        "resumed_from_epoch": res.resumed_from_epoch,
        "wall_s": round(wall, 1),
        "gates_evaluated": res.gates_evaluated,
        "gate_results": res.gate_results,
        "회계_ok": rep.ok, "회계_failures": rep.failures, "회계_notes": rep.notes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완주 {wall:.0f}s / {res.epochs_ran}ep / steps {res.optimizer_steps} "
          f"/ peak {res.peak_vram_gb:.2f}GB / 회계 ok={rep.ok}", flush=True)
    if not rep.ok:
        raise SystemExit(f"회계 실패 — 이 런은 게이트 판정에 쓸 수 없다: {rep.failures}")


# --------------------------------------------------------------------------
# 채점·판정
# --------------------------------------------------------------------------

def _load_yolo(npz_path: Path, class_names, imgsz: int):
    """npz 상태를 **본실험 프로파일 모델**(YOLO11s)에 주입한다.

    ## 왜 D 의 `load_yolo_from_npz` 를 그대로 못 쓰는가

    `evaluation/detect_infer.load_yolo_from_npz` 가 `cfg="yolo11n.yaml"` 을 **하드코딩**한다.
    파일럿 프로파일(YOLO11n)만 상정한 코드라 본실험 프로파일(YOLO11s)로 학습한 가중치를
    넣으면 첫 층에서 죽는다 — 실측:

        SerializeError: shape 불일치 [model.0.conv.weight]: (32,3,3,3) != (16,3,3,3)

    **`evaluation/` 는 트랙 D 소관이라 고치지 않았다.** 대신 여기서 같은 절차를 모델
    구성만 바꿔 수행한다. 갈라지는 것은 `cfg` 문자열 하나이고, 가중치 주입은 D 의
    독스트링이 스스로 지정한 대로 C 의 `detection.serialize` 를 그대로 지난다.
    **채점(`score_detection`·`cluster_bootstrap`)과 추론(`predict_cell`)은 전부 D 것을
    그대로 부른다** — 두 번째 채점기를 만들지 않는다.

    본실험 착수 전에 D 가 `load_yolo_from_npz` 에 모델 cfg 를 인자로 열어야 한다.
    지금 상태로는 **다섯 칸 중 어느 것도 본실험 프로파일로 채점할 수 없다.** 82번에 보고.
    """
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    from detection import serialize

    dm = DetectionModel(cfg="yolo11s.yaml", nc=len(class_names), verbose=False)
    keys = serialize.canonical_keys(dm.state_dict())
    z = np.load(npz_path)
    arrays = [z[f"arr_{i}"] for i in range(len(z.files))]
    ref = dm.state_dict()
    serialize.assert_compatible(arrays, keys, ref)
    dm.load_state_dict(serialize.ndarrays_to_state_dict(arrays, keys, ref), strict=True)
    dm.names = dict(enumerate(class_names))
    tmp = Path(npz_path).with_suffix(".tmp.pt")
    # fp32 그대로 저장한다 — `.half()` 사본을 만들지 않는다(D17: 예측 산출과 채점이
    # 모델 정밀도에 합의해야 한다).
    torch.save({"model": dm.float(), "train_args": {"imgsz": imgsz}}, tmp)
    yolo = YOLO(str(tmp))
    tmp.unlink(missing_ok=True)
    return yolo



#: 통과 기준 1 의 비교선. 40번 §4-6 이 사전 등록한 값(4결함 전량양성 자명하한)이다.
TRIVIAL_LINE = 0.2081
#: 통과 기준 2.
FAR_MAX = 0.5

#: 한 번에 `predict_cell` 에 넘기는 이미지 수. 전량(12,461)을 한 번에 주면 전처리가
#: 한 텐서로 뭉쳐 36.7GB 를 할당하려다 죽는다. 256장이면 약 0.8GB 다.
PREDICT_CHUNK = 256


def cmd_score() -> None:
    """평가셋 전량 추론 → 통과 기준 3종 판정.

    **채점 코드를 새로 짜지 않는다.** `evaluation/`(트랙 D 소유)의 `predict_cell`·
    `score_detection`·`cluster_bootstrap` 을 그대로 부른다. 게이트가 다섯 칸과 다른
    채점기를 쓰면 그 게이트가 무엇을 판정한 것인지 알 수 없다.
    """
    import numpy as np

    from evaluation.detect_infer import predict_cell
    from evaluation.eval_set import eval_rows, read_gold, read_manifest
    from evaluation.metrics.detection import score_detection
    from evaluation.params import ScoringParams
    from evaluation.stats import cluster_bootstrap
    from fl.run_gates import apply_run_gates

    last = OUT / "weights" / "last.npz"
    if not last.exists():
        raise SystemExit(f"가중치가 없다: {last} — 먼저 `train` 을 돌려라")

    gates = apply_run_gates(cell="gate46_score")

    # **imgsz 는 학습과 같아야 한다.** `ScoringParams.IMGSZ` 기본값은 416(파일럿
    # 프로파일)인데 이 게이트는 본실험 프로파일 640 으로 학습했다. 다른 해상도로 추론하면
    # 그 차이가 곧 게이트 판정의 오차가 된다.
    params = ScoringParams(snapshot=Path(SNAPSHOT_DIR), pilot=OUT, out=OUT,
                           seed=BASE_SEED, imgsz=640)
    # `params.class_names` 는 표시명('crack'…)이고 채점기가 비교하는 것은 ISO 코드다
    # (`predict_cell` 도 `lm.iso_code` 로 쓴다). `evaluation/cells.py:83` 과 같은 변환.
    from data.label_map import load_label_map

    lm = load_label_map()
    classes = [lm.iso_code(n) for n in params.class_names]
    rows_all = read_manifest(SNAPSHOT_DIR)
    rows = eval_rows(rows_all)
    eval_ids = {r["image_id"] for r in rows}
    gold, _gold_boxes = read_gold(SNAPSHOT_DIR, eval_ids)
    print(f"평가셋 {len(rows):,}장 / 정답 라벨 {len(gold):,}건", flush=True)

    # 추론 결과를 먼저 디스크에 남긴다. 12,461장 추론이 17분인데 하류 한 줄이 틀리면
    # 그걸 다시 치르게 된다 — 실제로 한 번 그랬다. 재실행 시 캐시를 읽는다.
    cache = OUT / "predictions.jsonl"
    if cache.exists():
        from evaluation.schema import PredictionRecord

        preds = [PredictionRecord.model_validate_json(l)
                 for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"예측 캐시 재사용 {len(preds):,}건 → {cache}", flush=True)
        return _verdict_from(preds, rows, gold, eval_ids, params, classes, gates)

    yolo = _load_yolo(last, params.class_names, params.imgsz)
    t0 = time.perf_counter()
    # **묶어서 부른다.** `predict_cell` 은 받은 행 전부의 경로를 한 번에 Ultralytics 에
    # 넘기는데, 파일럿 평가셋 653장에서는 문제가 없다가 본실험 평가셋 12,461장에서
    # 전처리가 통째로 한 텐서가 되어 **36.7GB 할당 시도로 죽는다**(실측).
    #
    #     RuntimeError: DefaultCPUAllocator: not enough memory:
    #                   you tried to allocate 36748984320 bytes
    #
    # `evaluation/` 는 D 소관이라 함수를 고치지 않고 **같은 함수를 나눠 부른다** —
    # 추론 경로는 여전히 하나다. D 가 본실험 전에 내부 청킹을 넣어야 한다(82번 보고).
    preds = []
    for i in range(0, len(rows), PREDICT_CHUNK):
        part = rows[i:i + PREDICT_CHUNK]
        # 칸 이름은 계약 #4 의 다섯 리터럴 중 하나여야 한다(스키마가 거부한다).
        # §4-6 은 "분리·중앙 1시드" 이므로 이 런의 칸은 실제로 sep_central 이다 —
        # 게이트라는 신원은 산출 경로와 verdict.json 이 들고 있다.
        preds.extend(predict_cell(yolo, part, Path.cwd(), "sep_central", None,
                                  params, conf=params.conf.value))
        if (i // PREDICT_CHUNK) % 10 == 0:
            print(f"  추론 {len(preds):,}/{len(rows):,} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"추론 {len(preds):,}건 / {time.perf_counter()-t0:.0f}s "
          f"(conf {params.conf.value}, 출처 {params.conf.source})", flush=True)
    with cache.open("w", encoding="utf-8") as fh:
        for r in preds:
            fh.write(r.model_dump_json() + "\n")
    print(f"예측 저장 → {cache}", flush=True)
    return _verdict_from(preds, rows, gold, eval_ids, params, classes, gates)


def _verdict_from(preds, rows, gold, eval_ids, params, classes, gates) -> None:
    """예측에서 통과 기준 3종을 판정한다. 추론과 분리해 두어 재실행이 싸다."""
    import numpy as np

    from evaluation.metrics.detection import score_detection
    from evaluation.stats import cluster_bootstrap

    by_id = {r["image_id"]: r for r in rows}
    # `PredictionRecord.defects` 는 pydantic `Defect` 객체다(dict 아님).
    # D 의 `evaluation/score.py:32` 와 같은 추출을 쓴다.
    pred_codes = {p.image_id: sorted(p.iso_codes) for p in preds}
    # D9 대응 — 정상 이미지도 모집단에 남는다. `gold` 는 결함 있는 이미지만 키를 갖는
    # 경우가 있으므로 평가셋 전량으로 back-fill 한다(빈 집합 = 정상).
    gold_codes = {i: sorted(gold.get(i, set())) for i in sorted(eval_ids)}
    assert len(gold_codes) == len(eval_ids), "모집단이 평가셋 전량이어야 한다"
    group_of = {r["image_id"]: r.get("group_id", r["image_id"]) for r in rows}
    ids_by_group: dict[str, list[str]] = {}
    for i, g in group_of.items():
        ids_by_group.setdefault(g, []).append(i)

    rep = score_detection(pred_codes, gold_codes, classes)
    macro = float(rep.macro_f1)

    # 기준 1 — 묶음 단위 클러스터 부트스트랩. 이미지 단위로 재표집하면 CI 가 좁아져
    # "유의하게 초과"가 부풀려진다(불변조건 1-5 와 같은 이유다).
    def macro_of(groups):
        ids = [i for g in groups for i in ids_by_group[g]]
        return float(score_detection({i: pred_codes.get(i, []) for i in ids},
                                     {i: gold_codes.get(i, []) for i in ids},
                                     classes).macro_f1)

    ci = cluster_bootstrap(sorted(ids_by_group), macro_of, seed=BASE_SEED)

    # 기준 2 — 정상 이미지에서 결함을 1건 이상 주장한 비율
    normals = [i for i in gold_codes if not gold_codes[i]]
    far = (sum(1 for i in normals if pred_codes.get(i)) / len(normals)) if normals else float("nan")

    # 기준 3 — 출처별 FPR 격차. **값만 기록한다**(명세가 이 시점 판정을 금지한다).
    #
    # 축은 `manifest.source`(데이터셋 출처, 전부 aihub71761)가 아니라 **`tiles.provenance`**
    # (N-crop / N-tile / N-band)다. 40번이 말한 "크롭 출신 정상 대 타일 출신 정상"이
    # 이 축이고, 함정표 #11(결함/정상 이미지 규격 지름길)이 겨냥하는 것도 여기다.
    from data.manifest_io import load_snapshot

    tiles = load_snapshot(SNAPSHOT_DIR).tiles
    prov = ({str(r.image_id): str(r.provenance) for r in tiles.itertuples()}
            if tiles is not None else {})

    src_groups: dict[str, list[str]] = {}
    for i in normals:
        src_groups.setdefault(prov.get(i, "unknown"), []).append(i)
    fpr_by_src = {k: round(sum(1 for i in v if pred_codes.get(i)) / len(v), 6)
                  for k, v in sorted(src_groups.items()) if v}
    n_by_src = {k: len(v) for k, v in sorted(src_groups.items())}
    crop = [k for k in fpr_by_src if "crop" in k.lower()]
    tile = [k for k in fpr_by_src if "tile" in k.lower()]
    gap = (abs(fpr_by_src[crop[0]] - fpr_by_src[tile[0]])
           if crop and tile else None)

    c1 = bool(ci.lo > TRIVIAL_LINE)
    c2 = bool(far < FAR_MAX)
    verdict = "PASS" if (c1 and c2) else "FAIL"

    out = {
        "판정": verdict,
        "명세": {"출처": "40번 §4-6", "조건": "분리·중앙 1시드 · 학습 풀 20% · epoch 1/3 · 본실험 프로파일"},
        "실행조건_실측": {
            "train_images": json.loads((OUT / "view_meta.json").read_text(encoding="utf-8"))["images_picked"],
            "epochs": EPOCHS, "profile": PROFILE, "model": MODEL, "seeds": 1,
            "eval_images": len(rows), "conf": params.conf.value,
            "conf_source": params.conf.source,
            "imgsz": params.imgsz, "classes_iso": classes,
            "device": params.device, "max_det": params.max_det,
        },
        "기준1_macro_f1": {
            "값": round(macro, 6), "CI_하한": round(ci.lo, 6), "CI_상한": round(ci.hi, 6),
            "비교선": TRIVIAL_LINE, "묶음_수": len(ids_by_group),
            "통과": c1,
            "설명": "묶음 단위 클러스터 부트스트랩 CI 하한이 자명하한을 넘어야 한다",
        },
        "기준2_far_at_normal": {
            "값": round(far, 6), "상한": FAR_MAX, "정상_이미지": len(normals), "통과": c2,
        },
        "기준3_출처별_FPR": {
            "축": "tiles.provenance (N-crop / N-tile / N-band)",
            "출처별_FPR": fpr_by_src, "출처별_정상장수": n_by_src, "격차": gap,
            "설명": "이 시점에서는 판정하지 않고 값만 기록한다(명세 §4-6 기준 3). "
                    "함정표 #11 이 겨냥하는 축이다.",
        },
        "클래스별": [
            {"iso_code": c.iso_code, "support": c.support, "tp": c.tp, "fp": c.fp,
             "fn": c.fn, "f1": round(c.f1, 6)} for c in rep.per_class
        ],
        "skipped_classes": list(rep.skipped_classes),
        **gates,
        "미달_시_규칙": "본실험 착수를 중지하고 총괄 판단 요청. 40번 §4-9 차선안으로 분기한다.",
    }
    (OUT / "verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("판정", "기준1_macro_f1", "기준2_far_at_normal",
                                          "기준3_출처별_FPR")}, ensure_ascii=False, indent=1))
    print(f"\n→ {OUT / 'verdict.json'}")


def cmd_audit() -> None:
    """디스크의 `accounting.csv` 를 그대로 다시 감사한다.

    **재계산이 아니라 재판정이다.** 셀의 값은 학습이 기록한 실측 그대로 읽고, 검사만
    다시 건다. §4-6 첫 판정에서 검사 (4)가 재개 런을 오판했다 — `stopper_calls`(이 프로세스가
    돈 2회)를 `epochs_ran`(라운드 누적 33회)과 비교했다. 값이 틀린 게 아니라 검사가
    틀렸으므로 학습을 다시 돌리지 않고 같은 기록에 고친 검사를 적용한다.

    상수로 채우는 필드가 없다 — 전부 CSV 에 있던 값이라 `value_source` 는 그대로다.
    """
    import csv

    from detection.budget_audit import AccountingCell, AccountingMatrix

    path = OUT / "accounting.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"회계가 비었다: {path}")

    def _opt(v, cast):
        return None if v in ("", None) else cast(v)

    acc = AccountingMatrix(num_rounds=1, client_ids=[0], local_epochs=EPOCHS,
                           total_epochs=EPOCHS)
    for r in rows:
        acc.record(AccountingCell(
            round_idx=int(r["round_idx"]), client_idx=int(r["client_idx"]),
            epochs_ran=int(r["epochs_ran"]), optimizer_steps=int(r["optimizer_steps"]),
            num_examples=int(r["num_examples"]), seed=int(r["seed"]),
            param_l2_norm=float(r["param_l2_norm"]), payload_bytes=int(r["payload_bytes"]),
            optimizer=r["optimizer"], lr=float(r["lr"]),
            arg_optimizer=r["arg_optimizer"], arg_lr0=float(r["arg_lr0"]),
            budget_fired_at=_opt(r["budget_fired_at"], int),
            stopper_class=r["stopper_class"],
            stopper_true_count=_opt(r["stopper_true_count"], int),
            stopper_calls=_opt(r["stopper_calls"], int),
            optimizer_updates=int(r["optimizer_updates"]),
            resumed_from_epoch=_opt(r["resumed_from_epoch"], int),
            value_source=r["value_source"],
            fedavg_weight=_opt(r["fedavg_weight"], float),
            fedavg_weight_unit=r["fedavg_weight_unit"],
            supervised_tokens=int(r["supervised_tokens"] or 0),
        ))
    rep = acc.audit()
    acc.to_json(OUT / "accounting.json")
    print(json.dumps({
        "ok": rep.ok, "failures": rep.failures, "notes": rep.notes,
        "resumed_cells": rep.resumed_cells,
        "total_optimizer_steps": rep.total_optimizer_steps,
        "total_epochs_by_client": rep.total_epochs_by_client,
    }, ensure_ascii=False, indent=2))
    if not rep.ok:
        raise SystemExit("회계 재판정도 실패다 — 검사가 아니라 런에 문제가 있다.")


def cmd_status() -> None:
    for name in ("view_meta.json", "train_result.json", "verdict.json"):
        p = OUT / name
        print(f"{'O' if p.exists() else 'X'} {p}")
    csv = OUT / "run" / "r000_c0" / "results.csv"
    if csv.exists():
        lines = csv.read_text(encoding="utf-8").splitlines()
        print(f"  results.csv {len(lines)-1} epoch 기록 / 목표 {EPOCHS}")
        if len(lines) > 1:
            print(f"  마지막: {lines[-1][:120]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["view", "train", "score", "audit", "status"])
    {"view": cmd_view, "train": cmd_train, "score": cmd_score,
     "audit": cmd_audit, "status": cmd_status}[ap.parse_args().cmd]()
