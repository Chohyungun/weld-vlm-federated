"""본실험 검출 export → 계약 #4 레코드. 채점 앞단의 유일한 칸 분기다.

    uv run python scripts/probe/adapt_main_detections.py \
        --snapshot data/interim/manifest_v1 \
        --pilot outputs/main_c/seed1 --out outputs/main_d/seed1 \
        --seed 20260828 --profile main

C 는 시드·칸 완주마다 `outputs/main_c/seed{k}/predictions/{tag}.detections.jsonl` 을 낸다
(13_spec_D §2-3 · 10번 §4). 파일럿에서는 D 가 직접 추론해 계약 #4 레코드를 만들었지만
(`score_cells.py predict`), 본실험에서는 **C 가 이미 CPU 추론을 끝냈다** — 평가셋 12,461장
× 5칸을 다시 돌리는 것은 같은 수를 17분×5 들여 다시 만드는 일이고, 그 사이 학습이 GPU 를
쓰고 있다.

그래서 이 스크립트가 하는 일은 **형식 변환뿐**이다. 지표는 하나도 계산하지 않고 좌표도
건드리지 않는다. 산출물은 파일럿과 같은 자리·같은 이름이라(`{tag}_s{seed}.jsonl`) 뒤의
채점·스윕·층화 경로가 파일럿과 비트 단위로 같은 코드를 탄다.

**모집단 대조를 여기서 한다.** 평가셋 id 와 레코드 id 가 정확히 같은 집합인지(결측·초과·
중복) 칸마다 확인하고 어긋나면 죽는다. 채점기까지 끌고 가면 "n_scored 가 다르다"로만
드러나고 어느 이미지인지 잃는다.

## 두 벌을 쓴다 — 하한과 운용 임계

C 의 export 는 `conf = 0.01`(하한) 이다. 채점 임계를 C 가 선점하지 않으려는 설계이므로
(13_spec_D §2-3), **임계를 거는 것은 D 의 일이다.** 파일럿에서 `predict` 가 하던 두 갈래를
그대로 재현한다.

    outputs/…/sweep/{tag}_raw_s{seed}.jsonl   하한 0.01 전량 — 스윕 입력
    outputs/…/{tag}_s{seed}.jsonl             운용 임계로 거른 것 — **본채점 입력**

한 벌만 두면 `score_cells_v1.json` 이 `params.conf = 0.25` 를 싣고도 실제로는 0.01 레코드를
채점한 산출물이 된다 — 산출물이 자기 임계에 대해 거짓말을 한다. 감사 D-1 이 지적한 것이
정확히 그 형태(분리형만 임계로 잘린 뒤 통합형과 비교)였다.

하한 1회 추론 + 사후 필터가 임계별 재추론과 같다는 동치는 NMS 의 성질에서 오고
(`evaluation.params.CONF_FLOOR` 주석), 파일럿에서 비트 대조로 실측했다
(`sweep_detection_conf.py --verify-parity`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.adapters import DETECTION_EXPORT_TAGS, adapt_detection_export
from evaluation.cells import load_population
from evaluation.detect_infer import cell_tag, filter_by_conf
from evaluation.params import add_common_args, params_from_args

REPORT_NAME = "adapt_detections_v1.json"


def _write(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(r.model_dump_json() + "\n")
    return sum(len(r.defects) for r in records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--root", default=".")
    ap.add_argument("--at-conf", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="이미 있는 계약 #4 레코드를 다시 만든다(기본은 건너뛴다)")
    args = ap.parse_args()

    params = params_from_args(args)
    params.out.mkdir(parents=True, exist_ok=True)
    pop = load_population(params)
    lm = load_label_map()
    src_dir = params.pilot / "predictions"
    print(f"평가셋 {pop.n_eval:,}장 (정상 {pop.n_normal:,}) · 입력 {src_dir}")
    print(f"클래스 {list(params.class_names)} → {[lm.iso_code(n) for n in params.class_names]}")

    conf = params.conf.value
    print(f"운용 임계 {conf} ({params.conf.source}) · 하한 {params.conf_floor}")

    cells: dict[str, dict] = {}
    for src_tag, cell, client in DETECTION_EXPORT_TAGS:
        src = src_dir / f"{src_tag}.detections.jsonl"
        tag = cell_tag(cell, client)
        dest = params.out / f"{tag}_s{params.seed}.jsonl"
        raw_dest = params.out / "sweep" / f"{tag}_raw_s{params.seed}.jsonl"
        if not src.exists():
            raise SystemExit(f"{src_tag}: export 없음 {src} — C 의 내보내기가 끝나지 않았다")
        if dest.exists() and raw_dest.exists() and not args.overwrite:
            print(f"[{tag}] 존재 — 건너뜀 ({dest.name})")
            cells[tag] = {"skipped": True, "dest": str(dest), "raw_dest": str(raw_dest)}
            continue

        rep = adapt_detection_export(
            src.read_text(encoding="utf-8").splitlines(),
            cell=cell, client=client, seed=params.seed,
            class_names=params.class_names, iso_code_of=lm.iso_code,
            image_size=pop.sizes,
        )
        ids = [r.image_id for r in rep.records]
        missing = pop.eval_ids - set(ids)
        extra = set(ids) - pop.eval_ids
        dup = len(ids) - len(set(ids))
        if missing or extra or dup:
            raise SystemExit(
                f"{tag}: 평가셋 불일치 — 결측 {len(missing)} 초과 {len(extra)} 중복 {dup}. "
                f"예: 결측 {sorted(missing)[:3]} 초과 {sorted(extra)[:3]}"
            )
        spaces = sorted({r.coord_space for r in rep.records})
        scores = [d.score for r in rep.records for d in r.defects if d.score is not None]
        n_raw = _write(raw_dest, rep.records)                     # 하한 전량 — 스윕 입력
        at_conf = filter_by_conf(rep.records, conf)
        n_at_conf = _write(dest, at_conf)                         # 운용 임계 — 본채점 입력

        meta_path = src_dir / f"{src_tag}.detections.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        cells[tag] = {
            "source": str(src), "source_tag": src_tag,
            "dest": str(dest), "raw_dest": str(raw_dest),
            "cell": cell, "client": client,
            "coord_space_values": spaces,
            "conf_operating": conf, "conf_floor": params.conf_floor,
            "n_boxes_floor": n_raw, "n_boxes_at_conf": n_at_conf,
            "min_score_seen": min(scores) if scores else None,
            "max_score_seen": max(scores) if scores else None,
            "export_meta": meta,
            **rep.as_dict(),
        }
        print(f"[{tag}] {len(rep.records):,}레코드 · 하한 {n_raw:,}박스 → 임계 {conf} "
              f"{n_at_conf:,}박스 ({n_at_conf / n_raw:.1%}) "
              f"(경계이탈 {rep.n_boxes_out_of_bounds:,} · 버린 항목 {rep.n_bad_items:,} · "
              f"클래스 밖 {rep.n_unknown_code:,}) · coord_space {spaces}")

    payload = {
        "params": params.as_dict(),
        "source_dir": str(src_dir),
        "n_eval": pop.n_eval,
        "n_normal": pop.n_normal,
        "class_names": list(params.class_names),
        "iso_codes": [lm.iso_code(n) for n in params.class_names],
        "tag_map": {s: {"cell": c, "client": cl} for s, c, cl in DETECTION_EXPORT_TAGS},
        "conf_operating": conf, "conf_operating_source": params.conf.source,
        "conf_floor": params.conf_floor,
        "cells": cells,
        "note": (
            "형식 변환만 한다 — 좌표·coord_space 를 고치지 않는다. 파일이 말하는 규약을 "
            "그대로 레코드에 실어 coord_space_contract 게이트가 판정하게 둔다. "
            "임계는 D 가 건다: 하한 전량은 sweep/, 운용 임계로 거른 것이 본채점 입력이다"
        ),
    }
    dest = params.out / REPORT_NAME
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
