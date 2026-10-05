"""경계 규약이 **이미 낸 수치를 건드리는가** — 저장된 예측과 정답만 읽어서 센다.

M1 의 경계 판정을 이진(binary64) 규약으로 못 박았다(`evaluation/metrics/box_f1.py`).
분리형 사전실험 세 시드는 **pycocotools COCOeval** 이 채점했고 그쪽 규약은 또 다르다.
규약을 채택하는 순간 이미 공개한 수치가 움직일 수 있는지 먼저 재야 한다.

## 세 규약

| 이름 | 판정식 | 어디 |
|---|---|---|
| `float_ratio` | `inter / union >= 0.5` (float 나눗셈) | **이 스크립트가 다시 쓴 식**이다. pycocotools 의 판정이 아니다 — 아래 |
| `binary` | `2·inter >= union` (배정밀도 값의 정확한 유리수) | `box_f1.is_candidate` — **새 정본** |
| `decimal` | 같은 식, 단 **기록된 십진 문자열**의 유리수 | 채택하지 않은 대안 |

셋이 갈리는 쌍이 하나도 없으면 규약을 바꿔도 매칭이 바뀔 수 없다. 있으면 몇 개인지가 답이다.

**`float_ratio` 는 pycocotools 가 아니다.** pycocotools 는 상자를 `[x, y, w, h]` 로 옮긴 뒤 컴파일된 `maskUtils.iou`
로 IoU 를 내고 `ious < iou` 로 비교한다. 좌상·우하 좌표로 다시 쓴 이 식과 연산 순서가 같다는 근거가 없고,
경계 근처에서는 바로 그 차이가 판정을 가른다. 그래서 `float_ratio≠binary` 는 "이 재현식과 이진 규약의 차이" 이지
"pycocotools 와 새 규약의 차이" 가 아니다. 앞 판은 이 열을 `coco` 라 부르고 "pycocotools 가 쓴 것" 이라 적었다.

## 빠진 입력은 멈춘다

예측 파일이 하나라도 없으면 멈춘다. 앞 판은 한 줄 찍고 건너뛰어, 다섯 태그가 전부 빠져도 합계가 `{}` 로 나가
"갈림 없음" 으로 읽혔다. 일부만 보려면 `--allow-missing` 을 주고, 그때 산출이 빠진 입력의 목록과
기대 입력 수·읽은 수를 싣는다.

## 재지 않는 것

**mAP 를 다시 계산하지 않는다.** 이 스크립트는 후보 판정이 갈리는 쌍의 수만 센다. 실제 mAP 이동은
매칭 알고리즘(점수 정렬 탐욕)과 누적 규칙을 거쳐야 나오므로, 여기 수는 **영향받을 수 있는 쌍의
상한**이다. 0 이면 상한이 0 이므로 이동도 0 이다.

정답은 동결 스냅샷의 주석을 스트리밍으로 읽는다. 채점기가 세는 지지량(22,549)과 여기 계수(22,565)는
무시 표시 때문에 다를 수 있다 — 그 차이는 갈림 판정에 영향을 주지 않으므로 그대로 둔다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.params import CLASS_NAMES
from evaluation.provenance import write_new_text

TAGS = ("sep_central", "sep_fed", "sep_local_C1", "sep_local_C2", "sep_local_C3")
SCORING = tuple(load_label_map().iso_code(n) for n in CLASS_NAMES)
"""채점 4클래스 — 코드를 적지 않고 사상표에서 읽는다(불변조건 1-8). 채점기와 같은 규칙이다."""
REL = 1e-9

CONVENTIONS = {
    "float_ratio": ("inter/union >= 0.5 (float 나눗셈) — 이 스크립트가 좌상·우하 좌표로 다시 쓴 식. "
                    "pycocotools 의 판정이 아니다(그쪽은 [x,y,w,h] 와 컴파일된 maskUtils.iou)"),
    "binary": "2·inter >= union (배정밀도 값의 유리수) — 새 정본",
    "decimal": "2·inter >= union (기록된 십진의 유리수) — 미채택 대안",
}
"""산출에 싣는 규약 설명. 첫째 열이 재현식이라는 것을 이름과 문장 둘 다에 적는다."""

WHY = ("공개된 map_50 은 pycocotools COCOeval 이 냈다 — 이미 낸 값이라 새 규약이 그 값을 바꾸지 않는다. "
       "지금 어느 채점 경로가 새 규약을 부르는지는 이 스크립트가 재지 않는다")
"""이동이 0 인 까닭. **실행 때의 코드 상태를 단언하지 않는다** — 진입점이 생기면 조용히 거짓이 되는 문장을 싣지 않는다."""

LOADED: dict = {}
"""정답을 읽으며 남긴 계수. 산출물에 그대로 싣는다."""


def _pairs(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    return iw, ih


def judge(a, b) -> tuple[bool, bool, bool, bool]:
    """`(float_ratio, binary, decimal, 경계근처)`. 겹침이 없으면 넷 다 거짓이다.

    첫째 값은 **이 스크립트의 재현식**이다 — pycocotools 의 판정이 아니다(모듈 머리말).
    """
    iw, ih = _pairs(a, b)
    if iw <= 0 or ih <= 0:
        return False, False, False, False
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if union <= 0:
        return False, False, False, False
    coco = (inter / union) >= 0.5
    slack = 2 * inter - union
    near = abs(slack) <= REL * max(1.0, union)

    fa = [Fraction(v) for v in a]
    fb = [Fraction(v) for v in b]
    fiw = min(fa[2], fb[2]) - max(fa[0], fb[0])
    fih = min(fa[3], fb[3]) - max(fa[1], fb[1])
    fi = fiw * fih if fiw > 0 and fih > 0 else Fraction(0)
    fu = (fa[2] - fa[0]) * (fa[3] - fa[1]) + (fb[2] - fb[0]) * (fb[3] - fb[1]) - fi
    binary = fu > 0 and 2 * fi >= fu

    da = [Fraction(Decimal(repr(v))) for v in a]
    db = [Fraction(Decimal(repr(v))) for v in b]
    diw = min(da[2], db[2]) - max(da[0], db[0])
    dih = min(da[3], db[3]) - max(da[1], db[1])
    di = diw * dih if diw > 0 and dih > 0 else Fraction(0)
    du = (da[2] - da[0]) * (da[3] - da[1]) + (db[2] - db[0]) * (db[3] - db[1]) - di
    decimal = du > 0 and 2 * di >= du
    return coco, binary, decimal, near


def load_gold(snapshot: Path) -> dict[str, dict[str, list]]:
    eval_ids = set()
    with open(snapshot / "manifest.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "eval":
                eval_ids.add(r["image_id"])
    gold: dict[str, dict[str, list]] = {}
    n = n_empty = 0
    cols = ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
    with open(snapshot / "annotations.csv", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["image_id"] not in eval_ids or r["iso_code"] not in SCORING:
                continue
            if any(r[c] == "" for c in cols):
                # 박스가 비어 있는 주석이 있다. 채점기도 이런 행을 셀 수 없으므로 건너뛰고
                # 수를 남긴다 — 매니페스트 계수와 채점 지지량이 갈리는 자리로 보인다.
                n_empty += 1
                continue
            box = tuple(float(r[c]) for c in cols)
            gold.setdefault(r["image_id"], {}).setdefault(r["iso_code"], []).append(box)
            n += 1
    print(f"  정답 {n}박스 · 결함 이미지 {len(gold)}장 · 빈 박스 {n_empty}건 제외", flush=True)
    LOADED.update({"n_gold_boxes": n, "n_gold_images": len(gold), "n_empty_bbox": n_empty})
    return gold


def audit_file(path: Path, gold: dict, log) -> dict:
    """한 모델의 예측을 스트리밍하며 같은 클래스 쌍을 전부 맞댄다."""
    n_pairs = n_overlap = 0
    near = Counter()
    disagree = Counter()
    examples: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for ln, raw in enumerate(fh, 1):
            row = json.loads(raw)
            g_by_class = gold.get(row["image_id"])
            if not g_by_class:
                continue
            p_by_class: dict[str, list] = {}
            for d in row.get("defects") or ():
                if d.get("bbox_px") and d["iso_code"] in g_by_class:
                    p_by_class.setdefault(d["iso_code"], []).append(tuple(d["bbox_px"]))
            for code, gs in g_by_class.items():
                for g in gs:
                    for p in p_by_class.get(code, ()):
                        n_pairs += 1
                        c, b, d, nr = judge(g, p)
                        if c or b or d:
                            n_overlap += 1
                        if nr:
                            near["경계근처"] += 1
                        if c != b:
                            disagree["float_ratio≠binary"] += 1
                        if b != d:
                            disagree["binary≠decimal"] += 1
                        if c != d:
                            disagree["float_ratio≠decimal"] += 1
                        if (c != b or b != d) and len(examples) < 10:
                            examples.append({"line": ln, "iso_code": code,
                                             "gold": list(g), "pred": list(p),
                                             "float_ratio": c, "binary": b, "decimal": d,
                                             "near_boundary": nr})
    log(f"  [{path.stem}] 쌍 {n_pairs:,} · 후보 {n_overlap:,} · "
        f"경계근처 {near['경계근처']} · 갈림 {sum(disagree.values())}")
    return {"n_pairs": n_pairs, "n_candidate_any": n_overlap,
            "n_near_boundary": int(near["경계근처"]),
            "disagree": dict(disagree), "examples": examples}


def input_paths(root: Path, seeds, values) -> list[tuple[str, str, Path]]:
    """읽어야 할 예측 파일 전량 — `(시드 번호, 태그, 경로)`."""
    if len(values) != len(seeds):
        raise SystemExit("시드 값의 수가 시드 수와 다르다")
    return [(str(n), tag, root / f"seed{n}" / "sweep" / f"{tag}_raw_s{sv}.jsonl")
            for n, sv in zip(seeds, values, strict=True) for tag in TAGS]


def audit(root: Path, snapshot: Path, seeds, values, *, allow_missing: bool, log=print) -> dict:
    """감사 본체. **빠진 입력을 0 뒤에 숨기지 않는다** — 없으면 멈추고, 허용하면 목록을 싣는다."""
    wanted = input_paths(root, seeds, values)
    missing = [f"seed{n}/{tag}" for n, tag, path in wanted if not path.exists()]
    if missing and not allow_missing:
        raise SystemExit(f"예측 파일 {len(missing)}/{len(wanted)} 개가 없다 — {missing[:5]}. "
                         "일부만 보려면 --allow-missing")
    gold = load_gold(snapshot)
    out: dict = {str(n): {} for n in seeds}
    for n, tag, path in wanted:
        if path.exists():
            out[n][tag] = audit_file(path, gold, log)
    return {"per_seed": out,
            "inputs": {"expected": len(wanted), "read": len(wanted) - len(missing),
                       "missing": missing, "complete": not missing}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="outputs/main_d", type=Path)
    ap.add_argument("--snapshot", default="data/interim/manifest_v1", type=Path)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--seed-values", default="20260828,20260829,20260830",
                    help="시드 번호 순서의 파일명 시드 값. 시드마다 다르다")
    ap.add_argument("--allow-missing", action="store_true",
                    help="예측 파일이 일부 없어도 돈다. 산출이 빠진 목록을 싣고 complete=false 다")
    ap.add_argument("--dest", default="outputs/main_d/seed3set/boundary_convention_audit_v1.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    seeds = [int(x) for x in args.seeds.split(",")]
    got = audit(REPO / args.root, REPO / args.snapshot, seeds, args.seed_values.split(","),
                allow_missing=args.allow_missing)
    out = got["per_seed"]

    total = sum(t["n_pairs"] for s in out.values() for t in s.values())
    near = sum(t["n_near_boundary"] for s in out.values() for t in s.values())
    dis = Counter()
    for s in out.values():
        for t in s.values():
            dis.update(t["disagree"])
    payload = {
        "kind": "boundary_convention_audit",
        "question": "새 경계 규약이 이미 낸 분리형 세 시드 수치를 움직이는가",
        "conventions": CONVENTIONS,
        "applies_to_published_numbers": False,
        "why": WHY,
        "inputs": got["inputs"],
        "gold": dict(LOADED),
        "totals": {"n_pairs": total, "n_near_boundary": near, "disagree": dict(dis),
                   "complete": got["inputs"]["complete"]},
        "map_50_movement": ("0. 공개된 값은 이미 pycocotools 가 냈다. 아래 갈림 수는 "
                            "'적용했다면 영향받을 수 있는 쌍의 상한' 이지 실제 이동이 아니다"),
        "caveat": ("쌍 단위 후보 판정만 맞댔다. 실제 매칭은 점수 정렬 탐욕과 누적 규칙을 "
                   "거치므로 갈림 쌍이 그대로 매칭 변화가 되지는 않는다. float_ratio 열은 pycocotools 의 "
                   "판정이 아니라 재현식이다"),
        "per_seed": out,
    }
    dest = REPO / args.dest
    write_new_text(dest, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"\n{dest.relative_to(REPO)}")
    print(f"총 쌍 {total:,} · 경계근처 {near} · 갈림 {dict(dis)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
