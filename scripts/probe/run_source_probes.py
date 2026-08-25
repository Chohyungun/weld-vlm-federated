"""P2 출처 판별 · P3 셔플·저해상 프로브를 최종 분할 위에서 실행한다.

    uv run python scripts/probe/run_source_probes.py \
        --manifest data/interim/manifest_v1/manifest.csv \
        --provenance data/interim/manifest_v1/encode_progress.jsonl \
        --root . --out outputs/probe/source_v1.json

**학습 풀 내부 홀드아웃에서만 학습·채점한다.** `split == "eval"` 행은 로더가 아예 읽지
않는다(`load_rows`). 홀드아웃은 묶음 단위라 같은 용접부가 양쪽에 갈리지 않는다.

**last 채점.** 3 epoch 을 끝까지 돌리고 마지막 상태로 잰다. best 선택은 암묵적 조기 종료다.

조건 3종을 같은 홀드아웃 위에서 돌린다.

| 조건 | 무엇이 남는가 | 게이트 |
|---|---|---|
| `raw` (P2) | 전부 | AUC ≤ 0.60, CI 상한 < 0.65 |
| `shuffle` (P3) | 텍스처만 | AUC ≤ 0.65 |
| `lowres` (P3) | 전역 통계만 | 출처 AUC 참고 + 결함 판별은 별도 |
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.probes.pilot import judge_p2
from evaluation.probes.source_probe import (
    EPOCHS,
    INPUT_SIZE,
    LOWRES_SIZE,
    PATCH_SIZE,
    SEED,
    ProbeRow,
    downscale,
    group_holdout,
    load_provenance,
    load_rows,
    patch_shuffle,
    run_source_probe,
    summarize_sources,
)


class _ProbeDataset(Dataset):
    """모듈 수준 클래스여야 한다. Windows 는 spawn 이라 지역 클래스를 pickle 하지 못하고,
    `num_workers > 0` 에서 로더가 통째로 죽는다."""

    def __init__(self, rows: list[ProbeRow], root: Path, condition: str) -> None:
        self.rows = rows
        self.root = root
        self.condition = condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(self.root / r.rel_path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.uint8)
        if self.condition == "shuffle":
            arr = patch_shuffle(arr, PATCH_SIZE, seed=SEED)
        elif self.condition == "lowres":
            arr = downscale(arr, LOWRES_SIZE)
        t = torch.from_numpy(np.ascontiguousarray(arr).copy()).float().div_(255.0)
        t = t.unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False
        )
        return t.squeeze(0).repeat(3, 1, 1), r.is_tile


def make_scorer(root: Path, condition: str, device: str, batch: int, workers: int):
    """ResNet-18 을 학습해 홀드아웃 점수를 낸다. **last 상태로 채점한다.**"""

    def score(train: list[ProbeRow], test: list[ProbeRow]) -> list[float]:
        torch.manual_seed(SEED)
        model = resnet18(weights=None, num_classes=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        loader = DataLoader(
            _ProbeDataset(list(train), root, condition), batch_size=batch,
            shuffle=True, num_workers=workers, drop_last=False,
        )
        model.train()
        for ep in range(EPOCHS):
            total = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = lossf(model(x), y)
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(y)
            print(f"    [{condition}] epoch {ep + 1}/{EPOCHS} "
                  f"loss {total / len(train):.4f}", flush=True)
        # 조기 종료도 best 선택도 없다. 마지막 상태 그대로 채점한다.
        model.eval()
        out: list[float] = []
        test_loader = DataLoader(
            _ProbeDataset(list(test), root, condition),
            batch_size=batch, num_workers=workers,
        )
        with torch.no_grad():
            for x, _unused in test_loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
                out.extend(p.detach().cpu().tolist())
        return out

    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--provenance", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="outputs/probe/source_v1.json")
    ap.add_argument("--conditions", default="raw,shuffle,lowres")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument(
        "--subset", default="all", choices=["all", "normals"],
        help=(
            "normals 면 정상 이미지만 쓴다. 창원 데이터는 N-tile 이 100%% 정상이고 "
            "N-crop 이 90.7%% 결함이라, 전체로 재면 출처 판별이 결함 판별과 뒤섞인다"
        ),
    )
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = Path(args.root)
    prov = load_provenance(args.provenance)
    rows, unmatched = load_rows(args.manifest, prov)
    if args.subset == "normals":
        rows = [r for r in rows if not r.iso_codes]
        print("정상 이미지만 사용한다 — 출처와 클래스의 교란을 제거하기 위해서다")
    print(f"학습 풀 표본 {len(rows)} · 출처 조인 실패 {len(unmatched)}")
    print(f"출처 분포 {summarize_sources(rows)}")
    if not rows:
        print("검사 대상 0장 — 통과로 처리하지 않는다")
        return 1
    if unmatched:
        print(f"출처를 모르는 이미지가 {len(unmatched)}건이다. 조인을 먼저 고친다")
        return 1

    train, test = group_holdout(rows, holdout_frac=args.holdout_frac)
    leak = {r.group_id for r in train} & {r.group_id for r in test}
    if leak:
        print(f"묶음 누수 {len(leak)}건 — 중단한다")
        return 1
    print(f"홀드아웃 학습 {len(train)} / 채점 {len(test)} "
          f"(묶음 {len({r.group_id for r in test})}개)")

    results, verdicts = [], []
    for cond in [c.strip() for c in args.conditions.split(",") if c.strip()]:
        print(f"  조건 {cond} 학습 시작")
        r = run_source_probe(
            train, test,
            make_scorer(root, cond, args.device, args.batch, args.workers),
            condition=cond,
        )
        results.append(r)
        print(f"  [{cond}] AUC {r.auc:.4f} CI [{r.ci_lo:.4f}, {r.ci_hi:.4f}]")
        if cond == "raw":
            verdicts.append(judge_p2(r.auc, r.ci_hi, n_clusters=r.n_test_groups))

    payload = {
        "manifest": args.manifest,
        "provenance": args.provenance,
        "subset": args.subset,
        "n_pool": len(rows),
        "source_counts": summarize_sources(rows),
        "n_train": len(train),
        "n_test": len(test),
        "n_test_groups": len({r.group_id for r in test}),
        "seed": SEED,
        "epochs": EPOCHS,
        "results": [r.as_dict() for r in results],
        "verdicts": [v.as_dict() for v in verdicts],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    for v in verdicts:
        print(f"  [{v.probe}] {v.detail}")
    print(f"결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
