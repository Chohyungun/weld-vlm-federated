"""통합형 export 진입점 — 리허설 2판 §2(C8). 묶음 계약의 쓰는 쪽을 명령줄로 연다.

    python -X utf8 scripts/export_uni.py --purpose rehearsal --mode model --tag uni_central --seed-index 1 \\
        --receipt R --list L --snapshot S --run-root ROOT --plan P --adapter-dir A --folder F --device D

| 인자 | 뜻 |
|---|---|
| `--purpose` · `--mode` · `--tag` · `--seed-index` | 목적(`main` · `rehearsal` · `frame_diag`) · 모드(`model` · `echo`) · 칸 · 시드 번째 |
| `--receipt` · `--list` · `--snapshot` | 영수증 · 생성(모델) 또는 에코 목록 파일 · 동결 스냅샷. 목록은 **바이트 해시로** 등록에 묶인다 |
| `--run-root` · `--plan` | 리허설 · 진단의 루트와 계획 파일. 본실험은 둘 다 받지 않는다 |
| `--adapter-dir` | 모델 모드의 학습 폴더(`adapter_last.npz` · meta · 원장) |
| `--folder` | 묶음 폴더. 모델과 에코 묶음이 **같은 폴더**를 쓴다 — 평가 쪽 진입점이 폴더 하나(`--bundles`)를 읽는다 |
| `--with-tokens` · `--stop-after-lines K` · `--abort-mismatched` | 토큰 파일 · 계획된 멈춤 · 도장 불일치 때의 개명(§2-4 의 3) |
| `--device` | 곁 파일의 장비 문자열 |

종료 코드 — 0 은 쓴 시도가 끝 줄까지 갔다(봉인이든 계획된 멈춤이든), 75 는 고의 중단, 2 는 거부(사유 코드를 stderr 첫 줄에 쓴다),
1 은 그 밖의 예외다.

**이음새는 파이썬 인자로만**(2판 §1-4). 명령줄에는 구현을 고르는 인자가 없다 — `main(argv, measure_env=..., load_generator=...)` 로만
바꾼다. 기본값은 승인된 실제 구현이다 — 실측은 `vlm.export_measure.measure_env`, 생성기 적재는 두 모드가 같이 지나는 `vlm.export_generator.load_generator`.
**모델 생성기의 실제 구현은 아직 없다** — 기본값이 없어 모델 모드는 이음새 없이 시작하지 않는다.

**이미지 경로.** 리허설 · 진단과 에코 모드는 설정의 본실험 페어(`uni_pairs.path`)의 val 행 `image_path` 로 연다 — 학습과 같은 규칙이다
(상대 경로는 저장소 루트 기준). 본실험의 모델 모드는 평가 이미지를 열어야 하는데 **그 경로를 낼 자리가 정해지지 않았다**
(열 판독기는 평가 행의 경로 열을 내지 않는다) — 경로를 지어내지 않고 거부한다.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2

__all__ = ["EXIT_OK", "EXIT_REFUSED", "main", "pair_image_paths"]


def pair_image_paths(gen_cfg: Any, *, purpose: str) -> dict[str, Path]:
    """페어 val 행의 `image_id` → 이미지 경로. 본실험 모델 모드는 거부한다(머리말의 이미지 경로)."""
    from vlm.export_writer import ExportRefused
    from vlm.pilot_vlm import load_pairs

    if gen_cfg.mode == "model" and purpose == "main":
        raise ExportRefused("main_image_paths_undefined", "본실험 모델 모드의 평가 이미지 경로를 낼 자리가 정해지지 않았다")
    if not gen_cfg.pairs_path:
        raise ExportRefused("pairs_path_missing", "이미지 경로를 낼 페어 경로가 설정에 없다(uni_pairs.path)")
    out: dict[str, Path] = {}
    for r in load_pairs("val", None, pairs_path=gen_cfg.pairs_path):
        p = Path(r["image_path"])
        out[str(r["image_id"])] = p if p.is_absolute() else REPO_ROOT / p
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--purpose", required=True, choices=("main", "rehearsal", "frame_diag"))
    ap.add_argument("--mode", required=True, choices=("model", "echo"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed-index", required=True, type=int)
    for name in ("receipt", "list", "snapshot", "folder"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--adapter-dir", type=Path, default=None)
    ap.add_argument("--with-tokens", action="store_true")
    ap.add_argument("--stop-after-lines", type=int, default=None)
    ap.add_argument("--abort-mismatched", action="store_true")
    ap.add_argument("--device", required=True)
    return ap


def main(argv: list[str] | None = None, *, measure_env: Callable | None = None,
         load_generator: Callable | None = None, image_path_of: Any = None, repo: Path | None = None,
         code_repo: Path | None = None, exposure_ledger: Path | None = None, non_main_parent: Path | None = None,
         clock: Callable | None = None, standin_allowed: bool = False) -> int:
    """명령줄 입구. 키워드 인자는 **시험의 이음새**다 — 명령줄로는 바꿀 수 없다."""
    from vlm.export_generator import load_generator as load_real
    from vlm.export_generator import model_generator_available
    from vlm.export_measure import measure_env as measure_real
    from vlm.export_run import run_export
    from vlm.export_writer import ExportRefused
    from vlm.fault import FaultRefused
    from vlm.seams import SeamRejected

    a = _parser().parse_args(argv)
    if load_generator is None:
        if a.mode == "model" and not model_generator_available():
            print("[model_generator_missing] 모델 생성기의 실제 구현이 아직 없다 — 모델 모드는 시작하지 않는다", file=sys.stderr)
            return EXIT_REFUSED
        load_generator = load_real
    if a.stop_after_lines is not None and a.stop_after_lines < 1:
        print(f"[stop_after_lines] --stop-after-lines 는 1 이상이다: {a.stop_after_lines}", file=sys.stderr)
        return EXIT_REFUSED
    paths = image_path_of if image_path_of is not None else (
        lambda gen_cfg: pair_image_paths(gen_cfg, purpose=a.purpose))
    try:
        res = run_export(folder=a.folder, tag=a.tag, seed_index=a.seed_index, mode=a.mode, purpose=a.purpose,
                         receipt_path=a.receipt, list_path=a.list, snapshot_root=a.snapshot, image_path_of=paths,
                         measure_env=measure_env or measure_real, load_generator=load_generator, device=a.device,
                         run_root=a.run_root, non_main_parent=non_main_parent, adapter_dir=a.adapter_dir,
                         plan_path=a.plan, with_tokens=a.with_tokens, stop_after_lines=a.stop_after_lines,
                         abort_mismatched=a.abort_mismatched, clock=clock, standin_allowed=standin_allowed,
                         repo=repo, code_repo=code_repo, exposure_ledger=exposure_ledger)
    except (ExportRefused, FaultRefused, SeamRejected) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"[{code}] {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{'sealed' if res.sealed else 'stopped'} attempt={res.plan.attempt_no} reason={res.plan.reason} "
          f"n_written={res.n_written}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
