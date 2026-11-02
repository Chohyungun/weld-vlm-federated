"""통합형 export 진입점 — 리허설 2판 §2(C8). 묶음 계약의 쓰는 쪽을 명령줄로 연다.

    python -X utf8 scripts/export_uni.py --purpose rehearsal --mode model --tag uni_central --seed-index 1 \\
        --receipt R --list L --snapshot S --run-root ROOT --plan P --adapter-dir A --folder F --device D

| 인자 | 뜻 |
|---|---|
| `--purpose` · `--mode` · `--tag` · `--seed-index` | 목적(`main` · `rehearsal` · `frame_diag` · `early_coord_check`) · 모드(`model` · `echo`) · 칸 · 시드 번째 |
| `--check-step` | 조기 좌표 점검의 점검 step — 그 목적의 모델 모드에서만, 그때는 반드시. 그 step 의 어댑터를 읽는다(11번 §3-3) |
| `--receipt` · `--list` · `--snapshot` | 영수증 · 생성(모델) 또는 에코 목록 파일 · 동결 스냅샷. 목록은 **바이트 해시로** 등록에 묶인다 |
| `--run-root` · `--plan` | 리허설 · 진단의 루트와 계획 파일. 본실험은 둘 다 받지 않는다 |
| `--adapter-dir` | 모델 모드의 학습 폴더(`adapter_last.npz` · meta · 원장) |
| `--folder` | 묶음 폴더. 모델과 에코 묶음이 **같은 폴더**를 쓴다 — 평가 쪽 진입점이 폴더 하나(`--bundles`)를 읽는다 |
| `--with-tokens` · `--stop-after-lines K` · `--abort-mismatched` | 토큰 파일 · 계획된 멈춤 · 도장 불일치 때의 개명(§2-4 의 3) |
| `--image-root` · `--image-list` | 본실험 모델 모드의 평가 이미지 — 뿌리와 승인 목록 파일(`rel_path,…,sha256`). 그 모드에서만, 둘을 함께 |
| `--side` | 표본 모드 본실험 모델 묶음의 곁 파일(`image_id,stratum,G_h,n_h`) — 봉인한 `side_sha256` 과 맞댄다 |
| `--device` | 곁 파일의 장비 문자열 — 채점기의 꼴 `{이름}:{인덱스}`. 새 시작은 그 꼴이 아니면 `export_device_form` 으로 시작하지 않는다. 재개는 남은 도장(첫 시작 줄)을 따른다: 정본 도장이면 정본만, 비정본 도장(꼴 검사 전에 시작한 묶음)이면 그 문자열을 잇는다(`vlm.export_writer.resume_device`) |

종료 코드 — 0 은 쓴 시도가 끝 줄까지 갔다(봉인이든 계획된 멈춤이든), 75 는 고의 중단, 2 는 거부(사유 코드를 stderr 첫 줄에 쓴다),
1 은 그 밖의 예외다.

**이음새는 파이썬 인자로만**(2판 §1-4). 명령줄에는 구현을 고르는 인자가 없다 — `main(argv, measure_env=..., load_generator=...)` 로만
바꾼다. 기본값은 승인된 실제 구현이다 — 실측은 `vlm.export_measure.measure_env`, 생성기 적재는 두 모드가 같이 지나는 `vlm.export_generator.load_generator`.
**모델 생성기의 실제 구현은 아직 없다** — 기본값이 없어 모델 모드는 이음새 없이 시작하지 않는다.

**이미지 경로.** 리허설 · 진단과 에코 모드는 설정의 본실험 페어(`uni_pairs.path`)의 val 행 `image_path` 로 연다 — 학습과 같은 규칙이다
(상대 경로는 저장소 루트 기준). 본실험의 모델 모드는 `--image-root` 와 `--image-list` 로 연다(1016k) — 열 판독기는 평가 행의 경로
열을 내지 않으므로(`data.manifest_view.EVAL_READABLE`) 경로는 승인 목록 파일의 `rel_path`(원본 — 타일이 아니다)에서 받고, id 는 동결
매니페스트의 `sha256` 으로 잇는다. 모델을 올리기 전에 장마다 파일 바이트의 sha256 을 매니페스트 값과 맞댄다(`main_image_paths`).
둘이 없으면 경로를 지어내지 않고 거부한다.
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

__all__ = ["EXIT_OK", "EXIT_REFUSED", "main", "main_image_paths", "pair_image_paths"]


def main_image_paths(image_root: Path, *, image_list: Path, snapshot: Path, list_path: Path) -> dict[str, Path]:
    """본실험 모델 모드의 평가 이미지 — 목록의 `image_id` → `<image_root>/<rel_path>`(1016k, 03 §2 (가)).

    경로는 승인 목록 파일(머리줄에 `rel_path` · `sha256` 이 있는 CSV, `#` 로 시작하는 줄은 설명)에서 받고, id 는 동결 매니페스트의
    `sha256` 으로 잇는다 — 열 판독기는 평가 행의 경로 열을 내지 않는다. 목록의 장마다 파일 바이트의 sha256 이 매니페스트 값과
    같아야 한다(채점 쪽 묶음 검증의 줄 사실 검사 `bundle_v14` 의 `image_sha256` 과 같은 값을 모델을 올리기 전에 본다). 경로는
    뿌리 아래의 상대 경로만 받는다. 어긋나면 `ExportRefused` — 사유에는 건수만 싣는다.
    """
    import hashlib
    import re
    from pathlib import PurePosixPath

    from evaluation.entry_gate import EntryRejected, read_manifest_meta
    from evaluation.eval_list import validate_eval_list
    from evaluation.reject_v14 import BundleRejected
    from vlm.export_writer import ExportRefused

    try:
        el = validate_eval_list(Path(list_path).read_bytes(), where=Path(list_path).name)
        meta = read_manifest_meta(Path(snapshot))
    except (BundleRejected, EntryRejected) as exc:
        raise ExportRefused("main_image_list", f"목록 · 매니페스트를 읽지 못했다: {exc}") from None
    try:
        text = Path(image_list).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExportRefused("main_image_list", f"승인 목록 파일을 읽지 못했다: {type(exc).__name__}") from None
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    head = lines[0].split(",") if lines else []
    if "rel_path" not in head or "sha256" not in head:
        raise ExportRefused("main_image_list", "승인 목록 파일의 머리줄에 rel_path · sha256 이 없다")
    ir, ish = head.index("rel_path"), head.index("sha256")
    by_sha: dict[str, str] = {}
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) != len(head) or not re.fullmatch(r"[0-9a-f]{64}", parts[ish]):
            raise ExportRefused("main_image_list", "승인 목록 파일의 줄 꼴이 머리줄과 다르다")
        by_sha.setdefault(parts[ish], parts[ir])
    out: dict[str, Path] = {}
    unlisted = bad_path = missing = mismatch = 0
    for i in el.ids:
        want = meta.sha256_of.get(i)
        rel = by_sha.get(want) if want else None
        if rel is None:
            unlisted += 1
            continue
        pp = PurePosixPath(rel.replace("\\", "/"))
        if not pp.parts or pp.is_absolute() or ".." in pp.parts or ":" in rel:
            bad_path += 1
            continue
        p = Path(image_root).joinpath(*pp.parts)
        if not p.is_file():
            missing += 1
        elif hashlib.sha256(p.read_bytes()).hexdigest() != want:
            mismatch += 1
        else:
            out[i] = p
    if unlisted or bad_path or missing or mismatch:
        raise ExportRefused("main_image_files", f"평가 이미지 {len(el)}장 가운데 목록 파일에 없음 {unlisted} · 경로 꼴 {bad_path} · "
                                                f"파일 없음 {missing} · 바이트가 매니페스트와 다름 {mismatch}")
    return out


def pair_image_paths(gen_cfg: Any, *, purpose: str, image_root: Path | None = None, image_list: Path | None = None,
                     snapshot: Path | None = None, list_path: Path | None = None) -> dict[str, Path]:
    """페어 val 행의 `image_id` → 이미지 경로. 본실험 모델 모드는 `main_image_paths`(뿌리 · 승인 목록 파일이 없으면 거부).

    조기 좌표 점검은 학습 분할의 결함 장을 점검하므로(평가 쪽 10번 §2 — 목적의 분할 `train`) train 행에서 낸다.
    """
    from vlm.export_writer import ExportRefused
    from vlm.pilot_vlm import load_pairs

    if gen_cfg.mode == "model" and purpose == "main":
        if image_root is None or image_list is None or snapshot is None or list_path is None:
            raise ExportRefused("main_image_paths_undefined",
                                "본실험 모델 모드는 --image-root · --image-list 가 있어야 한다 — 평가 이미지 경로를 지어내지 않는다")
        return main_image_paths(Path(image_root), image_list=Path(image_list), snapshot=Path(snapshot),
                                list_path=Path(list_path))
    if not gen_cfg.pairs_path:
        raise ExportRefused("pairs_path_missing", "이미지 경로를 낼 페어 경로가 설정에 없다(uni_pairs.path)")
    out: dict[str, Path] = {}
    for r in load_pairs("train" if purpose == "early_coord_check" else "val", None, pairs_path=gen_cfg.pairs_path):
        p = Path(r["image_path"])
        out[str(r["image_id"])] = p if p.is_absolute() else REPO_ROOT / p
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--purpose", required=True, choices=("main", "rehearsal", "frame_diag", "early_coord_check"))
    ap.add_argument("--check-step", type=int, default=None)
    # 점검 export 의 비용 가드(11번 §5 · §119 I-08) — 모델을 올리기 전에 `쓴 시간 + 예상` 을 규칙의 상한과 맞댄다
    ap.add_argument("--spent-hours", type=float, default=None)
    ap.add_argument("--next-hours", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--max-won", type=float, default=None)
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
    ap.add_argument("--image-root", type=Path, default=None)
    ap.add_argument("--image-list", type=Path, default=None)
    ap.add_argument("--side", type=Path, default=None)
    ap.add_argument("--device", required=True, help="장비 {이름}:{인덱스}(채점기와 같은 꼴) — 재개는 남은 도장을 따른다")
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
    budget_args = (a.spent_hours, a.next_hours, a.max_hours, a.max_won)
    if a.purpose != "early_coord_check" and any(v is not None for v in budget_args):
        print("[ecc_budget_args] 비용 가드 인자(--spent-hours · --next-hours · …)는 early_coord_check 에서만 받는다", file=sys.stderr)
        return EXIT_REFUSED
    if a.purpose == "early_coord_check" and a.mode == "model":
        if a.spent_hours is None or a.next_hours is None or a.adapter_dir is None:
            print("[ecc_budget_args] 점검 export 는 --spent-hours · --next-hours · --adapter-dir 가 있어야 한다(11번 §5)", file=sys.stderr)
            return EXIT_REFUSED
        from vlm.ecc import EccRefused, budget_check, effective_limits, load_check_rules

        try:
            lim = effective_limits(load_check_rules()["rules"], max_hours=a.max_hours, max_won=a.max_won)
            budget_check(Path(a.adapter_dir), spent_hours=a.spent_hours, next_hours=a.next_hours, max_hours=lim["max_hours"],
                         max_won=lim["max_cost_krw"], won_per_hour=lim["cost_krw_of_hour"], what="export",
                         price_detail={k: lim[k] for k in ("price_krw_per_hour", "safety_factor", "price_source")})
        except EccRefused as exc:
            print(f"[{exc.code}] {exc}", file=sys.stderr)
            return EXIT_REFUSED
    if load_generator is None:
        if a.mode == "model" and not model_generator_available():
            print("[model_generator_missing] 모델 생성기의 실제 구현이 아직 없다 — 모델 모드는 시작하지 않는다", file=sys.stderr)
            return EXIT_REFUSED
        load_generator = load_real
    if (a.image_root is not None or a.image_list is not None) and not (a.purpose == "main" and a.mode == "model"):
        print("[image_root_scope] --image-root · --image-list 는 본실험 모델 모드에서만 받는다", file=sys.stderr)
        return EXIT_REFUSED
    if (a.image_root is None) != (a.image_list is None):
        print("[image_root_args] --image-root 와 --image-list 는 함께 준다", file=sys.stderr)
        return EXIT_REFUSED
    if a.stop_after_lines is not None and a.stop_after_lines < 1:
        print(f"[stop_after_lines] --stop-after-lines 는 1 이상이다: {a.stop_after_lines}", file=sys.stderr)
        return EXIT_REFUSED
    paths = image_path_of if image_path_of is not None else (
        lambda gen_cfg: pair_image_paths(gen_cfg, purpose=a.purpose, image_root=a.image_root, image_list=a.image_list,
                                         snapshot=a.snapshot, list_path=a.list))
    try:
        res = run_export(folder=a.folder, tag=a.tag, seed_index=a.seed_index, mode=a.mode, purpose=a.purpose,
                         receipt_path=a.receipt, list_path=a.list, snapshot_root=a.snapshot, image_path_of=paths,
                         measure_env=measure_env or measure_real, load_generator=load_generator, device=a.device,
                         run_root=a.run_root, non_main_parent=non_main_parent, adapter_dir=a.adapter_dir,
                         plan_path=a.plan, with_tokens=a.with_tokens, stop_after_lines=a.stop_after_lines,
                         abort_mismatched=a.abort_mismatched, clock=clock, standin_allowed=standin_allowed,
                         repo=repo, code_repo=code_repo, exposure_ledger=exposure_ledger, check_step=a.check_step,
                         side_path=a.side)
    except (ExportRefused, FaultRefused, SeamRejected) as exc:
        code = getattr(exc, "code", type(exc).__name__)
        print(f"[{code}] {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"{'sealed' if res.sealed else 'stopped'} attempt={res.plan.attempt_no} reason={res.plan.reason} "
          f"n_written={res.n_written}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
