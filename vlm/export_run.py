"""export 한 묶음 — 가드 → 사전 점검 → 시작 절차 → 줄마다 생성 · 파싱 · 역변환 → 끝 줄 → 곁 파일 → 노출 원장(리허설 2판 §1-4 · §2-4).

## 가드의 순서(2판 §1-4)

1. 인자 — 목적 · 모드 · 태그(평가 쪽 `tag_parts` 로 칸 · 참여자를 낸다).
2. 고의 중단 변수 — 본실험 · 진단은 보이면 거부한다. 리허설은 둘이 함께 있고 계획 기록과 같을 때만 켠다(2판 §1-3).
   지점은 넷이다 — `after_lines`(그 줄을 쓴 뒤) · `torn_line`(그 줄의 앞 절반만 쓴 뒤) · `after_all_lines`(목록을 다 쓴 뒤 · 끝 줄 전) ·
   `after_stamp`(새 도장 뒤 · 시작 줄 전, 합성 시험). `n` 은 **생성 파일 전체의 줄 번호**(1부터)다. "한 번만" 은 멱등 판정과 같은 자리에서 본다.
3. 이음새 — 실측(`measure_env`, 프로세서만 연다) · 생성기 적재(`load_generator`). **파이썬 인자로만** 받는다. 본실험은 승인된 실제 구현만
   받는다(`vlm/seams.py`). 실측의 실제 구현은 `vlm.export_measure.measure_env`, 생성기 적재의 실제 구현은 두 모드가 같이 지나는
   `vlm.export_generator.load_generator` 다(모드별 구현 결속 — 등록의 구현 식별자가 한 칸이다).
   **모델 생성기의 실제 구현은 아직 없다** — 그래서 본실험의 모델 export 는 아직 시작하지 않는다.
4. 루트와 경로 — 리허설 · 진단은 출력 · 영수증 · 목록 · 어댑터 · 계획이 모두 루트 아래, 본실험은 어느 것도 비본실험 부모 아래가 아니다.
5. 적합성 벡터 — 이 프로세스 안에서 C 파서를 평가 쪽 벡터에 태우고, 모든 사례에서 줄의 두 필드가 짝을 지키는지 본다.
6. 사전 점검(`vlm.export_preflight`) — 평가 쪽 실제 함수로 영수증 · 등록 · 닻 · 목록 · 분할 · 노출 기록 · 실측값 · 학습 출처 · 청결.
   진단의 커밋된 영수증이면 대역 이음새를 거부한다(2판 §1-4 의 순서 3, 진단 열).

1~6 은 파일을 쓰지 않고 이미지를 열지 않고 모델을 올리지 않는다. 생성기는 시작 줄이 `first_id` 를 가질 때만 적재한다.

## 줄 하나를 쓰는 순서(§2-4 끝)

이미지 **바이트를 읽어 해시**하고 같은 바이트로 이미지를 연다 → 생성기(프로세서 · greedy 1회) → 파싱 · 역변환(`vlm.gen_parse`,
`to_px` 한 번) → 토큰 줄(켰으면) → **생성 줄을 마지막에** 쓴다. 파싱이 실패해도 그 줄을 쓴다(사유와 함께). 이미지를 열지 못하는
기반 오류는 줄을 쓰지 않고 멈춘다(끝 줄 없는 시도로 남는다).

짝 규칙이 어긋난 줄도 **멈추지 않고 그대로 쓴다** — 그 줄이 배관 고장의 표시이고, 평가 쪽이 그 묶음을 거부한다(§2-4 의 짝 규칙).
`gen_stop` 을 정하지 못하면 `stop_undetermined` 를 적고 계속한다(계약 §16-2 m-2).
에코 줄의 `latency_ms` 는 **파싱 경로만의 시간**이다(§2-4 의 에코 모드).

곁 파일을 쓴 뒤 리허설 · 진단은 노출 원장에 `generated` 줄을 덧붙인다(§5-4).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vlm.export_generator import load_generator as _generator_real
from vlm.export_generator import model_generator_available
from vlm.export_measure import measure_env as _measure_env_real
from vlm.export_writer import MODES, ExportRefused, ExportWriter, StartPlan

__all__ = [
    "APPROVED_GENERATOR_LOADERS",
    "APPROVED_MEASURES",
    "PURPOSES",
    "ExportResult",
    "GenOut",
    "conformance_problems",
    "grid_key",
    "parser_sha256",
    "run_export",
]

PURPOSES = ("main", "rehearsal", "frame_diag")
#: 승인된 실제 구현 — import 시점에 잡아 둔 참조다. 생성기는 모드마다 따로 본다 — 모델 생성기의 실제 구현은 아직 없다.
APPROVED_MEASURES: tuple = (_measure_env_real,)
APPROVED_GENERATOR_LOADERS: dict[str, tuple] = {"model": (_generator_real,) if model_generator_available() else (),
                                                 "echo": (_generator_real,)}
#: 등록하는 생성 조건 — greedy 1회 · 배치 1 의 왼쪽 채움(07번 §13-3 다-7). 등록 단계가 이 값을 적고 모델 생성기가 이 값으로 돈다.
DECODING: dict[str, Any] = {"do_sample": False, "num_beams": 1}
PADDING_SIDE = "left"


@dataclass(frozen=True)
class GenOut:
    """생성기 한 번의 결과."""

    text: str
    gen_stop: str
    """`eos` · `length` · 정하지 못하면 `stop_undetermined`."""
    n_new_tokens: int
    model_input_wh: tuple[int, int]
    grid_thw: tuple[int, int, int] | None = None
    token_ids: Sequence[int] | None = None
    token_logprobs: Sequence[float] | None = None
    latency_ms: float | None = None


@dataclass
class ExportResult:
    plan: StartPlan
    n_written: int
    sealed: bool
    meta: dict[str, Any] | None = None
    impl_ids: dict[str, Any] = field(default_factory=dict)
    generator_loaded: bool = False
    stand_in: bool = False
    exposure_line: bytes | None = None


def parser_sha256() -> str:
    """파서 모듈(`vlm/gen_parse.py`)의 **원시 바이트** sha256 — 도장의 `parser_sha256`. 코드를 고친 뒤 이어 쓰지 않게 한다."""
    return hashlib.sha256((Path(__file__).resolve().parent / "gen_parse.py").read_bytes()).hexdigest()


def grid_key(g: Sequence[int]) -> str:
    """`image_grid_thw_observed` 의 키 — `json.dumps([t, h, w])`(기본 구분자). 모든 묶음이 이 함수 하나로 만든다(2판 §2-5)."""
    return json.dumps([int(v) for v in g])


def conformance_problems() -> list[str]:
    """평가 쪽 적합성 벡터에 C 파서를 태운다. 어긋난 사례 이름 목록 — 비어야 시작한다(2판 §2-4 의 1)."""
    from evaluation.conformance_v14 import KNOWN_CODES, SCORING_CODES, all_cases, check_case
    from vlm.coords import CoordCfg, ImageGeom
    from vlm.gen_parse import conformance_tuple, line_fields, pair_ok, parse_generation

    out = []
    geom, cfg = ImageGeom(orig_w=1280, orig_h=720), CoordCfg(coord_space="ABS_ORIG")
    for case in all_cases():
        res = check_case(case, parse=lambda t: conformance_tuple(t, known_iso_codes=KNOWN_CODES,
                                                                 scoring_iso_codes=SCORING_CODES))
        if not res.passed:
            out.append(f"{case.name}: 값이 다르다")
        lf = line_fields(parse_generation(case.text), geom, cfg)
        if not pair_ok(lf["bbox_px_parsed"], lf["parse_error"]):
            out.append(f"{case.name}: 짝이 어긋난다")
    return out


def _root_problems(purpose: str, paths: Sequence[Path], run_root, non_main_parent) -> list[str]:
    from vlm.run_root import REPO_ROOT, run_root_problems
    from vlm.uni_config import NON_MAIN_PARENTS

    if purpose != "main":
        return run_root_problems(purpose, run_root, list(paths), parent=non_main_parent)
    parents = NON_MAIN_PARENTS if non_main_parent is None else (non_main_parent,)
    out = []
    for p in paths:
        q = Path(p).resolve()
        out += [f"본실험 경로 {p} 가 비본실험 부모 {par} 아래에 있다" for par in parents
                if q.is_relative_to((REPO_ROOT / par).resolve())]
    return out


def _export_fault_events(run_root, tag: str, w: ExportWriter) -> list[dict[str, Any]]:
    """이 묶음(export · 태그)에서 켜진 중단의 기록과 그 뒤에 복구한 시도(`attempt_no`) — 2판 §1-3 의 신원과 이력."""
    from vlm.export_writer import check_attempts_file
    from vlm.fault import fault_events

    starts = sorted((r for r in w._rows_of(check_attempts_file(w.p_att), w.stamp_sha) if r["event"] == "start"),
                    key=lambda r: r["attempt_no"])
    segs = [{"started_at": datetime.fromisoformat(r["started_at"]).timestamp(), "attempt_no": r["attempt_no"]}
            for r in starts]
    out = []
    for ev in fault_events(run_root, stage="export", tags=[tag], segments=segs):
        t = ev.get("recovered_process_started_unix")
        ev["recovered_attempt_no"] = next((s["attempt_no"] for s in segs if s["started_at"] == t), None)
        out.append(ev)
    return out


def run_export(*, folder: str | Path, tag: str, seed_index: int, mode: str, purpose: str,
               receipt_path: str | Path, list_path: str | Path, snapshot_root: str | Path,
               image_path_of: Mapping[str, str | Path] | Callable, measure_env: Callable, load_generator: Callable,
               device: str, run_root: str | Path | None = None, non_main_parent: str | Path | None = None,
               adapter_dir: str | Path | None = None, plan_path: str | Path | None = None,
               with_tokens: bool = False, stop_after_lines: int | None = None, abort_mismatched: bool = False,
               clock: Callable | None = None, standin_allowed: bool = False, env: Mapping[str, str] | None = None,
               repo: str | Path | None = None, code_repo: str | Path | None = None,
               exposure_ledger: str | Path | None = None) -> ExportResult:
    """한 묶음을 쓴다. `stop_after_lines` 는 계획된 멈춤이다(이 시도에서 K 줄을 쓰고 끝 줄을 적는다 — 중단이 아니다).

    `repo` · `code_repo` · `exposure_ledger` · `non_main_parent` 는 시험의 이음새다(파이썬 인자로만) — 정본은 본체 체크아웃 ·
    이 작업 트리 · 본체의 `outputs/exposure/` · `outputs/rehearsal_u` 다. `image_path_of` 가 함수면 사전 점검이 낸 생성 설정으로
    부른다(페어의 val 행에서 경로를 낸다 — CLI).
    """
    from evaluation.entry_gate import COMMITTED_KINDS, is_ancestor, main_checkout
    from evaluation.schema_v14 import tag_parts
    from vlm.coords import CoordCfg, ImageGeom, coord_cfg_hash
    from vlm.export_preflight import export_preflight
    from vlm.exposure import append_row
    from vlm.exposure import default_ledger as exposure_default
    from vlm.fault import FaultContext, FaultRefused, check_fault_env
    from vlm.gen_parse import line_fields, parse_generation
    from vlm.seams import check_seam

    # 1 인자
    if purpose not in PURPOSES:
        raise ExportRefused("purpose", f"목적은 {PURPOSES} 가운데 하나다: {purpose!r}")
    if mode not in MODES:
        raise ExportRefused("mode", f"모드는 {MODES} 가운데 하나다: {mode!r}")
    cell, client = tag_parts(tag)
    folder = Path(folder)
    # 2 고의 중단 변수 — 본실험 · 진단은 보이면 거부, 리허설은 계획 기록과 같을 때만 켠다
    armed = check_fault_env(purpose, run_root=run_root, stage="export", tags=[tag], env=env)
    # 3 이음새 — 부르기 전에 목적과 맞댄다
    impl_ids = {
        "export_measure": check_seam(measure_env, APPROVED_MEASURES, seam="export_measure", purpose=purpose,
                                     standin_allowed=standin_allowed),
        "export_generator": check_seam(load_generator, APPROVED_GENERATOR_LOADERS[mode], seam="export_generator",
                                       purpose=purpose, standin_allowed=standin_allowed),
    }
    # 3″ 바깥 적재 함수가 실제 구현이면 그것이 부를 안쪽 구현(에코 · 모델 생성기 · 모델 적재기)도 지금 맞대고 식별자를 싣는다.
    #    영수증 가드(3′)보다 먼저 — 바깥은 승인이고 안쪽만 대역인 진단도 같은 자리에서 막는다(외부 검토 재확인 추가 1).
    if load_generator is _generator_real:
        from vlm.export_generator import inner_impl_ids

        impl_ids.update(inner_impl_ids(purpose=purpose, standin_allowed=standin_allowed))
    # 3′ 진단의 커밋된 영수증이면 대역 이음새를 받지 않는다 — **대역을 한 번도 부르기 전에**(2판 §1-4 의 순서 3, 외부 검토 8).
    #    조상 검사는 이음새의 저장소가 아니라 이 모듈이 찾은 본체 체크아웃으로 한다.
    if purpose == "frame_diag" and not all(i["approved"] for i in impl_ids.values()):
        from evaluation.prereg_unified import ReceiptMissing, read_receipt

        try:
            rc0 = read_receipt(Path(receipt_path))
        except (ReceiptMissing, OSError, ValueError) as exc:
            raise ExportRefused("entry_RECEIPT", f"영수증을 읽을 수 없다: {exc}") from None
        if rc0.kind in COMMITTED_KINDS and is_ancestor(main_checkout(), rc0.main_commit):
            raise ExportRefused("seam_on_committed_receipt", "진짜 진단 등록(커밋된 영수증)은 대역 이음새를 받지 않는다")
    # 4 루트와 경로 — 출력과 모든 입력
    paths = [folder, Path(receipt_path), Path(list_path)] + [Path(p) for p in (adapter_dir, plan_path) if p is not None]
    problems = _root_problems(purpose, paths, run_root, non_main_parent)
    if problems:
        raise ExportRefused("run_root", " · ".join(problems))
    # 5 적합성 벡터와 짝
    bad = conformance_problems()
    if bad:
        raise ExportRefused("conformance", f"적합성 벡터에서 C 파서가 어긋난다: {bad}")
    # 6 사전 점검 — 평가 쪽 실제 함수로. 이미지를 열기 전 · 모델을 올리기 전에 끝난다
    now = (clock or (lambda: datetime.now().astimezone()))()
    pre = export_preflight(purpose=purpose, mode=mode, tag=tag, seed_index=int(seed_index),
                           receipt_path=Path(receipt_path), list_path=Path(list_path),
                           snapshot_root=Path(snapshot_root), adapter_dir=None if adapter_dir is None else Path(adapter_dir),
                           plan_path=None if plan_path is None else Path(plan_path),
                           run_root=None if run_root is None else Path(run_root), measure_env=measure_env,
                           impl_ids=impl_ids, parser_sha256=parser_sha256(), repo=repo, code_repo=code_repo,
                           exposure_ledger=exposure_ledger, now=now)

    n_list = len(pre.eval_list.ids)
    if armed is not None and armed.spec.k is not None and not 1 <= int(armed.spec.k) <= n_list:
        raise FaultRefused("fault_range", f"줄 번호 {armed.spec.k} 가 목록 1..{n_list} 밖이다")
    paths_of = image_path_of(pre.gen_cfg) if callable(image_path_of) else image_path_of

    w = ExportWriter(folder, tag=tag, seed_index=seed_index, mode=mode, eval_list=pre.eval_list,
                     stamp_fields=pre.stamp_fields, device=device, with_tokens=with_tokens, clock=clock,
                     abort_mismatched=abort_mismatched)
    # "한 번만" — 가드를 다 지난 뒤, 멱등 판정(완결 묶음 거부)과 같은 자리(2판 §1-4 의 순서 7)
    if armed is not None and armed.already_activated():
        import sys

        print(f"[fault] {armed.spec.raw} 의 활성화 기록이 이미 있다 — 다시 켜지 않는다", file=sys.stderr, flush=True)
        armed = None
    ctx = FaultContext(purpose=purpose, run_root=Path(run_root)) if armed is not None else None

    def fire(point: str, k: int | None, effect=None) -> None:
        if armed is not None and armed.spec.point == point:
            armed.fire(ctx, point=point, tag=tag, k=k, run_id=Path(run_root).name, effect=effect)

    plan = w.begin(after_stamp=lambda: fire("after_stamp", None))
    coord_cfg = CoordCfg(coord_space=pre.gen_cfg.coord_space)
    chash = coord_cfg_hash(coord_cfg)
    loaded = False
    if plan.first_id is not None:
        generator = load_generator(pre.gen_cfg)          # 쓸 줄이 있을 때만 올린다
        loaded = True
        # 실제 모델 생성기가 부른 적재기가 이음새 단계에 적은 것과 다르면 쓰지 않는다 — 곁 파일이 출처를 잘못 적는다(재확인 추가 2)
        used = (getattr(generator, "impl_ids", None) or {}).get("model_loader")
        if used is not None and (impl_ids.get("export_generator_model") or {}).get("approved") \
                and used != impl_ids.get("model_loader"):
            raise ExportRefused("impl_drift", "생성기가 부른 모델 적재기가 이음새 단계에 적은 식별자와 다르다")
        w.open_generations()
        while (image_id := w.next_id()) is not None:
            if stop_after_lines is not None and w.n_written >= stop_after_lines:
                break
            raw = Path(paths_of[image_id]).read_bytes()             # 바이트를 읽어 해시하고 같은 바이트로 연다
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.load()
            t0 = time.perf_counter()
            out = generator(img, image_id)
            t1 = time.perf_counter()
            fields = line_fields(parse_generation(out.text), ImageGeom(orig_w=img.size[0], orig_h=img.size[1]),
                                 coord_cfg)
            t2 = time.perf_counter()
            if mode == "echo":
                latency = (t2 - t1) * 1e3
            else:
                latency = out.latency_ms if out.latency_ms is not None else (t1 - t0) * 1e3
            row = {"image_id": image_id, "image_sha256": hashlib.sha256(raw).hexdigest(), "text": out.text,
                   **fields, "model_input_wh": [int(v) for v in out.model_input_wh],
                   "coord_space": coord_cfg.coord_space, "coord_cfg_hash": chash,
                   "latency_ms": round(float(latency), 3), "gen_stop": out.gen_stop,
                   "n_new_tokens": int(out.n_new_tokens)}
            if out.grid_thw is not None:
                # 관측 격자 — 줄에 싣고, 곁 파일은 생성 파일 **전체**에서 센다(이어 쓴 묶음도 앞 시도의 줄을 센다).
                row["image_grid_thw"] = [int(v) for v in out.grid_thw]
            tok = None
            if with_tokens:
                tok = {"image_id": image_id, "token_ids": [int(t) for t in (out.token_ids or [])],
                       "token_logprobs": [round(float(x), 6) for x in (out.token_logprobs or [])]}
            line_no = plan.m + w.n_written + 1                       # 생성 파일 전체의 줄 번호(1부터)
            if armed is not None and armed.spec.point == "torn_line" and armed.spec.k == line_no:
                fire("torn_line", line_no, effect=lambda r=row, k=tok: w.write_half(r, k))
            w.write(row, tok)
            fire("after_lines", line_no)
        if w.next_id() is None:
            fire("after_all_lines", None)
    w.end()
    if w.next_id() is not None and plan.first_id is not None:
        return ExportResult(plan, w.n_written, False, None, impl_ids, loaded, pre.stand_in)
    grid: dict[str, int] = {}
    for line in w.p_gen.read_bytes().splitlines():
        g = json.loads(line).get("image_grid_thw") if line else None
        if g is not None:
            grid[grid_key(g)] = grid.get(grid_key(g), 0) + 1
    measured = {"cell": cell, "client": client, "seed_value": pre.seed_value, "coord_space": coord_cfg.coord_space,
                "coord_cfg_hash": chash, "image_grid_thw_observed": grid, "impl_ids": impl_ids,
                "fault_events": [] if purpose == "main" else _export_fault_events(run_root, tag, w)}
    if purpose != "main" and run_root is not None:
        measured["rehearsal_id"] = Path(run_root).name
    meta = w.seal(pre.sidecar_values, measured)
    line = None
    if purpose != "main":
        # 노출 원장의 생성 줄 — 끝 줄과 곁 파일을 쓴 뒤(§5-4).
        ledger = Path(exposure_ledger) if exposure_ledger is not None else exposure_default()
        line = append_row(ledger, {
            "event": "generated", "run_id": Path(run_root).name, "purpose": purpose, "tag": tag,
            "seed_index": int(seed_index), "mode": mode, "list_kind": "echo" if mode == "echo" else "gen",
            "list_path": Path(list_path).as_posix(), "list_file_sha256": pre.eval_list.file_sha256,
            "list_set_sha256": pre.eval_list.set_sha256, "n": len(pre.eval_list.ids),
            "snapshot_digest": meta["snapshot_digest"], "commit": meta["export_commit"],
            "at": datetime.now().astimezone().isoformat(timespec="microseconds"), "n_written": meta["n_lines"]})
    return ExportResult(plan, w.n_written, True, meta, impl_ids, loaded, pre.stand_in, line)
