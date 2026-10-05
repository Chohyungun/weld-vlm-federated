"""리허설 오케스트레이터의 합성 세계 — 모델 없이 단계 표를 끝까지 돈다(리허설 2판 §6-1).

| 무엇 | 어디 |
|---|---|
| 본체 체크아웃 자리 | `tmp/repo`(`git init` · 본줄기에 설정과 골든 픽스처를 커밋) — 비본실험 부모 `repo/outputs/rehearsal_u` · 채점 원장 `repo/outputs/main_u/` |
| 동결 스냅샷 | `tmp/snap` — train · val · eval 행, val 주석(에코 정답) |
| 본실험 페어 | `tmp/pairs/pairs.jsonl` + `SNAPSHOT.sha256` — train · val 행(이미지는 1280×720 단색 PNG) |
| 근사 중복 간선 | `tmp/edges.csv` — val 한 장과 eval 한 장을 잇는다(그 묶음 전체가 빠진다) |
| 노출 원장 | `repo/outputs/exposure/uni_exposure_ledger.jsonl` — 정본과 같은 꼴(체크아웃의 `outputs/` 아래). 실행 전부터 있던 파일 `repo/outputs/keep/old.txt` 도 둔다 |
| 코드 저장소 | `tmp/code` — 깨끗한 저장소(export 의 작업 트리 청결 검사가 이것을 본다) |
| 계획 | `tmp/plan.yaml` — 작은 크기 · 중단 계획(줄 A 넷 · C3 둘) · 단계 상한 |

대역은 `tests/uni_fakes.py` 의 작은 모델(`fake_loader`, 프로세서 `FakeProcessor`)과 그 프로세서로 잰 실측, 그리고 학습 타깃을 생성문으로 내는
가짜 생성기다. 자식 프로세스는 `tests/uni_rehearsal_child.py` 가 이 세계에 다시 붙어 이음새를 **파이썬 인자로** 넘긴다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ("C1", "C2", "C3")
PROMPT_REL = "vlm/prompts/unified_v2_absorig.txt"
FIXTURES_REL = "vlm/coords_fixtures/golden_fixtures.json"
SEEDS = (20260828, 20260829, 20260830)
BOX = [100, 120, 300, 340]


def hx(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          capture_output=True, check=True).stdout.decode("utf-8").strip()


def git_blob(repo: Path, rev: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{rev}:{path}"], capture_output=True,
                          check=True).stdout


def fake_d_verdict_then_crash(w, argv) -> int:
    """평가 쪽 진단 호출이 **판정 줄을 쓴 직후 죽은** 꼴 — 정본 원장에 `call` 줄과 판정 줄을 쓰고 산출물 없이 끝난다(종료 1).
    판정 줄의 값은 시험이 지어낸 것이다(판정을 계산하지 않는다). 다시 만들기는 실제 평가 쪽 진입점이 이 줄에서 한다."""
    from datetime import datetime

    from evaluation.prereg_unified import read_receipt
    from evaluation.scoring_ledger import append_line

    a = dict(zip(argv[::2], argv[1::2]))
    gen = read_receipt(Path(a["--receipt"])).generation_sha256
    bundles = [{"tag": "uni_central", "seed_index": 1, "mode": "echo" if ".echo." in p.name else "model",
                "generations_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
               for p in sorted(Path(a["--bundles"]).glob("*.generations.jsonl"))]
    ledger = w.repo / "outputs" / "main_u" / "scoring_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    call = append_line(ledger, "call", {"purpose": "frame_diag", "run_kind": "frame_diag", "generation_sha256": gen,
                                        "bundles": bundles}, at=now)
    out_rel = (Path(a["--out"]) / "score" / "frame_diag_score.json").resolve().relative_to(w.repo.resolve()).as_posix()
    append_line(ledger, "frame_diag_verdict",
                {"run_kind": "frame_diag", "call_sha256": call, "generation_sha256": gen,
                 "bundles": [b for b in bundles if b["mode"] == "model"], "status": "통과", "candidate": None,
                 "reason": None, "n_images": 3, "n_pairs": 0, "n_no_pred": 0, "n_count_mismatch": 0, "n_groups": 3,
                 "n_image_groups": 3, "n_pair_groups": 0, "samples": {}, "n_limit_no_pair": 0, "output": out_rel}, at=now)
    return 1


def fake_echo_gate(w, argv, *, status: str, reasons: list[str], generation_sha256: str | None = None,
                   other_echo_list: bool = False) -> int:
    """평가 쪽 **에코 전용 호출**의 대역 — 인자 계약(평가 쪽 13번 보고)대로 `<out>/score/echo_gate.json` 을 쓰고 종료 0 · 2. 대역 묶음으로는
    실제 진입점이 통과를 내지 않으므로(`stand_in`) 통과 · 불일치의 갈래를 이 대역으로 본다. 에코 전용이 아닌 호출은 실제 진입점으로 넘긴다."""
    if "--echo-only" not in argv:
        import scripts.probe.score_unified as SU

        return SU.main(argv, _seam=SU.Seam(checkout=w.repo, repo=w.repo))
    from evaluation.prereg_unified import read_receipt

    rest = [x for x in argv if x != "--echo-only"]
    a = dict(zip(rest[::2], rest[1::2]))
    rc = read_receipt(Path(a["--receipt"]))

    def sha(p) -> str:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    from scripts.probe.score_unified import (  # 평가 쪽이 묶음을 고르는 규칙 그대로
        _NAME,
        _bundle_files,
    )

    inputs = {p.name: sha(p) for p in _bundle_files(Path(a["--bundles"])) if _NAME.match(p.name)["echo"]}
    inputs |= {"generation_list": sha(a["--generation-list"]), "echo_list": sha(a["--echo-list"]), "receipt": sha(a["--receipt"])}
    if other_echo_list:                                       # 다른 에코 목록을 본 산출(등록 지문은 맞다)
        inputs["echo_list"] = "0" * 64
    body = {"kind": "echo_gate", "status": status, "reasons": reasons, "purpose": a["--purpose"], "registration_mode": "probe",
            "fingerprints": {"generation_sha256": generation_sha256 or rc.generation_sha256,
                             "registration_file_sha256": rc.registration_file_sha256},
            "inputs": inputs, "ledger": "쓰지 않았다 — 대역"}
    out = Path(a["--out"]) / "score" / "echo_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "pass" else 2


def token_counter(rows, *, config):
    """목록 단계의 타깃 길이 대역 — 결함 하나에 100 토큰. 프로세서를 열지 않는다."""
    return {str(r["image_id"]): 100 * len(r["skeleton"]["defects"]) + 10 for r in rows}


def _man(image_id: str, split: str, group: str, sha: str, client: str, stratum: str) -> dict:
    from data.manifest_io import MANIFEST_COLUMNS

    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{image_id}.jpg", "sha256": sha,
        "width_px": 1280, "height_px": 720, "modality": "RT", "material": stratum.split("|")[0], "has_defect": True,
        "n_defects": 1, "defect_types": "porosity", "iso_codes": "2011", "src_labels_raw": "기공", "label_type": "polygon",
        "has_localization": True, "phash_hex": "0" * 16, "group_id": group, "group_size": 1, "strata_key": stratum,
        "split": split, "client": client if split != "eval" else "", "eval_subset": "", "thickness_mm": None,
        "thickness_source": "unavailable", "px_per_mm": None, "scale_source": "unavailable", "quality_level": "",
        "ingest_version": "t", "label_map_version": 1, "notes": "",
    }


def box_of(k: int) -> list[int]:
    """장마다 다른 박스 — 참여자마다 학습 내용이 달라야 어댑터가 겹치지 않는다(평가 쪽의 어댑터 중복 검사)."""
    return [BOX[0] + 7 * k, BOX[1] + 3 * k, BOX[2] + 5 * k, BOX[3] + 2 * k]


def _ann(image_id: str, box: list[int]) -> dict:
    from data.manifest_io import ANNOTATION_COLUMNS

    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#0", "image_id": image_id, "src_label_raw": "기공", "defect_type": "porosity",
        "iso_code": "2011", "polygon_json": "[]", "bbox_x1_px": box[0], "bbox_y1_px": box[1], "bbox_x2_px": box[2],
        "bbox_y2_px": box[3], "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


PLAN = """kind: rehearsal
version: 1
sampling_seed: 7
seed_index: 1
cells: [{cells}]
gen: {{split: val, defect_only: false, target_n: 4, max_n: 8, min_groups_per_stratum: 1, max_target_tokens: null}}
echo: {{same_as_gen: true}}
train: {{split: train, rows: {{C1: 3, C2: 2, C3: 2}}, avoid_multiple_of_accum: true, central_rows: union, fed_rows: same_as_local}}
amount: {{n: 2, r: 2, e: 1}}
exclude: {{near_dup_edges: {{path: '{edges}', sha256: {edges_sha}}}, drop_components_touching_eval: true}}
gen_limit: {{max_new_tokens: 512, provisional: true}}
faults:
  train:
    uni_central: 'train:after_ckpt:uni_central:ep=0'
    uni_local_C3: 'train:before_ckpt:uni_local_C3:ep=0'
  export:
    uni_central:
      - {{fault: 'export:after_lines:uni_central:n=1'}}
      - {{fault: 'export:torn_line:uni_central:n=2'}}
      - {{stop_after_lines: 1}}
      - {{to_end: true}}
    uni_local_C3:
      - {{fault: 'export:after_all_lines:uni_local_C3:-'}}
      - {{to_end: true}}
negatives: [policy_blank, identity_key_missing, identity_key_extra, format_mismatch]
timeouts_s: {{lists: 240, register: 600, train: 900, export: 600, score: 600}}
optional: {{t_eq_uni: false, dtype_compare: false, logprob_invariance: false}}
"""


def build(tmp: Path, *, cells: str = "uni_central, uni_local_C1, uni_local_C2, uni_local_C3",
          val_defects: bool = True) -> SimpleNamespace:
    """`val_defects=False` 면 val 장에 결함이 없다(주석도 없다) — 에코의 기대 박스 수가 0 인 세계."""
    import pandas as pd
    from PIL import Image

    from data.manifest_io import write_snapshot

    tmp = Path(tmp)
    img = tmp / "img"
    img.mkdir(parents=True)
    man, ann, pairs = [], [], []
    k = 0

    def picture(i: str) -> str:
        nonlocal k
        k += 1
        p = img / f"{i}.png"
        Image.new("L", (1280, 720), (k * 13) % 250).save(p)
        return hashlib.sha256(p.read_bytes()).hexdigest()

    # val — 층 둘 · 묶음 다섯(묶음 gv0 은 eval 과 간선으로 이어져 통째로 빠진다)
    for g, (stratum, n) in enumerate((("ST|porosity", 2), ("ST|porosity", 1), ("AL|porosity", 2), ("AL|porosity", 1),
                                      ("ST|porosity", 1))):
        for j in range(n):
            i = f"v{g}{j}"
            sha = picture(i)
            box = box_of(k)
            man.append(_man(i, "val", f"gv{g}", sha, "C1", stratum))
            if val_defects:
                ann.append(_ann(i, box))
            pairs.append({"image_id": i, "image_path": str(img / f"{i}.png"), "client": "C1", "split": "val",
                          "skeleton": {"defects": [{"type": "2011", "bbox_px": box}] if val_defects else [],
                                       "verdict": "불합격" if val_defects else "합격", "clauses": []}})
    # train — 참여자마다 한 장 묶음 넷
    for c in CLIENTS:
        for g in range(4):
            i = f"t{c}{g}"
            sha = picture(i)
            man.append(_man(i, "train", f"gt{c}{g}", sha, c, "ST|porosity"))
            pairs.append({"image_id": i, "image_path": str(img / f"{i}.png"), "client": c, "split": "train",
                          "skeleton": {"defects": [{"type": "2011", "bbox_px": box_of(k)}], "verdict": "불합격",
                                       "clauses": []}})
    for j in range(2):
        man.append(_man(f"e{j}", "eval", f"ge{j}", hx(f"e{j}"), "", "ST|porosity"))
    snap = tmp / "snap"
    from data.manifest_io import ANNOTATION_COLUMNS

    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame(ann, columns=list(ANNOTATION_COLUMNS)),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    pdir = tmp / "pairs"
    pdir.mkdir()
    pf = pdir / "pairs.jsonl"
    pf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs), encoding="utf-8")
    h = hashlib.sha256(pf.read_bytes()).hexdigest()
    pdig = hashlib.sha256(h.encode()).hexdigest()
    (pdir / "SNAPSHOT.sha256").write_text(f"{h}  pairs.jsonl\n# snapshot_digest {pdig}\n", encoding="utf-8")
    edges = tmp / "edges.csv"
    edges.write_text("a_id,b_id\nv00,e0\n", encoding="utf-8")

    repo = tmp / "repo"
    (repo / "configs").mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    cfg = repo / "configs" / "base.yaml"
    cfg.write_text(
        "experiment:\n  seeds: [" + ", ".join(map(str, SEEDS)) + "]\n"
        "fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n"
        f"  snapshot_digest: {digest}\n  uni_processor_kwargs: {{max_pixels: 1048576}}\n"
        "  uni_model: {id: local/tiny, revision: '" + "b" * 40 + "'}\n"
        f"  uni_pairs: {{path: '{pf.as_posix()}', digest: {pdig}}}\n"
        f"  uni_prompt_path: {PROMPT_REL}\n  uni_coord_space: ABS_ORIG\n"
        "  uni_chat_template_kwargs: {enable_thinking: false}\n  uni_batch_size: 1\n  uni_max_new_tokens: null\n",
        encoding="utf-8")
    (repo / FIXTURES_REL).parent.mkdir(parents=True)
    (repo / FIXTURES_REL).write_bytes((ROOT / FIXTURES_REL).read_bytes())
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "설정")
    code = tmp / "code"
    code.mkdir()
    git(code, "init", "-q", "-b", "main")
    (code / "m.py").write_text("x = 1\n", encoding="utf-8")
    git(code, "add", "-A")
    git(code, "commit", "-q", "-m", "코드")
    plan = tmp / "plan.yaml"
    plan.write_text(PLAN.format(cells=cells, edges=edges.as_posix(),
                                edges_sha=hashlib.sha256(edges.read_bytes()).hexdigest()), encoding="utf-8")
    w = SimpleNamespace(tmp=tmp, repo=repo, config=cfg, snap=snap, digest=digest, pairs=pf, edges=edges, code=code,
                        plan=plan, parent=repo / "outputs" / "rehearsal_u",
                        exposure=repo / "outputs" / "exposure" / "uni_exposure_ledger.jsonl",
                        scoring=repo / "outputs" / "main_u" / "scoring_ledger.jsonl")
    keep = repo / "outputs" / "keep" / "old.txt"                 # 실행 전에 있던 루트 밖 파일 — 보존되어야 한다(2판 §3 ④)
    keep.parent.mkdir(parents=True)
    keep.write_text("앞 실행의 산출물", encoding="utf-8")
    (tmp / "world.json").write_text(json.dumps({k: str(v) for k, v in vars(w).items()}), encoding="utf-8")
    return w


def attach(tmp: Path) -> SimpleNamespace:
    d = json.loads((Path(tmp) / "world.json").read_text(encoding="utf-8"))
    return SimpleNamespace(**{k: Path(v) if k != "digest" else v for k, v in d.items()})


# ---------------------------------------------------------------- 대역
class _Measurable:
    """`FakeProcessor` 를 실측 함수가 읽는 꼴로 보인다 — 이미지 프로세서의 `to_dict` 는 그대로(설정 해시가 학습과 같다)."""

    def __init__(self, proc):
        self._p = proc
        cfg = proc.image_processor.cfg
        ip = proc.image_processor
        ip.size = {"shortest_edge": int(cfg["min_pixels"]), "longest_edge": int(cfg["max_pixels"])}
        ip.patch_size, ip.merge_size = int(cfg["patch_size"]), int(cfg["merge_size"])
        self.image_processor = ip

    def apply_chat_template(self, *a, **k):
        return self._p.apply_chat_template(*a, **k)


def measure(spec):
    """실측 이음새의 대역 — 학습 대역과 같은 프로세서를 연다."""
    from tests.uni_fakes import FakeProcessor
    from vlm.export_measure import measure_processor

    return measure_processor(_Measurable(FakeProcessor(**dict(spec.processor_kwargs or {}))),
                             prompt_text=spec.prompt_text, chat_template_kwargs=spec.chat_template_kwargs)


def fake_model_loader(*a, **k):
    from tests.uni_fakes import fake_loader

    return fake_loader(*a, **k)


def fed_runner(non_main_parent: Path):
    """연합 단계의 flwr 실행 대역 — **전송 계층만 뺀다.** 서버 쪽은 두 서버 경로가 함께 쓰는 부품(`UniFedRun` · 라운드 기록 ·
    회계 감사 · 저장 · 끝 행)과 초기 어댑터 적재(`fl.server_app._load_initial` — `uni-init-adapter` 를 지난다)를 그대로 부르고,
    클라이언트는 실제 `run_client_round`(리허설 목록의 행 — `uni-train-lists`)로 돈다. 모델 적재기만 합성 대역(CPU)이다.
    집계는 산술 평균으로 대신한다(집계 산술은 이 대역의 대상이 아니다). 실제 `flwr run` 경로는 GPU 창이 본다(2판 §6-2 의 5)."""

    def run(items: dict, log_path: Path) -> None:
        import numpy as np

        from fl.atomic_log import AtomicLog, RoundTimer, new_run_id
        from fl.client_vlm import run_client_round
        from fl.round_wiring import finalize_accounting, make_round_recorder
        from fl.server_app import (
            _cell_from_metrics,
            _load_initial,
            _with_uni_observer,
            build_accounting,
            cell_out_dir,
        )
        from fl.uni_fed import UniFedRun, assert_fresh_ledger
        from fl.uni_run_config import client_run_cfg, down_config
        from vlm import pilot_vlm

        pilot_vlm._load_model = fake_model_loader          # 연합 경로의 적재기 — 리허설만 받는다(대역으로 적힌다)
        def get(k, d):                                     # 서버 설정을 읽는 함수의 꼴
            return items.get(k, d)

        R, E = int(items["num-server-rounds"]), int(items["local-epochs"])
        tags = str(items["client-tags"]).split(",")
        out = cell_out_dir("uni_fed", items)
        run_id = new_run_id("uni_fed", int(items["base-seed"]), str(items["run-stamp"]))
        uni_run = UniFedRun(get=get, out_dir=out, run_id=run_id, base_seed=int(items["base-seed"]),
                            split_hash=str(items["split-hash"]), num_rounds=R, local_epochs=E,
                            total_epochs=int(items["total-epochs"]), client_tags=tags, non_main_parent=non_main_parent)
        record, keys, _ref = _load_initial("uni_fed", items)
        arrays = record.to_numpy_ndarrays()
        uni_run.set_initial(arrays, keys)
        ledger = Path(out) / "atomic_log.csv"
        assert_fresh_ledger(ledger)
        atomic = AtomicLog(ledger, run_id=run_id, seed=int(items["base-seed"]), cell="uni_fed",
                           split_hash=str(items["split-hash"]))
        accounting = build_accounting(num_rounds=R, client_ids=list(range(len(tags))), local_epochs=E,
                                      total_epochs=R * E)
        on_round_end = _with_uni_observer(make_round_recorder(
            accounting=accounting, atomic=atomic, timer=RoundTimer(), cell_from_metrics=_cell_from_metrics), uni_run)
        uni = client_run_cfg(down_config(get))
        glob = [a.copy() for a in arrays]
        for sr in range(1, R + 1):
            cells, outs = [], []
            for i, tag in enumerate(tags):
                arr, metrics, strings = run_client_round(
                    adapter_in=glob, canonical_keys=keys, round_idx=sr - 1, client_idx=i,
                    cfg={"client_tag": tag, "local_epochs": E, "num_rounds": R, "base_seed": int(items["base-seed"]),
                         "resume_root": None, "run_id": str(items["run-stamp"]), "uni": uni})
                cells.append({"node_id": i, **metrics, **strings})
                outs.append(arr)
            glob = [np.mean([o[j] for o in outs], axis=0).astype(outs[0][j].dtype) for j in range(len(keys))]
            on_round_end(sr, cells, SimpleNamespace(ndarrays=glob, total_examples=sum(1 for _ in outs),
                                                    global_norm=0.0, bn_buffer_divergence=0.0,
                                                    missing_variance_ratio=0.0))
        report = finalize_accounting(accounting=accounting, atomic=atomic, out_dir=Path(out), num_rounds=R,
                                     client_ids=list(range(len(tags))), raise_on_failure=False)
        if not report.ok:
            raise RuntimeError(f"연합 회계 감사가 지나지 않았다: {report.failures}")
        uni_run.finish(atomic=atomic, report=report)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(json.dumps({"runner": "in_process", "rounds": R, "clients": tags}) + "\n",
                                  encoding="utf-8")

    return run


class _GridProc:
    """에코 생성기와 가짜 모델 생성기의 프로세서 — 학습 대역과 같은 격자 규칙(`[1, h//16, w//16]`)."""

    class _IP:
        patch_size = 16

        def __call__(self, images, return_tensors=None):
            w, h = images[0].size
            return {"image_grid_thw": [[1, max(h // 16, 1), max(w // 16, 1)]]}

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": list(range(len(text) // 4 + 1))}

    image_processor = _IP()
    tokenizer = _Tok()


def echo_loader(cfg):
    from vlm.export_echo import load_echo_generator

    return load_echo_generator(cfg, processor=_GridProc())


def model_generator_loader(cfg):
    """가짜 모델 생성기 — val 행의 학습 타깃을 생성문으로 낸다(형식이 맞는 줄). 모델을 부르지 않는다."""
    from dataclasses import replace

    from vlm.export_echo import load_echo_generator
    from vlm.export_run import GenOut

    echo = load_echo_generator(replace(cfg, mode="echo"), processor=_GridProc())

    def g(img, image_id):
        o = echo(img, image_id)
        return GenOut(text=o.text, gen_stop="eos", n_new_tokens=o.n_new_tokens, model_input_wh=o.model_input_wh,
                      grid_thw=o.grid_thw, latency_ms=3.0)

    return g


def generator_loader(cfg):
    """두 모드가 같이 지나는 대역 적재 함수 — 실제 구현(`vlm.export_generator.load_generator`)과 같은 꼴이다."""
    return echo_loader(cfg) if cfg.mode == "echo" else model_generator_loader(cfg)


def child_cmd(tmp: Path):
    """오케스트레이터의 `launcher` — 자식이 이 세계에 다시 붙어 대역을 파이썬 인자로 넘긴다."""
    shim = ROOT / "tests" / "uni_rehearsal_child.py"

    def launcher(entry, argv):
        return [sys.executable, "-X", "utf8", str(shim), str(tmp), entry, *argv]

    return launcher
