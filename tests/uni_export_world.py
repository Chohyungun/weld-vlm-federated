"""export 시험의 합성 세계 — 평가 쪽 실제 함수(등록 로더 · 영수증 · 닻 · 목록 · 분할 · 실측값 대조)가 그대로 도는 최소 구성.

| 무엇 | 자리 |
|---|---|
| 본체 체크아웃 대역 | `<tmp>/repo` — git 저장소. `configs/base.yaml`(닻 · 프로세서 인자 · 통합형 설정 키)을 `main` 에 커밋한다 |
| 동결 스냅샷 | `<tmp>/snap` — `data.manifest_io.write_snapshot`. 목록의 넷은 val, 하나는 eval |
| 리허설 루트 | `<tmp>/repo/outputs/rehearsal_u/reh-…` — 목록 · 계획 · 등록과 영수증 · 학습 산출물 · export 폴더 |
| 등록 | 평가 쪽 `prereg_unified` 로 만들고 `dump_registration` 바이트로 쓴다 — 이 모듈이 지어내는 것은 값뿐이다 |
| 노출 원장 | `<tmp>/exposure/uni_exposure_ledger.jsonl` — 두 목록의 `planned` 줄 |
| 코드 저장소 대역 | `<tmp>/code` — 커밋된 파일 하나뿐인 깨끗한 저장소(작업 트리 청결 검사의 이음새) |

이음새 대역 둘 — 실측(`measure`)은 등록과 같은 값을, 생성기 적재(`Loader`)는 정해진 생성문을 낸다. 둘 다 리허설 목적에서만 받는다.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
REH = "reh-20261001T000000"
IDS = ["aihub000001", "aihub000002", "aihub000003", "aihub000004"]
EVAL_ID = "aihub000009"
PROMPT_REL = "vlm/prompts/unified_v2_absorig.txt"
MAX_PIXELS = 1280 * 704
DEVICE = "NVIDIA GeForce RTX 5060 Ti:0"
RUN_ID = "uni_local_C1_s7_t1"


def hx(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                          capture_output=True, check=True).stdout.decode("utf-8").strip()


MEASURED = {"processor_config_sha256": hx("proc"), "processor_min_pixels": 256, "processor_max_pixels": MAX_PIXELS,
            "patch_size": 16, "merge_size": 2, "gen_prefix_sha256": hx("prefix"), "transformers_version": "5.15.0"}


def measure(spec):
    """실측 이음새의 대역 — 프로세서를 열지 않고 등록과 같은 값을 낸다."""
    return dict(MEASURED)


def gen(text_of=None, latency_ms=1.0):
    from vlm.export_run import GenOut

    def g(img, image_id):
        text = text_of(image_id) if text_of else (
            '{"defects": [{"iso_code": "2011", "bbox_2d": [10, 20, 40, 55]}], "verdict": "합격", "cited_clauses": []}')
        return GenOut(text=text, gen_stop="eos", n_new_tokens=5, model_input_wh=(1280, 704), grid_thw=(1, 44, 80),
                      token_ids=[1, 2, 3, 4, 5], token_logprobs=[-0.1] * 5, latency_ms=latency_ms)
    return g


class Loader:
    """생성기 적재 대역 — 불린 횟수와 받은 생성 설정을 센다."""

    def __init__(self, g=None):
        self.gen = g or gen()
        self.calls = 0
        self.cfg = None

    def __call__(self, gen_cfg):
        self.calls += 1
        self.cfg = gen_cfg
        return self.gen


def prompt_sha() -> str:
    from vlm.pilot_vlm import load_prompt

    return load_prompt(ROOT / PROMPT_REL)[1]


def train_config_text() -> str:
    """학습 쪽 함수가 낸 학습 설정 원문 — 손으로 적지 않는다."""
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import TARGET_CONTRACT_SHA256, canonical_json
    from vlm.train_cell import train_config_object

    metrics = {"processor_config_sha256": hx("proc"),
               "config_blocks": {k: {} for k in ("lora", "quant", "optimizer", "loss", "lr_schedule")},
               "prompt_sha256": prompt_sha(), "template_mode": "enable_thinking=False",
               "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
               "target_contract_sha256": TARGET_CONTRACT_SHA256}
    return canonical_json(train_config_object(
        model_id="Qwen/Qwen3.5-4B", model_revision="a" * 40, pairs_digest=hx("pairs"),
        chat_template_kwargs={"enable_thinking": False}, coord_space="ABS_ORIG", num_rounds=2, local_epochs=1,
        total_epochs=2, prompt_sha256=prompt_sha(), init_digest=hx("init"), metrics=metrics))


def _man(image_id: str, split: str, sha: str | None = None) -> dict:
    from data.manifest_io import MANIFEST_COLUMNS

    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{image_id}.jpg", "sha256": sha or hx(image_id),
        "width_px": 1280, "height_px": 720, "modality": "RT", "material": "ST", "has_defect": True, "n_defects": 1,
        "defect_types": "porosity", "iso_codes": "2011", "src_labels_raw": "기공", "label_type": "polygon",
        "has_localization": True, "phash_hex": "0" * 16, "group_id": f"g{image_id}", "group_size": 1,
        "strata_key": "ST|porosity", "split": split, "client": "C1" if split != "eval" else "",
        "eval_subset": "", "thickness_mm": None, "thickness_source": "unavailable", "px_per_mm": None,
        "scale_source": "unavailable", "quality_level": "", "ingest_version": "t", "label_map_version": 1, "notes": "",
    }


def _ann(image_id: str) -> dict:
    from data.manifest_io import ANNOTATION_COLUMNS

    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#0", "image_id": image_id, "src_label_raw": "기공", "defect_type": "porosity",
        "iso_code": "2011", "polygon_json": "[]", "bbox_x1_px": 10, "bbox_y1_px": 20, "bbox_x2_px": 30,
        "bbox_y2_px": 40, "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


def registration(w, *, edit=None):
    """평가 쪽 `UnifiedRegistration` 을 채운다 — 값은 이 세계의 설정 · 목록 · 실측과 같다."""
    from evaluation.actuals import impl_key
    from evaluation.prereg_unified import UnifiedRegistration
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.seams import impl_id

    r = UnifiedRegistration()
    g = r.generation
    g.purpose, g.list_split, g.snapshot_digest = "rehearsal", "val", w.digest
    g.eval_list_file_sha256 = g.echo_list_file_sha256 = w.elist.file_sha256
    g.eval_list_set_sha256 = g.echo_list_set_sha256 = w.elist.set_sha256
    g.prompt_sha256, g.chat_template_kwargs = prompt_sha(), {"enable_thinking": False}
    g.gen_prefix_sha256 = MEASURED["gen_prefix_sha256"]
    g.max_new_tokens, g.decoding, g.batch_size, g.padding_side = 512, {"do_sample": False, "num_beams": 1}, 1, "left"
    for k in ("processor_config_sha256", "processor_min_pixels", "processor_max_pixels", "patch_size", "merge_size",
              "transformers_version"):
        setattr(g, k, MEASURED[k])
    g.coord_space, g.coord_cfg_hash = "ABS_ORIG", coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG"))
    g.base_model_id, g.base_model_revision, g.init_adapter_digest = "Qwen/Qwen3.5-4B", "a" * 40, hx("init")
    g.train_config_sha256 = hashlib.sha256(train_config_text().encode("utf-8")).hexdigest()
    g.budget_n, g.budget_r, g.budget_e = 2, 2, 1
    g.impl_ids = tuple(sorted(impl_key(impl_id(fn, seam=s, approved=False))
                              for fn, s in ((measure, "export_measure"), (Loader(), "export_generator"))))
    g.train_rows_digest = hx("rows")
    g.plan_sha256 = hashlib.sha256(w.plan.read_bytes()).hexdigest()
    r.metric.class_codes, r.metric.coupling_rule = ("100", "2011", "301", "401"), "decoupled_v2"
    r.seeds = {"1": 7}
    if edit is not None:
        edit(r)
    return r


def write_registration(w, r, *, name="rehearsal-20261001-1") -> Path:
    """등록 파일 · 학습 설정 원문 · 영수증을 리허설 루트의 `registration/` 에 쓴다. 영수증 경로를 돌려준다."""
    from evaluation.prereg_unified import dump_registration

    d = w.root / "registration"
    d.mkdir(parents=True, exist_ok=True)
    raw = dump_registration(r)
    (d / f"{name}.json").write_bytes(raw)
    (d / "train_config.json").write_bytes(train_config_text().encode("utf-8"))
    receipt = {"generation_sha256": r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": w.head, "kind": "rehearsal",
               "registration_path": (d / f"{name}.json").relative_to(w.repo).as_posix(),
               "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    p = d / f"{name}.receipt.json"
    p.write_text(json.dumps(receipt), encoding="utf-8")
    return p


def write_adapter(w, *, tag="uni_local_C1", step=12, purpose="rehearsal", close=True, meta_over=None,
                  base: Path | None = None) -> Path:
    """학습 산출물 — 키 있는 어댑터 · meta · 닫힌 원장. 원장은 태그의 칸 꼴이다 — 로컬(참여자 하나) · 중앙(`central`) ·
    연합(`atomic_log.csv` — 라운드마다 참여자 셋의 `optimizer_steps` 와 `server` 행, 끝 행 `server` · 값 = 등록의 R).
    **합성 학습 산출물이다** — 연합 학습을 돈 것이 아니다. export · 평가 쪽 검증의 입력 꼴만 맞춘다."""
    import shutil

    import numpy as np

    from evaluation.schema_v14 import tag_parts
    from fl.atomic_log import AtomicLog
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.train_cell import _sha256_file

    d = (base or w.root / "train") / f"{tag}_s1"
    if d.exists():
        shutil.rmtree(d)                     # 시험의 임시 폴더 안이다 — 산출물을 새로 쓴다
    d.mkdir(parents=True)
    # 모델마다 다른 바이트 — 평가 쪽이 묶음 사이의 같은 어댑터를 거부한다(adapter_duplicate). C1 은 종전 값 그대로다
    fill = 1.0 if tag == "uni_local_C1" else 2.0 + ("uni_central", "uni_local_C2", "uni_local_C3", "uni_fed").index(tag)
    np.savez(d / "adapter_last.npz", **{"a.lora_A.weight": np.full((2, 2), fill, dtype=np.float32)})
    cell, client = tag_parts(tag)
    if cell == "uni_fed":
        rounds = int(w.reg.generation.budget_r)
        log = AtomicLog(d / "atomic_log.csv", run_id=RUN_ID, seed=7, cell=cell, split_hash=w.digest)
        for r in range(rounds):
            for c in range(3):
                log.log_round(round_idx=r, client_id=c, n_train_samples=4, metrics={"optimizer_steps": 6.0})
            log.log_round(round_idx=r, client_id="server", n_train_samples=12, metrics={"bytes_round": 100.0})
        if close:
            log.close(client_id="server", step=rounds)
        step = rounds                        # 연합 meta 의 adapter_step 은 라운드 수다(원장 끝 행의 값)
    else:
        cid = client or "central"
        log = AtomicLog(d / "train_ledger.csv", run_id=RUN_ID, seed=7, cell=cell, split_hash=w.digest)
        for ep, s in ((0, step // 2), (1, step)):
            log.log_round(round_idx=ep, client_id=cid, n_train_samples=4, metrics={"optimizer_steps": float(s)})
        if close:
            log.close(client_id=cid, step=step, n_train_samples=4)
    meta = {"adapter_sha256": _sha256_file(d / "adapter_last.npz"), "adapter_step": step,
            "identity": {"purpose": purpose, "tag": tag, "seed_index": 1}, "train_run_id": RUN_ID,
            "processor_config_sha256": MEASURED["processor_config_sha256"],
            "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")), "train_rows_digest": hx("rows"),
            "train_config": train_config_text(), "init_adapter_digest": hx("init"), **(meta_over or {})}
    (d / "adapter_last.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def build(tmp: Path, *, ids=IDS) -> SimpleNamespace:
    """합성 세계 하나. 영수증 · 학습 산출물까지 만든다."""
    import pandas as pd
    from PIL import Image

    from data.manifest_io import write_snapshot
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from vlm.exposure import append_row

    img = tmp / "img"
    img.mkdir(parents=True)
    images = {}
    for i, iid in enumerate(ids):
        Image.new("L", (1280, 720), i * 20).save(img / f"{iid}.png")
        images[iid] = img / f"{iid}.png"
    repo = tmp / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    snap = tmp / "snap"
    eval_img = img / f"{EVAL_ID}.png"                       # 평가 이미지도 실제 바이트로 — 엄격 경로가 화소 지문을 맞댄다
    Image.new("L", (1280, 720), 222).save(eval_img)
    man = [_man(i, "val", hashlib.sha256(images[i].read_bytes()).hexdigest()) for i in ids] + \
        [_man(EVAL_ID, "eval", hashlib.sha256(eval_img.read_bytes()).hexdigest())]
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([_ann(i) for i in ids]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    (repo / "configs").mkdir()
    (repo / "configs/base.yaml").write_text(
        "fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n"
        f"  snapshot_digest: {digest}\n  uni_processor_kwargs: {{max_pixels: {MAX_PIXELS}}}\n"
        "  uni_model: {id: Qwen/Qwen3.5-4B, revision: " + "a" * 40 + "}\n"
        f"  uni_prompt_path: {PROMPT_REL}\n  uni_coord_space: ABS_ORIG\n"
        "  uni_chat_template_kwargs: {enable_thinking: false}\n  uni_batch_size: 1\n  uni_max_new_tokens: null\n",
        encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "설정")
    head = git(repo, "rev-parse", "HEAD")

    parent = repo / "outputs" / "rehearsal_u"
    root = parent / REH
    (root / "lists").mkdir(parents=True)
    raw = canonical_bytes(list(ids))
    elist = validate_eval_list(raw)
    lists = {"model": root / "lists" / "gen.txt", "echo": root / "lists" / "echo.txt"}
    for p in lists.values():
        p.write_bytes(raw)
    plan = root / "plan.yaml"
    plan.write_text("kind: rehearsal\nversion: 1\ngen_limit: {max_new_tokens: 512, provisional: true}\n",
                    encoding="utf-8")
    exposure = tmp / "exposure" / "uni_exposure_ledger.jsonl"
    for kind in ("gen", "echo"):
        append_row(exposure, {"event": "planned", "run_id": REH, "purpose": "rehearsal", "tag": None,
                              "seed_index": 1, "mode": None, "list_kind": kind,
                              "list_path": lists["echo" if kind == "echo" else "model"].as_posix(),
                              "list_file_sha256": elist.file_sha256, "list_set_sha256": elist.set_sha256,
                              "n": len(ids), "snapshot_digest": digest, "commit": head,
                              "at": "2026-10-01T09:00:00.000000+09:00"})
    code = tmp / "code"
    code.mkdir()
    git(code, "init", "-q", "-b", "main")
    (code / "m.py").write_text("x = 1\n", encoding="utf-8")
    git(code, "add", "-A")
    git(code, "commit", "-q", "-m", "코드")

    w = SimpleNamespace(tmp=tmp, repo=repo, head=head, snap=snap, digest=digest, parent=parent, root=root,
                        lists=lists, elist=elist, plan=plan, exposure=exposure, images=images, code=code)
    w.reg = registration(w)
    w.receipt = write_registration(w, w.reg)
    w.adapter = write_adapter(w)
    return w


class Clock:
    def __init__(self, start=datetime(2026, 10, 1, 12, 0, 0, tzinfo=KST)):
        self.t = start

    def __call__(self):
        self.t = self.t + timedelta(milliseconds=7)
        return self.t


def run(w, *, mode="model", tag="uni_local_C1", loader=None, clock=None, purpose="rehearsal", folder=None,
        measure_fn=None, **kw):
    """`run_export` — 세계의 모든 경로와 이음새를 넘긴다. `kw` 가 덮어쓴다."""
    from vlm.export_run import run_export

    args = dict(folder=folder or w.root / ("export_echo" if mode == "echo" else "export"), tag=tag, seed_index=1,
                mode=mode, purpose=purpose, receipt_path=w.receipt, list_path=w.lists[mode], snapshot_root=w.snap,
                image_path_of=w.images, measure_env=measure_fn or measure, load_generator=loader or Loader(),
                device=DEVICE, run_root=w.root, non_main_parent=w.parent,
                adapter_dir=w.adapter if mode == "model" else None, plan_path=w.plan, repo=w.repo,
                code_repo=w.code, exposure_ledger=w.exposure, clock=clock or Clock(), env={})
    args.update(kw)
    return run_export(**args)


def verify(w, *, mode="model", tag="uni_local_C1", folder=None):
    """평가 쪽 검증기에 **평가 쪽 투영**(`actuals.registered_values`)과 원장 판독기(`read_ledger_view`)로 태운다."""
    from evaluation.actuals import registered_values
    from evaluation.bundle_v14 import BundleFiles, Registration, TrainingArtifact, verify_bundle
    from evaluation.ledger_v14 import read_ledger_view

    reg = w.reg
    proj = Registration(generation_sha256=reg.generation_sha256(), coupling_rule="decoupled_v2",
                        seeds={int(k): v for k, v in reg.seeds.items()}, values=registered_values(reg, mode=mode),
                        eval_list=w.elist, purpose="rehearsal", echo_list=w.elist, coord_space="ABS_ORIG",
                        train_rows_digest=reg.generation.train_rows_digest)
    art = led = None
    if mode == "model":
        m = json.loads((w.adapter / "adapter_last.meta.json").read_text(encoding="utf-8"))
        art = TrainingArtifact(m["adapter_sha256"], m["adapter_step"], m["train_rows_digest"])
        led = read_ledger_view(w.adapter / "train_ledger.csv")
    stem = f"{tag}_s1" + (".echo" if mode == "echo" else "")
    folder = folder or w.root / ("export_echo" if mode == "echo" else "export")
    shas = {i: hashlib.sha256(p.read_bytes()).hexdigest() for i, p in w.images.items()}
    return verify_bundle(BundleFiles(folder / f"{stem}.generations.jsonl"), registration=proj, artifact=art,
                         ledger=led, receipt_time=datetime(2026, 9, 21, 9, 0, tzinfo=KST), image_sha256_of=shas)


def attach(tmp: Path) -> SimpleNamespace:
    """이미 만든 세계에 다시 붙는다 — 자식 프로세스와 부모가 같은 세계를 본다(파일만 읽는다)."""
    from evaluation.eval_list import validate_eval_list
    from evaluation.prereg_unified import load_registration

    repo = tmp / "repo"
    parent = repo / "outputs" / "rehearsal_u"
    root = parent / REH
    lists = {"model": root / "lists" / "gen.txt", "echo": root / "lists" / "echo.txt"}
    reg_files = sorted((root / "registration").glob("rehearsal-*-?.json"))
    w = SimpleNamespace(tmp=tmp, repo=repo, head=git(repo, "rev-parse", "HEAD"), snap=tmp / "snap", parent=parent,
                        root=root, lists=lists, elist=validate_eval_list(lists["model"].read_bytes()),
                        plan=root / "plan.yaml", exposure=tmp / "exposure" / "uni_exposure_ledger.jsonl",
                        images={i: tmp / "img" / f"{i}.png" for i in IDS}, code=tmp / "code",
                        adapter=root / "train" / "uni_local_C1_s1")
    w.reg = load_registration(reg_files[-1].read_bytes())
    w.digest = w.reg.generation.snapshot_digest
    w.receipt = reg_files[-1].with_name(reg_files[-1].stem + ".receipt.json")
    return w
