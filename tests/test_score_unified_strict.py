"""엄격 본채점 **표본 모드** — 진입점을 부른다(35번 §1 · 외부 검토 137-2 I-05).

합성 세계 — `git init` 한 저장소(본체 체크아웃 자리)에 base 설정 · 평가 등록 설정 · 본실험 등록을 커밋하고, 그 커밋을 가리키는 `main`
영수증 · 봉인 목록과 곁 파일 · 등록 대조선 예측 · 다섯 모델과 에코 묶음 · 학습 원장을 둔다. 평가 정답은 데이터 쪽 엄격 판독기 자리
(`SU.EVAL_VIEW`)를 시험 판독기로 바꿔 끼운다 — 그 판독기는 이음새가 없다(본체 저장소를 본다). 실물 평가 자산을 열지 않는다.

지키는 것.

1. 승인된 표본 실행은 엄격 채점을 끝까지 지나 산출 하나를 배타 생성하고 종료 0 이다 — 원장의 `call` 한 줄이 표본 입력의 해시를 싣는다.
2. 곁 파일 · 대조선 · 봉인 · 결속이 어긋나면 **원장 줄 전에** 종료 3 이다(정답 · 묶음 검증이 불리지 않는다).
3. 묶음이 쪼개진 표본은 목록 단계에서 거부한다. 같은 모집단의 두 번째 산출은 덮지 않는다.
4. 거부된 모델 칸은 산출에 이름으로 남고 파생값은 `산출 불가` 이며 종료 2 다.
5. 평가 등록 설정은 영수증의 커밋에서 읽는다 — 작업 트리를 바꿔도 산출이 따르지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd
import pytest
import yaml

import scripts.probe.score_unified as SU
from data.eval_sample import SEED as SAMPLE_SEED
from data.eval_sample import SIZES
from data.manifest_io import write_snapshot
from evaluation import early_coord_check as E
from evaluation import ecc_frame_check as FC
from evaluation import known_answer as KA
from evaluation.coord_check import coord_fixture_digest
from evaluation.eval_list import canonical_bytes, validate_eval_list
from evaluation.eval_registration import EXTRA_SEALED_SIZES
from evaluation.prereg_unified import CoordCfgRecord, dump_registration
from evaluation.sample_weights import SIDE_HEADER
from evaluation.scoring_ledger import read_ledger
from tests import test_early_coord_check as TEC
from tests import test_ecc_frame_check as TFC
from tests.test_bundle_v14 import IDS, IMPL_KEYS, hx, make, train_config_text
from tests.test_prereg_unified import filled
from tests.test_score_unified import KST, REPO, _ann, _man, git
from tests.test_scoring_layer import gen_row

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=KST)
CLASSES = ("100", "2011", "301", "401")
REG_REL = "configs/registration/main-20261014-1.json"
PID = "main-sample-3000"
BOX = [10.0, 20.0, 30.0, 40.0]
FAR = [500.0, 500.0, 520.0, 520.0]

# 묶음 → (재질, 이미지들). 층 = 재질|구간. gZ 는 평가셋이지만 표본 밖 — 표본이 평가 전량이 아님을 보인다.
SAMPLE = {"gA": ("ST", ["aihub100001", "aihub100002"]), "gB": ("ST", ["aihub100003", "aihub100004"]),
          "gC": ("AL", ["aihub100005"]), "gD": ("AL", ["aihub100006", "aihub100007"])}
OTHER = {"gZ": ("ST", ["aihub100008", "aihub100009"])}
STRATUM = {"gA": ("ST|1", 3, 2), "gB": ("ST|1", 3, 2), "gC": ("AL|0", 2, 2), "gD": ("AL|0", 2, 2)}
GOLD = {"aihub100001": [("100", BOX)], "aihub100002": [("2011", BOX)], "aihub100003": [("301", BOX)],
        "aihub100004": [("401", BOX)], "aihub100005": [("2011", BOX)], "aihub100006": [],
        "aihub100007": [("100", BOX), ("2011", BOX)], "aihub100008": [("2011", BOX)], "aihub100009": []}
SAMPLE_IDS = [i for _, ids in SAMPLE.values() for i in ids]
TAGS = ("uni_central", "uni_fed", "uni_local_C1", "uni_local_C2", "uni_local_C3")
HIT = {"uni_central": 7, "uni_fed": 5, "uni_local_C1": 3, "uni_local_C2": 4, "uni_local_C3": 2}
"""모델마다 정답 상자를 맞히는 이미지 수(목록 순서의 앞에서부터) — 나머지는 엉뚱한 자리에 예측한다."""


def _gold_rows(ids):
    return [{"image_id": i, "iso_code": c, "bbox_x1_px": b[0], "bbox_y1_px": b[1], "bbox_x2_px": b[2], "bbox_y2_px": b[3]}
            for i in ids for c, b in GOLD[i]]


def _ledger(tag: str) -> tuple[bytes, int]:
    head = ["run_id", "seed", "cell", "split_hash", "client_id", "round", "n_train_samples", "metric_name",
            "metric_value", "bytes_up", "bytes_down", "wall_time"]
    run = f"run_{tag}"
    rows = [head]
    if tag == "uni_fed":
        for rnd in range(3):
            rows.append([run, "20260828", "uni_fed", "d", "server", str(rnd), "0", "aggregate", "1", "0", "0", "1"])
            for c in range(3):
                rows.append([run, "20260828", "uni_fed", "d", str(c), str(rnd), "10", "optimizer_steps", "5", "0", "0", "1"])
        rows.append([run, "20260828", "uni_fed", "d", "server", "3", "0", "final_adapter_step", "3", "0", "0", "1"])
        final = 3
    else:
        cell = "uni_central" if tag == "uni_central" else "uni_local"
        client = "central" if tag == "uni_central" else tag.rsplit("_", 1)[1]
        for e, steps in ((1, 50), (2, 100)):
            rows.append([run, "20260828", cell, "d", client, str(e), "10", "optimizer_steps", str(steps), "0", "0", "1"])
        rows.append([run, "20260828", cell, "d", client, "100", "10", "final_adapter_step", "100", "0", "0", "1"])
        final = 100
    return ("\r\n".join(",".join(r) for r in rows) + "\r\n").encode("utf-8"), final


def _rows(tag: str, *, literal_break: bool = False):
    out = []
    for k, i in enumerate(SAMPLE_IDS):
        gold = GOLD[i]
        if k < HIT[tag] and gold:
            boxes = tuple((c, list(b)) for c, b in gold)
        else:
            boxes = (("2011", FAR),)
        # 문자 일치가 어긋나는 줄 — 생성문의 좌표와 파싱 · 역변환 결과가 다르다
        text_boxes = (("2011", [11.0, 20.0, 30.0, 40.0]),) if literal_break and k == 0 else None
        out.append(gen_row(i, boxes=boxes, text_boxes=text_boxes))
    return out


def _ecc_rules_raw() -> bytes:
    """조기 좌표 점검의 합성 규칙 파일 바이트 — 점검 시험의 `ready_rules(TGT)` 와 같은 칸(합성 대상 다섯 장 · 한 번 통과)."""
    b = TEC.body()
    b.update({"sequential_rule": "single", "max_attempts": 2,
              "limits": {"max_hours": 3, "max_cost_krw": 20000, "price_krw_per_hour": 2500, "price_source": "합성",
                         "safety_factor": 1.2},
              "target_list": {"sha256": TEC.TGT.file_sha256, "n": len(TEC.TGT)}})
    b["discrimination"] = TEC.synthetic_receipt(b)
    return TEC.raw_of(b)


def _permit_raw(rules_raw: bytes, *, step_status: str = E.PASS) -> bytes:
    """그 규칙의 통과 파일(평가 쪽 산출 함수 그대로) — 시드 폴더의 `ecc_permit.json` 자리."""
    obj = E.build_output(rules=E.load_rules(rules_raw), target=TEC.TGT, model_free=TEC._mf(),
                         steps=[TEC._step(status=step_status)], evidence=TEC._evidence())
    return E.canonical(obj)


def build(tmp, *, sample_ids=None, side_edit=None, baseline_edit=None, doc_edit=None, reg_edit=None,
          bundle_edit=None, skip=(), literal_break=(), known_answer_fixture=True, permit_step=E.PASS,
          frame_mode="permit", log_section=True, text_suffix=None) -> dict:
    sample_ids = list(SAMPLE_IDS if sample_ids is None else sample_ids)
    repo = tmp / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    snap = tmp / "snap"
    group_of = {i: g for g, (_, ids) in {**SAMPLE, **OTHER}.items() for i in ids}
    material = {i: m for g, (m, ids) in {**SAMPLE, **OTHER}.items() for i in ids}
    man = [_man(i, "val") for i in IDS] + [_man(i, "eval") | {"group_id": group_of[i], "material": material[i]}
                                          for i in group_of]
    digest = write_snapshot(snap, pd.DataFrame(man), pd.DataFrame([_ann(i) for i in IDS]),
                            {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only"}})
    inputs = repo / "inputs"
    (inputs / "lists").mkdir(parents=True)
    gen_raw = canonical_bytes(sample_ids)
    gen = validate_eval_list(gen_raw)
    echo_raw = canonical_bytes(IDS)
    echo = validate_eval_list(echo_raw)
    for name, raw in (("generation.list", gen_raw), ("scoring.list", gen_raw), ("echo.list", echo_raw)):
        (inputs / "lists" / name).write_bytes(raw)
    side_lines = [SIDE_HEADER] + [",".join([i, STRATUM[group_of[i]][0], str(STRATUM[group_of[i]][1]),
                                            str(STRATUM[group_of[i]][2])]) for i in gen.ids]
    side_sealed = ("\n".join(side_lines) + "\n").encode("utf-8")      # 봉인한 바이트 — 고친 파일은 이것과 달라진다
    (inputs / "side.csv").write_bytes(side_edit(side_sealed) if side_edit else side_sealed)
    base_rows = [{"image_id": i, "defects": [{"iso_code": "2011", "bbox_px": BOX}] if k % 2 == 0 else []}
                 for k, i in enumerate(sorted(group_of))]
    baseline_raw = "".join(json.dumps(r) + "\n" for r in base_rows).encode("utf-8")
    if baseline_edit:
        baseline_raw = baseline_edit(baseline_raw)
    (inputs / "baseline.jsonl").write_bytes(baseline_raw)

    (repo / "configs/registration").mkdir(parents=True)
    (repo / "configs/base.yaml").write_text(
        f"fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n  snapshot_digest: {digest}\n"
        f"  uni_processor_kwargs: {{max_pixels: {1280 * 704}}}\n  uni_max_new_tokens: 512\n", encoding="utf-8")
    n_groups = len({group_of[i] for i in sample_ids})
    sealed = {str(s): {"images": s, "groups": s // 10, "file_sha256": hx(f"f{s}"), "set_sha256": hx(f"s{s}"),
                       "side_sha256": hx(f"d{s}")} for s in (*SIZES, *EXTRA_SEALED_SIZES)}   # 판독기가 받는 봉인 키 집합 그대로
    sealed["3000"] = {"images": len(gen), "groups": n_groups, "file_sha256": gen.file_sha256,
                      "set_sha256": gen.set_sha256, "side_sha256": hashlib.sha256(side_sealed).hexdigest()}
    doc = {"kind": "eval_registration", "version": 1, "decided": "2026-10-13",
           "representative": {"metric_id": "m1_box_f1_iou_1_2_macro4", "iou_threshold": "1/2",
                              "class_codes": list(CLASSES), "macro_support_rule": "gold_support",
                              "coupling_rule": "decoupled_v2", "sensitivity_variant": "fp_per_class_every_failed_record"},
           "auxiliary": ["class_jaccard", "image_presence_f1", "m1_micro", "per_class_f1", "latency_s_per_image",
                         "exchange_bytes"],
           "population": {"snapshot_digest": digest, "split": "eval", "n_full": 12461,
                          "sample": {"seed": SAMPLE_SEED, "cluster_unit": "group",
                                     "strata": "material_x_group_mean_defect_bucket",
                                     "bucket_edges": ["0", "1", "2-3", "4+"], "min_groups_per_stratum": 2,
                                     "main_size": 3000, "weighting": "stratified_g_over_n",
                                     "side_file_header": SIDE_HEADER, "sealed": sealed, "full_addendum_d_min": 0.115,
                                     "not_judged_on_sample": ["al_subset_macro_m1", "p9_equivalence"]}},
           "generation": {"max_new_tokens": 512}, "seeds": [1]}
    if doc_edit:
        doc_edit(doc)
    (repo / "configs/eval_registration.yaml").write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                                                       encoding="utf-8")
    fixtures = REPO / "vlm/coords_fixtures/golden_fixtures.json"
    (repo / "vlm/coords_fixtures").mkdir(parents=True)
    (repo / "vlm/coords_fixtures/golden_fixtures.json").write_bytes(fixtures.read_bytes())

    r = filled(purpose="main")
    g = r.generation
    g.snapshot_digest = digest
    g.eval_list_file_sha256, g.eval_list_set_sha256 = gen.file_sha256, gen.set_sha256
    g.echo_list_file_sha256, g.echo_list_set_sha256 = echo.file_sha256, echo.set_sha256
    g.prompt_sha256, g.chat_template_kwargs, g.gen_prefix_sha256 = hx("prompt"), {"add_generation_prompt": True}, hx("prefix")
    g.max_new_tokens, g.decoding, g.batch_size, g.padding_side = 512, {"do_sample": False}, 1, "left"
    g.processor_config_sha256, g.processor_min_pixels, g.processor_max_pixels = hx("proc"), 256, 1280 * 704
    g.patch_size, g.merge_size, g.coord_space, g.coord_cfg_hash = 16, 2, "ABS_ORIG", hx("coord")
    g.base_model_id, g.base_model_revision, g.init_adapter_digest = "local/qwen3.5-4b", "frozen", hx("init")
    g.start_checkpoint_sha256 = None                    # 출발점은 하나만 — 묶음 곁 파일은 초기 어댑터를 싣는다
    g.train_config_sha256 = hashlib.sha256(train_config_text().encode("utf-8")).hexdigest()
    g.budget_n, g.budget_r, g.budget_e, g.transformers_version = 30, 3, 10, "4.57.0"
    g.impl_ids = IMPL_KEYS
    p = r.population
    p.population_id, p.weighting, p.cluster_unit = PID, "stratified_g_over_n", "group"
    p.eval_list_file_sha256, p.eval_list_set_sha256 = gen.file_sha256, gen.set_sha256
    p.n_images, p.n_clusters = len(gen), n_groups
    r.ci.cluster_unit, r.ci.n_resamples, r.ci.rng_seed, r.ci.min_defined_draws = "group", 200, 20261014, 100
    m = r.metric
    m.metric_id, m.iou_threshold, m.class_codes = "m1_box_f1_iou_1_2_macro4", "1/2", CLASSES
    m.coupling_rule, m.sensitivity_variant_id, m.macro_support_rule = ("decoupled_v2", "fp_per_class_every_failed_record",
                                                                       "gold_support")
    r.baseline.prediction_sha256 = hashlib.sha256(baseline_raw).hexdigest()
    r.canary.coord_cfg = CoordCfgRecord(accepted=(hx("coord"),), frozen_value=hx("coord"), frozen_at="2026-10-01",
                                        current_value=hx("coord"), change_scope="first_registration",
                                        equivalence_evidence=("동결 export 없음 — 첫 등록",))
    r.canary.coord_fixture_sha256 = coord_fixture_digest(fixtures.read_bytes())
    r.seeds, r.planned_seed_count = {"1": 20260828}, 1
    ka_raw = (REPO / KA.FIXTURE_REL).read_bytes()          # 기지답 픽스처 — 영수증 커밋에 두고 그 계수표 sha 를 등록한다
    if known_answer_fixture:
        (repo / KA.FIXTURE_REL).parent.mkdir(parents=True)
        (repo / KA.FIXTURE_REL).write_bytes(ka_raw)
    r.canary.known_answer_sha256 = KA.known_answer_sha256(ka_raw)
    rules_raw = _ecc_rules_raw()                            # 조기 점검 규칙(영수증 커밋) · 허가 파일(시드 폴더 자리)
    (repo / SU.ECC_RULES_REL).write_bytes(rules_raw)
    permit = _permit_raw(rules_raw, step_status=permit_step)
    (inputs / "ecc_permit.json").write_bytes(permit)
    r.canary.frame_diag_sha256 = hashlib.sha256(permit).hexdigest()
    if frame_mode == "model_free":                      # 허가 면제 — 모델 없는 통과 산출 · 면제 파일 · 의사결정로그(영수증 커밋)
        adm = TFC._admission(E.load_rules(rules_raw))
        wv = TFC._waiver(adm)
        (inputs / "admission.json").write_bytes(adm)
        (inputs / "waiver.json").write_bytes(wv)
        (repo / SU.DECISION_LOG_REL).parent.mkdir(parents=True, exist_ok=True)
        (repo / SU.DECISION_LOG_REL).write_text(TFC.LOG if log_section else "# 의사결정로그\n", encoding="utf-8")
        r.canary.frame_diag_sha256 = FC.model_free_bundle_sha256(adm, wv)
    if reg_edit:
        reg_edit(r)
    raw = dump_registration(r)
    (repo / REG_REL).write_bytes(raw)
    git(repo, "add", "configs", "vlm", *(["evaluation"] if known_answer_fixture else []),
        *(["docs"] if frame_mode == "model_free" else []))
    git(repo, "commit", "-q", "-m", "본실험 등록")
    head = git(repo, "rev-parse", "HEAD")
    receipt = {"generation_sha256": r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "main",
               "registration_path": REG_REL, "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    (inputs / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    bundles = inputs / "bundles"
    attempts = b""
    common = {"generation_sha256": r.generation_sha256(), "snapshot_digest": digest, "list_split": "eval", "batch_size": 1,
              "max_new_tokens": 512}
    for tag in (t for t in TAGS if t not in skip):
        led_raw, final = _ledger(tag)
        led = inputs / "train" / f"{tag}_s1" / "train_ledger.csv"
        led.parent.mkdir(parents=True)
        led.write_bytes(led_raw)
        adapter = hx(f"adapter_{tag}")
        (led.parent / "adapter_last.meta.json").write_text(
            json.dumps({"adapter_sha256": adapter, "adapter_step": final}), encoding="utf-8")
        led_sha = hashlib.sha256(led_raw).hexdigest()
        meta = common | {"train_run_id": f"run_{tag}", "train_ledger_sha256": led_sha,
                         "train_ledger_path": led.relative_to(repo).as_posix(), "scored_adapter_sha256": adapter,
                         "scored_adapter_step": final}
        if bundle_edit:
            meta = bundle_edit(tag, meta)
        rows = _rows(tag, literal_break=tag in literal_break) if sample_ids == SAMPLE_IDS else None
        if text_suffix and rows:                         # 생성문 뒤에 서술을 덧붙인 묶음(03 2판 §5a — M1 쪽 칸은 같아야 한다)
            for row in rows:
                row["text"] = row["text"] + text_suffix
        make(bundles, tag=tag, ids=sample_ids, rows=rows, purpose="main",
             meta_over=meta, start_over={"generation_sha256": r.generation_sha256(), "train_run_id": f"run_{tag}",
                                         "train_ledger_sha256": led_sha, "scored_adapter_sha256": adapter})
        attempts += (bundles / "attempts.jsonl").read_bytes()
    make(bundles, tag="uni_central", ids=IDS, rows=[gen_row(i) for i in IDS], purpose="main", echo=True,
         meta_over={k: v for k, v in common.items() if k != "list_split"},
         start_over={"generation_sha256": r.generation_sha256()})
    attempts += (bundles / "attempts.jsonl").read_bytes()
    (bundles / "attempts.jsonl").write_bytes(attempts)
    return {"repo": repo, "inputs": inputs, "snap": snap, "reg": r, "head": head, "gen": gen, "doc": doc}


def argv(w, *, side=True, baseline=True, ecc=True, admission=False, substitute=False) -> list[str]:
    i = w["inputs"]
    out = ["--registration", "strict", "--purpose", "main", "--receipt", str(i / "receipt.json"),
           "--snapshot", str(w["snap"]), "--bundles", str(i / "bundles"),
           "--generation-list", str(i / "lists/generation.list"), "--scoring-list", str(i / "lists/scoring.list"),
           "--echo-list", str(i / "lists/echo.list"), "--train-root", str(i / "train"),
           "--out", str(w["repo"] / "outputs/main_u"), "--device", "cpu:0"]
    if side:
        out += ["--side", str(i / "side.csv")]
    if baseline:
        out += ["--baseline", str(i / "baseline.jsonl")]
    if ecc:
        out += ["--ecc-permit", str(i / "ecc_permit.json")]
    if admission:
        out += ["--frame-admission", str(i / "admission.json"), "--ecc-waiver", str(i / "waiver.json")]
    if substitute:                                   # A3 대체 승인서(D7) — 이 세계에는 파일이 없다(인자 배타 · 부재만 본다)
        out += ["--ecc-substitute", str(i / "ecc_a3_substitute.json")]
    return out


@pytest.fixture
def view(monkeypatch):
    """데이터 쪽 엄격 판독기 자리 — 받은 id 의 합성 정답을 돌려주고 호출을 적는다."""
    calls = []

    def fake(root, ids, *, receipt):
        ids = list(ids)
        calls.append({"ids": ids, "receipt": str(receipt)})
        return pd.DataFrame(_gold_rows(ids), columns=["image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px",
                                                      "bbox_y2_px"])

    monkeypatch.setattr(SU, "EVAL_VIEW", fake)
    return calls


@pytest.fixture
def verify_calls(monkeypatch):
    seen = []
    real = SU.verify_bundle

    def spy(files, **kw):
        seen.append(files.stem)
        return real(files, **kw)

    monkeypatch.setattr(SU, "verify_bundle", spy)
    return seen


def run(w, **kw) -> int:
    return SU.main(argv(w, **kw), _seam=SU.Seam(checkout=w["repo"], repo=w["repo"]), _now=NOW)


def out_path(w):
    return w["repo"] / "outputs/main_u" / f"{SU.STRICT_OUT_PREFIX}{PID}_{w['reg'].scoring_sha256()[:12]}.json"


def ledger(w):
    return read_ledger(SU.canonical_ledger(w["repo"]))


# ---------------------------------------------------------------- 1 승인된 표본 실행

def test_허가_면제면_통과_산출과_면제_파일의_묶음으로_서고_모델_없는_검사만이라고_표지한다(tmp_path, view, capsys) -> None:
    """사용자 결정(의사결정로그 2026-10-17)으로 모델 점검 허가가 면제되면 등록 칸은 두 파일의 묶음 sha 다."""
    w = build(tmp_path, frame_mode="model_free")
    assert run(w, ecc=False, admission=True) == SU.EXIT_OK, capsys.readouterr().err
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    f = body["canaries"]["frame"]
    assert (f["status"], f["frame_diag_sha256"], f["meaning"]) == (
        "FRAME_CHECKED_MODEL_FREE_ONLY", w["reg"].canary.frame_diag_sha256, "model_free_admission_waiver_bundle_sha256")
    i = w["inputs"]
    assert (f["admission_sha256"], f["waiver_sha256"]) == (hashlib.sha256((i / "admission.json").read_bytes()).hexdigest(),
                                                         hashlib.sha256((i / "waiver.json").read_bytes()).hexdigest())
    assert "미세조정" in f["scope"] and "ecc_permit" not in body["inputs"]
    assert body["inputs"]["frame_admission"] == f["admission_sha256"] and body["inputs"]["ecc_waiver"] == f["waiver_sha256"]


def test_산출은_조기_점검_허가로_프레임을_표지하고_기지답을_재현한_기록을_싣는다(tmp_path, view, capsys) -> None:
    """목적 main 의 `canary.frame_diag_sha256` 은 조기 좌표 점검 허가 파일의 sha256 이다(의사결정로그 10-07 · 등록 판 5 추기)."""
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK, capsys.readouterr().err
    c = json.loads(out_path(w).read_text(encoding="utf-8"))["canaries"]
    permit_sha = hashlib.sha256((w["inputs"] / "ecc_permit.json").read_bytes()).hexdigest()
    assert (c["frame"]["status"], c["frame"]["ecc_permit_sha256"], c["frame"]["meaning"]) == (
        "FRAME_CHECKED_BY_ECC", permit_sha, "ecc_permit_sha256")
    assert permit_sha == w["reg"].canary.frame_diag_sha256 and "not_checked" not in json.dumps(c)
    assert c["known_answer"] == {"status": "reproduced", "sha256": w["reg"].canary.known_answer_sha256,
                                 "fixture": KA.FIXTURE_REL,
                                 "fixture_sha256": hashlib.sha256((REPO / KA.FIXTURE_REL).read_bytes()).hexdigest()}


def test_승인된_표본_실행은_엄격_채점을_지나_산출_하나를_쓴다(tmp_path, view, capsys) -> None:
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK, capsys.readouterr().err
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    assert (body["status"], body["mode"], body["registration_mode"], body["population_id"]) == ("pass", "sample", "strict", PID)
    rep = body["report"]
    assert rep["weighting"] == "stratified_g_over_n" and rep["n_images"] == len(SAMPLE_IDS) and rep["n_groups"] == 4
    assert all(rep["models"][f"{t}_s1"]["status"] == "scored" for t in TAGS)
    assert rep["rule5"]["status"] == "computed" and rep["rule5"]["verdict_authority"] == "registered"
    assert rep["rq1_recovery"]["main"]["status"] == "computed"
    assert set(rep["rq2_participants"]["main"]["rows"]) == {"uni_local_C1", "uni_local_C2", "uni_local_C3"}
    assert rep["bootstrap"]["n_resamples"] == 200 and rep["bootstrap"]["resampling"] == "within_stratum_group"
    fp = body["fingerprints"]
    assert fp["sealed"]["file_sha256"] == w["gen"].file_sha256 and fp["eval_registration_rev"] == w["head"]
    assert fp["side_sha256"] == hashlib.sha256((w["inputs"] / "side.csv").read_bytes()).hexdigest()
    assert body["canaries"]["axis_ratio"]["status"] == "not_run" and body["canaries"]["frame"]["status"] == "FRAME_CHECKED_BY_ECC"
    assert body["prior_calls"] == [] and body["code_drift"]["drifted"] is False
    assert {f"{t}_s1.generations.jsonl" for t in TAGS} <= set(body["inputs"]) and body["gold"]["n_annotation_rows_without_box"] == 0
    # 정답은 엄격 판독기로 한 번 — 표본 id 전부, 영수증 경로와 함께
    assert view == [{"ids": list(w["gen"].ids), "receipt": str(w["inputs"] / "receipt.json")}]
    calls = [row for _, row in ledger(w) if row["kind"] == "call"]
    assert len(calls) == 1 and calls[0]["registration_mode"] == "strict"
    assert {"side", "baseline", "eval_registration"} <= set(calls[0]["inputs"])


def test_산출은_같은_모집단으로_다시_쓰지_않고_시도로도_세지_않는다(tmp_path, view, capsys) -> None:
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    first = out_path(w).read_bytes()
    n_view = len(view)
    assert run(w) == SU.EXIT_ENTRY
    assert "[OUT_EXISTS]" in capsys.readouterr().err
    assert out_path(w).read_bytes() == first
    assert sum(1 for _, row in ledger(w) if row["kind"] == "call") == 1       # 원장 줄 전의 거부 — 정답도 다시 읽지 않는다
    assert len(view) == n_view


# ---------------------------------------------------------------- 2 진입 전 거부 — 원장 줄 전

@pytest.mark.parametrize(("kw", "edit", "code"), [
    ({"side": False}, {}, "SIDE_MISSING"),
    ({"baseline": False}, {}, "BASELINE_MISSING"),
    ({}, {"side_edit": lambda raw: raw.replace(b"ST|1,3,2", b"ST|1,3,2 ", 1)}, "SIDE_HASH"),
    ({}, {"reg_edit": lambda r: setattr(r.baseline, "prediction_sha256", hx("other"))}, "BASELINE_HASH"),
    ({}, {"doc_edit": lambda d: d["population"]["sample"]["sealed"]["3000"].update(images=99)}, "EVAL_REGISTRATION"),
    ({}, {"doc_edit": lambda d: d.update(seeds=[1, 2])}, "EVAL_REGISTRATION"),
    ({}, {"reg_edit": lambda r: setattr(r.ci, "cluster_unit", "component")}, "EVAL_REGISTRATION"),
    ({}, {"doc_edit": lambda d: d["representative"].update(metric_id="m1_other")}, "EVAL_REGISTRATION"),
    ({}, {"doc_edit": lambda d: d["generation"].update(max_new_tokens=256)}, "EVAL_REGISTRATION"),
    ({}, {"doc_edit": lambda d: d["population"]["sample"].update(sealed="x")}, "EVAL_REGISTRATION"),
    ({}, {"reg_edit": lambda r: setattr(r, "planned_seed_count", 3)}, "RULE5_REGISTRATION"),
    ({}, {"reg_edit": lambda r: setattr(r.population, "population_id", "본실험 표본")}, "POPULATION_ID"),
    ({}, {"reg_edit": lambda r: setattr(r.canary, "known_answer_sha256", hx("other"))}, "KNOWN_ANSWER_MISMATCH"),
    ({}, {"known_answer_fixture": False}, "KNOWN_ANSWER_FIXTURE"),
    ({"ecc": False}, {}, "FRAME_PERMIT_MISSING"),
    ({}, {"reg_edit": lambda r: setattr(r.canary, "frame_diag_sha256", hx("other"))}, "FRAME_PERMIT"),
    ({}, {"permit_step": E.FAIL}, "FRAME_PERMIT"),
    ({"admission": True}, {"frame_mode": "model_free"}, "FRAME_INPUT_CONFLICT"),
    ({"ecc": False, "admission": True}, {}, "FRAME_PERMIT_MISSING"),
    ({"substitute": True}, {}, "FRAME_INPUT_CONFLICT"),
    ({"ecc": False, "substitute": True, "admission": True}, {"frame_mode": "model_free"}, "FRAME_INPUT_CONFLICT"),
    ({"ecc": False, "substitute": True}, {}, "FRAME_PERMIT_MISSING"),
    ({"ecc": False, "admission": True}, {"frame_mode": "model_free", "log_section": False}, "FRAME_PERMIT"),
    ({"ecc": False, "admission": True}, {"frame_mode": "model_free", "reg_edit": lambda r: setattr(
        r.canary, "frame_diag_sha256", hx("other"))}, "FRAME_PERMIT"),
])
def test_표본_입력이_봉인_등록과_어긋나면_원장_줄_전에_거부한다(tmp_path, view, verify_calls, capsys, kw, edit, code) -> None:
    w = build(tmp_path, **edit)
    assert run(w, **kw) == SU.EXIT_ENTRY
    assert f"[{code}]" in capsys.readouterr().err
    assert ledger(w) == [] and view == [] and verify_calls == []


@pytest.mark.parametrize(("kw", "code"), [({}, "SIDE_IN_FULL_MODE"), ({"side": False}, "BASELINE_IN_FULL_MODE")])
def test_전량_모드에_표본_입력을_주면_거부한다(tmp_path, view, capsys, kw, code) -> None:
    w = build(tmp_path, reg_edit=lambda r: setattr(r.population, "weighting", "none"))
    assert run(w, **kw) == SU.EXIT_ENTRY
    assert f"[{code}]" in capsys.readouterr().err


# ---------------------------------------------------------------- 3 목록 단계 — 묶음이 쪼개진 표본

def test_묶음이_쪼개진_표본은_목록_단계에서_거부한다(tmp_path, view, verify_calls, capsys) -> None:
    split = [i for i in SAMPLE_IDS if i != "aihub100002"]          # gA 의 한 장만
    w = build(tmp_path, sample_ids=split)
    assert run(w) == SU.EXIT_ENTRY
    assert "[MAIN_SAMPLE_GROUP_SPLIT]" in capsys.readouterr().err
    assert view == [] and verify_calls == []
    assert [row["kind"] for _, row in ledger(w)] == ["call"]       # 원장 줄 뒤의 거부 — 시도로 남는다


# ---------------------------------------------------------------- 4 거부된 모델 칸

def test_거부된_모델_칸은_이름으로_남고_파생값은_산출_불가이며_종료_2(tmp_path, view, capsys) -> None:
    w = build(tmp_path, bundle_edit=lambda tag, meta: meta | {"scored_adapter_step": 4} if tag == "uni_fed" else meta)
    assert run(w) == SU.EXIT_REJECTED
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    rep = body["report"]
    assert body["status"] == "partial" and rep["models"]["uni_fed_s1"]["status"] == "rejected"
    assert "uni_fed_s1" in body["bundles"]["rejected"]
    assert rep["rq1_recovery"]["main"] == {"status": "산출 불가", "missing_cells": ["uni_fed_s1"]}
    assert rep["rule5"]["status"] == "산출 불가"


# ---------------------------------------------------------------- 5 영수증 커밋의 설정

def test_작업_트리의_평가_등록_설정을_바꿔도_영수증_커밋의_것을_읽는다(tmp_path, view) -> None:
    w = build(tmp_path)
    p = w["repo"] / "configs/eval_registration.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    doc["population"]["sample"]["full_addendum_d_min"] = 0.9
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    assert run(w) == SU.EXIT_OK
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    assert body["report"]["full_addendum"]["d_min"] == 0.115


# ---------------------------------------------------------------- 6 빠진 칸 · 환경 불일치(검토 F1 · F3)

def test_묶음이_없는_등록_칸은_빠진_칸으로_세고_종료_2(tmp_path, view) -> None:
    w = build(tmp_path, skip=("uni_local_C3",))
    assert run(w) == SU.EXIT_REJECTED
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    assert body["status"] == "partial" and body["bundles"]["rejected"]["uni_local_C3_s1"] == ["missing_bundle"]
    assert body["report"]["models"]["uni_local_C3_s1"] == {"status": "rejected", "reasons": ["missing_bundle"]}
    assert body["report"]["rq1_recovery"]["main"]["status"] == "산출 불가"
    assert body["report"]["rule5"]["status"] == "computed"            # 조건 모델(중앙 · 연합)은 다 있다


def test_묶음_사이_환경이_다르면_비교는_산출_불가이고_종료_2(tmp_path, view, monkeypatch) -> None:
    from evaluation.reject_v14 import Rejection, RejectCode

    w = build(tmp_path)
    monkeypatch.setattr(SU, "check_env_consistency",
                        lambda group: [Rejection(RejectCode.ENV_MISMATCH, group[0].files.stem, {"item": "x", "n_distinct": 2})]
                        if group and group[0].mode == SU.MODE_MODEL else [])
    assert run(w) == SU.EXIT_REJECTED
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    rep = body["report"]
    assert body["status"] == "partial" and "env:model" in body["bundles"]["rejected"]
    for key in ("bootstrap", "rq1_recovery", "rq2_participants", "rule5", "main_vs_sensitivity", "full_addendum"):
        assert rep[key]["status"] == "산출 불가" and rep[key]["reason"] == "env_mismatch", key
    assert all(m["status"] == "scored" for m in rep["models"].values())   # 모델마다의 값은 싣는다


# ---------------------------------------------------------------- 7 정답의 상자 없는 행 · 문자 일치(외부 검토)

def test_상자_없는_정답_행은_상자로_세지_않고_세어_싣는다(tmp_path, monkeypatch) -> None:
    """판독기는 정수 열의 빈 칸을 NA 로 준다 — float(NA) 로 죽지 않고, 공통 채점기처럼 이미지 수준 코드에만 남긴다."""
    def fake(root, ids, *, receipt):
        ids = list(ids)
        rows = _gold_rows(ids) + [{"image_id": ids[5], "iso_code": "301", "bbox_x1_px": None, "bbox_y1_px": None,
                                   "bbox_x2_px": None, "bbox_y2_px": None}]
        df = pd.DataFrame(rows, columns=["image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"])
        return df.astype({c: "Int64" for c in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")})

    monkeypatch.setattr(SU, "EVAL_VIEW", fake)
    w = build(tmp_path)
    assert run(w) == SU.EXIT_OK
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    assert body["gold"]["n_annotation_rows_without_box"] == 1
    f = body["report"]["models"]["uni_fed_s1"]["failures"]
    assert f["defect_images"]["n"] == sum(1 for i in SAMPLE_IDS if GOLD[i]) + 1      # 상자 없는 행의 이미지는 결함 이미지다


def test_모델_묶음의_문자_일치가_어긋나면_그_칸만_거부하고_종료_2(tmp_path, view, capsys) -> None:
    w = build(tmp_path, literal_break=("uni_local_C2",))
    assert run(w) == SU.EXIT_REJECTED
    body = json.loads(out_path(w).read_text(encoding="utf-8"))
    assert "literal_mismatch" in body["bundles"]["rejected"]["uni_local_C2_s1"]
    assert body["report"]["models"]["uni_local_C2_s1"]["status"] == "rejected"
    assert body["report"]["rule5"]["status"] == "computed"                       # 조건 모델은 그대로 판정한다
