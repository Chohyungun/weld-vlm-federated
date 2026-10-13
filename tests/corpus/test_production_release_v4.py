"""릴리스가 생성 v4(생성 정책 `image_report_regeneration_v4` — 생성 v2)의 행을 거른다 — C 24e I-1 의 결합 조건.

관찰 구조 진단(`observation_problems`)이 비지 않은 결함 행 · 템플릿 바이트가 다른 정상 행 · 잘린 결함 행은 릴리스에 들지 않는다.
**거부하지 않으면 실패하는 시험**이다. 생성 v4 의 쓰기 · 읽기(`infer`) · 렌더(`generation_policy`)는 실물이고 모델만 가짜다.
이 파일은 생성 v2 코드(외부 검토 도구의 17 파일)가 든 트리에서만 수집된다 — 없으면 수집에서 실패한다(건너뛰지 않는다).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from corpus.production import generation_policy as G
from corpus.production import infer, release
from corpus.production.generation_policy import (  # noqa: F401 — 생성 v2 코드가 있어야 한다
    is_v2_run,
    render_generation_row,
)
from corpus.production.release import ReleaseRefused, _v4_generation_problem, check_row
from data.label_map import load_label_map
from tests.corpus.test_production_release import Case

LABEL = load_label_map().defect_types["porosity"]
#: 근거와 맞는 관찰(결함 하나 · 영상 가운데) — 생성 정책의 구조 진단이 비는 글
GOOD_OBSERVATION = (f"{LABEL.name_ko} 1개가 영상 가운데에 표시되어 있다. "
                    "주석만으로는 결함의 모양과 크기를 알 수 없다.")


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _spec(family):
    return {"model_id": f"fixture/{family}", "revision": "a" * 40, "family": family, "quantization": "bf16",
            "local_files_only": True, "enable_thinking": False, "batch_size": 2, "max_input_tokens": 4096,
            "max_new_tokens": 384, "wall_seconds": 1800, "seed": 17}


def _task(i, defect):
    return {"id": f"case-{i:03d}", "kind": "image_report", "split": "train", "client": "C1",
            "lineage_group_id": f"lineage-{i}", "source_ids": [f"annotation-{i}"],
            "image_path": f"never-opened/{i}.png", "image_sha256": _digest(["image", i]),
            "train_user_text": "이 영상의 주석을 설명하라.", "prompt": "부모의 이전 설명 지시",
            "locked_target": {"defects": [{"type": LABEL.iso_code, "bbox_px": [1, 2, 3, 4]}] if defect else [],
                              "defect_counts": {LABEL.iso_code: 1} if defect else {}, "clauses": [], "verdict": None},
            "evidence": {"material": "강재", "defect_counts_by_name": {LABEL.name_ko: 1} if defect else {},
                         "defect_positions": [{"name": LABEL.name_ko, "where": "가운데-가운데"}] if defect else [],
                         "has_applicable_clause": False, "size_mm_available": False,
                         "limitation": "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."}}


def v4_run(tmp_path, raw_text=GOOD_OBSERVATION, finish="eos", normal=2, defect=2):
    """부모 명부 → 생성 v4 재생성 명부(`prepare_regeneration`) → 가짜 모델로 생성(transformers 를 자칭) → 읽은 생성 행."""
    root = tmp_path / "repository"
    root.mkdir()
    (root / ".git").mkdir()
    parent = root / "parent"
    parent.mkdir()
    tasks = [_task(i, i >= normal) for i in range(normal + defect)]
    config = {"schema_version": 1, "recipe": "parent_fixture", "shard_size": 2, "generation": _spec("qwen"),
              "judge": _spec("phi"), "inputs": {"expected_image_train": len(tasks), "expected_image_val": 0}}
    raw = b"".join(_canonical(t) + b"\n" for t in tasks)
    (parent / "tasks.jsonl").write_bytes(raw)
    basis = {"schema_version": 1, "config_sha256": _digest(config), "tasks_sha256": hashlib.sha256(raw).hexdigest(),
             "pairs_sha256": "a" * 64, "pairs_snapshot_digest": "b" * 64, "manifest_snapshot_digest": "c" * 64,
             "source_bank_sha256": "d" * 64, "prepare_code_sha256": "e" * 64}
    run = {"schema_version": 1, "identity_basis": basis, "identity_sha256": _digest(basis), "config": config,
           "config_sha256": basis["config_sha256"], "tasks_sha256": basis["tasks_sha256"],
           "counts": {"tasks": len(tasks), "image_report": {"train": len(tasks), "val": 0},
                      "source_qa": {"train": 0, "val": 0}, "by_kind_split_client": {"image_report|train|C1": len(tasks)}}}
    for name, value in (("RUN.json", run), ("config.json", config)):
        (parent / name).write_bytes(_canonical(value) + b"\n")
    dest = root / "generation-v4"
    G.prepare_regeneration(parent, dest, config, repo_root=root)
    cfg = json.loads((dest / "config.json").read_text(encoding="utf-8"))

    class Backend:
        def __init__(self, spec, stage):
            self.identity = {"backend": "transformers", "model_id": spec["model_id"], "revision": spec["revision"],
                             "family": spec["family"], "model_type": spec["family"], "quantization": spec["quantization"],
                             "tokenizer_sha256": "c" * 64}
            self.last_metrics = {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2}

        def generate_batch(self, prompts, system, deadline):
            return [{"text": raw_text, "finish_reason": finish, "input_tokens": 71, "output_tokens": 23} for _ in prompts]

        def close(self):
            pass

    assert infer.run_stage(dest, "generate", cfg, backend_factory=Backend)["exit_code"] == 0
    run4 = json.loads((dest / "RUN.json").read_text(encoding="utf-8"))
    tasks4 = [json.loads(x) for x in (dest / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    return run4, tasks4, infer.read_stage_rows(dest, "generate", cfg)


def _problems(run, tasks, rows):
    assert is_v2_run(run) and [r["id"] for r in rows] == [t["id"] for t in tasks]
    return {t["id"]: _v4_generation_problem(G, t, g) for t, g in zip(tasks, rows)}


def test_v4_바른_정상_템플릿과_결함_관찰은_지난다(tmp_path):
    run, tasks, rows = v4_run(tmp_path)
    assert {r["generator"] for r in rows} == {"template", "model"} and {r["_backend"] for r in rows} == {"template", "transformers"}
    assert set(_problems(run, tasks, rows).values()) == {None}


@pytest.mark.parametrize("change", [
    lambda g: g.update(text=g["text"] + " "),                                    # 바이트 하나
    lambda g: g.update(text=G.NORMAL_ABSENCE_FORMS[-1] + " " + G.NORMAL_LIMITATION_FORMS[-1] + "."),
    lambda g: g.update(raw_text="모델이 쓴 글"),
    lambda g: g.update(observation_problems=["obs_language"]),
])
def test_v4_템플릿_바이트가_다른_정상_행은_거부(tmp_path, change):
    _run, tasks, rows = v4_run(tmp_path)
    t, g = next((t, dict(g)) for t, g in zip(tasks, rows) if g["generator"] == "template")
    change(g)
    assert _v4_generation_problem(G, t, g)[0] == "generation_template_mismatch"


@pytest.mark.parametrize("raw_text", [
    "가운데에 결함 하나가 있다. 주석만으로는 결함의 모양과 크기를 알 수 없다.",                 # 이름 · 개수 빠짐
    f"{LABEL.name_ko} 1개가 영상 가운데에 표시되어 있다. 모재 두께를 알 수 없다.",             # 제한 말을 관찰에
    f"{LABEL.name_ko} 1个가 영상 가운데에 표시되어 있다. 주석만으로는 결함의 모양과 크기를 알 수 없다.",  # 한자
])
def test_v4_관찰_구조_진단이_비지_않은_결함_행은_거부(tmp_path, raw_text):
    _run, tasks, rows = v4_run(tmp_path, raw_text=raw_text)
    model = [(t, g) for t, g in zip(tasks, rows) if g["generator"] == "model"]
    assert model and all(g["observation_problems"] for _, g in model)                  # 읽기가 진단을 싣는다
    assert {_v4_generation_problem(G, t, g)[0] for t, g in model} == {"generation_observation_problems"}
    t, g = model[0]                                                                     # 행에 적힌 진단을 지워도 다시 계산한다
    assert _v4_generation_problem(G, t, {**g, "observation_problems": []})[0] == "generation_v4_row"


def test_v4_잘린_결함_행은_거부(tmp_path):
    _run, tasks, rows = v4_run(tmp_path, finish="length")
    model = [(t, g) for t, g in zip(tasks, rows) if g["generator"] == "model"]
    assert {_v4_generation_problem(G, t, g)[0] for t, g in model} == {"generation_truncated"}


def test_v4_정상_작업에_모델_행_결함_작업에_템플릿_행은_거부(tmp_path):
    _run, tasks, rows = v4_run(tmp_path)
    normal = next(t for t, g in zip(tasks, rows) if g["generator"] == "template")
    defect, model_row = next((t, g) for t, g in zip(tasks, rows) if g["generator"] == "model")
    template_row = next(g for g in rows if g["generator"] == "template")
    assert _v4_generation_problem(G, normal, {**model_row, "id": normal["id"]})[0] == "generation_v4_row"
    assert _v4_generation_problem(G, defect, {**template_row, "id": defect["id"]})[0] == "generation_v4_row"


@pytest.mark.parametrize("code", ["generation_observation_problems", "generation_template_mismatch", "generation_truncated"])
def test_v4_릴리스는_행마다_그_검사를_불러_거부한다(tmp_path, monkeypatch, code):
    """연결 시험 — 릴리스가 생성 v4 실행에서 행마다 위 검사를 부르고 그 사유로 멈춘다(부르지 않으면 이 시험이 실패한다)."""
    c = Case(tmp_path)
    monkeypatch.setattr(release, "_generation_v4", lambda run: G)
    calls = []

    def problem(gen, task, g):
        calls.append(task["id"])
        return (code, "시험") if len(calls) == 2 else None

    monkeypatch.setattr(release, "_v4_generation_problem", problem)
    c.refused(code)
    assert len(calls) == 2


def _template_row(case):
    rows = [json.loads(x) for x in (case.audit_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(r for r in rows if r["kind"] == "image_report")
    row["provenance"]["generate"] = {"backend": "template", "generator": "template", "shard_sha256": "a" * 64,
                                     "text_sha256": "b" * 64}
    row["provenance"]["judge"] = {"applicable": False, "identity_sha256": "c" * 64}
    audit = json.loads(json.dumps(case.audit))
    audit["judge_identity"]["parser_schema"] = "judge_structured_v7"
    return row, audit


def test_행_검사는_템플릿_생성_출처를_네_칸으로만_받는다(tmp_path):
    case = Case(tmp_path)
    row, audit = _template_row(case)
    check_row(row, audit)
    for bad in ({"family": "qwen"}, {"generator": "model"}, {"model_id": "x"}):
        broken = json.loads(json.dumps(row))
        broken["provenance"]["generate"].update(bad)
        with pytest.raises(ReleaseRefused) as err:
            check_row(broken, audit)
        assert err.value.code == "row_backend"
