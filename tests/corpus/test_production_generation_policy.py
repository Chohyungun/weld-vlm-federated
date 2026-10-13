"""재생성 명부의 CPU 계약 시험. 합성 입력만 쓰며 모델·영상의 의미 품질을 시험하지 않는다."""

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from corpus.production import generation_policy as G
from corpus.production.common import canonical_bytes, iter_jsonl, read_json, sha256_file


def _digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _spec(family):
    return {"model_id": f"local/{family}", "revision": "a" * 40, "family": family,
            "local_files_only": True, "quantization": "bf16", "batch_size": 2,
            "max_new_tokens": 384, "max_input_tokens": 4096, "seed": 17,
            "wall_seconds": 1800, "enable_thinking": False}


def _task(index, *, defects=1, split="train", client="C1"):
    return {
        "id": f"synthetic-{index}", "kind": "image_report", "split": split, "client": client,
        "lineage_group_id": f"lineage-{index}", "image_path": f"not-opened/{index}.png",
        "image_sha256": _digest({"image": index}), "source_ids": [f"synthetic-annotation:{index}"],
        "locked_target": {"defects": [{"type": "2011", "bbox_px": [10, 20, 30, 40]}
                                       for _ in range(defects)],
                          "defect_counts": {"2011": defects} if defects else {}, "clauses": [], "verdict": None},
        "evidence": {"material": "강재", "defect_counts_by_name": {"기공": defects} if defects else {},
                     "defect_positions": [{"name": "기공", "where": "가운데-가운데"} for _ in range(defects)],
                     "has_applicable_clause": False, "size_mm_available": False,
                     "limitation": G.LIMITATION_FORMS[0]},
        "prompt": "이전 영상 설명 지시", "train_user_text": "이 영상의 주석을 설명하라.",
        "extra_metadata": {"preserve": [index, {"value": "변경 금지"}]},
    }


def _write_run(parent, run):
    (parent / "RUN.json").write_bytes(canonical_bytes(run) + b"\n")


def _reseal_tasks(parent, tasks):
    """계약 위반을 해시 오류에 가리지 않도록 합성 명부를 다시 봉인한다."""
    (parent / "tasks.jsonl").write_bytes(b"".join(canonical_bytes(t) + b"\n" for t in tasks))
    run = read_json(parent / "RUN.json")
    run["tasks_sha256"] = sha256_file(parent / "tasks.jsonl")
    run["identity_basis"]["tasks_sha256"] = run["tasks_sha256"]
    run["identity_sha256"] = _digest(run["identity_basis"])
    _write_run(parent, run)


@pytest.fixture
def case(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    parent = repo / "parent"
    parent.mkdir()
    tasks = [_task(0, defects=0), _task(1), _task(2, defects=2, split="val", client="C2")]
    config = {"schema_version": 1, "recipe": "parent_recipe", "shard_size": 2,
              "inputs": {"expected_image_train": 2, "expected_image_val": 1},
              "generation": _spec("qwen"), "judge": _spec("phi"),
              "quality": {"human_review_required": True, "no_partial_release": True}}
    raw = b"".join(canonical_bytes(t) + b"\n" for t in tasks)
    (parent / "tasks.jsonl").write_bytes(raw)
    (parent / "config.json").write_bytes(canonical_bytes(config) + b"\n")
    basis = {"schema_version": 1, "config_sha256": _digest(config),
             "tasks_sha256": sha256_file(parent / "tasks.jsonl"), "pairs_sha256": "a" * 64,
             "pairs_snapshot_digest": "b" * 64, "manifest_snapshot_digest": "c" * 64,
             "source_bank_sha256": "d" * 64, "prepare_code_sha256": "e" * 64}
    counts = Counter((t["split"], t["client"]) for t in tasks)
    run = {"schema_version": 1, "identity_basis": basis, "identity_sha256": _digest(basis),
           "config": config, "config_sha256": basis["config_sha256"], "tasks_sha256": basis["tasks_sha256"],
           "counts": {"tasks": 3, "image_report": {"train": 2, "val": 1},
                      "source_qa": {"train": 0, "val": 0},
                      "by_kind_split_client": {f"image_report|{s}|{c}": n for (s, c), n in counts.items()}}}
    _write_run(parent, run)
    for stage in ("generated", "judged"):
        (parent / stage).mkdir()
        # 샤드 검증을 우회하려는 자료가 아니다. 이 helper가 샤드를 열지 않는다는 경계의 표지다.
        (parent / stage / "000000.json").write_bytes(f"{stage}: 열거나 고치지 않는 표지".encode())
    return repo, parent, tasks, config


def _bytes_under(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_부모_전부와_설정_인자를_보존하고_새_신원만_쓴다(case):
    repo, parent, tasks, config = case
    before, config_before = _bytes_under(parent), deepcopy(config)
    old = read_json(parent / "RUN.json")
    destination = repo / "new"
    result = G.prepare_regeneration(parent, destination, config, repo_root=repo)
    assert _bytes_under(parent) == before
    assert config == config_before
    assert set(_bytes_under(destination)) == {"config.json", "tasks.jsonl", "RUN.json"}
    new = read_json(destination / "RUN.json")
    copied_config = read_json(result["config_path"])
    assert copied_config == new["config"]
    assert copied_config["recipe"] == G.POLICY_VERSION
    assert copied_config["generation_policy"] == G.policy_identity()
    assert copied_config["generation_policy"]["spec_sha256"] == G.SPEC_SHA256
    assert {k: v for k, v in copied_config.items() if k not in ("recipe", "generation_policy")} == {
        k: v for k, v in config.items() if k not in ("recipe", "generation_policy")}
    assert new["identity_sha256"] == _digest(new["identity_basis"]) != old["identity_sha256"]
    assert new["config_sha256"] == _digest(copied_config)
    assert new["tasks_sha256"] == sha256_file(destination / "tasks.jsonl") != old["tasks_sha256"]
    reference = new["identity_basis"]["regeneration_parent"]
    assert reference == {"run_sha256": sha256_file(parent / "RUN.json"),
                         "identity_sha256": old["identity_sha256"], "tasks_sha256": old["tasks_sha256"],
                         "config_sha256": old["config_sha256"]}
    assert new["counts"] == old["counts"] == result["counts"]
    got = list(iter_jsonl(destination / "tasks.jsonl"))
    assert {t["id"] for t in got} == {t["id"] for t in tasks}
    by_id = {t["id"]: t for t in tasks}
    for changed in got:
        original = by_id[changed["id"]]
        assert {k: v for k, v in original.items() if k not in ("prompt", "evidence")} == {
            k: v for k, v in changed.items() if k not in ("prompt", "evidence")}
        assert all(changed["evidence"][k] == value for k, value in original["evidence"].items())
        assert changed["prompt"] != original["prompt"]
    assert got[0]["evidence"]["row_class"] == "defect_free"
    assert got[1]["evidence"]["row_class"] == "defect"


def test_새_명부는_추론_읽기_계약을_통과하지만_샤드는_없다(case):
    from corpus.production.infer import StageRefused, read_stage_rows

    repo, parent, _, config = case
    result = G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert read_stage_rows(result["run_dir"], "generate", read_json(result["config_path"])) == []
    with pytest.raises(StageRefused, match="config_changed"):
        read_stage_rows(result["run_dir"], "generate", config)


def test_정상_N5는_생성_필수이며_예시는_근거에_있다():
    task = _task(0, defects=0)
    before = deepcopy(task)
    evidence = G.build_image_evidence(task)
    prompt = G.build_image_prompt(evidence)
    instruction, embedded = prompt.rsplit("\n주석: ", 1)
    assert task == before
    assert "반드시 두 문장" in instruction and "둘째 문장" in instruction
    assert "normal_absence_forms 중 하나를 골라 그대로 쓴다" in instruction
    assert "normal_limitation_forms 중 하나를 골라 그대로 쓴다" in instruction
    assert "품질이나 안전성을 보증할 수 없다" not in instruction
    assert evidence["normal_limitation_forms"] == [
        "결함이 표시되지 않았다는 것이 합격이나 안전을 뜻하지는 않는다.",
        "이것이 용접부의 합격이나 안전을 보장하지 않는다."]
    for form in evidence["normal_absence_forms"] + evidence["normal_limitation_forms"]:
        assert form in embedded and form not in instruction


def test_결함_한계도_근거의_문장을_그대로_쓰며_본문에는_복제하지_않는다():
    evidence = G.build_image_evidence(_task(1))
    instruction, embedded = G.build_image_prompt(evidence).rsplit("\n주석: ", 1)
    assert "limitation_forms 중 하나를 골라 그대로 쓴 한 문장" in instruction
    assert "합격 여부를 결정할 수 없다" not in instruction
    for form in evidence["limitation_forms"]:
        assert form in embedded and form not in instruction
    assert G.POLICY_VERSION == "image_report_regeneration_v3"
    assert G.policy_identity()["version"] == G.POLICY_VERSION


@pytest.mark.parametrize("absence", G.NORMAL_ABSENCE_FORMS)
@pytest.mark.parametrize("limitation", G.NORMAL_LIMITATION_FORMS)
def test_모든_정상_생성_예시_조합이_판정_문법과_N5_표지에_맞는다(absence, limitation):
    """고정 예시의 교차계약이다. 실제 모델의 지시 준수나 생성 품질은 측정하지 않는다."""
    from corpus.production import judge_policy as P

    task = _task(0, defects=0)
    task["evidence"] = G.build_image_evidence(task)
    text = f"{absence} {limitation}"
    assert P.deterministic_reasons(task, text) == []
    assert P.deterministic_flags(task, text) == []


@pytest.mark.parametrize("limitation", G.LIMITATION_FORMS)
def test_모든_결함_한계_예시가_판정_문법에_맞는다(limitation):
    """L 문장 계약만 본다. 소견의 의미 정확도나 B17 전체 형식의 통과 시험이 아니다."""
    from corpus.production import judge_policy as P

    task = _task(1)
    task["evidence"] = G.build_image_evidence(task)
    assert P.deterministic_reasons(task, limitation) == []


def test_희소_주석에서_정렬과_형상이나_부재를_만들도록_강요하지_않는다():
    prompt = G.build_image_prompt(G.build_image_evidence(_task(1)))
    assert "단일 결함이나 단일 위치에서 형상·정렬·몰림을 만들어내지 않는다" in prompt
    assert "알 수 없는 관찰상의 한계" in prompt
    assert "그런 속성이 실제로 없다고 꾸며내지 않는다" in prompt
    assert "4~7문장" in prompt and "소견은 2~4문장" in prompt


def test_제한_예시는_한_문장_L1_L2이고_길이_단위가_없다():
    for form in G.LIMITATION_FORMS:
        assert form.count(".") == 1 and form.endswith(".")
        assert "두께" in form and "실치수" in form
        assert "없어" in form or "주어지지 않아" in form
        assert "판정은 내리지 않는다" in form or "판단하지 않는다" in form
        assert not any(unit in form for unit in ("mm", "cm", "밀리", "센티"))


def test_근거_예시를_옮겨도_지시문_유출로_잘못_잡지_않는다():
    from corpus.production.quality import leak_problems

    for task in (_task(0, defects=0), _task(1)):
        task["evidence"] = G.build_image_evidence(task)
        task["prompt"] = G.build_image_prompt(task["evidence"])
        forms = list(G.LIMITATION_FORMS)
        if not task["locked_target"]["defects"]:
            forms += list(G.NORMAL_ABSENCE_FORMS) + list(G.NORMAL_LIMITATION_FORMS)
        for form in forms:
            assert leak_problems(form, task) == []


@pytest.mark.parametrize("marker", ["CTO_GATE_OPEN", "MERGE_HEAD"])
def test_편집_허가와_별개로_실행_게이트가_열리면_아무것도_쓰지_않는다(case, marker):
    repo, parent, _, config = case
    before = _bytes_under(parent)
    (repo / ".git" / marker).write_text("열린 게이트", encoding="utf-8")
    with pytest.raises(ValueError, match="gate/merge"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new").exists()
    assert _bytes_under(parent) == before


@pytest.mark.parametrize("which", ["file", "directory", "parent", "inside_parent"])
def test_기존_경로와_부모_내부에는_쓰지_않는다(case, which):
    repo, parent, _, config = case
    destination = repo / "new"
    if which == "file":
        destination.write_bytes(b"existing")
    elif which == "directory":
        destination.mkdir()
    elif which == "parent":
        destination = parent
    else:
        destination = parent / "new"
    before = _bytes_under(parent)
    with pytest.raises(G.RegenerationRefused, match="run_dir_exists|output_overlaps_parent"):
        G.prepare_regeneration(parent, destination, config, repo_root=repo)
    assert _bytes_under(parent) == before
    if which == "file":
        assert destination.read_bytes() == b"existing"
    if which == "inside_parent":
        assert not destination.exists()


@pytest.mark.parametrize("kind,code", [("identity", "parent_identity"), ("basis", "parent_identity"),
                                        ("config", "parent_config"), ("tasks", "parent_tasks"),
                                        ("counts", "parent_counts")])
def test_부모의_변조나_잘못된_회계를_거부한다(case, kind, code):
    repo, parent, _, config = case
    run = read_json(parent / "RUN.json")
    if kind == "identity":
        run["identity_sha256"] = "f" * 64
    elif kind == "basis":
        run["identity_basis"]["tasks_sha256"] = "f" * 64
        run["identity_sha256"] = _digest(run["identity_basis"])
    elif kind == "config":
        run["config"]["generation"]["seed"] += 1
    elif kind == "tasks":
        with (parent / "tasks.jsonl").open("ab") as stream:
            stream.write(b"\n")
    else:
        run["counts"]["source_qa"]["train"] = 1
    _write_run(parent, run)
    with pytest.raises(G.RegenerationRefused, match=code):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new").exists()


def test_부모와_다른_모델_설정을_인자로_받지_않는다(case):
    repo, parent, _, config = case
    changed = deepcopy(config)
    changed["generation"]["max_new_tokens"] += 1
    with pytest.raises(G.RegenerationRefused, match="config_changed"):
        G.prepare_regeneration(parent, repo / "new", changed, repo_root=repo)
    assert not (repo / "new").exists()


@pytest.mark.parametrize("kind,code", [("source_qa", "image_only"), ("duplicate", "task_id"),
                                        ("eval", "task_split"), ("lineage", "lineage_split_overlap"),
                                        ("evidence", "evidence_counts")])
def test_해시를_다시_맞춰도_잘못된_명부를_거부한다(case, kind, code):
    repo, parent, tasks, config = case
    if kind == "source_qa":
        tasks[0].update(kind="source_qa", locked_target=None)
    elif kind == "duplicate":
        tasks[1]["id"] = tasks[0]["id"]
    elif kind == "eval":
        tasks[0]["split"] = "eval"
    elif kind == "lineage":
        tasks[2]["lineage_group_id"] = tasks[0]["lineage_group_id"]
    else:
        tasks[1]["evidence"]["defect_counts_by_name"]["기공"] = 2
    _reseal_tasks(parent, tasks)
    with pytest.raises(G.RegenerationRefused, match=code):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new").exists()


def test_작성_중_부모_바이트가_바뀌면_RUN을_발행하지_않는다(case, monkeypatch):
    repo, parent, _, config = case
    original = G.build_image_prompt
    changed = False

    def mutate_parent(evidence):
        nonlocal changed
        if not changed:
            with (parent / "RUN.json").open("ab") as stream:
                stream.write(b" ")
            changed = True
        return original(evidence)

    monkeypatch.setattr(G, "build_image_prompt", mutate_parent)
    with pytest.raises(G.RegenerationRefused, match="parent_changed"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert (repo / "new" / "tasks.jsonl").exists()
    assert not (repo / "new" / "RUN.json").exists()


def test_쓰는_도중_게이트가_열리면_RUN을_발행하지_않고_실패_출력은_보존한다(case, monkeypatch):
    repo, parent, _, config = case
    original = G.build_image_prompt
    before = _bytes_under(parent)

    def open_gate(evidence):
        (repo / ".git" / "CTO_GATE_OPEN").write_bytes(b"open")
        return original(evidence)

    monkeypatch.setattr(G, "build_image_prompt", open_gate)
    with pytest.raises(ValueError, match="gate/merge"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new" / "RUN.json").exists()
    assert (repo / "new" / "config.json").exists()
    assert _bytes_under(parent) == before


def test_파일_쓰기_실패_뒤_부분_출력을_덮어쓰지_않는다(case, monkeypatch):
    repo, parent, _, config = case
    before = _bytes_under(parent)
    original = G.write_json_new

    def fail_run(path, value):
        if Path(path).name == "RUN.json":
            raise OSError("합성 발행 실패")
        return original(path, value)

    monkeypatch.setattr(G, "write_json_new", fail_run)
    with pytest.raises(OSError, match="합성 발행 실패"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    partial = _bytes_under(repo / "new")
    assert set(partial) == {"config.json", "tasks.jsonl"}
    assert _bytes_under(parent) == before
    with pytest.raises(G.RegenerationRefused, match="run_dir_exists"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert _bytes_under(repo / "new") == partial


def test_같은_부모와_정책은_다른_새_경로에서도_같은_명부와_신원이다(case):
    repo, parent, _, config = case
    first = G.prepare_regeneration(parent, repo / "new-a", config, repo_root=repo)
    second = G.prepare_regeneration(parent, repo / "new-b", config, repo_root=repo)
    assert first["identity_sha256"] == second["identity_sha256"]
    assert _bytes_under(repo / "new-a") == _bytes_under(repo / "new-b")


def test_부모_쓰기가_진행_중이면_새_명부를_만들지_않는다(case):
    repo, parent, _, config = case
    (parent / ".inference.lock").write_bytes(b"other writer")
    before = _bytes_under(parent)
    with pytest.raises(G.RegenerationRefused, match="parent_locked"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new").exists()
    assert _bytes_under(parent) == before


def test_수량이_같아도_근거의_이름이_잠긴_코드와_다르면_거부한다(case):
    repo, parent, tasks, config = case
    tasks[1]["evidence"]["defect_counts_by_name"] = {"융합불량": 1}
    tasks[1]["evidence"]["defect_positions"][0]["name"] = "융합불량"
    _reseal_tasks(parent, tasks)
    with pytest.raises(G.RegenerationRefused, match="evidence_names"):
        G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    assert not (repo / "new").exists()


@pytest.mark.parametrize("change,code", [("type", "locked_type"), ("bbox", "locked_bbox"),
                                         ("counts", "locked_counts"), ("clauses", "evidence_clauses")])
def test_잠긴_정답의_실제_생산_계약을_검사한다(change, code):
    task = _task(1)
    if change == "type":
        task["locked_target"]["defects"][0]["type"] = "unknown"
    elif change == "bbox":
        task["locked_target"]["defects"][0]["bbox_px"] = [40, 20, 10, 40]
    elif change == "counts":
        task["locked_target"]["defect_counts"] = {"2011": 2}
    else:
        task["locked_target"]["clauses"] = ["synthetic-clause"]
    with pytest.raises(G.RegenerationRefused, match=code):
        G.build_image_evidence(task)


def test_폭과_높이_없는_부모의_거친_위치_대조를_완료라고_기록하지_않는다(case):
    repo, parent, tasks, config = case
    # 같은 bbox도 영상 크기가 없으면 어느 아홉 칸인지 재계산할 수 없다.
    tasks[1]["evidence"]["defect_positions"][0]["where"] = "아래-오른쪽"
    _reseal_tasks(parent, tasks)
    G.prepare_regeneration(parent, repo / "new", config, repo_root=repo)
    new = read_json(repo / "new" / "RUN.json")
    scope = new["regeneration"]["validation_scope"]
    assert scope["coarse_positions"] == "preserved_parent_tasks_not_recomputed_without_image_dimensions"
    assert scope["image_bytes"] == "not_opened_requires_separate_deployment_gate"
    assert list(iter_jsonl(repo / "new" / "tasks.jsonl"))[1]["evidence"] == G.build_image_evidence(tasks[1])


def _set_synthetic_population(parent, tasks):
    """이 시험이 만든 임시 부모의 수량만 바꾼다. 봉인 fixture/실제 RUN은 사용하지 않는다."""
    run = read_json(parent / "RUN.json")
    counts = Counter((task["split"], task["client"]) for task in tasks)
    split_counts = {split: sum(n for (s, _), n in counts.items() if s == split) for split in ("train", "val")}
    config = deepcopy(run["config"])
    config["inputs"] = {f"expected_image_{split}": count for split, count in split_counts.items()}
    run.update(config=config, config_sha256=_digest(config), counts={
        "tasks": len(tasks), "image_report": split_counts, "source_qa": {"train": 0, "val": 0},
        "by_kind_split_client": {f"image_report|{s}|{c}": n for (s, c), n in counts.items()}})
    run["identity_basis"]["config_sha256"] = run["config_sha256"]
    _write_run(parent, run)
    (parent / "config.json").write_bytes(canonical_bytes(config) + b"\n")
    _reseal_tasks(parent, tasks)
    return config


def _stratum(task):
    return task["client"], task["split"], "defect" if task["locked_target"]["defects"] else "defect_free"


@pytest.mark.parametrize("population", ["all_twelve", "sparse", "single"])
def test_존재층_round_robin과_부모_층내순서_전행보존_및_신원결속(case, population):
    repo, parent, _, _ = case
    order = [(client, split, kind) for client in ("C1", "C2", "C3")
             for split in ("train", "val") for kind in ("defect_free", "defect")]
    if population == "all_twelve":
        selected = order
    elif population == "sparse":
        selected = [order[0], order[10], order[11]]
    else:
        selected = [order[7]]
    tasks = []
    for group_index, (client, split, kind) in enumerate(reversed(selected)):
        for _ in range(group_index % 3 + 1):
            # id 정렬로 층내 부모 순서를 대체하면 깨지도록 내림차순 식별자를 쓴다.
            tasks.append(_task(1000 - len(tasks), defects=int(kind == "defect"), split=split, client=client))
    config = _set_synthetic_population(parent, tasks)
    before = _bytes_under(parent)
    old = read_json(parent / "RUN.json")
    buckets = {key: [task for task in tasks if _stratum(task) == key] for key in order}
    expected = [task for ordinal in range(max(map(len, buckets.values())))
                for key in order for task in buckets[key][ordinal:ordinal + 1]]

    result = G.prepare_regeneration(parent, repo / "ordered", config, repo_root=repo)
    got = list(iter_jsonl(repo / "ordered" / "tasks.jsonl"))
    assert [task["id"] for task in got] == [task["id"] for task in expected]
    assert len(got) == len({task["id"] for task in got}) == len(tasks)
    assert {task["id"] for task in got} == {task["id"] for task in tasks}
    by_id = {task["id"]: task for task in tasks}
    for changed in got:
        original = by_id[changed["id"]]
        assert {key: value for key, value in changed.items() if key not in ("evidence", "prompt")} == {
            key: value for key, value in original.items() if key not in ("evidence", "prompt")}
        assert all(changed["evidence"][key] == value for key, value in original["evidence"].items())
    for key in order:
        assert [task["id"] for task in got if _stratum(task) == key] == [task["id"] for task in buckets[key]]

    new = read_json(repo / "ordered" / "RUN.json")
    assert new["counts"] == old["counts"] == result["counts"]
    ordering = new["identity_basis"]["regeneration_ordering"]
    assert ordering == new["regeneration"]["ordering"] == result["ordering"]
    assert ordering["policy"] == {
        "version": "stratified_round_robin_v1", "axes": ["client", "split", "row_class"],
        "client_order": ["C1", "C2", "C3"], "split_order": ["train", "val"],
        "row_class_order": ["defect_free", "defect"], "within_stratum": "parent_tasks_order",
        "empty_strata": "skip", "exhausted_strata": "skip"}
    assert new["config"]["generation_policy"]["task_ordering"] == ordering["policy"]
    assert new["identity_basis"]["generation_policy"]["task_ordering"] == ordering["policy"]
    assert new["identity_sha256"] == _digest(new["identity_basis"])
    assert ordering["strata_counts"] == {"|".join(key): len(buckets[key]) for key in order}
    active = [key for key in order if buckets[key]]
    coverage = ordering["prefix_coverage"]
    assert coverage["rows_to_cover_all_nonempty_strata"] == len(active)
    assert [_stratum(task) for task in got[:len(active)]] == active
    assert coverage["first_row_1based_by_stratum"] == {"|".join(key): i + 1 for i, key in enumerate(active)}
    assert coverage["scope"] == "planned_task_order_not_judged_or_accepted_coverage"
    assert _bytes_under(parent) == before
    again = G.prepare_regeneration(parent, repo / "ordered-again", config, repo_root=repo)
    assert again["identity_sha256"] == result["identity_sha256"]
    assert _bytes_under(repo / "ordered-again") == _bytes_under(repo / "ordered")
    assert _bytes_under(parent) == before


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "within_stratum_reversed"])
def test_순회에_누락_중복_층내역전이_있으면_RUN을_발행하지_않는다(case, monkeypatch, corruption):
    repo, parent, _, _ = case
    tasks = [_task(3), _task(1), _task(2, defects=0)]
    config = _set_synthetic_population(parent, tasks)
    before = _bytes_under(parent)
    original = G._round_robin

    def broken_order(strata):
        entries = list(original(strata))
        if corruption == "missing":
            entries.pop()
        elif corruption == "duplicate":
            entries.insert(1, entries[0])
        else:
            entries[1], entries[2] = entries[2], entries[1]
        yield from entries

    monkeypatch.setattr(G, "_round_robin", broken_order)
    with pytest.raises(G.RegenerationRefused, match="task_ordering"):
        G.prepare_regeneration(parent, repo / "unfinished", config, repo_root=repo)
    assert not (repo / "unfinished" / "RUN.json").exists()
    assert (repo / "unfinished" / "tasks.jsonl").exists()
    assert _bytes_under(parent) == before
