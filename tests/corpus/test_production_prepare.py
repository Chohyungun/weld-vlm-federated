"""작업 명부(prepare_run) 계약 — 가짜 입력만 쓴다(실데이터 · GPU 없음)."""
from __future__ import annotations

import json

import pytest

# 은행 진입점이 없으면 건너뛰지 않고 수집에서 실패한다 — 진입점 부재는 계약 위반이다(B 05f I-1)
from corpus.parse.passage_bank import select_bank_for_generation  # noqa: F401
from corpus.production import infer
from corpus.production.common import canonical_bytes, sha256_file
from corpus.production.prepare import SOURCE_PERSPECTIVES, PrepareRefused
from tests.corpus.production_fixtures import IMAGES, PASSAGES, QUARANTINED, USABLE, World, sha


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _tasks(run_dir):
    return [json.loads(x) for x in (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]


def _refused(world, code, run_name="run"):
    with pytest.raises(PrepareRefused) as err:
        world.prepare(run_name)
    assert err.value.code == code
    assert not (world.tmp / run_name).exists()
    assert not list(world.tmp.glob(f".{run_name}.*.tmp"))       # 거부하면 아무것도 남기지 않는다
    return err.value


def test_명부_수와_RUN_계약(world):
    result = world.prepare()
    run_dir = world.tmp / "run"
    tasks = _tasks(run_dir)
    run = json.loads((run_dir / "RUN.json").read_bytes())
    images = [t for t in tasks if t["kind"] == "image_report"]
    sources = [t for t in tasks if t["kind"] == "source_qa"]
    assert sum(t["split"] == "train" for t in images) == 4 and sum(t["split"] == "val" for t in images) == 2
    assert len(sources) == len(USABLE) * len(SOURCE_PERSPECTIVES) == run["counts"]["source_qa"]["train"] + run["counts"]["source_qa"]["val"]
    sb = run["counts"]["source_bank"]                       # 은행 진입점의 요약 그대로
    assert (sb["bank_rows"], sb["selected"], sb["excluded"]) == (len(PASSAGES), len(USABLE), len(PASSAGES) - len(USABLE))
    assert sb["reasons"]["review:none"] == 1 and sb["reasons"]["hold:source_policy:paid_standard_numeric"] == 1
    assert run["identity_basis"]["source_bank_inputs_sha256"] == sb["inputs_sha256"]
    assert run["identity_basis"]["source_bank_sha256"] == world.bank_sha and run["identity_basis"]["source_text_sha256"] == sha(world.bank_text.read_bytes())
    assert run["schema_version"] == 1 and run["config"] == world.config
    assert run["config_sha256"] == sha(canonical_bytes(world.config))
    assert run["tasks_sha256"] == sha256_file(run_dir / "tasks.jsonl")
    assert run["identity_sha256"] == sha(canonical_bytes(run["identity_basis"])) == result["identity_sha256"]
    basis = run["identity_basis"]
    assert basis["pairs_sha256"] == world.config["inputs"]["pairs_sha256"]
    assert basis["manifest_snapshot_digest"] == world.manifest_digest
    assert run["counts"]["image_report"] == {"train": 4, "val": 2}
    # 소비자가 그대로 받는다
    got_run, got_tasks = infer._read_inputs(run_dir, world.config)
    assert got_run == run and got_tasks == tasks


def test_계보와_영상_해시는_매니페스트와_실제_파일에서(world):
    world.prepare()
    by_id = {r["image_id"]: r for r in world.manifest_rows}
    for t in _tasks(world.tmp / "run"):
        if t["kind"] != "image_report":
            continue
        iid = t["id"].split(":", 1)[1]
        m = by_id[iid]
        assert (t["split"], t["client"], t["lineage_group_id"], t["image_path"]) == (m["split"], m["client"], m["group_id"], m["rel_path"])
        assert t["image_sha256"] == sha((world.repo / m["rel_path"]).read_bytes())
    ids = [t["id"] for t in _tasks(world.tmp / "run")]
    assert "image_report:img:0099" not in ids and f"image_report:{QUARANTINED}" not in ids


def test_골격은_잠긴_필드에만_있고_prompt_에는_수치가_없다(world):
    world.prepare()
    for t in _tasks(world.tmp / "run"):
        if t["kind"] != "image_report":
            continue
        n = next(n for iid, *_r, n in IMAGES if t["id"].endswith(iid))
        assert len(t["locked_target"]["defects"]) == n
        assert {"defects", "defect_counts", "verdict", "clauses"} == set(t["locked_target"])
        assert "KRA27" not in t["prompt"] and "344" not in t["prompt"] and "mm 이하" not in t["prompt"]
        assert "KRA27" not in json.dumps(t["evidence"], ensure_ascii=False)


def test_영상_작업의_학습_질문은_하나이고_골격이_없다(world):
    import re

    from data.label_map import load_label_map

    world.prepare()
    texts = {t["train_user_text"] for t in _tasks(world.tmp / "run") if t["kind"] == "image_report"}
    assert len(texts) == 1
    (text,) = texts
    names = {dt.name_ko for dt in load_label_map().defect_types.values()}
    assert text.strip() and not re.search(r"[0-9{}\[\]]", text) and not any(n in text for n in names)
    assert all("train_user_text" not in t for t in _tasks(world.tmp / "run") if t["kind"] == "source_qa")


def test_원문_QA_는_넘길_수_있는_구절만_세_관점_계보는_은행의_것(world):
    world.prepare()
    sources = [t for t in _tasks(world.tmp / "run") if t["kind"] == "source_qa"]
    assert sorted(t["id"] for t in sources) == sorted(f"source_qa:{p}:q{j}" for p in USABLE for j in (1, 2, 3))
    spec = {p["passage_id"]: p for p in PASSAGES}
    for t in sources:
        p = spec[t["source_ids"][0]]
        assert t["client"] == "public" and t["image_path"] is None and t["locked_target"] is None
        assert t["lineage_group_id"] == p["lineage_group_id"] == t["evidence"]["lineage_group_id"]
        assert t["evidence"]["passage"] == p["text"] and t["evidence"]["source_id"] == p["doc_id"]
        assert ("역사적 서술" in t["prompt"]) == (p["passage_id"] == "HIST-P01")
        assert "pdf_page_start" not in json.dumps(t["evidence"]) and t["source_ref"]["locator"]["pdf_page_start"] == 3
    p02 = next(t for t in sources if t["id"] == "source_qa:DOC1-P02:q1")
    assert p02["evidence"]["context_passages"] == {"DOC1-P01": spec["DOC1-P01"]["text"]}
    assert "Do not turn experimental settings into general rules" in p02["prompt"] and spec["DOC1-P01"]["text"] in p02["prompt"]


def _sp(lin, doc, n):
    return [{"lineage_group_id": lin, "doc_id": doc} for _ in range(n)]


def test_원문_QA_분할은_계보_단위이고_문서가_val_의_절반을_넘지_않는다():
    from corpus.production.prepare import _source_split

    # v3 꼴: 큰 한 문서 계보(57 %)는 목표 10 % 안에 못 들어 train 에 남는다
    ps = _sp("BIG", "D-big", 570) + _sp("MID", "D-mid", 230) + [x for k in range(10) for x in _sp(f"S{k}", f"D{k}", 20)]
    split, info = _source_split(ps)
    assert split["BIG"] == "train" and info["target_val_passages"] == 100
    assert info["val_passages"] <= 100 and (info["max_doc_share_of_val"] or 0) <= 0.5
    assert set(info["val_lineages"]) == {g for g, s in split.items() if s == "val"}
    assert split == _source_split(list(reversed(ps)))[0]           # 입력 순서와 상관없다


def test_원문_QA_분할의_실제_val_문서_상한():
    from corpus.production.prepare import _source_split

    # 목표 안에 든 계보가 하나뿐이면 그 문서가 val 전부가 된다 — 끝의 상한이 그 계보를 뺀다
    split, info = _source_split(_sp("ONE", "D1", 4) + _sp("HUGE", "D2", 96))
    assert info["target_val_passages"] == 10 and info["val_passages"] == 0 and set(split.values()) == {"train"}


def test_같은_입력이면_같은_바이트(world):
    a, b = world.prepare("a"), world.prepare("b")
    assert a["identity_sha256"] == b["identity_sha256"]
    assert (world.tmp / "a" / "tasks.jsonl").read_bytes() == (world.tmp / "b" / "tasks.jsonl").read_bytes()


@pytest.mark.parametrize("key,code", [("pairs_sha256", "pairs_sha256"), ("pairs_snapshot_digest", "pairs_snapshot_digest"),
                                      ("source_bank_sha256", "source_bank_sha256")])
def test_지문_불일치는_거부(world, key, code):
    world.config["inputs"][key] = "0" * 64
    _refused(world, code)


def test_구절_은행_지문이_없으면_거부(world):
    world.config["inputs"]["source_bank_sha256"] = None
    _refused(world, "source_bank_unregistered")


def test_봉인_뒤에_바뀐_페어는_거부(world):
    with (world.pairs_dir / "pairs.jsonl").open("ab") as fh:
        fh.write(b"\n")
    _refused(world, "pairs_member_changed")


def test_매니페스트가_봉인과_다르면_거부(world):
    with (world.manifest_dir / "manifest.csv").open("a", encoding="utf-8", newline="") as fh:
        fh.write("img:0100,x.jpg," + "0" * 64 + ",grp_x,train,C1\n")
    _refused(world, "manifest_snapshot")


def test_eval_영상_혼입은_거부(world):
    world.pair_rows.append(World.pair("img:0099", "train", "C1", 1))
    world.seal_pairs()
    _refused(world, "eval_leak")


def test_eval_split_행은_거부(world):
    row = World.pair("img:0099", "eval", "", 1)
    world.pair_rows.append(row)
    world.seal_pairs()
    _refused(world, "eval_leak")


def test_eval_묶음과_같은_묶음은_거부(world):
    world.manifest_rows[0]["group_id"] = "grp_z"            # train 영상이 eval 묶음에 들어 있다
    world.write_manifest()
    world.seal_pairs()                                       # 페어 META 가 새 매니페스트 digest 를 가리킨다
    _refused(world, "eval_group_leak")


@pytest.mark.parametrize("field,value", [("split", "val"), ("client", "C2")])
def test_재배정은_거부(world, field, value):
    world.pair_rows[0][field] = value                        # img:0001 은 매니페스트에서 train/C1
    world.seal_pairs()
    world.config["inputs"]["expected_image_train"] = 4 - (field == "split")
    world.config["inputs"]["expected_image_val"] = 2 + (field == "split")
    _refused(world, "split_reassigned")


def test_기대_수와_다르면_거부(world):
    world.config["inputs"]["expected_image_train"] = 5
    _refused(world, "image_counts")


def test_영상_파일이_매니페스트와_다르면_거부(world):
    (world.repo / world.manifest_rows[2]["rel_path"]).write_bytes(b"changed")
    _refused(world, "image_changed")


def test_같은_영상이_두_번_나오면_거부(world):
    world.pair_rows.append(dict(world.pair_rows[0]))
    world.seal_pairs()
    _refused(world, "pairs_duplicate")


def test_사상표에_없는_코드는_거부(world):
    world.pair_rows[0]["skeleton"]["defects"][0]["type"] = "9999"
    world.seal_pairs()
    _refused(world, "label_code")


def test_있는_디렉터리에는_쓰지_않는다(world):
    (world.tmp / "run").mkdir()
    with pytest.raises(PrepareRefused) as err:
        world.prepare()
    assert err.value.code == "run_dir_exists"


def test_게이트가_열려_있으면_쓰지_않는다(world):
    (world.repo / ".git" / "CTO_GATE_OPEN").write_text("x")
    with pytest.raises(ValueError, match="gate"):
        world.prepare()
    assert not (world.tmp / "run").exists()


def test_은행_계약_밖의_꼴은_은행_진입점이_멈춘다(world):
    draft = [{"passage_id": "X-1", "source_id": "S", "source_sha256": "a" * 64, "passage": "text", "use_boundary": "u"}]
    world.bank_text.write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in draft))
    world.bank.write_bytes(world.bank_text.read_bytes())
    world.config["inputs"]["source_bank_sha256"] = sha(world.bank.read_bytes())
    (world.tmp / "bank_v3.build_report.json").write_bytes(canonical_bytes(
        {"outputs": {"pointer_bank_sha256": sha(world.bank.read_bytes()), "full_bank_sha256": sha(world.bank_text.read_bytes())}}))
    _refused(world, "source_bank:bank_contract")


@pytest.mark.parametrize("which,code", [("pointer", "pointer_bank_sha_mismatch"), ("full", "full_bank_sha_mismatch"),
                                        ("missing", "build_report_missing")])
def test_빌드_보고와_지문이_다르면_은행_진입점이_멈춘다(world, which, code):
    report = world.tmp / "bank_v3.build_report.json"
    if which == "missing":
        report.unlink()
    else:
        outputs = json.loads(report.read_bytes())["outputs"]
        outputs[f"{which}_bank_sha256"] = "0" * 64
        report.write_bytes(canonical_bytes({"outputs": outputs}))
    _refused(world, f"source_bank:{code}")


@pytest.mark.parametrize("over,reason", [({"source_text_path": None}, "no_full_bank"),
                                         ({"source_rights_path": None}, "rights:no_decision")])
def test_본문_은행이나_권리_결정이_없으면_넘기는_구절이_0_이고_사유를_적는다(world, over, reason):
    world.prepare(**over)
    run = json.loads((world.tmp / "run" / "RUN.json").read_bytes())
    assert run["counts"]["source_qa"] == {"train": 0, "val": 0}
    assert run["counts"]["source_bank"]["selected"] == 0 and run["counts"]["source_bank"]["reasons"][reason] == len(PASSAGES)


def test_필수_문맥은_사슬_전체를_싣고_순환을_닫는다(world):
    chain = [{"passage_id": "C-1", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "First link text."},
             {"passage_id": "C-2", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "Second link text.",
              "required_context_passage_ids": ["C-1"]},
             {"passage_id": "C-3", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "Third link text.",
              "required_context_passage_ids": ["C-2"]},
             {"passage_id": "M-1", "doc_id": "DOC-M", "lineage_group_id": "LG-M", "text": "Mutual one.", "required_context_passage_ids": ["M-2"]},
             {"passage_id": "M-2", "doc_id": "DOC-M", "lineage_group_id": "LG-M", "text": "Mutual two.", "required_context_passage_ids": ["M-1"]}]
    rows = [World.passage(p) for p in chain]
    world.write_bank(rows)
    world.write_records(rows, no_review=frozenset())
    world.prepare()
    ev = {t["source_ids"][0]: t["evidence"] for t in _tasks(world.tmp / "run") if t["kind"] == "source_qa"}
    assert sorted(ev["C-3"]["context_passages"]) == ["C-1", "C-2"]              # 직접 문맥 C-2 와 그 문맥 C-1
    assert sorted(ev["M-1"]["context_passages"]) == ["M-2"] and sorted(ev["M-2"]["context_passages"]) == ["M-1"]
    p3 = next(t["prompt"] for t in _tasks(world.tmp / "run") if t["id"] == "source_qa:C-3:q1")
    assert "First link text." in p3 and "Second link text." in p3


def test_사슬의_끝이_빠지면_사슬_전체가_빠진다(world):
    chain = [{"passage_id": "C-1", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "First link text."},
             {"passage_id": "C-2", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "Second link text.",
              "required_context_passage_ids": ["C-1"]},
             {"passage_id": "C-3", "doc_id": "DOC-C", "lineage_group_id": "LG-C", "text": "Third link text.",
              "required_context_passage_ids": ["C-2"]}]
    rows = [World.passage(p) for p in chain]
    world.write_bank(rows)
    world.write_records(rows, no_review=frozenset({"C-1"}))
    world.prepare()
    run = json.loads((world.tmp / "run" / "RUN.json").read_bytes())
    assert run["counts"]["source_bank"]["selected"] == 0 and run["counts"]["source_qa"] == {"train": 0, "val": 0}


def _same_bytes(world, src, dst):
    a = next(r for r in world.manifest_rows if r["image_id"] == src)
    b = next(r for r in world.manifest_rows if r["image_id"] == dst)
    (world.repo / b["rel_path"]).write_bytes((world.repo / a["rel_path"]).read_bytes())
    b["sha256"] = a["sha256"]
    world.write_manifest()
    world.seal_pairs()


def test_같은_계보_안의_같은_바이트_영상은_허용하고_센다(world):
    _same_bytes(world, "img:0001", "img:0002")                    # 둘 다 grp_a
    world.prepare()
    run = json.loads((world.tmp / "run" / "RUN.json").read_bytes())
    assert run["counts"]["duplicate_image_same_lineage"] == 1
    assert run["identity_basis"]["duplicate_image_policy"] == "same_lineage_allowed"


def test_다른_계보의_같은_바이트_영상은_거부(world):
    _same_bytes(world, "img:0001", "img:0003")                    # grp_a · grp_b
    _refused(world, "image_duplicate_cross_lineage")


def test_문맥_폐쇄에서_없는_구절은_거부한다():
    from corpus.production.prepare import _context_closure

    with pytest.raises(PrepareRefused) as err:
        _context_closure("A", {"A": {"required_context_passage_ids": ["B"]}, "B": {"required_context_passage_ids": ["Z"]}})
    assert err.value.code == "source_context_missing"
