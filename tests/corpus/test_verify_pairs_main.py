"""본실험 페어 산출물 검사기(`verify_pairs_main`) — 깨뜨린 산출물이 실제로 걸리는가.

검사기가 "실패 0" 을 내는 것만으로는 아무것도 말하지 않는다. 작은 합성 스냅샷으로 빌드한 뒤 산출 파일을
한 군데씩 깨뜨려, **그 결함에 해당하는 사유**가 나오는지 본다. 봉인 자산·실물 매니페스트는 읽지 않는다.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from corpus.generate import make_pairs_main as MM
from corpus.generate import make_pairs_pilot as M
from corpus.generate.run_cycle_corpus import defect_names
from corpus.rules import limits_loader
from corpus.rules.skeleton_gen import load_defect_lexicon
from corpus.validate import verify_pairs_main as V
from data.manifest_io import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, load_snapshot, write_snapshot

W, H = 1280, 720
NAMES = defect_names()
CODES = sorted(NAMES)                       # 코드 문자열을 박지 않는다 — 사상표에서 읽는다
TABLE = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
COVERED = [c for c in CODES if M.clause_basis(TABLE, "ST").get(c)]
NP = next(c for c in COVERED if "none_permitted" in M.clause_basis(TABLE, "ST")[c]["rule_kinds"])
DIM = next(c for c in COVERED if c != NP)

CAPS = {"snapshot_id": "test_pairs", "created_at": "2026-09-25T00:00:00Z",
        "capabilities": {"verdict_mode": "clause_only", "localization": True, "is_mock": True},
        "split_meta": {"dirichlet": None}}


def _row(iid, *, material="ST", has_defect=True, n_defects=1, split="train", client="C1", group=None):
    """매니페스트 **전 컬럼**을 채운다. 잘라낸 표로는 정식 적재기·스냅샷 대조를 지날 수 없다."""
    r = dict.fromkeys(MANIFEST_COLUMNS, "")
    r.update(image_id=iid, source="synthetic", rel_path=f"tiles\\{iid}.png",
             sha256=hashlib.sha256(iid.encode()).hexdigest(), width_px=W, height_px=H, modality="RT",
             material=material, has_defect=has_defect, n_defects=n_defects, label_type="polygon",
             has_localization=True, phash_hex="0" * 16, group_id=group or "g-" + iid, group_size=1,
             strata_key=material + "|RT", split=split, client=client, ingest_version="test",
             label_map_version=1)
    return r


def _ann(iid, k, code, box=(10, 10, 50, 50), ok=True):
    r = dict.fromkeys(ANNOTATION_COLUMNS, "")
    r.update(ann_id=f"{iid}#{k}", image_id=iid, src_label_raw="x", defect_type="d", iso_code=code,
             polygon_json="[]", bbox_x1_px=box[0], bbox_y1_px=box[1], bbox_x2_px=box[2], bbox_y2_px=box[3],
             area_px=64.0, major_axis_px=40.0, minor_axis_px=8.0, equiv_diameter_px=30.5,
             geom_valid=ok, geom_flags="")
    return r


ROWS = [_row("img-0001", n_defects=3), _row("img-0002", has_defect=False, n_defects=0, client="C2"),
        _row("img-0003", material="AL", client="C3", split="val"), _row("img-0004", n_defects=1),
        _row("img-0005", n_defects=1, client="C2"),
        _row("img-9001", split="eval", client="")]
ANNS = [_ann("img-0001", 0, DIM), _ann("img-0001", 1, NP, (0, 0, W, H)), _ann("img-0001", 2, DIM, ok=False),
        _ann("img-0003", 0, DIM), _ann("img-0004", 0, DIM, ok=False), _ann("img-0005", 0, NP),
        _ann("img-9001", 0, DIM)]


def _build(tmp_path, rows=None, anns=None):
    """정식 스냅샷을 쓰고 정식 적재기를 지나 빌드한다 — 본실험 경로와 같은 계약이다."""
    rows, anns = ROWS if rows is None else rows, ANNS if anns is None else anns
    snap, out = tmp_path / "snap", tmp_path / "build"
    digest = write_snapshot(snap, pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS)),
                            pd.DataFrame(anns, columns=list(ANNOTATION_COLUMNS)), CAPS)
    snapshot = load_snapshot(snap)
    # 경로 주장이 **참이 되게** 실물을 만든다. 없는 파일을 확인했다고 적으면 검사기가 잡는다(그것이 맞다).
    for rel in snapshot.manifest["rel_path"]:
        f = tmp_path / str(rel).replace(chr(92), "/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    res = M.build_pairs(snapshot.manifest, snapshot.annotations, TABLE, NAMES, load_defect_lexicon(),
                        root=tmp_path, check_paths=True)
    meta = MM.build_meta(res, snapshot.manifest, manifest_digest=digest, date="2026-09-25")
    M.write_outputs(out, res, asset="d4_pairs_test", version="v0", date="2026-09-25",
                    limits_csv=M.LIMITS_CSV, manifest_csv=snap / "manifest.csv", meta=meta)
    return out, snap, tmp_path


@pytest.fixture()
def built(tmp_path):
    return _build(tmp_path)


def _edit(out, fn, *, reseal=True):
    """pairs.jsonl 의 레코드 목록을 `fn` 으로 바꾸고, 계약서를 다시 쓴다(내용 결함만 남기려고)."""
    p = out / "pairs.jsonl"
    recs = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    changed = fn(recs)
    recs = changed if isinstance(changed, list) else recs      # 제자리 변경은 None 이나 뺀 값을 돌려준다
    M.write_jsonl(p, recs)
    if reseal:
        M.snapshot_digest(out, V.MEMBERS)


def _kinds(out, snap, root):
    rep = V.verify(out, snap, root)
    assert rep["ok"] == (not rep["failures"])
    return set(rep["failures"])


def test_손대지_않은_빌드는_실패가_없고_전_축을_대조한다(built):
    out, snap, root = built
    rep = V.verify(out, snap, root)
    assert rep["ok"] and rep["failures"] == {}
    # "적발이 없다" 와 "전부 봤다" 는 다른 사실이다 — 섞으면 못 본 것이 통과로 센다
    assert rep["coverage"] == "full" and rep["not_checked"] == [] and rep["verdict"] == "일치"
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 0
    assert (rep["n_records"], rep["n_defect"], rep["n_normal"], rep["n_discarded"]) == (4, 3, 1, 1)
    assert rep["eval_overlap"] == {"image_id": 0, "group_id": 0} and rep["n_eval_images"] == 1
    assert rep["n_annotations_skipped_in_records"] == 1


def _by_id(recs, iid):
    return next(r for r in recs if r["image_id"] == iid)


CORRUPTIONS = {
    # 검사기는 원천에서 레코드를 다시 조립해 **직렬화 바이트**로 맞댄다 — 필드마다 사유를 나누지 않는다.
    "상자 한 칸": (lambda rs: _by_id(rs, "img-0001")["skeleton"]["defects"][0]["bbox_px"].__setitem__(0, 11),
               {"E_record_differs_from_source"}),
    "결함 순서": (lambda rs: _by_id(rs, "img-0001")["skeleton"]["defects"].reverse(),
              {"E_record_differs_from_source", "A_gate_recheck"}),
    "AL 에 강재 조항": (lambda rs: _by_id(rs, "img-0003")["skeleton"].__setitem__(
        "clauses", [M.clause_basis(TABLE, "ST")[DIM]["clause_id"]]), {"A_gate_recheck"}),
    "부등호 한 글자": (lambda rs: _by_id(rs, "img-0001").__setitem__(
        "target_text", _by_id(rs, "img-0001")["target_text"].replace("이하", "미만", 1)), {"A_gate_recheck"}),
    "참여자 바꿔치기": (lambda rs: _by_id(rs, "img-0001").__setitem__("client", "C2"),
                 {"F_attribution", "F_counts_clients", "F_meta_counts_by_split"}),
    "레코드 하나 삭제": (lambda rs: [r for r in rs if r["image_id"] != "img-0002"],
                  {"F_id_accounting", "F_discard_accounting"}),
    "행 순서": (lambda rs: rs[::-1], {"F_order_or_duplicate_ids"}),
    "정상 페어에 조항": (lambda rs: _by_id(rs, "img-0002")["skeleton"].__setitem__("clauses", ["X-1"]),
                  {"D_normal_record", "A_gate_recheck"}),
    "eval 이미지를 실음": (lambda rs: _by_id(rs, "img-0002").__setitem__("image_id", "img-9001"),
                     {"F_eval_overlap", "F_not_in_train_val", "F_id_accounting"}),
    "기하 무효 수를 뺌": (lambda rs: _by_id(rs, "img-0001").pop(V.SKIPPED_KEY), {"E_skipped_count"}),
    "좌표 규약": (lambda rs: _by_id(rs, "img-0001").__setitem__("coord_space", "NORM_1000"),
              {"E_record_differs_from_source", "A_gate_recheck"}),
    "이미지 크기": (lambda rs: _by_id(rs, "img-0001").__setitem__("width_px", W + 1), {"F_attribution", "A_gate_recheck"}),
}


@pytest.mark.parametrize("label", sorted(CORRUPTIONS))
def test_깨뜨린_산출물은_그_결함의_사유로_걸린다(built, label):
    out, snap, root = built
    fn, want = CORRUPTIONS[label]
    _edit(out, fn)
    got = _kinds(out, snap, root)
    assert want <= got, (label, got)
    assert not {k for k in got if k.startswith("G_contract")}, got      # 계약서는 다시 썼다 — 걸린 것은 내용이다


def test_키_순서가_바뀌면_걸린다(built):
    out, snap, root = built

    def reorder(rs):
        r = _by_id(rs, "img-0002")
        r["target_text"], r["skeleton"] = r.pop("target_text"), r.pop("skeleton")
    _edit(out, reorder)
    assert "G_key_order" in _kinds(out, snap, root)


def test_계약서를_다시_쓰지_않은_변경과_CRLF_는_바이트에서_걸린다(built):
    out, snap, root = built
    _edit(out, lambda rs: _by_id(rs, "img-0001").__setitem__("split", "val"), reseal=False)
    assert "G_contract_hash" in _kinds(out, snap, root)
    p = out / "pairs.jsonl"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert "G_cr_in_output" in _kinds(out, snap, root)


def test_회계와_메타를_고치면_걸린다(built):
    out, snap, root = built
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    counts["discarded"]["quarantine"] += 1
    M.write_json(out / "counts.json", counts)
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["eval_isolation"]["eval_group_id_overlap"] = 3
    meta["discard_reasons"] = {}
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    assert {"F_discard_accounting", "F_meta_isolation", "F_meta_discard_reasons"} <= _kinds(out, snap, root)


def test_대조하지_못한_축은_통과로_세지_않는다(built):
    """적발이 없는 것과 볼 수 있는 것만 본 것을 가른다. 실패는 아니므로 종료 코드는 2 다.

    표 해시 선언이 **없는 것**은 여기 있지 않다 — 그것은 대조 못 함이 아니라 계약 위반이다(검토 §27-19).
    """
    label, axis = "경로 주장이 옛 참·거짓 형식", "meta.image_paths_checked"
    out, snap, root = built
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["image_paths_checked"] = True
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    rep = V.verify(out, snap, root)
    assert rep["ok"] and rep["failures"] == {}, label
    assert rep["coverage"] == "partial" and rep["not_checked"] == [axis], label
    assert rep["verdict"] != "일치", label
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 2, label


def test_경로_주장의_건수가_투입과_다르면_적발이다(built):
    out, snap, root = built
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["image_paths_checked"] = {"n": 1, "of": 99, "basis": "x"}
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    rep = V.verify(out, snap, root)
    assert "F_meta_paths_partial" in rep["failures"]
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 1


def test_원천의_어노테이션_수가_매니페스트와_다르면_걸린다(built):
    """원천을 고치면 스냅샷 계약도 함께 깨진다 — 둘 다 적발이어야 한다."""
    out, snap, root = built
    col = list(MANIFEST_COLUMNS).index("n_defects")
    lines = (snap / "manifest.csv").read_text(encoding="utf-8").splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("img-0005,"):
            cells = ln.split(",")
            cells[col] = "2"
            lines[i] = ",".join(cells)
    (snap / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    got = _kinds(out, snap, root)
    assert "E_source_n_defects" in got
    # 원천이 바뀌었으면 출처 주장도 더는 참이 아니다. 조용히 지나가면 digest 가 장식이 된다.
    assert "F_snapshot_unverifiable" in got


# ------------------------------------------------------------------ 검토 §27-19 의 사례


def test_정상적으로_비어_있는_폐기_파일은_이상이_아니다(tmp_path):
    """폐기할 항목이 없는 입력에서 빌더가 만든 0바이트 파일을 개행 누락으로 거부하던 자리다."""
    rows = [r for r in ROWS if r["image_id"] != "img-0004"]      # 유효 기하가 0 인 결함 이미지를 뺀다
    anns = [a for a in ANNS if a["image_id"] != "img-0004"]
    out, snap, root = _build(tmp_path, rows, anns)
    assert (out / "discarded.jsonl").stat().st_size == 0
    rep = V.verify(out, snap, root)
    assert rep["ok"] and rep["coverage"] == "full" and rep["failures"] == {}
    assert rep["n_discarded"] == 0 and rep["verdict"] == "일치"
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 0


def test_정상_항목을_근거_없이_폐기하고_회계를_맞춰도_걸린다(built):
    """폐기를 원천에서 **다시 유도**한다. 회계만 맞추면 지나가던 자리다."""
    out, snap, root = built
    recs = [json.loads(ln) for ln in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    victim = _by_id(recs, "img-0002")                            # 멀쩡한 정상 페어를 폐기로 옮긴다
    M.write_jsonl(out / "pairs.jsonl", [r for r in recs if r["image_id"] != victim["image_id"]])
    disc = [json.loads(ln) for ln in (out / "discarded.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    disc.append({"image_id": victim["image_id"], "reasons": ["invented_reason"]})
    M.write_jsonl(out / "discarded.jsonl", sorted(disc, key=lambda d: d["image_id"]))
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    counts["clients"][victim["client"]]["n_total"] -= 1
    counts["clients"][victim["client"]]["n_normal"] -= 1
    counts["discarded"]["quarantine"] = len(disc)
    M.write_json(out / "counts.json", counts)
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["discard_reasons"] = {x: 1 for d in disc for x in d["reasons"]}
    meta["counts_by_split"]["train"].pop(victim["client"])
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    got = _kinds(out, snap, root)
    assert {"F_discard_unjustified", "F_discard_reason_unknown"} <= got, got


META_MUTATIONS = {
    "입력 digest 를 바꾼다": (lambda m: m["input_snapshot"].__setitem__("snapshot_digest", "0" * 64),
                        "F_meta_input_digest"),
    "출처 선언을 지운다": (lambda m: m.pop("input_snapshot"), "F_meta_input_snapshot_missing"),
    "검수 주체만 사람으로 바꾼다": (lambda m: m.__setitem__("validated_by", "human_double_checked"),
                          "F_meta_validated_by_unsupported"),
    "표 해시를 지운다": (lambda m: m["limits_table"].pop("sha256"), "F_meta_limits_hash_missing"),
    "표 선언을 통째로 지운다": (lambda m: m.pop("limits_table"), "F_meta_limits_hash_missing"),
}


@pytest.mark.parametrize("label", sorted(META_MUTATIONS))
def test_메타의_출처와_검수_주체를_대조한다(built, label):
    """다섯 다 통과하던 자리다. 선언만 있고 대조가 없으면 아무 값이나 적어도 지나간다."""
    out, snap, root = built
    fn, want = META_MUTATIONS[label]
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    fn(meta)
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    rep = V.verify(out, snap, root)
    assert want in rep["failures"], (label, rep["failures"])
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 1


def test_기록_파일을_가리켜도_사람_검수라는_이름은_받지_않는다(built, tmp_path):
    """파일의 실재는 아무것도 뒷받침하지 않는다 — `README.md` 를 가리켜도 지나가던 자리다(교차 검수 I-3).

    검수 기록의 규약(대상 빌드 digest·표본 수·서명)이 서기 전에는 `rule` 외의 값을 코드가 받지 않는다.
    """
    out, snap, root = built
    rec = tmp_path / "gold/pairs_review.md"
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text("표본 검수 기록", encoding="utf-8")
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["validated_by"] = "rule+human_sample"
    meta["human_verification"] = {"record_path": "gold/pairs_review.md"}
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)
    assert "F_meta_validated_by_unsupported" in _kinds(out, snap, root)


def _move_to_discard(out, iid, reasons):
    """`iid` 를 채택에서 빼 폐기로 옮기고 **회계·메타를 전부 맞춰 둔다.** 회계로는 흠잡을 데가 없게."""
    recs = [json.loads(ln) for ln in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    victim = _by_id(recs, iid)
    M.write_jsonl(out / "pairs.jsonl", [r for r in recs if r["image_id"] != iid])
    disc = [json.loads(ln) for ln in (out / "discarded.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    disc.append({"image_id": iid, "reasons": reasons} if reasons is not None else {"image_id": iid})
    M.write_jsonl(out / "discarded.jsonl", sorted(disc, key=lambda d: d["image_id"]))
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    counts["clients"][victim["client"]]["n_total"] -= 1
    counts["clients"][victim["client"]]["n_normal" if not victim["skeleton"]["defects"] else "n_defect"] -= 1
    counts["discarded"]["quarantine"] = len(disc)
    M.write_json(out / "counts.json", counts)
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    meta["discard_reasons"] = {x: 1 for d in disc for x in d.get("reasons", [])}
    meta["counts_by_split"][victim["split"]].pop(victim["client"])
    M.write_json(out / "PAIRS_META.json", meta)
    M.snapshot_digest(out, V.MEMBERS)


def test_사유가_빈_목록인_폐기는_폐기가_아니다(built):
    """빈 목록은 재유도의 기대값과 **등치가 성립해** 조용히 지나갔다. 회계까지 맞춰 둔 최악의 모양으로 건다."""
    out, snap, root = built
    _move_to_discard(out, "img-0002", [])
    got = _kinds(out, snap, root)
    assert "F_discard_record_schema" in got, got
    assert "F_discard_reason_unknown" not in got      # 어휘 문제가 아니다 — 사유가 아예 없는 것이다


def test_사유가_있어도_다시_조립해_통과하면_부당_폐기다(built):
    """어휘 안의 사유를 골라 붙여도 원천에서 통과하는 레코드는 폐기 대상이 아니다."""
    out, snap, root = built
    _move_to_discard(out, "img-0002", [M.NO_VALID_GEOMETRY])
    got = _kinds(out, snap, root)
    assert "F_discard_unjustified" in got, got


@pytest.mark.parametrize("bad", [
    {"reasons": ["x"]},                               # image_id 가 없다
    {"image_id": "img-0002"},                         # reasons 가 없다
    {"image_id": "img-0002", "reasons": "schema:x"},  # 목록이 아니다
    {"image_id": "img-0002", "reasons": [3]},         # 사유가 문자열이 아니다
    ["img-0002"],                                     # 사전이 아니다
])
def test_폐기_항목의_스키마가_깨져도_죽지_않는다(built, bad):
    """예외로 죽으면 판정이 아니라 사고 보고가 된다. 집계·재유도가 KeyError 를 내던 자리다."""
    out, snap, root = built
    disc = [json.loads(ln) for ln in (out / "discarded.jsonl").read_text(encoding="utf-8").splitlines() if ln]
    M.write_jsonl(out / "discarded.jsonl", [*disc, bad])
    M.snapshot_digest(out, V.MEMBERS)
    assert "F_discard_record_schema" in _kinds(out, snap, root)


@pytest.mark.parametrize("name", ["discarded.jsonl", "counts.json", "SNAPSHOT.sha256"])
def test_계약_구성원이_없으면_죽지_않고_판정으로_낸다(built, name):
    """읽다가 예외로 죽으면 봉인 판정이 아니라 사고 보고가 된다. 뒤의 대조는 성립하지 않으므로 범위 제한으로 적는다."""
    out, snap, root = built
    (out / name).unlink()
    rep = V.verify(out, snap, root)
    assert "G_member_missing" in rep["failures"] and not rep["ok"] and rep["coverage"] == "partial"
    assert V.main(["--build", str(out), "--snapshot", str(snap), "--root", str(root)]) == 1


def test_구성원이_비면_이상이다_폐기_파일만_예외다(built):
    out, snap, root = built
    (out / "counts.json").write_bytes(b"")
    assert "G_member_empty" in _kinds(out, snap, root)
