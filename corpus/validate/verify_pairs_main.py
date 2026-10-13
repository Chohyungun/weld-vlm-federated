"""본실험 페어 빌드의 산출물 검사 — **읽기 전용.** 빌드 디렉터리 하나를 받아 파일 수준에서 다시 본다.

빌더의 게이트는 메모리 안의 레코드를 본다. 여기서는 **디스크에 떨어진 바이트**를 원천과 맞댄다(06번 §5).
매니페스트와 어노테이션은 적재기를 거치지 않고 `csv` 로 직접 읽는다 — 검산에 한한 예외다. eval 행은 id 집합에만 쓰고
eval 의 어노테이션은 색인에 넣지 않는다(개발규약 1-4).

| 관점 | 여기서 보는 것 |
|---|---|
| A 부등식·구간 | 결함 레코드 전부를 파일에서 읽어 게이트 ③(허용치 행과의 등치)에 다시 건다. 서술에서 읽은 기준 구조의 종류 수가 표의 행과 맞는다 |
| B 조합 | (재질 × 첫 등장 코드열 × 기하 무효 유무 × 정상)을 **원천 CSV 에서** 열거해 레코드의 조합과 1:1 로 맞춘다. 개수를 가린 서술이 조합마다 하나다 |
| D 정상 어휘 | 정상 레코드는 고정 문장이고 조항·행·미특정이 비었고 결함 어휘가 없다 |
| E 상자·좌표 | 레코드의 결함 목록이 원천 어노테이션(유효 기하, `ann_id` 문자열 순)과 코드·상자·크기까지 같다. 경계 안, 좌표 규약 명시 |
| F 격리·귀속·회계 | eval 과의 이미지·묶음 교차 0(직접 재집계), 참여자·분할·재질·경로·크기 = 매니페스트, counts·메타의 건수 = 재집계, 폐기 = 투입 − 채택 |
| G 결정론·바이트 | 계약서의 해시 = 파일, CR 0, 레코드마다 키 순서 고정, 줄 = 재직렬화 |

관점 C(빌더를 import 하지 않는 독립 재구현)는 E 가 맡는다 — 이 스크립트는 빌더와 같은 트랙의 검사다.

실행: uv run python -m corpus.validate.verify_pairs_main --build <빌드 디렉터리>
결과는 표준 출력의 JSON 하나다. 종료 코드는 공개 검수기와 같은 규약이다 —
**0 = 적발 없음 + 전 축 대조, 1 = 적발, 2 = 적발은 없으나 대조하지 못한 축이 있다.**
봉인 조건은 **0** 이다. 파일을 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "data/interim/manifest_v1"
RECORD_KEYS = ["image_id", "image_path", "client", "split", "material", "width_px", "height_px",
               "coord_space", "skeleton", "target_text"]
SKELETON_KEYS = ["defects", "verdict", "verdict_mode", "clauses", "candidate_rules", "uncited_codes"]
DEFECT_KEYS = ["type", "bbox_px", "size_px", "size_mm"]
SKIPPED_KEY = "n_annotations_skipped_geom_invalid"
MEMBERS = ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json")
#: **정상적으로 비어 있을 수 있는** 구성원. 폐기할 항목이 없는 입력에서 빈 파일이 나온다.
#: 나머지는 비어 있으면 이상이다 — 채택 0건은 빌드가 애초에 거부한다.
MAY_BE_EMPTY = frozenset({"discarded.jsonl"})
#: 메타의 경로 확인 주장을 되볼 때 열어 볼 표본 수. 전수는 빌더가 본다 — 여기서 되푸는 것은 주장의 진위다.
PATH_SAMPLE = 500
#: 서술의 개수를 가려 틀만 남긴다. ASCII 숫자만 본다 — 다른 숫자 체계를 같은 틀로 뭉개지 않는다.
_COUNT_RE = re.compile(r"[0-9]+개")


def _reject_constant(name: str):
    raise ValueError(f"표준 JSON 이 아니다: {name}")


def _truth(s: str) -> bool:
    if s not in ("True", "False"):
        raise ValueError(f"참·거짓이 아니다: {s!r}")
    return s == "True"


def read_sources(snap: Path):
    """매니페스트의 train·val 행, eval 의 id 집합, train·val 이미지의 어노테이션."""
    tv, eval_ids, eval_groups = {}, set(), set()
    with (snap / "manifest.csv").open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "eval":
                eval_ids.add(r["image_id"])
                eval_groups.add(r["group_id"])
            elif r["split"] in ("train", "val"):
                assert r["image_id"] not in tv, r["image_id"]
                tv[r["image_id"]] = r
            else:
                raise ValueError(f"모르는 split: {r['split']!r}")
    anns = defaultdict(list)
    with (snap / "annotations.csv").open(encoding="utf-8", newline="") as fh:
        for a in csv.DictReader(fh):
            if a["image_id"] in tv:
                anns[a["image_id"]].append(a)
    assert not (set(anns) & eval_ids), "색인에 eval 이미지의 어노테이션이 있다"
    return tv, eval_ids, eval_groups, anns


def first_seen(codes):
    out = []
    for c in codes:
        if c not in out:
            out.append(c)
    return out


def verify(build: Path, snap: Path = SNAP, root: Path = REPO) -> dict:
    from corpus.generate import make_pairs_pilot as P
    from corpus.generate.counts_builder import CountsError, validate_counts
    from corpus.generate.run_cycle_corpus import defect_names
    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon
    from corpus.validate.pair_criteria_gate import PairRules, check_text, criterion_of_row
    from data.manifest_io import verify_snapshot

    fails: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    #: **대조하지 못한 축.** 없는 키를 조용히 지나가면 "봤는데 같았다" 와 "볼 수 없었다" 가 섞인다.
    not_checked: list[str] = []

    def skip(axis: str) -> None:
        not_checked.append(axis)

    def fail(kind: str, what) -> None:
        fails[kind] += 1
        if len(examples[kind]) < 3:
            examples[kind].append(str(what)[:160])

    # ---------------------------------------------------------------- G 바이트
    # 계약서 자체가 없을 수 있다. 읽다가 죽으면 봉인 판정이 아니라 사고 보고가 된다 — 아래에서 적발로 낸다.
    contract_file = build / "SNAPSHOT.sha256"
    contract = contract_file.read_text(encoding="utf-8") if contract_file.is_file() else ""
    listed = dict(reversed(ln.split("  ", 1)) for ln in contract.splitlines() if ln and not ln.startswith("#"))
    if sorted(listed) != sorted(MEMBERS):
        fail("G_contract_members", sorted(listed))
    raw = {}
    for name in (*MEMBERS, "SNAPSHOT.sha256"):
        f = build / name
        if not f.is_file():
            # 계약 구성원이 아예 없다. 읽다가 죽지 않고 **판정으로** 낸다 — 봉인 조건에서 걸러야 한다.
            fail("G_member_missing", name)
            raw[name] = b""
            continue
        raw[name] = f.read_bytes()
        if b"\r" in raw[name]:
            fail("G_cr_in_output", name)
    for name in MEMBERS:
        if hashlib.sha256(raw[name]).hexdigest() != listed.get(name):
            fail("G_contract_hash", name)
    digest = hashlib.sha256("".join(listed[n] for n in MEMBERS if n in listed).encode()).hexdigest()
    if f"# snapshot_digest {digest}" not in contract:
        fail("G_contract_digest", digest)

    for name in (*MEMBERS, "SNAPSHOT.sha256"):
        if raw[name].startswith(codecs.BOM_UTF8):
            fail("G_bom", name)                     # BOM 은 소비처마다 다르게 읽힌다
        if name in MAY_BE_EMPTY and not raw[name]:
            # **0 바이트가 곧 이상이 아니다.** 폐기할 항목이 없으면 빌더가 빈 파일을 만든다 —
            # 그것을 개행 누락으로 거부하면 정상 입력이 봉인되지 못한다(검토 §27-19).
            continue
        if not raw[name].endswith(b"\n"):
            fail("G_file_no_trailing_newline", name)
        if not raw[name] and name not in MAY_BE_EMPTY:
            fail("G_member_empty", name)
    if fails["G_member_missing"] or fails["G_member_empty"]:
        # 내용 대조를 **할 수 없다.** 빈 바이트를 파싱하다 예외로 죽으면 판정이 아니라 사고 보고가 되고,
        # 기본값으로 메우면 없는 파일이 "같았다" 로 센다. 끊고 못 본 범위를 밝힌다(검토 §27-19).
        return {"build": str(build).replace("\\", "/"),
                "failures": dict(fails), "failure_examples": dict(examples),
                "not_checked": ["E 레코드", "F 격리·귀속·회계", "A 게이트 재검", "B 조합", "D 정상"],
                "coverage": "partial", "ok": False,
                "verdict": "불일치 — 계약 구성원이 없거나 비어 대조를 이어갈 수 없다"}
    strict = {"parse_constant": _reject_constant}   # NaN·Infinity 는 표준 JSON 이 아니다(검토 14)
    lines = raw["pairs.jsonl"].decode("utf-8").split("\n")
    if lines[-1] != "":
        fail("G_no_trailing_newline", "pairs.jsonl")
    recs = [json.loads(ln, **strict) for ln in lines[:-1]]
    for ln, r in zip(lines, recs):
        if json.dumps(r, ensure_ascii=False) != ln:
            fail("G_not_canonical_serialization", r.get("image_id"))
        keys = [k for k in r if k != SKIPPED_KEY]
        if keys != RECORD_KEYS or list(r["skeleton"]) != SKELETON_KEYS or (SKIPPED_KEY in r and list(r)[-1] != SKIPPED_KEY):
            fail("G_key_order", r.get("image_id"))
        if any(list(d) != DEFECT_KEYS for d in r["skeleton"]["defects"]):
            fail("G_defect_key_order", r.get("image_id"))
    discarded = [json.loads(ln, **strict) for ln in raw["discarded.jsonl"].decode("utf-8").splitlines() if ln]
    #: 스키마가 선 것만 재유도에 건다. **사유가 빈 목록인 폐기는 폐기가 아니다** — 정상 레코드를 다시 조립하면
    #: 기대 사유도 비어서 등치가 성립하고, 어휘 검사도 빈 목록에는 아무것도 더하지 않는다. 멀쩡한 페어를
    #: 사유 없이 버리고 회계만 맞추면 그대로 지나가던 자리다(교차 검수 I-1).
    well_formed = []
    for d in discarded:
        if (not isinstance(d, dict) or not isinstance(d.get("image_id"), str) or not d["image_id"]
                or not isinstance(d.get("reasons"), list) or not d["reasons"]
                or any(not isinstance(x, str) or not x for x in d["reasons"])):
            fail("F_discard_record_schema", str(d)[:120])
            continue
        well_formed.append(d)
    counts = json.loads(raw["counts.json"], **strict)
    meta = json.loads(raw["PAIRS_META.json"], **strict)
    extra = sorted(x.name for x in build.iterdir() if x.name not in (*MEMBERS, "SNAPSHOT.sha256"))
    if extra:
        fail("G_extra_files", extra)                # 계약 밖 파일이 봉인 디렉터리에 같이 들어가면 안 된다

    # ---------------------------------------------------------------- F 격리·귀속·회계
    tv, eval_ids, eval_groups, anns = read_sources(snap)
    ids = [r["image_id"] for r in recs]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        fail("F_order_or_duplicate_ids", len(ids))
    disc_ids = {d["image_id"] for d in well_formed}
    if set(ids) | disc_ids != set(tv) or set(ids) & disc_ids:
        fail("F_id_accounting", f"made {len(ids)} + discarded {len(disc_ids)} vs train·val {len(tv)}")
    overlap_i = len(set(ids) & eval_ids)
    overlap_g = len({tv[i]["group_id"] for i in ids if i in tv} & eval_groups)
    if overlap_i or overlap_g or set(tv) & eval_ids or {r["group_id"] for r in tv.values()} & eval_groups:
        fail("F_eval_overlap", (overlap_i, overlap_g))
    if meta.get("eval_isolation", {}).get("eval_image_id_overlap") != overlap_i or \
            meta.get("eval_isolation", {}).get("eval_group_id_overlap") != overlap_g or \
            meta.get("eval_isolation", {}).get("n_eval_images") != len(eval_ids):
        fail("F_meta_isolation", meta.get("eval_isolation"))
    for r in recs:
        m = tv.get(r["image_id"])
        if m is None:
            fail("F_not_in_train_val", r["image_id"])
            continue
        same = (r["client"] == m["client"] and r["split"] == m["split"] and r["material"] == m["material"]
                and r["image_path"] == m["rel_path"].replace("\\", "/")
                and (r["width_px"], r["height_px"]) == (int(m["width_px"]), int(m["height_px"])))
        if not same:
            fail("F_attribution", r["image_id"])
    per = {c: {"n_total": 0, "n_defect": 0, "n_normal": 0} for c in counts["clients"]}
    by_split: dict = {}
    for r in recs:
        kind = "n_defect" if r["skeleton"]["defects"] else "n_normal"
        for cell in (per.setdefault(r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0}),
                     by_split.setdefault(r["split"], {}).setdefault(
                         r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0})):
            cell["n_total"] += 1
            cell[kind] += 1
    if per != counts["clients"]:
        fail("F_counts_clients", per)
    if {s: dict(sorted(v.items())) for s, v in sorted(by_split.items())} != meta.get("counts_by_split"):
        fail("F_meta_counts_by_split", by_split)
    if counts["n_generated"] != len(tv) or sum(counts["discarded"].values()) != len(discarded) \
            or len(tv) - len(recs) != len(discarded):
        fail("F_discard_accounting", (counts["n_generated"], counts["discarded"], len(discarded)))
    reason_count = Counter(x for d in well_formed for x in d["reasons"])
    if dict(sorted(reason_count.items())) != meta.get("discard_reasons"):
        fail("F_meta_discard_reasons", dict(reason_count))
    unknown = sorted(x for x in reason_count if not P.known_reason(x))
    if unknown:
        # 어휘 밖 사유는 지어낸 것이다. 회계를 맞춰도 여기서 걸린다.
        fail("F_discard_reason_unknown", unknown)
    # counts.json 의 정합 규칙(§7-5)을 산출기와 같은 코드로 다시 본다 — 스키마 키·중복·합계(검토 6)
    try:
        validate_counts(counts, [{"image_id": r["image_id"], "defects": r["skeleton"]["defects"]} for r in recs])
    except CountsError as e:
        fail("F_counts_invalid", str(e))
    # 원천 해시 선언. 어긋난 counts 가 여기를 지나면 학습 착수 시점에야 죽는다.
    for key, path in (("manifest_snapshot", snap / "manifest.csv"), ("limits_snapshot", P.LIMITS_CSV)):
        if counts.get(key) != "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest():
            fail("F_counts_snapshot_hash", key)
    want_limits = hashlib.sha256(P.LIMITS_CSV.read_bytes()).hexdigest()
    declared = meta.get("limits_table", {}).get("sha256")
    if declared is None:
        # 필수 선언이 **없는 것**은 대조 못 함이 아니라 계약 위반이다(검토 §27-19).
        fail("F_meta_limits_hash_missing", "meta.limits_table.sha256")
    elif declared != want_limits:
        fail("F_meta_limits_hash", declared)
    # 입력 스냅샷 출처. 메타가 적은 digest 가 실제 스냅샷의 것과 같아야 한다 —
    # 선언만 있고 대조가 없으면 아무 값이나 적어도 지나간다.
    src = meta.get("input_snapshot", {})
    if not isinstance(src, dict) or not src.get("snapshot_digest"):
        fail("F_meta_input_snapshot_missing", src)
    else:
        try:
            real = verify_snapshot(snap)
        except Exception as e:                      # noqa: BLE001 — 스냅샷이 깨졌으면 그것도 판정이다
            fail("F_snapshot_unverifiable", str(e)[:120])
        else:
            if src["snapshot_digest"] != real:
                fail("F_meta_input_digest", src["snapshot_digest"])
    # 검수 주체. **`rule` 만 받는다.** 파일이 있다는 것은 아무것도 뒷받침하지 않는다 — `README.md` 를 가리켜도
    # 지나갔다(교차 검수 I-3). 사람 검수라는 이름은 규약(대상 빌드 digest·표본 수·서명)이 선 뒤에 코드가 받는다.
    # 개발규약 3-4 와 봉인 절차 §7 이 그 이름을 함부로 쓰지 말라고 한 자리다. 총괄 판정으로 E 안에 맞춘다.
    if meta.get("validated_by") != "rule":
        fail("F_meta_validated_by_unsupported", meta.get("validated_by"))
    if meta.get("eval_isolation", {}).get("n_eval_groups") != len(eval_groups):
        fail("F_meta_isolation", meta.get("eval_isolation"))
    if meta.get("coord_space") != P.COORD_SPACE or meta.get("verdict_mode") != "clause_only":
        fail("F_meta_axis", (meta.get("coord_space"), meta.get("verdict_mode")))
    n_skipped_src = sum(len(anns.get(i, [])) - sum(1 for a in anns.get(i, []) if _truth(a["geom_valid"])) for i in tv)
    if (meta.get("n_annotations_skipped_in_records"), meta.get("n_annotations_skipped_geom_invalid")) !=             (sum(r.get(SKIPPED_KEY, 0) for r in recs), n_skipped_src):
        fail("F_meta_skipped", (meta.get("n_annotations_skipped_in_records"),
                                meta.get("n_annotations_skipped_geom_invalid")))
    if sorted(meta.get("materials", {}).get("present", [])) != sorted({m["material"] for m in tv.values()}):
        fail("F_meta_materials", meta.get("materials"))
    # "경로를 확인했다" 는 주장은 검사기가 **표본으로 되본다.** 주장만 믿으면 검사를 끈 빌드와 구별되지 않는다.
    claim = meta.get("image_paths_checked")
    n_sampled = 0
    if not isinstance(claim, dict):
        # 옛 빌드는 참·거짓만 실었다. 되볼 건수가 없으니 **대조하지 못한 축**이다 — 실패가 아니다.
        skip("meta.image_paths_checked")
    else:
        if claim.get("n") != len(tv):
            fail("F_meta_paths_partial", claim)
        step = max(1, len(recs) // PATH_SAMPLE)
        for r in recs[::step]:
            n_sampled += 1
            if not (Path(root) / r["image_path"]).is_file():
                fail("F_image_path_missing", r["image_path"])

    # ---------------------------------------------------------------- E 레코드 (원천에서 다시 조립해 바이트로)
    # 파이썬 등치는 `10.0 == 10` 도 `True == 1` 도 참이라, 자료형이 바뀐 산출물을 그냥 통과시킨다(검토 5).
    # 원천 행에서 레코드를 **다시 조립해 직렬화 바이트**로 맞댄다 — 키 집합·키 순서·수의 표기·verdict_mode 가
    # 한 번에 닫힌다. (빌더를 부르지 않는 독립 재구현은 따로 있다 — 이 검사는 "파일이 지금 코드의 산출과 같은가" 다.)
    names_for_build = defect_names()
    table = limits_loader.load_limits(str(P.LIMITS_CSV), pilot=True)
    bases = {mat: P.clause_basis(table, mat) for mat in sorted({m["material"] for m in tv.values()})}
    src_combo: Counter = Counter()
    for iid, m in tv.items():
        rows = sorted(anns.get(iid, []), key=lambda a: a["ann_id"])
        valid = [a for a in rows if _truth(a["geom_valid"])]
        if len(rows) != int(m["n_defects"]):
            fail("E_source_n_defects", iid)
        if _truth(m["has_defect"]) != (int(m["n_defects"]) > 0):
            fail("E_source_label_mismatch", iid)
        if _truth(m["has_defect"]) and not valid:
            if iid not in disc_ids:
                fail("E_should_be_discarded", iid)
            continue
        if iid in disc_ids:
            continue                    # 게이트가 폐기한 이미지는 조합 대조에서 뺀다(검토 14의 거짓 경보)
        src_combo[(m["material"], tuple(first_seen(a["iso_code"] for a in valid)), len(rows) > len(valid))] += 1
    rec_combo: Counter = Counter()
    templates: dict = defaultdict(set)
    for r in recs:
        iid = r["image_id"]
        src = tv.get(iid)
        if src is not None:
            row = dict(src, width_px=int(src["width_px"]), height_px=int(src["height_px"]),
                       has_defect=_truth(src["has_defect"]))
            want, skipped = P.assemble_record(row, anns.get(iid, []), names_for_build, bases[src["material"]])
            if want is None or json.dumps(want, ensure_ascii=False) != json.dumps(r, ensure_ascii=False):
                fail("E_record_differs_from_source", iid)
            if r.get(SKIPPED_KEY, 0) != skipped:
                fail("E_skipped_count", iid)
        for d in r["skeleton"]["defects"]:
            x1, y1, x2, y2 = d["bbox_px"]
            if not (0 <= x1 < x2 <= r["width_px"] and 0 <= y1 < y2 <= r["height_px"]):
                fail("E_bbox_bounds", iid)
        combo = (r["material"], tuple(first_seen(d["type"] for d in r["skeleton"]["defects"])), SKIPPED_KEY in r)
        rec_combo[combo] += 1
        templates[combo[:2]].add(_COUNT_RE.sub("N개", r["target_text"]))

    # ---------------------------------------------------------------- B 조합
    if src_combo != rec_combo:
        fail("B_combo_counts", f"원천 {len(src_combo)}종 vs 레코드 {len(rec_combo)}종")
    for combo, texts in templates.items():
        if len(texts) != 1:
            fail("B_combo_text_not_unique", combo)

    # ---------------------------------------------------------------- 폐기의 재유도
    # 회계만 맞으면 지나가던 자리다. 폐기된 이미지를 원천에서 **다시 조립해 같은 사유가 나오는지** 본다 —
    # 정상 항목을 근거 없이 폐기하고 회계를 맞춰도 여기서 걸린다(검토 §27-19).
    for d in well_formed:
        iid = d["image_id"]
        src = tv.get(iid)
        if src is None:
            fail("F_discarded_not_in_source", iid)
            continue
        row = dict(src, width_px=int(src["width_px"]), height_px=int(src["height_px"]),
                   has_defect=_truth(src["has_defect"]))
        rec2, _ = P.assemble_record(row, anns.get(iid, []), names_for_build, bases[src["material"]])
        want = ([P.NO_VALID_GEOMETRY] if rec2 is None else
                P.check_pair(rec2, names_for_build, PairRules(table.rows, names_for_build),
                             load_defect_lexicon(), wh=(row["width_px"], row["height_px"])))
        if not want:
            # 다시 조립했더니 **통과하는 레코드**다. 폐기 대상이 아니므로 사유가 무엇이든 부당 폐기다.
            fail("F_discard_unjustified", (iid, d["reasons"], "재조립 결과 통과 — 폐기 대상이 아니다"))
        elif sorted(d["reasons"]) != sorted(want):
            fail("F_discard_unjustified", (iid, d["reasons"], want))

    # ---------------------------------------------------------------- A 게이트 재검 · D 정상
    names, lexicon = names_for_build, load_defect_lexicon()
    rules = PairRules(table.rows, names)
    seen_criteria: dict = defaultdict(set)
    n_defect = n_normal = 0
    for r in recs:
        src = tv.get(r["image_id"])                  # 없으면 F 에서 이미 걸렸다 — 여기서는 크기 대조만 건너뛴다
        bad = P.check_pair(r, names, rules, lexicon,
                           wh=(int(src["width_px"]), int(src["height_px"])) if src else None)
        if bad:
            fail("A_gate_recheck", (r["image_id"], bad))
        if r["skeleton"]["defects"]:
            n_defect += 1
            _, parsed = check_text(r, rules)
            for _name, clause_id, crit, _note in (parsed.blocks if parsed else ()):
                seen_criteria[clause_id].update(crit)
        else:
            n_normal += 1
            sk = r["skeleton"]
            if r["target_text"] != P.NORMAL_TEXT or sk["clauses"] or sk["candidate_rules"] or sk["uncited_codes"]:
                fail("D_normal_record", r["image_id"])
    want_criteria: dict = defaultdict(set)
    for mat in sorted({r["material"] for r in recs}):
        for code in sorted({d["type"] for r in recs for d in r["skeleton"]["defects"]}):
            for row in rules.rows_for(mat, code):
                want_criteria[row.clause_id].add(criterion_of_row(row))
    if {k: v for k, v in seen_criteria.items()} != {k: v for k, v in want_criteria.items()}:
        fail("A_criteria_seen_vs_rows", {k: len(v) for k, v in seen_criteria.items()})

    return {
        "build": str(build).replace("\\", "/"),
        "snapshot_digest": digest,
        "n_records": len(recs), "n_defect": n_defect, "n_normal": n_normal, "n_discarded": len(discarded),
        "n_train_val_in_manifest": len(tv), "n_eval_images": len(eval_ids), "n_eval_groups": len(eval_groups),
        "eval_overlap": {"image_id": overlap_i, "group_id": overlap_g},
        "n_combos": len(rec_combo), "n_text_templates": sum(len(v) for v in templates.values()),
        "criteria_structures_by_clause": {k: len(v) for k, v in sorted(seen_criteria.items())},
        "n_annotations_in_records": sum(len(r["skeleton"]["defects"]) for r in recs),
        "n_annotations_skipped_in_records": sum(r.get(SKIPPED_KEY, 0) for r in recs),
        "n_image_paths_sampled": n_sampled,
        "bytes": {n: len(raw[n]) for n in MEMBERS},
        "failures": dict(fails), "failure_examples": dict(examples),
        # `ok` 는 "적발이 없다" 이고 `coverage` 는 "전부 봤다" 다. 둘을 섞으면 못 본 것이 통과로 센다.
        "not_checked": not_checked,
        "coverage": "full" if not not_checked else "partial",
        "ok": not fails,
        "verdict": ("일치" if not fails and not not_checked else
                    "대조 범위 제한 — 본 것은 전부 같았다" if not fails else "불일치"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--snapshot", type=Path, default=SNAP,
                    help="빌드가 나온 매니페스트 스냅샷. 기본은 본실험 동결본이다 —"
                         " 다른 스냅샷의 빌드를 검사하려면 명시한다")
    ap.add_argument("--root", type=Path, default=REPO,
                    help="`image_path` 를 푸는 기준. 레코드의 경로는 저장소 루트 기준 상대경로다")
    args = ap.parse_args(argv)
    report = verify(args.build, args.snapshot, args.root)
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)
    sys.stdout.write("\n")
    if not report["ok"]:
        return 1
    return 0 if report["coverage"] == "full" else 2      # 공개 검수기와 같은 규약


if __name__ == "__main__":
    raise SystemExit(main())
