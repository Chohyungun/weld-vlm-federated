"""본채점 진입점의 진입 전 관문 — 진입점 미니스펙 3판 1-1-가 · 1-1-나 · 3-5 · 07번 §32-1 · §32-3 · §32-6.

**합성 저장소와 합성 스냅샷만 쓴다.** 저장소는 시험마다 `git init` 으로 새로 만들고 등록 파일 · 설정 파일을 커밋한다.
동결 자산을 읽지 않는다. 본체 저장소는 `guard_seam` 시험에서 **커밋 SHA 하나**(루트 커밋)만 읽는다.

지키는 것.

1. 등록은 **영수증 커밋의 바이트**로 만든다 — 작업 트리를 고쳐도 커밋의 것이 쓰인다. 파일 해시가 다르면 멈춘다.
2. 커밋이 본줄기(`refs/heads/main`)의 조상이 아니면 멈춘다.
3. 닻은 설정 파일의 커밋 바이트에서 읽는다 — 엄격 · 진단은 영수증 커밋, 리허설은 `refs/heads/main`.
4. 스냅샷은 digest 가 닻과 같을 때만 쓴다. 흡수 상태를 돌려준다.
5. 탐색 채점은 본실험 목적 · 영수증을 받지 않는다.
6. 목록의 id 는 그 목적의 분할이어야 한다. 매니페스트는 네 열만 읽는다 — 평가 행의 라벨 열에 독을 심어도 닿지 않는다.
7. 커밋된 영수증으로는 이음새를 쓸 수 없다.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from data.manifest_io import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, write_snapshot
from evaluation import entry_gate as G
from evaluation.prereg_unified import Receipt, UnifiedRegistration, dump_registration

REG_PATH = "configs/registration/main-20261001-1.json"
POISON = "__POISON__"


# ---------------------------------------------------------------- 합성 자산

def _man_row(image_id, split, client="C1"):
    tail = image_id.split(":")[-1]
    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{tail}.jpg",
        "sha256": tail.zfill(8), "width_px": 1280, "height_px": 720, "modality": "RT", "material": "ST",
        "has_defect": True, "n_defects": 1, "defect_types": "porosity", "iso_codes": "2011",
        "src_labels_raw": "기공", "label_type": "polygon", "has_localization": True, "phash_hex": "0" * 16,
        "group_id": f"g{tail}", "group_size": 1, "strata_key": "ST|porosity", "split": split,
        "client": client, "eval_subset": "", "thickness_mm": None, "thickness_source": "unavailable",
        "px_per_mm": None, "scale_source": "unavailable", "quality_level": "", "ingest_version": "t",
        "label_map_version": 1, "notes": "",
    }


def _ann_row(image_id):
    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#0", "image_id": image_id, "src_label_raw": "기공", "defect_type": "porosity",
        "iso_code": "2011", "polygon_json": "[]", "bbox_x1_px": 10, "bbox_y1_px": 10, "bbox_x2_px": 20,
        "bbox_y2_px": 20, "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


TRAIN, VAL, VAL2, EVAL = "aihub71761:11", "aihub71761:12", "aihub71761:15", "aihub71761:13"


def make_snapshot(root: Path, *, absorption: dict | None = None) -> str:
    """평가 행의 라벨 열 · 층 열에 **값으로 만들면 터지는 독**을 심는다 — 네 열 판독이 닿지 않아야 한다."""
    man = [_man_row(TRAIN, "train"), _man_row(VAL, "val", "C2"), _man_row(VAL2, "val", "C2"),
           _man_row(EVAL, "eval", "")]
    for r in man:
        if r["split"] == "eval":
            r.update({c: POISON for c in ("has_defect", "n_defects", "defect_types", "iso_codes",
                                         "src_labels_raw", "strata_key")})
    caps = {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only", "localization": True}}
    if absorption is not None:
        caps["absorption"] = absorption
    return write_snapshot(root, pd.DataFrame(man), pd.DataFrame([_ann_row(TRAIN), _ann_row(EVAL)]), caps)


def git(repo: Path, *args: str) -> str:
    res = subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                          "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false", *args],
                         capture_output=True, check=True)
    return res.stdout.decode("utf-8").strip()


def base_yaml(digest: str, processor: str | None = "{max_pixels: 921600}") -> bytes:
    proc = f"  uni_processor_kwargs: {processor}\n" if processor is not None else ""
    return (f"fixed_before_main_runs:\n  snapshot_id: synthetic_v1\n  snapshot_digest: {digest}\n{proc}"
            f"experiment:\n  seeds: [1, 2]\n").encode("utf-8")


def registration(digest: str, purpose: str = "main") -> UnifiedRegistration:
    r = UnifiedRegistration()
    r.generation.purpose = purpose
    r.generation.list_split = "eval" if purpose == "main" else "val"
    r.generation.snapshot_digest = digest
    r.generation.processor_max_pixels = 921600
    return r


def receipt(r: UnifiedRegistration, commit: str, *, kind: str | None = None, path: str = REG_PATH,
            file_sha: str | None = None) -> Receipt:
    return Receipt(generation_sha256=r.generation_sha256(), scoring_sha256=r.scoring_sha256(),
                   registered_at="2026-10-01T10:00:00+09:00", main_commit=commit,
                   kind=kind or r.generation.purpose, registration_path=path,
                   registration_file_sha256=file_sha or hashlib.sha256(dump_registration(r)).hexdigest())


@pytest.fixture
def world(tmp_path: Path):
    """합성 저장소 하나 — `main` 에 설정 파일과 본실험 등록을 커밋했다. 스냅샷은 저장소 밖에 둔다."""
    snap = tmp_path / "snap"
    digest = make_snapshot(snap)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "configs/registration").mkdir(parents=True)
    (repo / "configs/base.yaml").write_bytes(base_yaml(digest))
    reg = registration(digest)
    (repo / REG_PATH).write_bytes(dump_registration(reg))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "등록")
    return {"repo": repo, "snap": snap, "digest": digest, "reg": reg, "commit": git(repo, "rev-parse", "HEAD")}


# ---------------------------------------------------------------- 1 · 2 등록과 영수증

def test_영수증_커밋의_바이트로_등록을_만든다(world) -> None:
    got = G.load_registration_for(receipt(world["reg"], world["commit"]), repo=world["repo"])
    assert got.generation_sha256() == world["reg"].generation_sha256()


def test_작업_트리의_등록을_고쳐도_커밋의_것을_쓴다(world) -> None:
    """3판 가-10 의 둘째 · 진-1 — "영수증이 가리키는 등록" 과 "지금 쓰는 등록" 이 같은 것이어야 한다."""
    edited = registration(world["digest"])
    edited.generation.max_new_tokens = 4096
    (world["repo"] / REG_PATH).write_bytes(dump_registration(edited))
    got = G.load_registration_for(receipt(world["reg"], world["commit"]), repo=world["repo"])
    assert got.generation.max_new_tokens is None


def test_파일_해시가_영수증과_다르면_멈춘다(world) -> None:
    rc = receipt(world["reg"], world["commit"], file_sha="0" * 64)
    with pytest.raises(G.EntryRejected) as exc:
        G.load_registration_for(rc, repo=world["repo"])
    assert exc.value.code == "REGISTRATION_FILE_HASH"


def test_본줄기의_조상이_아닌_커밋은_멈춘다(world) -> None:
    repo = world["repo"]
    git(repo, "checkout", "-q", "-b", "side")
    (repo / "note.txt").write_text("x", encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "곁가지")
    side = git(repo, "rev-parse", "HEAD")
    git(repo, "checkout", "-q", "main")
    with pytest.raises(G.EntryRejected) as exc:
        G.load_registration_for(receipt(world["reg"], side), repo=repo)
    assert exc.value.code == "RECEIPT_NOT_ANCESTOR"


def test_저장소에_없는_커밋은_조상이_아니다(world) -> None:
    assert G.is_ancestor(world["repo"], "0123456789abcdef0123456789abcdef01234567") is False
    assert G.is_ancestor(world["repo"], world["commit"]) is True


def test_리허설은_로컬_등록_파일을_읽는다(world) -> None:
    repo = world["repo"]
    r = registration(world["digest"], "rehearsal")
    path = "outputs/rehearsal_u/r1/registration/rehearsal-20261001-1.json"
    (repo / path).parent.mkdir(parents=True)
    (repo / path).write_bytes(dump_registration(r))
    rc = receipt(r, "f" * 40, path=path)
    assert G.load_registration_for(rc, repo=repo).generation.purpose == "rehearsal"


def test_커밋에서_읽을_경로가_저장소_밖이면_멈춘다(world) -> None:
    with pytest.raises(G.EntryRejected) as exc:
        G.blob_at(world["repo"], world["commit"], "../x.json")
    assert exc.value.code == "PATH_INVALID"


# ---------------------------------------------------------------- 3 닻

def test_닻은_영수증_커밋의_설정에서_읽는다(world) -> None:
    (world["repo"] / "configs/base.yaml").write_bytes(base_yaml("f" * 64))
    anchor = G.load_anchor(receipt(world["reg"], world["commit"]), repo=world["repo"])
    assert anchor.snapshot_digest == world["digest"], "작업 트리의 설정을 읽지 않는다"


def test_리허설의_닻은_본줄기의_설정이다(world) -> None:
    repo = world["repo"]
    (repo / "configs/base.yaml").write_bytes(base_yaml("e" * 64))
    git(repo, "commit", "-q", "-am", "닻을 바꾼다")
    r = registration(world["digest"], "rehearsal")
    rc = receipt(r, world["commit"], kind="rehearsal")
    assert G.load_anchor(rc, repo=repo).snapshot_digest == "e" * 64
    assert G.load_anchor(receipt(world["reg"], world["commit"]), repo=repo).snapshot_digest == world["digest"]


def test_등록의_동결본이_닻과_다르면_멈춘다(world) -> None:
    other = registration("d" * 64)
    with pytest.raises(G.EntryRejected) as exc:
        G.check_anchor(other, G.Anchor("synthetic_v1", world["digest"]))
    assert exc.value.code == "SNAPSHOT_NOT_ANCHOR"
    G.check_anchor(world["reg"], G.Anchor("synthetic_v1", world["digest"]))


@pytest.mark.parametrize("raw", [b"experiment: {}\n", b"fixed_before_main_runs:\n  snapshot_id: x\n",
                                 b"fixed_before_main_runs:\n  snapshot_id: x\n  snapshot_digest: abc\n", b"\xff"])
def test_닻이_없거나_꼴이_틀리면_멈춘다(raw: bytes) -> None:
    with pytest.raises(G.EntryRejected) as exc:
        G.read_anchor(raw)
    assert exc.value.code == "ANCHOR_MISSING"


def test_지금_저장소의_설정에서_닻을_읽는다() -> None:
    """본체의 `configs/base.yaml` 이 이 모듈이 읽는 꼴인지 — 값은 보지 않는다."""
    raw = (Path(__file__).resolve().parents[1] / "configs/base.yaml").read_bytes()
    anchor = G.read_anchor(raw)
    assert anchor.snapshot_id and len(anchor.snapshot_digest) == 64


# ---------------------------------------------------------------- 3′ 학습 쪽 프로세서 설정 (지시 20261001h 의 3)

def test_등록의_프로세서_값이_설정과_같으면_지난다(world) -> None:
    kwargs = G.load_processor_kwargs(receipt(world["reg"], world["commit"]), repo=world["repo"])
    assert kwargs == {"max_pixels": 921600}
    G.check_processor(world["reg"], kwargs)


def test_프로세서_인자는_영수증_커밋의_설정에서_읽는다(world) -> None:
    (world["repo"] / "configs/base.yaml").write_bytes(base_yaml(world["digest"], "{max_pixels: 1}"))
    assert G.load_processor_kwargs(receipt(world["reg"], world["commit"]), repo=world["repo"]) == {"max_pixels": 921600}


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"max_pixels": 1003520}, "PROCESSOR_MISMATCH"),
    ({"max_pixels": 921600.0}, "PROCESSOR_MISMATCH"),
    ({"max_pixels": 921600, "min_pixels": 3136}, "PROCESSOR_MISMATCH"),
    ({"max_pixels": 921600, "do_resize": False}, "PROCESSOR_KWARG_UNKNOWN"),
    ({}, "PROCESSOR_KWARGS_EMPTY"),
    ({"max_pixels": 921600, "min_pixels": None}, "PROCESSOR_MISMATCH"),
])
def test_등록과_설정의_프로세서_값이_다르면_멈춘다(world, kwargs: dict, code: str) -> None:
    """자료형까지 본다. 설정에 있는데 등록이 비었어도 다르다 — 설정 값이 `null` 이고 등록도 비어도 다르다.
    등록에 맞댈 칸이 없는 인자 · 빈 사전은 받지 않는다(검수 16번 M-3)."""
    with pytest.raises(G.EntryRejected) as exc:
        G.check_processor(world["reg"], kwargs)
    assert exc.value.code == code


@pytest.mark.parametrize("processor", [None, "[1, 2]"])
def test_설정에_프로세서_인자가_없으면_멈춘다(processor) -> None:
    with pytest.raises(G.EntryRejected) as exc:
        G.read_processor_kwargs(base_yaml("a" * 64, processor))
    assert exc.value.code == "PROCESSOR_KWARGS_MISSING"


# ---------------------------------------------------------------- 4 스냅샷

def test_스냅샷은_digest_가_닻과_같을_때만_쓴다(world) -> None:
    assert G.check_snapshot(world["snap"], G.Anchor("synthetic_v1", world["digest"])) is None
    with pytest.raises(G.EntryRejected) as exc:
        G.check_snapshot(world["snap"], G.Anchor("synthetic_v1", "c" * 64))
    assert exc.value.code == "SNAPSHOT_NOT_ANCHOR"


def test_손댄_스냅샷은_검증에서_멈춘다(world) -> None:
    p = world["snap"] / "manifest.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(G.EntryRejected) as exc:
        G.check_snapshot(world["snap"], G.Anchor("synthetic_v1", world["digest"]))
    assert exc.value.code == "SNAPSHOT_UNVERIFIED"


def test_흡수_상태를_돌려준다(tmp_path: Path) -> None:
    digest = make_snapshot(tmp_path / "s", absorption={"status": "provisional"})
    assert G.check_snapshot(tmp_path / "s", G.Anchor("synthetic_v1", digest)) == "provisional"


# ---------------------------------------------------------------- 5 모드와 목적

@pytest.mark.parametrize(("mode", "purpose"), [("strict", "main"), ("probe", "rehearsal"), ("probe", "frame_diag")])
def test_허용되는_조합(mode: str, purpose: str) -> None:
    G.check_mode(mode, purpose)


@pytest.mark.parametrize(("mode", "purpose", "code"), [
    ("probe", "main", "PROBE_MAIN"), ("strict", "rehearsal", "MODE_PURPOSE"),
    ("strict", "frame_diag", "MODE_PURPOSE"), ("loose", "main", "MODE_INVALID"),
])
def test_막는_조합(mode: str, purpose: str, code: str) -> None:
    with pytest.raises(G.EntryRejected) as exc:
        G.check_mode(mode, purpose)
    assert exc.value.code == code


def test_탐색_채점에_본실험_영수증이_오면_멈춘다(world) -> None:
    rc = receipt(world["reg"], world["commit"])
    with pytest.raises(G.EntryRejected) as exc:
        G.check_receipt_kind("probe", "rehearsal", rc)
    assert exc.value.code == "PROBE_MAIN"


def test_영수증의_종류가_경로의_목적과_다르면_멈춘다(world) -> None:
    r = registration(world["digest"], "frame_diag")
    with pytest.raises(G.EntryRejected) as exc:
        G.check_receipt_kind("probe", "rehearsal", receipt(r, world["commit"]))
    assert exc.value.code == "RECEIPT_KIND"


# ---------------------------------------------------------------- 6 기준 집합

def test_매니페스트는_네_열만_읽고_독에_닿지_않는다(world) -> None:
    meta = G.read_manifest_meta(world["snap"])
    assert meta.split_of[EVAL] == "eval" and meta.group_of[EVAL] == "g13"
    assert G.META_COLUMNS == ("image_id", "split", "sha256", "group_id")


@pytest.mark.parametrize(("purpose", "ids", "ok"), [
    ("rehearsal", [VAL, VAL2], True), ("frame_diag", [VAL], True), ("main", [EVAL], True),
    ("rehearsal", [VAL, EVAL], False), ("main", [VAL], False), ("frame_diag", [TRAIN], False),
    ("rehearsal", ["aihub71761:99"], False),
])
def test_목록의_id_는_그_목적의_분할이다(world, purpose: str, ids: list[str], ok: bool) -> None:
    meta = G.read_manifest_meta(world["snap"])
    if ok:
        G.check_list_split(ids, meta, purpose, where="생성 목록")
    else:
        with pytest.raises(G.EntryRejected) as exc:
            G.check_list_split(ids, meta, purpose, where="생성 목록")
        assert exc.value.code == "LIST_SPLIT"


def test_에코_목록은_본실험에서도_val_이다(world) -> None:
    meta = G.read_manifest_meta(world["snap"])
    G.check_list_split([VAL], meta, "main", where="에코 목록", echo=True)
    with pytest.raises(G.EntryRejected):
        G.check_list_split([EVAL], meta, "main", where="에코 목록", echo=True)


# ---------------------------------------------------------------- 7 이음새

def _root_commit() -> str:
    """본체 저장소의 루트 커밋 — 본줄기의 조상인 실제 커밋. 이것 하나만 읽는다."""
    here = Path(__file__).resolve().parent
    out = subprocess.run(["git", "-C", str(here), "rev-list", "--max-parents=0", G.MAIN_REF],
                         capture_output=True, check=True).stdout.decode("utf-8").split()
    return out[-1]


@pytest.mark.parametrize("kind", ["main", "frame_diag"])
def test_커밋된_영수증으로는_이음새를_쓸_수_없다(kind: str) -> None:
    r = registration("a" * 64, kind)
    with pytest.raises(G.EntryRejected) as exc:
        G.guard_seam(receipt(r, _root_commit()))
    assert exc.value.code == "SEAM_ON_COMMITTED_RECEIPT"


def test_합성_영수증과_리허설_영수증은_이음새를_쓴다() -> None:
    G.guard_seam(receipt(registration("a" * 64), "0123456789abcdef0123456789abcdef01234567"))
    G.guard_seam(receipt(registration("a" * 64, "rehearsal"), _root_commit()))


def test_정본_원장은_본체_체크아웃_아래에_있다() -> None:
    checkout = G.main_checkout()
    assert (checkout / ".git").is_dir(), "본체 체크아웃은 .git 디렉터리를 가진다(워크트리는 파일이다)"
    assert G.canonical_ledger(checkout) == checkout / "outputs" / "main_u" / "scoring_ledger.jsonl"
