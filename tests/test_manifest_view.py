"""평가 정답을 올리지 않는 판독기 둘을 고정한다.

**합성 스냅샷만 쓴다. 동결 자산을 읽지 않는다.**

덫: 평가 행의 라벨 열과 평가 이미지의 주석 행에 **값으로 만들면 터지는 독 값**을 심는다. 전체를 올리는 로더는
그 스냅샷에서 예외를 낸다(전제로 먼저 본다). 판독기의 정상 호출은 그 값을 값으로 만드는 단계에 한 번도
넘기지 않아야 한다 — 변환 함수에 들어온 값을 전부 기록해 독 값이 없음을 본다.
"""
from __future__ import annotations

import ast
import hashlib
import json as _json
import re
import subprocess as _subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from data import manifest_view as mv
from data.manifest_io import (ANNOTATION_COLUMNS, MANIFEST_COLUMNS, ManifestError,
                              SnapshotVerificationError, load_snapshot, read_annotations,
                              verify_snapshot, write_snapshot)
from evaluation.eval_list import set_digest
from evaluation.prereg_unified import dump_registration, read_receipt
from tests.test_prereg_unified import filled

POISON = "__POISON__"
LABEL_COLS = ("has_defect", "n_defects", "defect_types", "iso_codes", "src_labels_raw")


def _man_row(image_id, split="train", client="C1", has_defect=True, strata="ST|porosity"):
    tail = image_id.split(":")[-1]
    return {c: "" for c in MANIFEST_COLUMNS} | {
        "image_id": image_id, "source": "aihub71761", "rel_path": f"x/{tail}.jpg",
        "sha256": tail.zfill(8), "width_px": 1280, "height_px": 720,
        "modality": "RT", "material": "ST", "has_defect": has_defect,
        "n_defects": 1 if has_defect else 0, "defect_types": "porosity" if has_defect else "",
        "iso_codes": "2011" if has_defect else "", "src_labels_raw": "기공" if has_defect else "",
        "label_type": "polygon", "has_localization": True, "phash_hex": "0" * 16,
        "group_id": f"g{tail}", "group_size": 1, "strata_key": strata, "split": split,
        "client": client, "eval_subset": "", "thickness_mm": None, "thickness_source": "unavailable",
        "px_per_mm": None, "scale_source": "unavailable", "quality_level": "",
        "ingest_version": "t", "label_map_version": 1, "notes": "",
    }


def _ann_row(image_id, idx=0, box=(10, 10, 20, 20)):
    return {c: "" for c in ANNOTATION_COLUMNS} | {
        "ann_id": f"{image_id}#{idx}", "image_id": image_id, "src_label_raw": "기공",
        "defect_type": "porosity", "iso_code": "2011", "polygon_json": "[]",
        "bbox_x1_px": box[0], "bbox_y1_px": box[1], "bbox_x2_px": box[2], "bbox_y2_px": box[3],
        "area_px": 1.5, "major_axis_px": 1.0, "minor_axis_px": 1.0, "equiv_diameter_px": 1.0,
        "major_axis_mm": None, "equiv_diameter_mm": None, "geom_valid": True, "geom_flags": "",
    }


TRAIN, VAL, EVAL_D, EVAL_N = "aihub71761:11", "aihub71761:12", "aihub71761:13", "aihub71761:14"


def _snapshot(root: Path, *, poison: bool, absorb: bool = False) -> Path:
    man = [_man_row(TRAIN), _man_row(VAL, split="val", client="C2", strata="AL|crack"),
           _man_row(EVAL_D, split="eval", client=""), _man_row(EVAL_N, split="eval", client="", has_defect=False,
                                                               strata="ST|__normal__")]
    ann = [_ann_row(TRAIN), _ann_row(VAL), _ann_row(EVAL_D, 0), _ann_row(EVAL_D, 1)]
    if poison:
        for r in man:
            if r["split"] == "eval":
                r.update({c: POISON for c in LABEL_COLS} | {"strata_key": POISON})
        for r in ann:
            if r["image_id"] == EVAL_D:
                r.update({"bbox_x1_px": POISON, "geom_valid": POISON, "defect_type": POISON, "area_px": POISON})
    extra = {}
    if absorb:
        from data.manifest_io import ABSORB_DEFECT_COLUMNS, ABSORB_IMAGE_COLUMNS, ABSORB_REGION_COLUMNS
        extra = {"absorb_image": pd.DataFrame(columns=list(ABSORB_IMAGE_COLUMNS)),
                 "absorb_defect": pd.DataFrame(columns=list(ABSORB_DEFECT_COLUMNS)),
                 "absorb_region": pd.DataFrame(columns=list(ABSORB_REGION_COLUMNS))}
    write_snapshot(root, pd.DataFrame(man), pd.DataFrame(ann),
                   {"snapshot_id": "synthetic_v1", "capabilities": {"verdict_mode": "clause_only", "localization": True}},
                   **extra)
    return root


@pytest.fixture
def poisoned(tmp_path):
    return _snapshot(tmp_path / "poisoned", poison=True)


@pytest.fixture
def clean(tmp_path):
    return _snapshot(tmp_path / "clean", poison=False)


@pytest.fixture
def opens(monkeypatch):
    """이 모듈이 연 파일 이름을 순서대로 기록한다."""
    seen: list[str] = []
    real = mv._open_csv
    monkeypatch.setattr(mv, "_open_csv", lambda p: (seen.append(Path(p).name), real(p))[1])
    return seen


@pytest.fixture
def converted(monkeypatch):
    """값으로 만드는 단계에 들어온 (열, 값) 을 전부 기록한다."""
    seen: list[tuple[str, str]] = []
    real = mv._convert
    monkeypatch.setattr(mv, "_convert", lambda c, vals: (seen.extend((c, v) for v in vals), real(c, vals))[1])
    return seen


# --- 전제: 독 값은 살아 있다 ----------------------------------------------------------

def test_전제_독이_든_스냅샷은_검증을_지나고_전체_로더는_터진다(poisoned):
    verify_snapshot(poisoned)                              # 바이트는 잠금과 맞는다
    # 로더의 변환은 정수 열에서 `ValueError`, 불 열에서 `ManifestError`(그 하위)를 낸다.
    with pytest.raises(ValueError):
        load_snapshot(poisoned)                            # 평가 행의 라벨 열을 값으로 만들다 터진다
    with pytest.raises(ValueError):
        read_annotations(poisoned / "annotations.csv")     # 평가 이미지의 주석 행도 같다
    assert issubclass(ManifestError, ValueError)


# --- 덫 ------------------------------------------------------------------------------

def test_덫_정상_호출은_독_값을_값으로_만들지_않는다(poisoned, converted):
    tv = mv.read_manifest_columns(poisoned, list(MANIFEST_COLUMNS), split_filter={"train", "val"})
    ev = mv.read_manifest_columns(poisoned, ["image_id", "split", "sha256", "group_id"],
                                  split_filter={"train", "val", "eval"})
    av = mv.read_annotations_view(poisoned, {TRAIN, VAL})
    assert converted, "변환 기록이 비었다 — 덫이 아무것도 보지 않는다"
    touched = sorted({c for c, v in converted if v == POISON})
    assert not touched, f"독 값이 값으로 만들어졌다: {touched}"
    assert set(tv["image_id"]) == {TRAIN, VAL} and set(tv["split"]) == {"train", "val"}
    assert set(ev["image_id"]) == {TRAIN, VAL, EVAL_D, EVAL_N} and list(ev.columns) == ["image_id", "split", "sha256", "group_id"]
    assert set(av["image_id"]) == {TRAIN, VAL} and len(av) == 2
    # 변환을 거치지 않고 돌려주는 길이 있어도 잡히게 **반환 프레임도** 본다.
    for name, frame in (("열 판독(train·val)", tv), ("열 판독(평가 넷)", ev), ("주석 뷰", av)):
        assert not (frame.astype("string") == POISON).fillna(False).any().any(), f"{name} 이 독 값을 돌려줬다"


# --- 평가 행의 열 -----------------------------------------------------------------------

@pytest.mark.parametrize("col", ["strata_key", "has_defect", "defect_types", "iso_codes", "client", "rel_path"])
@pytest.mark.parametrize("splits", [{"eval"}, {"train", "val", "eval"}], ids=["eval만", "셋다"])
def test_평가_행의_금지_열을_요구하면_아무것도_열지_않고_거부한다(poisoned, opens, monkeypatch, col, splits):
    verified = []
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: verified.append(r))
    with pytest.raises(mv.ManifestViewError, match="평가 행에서 읽을 수 없는 열"):
        mv.read_manifest_columns(poisoned, ["image_id", col], split_filter=splits)
    assert opens == [] and verified == [], "거부하기 전에 무언가를 읽었다"


def test_층은_train_val_행에서만_읽힌다(clean):
    df = mv.read_manifest_columns(clean, ["image_id", "strata_key"], split_filter={"train", "val"})
    assert dict(zip(df["image_id"], df["strata_key"])) == {TRAIN: "ST|porosity", VAL: "AL|crack"}


# --- 주석 뷰 ---------------------------------------------------------------------------

def test_평가_id_가_든_집합은_주석을_열기_전에_거부한다(poisoned, opens):
    with pytest.raises(mv.ManifestViewError, match="평가 id"):
        mv.read_annotations_view(poisoned, {TRAIN, EVAL_D})
    assert "annotations.csv" not in opens, "평가 id 를 거부하기 전에 주석 파일을 열었다"


def test_매니페스트에_없는_id_는_거부한다(poisoned, opens):
    with pytest.raises(mv.ManifestViewError, match="없는 id"):
        mv.read_annotations_view(poisoned, {TRAIN, "aihub71761:999"})
    assert "annotations.csv" not in opens


def test_준_집합의_행만_남긴다(clean):
    df = mv.read_annotations_view(clean, {VAL})
    assert list(df["ann_id"]) == [f"{VAL}#0"] and list(df.columns) == list(ANNOTATION_COLUMNS)
    assert mv.read_annotations_view(clean, set()).empty


# --- 검증이 먼저 -----------------------------------------------------------------------

@pytest.mark.parametrize("member", ["manifest.csv", "annotations.csv"])
@pytest.mark.parametrize("call", ["columns", "annotations"])
def test_검증이_실패하면_아무것도_읽지_않는다(clean, opens, member, call):
    p = clean / member
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SnapshotVerificationError):
        if call == "columns":
            mv.read_manifest_columns(clean, ["image_id"], split_filter={"train"})
        else:
            mv.read_annotations_view(clean, {TRAIN})
    assert opens == [], f"검증이 실패했는데 {opens} 를 열었다"


def test_검증이_파일을_열기_전에_돈다(clean, monkeypatch):
    order: list[str] = []
    real_v, real_o = mv.verify_snapshot, mv._open_csv
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: (order.append("verify"), real_v(r))[1])
    monkeypatch.setattr(mv, "_open_csv", lambda p: (order.append(Path(p).name), real_o(p))[1])
    mv.read_manifest_columns(clean, ["image_id"], split_filter={"train"})
    mv.read_annotations_view(clean, {TRAIN})
    assert order == ["verify", "manifest.csv", "verify", "manifest.csv", "annotations.csv"]


def test_반환한_지문이_검증한_지문이다(clean):
    d = verify_snapshot(clean)
    assert mv.read_manifest_columns(clean, ["image_id"], split_filter={"val"}).attrs["snapshot_digest"] == d
    assert mv.read_annotations_view(clean, {TRAIN}).attrs["snapshot_digest"] == d


# --- 정상 경로는 로더와 같은 값을 낸다 ---------------------------------------------------

def test_로더와_같은_값과_자료형을_낸다(clean):
    snap = load_snapshot(clean)
    tv_ids = set(snap.manifest.loc[snap.manifest["split"] != "eval", "image_id"])
    got = mv.read_manifest_columns(clean, list(MANIFEST_COLUMNS), split_filter={"train", "val"})
    want = snap.manifest[snap.manifest["split"] != "eval"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got.sort_values("image_id").reset_index(drop=True),
                                  want.sort_values("image_id").reset_index(drop=True))
    got_a = mv.read_annotations_view(clean, tv_ids)
    want_a = snap.annotations[snap.annotations["image_id"].isin(tv_ids)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got_a.sort_values("ann_id").reset_index(drop=True),
                                  want_a.sort_values("ann_id").reset_index(drop=True))


# --- 머리행과 인자 -----------------------------------------------------------------------

def _relock(root: Path) -> None:
    """합성 스냅샷의 잠금을 지금 바이트로 다시 쓴다. 형식은 `write_snapshot` 과 같다."""
    lock = root / "SNAPSHOT.sha256"
    names = [ln.split(maxsplit=1)[1] for ln in lock.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")]
    lines = [f"{hashlib.sha256((root / n).read_bytes()).hexdigest()}  {n}" for n in names]
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    lock.write_text("\n".join(lines) + "\n" + f"# snapshot_digest {digest}\n", encoding="utf-8", newline="\n")


def test_머리행이_계약과_다르면_읽지_않는다(clean):
    p = clean / "manifest.csv"
    text = p.read_text(encoding="utf-8").split("\n")
    head = text[0].split(",")
    head[0], head[1] = head[1], head[0]
    p.write_text("\n".join([",".join(head), *text[1:]]), encoding="utf-8", newline="\n")
    _relock(clean)
    verify_snapshot(clean)                                 # 잠금은 새 바이트와 맞는다
    with pytest.raises(mv.ManifestViewError, match="머리행"):
        mv.read_manifest_columns(clean, ["image_id"], split_filter={"train"})


@pytest.mark.parametrize("columns, splits, msg", [
    (["image_id"], "train", "문자열 하나가 아니다"),
    (["image_id"], {"test"}, "분할은"),
    (["image_id"], set(), "분할은"),
    ([], {"train"}, "비어 있지 않은"),
    ("image_id", {"train"}, "비어 있지 않은"),
    (["image_id", "nope"], {"train"}, "없는 열"),
    (["image_id", "image_id"], {"train"}, "두 번"),
])
def test_잘못된_인자는_아무것도_열지_않고_거부한다(clean, opens, columns, splits, msg):
    with pytest.raises(mv.ManifestViewError, match=msg):
        mv.read_manifest_columns(clean, columns, split_filter=splits)
    assert opens == []


def test_ids_가_문자열_하나면_거부한다(clean, opens):
    with pytest.raises(mv.ManifestViewError, match="문자열 하나가 아니다"):
        mv.read_annotations_view(clean, TRAIN)
    assert opens == []


# --- 긴 칸 ------------------------------------------------------------------------------

def test_파이썬_csv_기본_상한보다_긴_칸을_읽고_상한을_되돌린다(tmp_path):
    """`csv` 의 기본 한 칸 상한은 131,072 자다. 긴 `polygon_json` 한 칸에서 판독기가 멈추면 안 된다."""
    import csv
    root = tmp_path / "long"
    long_poly = "[" + ",".join(f"[{i},{i}]" for i in range(20_000)) + "]"
    assert len(long_poly) > 131_072
    man = [_man_row(TRAIN), _man_row(EVAL_D, split="eval", client="")]
    ann = [_ann_row(TRAIN) | {"polygon_json": long_poly}, _ann_row(EVAL_D)]
    write_snapshot(root, pd.DataFrame(man), pd.DataFrame(ann),
                   {"snapshot_id": "s", "capabilities": {"verdict_mode": "clause_only", "localization": True}})
    before = csv.field_size_limit()
    df = mv.read_annotations_view(root, {TRAIN})
    assert df.loc[0, "polygon_json"] == long_poly
    assert csv.field_size_limit() == before, "전역 상한을 되돌리지 않았다"


# --- 흡수 곁파일 -------------------------------------------------------------------------

def test_흡수_곁파일을_열지_않는다(tmp_path, opens):
    root = _snapshot(tmp_path / "abs", poison=True, absorb=True)
    assert (root / "absorb_image.csv").exists()
    mv.read_manifest_columns(root, ["image_id", "split"], split_filter={"train", "val", "eval"})
    mv.read_annotations_view(root, {TRAIN})
    assert set(opens) == {"manifest.csv", "annotations.csv"}, opens


def test_모듈이_흡수_모듈과_곁파일_이름을_가져다_쓰지_않는다():
    tree = ast.parse(Path(mv.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    bad = sorted(n for n in names if "absorb" in n.lower() or n in {"load_snapshot", "read_annotations",
                                                                     "read_manifest", "join_defects"})
    assert not bad, bad


#: 판독기 모듈이 쓰면 안 되는 이름. 파일을 직접 여는 길과 이름을 문자열로 푸는 동적 접근이다.
#: `open` 은 `_open_csv` 안의 한 곳만 허용한다 — 이 모듈이 여는 파일은 전부 거기를 지난다.
DIRECT_READ = frozenset({"read_csv", "read_table", "read_fwf", "read_excel", "read_parquet", "read_json",
                         "read_text", "read_bytes", "open", "loadtxt", "genfromtxt"})
DYNAMIC = frozenset({"getattr", "importlib", "import_module", "__import__", "eval", "exec"})
OPEN_SITES = frozenset({"_open_csv", "_read_small", "_append_call_log"})


def test_모듈이_직접_읽기와_동적_접근을_쓰지_않는다():
    """구문 트리의 이름을 본다. **막는 것은 실수다** — 문자열을 조립해 이름을 푸는 고의의 우회는 코드 검수의 일이다.

    `open` 은 파일을 여는 세 자리(`OPEN_SITES`)에서만 허용한다.
    """
    tree = ast.parse(Path(mv.__file__).read_text(encoding="utf-8"))
    # 파일을 여는 자리는 셋이다 — CSV(`_open_csv`) · 영수증(`_read_small`) · 호출 기록(`_append_call_log`)
    sites = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in OPEN_SITES]
    assert {n.name for n in sites} == OPEN_SITES, "파일을 여는 자리가 목록과 다르다"
    inside = {id(x) for n in sites for x in ast.walk(n)}
    hits = []
    for node in ast.walk(tree):
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            hits += [m for m in mods if m.split(".")[0] in DYNAMIC]
        if name is None:
            continue
        if name == "open" and id(node) in inside:
            continue
        if name in DIRECT_READ or name in DYNAMIC:
            hits.append(f"{name}@{node.lineno}")
    assert not hits, hits


# ======================================================================================
# 엄격 경로의 평가 정답 뷰
# ======================================================================================
#
# 합성 저장소(git)에 등록과 닻을 적은 설정을 커밋하고 그 커밋을 가리키는 영수증을 만든다. 저장소·호출 기록 자리는
# 모듈의 상수를 바꿔 합성 쪽을 가리키게 한다 — 함수 서명에는 그런 이음새가 없다. 닻은 영수증 커밋의 설정에서 읽힌다.

REG_DIR = "configs/registration"


def _g(repo: Path, *args: str) -> str:
    r = _subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                        cwd=repo, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _base_yaml(digest: str) -> str:
    return f"fixed_before_main_runs:\n  snapshot_digest: {digest}\n"


def _reg(digest: str, *, purpose="main", split="eval", generation_only=False):
    r = filled(generation_only=generation_only)
    r.generation.purpose, r.generation.list_split, r.generation.snapshot_digest = purpose, split, digest
    return r


@pytest.fixture
def strict(tmp_path, monkeypatch):
    clean_root = _snapshot(tmp_path / "clean", poison=False)
    digest = verify_snapshot(clean_root)
    regs = {"main": _reg(digest), "rehearsal": _reg(digest, purpose="rehearsal", split="val"),
            "gen_only": _reg(digest, generation_only=True), "other_anchor": _reg("a" * 64)}
    repo = tmp_path / "repo"
    (repo / REG_DIR).mkdir(parents=True)
    _g(repo, "init", "-q")
    _g(repo, "symbolic-ref", "HEAD", "refs/heads/main")
    reg_bytes = {name: dump_registration(r) for name, r in regs.items()}
    for name, raw in reg_bytes.items():
        (repo / REG_DIR / f"{name}.json").write_bytes(raw)
    base = repo / "configs" / "base.yaml"
    base.write_text(_base_yaml(digest), encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "등록")
    c_reg = _g(repo, "rev-parse", "HEAD")
    for name, raw in reg_bytes.items():          # 영수증이 적는 해시는 커밋에 든 등록 파일의 바이트로 낸다
        shown = _subprocess.run(["git", "show", f"{c_reg}:{REG_DIR}/{name}.json"], cwd=repo, capture_output=True)
        assert shown.returncode == 0 and shown.stdout == raw, f"커밋된 등록 바이트가 쓴 바이트와 다르다: {name}"
    (repo / "after.txt").write_text("영수증 커밋 자리", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "영수증")
    base.write_text(_base_yaml("a" * 64), encoding="utf-8")            # 본줄기 안에서 닻만 다른 커밋
    _g(repo, "commit", "-q", "-am", "닻이 다른 설정")
    c_other_anchor = _g(repo, "rev-parse", "HEAD")
    _g(repo, "rm", "-q", "configs/base.yaml")                          # 본줄기 안에서 설정이 없는 커밋
    _g(repo, "commit", "-q", "-m", "설정 없음")
    c_no_config = _g(repo, "rev-parse", "HEAD")
    base.write_text(_base_yaml(digest), encoding="utf-8")              # 본줄기 끝의 작업 트리는 맞는 닻이다
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "설정 되돌림")
    _g(repo, "checkout", "-q", "-b", "side")
    (repo / "side.txt").write_text("곁가지", encoding="utf-8")
    _g(repo, "add", "-A")
    _g(repo, "commit", "-q", "-m", "곁가지")
    c_side = _g(repo, "rev-parse", "HEAD")
    _g(repo, "checkout", "-q", "main")
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(mv, "_GIT_ROOT", repo)
    monkeypatch.setattr(mv, "CALL_LOG", log)

    def receipt(name="main", *, kind="main", commit=None, path=None, gen=None, file="receipt.json", raw=None):
        r = regs[name]
        data = {"generation_sha256": gen or r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
                "registered_at": "2026-10-01T12:00:00+09:00", "main_commit": commit or c_reg, "kind": kind,
                "registration_path": path or f"{REG_DIR}/{name}.json",
                "registration_file_sha256": hashlib.sha256(reg_bytes[name]).hexdigest()}
        p = tmp_path / file
        p.write_bytes(raw if raw is not None else _json.dumps(data).encode("utf-8"))
        return p

    def lines():
        return [_json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []

    return SimpleNamespace(root=clean_root, digest=digest, repo=repo, c_reg=c_reg, c_side=c_side,
                           c_other_anchor=c_other_anchor, c_no_config=c_no_config,
                           receipt=receipt, lines=lines, base=base, tmp=tmp_path)


def test_엄격_정상_경로는_평가_주석을_돌려주고_기록을_남긴다(strict):
    rp = strict.receipt()
    df = mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=rp)
    assert list(df["ann_id"]) == [f"{EVAL_D}#0", f"{EVAL_D}#1"] and list(df.columns) == list(ANNOTATION_COLUMNS)
    assert df.attrs["snapshot_digest"] == strict.digest
    (line,) = strict.lines()
    assert line["outcome"] == "allowed" and line["reason"] == ""
    assert line["receipt_sha256"] == hashlib.sha256(rp.read_bytes()).hexdigest()
    assert line["ids_set_sha256"] == set_digest({EVAL_D}) and line["n_ids"] == 1
    assert line["main_commit"] == strict.c_reg and line["receipt_kind"] == "main"
    assert line["snapshot_digest"] == strict.digest
    assert line["caller"].endswith(":test_엄격_정상_경로는_평가_주석을_돌려주고_기록을_남긴다")
    assert line["at"].endswith("+00:00") and isinstance(line["pid"], int)


def test_엄격_허용_호출은_기록을_쓴_뒤에_주석_파일을_연다(strict, monkeypatch):
    order: list[str] = []
    real_log, real_open = mv._append_call_log, mv._open_csv
    monkeypatch.setattr(mv, "_append_call_log", lambda e: (order.append("기록"), real_log(e))[1])
    monkeypatch.setattr(mv, "_open_csv", lambda p: (order.append(Path(p).name), real_open(p))[1])
    mv.read_annotations_view_eval(strict.root, {EVAL_D, EVAL_N}, receipt=strict.receipt())
    assert order == ["manifest.csv", "기록", "annotations.csv"], order


def test_엄격_기록을_쓰지_못하면_읽지_않는다(strict, opens, monkeypatch):
    def boom(entry):
        raise OSError("디스크")
    monkeypatch.setattr(mv, "_append_call_log", boom)
    with pytest.raises(mv.EvalViewRefused, match="호출 기록"):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    assert "annotations.csv" not in opens


def _refusal_cases():
    return [
        ("리허설_영수증", lambda s: dict(receipt=s.receipt("rehearsal", kind="rehearsal")), "종류가 'rehearsal'"),
        ("진단_영수증", lambda s: dict(receipt=s.receipt(kind="frame_diag")), "종류가 'frame_diag'"),
        ("종류_없는_영수증", lambda s: dict(receipt=s.receipt(raw=b'{"main_commit": "x"}')), "종류가 None"),
        ("읽은_영수증_객체", lambda s: dict(receipt=read_receipt(s.receipt())), "파일 경로"),
        ("영수증_파일_없음", lambda s: dict(receipt=s.tmp / "없다.json"), "영수증 파일을 읽지 못했다"),
        ("영수증_JSON_아님", lambda s: dict(receipt=s.receipt(raw=b"not json")), "영수증을 읽지 못했다"),
        ("영수증_JSON_배열", lambda s: dict(receipt=s.receipt(raw=b'["main"]')), "JSON 객체가 아니다"),
        ("영수증_키_빠짐", lambda s: dict(receipt=s.receipt(raw=b'{"kind": "main"}')), "이 없다"),
        ("영수증_꼴_틀림", lambda s: dict(receipt=s.receipt(commit="XYZ")), "영수증의 꼴이 틀렸다"),
        ("id_비었음", lambda s: dict(receipt=s.receipt(), ids=set()), "ids 가 비었다"),
        ("커밋의_닻이_다름", lambda s: dict(receipt=s.receipt(commit=s.c_other_anchor)), "영수증 커밋의 닻과 다르다"),
        ("커밋에_설정_없음", lambda s: dict(receipt=s.receipt(commit=s.c_no_config)), "설정(configs/base.yaml)을 읽지"),
        ("본줄기_조상_아님", lambda s: dict(receipt=s.receipt(commit=s.c_side)), "조상이 아니다"),
        ("모르는_커밋", lambda s: dict(receipt=s.receipt(commit="0" * 40)), "확인하지 못했다"),
        ("등록_파일_없음", lambda s: dict(receipt=s.receipt(path=f"{REG_DIR}/없다.json")), "등록 파일을 읽지 못했다"),
        ("영수증_지문_불일치", lambda s: dict(receipt=s.receipt(gen="b" * 64)), "등록이 영수증과 맞지 않거나"),
        ("채점_쪽_미등록", lambda s: dict(receipt=s.receipt("gen_only")), "등록이 영수증과 맞지 않거나"),
        ("영수증_main_등록_리허설", lambda s: dict(receipt=s.receipt("rehearsal")), "등록이 영수증과 맞지 않거나"),
        ("등록_지문이_닻과_다름", lambda s: dict(receipt=s.receipt("other_anchor")), "등록의 동결본 지문"),
        ("학습_id_섞임", lambda s: dict(receipt=s.receipt(), ids={EVAL_D, TRAIN}), "평가가 아닌 id"),
        ("검증_id_섞임", lambda s: dict(receipt=s.receipt(), ids={EVAL_D, VAL}), "평가가 아닌 id"),
        ("모르는_id", lambda s: dict(receipt=s.receipt(), ids={EVAL_D, "aihub71761:999"}), "없는 id"),
        ("id_문자열_하나", lambda s: dict(receipt=s.receipt(), ids=EVAL_D), "문자열 하나"),
    ]


@pytest.mark.parametrize("make, msg", [(m, msg) for _n, m, msg in _refusal_cases()],
                         ids=[n for n, _m, _msg in _refusal_cases()])
def test_엄격_조건을_어기면_주석_파일을_열기_전에_거부하고_기록한다(strict, opens, make, msg):
    kw = {"ids": {EVAL_D}} | make(strict)
    with pytest.raises(mv.EvalViewRefused, match=re.escape(msg)):
        mv.read_annotations_view_eval(strict.root, kw["ids"], receipt=kw["receipt"])
    assert "annotations.csv" not in opens, "거부하기 전에 주석 파일을 열었다"
    (line,) = strict.lines()
    assert line["outcome"] == "refused" and msg in line["reason"], line


def test_엄격_검증한_지문이_닻과_다르면_거부한다(strict, opens, tmp_path):
    other = _snapshot(tmp_path / "other", poison=True)        # 스스로는 잠금과 맞지만 닻과 다른 판
    with pytest.raises(mv.EvalViewRefused, match="검증한 스냅샷의 지문"):
        mv.read_annotations_view_eval(other, {EVAL_D}, receipt=strict.receipt())
    assert opens == [], "닻이 다른 판의 파일을 열었다"
    assert strict.lines()[-1]["snapshot_digest"] == verify_snapshot(other)


def test_엄격_검증이_실패하면_아무것도_읽지_않고_기록한다(strict, opens):
    p = strict.root / "annotations.csv"
    p.write_bytes(p.read_bytes() + b"\n")
    with pytest.raises(SnapshotVerificationError):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    assert opens == []
    assert strict.lines()[-1]["outcome"] == "refused"


def test_엄격_평가_쪽_대조가_목적을_건너뛰어도_목적과_분할을_본다(strict, opens, monkeypatch):
    """`require_receipt` 는 목적이 비면 대조를 건너뛴다. 그 대조가 무엇을 돌려주든 이 함수가 목적을 다시 본다."""
    import evaluation.prereg_unified as P
    fake = _reg(strict.digest, purpose="rehearsal", split="val")
    monkeypatch.setattr(P, "registration_for_receipt", lambda raw, rec, side: (fake, "x"))
    monkeypatch.setattr(P, "require_receipt", lambda reg, rec, side="generation": "x")
    with pytest.raises(mv.EvalViewRefused, match="등록의 목적이 'rehearsal'"):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    fake.generation.purpose = "main"
    with pytest.raises(mv.EvalViewRefused, match="기준 분할이 'val'"):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    assert "annotations.csv" not in opens


def test_엄격_닻은_영수증_커밋에서_읽고_작업_트리의_설정을_읽지_않는다(strict, opens, monkeypatch):
    """체크아웃의 한 줄을 고쳐도 판정이 바뀌지 않는다 — 닻은 영수증 커밋의 설정이다."""
    assert not hasattr(mv, "_BASE_CONFIG"), "작업 트리 설정을 가리키는 상수가 남아 있다"
    small: list[str] = []
    real_small = mv._read_small
    monkeypatch.setattr(mv, "_read_small", lambda p: (small.append(Path(p).name), real_small(p))[1])
    # 작업 트리의 닻이 틀려도 영수증 커밋의 닻으로 지난다
    strict.base.write_text(_base_yaml("c" * 64), encoding="utf-8")
    df = mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    assert df.attrs["snapshot_digest"] == strict.digest
    # 작업 트리의 닻이 맞아도 영수증 커밋의 닻이 다르면 멈춘다
    strict.base.write_text(_base_yaml(strict.digest), encoding="utf-8")
    opens.clear()
    with pytest.raises(mv.EvalViewRefused, match="영수증 커밋의 닻과 다르다"):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=strict.receipt(commit=strict.c_other_anchor))
    assert "annotations.csv" not in opens
    assert "base.yaml" not in small, "작업 트리의 설정 파일을 열었다"


def test_엄격_빈_ids_는_영수증도_열지_않고_거부한다(strict, opens, monkeypatch):
    monkeypatch.setattr(mv, "_read_small", lambda p: pytest.fail(f"파일을 읽었다: {Path(p).name}"))
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: pytest.fail("검증을 불렀다"))
    with pytest.raises(mv.EvalViewRefused, match="ids 가 비었다"):
        mv.read_annotations_view_eval(strict.root, [], receipt=strict.receipt())
    assert opens == []
    (line,) = strict.lines()
    assert line["outcome"] == "refused" and line["n_ids"] == 0 and line["receipt_sha256"] is None


def test_엄격_진단_영수증은_평가_쪽이_종류를_늘려도_이_함수가_거부한다(strict, opens, converted, monkeypatch):
    """평가 쪽 영수증이 `frame_diag` 를 받게 되어도 덫은 이 함수의 조건 검사에서 선다."""
    import evaluation.prereg_unified as P
    monkeypatch.setattr(P, "RECEIPT_KINDS", ("main", "rehearsal", "frame_diag"))
    rp = strict.receipt(kind="frame_diag")
    assert read_receipt(rp).kind == "frame_diag", "전제: 평가 쪽 파싱은 이제 진단 영수증을 받는다"
    verified: list = []
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: verified.append(r))
    monkeypatch.setattr(mv, "_git", lambda *a: pytest.fail(f"저장소를 물었다: {a}"))
    with pytest.raises(mv.EvalViewRefused, match="종류가 'frame_diag'"):
        mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=rp)
    assert opens == [] and converted == [] and verified == []
    assert strict.lines()[-1]["receipt_kind"] == "frame_diag"


def test_엄격_영수증은_한_번_읽은_바이트로_해시와_파싱을_함께_한다(strict, monkeypatch):
    """읽은 직후 파일이 바뀌어도, 기록한 해시와 대조한 내용은 같은 바이트에서 나온다."""
    rp = strict.receipt()
    first = rp.read_bytes()
    swapped = strict.receipt("other_anchor", file="swap.json").read_bytes()
    reads: list[str] = []
    real_small = mv._read_small

    def read_then_swap(p):
        out = real_small(p)
        if Path(p) == rp:
            reads.append("영수증")
            rp.write_bytes(swapped)                  # 읽은 뒤에 파일이 바뀐다
        return out
    monkeypatch.setattr(mv, "_read_small", read_then_swap)
    df = mv.read_annotations_view_eval(strict.root, {EVAL_D}, receipt=rp)
    assert df.attrs["snapshot_digest"] == strict.digest
    assert reads == ["영수증"], "영수증 파일을 두 번 읽었다"
    assert strict.lines()[-1]["receipt_sha256"] == hashlib.sha256(first).hexdigest()


@pytest.mark.parametrize("name, kind, device", [("main", "main", None), ("rehearsal", "rehearsal", None),
                                                ("gen_only", "main", "cuda:0")])
def test_영수증_파싱은_평가_쪽_read_receipt_와_같은_객체를_낸다(strict, name, kind, device):
    """평가 쪽 함수가 경로만 받아 같은 바이트로 파싱하려고 공개 클래스로 만든다. 둘이 갈리면 여기서 떨어진다."""
    rp = strict.receipt(name, kind=kind)
    if device is not None:
        data = _json.loads(rp.read_bytes())
        rp.write_bytes(_json.dumps(data | {"device": device}).encode("utf-8"))
    assert mv._receipt(mv._receipt_data(rp.read_bytes())) == read_receipt(rp)


@pytest.mark.parametrize("kind", ["rehearsal", "frame_diag"])
def test_엄격_덫_리허설과_진단_영수증으로는_아무것도_읽지_않는다(strict, opens, converted, monkeypatch, tmp_path, kind):
    """리허설·진단 목적의 합성 영수증으로 부르면 예외다. 독이 든 스냅샷을 줘도 검증조차 부르지 않는다."""
    poisoned_root = _snapshot(tmp_path / "poisoned_trap", poison=True)
    verified: list = []
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: verified.append(r))
    gits: list = []
    monkeypatch.setattr(mv, "_git", lambda *a: gits.append(a))
    name = "rehearsal" if kind == "rehearsal" else "main"
    with pytest.raises(mv.EvalViewRefused, match="영수증의 종류가"):
        mv.read_annotations_view_eval(poisoned_root, {EVAL_D}, receipt=strict.receipt(name, kind=kind))
    assert opens == [] and converted == [] and verified == [], "리허설·진단 영수증으로 읽기가 시작됐다"
    assert gits == [], "종류 검사보다 뒤의 검사가 거부했다"
    assert strict.lines()[-1]["outcome"] == "refused"


# --- 엄격 경로의 평가 이미지 경로 판독기 ----------------------------------------------------
#
# 본실험 export 가 평가 이미지를 여는 길이다. 조건은 정답 뷰와 같은 검사를 지난다 — 같은 거부 사례 표를 다시 쓴다.

def _watch_scan(monkeypatch) -> list[tuple[str, ...]]:
    asked: list[tuple[str, ...]] = []
    real = mv._scan_manifest
    monkeypatch.setattr(mv, "_scan_manifest",
                        lambda root, cols, splits: (asked.append(tuple(cols)), real(root, cols, splits))[1])
    return asked


def test_경로_정상_경로는_평가_장의_경로만_내고_기록을_남긴다(strict, opens, monkeypatch):
    asked = _watch_scan(monkeypatch)
    df = mv.read_image_paths_eval(strict.root, {EVAL_D, EVAL_N}, receipt=strict.receipt())
    assert list(df.columns) == list(mv.EVAL_PATH_COLUMNS) == ["image_id", "rel_path"]
    assert dict(zip(df["image_id"], df["rel_path"])) == {EVAL_D: "x/13.jpg", EVAL_N: "x/14.jpg"}
    assert df.attrs["snapshot_digest"] == strict.digest
    assert "annotations.csv" not in opens, "경로 판독기가 주석 파일을 열었다"
    assert ("image_id", "rel_path") in asked
    (line,) = strict.lines()
    assert line["view"] == "image_paths" and line["outcome"] == "allowed" and line["n_ids"] == 2
    assert line["caller"].endswith(":test_경로_정상_경로는_평가_장의_경로만_내고_기록을_남긴다")


def test_경로_한_장만_주면_그_장의_경로만_낸다(strict):
    df = mv.read_image_paths_eval(strict.root, {EVAL_N}, receipt=strict.receipt())
    assert list(df["image_id"]) == [EVAL_N]


@pytest.mark.parametrize("make, msg", [(m, msg) for _n, m, msg in _refusal_cases()],
                         ids=[n for n, _m, _msg in _refusal_cases()])
def test_경로_조건을_어기면_경로_열을_읽기_전에_거부하고_기록한다(strict, monkeypatch, make, msg):
    asked = _watch_scan(monkeypatch)
    kw = {"ids": {EVAL_D}} | make(strict)
    with pytest.raises(mv.EvalViewRefused, match=re.escape(msg)):
        mv.read_image_paths_eval(strict.root, kw["ids"], receipt=kw["receipt"])
    assert not any("rel_path" in c for c in asked), "거부하기 전에 경로 열을 읽었다"
    (line,) = strict.lines()
    assert line["outcome"] == "refused" and line["view"] == "image_paths" and msg in line["reason"], line


def test_경로_기록을_쓰지_못하면_읽지_않는다(strict, monkeypatch):
    asked = _watch_scan(monkeypatch)

    def boom(entry):
        raise OSError("디스크")
    monkeypatch.setattr(mv, "_append_call_log", boom)
    with pytest.raises(mv.EvalViewRefused, match="호출 기록"):
        mv.read_image_paths_eval(strict.root, {EVAL_D}, receipt=strict.receipt())
    assert not any("rel_path" in c for c in asked)


@pytest.mark.parametrize("kind", ["rehearsal", "frame_diag"])
def test_경로_덫_리허설과_진단_영수증으로는_아무것도_읽지_않는다(strict, opens, converted, monkeypatch, tmp_path, kind):
    """리허설 · 진단 목적의 영수증으로 부르면 예외다. 독이 든 스냅샷을 줘도 검증조차 부르지 않는다."""
    poisoned_root = _snapshot(tmp_path / "poisoned_trap", poison=True)
    verified: list = []
    monkeypatch.setattr(mv, "verify_snapshot", lambda r: verified.append(r))
    gits: list = []
    monkeypatch.setattr(mv, "_git", lambda *a: gits.append(a))
    name = "rehearsal" if kind == "rehearsal" else "main"
    with pytest.raises(mv.EvalViewRefused, match="영수증의 종류가"):
        mv.read_image_paths_eval(poisoned_root, {EVAL_D}, receipt=strict.receipt(name, kind=kind))
    assert opens == [] and converted == [] and verified == []
    assert gits == [], "종류 검사보다 뒤의 검사가 거부했다"
    assert strict.lines()[-1]["outcome"] == "refused"


def test_일반_판독기는_평가_행의_경로를_여전히_막는다(strict, opens):
    with pytest.raises(mv.ManifestViewError):
        mv.read_manifest_columns(strict.root, ["image_id", "rel_path"], split_filter={"eval"})
    assert opens == []
