"""본 칸 평가가 서게 하는 세 자리(1016k, 03 §0 의 ① · ② · ⑤) — 모델 · GPU 없이 합성 세계에서.

① 본실험 어댑터의 학습 행 대조(`vlm.export_preflight.check_main_train_rows`) — 페어의 참여자 행 전체에서 영수증 커밋의 학습 등록이
   정한 예산 비율 몫을 등록 시드로 다시 고른다. 비율 1.0 이면 이 판 전과 같은 값이다.
② 생성 쪽 표본 관문(`check_sample_eval_set`)은 채점 쪽 `check_main_sample` 과 공유 벡터(`tests/main_sample_vectors.py`)에서 같은
   판정이다. 사전 점검은 등록의 가중이 표본이면 영수증 커밋의 평가 등록 설정에서 주 표본 크기의 봉인을 고른다.
⑤ 본실험 모델 모드의 평가 이미지(`scripts.export_uni.main_image_paths`) — 승인 목록 파일의 `rel_path`, id 는 매니페스트의 sha256 으로
   잇고 장마다 바이트를 맞댄다.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vlm.export_run as XR
from evaluation import eval_registration as ER
from scripts import export_uni as CLI
from tests import main_sample_vectors as V
from tests import test_uni_joint_entry as J
from tests import uni_export_world as W
from tests import v6_link_world as VW
from vlm import budget as B
from vlm import export_preflight as XP
from vlm.export_writer import ExportRefused

REG_BYTES = (ROOT / "configs/train_registration.yaml").read_bytes()


def _registered_fraction() -> float:
    from vlm.registration import budget_fraction_of, registration_from_bytes

    return budget_fraction_of(registration_from_bytes(REG_BYTES))


# ================================================================ ① 학습 행 대조 — 함수
def _pairs(tmp: Path, sizes=None) -> Path:
    sizes = sizes or {"C1": 100, "C2": 40, "C3": 70}         # 셋 다 한 창(32 행)보다 크다 — 몫이 전체가 아니다
    rows = []
    for c, n in sizes.items():
        rows += [{"image_id": f"synth:{c}:{i:05d}", "client": c, "split": "train"} for i in range(n)]
        rows.append({"image_id": f"synth:{c}:val", "client": c, "split": "val"})
    p = tmp / "pairs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _digest(p: Path, c, f: float, seed: int) -> str:
    from vlm.pilot_vlm import load_pairs, train_rows_digest

    return train_rows_digest(B.apply_budget(load_pairs("train", c, pairs_path=p), fraction=f, seed_value=seed, pairs_path=p))


def _ident(f=None, seed=7, policy=True) -> dict:
    out = {"purpose": "main", "seed_value": seed}
    if f is not None:
        out["budget_fraction"] = f
        if policy:
            out["budget_policy"] = B.BUDGET_POLICY
    return out


def _code(rows_got, ident, p, *, tag="uni_local_C1", client="C1", seed=7, f=0.125):
    with pytest.raises(ExportRefused) as exc:
        XP.check_main_train_rows(rows_got, tag=tag, client=client, ident=ident, pairs_path=p, seed_value=seed, fraction=f)
    return exc.value.code


def test_비율_1_이면_전체_행_그대로다(tmp_path):
    p = _pairs(tmp_path)
    full = _digest(p, "C1", 1.0, 7)
    XP.check_main_train_rows(full, tag="uni_local_C1", client="C1", ident=_ident(), pairs_path=p, seed_value=7, fraction=1.0)
    fed = {c: _digest(p, c, 1.0, 7) for c in ("C1", "C2", "C3")}
    XP.check_main_train_rows(fed, tag="uni_fed", client=None, ident=_ident(), pairs_path=p, seed_value=7, fraction=1.0)
    with pytest.raises(ExportRefused, match="부분 목록") as exc:            # 이 판 전의 사유 그대로
        XP.check_main_train_rows(_digest(p, "C1", 0.5, 7), tag="uni_local_C1", client="C1", ident=_ident(), pairs_path=p,
                                 seed_value=7, fraction=1.0)
    assert exc.value.code == "train_rows"


@pytest.mark.parametrize(("tag", "client"), [("uni_local_C1", "C1"), ("uni_local_C3", "C3"), ("uni_central", None),
                                             ("uni_fed", None)])
def test_비율_칸은_등록의_비율과_시드로_다시_고른_행이면_지난다(tmp_path, tag, client):
    p = _pairs(tmp_path)
    got = ({c: _digest(p, c, 0.125, 7) for c in ("C1", "C2", "C3")} if tag == "uni_fed"
           else _digest(p, client, 0.125, 7))
    XP.check_main_train_rows(got, tag=tag, client=client, ident=_ident(0.125), pairs_path=p, seed_value=7, fraction=0.125)
    if tag != "uni_fed":
        assert got != _digest(p, client, 1.0, 7)                                        # 몫은 전체가 아니다


def test_다른_시드_다른_비율_전체_행은_거부한다(tmp_path):
    p = _pairs(tmp_path)
    sub = _digest(p, "C1", 0.125, 7)
    assert _code(_digest(p, "C1", 0.125, 8), _ident(0.125, seed=8), p) == "adapter_seed"      # 다른 시드로 학습 · 신원도 그 시드
    assert _code(_digest(p, "C1", 0.125, 8), _ident(0.125), p) == "train_rows"                # 신원은 등록 시드 · 행은 다른 시드
    assert _code(_digest(p, "C1", 0.25, 7), _ident(0.25), p) == "train_budget"                # 다른 비율
    assert _code(_digest(p, "C1", 1.0, 7), _ident(), p) == "train_budget"                     # 전체 행 어댑터(비율 칸 없음)
    assert _code(_digest(p, "C1", 1.0, 7), _ident(0.125), p) == "train_rows"                  # 비율 칸은 있는데 전체 행
    assert _code(sub, _ident(0.125), p, f=1.0) == "train_budget"                              # 등록은 1.0 인데 meta 는 비율
    assert _code(sub, _ident(0.125, policy=False), p) == "train_budget"                       # 정책 칸이 없다
    assert _code(sub, _ident(0.125, seed=True), p) == "adapter_seed"                          # 참 · 거짓은 시드가 아니다


def _repo(tmp: Path, files: dict[str, bytes]) -> tuple[Path, str]:
    repo = tmp / "r"
    repo.mkdir(parents=True)
    W.git(repo, "init", "-q", "-b", "main")
    (repo / "x.txt").write_text("x\n", encoding="utf-8")
    for rel, raw in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(raw)
    W.git(repo, "add", "-A")
    W.git(repo, "commit", "-q", "-m", "c")
    return repo, W.git(repo, "rev-parse", "HEAD")


def test_학습_등록의_비율은_영수증_커밋의_바이트에서_읽는다(tmp_path):
    reg = re.sub(rb"(?m)^budget_fraction: .*$", b"budget_fraction: 0.125", REG_BYTES)   # 등록의 현재 값(1.0 일 수도)과 무관하게 0.125 로 둔 바이트
    assert b"budget_fraction: 0.125" in reg
    repo, head = _repo(tmp_path, {"configs/train_registration.yaml": reg})
    assert XP._train_budget_fraction_at(repo, head) == 0.125
    (repo / "configs/train_registration.yaml").write_bytes(reg.replace(b"budget_fraction: 0.125", b"budget_fraction: 0.5"))
    assert XP._train_budget_fraction_at(repo, head) == 0.125                                   # 작업 트리가 아니라 커밋
    empty, h2 = _repo(tmp_path / "e", {})
    assert XP._train_budget_fraction_at(empty, h2) == 1.0                                     # 그 커밋에 등록이 없으면 전체 행
    bad, h3 = _repo(tmp_path / "b", {"configs/train_registration.yaml": b"kind: other\n"})
    with pytest.raises(ExportRefused) as exc:
        XP._train_budget_fraction_at(bad, h3)
    assert exc.value.code == "train_registration"


# ================================================================ ① · ② · ⑤ 배선 — 본실험 export 세계
@pytest.fixture
def approved(monkeypatch):
    s = J.Spies()
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (s.measure,))
    monkeypatch.setattr(XR, "APPROVED_GENERATOR_LOADERS", {"model": (s.loader,), "echo": (s.loader,)})
    return s


@pytest.fixture
def w(tmp_path):
    return J.joint(tmp_path)


def _main_world(w, s, *, train_reg=True, eval_reg: bytes | None = None, edit=None, n_rows=64):
    """`J.main_ready` 와 같은 본실험 세계 — 설정에 페어(C1 학습 행 `n_rows`) · 생성 한도, 영수증 커밋에 학습 등록(저장소의 실물
    바이트 — 비율 1/8)과 평가 등록 설정을 둔다. 돌려주는 값은 (영수증, 생성 목록, 등록, 페어)."""
    from evaluation.actuals import impl_key
    from vlm.seams import impl_id

    pairs = w.tmp / "main_pairs" / "pairs.jsonl"
    pairs.parent.mkdir()
    pairs.write_text("".join(json.dumps({"image_id": f"t{k:03d}", "image_path": "x.png", "client": "C1", "split": "train",
                                         "skeleton": {"defects": [], "verdict": "합격", "clauses": []}}) + "\n"
                             for k in range(n_rows)), encoding="utf-8")
    cfg = w.repo / "configs" / "base.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("uni_max_new_tokens: null", "uni_max_new_tokens: 512")
                   + f"  uni_pairs: {{path: '{pairs.as_posix()}', digest: {'p' * 64}}}\n", encoding="utf-8")
    add = ["configs/base.yaml"]
    if train_reg:
        (w.repo / "configs/train_registration.yaml").write_bytes(REG_BYTES)
        add.append("configs/train_registration.yaml")
    if eval_reg is not None:
        (w.repo / "configs/eval_registration.yaml").write_bytes(eval_reg)
        add.append("configs/eval_registration.yaml")
    W.git(w.repo, "add", *add)
    W.git(w.repo, "commit", "-q", "-m", "본실험 설정")
    keys = tuple(sorted(impl_key(impl_id(fn, seam=seam, approved=True))
                        for fn, seam in ((s.measure, "export_measure"), (s.loader, "export_generator"))))

    def ed(r):
        r.generation.impl_ids = keys
        if edit is not None:
            edit(r)

    rp, lst, _echo, r = J.main_registration(w, ids=[W.EVAL_ID], edit=ed)
    return rp, lst, r, pairs


def _adapter(w, r, pairs, *, f: float, rows_f: float | None = None, rows_seed: int | None = None):
    """본실험 C1 어댑터 — 신원에 시드 값(등록 시드표) · 비율 칸, 학습 행 digest 는 `rows_f` · `rows_seed` 의 몫(없으면 신원 그대로)."""
    seed = int(r.seeds["1"])
    ident = {"purpose": "main", "tag": "uni_local_C1", "seed_index": 1, "seed_value": seed}
    if f != 1.0:
        ident |= {"budget_fraction": f, "budget_policy": B.BUDGET_POLICY}
    dig = _digest(pairs, "C1", f if rows_f is None else rows_f, seed if rows_seed is None else rows_seed)
    W.write_adapter(w, purpose="main", base=w.repo / "main_in" / "train",
                    meta_over={"train_rows_digest": dig, "identity": ident})


def _run(w, s, rp, lst, **kw):
    try:
        W.run(w, purpose="main", mode="model", receipt_path=rp, list_path=lst, run_root=None, plan_path=None,
              folder=w.repo / "main_out" / "export", loader=s.loader, measure_fn=s.measure,
              image_path_of=kw.pop("image_path_of", s.paths(w)),
              adapter_dir=w.repo / "main_in" / "train" / "uni_local_C1_s1", **kw)
    except ExportRefused as exc:
        return exc.code
    return None


def test_본실험_export_는_학습_등록의_예산_몫으로_학습한_칸을_받는다(w, approved):
    rp, lst, r, pairs = _main_world(w, approved)
    _adapter(w, r, pairs, f=_registered_fraction())
    assert _run(w, approved, rp, lst) is None
    assert (w.repo / "main_out" / "export" / "uni_local_C1_s1.export_meta.json").is_file()


def _other_fraction(f: float) -> float:
    """등록 비율과 다른 비율 하나."""
    return 0.5 if f != 0.5 else 0.25


@pytest.mark.parametrize(("f", "rows_f", "rows_seed", "code"), [
    ("other", None, None, "train_budget"),    # 등록과 다른 비율의 어댑터
    ("reg", "other", None, "train_rows"),     # 비율 칸은 등록값인데 행은 다른 비율의 몫
    ("reg", None, 99, "train_rows"),          # 다른 시드로 고른 몫(등록 비율 1 이면 시드가 행을 바꾸지 않는다 — 건너뜀)
])
def test_본실험_export_는_학습_등록과_다른_행의_칸을_모델_전에_거부한다(w, approved, f, rows_f, rows_seed, code):
    reg = _registered_fraction()
    if rows_seed is not None and reg == 1.0:
        pytest.skip("등록 비율 1 — 시드가 학습 행을 바꾸지 않는다")
    f = _other_fraction(reg) if f == "other" else reg
    rows_f = _other_fraction(reg) if rows_f == "other" else rows_f
    rp, lst, r, pairs = _main_world(w, approved)
    _adapter(w, r, pairs, f=f, rows_f=rows_f, rows_seed=rows_seed)
    assert _run(w, approved, rp, lst) == code
    assert approved.calls["loader"] == 0 and approved.calls["image"] == 0


def test_영수증_커밋에_학습_등록이_없으면_전체_행_대조_그대로다(w, approved):
    rp, lst, r, pairs = _main_world(w, approved, train_reg=False)
    _adapter(w, r, pairs, f=0.125)
    assert _run(w, approved, rp, lst) == "train_budget"
    _adapter(w, r, pairs, f=1.0)
    assert _run(w, approved, rp, lst) is None


# ================================================================ ② 표본 관문
@pytest.mark.parametrize("case", V.CASES, ids=[c["name"] for c in V.CASES])
def test_생성_쪽과_채점_쪽_표본_관문이_같은_벡터에서_같은_판정을_낸다(case):
    from evaluation.entry_gate import EntryRejected, check_main_sample

    el, m, sealed, side = V.inputs(case)

    def code(fn):
        try:
            fn()
        except (EntryRejected, ExportRefused) as exc:
            return exc.code
        return None

    d = code(lambda: check_main_sample(el, el, m, sealed=sealed, side=side))
    c = code(lambda: XP.check_sample_eval_set(el, m, sealed=sealed, side=side))
    assert d == case["expect"] and c == (None if d is None else d.lower())


def test_목록_검사는_표본이_주어지면_전량_대신_표본을_본다():
    from types import SimpleNamespace

    el, m, sealed, side = V.inputs(V.CASES[0])
    g = SimpleNamespace(eval_list_file_sha256=el.file_sha256, eval_list_set_sha256=el.set_sha256)
    with pytest.raises(ExportRefused) as exc:
        XP.check_generation_list(el, g, mode="model", purpose="main", meta_of=lambda: m)
    assert exc.value.code == "main_list_not_full_eval"                                     # 전량 모드 그대로
    XP.check_generation_list(el, g, mode="model", purpose="main", meta_of=lambda: m, sample={"sealed": sealed, "side": side})


SIDE_HEAD = "image_id,stratum,G_h,n_h"


def _sample_world(w, s, *, main_size=3000, side_ok=True):
    """표본 모드 — 등록의 가중 `stratified_g_over_n`, 영수증 커밋의 평가 등록 설정에 2000 · 3000 · 4000 봉인(3000 이 이 세계의 한 장)."""
    from evaluation.eval_list import canonical_bytes, validate_eval_list
    from evaluation.sample_weights import WEIGHTING_SAMPLE

    el = validate_eval_list(canonical_bytes([W.EVAL_ID]))
    side_raw = f"{SIDE_HEAD}\n{W.EVAL_ID},S1,1,1\n".encode()
    other = {"images": 9, "groups": 9, "file_sha256": "a" * 64, "set_sha256": "b" * 64, "side_sha256": "c" * 64}
    sealed = {"2000": other, "3000": {"images": 1, "groups": 1, "file_sha256": el.file_sha256, "set_sha256": el.set_sha256,
                                      "side_sha256": hashlib.sha256(side_raw).hexdigest()}, "4000": other}
    # 1 차 꼴의 설정 전부 — 네 소비자 공통 판독(`load_bound_eval`)은 봉인만이 아니라 꼴 · 코드 상수 · base · 등록 결속까지 본다
    doc = copy.deepcopy(yaml.safe_load((ROOT / ER.CONFIG_REL).read_text(encoding="utf-8")))
    doc["population"]["snapshot_digest"] = w.digest
    doc["population"]["sample"].update({"main_size": main_size, "sealed": {**doc["population"]["sample"]["sealed"], **sealed}})
    doc["population"]["sample"]["sealed"].pop("1000", None)
    doc["population"]["sample"]["sealed"]["1000"] = other

    def edit(r):
        bound = VW.bind_to_doc(r, {**doc, "population": {**doc["population"], "sample": {**doc["population"]["sample"],
                                                                                        "main_size": 3000}}}, w.repo)
        for area in ("population", "metric", "generation", "ci"):
            setattr(r, area, getattr(bound, area))
        r.seeds, r.planned_seed_count = bound.seeds, bound.planned_seed_count
        r.population.weighting = WEIGHTING_SAMPLE

    rp, lst, r, pairs = _main_world(w, s, train_reg=False, edit=edit,
                                    eval_reg=yaml.safe_dump(doc, allow_unicode=True).encode("utf-8"))
    _adapter(w, r, pairs, f=1.0)
    side = w.tmp / "side.csv"
    side.write_bytes(side_raw if side_ok else side_raw.replace(b",1,1", b",2,1"))
    return rp, lst, side


def test_표본_모드의_본실험_export_는_봉인한_주_표본과_곁_파일을_받는다(w, approved):
    rp, lst, side = _sample_world(w, approved)
    assert _run(w, approved, rp, lst, side_path=side) is None
    assert _run(w, approved, rp, lst) == "side_missing"


@pytest.mark.parametrize(("over", "code"), [({"main_size": 1000}, "eval_binding"),     # 가짜 1,000 — 등록(3000 결속)과 어긋난다
                                            ({"side_ok": False}, "side_hash")])
def test_표본_모드는_봉인되지_않은_크기와_다른_곁_파일을_모델_전에_거부한다(w, approved, over, code):
    rp, lst, side = _sample_world(w, approved, **over)
    assert _run(w, approved, rp, lst, side_path=side) == code
    assert approved.calls["loader"] == 0 and approved.calls["image"] == 0


def test_전량_모드에는_곁_파일을_주면_거부하고_본실험_모델_밖에서는_받지_않는다(w, approved):
    rp, lst, r, pairs = _main_world(w, approved, train_reg=False)
    _adapter(w, r, pairs, f=1.0)
    side = w.tmp / "side.csv"
    side.write_text(f"{SIDE_HEAD}\n", encoding="utf-8")
    assert _run(w, approved, rp, lst, side_path=side) == "side_in_full_mode"
    with pytest.raises(ExportRefused) as exc:
        XP.export_preflight(purpose="rehearsal", mode="model", tag="uni_local_C1", seed_index=1, receipt_path=w.receipt,
                            list_path=w.lists["model"], snapshot_root=w.snap, adapter_dir=w.adapter, plan_path=w.plan,
                            run_root=w.root, measure_env=W.measure, impl_ids={}, parser_sha256="0" * 64, side_path=side)
    assert exc.value.code == "side_scope"


# ================================================================ ⑤ 본실험 이미지 경로
def _image_inputs(w, *, rel="sub/a.png", body: bytes | None = None, listed=True, head="rel_path,tile_rel,bytes,sha256"):
    from evaluation.eval_list import canonical_bytes

    src = w.tmp / "img" / f"{W.EVAL_ID}.png"
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    root = w.tmp / "evalroot"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    if body is None:
        shutil.copyfile(src, root / "sub" / "a.png")
    elif body:
        (root / "sub" / "a.png").write_bytes(body)
    lst = w.tmp / "eval_gen.list"
    lst.write_bytes(canonical_bytes([W.EVAL_ID]))
    image_list = w.tmp / "eval_paths.txt"
    row = f"{rel},tiles/a.png,{src.stat().st_size},{sha if listed else 'e' * 64}"
    image_list.write_text(f"# 설명 줄\n{head}\n{row}\n", encoding="utf-8")
    return root, image_list, lst


def test_본실험_이미지는_목록_파일의_rel_path_이고_바이트가_매니페스트와_같다(w):
    root, image_list, lst = _image_inputs(w)
    got = CLI.main_image_paths(root, image_list=image_list, snapshot=w.snap, list_path=lst)
    assert got == {W.EVAL_ID: root / "sub" / "a.png"}
    from types import SimpleNamespace

    G = SimpleNamespace(mode="model", pairs_path=None)
    assert CLI.pair_image_paths(G, purpose="main", image_root=root, image_list=image_list, snapshot=w.snap,
                                list_path=lst) == got
    with pytest.raises(ExportRefused) as exc:
        CLI.pair_image_paths(G, purpose="main")
    assert exc.value.code == "main_image_paths_undefined"


@pytest.mark.parametrize(("kw", "code"), [
    ({"body": b"other bytes"}, "main_image_files"),         # 바이트가 매니페스트와 다르다
    ({"body": b""}, "main_image_files"),                    # 파일이 없다
    ({"listed": False}, "main_image_files"),                # 목록 파일에 그 sha 가 없다
    ({"rel": "../a.png"}, "main_image_files"),              # 뿌리 밖
    ({"head": "rel_path,tile_rel,bytes"}, "main_image_list"),
])
def test_본실험_이미지는_어긋나면_모델_전에_거부한다(w, kw, code):
    root, image_list, lst = _image_inputs(w, **kw)
    with pytest.raises(ExportRefused) as exc:
        CLI.main_image_paths(root, image_list=image_list, snapshot=w.snap, list_path=lst)
    assert exc.value.code == code and W.EVAL_ID not in str(exc.value)


def test_명령줄의_본실험_모델_export_는_이미지_뿌리와_목록_파일로_연다(w, approved, capsys):
    rp, lst, r, pairs = _main_world(w, approved, train_reg=False)
    _adapter(w, r, pairs, f=1.0)
    root, image_list, _ = _image_inputs(w)
    folder = w.repo / "main_out" / "cli"
    argv = ["--purpose", "main", "--mode", "model", "--tag", "uni_local_C1", "--seed-index", "1", "--receipt", str(rp),
            "--list", str(lst), "--snapshot", str(w.snap), "--folder", str(folder), "--device", W.DEVICE,
            "--adapter-dir", str(w.repo / "main_in" / "train" / "uni_local_C1_s1")]
    seams = {"measure_env": approved.measure, "load_generator": approved.loader, "repo": w.repo, "code_repo": w.code,
             "exposure_ledger": w.exposure, "clock": W.Clock()}
    assert CLI.main(argv, **seams) == CLI.EXIT_REFUSED                                    # 뿌리 · 목록 파일 없음
    assert capsys.readouterr().err.startswith("[main_image_paths_undefined]") and approved.calls["loader"] == 0
    assert CLI.main(argv + ["--image-root", str(root), "--image-list", str(image_list)], **seams) == CLI.EXIT_OK
    assert (folder / "uni_local_C1_s1.export_meta.json").is_file()
    rows = [json.loads(x) for x in (folder / "uni_local_C1_s1.generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["image_sha256"] for x in rows] == [hashlib.sha256((root / "sub" / "a.png").read_bytes()).hexdigest()]


@pytest.mark.parametrize(("extra", "word"), [(["--image-root", "r", "--image-list", "l"], "[image_root_scope]"),
                                             (["--image-root", "r"], "[image_root_scope]")])
def test_이미지_뿌리는_본실험_모델_모드에서만_둘을_함께_받는다(tmp_path, capsys, extra, word):
    base = ["--mode", "model", "--tag", "uni_local_C1", "--seed-index", "1", "--receipt", "x", "--list", "x",
            "--snapshot", "x", "--folder", str(tmp_path / "f"), "--device", "cpu:0"]
    assert CLI.main(["--purpose", "rehearsal", *base, *extra]) == CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith(word)
    assert CLI.main(["--purpose", "main", *base, "--image-root", "r"]) == CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("[image_root_args]")
    assert not (tmp_path / "f").exists()


@pytest.mark.parametrize("kind", ["dotdot", "absolute"])
def test_본실험_이미지는_뿌리_밖_경로를_바이트가_맞아도_거부한다(w, kind):
    """G 05 m-2 — 뿌리 밖에 매니페스트와 같은 바이트의 파일을 둔다. 경로 꼴 검사만이 그것을 막는다(바이트 대조로는 지나간다)."""
    root, image_list, lst = _image_inputs(w)
    src = w.tmp / "img" / f"{W.EVAL_ID}.png"
    outside = w.tmp / "outside.png"
    shutil.copyfile(src, outside)
    rel = "../outside.png" if kind == "dotdot" else outside.resolve().as_posix()
    image_list.write_text(f"rel_path,tile_rel,bytes,sha256\n{rel},t,1,{hashlib.sha256(src.read_bytes()).hexdigest()}\n",
                          encoding="utf-8")
    with pytest.raises(ExportRefused) as exc:
        CLI.main_image_paths(root, image_list=image_list, snapshot=w.snap, list_path=lst)
    assert exc.value.code == "main_image_files" and "경로 꼴 1" in str(exc.value) and "파일 없음 0" in str(exc.value)


@pytest.mark.parametrize("kind", ["missing", "not_utf8"])
def test_승인_목록_파일이_없거나_UTF8_이_아니면_사유_코드로_거부한다(w, kind):
    """G 05 m-3 — traceback(종료 1)이 아니라 `main_image_list` 다."""
    root, image_list, lst = _image_inputs(w)
    if kind == "missing":
        image_list.unlink()
    else:
        image_list.write_bytes("rel_path,sha256\n".encode("utf-16"))
    with pytest.raises(ExportRefused) as exc:
        CLI.main_image_paths(root, image_list=image_list, snapshot=w.snap, list_path=lst)
    assert exc.value.code == "main_image_list"


def test_명령줄은_승인_목록_파일을_못_읽으면_종료_2_와_사유_코드다(w, approved, capsys):
    rp, lst, r, pairs = _main_world(w, approved, train_reg=False)
    _adapter(w, r, pairs, f=1.0)
    root, _image_list, _ = _image_inputs(w)
    argv = ["--purpose", "main", "--mode", "model", "--tag", "uni_local_C1", "--seed-index", "1", "--receipt", str(rp),
            "--list", str(lst), "--snapshot", str(w.snap), "--folder", str(w.repo / "main_out" / "cli2"), "--device",
            W.DEVICE, "--adapter-dir", str(w.repo / "main_in" / "train" / "uni_local_C1_s1"),
            "--image-root", str(root), "--image-list", str(w.tmp / "no_such_list.txt")]
    seams = {"measure_env": approved.measure, "load_generator": approved.loader, "repo": w.repo, "code_repo": w.code,
             "exposure_ledger": w.exposure, "clock": W.Clock()}
    assert CLI.main(argv, **seams) == CLI.EXIT_REFUSED
    assert capsys.readouterr().err.startswith("[main_image_list]") and approved.calls["loader"] == 0


# ================================================================ 학습 레시피(02 2판 §3-5 의 등록 사슬 · 외부 검토 194 M6 의 export 칸)
def _tc_text(recipe: str) -> str:
    """`W.train_config_text` 와 같은 입력에서 레시피만 고른 학습 설정 원문 — 학습 쪽 함수가 낸다(손으로 적지 않는다)."""
    from vlm.coords import CoordCfg, coord_cfg_hash
    from vlm.pilot_vlm import canonical_json, target_contract_sha256
    from vlm.train_cell import train_config_object

    metrics = {"processor_config_sha256": W.hx("proc"),
               "config_blocks": {k: {} for k in ("lora", "quant", "optimizer", "loss", "lr_schedule")},
               "prompt_sha256": W.prompt_sha(), "template_mode": "enable_thinking=False",
               "coord_cfg_hash": coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")),
               "target_contract_sha256": target_contract_sha256(recipe)}
    return canonical_json(train_config_object(
        model_id="Qwen/Qwen3.5-4B", model_revision="a" * 40, pairs_digest=W.hx("pairs"),
        chat_template_kwargs={"enable_thinking": False}, coord_space="ABS_ORIG", num_rounds=2, local_epochs=1,
        total_epochs=2, prompt_sha256=W.prompt_sha(), init_digest=W.hx("init"), metrics=metrics, recipe=recipe))


def test_1차_레시피의_학습_설정_원문은_세계의_원문과_같은_바이트다():
    assert _tc_text("json_only") == W.train_config_text()


@pytest.mark.parametrize("recipe", ["json_only", "json_line_then_basis"])
def test_본실험_export_는_학습_설정_원문의_계약에서_레시피를_되찾아_생성기_설정에_싣는다(tmp_path, monkeypatch, approved, recipe):
    """레시피는 export 가 따로 받지 않는다 — 등록된 학습 설정 원문의 계약 해시에서 되찾는다(에코 글도 그 레시피로 짓는다).
    새 레시피의 산출은 판 6 등록으로만 export 한다(지시 1023c — 등록 판 짝, `tests/test_regime2_end_to_end.py` 가 판 5 를 거부한다)."""
    seen: list[str] = []
    real = XP.export_preflight

    def spy(**kw):
        res = real(**kw)
        seen.append(res.gen_cfg.target_recipe)
        return res

    monkeypatch.setattr(XP, "export_preflight", spy)
    monkeypatch.setattr(W, "train_config_text", lambda: _tc_text(recipe))
    w = J.joint(tmp_path)
    rp, lst, r, pairs = _main_world(w, approved, edit=(lambda r: W.to_v6(r, inputs_dir=tmp_path / "v6_inputs")) if recipe == "json_line_then_basis" else None)
    _adapter(w, r, pairs, f=_registered_fraction())
    assert _run(w, approved, rp, lst) is None
    assert seen == [recipe]


def test_본실험_export_는_모르는_계약의_학습_설정_원문을_모델_전에_거부한다(tmp_path, monkeypatch, approved):
    from vlm.pilot_vlm import canonical_json

    bad = {**json.loads(W.train_config_text()), "target_contract_sha256": "0" * 64}
    monkeypatch.setattr(W, "train_config_text", lambda: canonical_json(bad))
    w = J.joint(tmp_path)
    rp, lst, r, pairs = _main_world(w, approved)
    _adapter(w, r, pairs, f=_registered_fraction())
    assert _run(w, approved, rp, lst) == "train_config_contract"
    assert approved.calls["loader"] == 0 and approved.calls["image"] == 0
