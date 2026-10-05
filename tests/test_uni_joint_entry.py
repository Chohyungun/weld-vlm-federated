"""쓰는 쪽(export)과 읽는 쪽(평가 쪽 본채점 진입점)의 공동 진입점 시험 — 리허설 2판 §6-3.

**같은 합성 등록 파일**을 평가 쪽 `prereg_unified` 로 만들고 두 쪽이 그 파일을 같이 읽는다. 두 쪽이 **각자** 거부해야 한다 —
한쪽만 막으면 다른 쪽의 결함이 드러나지 않는다. 거부는 종료 코드뿐 아니라 **사유 코드**를 맞댄다. 쓰는 쪽은 평가 쪽 진입 함수의
거부를 `entry_<코드>` 로 감싸고, 읽는 쪽은 stderr 에 `[코드]` 를 쓴다 — 같은 검사에 걸렸으면 코드가 같다.

평가 쪽 코드는 **부르기만 한다**(`scripts.probe.score_unified.main` 과 그 시험의 이음새 `Seam`, `tests.test_prereg_unified.filled`).
합성 세계는 `tests/uni_export_world.py` 다 — 본체 체크아웃 자리의 git 저장소 · 동결 스냅샷 · 리허설 루트 · 노출 원장.

| 표의 행 | 여기서 |
|---|---|
| 가-1 · 가-2 · 가-4 · 가-4′ · 가-6 · 가-8(리허설/val) · 가-9 · 가-10 · 추가 | 두 쪽 |
| 가-7 | 평가 쪽 진입점으로 — 진입점 안의 묶음 검증이 내는 사유를 검증기를 감싼 기록으로 본다(엄격 진입점은 채점 단계가 없어 칸 거부의 종료 2 대신 종료 3 이다) |
| 가-8(엄격/eval) | 쓰는 쪽 본실험 모델 export 가 봉인되고, 평가 쪽 엄격 진입점이 그 묶음을 **거부 없이 검증한다**. 에코는 아직 서지 않는다 — 본실험 · 진단 에코의 학습 설정 원문 자리가 정해지지 않아 쓰는 쪽이 `echo_train_config_place` 로 멈추고, 평가 쪽은 에코 묶음 없이 카나리아에서 종료 3 이다 |
| 가-8(진단/val) · 진-1 · 진-2 · 진-4 | 쓰는 쪽은 본다(로컬 합성 진단 영수증의 모델 export 봉인 · 작업 트리와 다른 등록은 커밋의 것 · 진단 규칙 칸이 빈 등록은 시작 거부 · 커밋된 진단 영수증 + 대역은 대역 호출 0 으로 거부). 평가 쪽 진단 경로(본줄기 `cc2c48c` 부터 구현됨)는 같은 사유 코드로 거부하거나(영수증 종류 · 빈 규칙 칸 · 커밋된 영수증 + 대역), 로컬 합성 영수증의 대역 묶음을 **판정하지 않는다**(`frame_diag_not_judged` — 묶음 거부) |
| 가-3 · 가-5 · 진-3 | 평가 쪽만의 행이다 — 평가 쪽 시험이 본다 |

**본실험 목적을 쓰는 쪽에서 지나게 하는 법.** 본실험은 승인된 실제 구현만 받는데 모델 생성기의 실제 구현이 아직 없다. 그래서 시험이 승인
목록을 **호출을 세는 대역**으로 바꿔 끼운다 — 순서 3 을 지나 내용 검사(순서 4 · 6)에 닿게 하려는 것이다. 거부 사유 코드로 그 자리를 맞대므로
대역 때문에 순서 3 에서 막힌 시험이 겉으로 통과하지 않는다. 대역의 호출 수(실측 · 생성기 적재 · 이미지 열기)가 0 이어야 한다(가-9).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import fields as dc_fields
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.probe.score_unified as SU
from evaluation import prereg_unified as P
from evaluation.eval_list import canonical_bytes, validate_eval_list
from tests import uni_export_world as W
from vlm import export_run as XR
from vlm.export_writer import ExportRefused

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=W.KST)
FIXTURES_REL = "vlm/coords_fixtures/golden_fixtures.json"
MAIN_REG_REL = "configs/registration/main-20261001-1.json"
BOX = [10, 20, 30, 40]                       # 합성 스냅샷의 주석 박스(`uni_export_world._ann`)와 같다


# ---------------------------------------------------------------- 세계
class _EchoImageProcessor:
    patch_size = 16

    def __call__(self, images, return_tensors=None):
        w, h = images[0].size
        return {"image_grid_thw": [[1, (h // 32) * 2, (w // 32) * 2]]}      # 1280×720 → 1280×704 의 격자 (1, 44, 80)


class _EchoTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text) // 4 + 1))}


class EchoProc:
    """에코 생성기에 넣는 합성 프로세서 — 격자와 토큰 수만 낸다(HF 캐시에 기대지 않는다)."""

    image_processor = _EchoImageProcessor()
    tokenizer = _EchoTokenizer()


def _canary(r) -> None:
    """에코 관문의 두 칸 — 평가 쪽 함수가 낸 값을 그대로 넣는다(쓰는 쪽이 지어내지 않는다, 2판 §1-2 의 0′)."""
    from evaluation.coord_check import coord_fixture_digest, unified_coord_cfg_record
    from vlm.coords import CoordCfg, coord_cfg_hash

    fd = coord_fixture_digest((ROOT / FIXTURES_REL).read_bytes())
    r.canary.coord_cfg = unified_coord_cfg_record(
        coord_cfg_hash(CoordCfg(coord_space="ABS_ORIG")), fixture_digest=fd, registered_on="2026-10-01",
        coords_source=(ROOT / "vlm/coords.py").read_bytes(),
        hash_record=(ROOT / "vlm/coords_fixtures/HASH_RECORD.yaml").read_bytes())
    r.canary.coord_fixture_sha256 = fd


def joint(tmp: Path):
    """합성 세계 + 본체 체크아웃 자리의 골든 픽스처 + 에코 관문 칸이 찬 리허설 등록 + 주석과 같은 박스의 페어 val 행."""
    w = W.build(tmp)
    (w.repo / FIXTURES_REL).parent.mkdir(parents=True)
    (w.repo / FIXTURES_REL).write_bytes((ROOT / FIXTURES_REL).read_bytes())
    W.git(w.repo, "add", "-A")
    W.git(w.repo, "commit", "-q", "-m", "골든 픽스처")
    w.reg = W.registration(w, edit=_canary)
    w.receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-2")
    w.pairs = tmp / "pairs.jsonl"
    w.pairs.write_text("".join(json.dumps({"image_id": i, "image_path": str(p), "client": "C1", "split": "val",
                                           "skeleton": {"defects": [{"type": "2011", "bbox_px": BOX}],
                                                        "verdict": "불합격", "clauses": []}}) + "\n"
                               for i, p in w.images.items()), encoding="utf-8")
    (w.repo / "main_in" / "bundles").mkdir(parents=True)
    (w.repo / "main_in" / "train").mkdir(parents=True)
    return w


@pytest.fixture
def w(tmp_path):
    return joint(tmp_path)


def echo_loader(w):
    from vlm.export_echo import load_echo_generator

    def load(cfg):                       # 리허설에서만 받는 대역 적재 — 실제 에코 생성기에 합성 프로세서와 페어 경로를 준다
        return load_echo_generator(replace(cfg, pairs_path=str(w.pairs)), processor=EchoProc())

    return load


def _commit_train_config(w, reg_rel: str) -> str:
    """등록 옆의 학습 설정 원문 — 같은 줄기 이름 `{줄기}.train_config.json`, 바이트는 등록 단계의 정규 문자열 그대로(끝 LF 없음).
    등록과 같은 커밋에 넣으려고 색인에만 올린다(평가 쪽 `configs/registration/README.md`)."""
    from vlm.export_preflight import echo_train_config_rel

    rel = echo_train_config_rel(reg_rel)
    (w.repo / rel).write_bytes(W.train_config_text().encode("utf-8"))
    W.git(w.repo, "add", rel)
    return rel


def main_registration(w, *, ids=None, edit=None, branch=None, train_config=True):
    """본실험 등록 — 채점 쪽은 평가 쪽 시험의 `filled`, 생성 쪽은 이 세계의 값. 본줄기(또는 `branch`)에 커밋하고 영수증을 쓴다.
    `train_config` 면 등록 옆의 학습 설정 원문도 같은 커밋에 둔다."""
    from tests.test_prereg_unified import filled

    ids = list(W.IDS if ids is None else ids)
    el = validate_eval_list(canonical_bytes(ids))
    base = W.registration(w, edit=_canary)
    r = filled(purpose="main")
    for f in dc_fields(P.GenerationSpec):
        setattr(r.generation, f.name, getattr(base.generation, f.name))
    g = r.generation
    g.purpose, g.list_split = "main", "eval"
    for name in P.PURPOSE_NULL["main"]:
        setattr(g, name.split(".", 1)[1], None)
    g.eval_list_file_sha256, g.eval_list_set_sha256 = el.file_sha256, el.set_sha256
    r.population.eval_list_file_sha256, r.population.eval_list_set_sha256 = el.file_sha256, el.set_sha256
    r.canary = replace(r.canary, coord_cfg=base.canary.coord_cfg, coord_fixture_sha256=base.canary.coord_fixture_sha256)
    r.metric.class_codes, r.metric.coupling_rule = base.metric.class_codes, base.metric.coupling_rule
    if edit is not None:
        edit(r)
    raw = P.dump_registration(r)
    if branch is not None:
        W.git(w.repo, "checkout", "-q", "-b", branch)
    (w.repo / MAIN_REG_REL).parent.mkdir(parents=True, exist_ok=True)
    (w.repo / MAIN_REG_REL).write_bytes(raw)
    W.git(w.repo, "add", MAIN_REG_REL)
    if train_config:
        _commit_train_config(w, MAIN_REG_REL)
    W.git(w.repo, "commit", "-q", "-m", "본실험 등록")
    head = W.git(w.repo, "rev-parse", "HEAD")
    if branch is not None:
        W.git(w.repo, "checkout", "-q", "main")
    receipt = {"generation_sha256": r.generation_sha256(), "scoring_sha256": r.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "main",
               "registration_path": MAIN_REG_REL, "registration_file_sha256": hashlib.sha256(raw).hexdigest()}
    rp = w.repo / "main_in" / "main-20261001-1.receipt.json"
    rp.write_text(json.dumps(receipt), encoding="utf-8")
    lst = w.repo / "main_in" / "gen.txt"
    lst.write_bytes(canonical_bytes(ids))
    echo = w.repo / "main_in" / "echo.txt"
    echo.write_bytes(canonical_bytes(list(W.IDS)))
    return rp, lst, echo, r


# ---------------------------------------------------------------- 두 쪽을 부르는 법
class Spies:
    """쓰는 쪽의 호출 기록 — 실측 · 생성기 적재 · 이미지 열기. 거부 시험에서 셋 다 0 이어야 한다."""

    def __init__(self):
        self.calls = {"measure": 0, "loader": 0, "image": 0}
        # 승인 판정은 객체의 `is` 다 — 메서드를 한 번만 묶어 같은 객체를 쓴다(속성을 읽을 때마다 새 묶음이 생긴다)
        self.measure, self.loader = self._measure, self._loader

    def _measure(self, spec):
        self.calls["measure"] += 1
        return dict(W.MEASURED)

    def _loader(self, cfg):
        self.calls["loader"] += 1
        return W.gen()

    def paths(self, w):
        spies = self
        images = {**w.images, W.EVAL_ID: w.tmp / "img" / f"{W.EVAL_ID}.png"}

        class Paths(dict):
            def __getitem__(self, k):
                spies.calls["image"] += 1
                return images[k]

        return Paths()


@pytest.fixture
def approved(monkeypatch):
    """승인 목록을 호출을 세는 대역으로 바꿔 끼운다(머리말) — 본실험 목적이 순서 3 을 지나 내용 검사에 닿게."""
    s = Spies()
    monkeypatch.setattr(XR, "APPROVED_MEASURES", (s.measure,))
    monkeypatch.setattr(XR, "APPROVED_GENERATOR_LOADERS", {"model": (s.loader,), "echo": (s.loader,)})
    return s


def c_main(w, s, *, receipt, list_path, mode="model", folder=None, adapter_dir=None):
    """쓰는 쪽 — 본실험 목적. 돌려주는 값은 거부 사유 코드(지나면 None)."""
    try:
        W.run(w, purpose="main", mode=mode, receipt_path=receipt, list_path=list_path, run_root=None, plan_path=None,
              folder=folder or w.repo / "main_out" / "export", loader=s.loader, measure_fn=s.measure,
              image_path_of=s.paths(w),
              adapter_dir=(adapter_dir or w.repo / "main_in" / "train" / "uni_local_C1_s1") if mode == "model" else None)
    except ExportRefused as exc:
        return exc.code
    return None


def c_reh(w, *, purpose="rehearsal", receipt=None, **kw):
    try:
        W.run(w, purpose=purpose, receipt_path=receipt or w.receipt, **kw)
    except ExportRefused as exc:
        return exc.code
    return None


@pytest.fixture
def d_spy(monkeypatch):
    """읽는 쪽의 호출 기록 — 정답 읽기 · 묶음 검증(가-9). 진입 전 거부에서 둘 다 0 이어야 한다."""
    calls = {"gold": 0, "verify": 0}

    def gold(*a, **k):
        calls["gold"] += 1
        raise AssertionError("진입 전 거부인데 정답을 읽었다")

    def verify(*a, **k):
        calls["verify"] += 1
        raise AssertionError("진입 전 거부인데 묶음을 열었다")

    monkeypatch.setattr(SU, "read_annotations_view", gold)
    monkeypatch.setattr(SU, "verify_bundle", verify)
    return calls


def d_main(w, capsys, *, registration, purpose, receipt, gen, echo, pop=None, bundles=None, train=None, out=None,
           plan=None) -> tuple[int, str | None]:
    """읽는 쪽 — 종료 코드와 stderr 의 `[사유 코드]`."""
    import re

    argv = ["--registration", registration, "--purpose", purpose, "--receipt", str(receipt), "--snapshot", str(w.snap),
            "--bundles", str(bundles or w.repo / "main_in" / "bundles"), "--generation-list", str(gen),
            "--scoring-list", str(pop or gen), "--echo-list", str(echo),
            "--train-root", str(train or w.repo / "main_in" / "train"),
            "--out", str(out or (w.repo / "outputs" / "main_u" if registration == "strict" else w.root)),
            "--device", "cpu:0"] + (["--plan", str(plan)] if plan is not None else [])
    capsys.readouterr()
    Path(argv[argv.index("--bundles") + 1]).mkdir(parents=True, exist_ok=True)   # 폴더가 없으면 읽는 쪽이 예외로 멈춘다(보고)
    code = SU.main(argv, _seam=SU.Seam(checkout=w.repo, repo=w.repo), _now=NOW)
    m = re.search(r"\[([A-Z_]+)\]", capsys.readouterr().err)
    return code, (m.group(1) if m else None)


def _same(c_code: str | None, d_code: str | None) -> None:
    assert c_code is not None and c_code.startswith("entry_"), c_code
    assert c_code == f"entry_{d_code}", (c_code, d_code)


# ---------------------------------------------------------------- 가-1 · 가-9
def test_가1_엄격과_val_목록이면_두_쪽이_같은_분할_검사로_거부하고_호출이_0이다(w, approved, d_spy, capsys):
    rp, lst, echo, _ = main_registration(w)
    c = c_main(w, approved, receipt=rp, list_path=lst)
    code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo)
    assert (code, d) == (SU.EXIT_ENTRY, "LIST_SPLIT")
    _same(c, d)
    assert approved.calls == {"measure": 0, "loader": 0, "image": 0} and d_spy == {"gold": 0, "verify": 0}
    assert not (w.repo / "main_out").exists()


# ---------------------------------------------------------------- 가-2
def test_가2_리허설_등록의_기준_분할이_eval_이면_두_쪽이_거부한다(w, d_spy, capsys):
    w.reg = W.registration(w, edit=lambda r: (_canary(r), setattr(r.generation, "list_split", "eval")))
    receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-3")
    loader = W.Loader()
    c = c_reh(w, receipt=receipt, loader=loader)
    code, d = d_main(w, capsys, registration="probe", purpose="rehearsal", receipt=receipt, gen=w.lists["model"],
                     echo=w.lists["echo"], bundles=w.root / "export", train=w.root / "train", plan=w.plan)
    # 두 쪽 다 등록 객체를 만들 때(목적과 분할의 관계) **같은 사유 코드**로 멈춘다 — 평가 쪽 예외가 사유 코드를 갖게 됐다(본줄기 cc2c48c)
    assert c == "entry_REGISTRATION_INVALID" and (code, d) == (SU.EXIT_ENTRY, "REGISTRATION_INVALID"), (c, code, d)
    assert loader.calls == 0 and d_spy == {"gold": 0, "verify": 0}


def test_가2_리허설_목록에_평가_id_가_섞이면_두_쪽이_같은_분할_검사로_거부한다(w, d_spy, capsys):
    raw = canonical_bytes(list(W.IDS[:3]) + [W.EVAL_ID])
    el = validate_eval_list(raw)
    for p in w.lists.values():
        p.write_bytes(raw)

    def edit(r):
        _canary(r)
        g = r.generation
        g.eval_list_file_sha256 = g.echo_list_file_sha256 = el.file_sha256
        g.eval_list_set_sha256 = g.echo_list_set_sha256 = el.set_sha256

    w.reg = W.registration(w, edit=edit)
    receipt = W.write_registration(w, w.reg, name="rehearsal-20261001-3")
    loader = W.Loader()
    c = c_reh(w, receipt=receipt, loader=loader)
    code, d = d_main(w, capsys, registration="probe", purpose="rehearsal", receipt=receipt, gen=w.lists["model"],
                     echo=w.lists["echo"], bundles=w.root / "export", train=w.root / "train", plan=w.plan)
    assert (code, d) == (SU.EXIT_ENTRY, "LIST_SPLIT")
    _same(c, d)
    assert loader.calls == 0 and d_spy == {"gold": 0, "verify": 0}


# ---------------------------------------------------------------- 가-4 · 가-4′
@pytest.mark.parametrize("which", ["adapter", "list"])
def test_가4_엄격의_입력이_리허설_루트_아래면_두_쪽이_거부한다(w, approved, d_spy, capsys, which):
    rp, lst, echo, _ = main_registration(w, ids=[W.EVAL_ID])
    if which == "list":
        inside = w.root / "lists" / "main_gen.txt"
        inside.write_bytes(lst.read_bytes())
        c = c_main(w, approved, receipt=rp, list_path=inside)
        code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=inside, echo=echo)
    else:
        c = c_main(w, approved, receipt=rp, list_path=lst, adapter_dir=w.adapter)
        code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo,
                         train=w.root / "train")
    assert c == "run_root" and (code, d) == (SU.EXIT_ENTRY, "INPUT_UNDER_REHEARSAL")
    assert approved.calls == {"measure": 0, "loader": 0, "image": 0} and d_spy == {"gold": 0, "verify": 0}


def test_가4_엄격의_출력이_비본실험_부모_아래면_두_쪽이_거부한다(w, approved, d_spy, capsys):
    rp, lst, echo, _ = main_registration(w, ids=[W.EVAL_ID])
    c = c_main(w, approved, receipt=rp, list_path=lst, folder=w.root / "export_main")
    code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo, out=w.root)
    assert c == "run_root" and (code, d) == (SU.EXIT_ENTRY, "OUT_ROOT")
    assert not (w.root / "export_main").exists() and approved.calls["measure"] == 0 and d_spy["verify"] == 0


# ---------------------------------------------------------------- 가-6 · 추가
@pytest.mark.parametrize("kind", ["rehearsal", "frame_diag"])
def test_가6_엄격에_리허설이나_진단_영수증을_주면_두_쪽이_영수증_종류로_거부한다(w, approved, d_spy, capsys, kind):
    rc = json.loads(w.receipt.read_text(encoding="utf-8"))
    p = w.repo / "main_in" / f"{kind}.receipt.json"
    p.write_text(json.dumps(rc | {"kind": kind}), encoding="utf-8")
    lst = w.repo / "main_in" / "gen.txt"
    lst.write_bytes(w.lists["model"].read_bytes())
    c = c_main(w, approved, receipt=p, list_path=lst)
    code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=p, gen=lst, echo=lst)
    assert (code, d) == (SU.EXIT_ENTRY, "RECEIPT_KIND")
    _same(c, d)
    assert approved.calls == {"measure": 0, "loader": 0, "image": 0} and d_spy == {"gold": 0, "verify": 0}


def test_추가_진단_영수증으로_리허설을_돌리면_두_쪽이_거부한다(w, d_spy, capsys):
    rc = json.loads(w.receipt.read_text(encoding="utf-8"))
    p = w.root / "registration" / "fdiag.receipt.json"
    p.write_text(json.dumps(rc | {"kind": "frame_diag"}), encoding="utf-8")
    loader = W.Loader()
    c = c_reh(w, receipt=p, loader=loader)
    code, d = d_main(w, capsys, registration="probe", purpose="rehearsal", receipt=p, gen=w.lists["model"],
                     echo=w.lists["echo"], bundles=w.root / "export", train=w.root / "train", plan=w.plan)
    assert (code, d) == (SU.EXIT_ENTRY, "RECEIPT_KIND")
    _same(c, d)
    assert loader.calls == 0 and d_spy == {"gold": 0, "verify": 0}


def test_추가_리허설_영수증으로_진단을_돌리면_쓰는_쪽이_영수증_종류로_거부한다(w, d_spy, capsys):
    """진단 루트(`fdiag-*`)에 리허설 영수증 · 목록 · 계획을 옮겨 둔다 — 루트 검사를 지나 영수증 종류에서 멈춘다.
    읽는 쪽의 진단 경로도 **같은 사유(영수증 종류)**로 진입 전에 멈춘다 — 정답 읽기 · 묶음 검증 0."""
    froot = w.parent / "fdiag-20261001T000000"
    shutil.copytree(w.root / "lists", froot / "lists")
    shutil.copytree(w.root / "registration", froot / "registration")
    shutil.copyfile(w.plan, froot / "plan.yaml")
    receipt = froot / "registration" / w.receipt.name
    loader = W.Loader()
    c = c_reh(w, purpose="frame_diag", receipt=receipt, loader=loader, run_root=froot, folder=froot / "export",
              list_path=froot / "lists" / "gen.txt", plan_path=froot / "plan.yaml", adapter_dir=froot / "train" / "x",
              standin_allowed=True)
    code, d = d_main(w, capsys, registration="probe", purpose="frame_diag", receipt=receipt,
                     gen=froot / "lists" / "gen.txt", echo=froot / "lists" / "echo.txt", bundles=froot / "export",
                     train=froot / "train", out=froot, plan=froot / "plan.yaml")
    assert c == "entry_RECEIPT_KIND" and loader.calls == 0, c
    assert (code, d) == (SU.EXIT_ENTRY, "RECEIPT_KIND") and d_spy == {"gold": 0, "verify": 0}


# ---------------------------------------------------------------- 가-10
def test_가10_영수증_커밋이_본줄기의_조상이_아니면_두_쪽이_같은_사유로_거부한다(w, approved, d_spy, capsys):
    rp, lst, echo, _ = main_registration(w, ids=[W.EVAL_ID], branch="side")
    c = c_main(w, approved, receipt=rp, list_path=lst)
    code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo)
    assert code == SU.EXIT_ENTRY and d is not None
    _same(c, d)
    assert approved.calls == {"measure": 0, "loader": 0, "image": 0} and d_spy == {"gold": 0, "verify": 0}


def test_가10_작업_트리의_등록이_커밋과_달라도_두_쪽이_커밋의_것으로_돈다(w, approved, d_spy, capsys):
    """작업 트리를 다른 바이트로 덮는다 — 그것을 읽었다면 파일 해시에서 멈춘다. 두 쪽 다 다음 검사(분할)까지 간다."""
    rp, lst, echo, _ = main_registration(w)
    p = w.repo / MAIN_REG_REL
    p.write_bytes(p.read_bytes().replace(b'"left"', b'"right"'))
    c = c_main(w, approved, receipt=rp, list_path=lst)
    code, d = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo)
    assert (code, d) == (SU.EXIT_ENTRY, "LIST_SPLIT")
    _same(c, d)


# ---------------------------------------------------------------- 가-8(리허설/val)
def test_가8_리허설_val_묶음을_쓰는_쪽이_쓰고_읽는_쪽이_거부_없이_받는다(w, capsys):
    folder = w.root / "export"
    assert c_reh(w, folder=folder) is None
    assert c_reh(w, mode="echo", folder=folder, loader=echo_loader(w)) is None
    code, _ = d_main(w, capsys, registration="probe", purpose="rehearsal", receipt=w.receipt, gen=w.lists["model"],
                     echo=w.lists["echo"], bundles=folder, train=w.root / "train", plan=w.plan)
    score = json.loads((w.root / SU.OUT_SCORE).read_text(encoding="utf-8"))
    assert score["bundles"]["rejected"] == {}, score["bundles"]["rejected"]
    assert sorted(score["bundles"]["verified"]) == ["uni_local_C1_s1", "uni_local_C1_s1.echo"]
    echo = score["canaries"]["echo"][0]
    assert echo["status"] == "pass", echo
    assert all(x["status"] == "pass" for x in score["canaries"]["literal"])
    # 이음새가 대역(합성 실측 · 합성 생성기)이라 읽는 쪽은 통과를 내지 않고 '대역 실행' 으로 적는다 — 종료 2
    assert (code, score["status"]) == (SU.EXIT_REJECTED, "stand_in")


UNI_TAGS5 = ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_fed")


def test_다섯_모델_묶음의_축_배율을_평가_쪽이_쓴_파일_그대로_판정기가_읽는다(w, capsys):
    """다섯 모델 묶음을 쓰는 쪽의 실제 export 로 쓰고, 평가 쪽 진입점이 그 다섯에 대해 **직접 쓴** 본채점 파일을 손대지 않고
    판정기의 관측 함수(`axis_observation` — `judge` 가 쓰는 그것)에 넣는다(외부 검토 축 배율 회신의 2).
    **학습 산출물은 합성이다** — 연합 칸의 어댑터 · 원장은 세계가 꾸민 것이고 연합 학습을 돈 것이 아니다. 이 시험이 보는 것은
    묶음 → 평가 쪽 채점 → 파일 → 판정기의 연결이다. 연합 리허설 한 바퀴의 실제 산출이 아니다."""
    from vlm.rehearsal_judge import AXIS_STATUSES, axis_observation

    folder = w.root / "export"
    for tag in UNI_TAGS5:
        adir = W.write_adapter(w, tag=tag)
        assert c_reh(w, folder=folder, tag=tag, adapter_dir=adir) is None, tag
    assert c_reh(w, mode="echo", folder=folder, loader=echo_loader(w)) is None
    d_main(w, capsys, registration="probe", purpose="rehearsal", receipt=w.receipt, gen=w.lists["model"],
           echo=w.lists["echo"], bundles=folder, train=w.root / "train", plan=w.plan)
    path = w.root / SU.OUT_SCORE
    raw = path.read_bytes()
    score = json.loads(raw)
    stems = {f"{t}_s1" for t in UNI_TAGS5}
    assert score["bundles"]["rejected"] == {}, score["bundles"]["rejected"]
    assert set(score["bundles"]["verified"]) == stems | {"uni_local_C1_s1.echo"}
    obs = axis_observation(score["canaries"], stems)
    assert obs["called"] is True and obs["problems"] == [], obs
    assert {e["bundle"] for e in obs["entries"]} == stems and all(e["status"] in AXIS_STATUSES for e in obs["entries"])
    assert path.read_bytes() == raw                                   # 평가 쪽 파일을 고치지 않았다
    # 같은 파일에서 한 묶음을 빼면(다른 것은 그대로) 호출로 보지 않는다
    one_less = {**score["canaries"], "axis_ratio": [e for e in score["canaries"]["axis_ratio"] if e["bundle"] != "uni_fed_s1"]}
    assert axis_observation(one_less, stems)["called"] is False


# ---------------------------------------------------------------- 가-7
def test_가7_리허설_묶음을_옮겨_본실험_등록으로_검증하면_내용_검사가_거부한다(w):
    from evaluation.actuals import registered_values
    from evaluation.bundle_v14 import BundleFiles, Registration, verify_bundle
    from evaluation.reject_v14 import BundleRejected

    assert c_reh(w, folder=w.root / "export") is None
    moved = w.repo / "main_in" / "bundles"
    for q in (w.root / "export").iterdir():
        shutil.copyfile(q, moved / q.name)
    _, _, _, main = main_registration(w)
    el = validate_eval_list(canonical_bytes(list(W.IDS)))
    proj = Registration(generation_sha256=main.generation_sha256(), coupling_rule="decoupled_v2",
                        seeds={int(k): v for k, v in main.seeds.items()}, values=registered_values(main, mode="model"),
                        eval_list=el, purpose="main", echo_list=el, coord_space="ABS_ORIG", train_rows_digest=None)
    shas = {i: hashlib.sha256(p.read_bytes()).hexdigest() for i, p in w.images.items()}
    with pytest.raises(BundleRejected) as exc:
        verify_bundle(BundleFiles(moved / "uni_local_C1_s1.generations.jsonl"), registration=proj, artifact=None,
                      ledger=None, receipt_time=datetime(2026, 9, 21, 9, 0, tzinfo=W.KST), image_sha256_of=shas)
    codes = sorted(c.value for c in exc.value.codes())
    assert "generation_fingerprint_mismatch" in codes, codes          # 경로가 아니라 내용(생성 지문)이 막는다


# ---------------------------------------------------------------- 가-8(엄격/eval) · 가-7(진입점으로)
@pytest.fixture
def verify_log(monkeypatch):
    """평가 쪽 진입점 안의 묶음 검증을 감싸 결과를 적는다 — 검증은 실제 함수가 한다."""
    from evaluation.reject_v14 import BundleRejected

    real = SU.verify_bundle
    seen: list = []

    def wrap(files, **kw):
        try:
            vb = real(files, **kw)
        except BundleRejected as exc:
            seen.append(("rejected", files.stem, sorted(c.value for c in exc.codes()),
                         [getattr(r, "detail", None) for r in getattr(exc, "rejections", [])]))
            raise
        seen.append(("verified", files.stem, [], []))
        return vb

    monkeypatch.setattr(SU, "verify_bundle", wrap)
    return seen


def main_ready(w, s, *, train_config=True):
    """본실험 모델 export 가 지나는 세계 — 설정에 페어 · 생성 한도, 본실험 어댑터(참여자 행 전체의 digest), 승인 대역의 구현 열쇠."""
    from evaluation.actuals import impl_key
    from vlm.pilot_vlm import load_pairs, train_rows_digest
    from vlm.seams import impl_id

    pairs = w.tmp / "main_pairs" / "pairs.jsonl"
    pairs.parent.mkdir()
    pairs.write_text("".join(json.dumps({"image_id": f"t{k}", "image_path": "x.png", "client": "C1", "split": "train",
                                         "skeleton": {"defects": [], "verdict": "합격", "clauses": []}}) + "\n"
                             for k in range(3)), encoding="utf-8")
    cfg = w.repo / "configs" / "base.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("uni_max_new_tokens: null", "uni_max_new_tokens: 512")
                   + f"  uni_pairs: {{path: '{pairs.as_posix()}', digest: {'p' * 64}}}\n", encoding="utf-8")
    W.git(w.repo, "commit", "-q", "-am", "본실험 설정")
    rows = train_rows_digest(load_pairs("train", "C1", pairs_path=pairs))
    W.write_adapter(w, purpose="main", base=w.repo / "main_in" / "train", meta_over={"train_rows_digest": rows})
    keys = tuple(sorted(impl_key(impl_id(fn, seam=seam, approved=True))
                        for fn, seam in ((s.measure, "export_measure"), (s.loader, "export_generator"))))

    def edit(r):
        r.generation.impl_ids = keys

    return main_registration(w, ids=[W.EVAL_ID], edit=edit, train_config=train_config)


def test_가8_엄격_eval_묶음을_쓰는_쪽이_봉인하고_읽는_쪽이_거부_없이_검증한다(w, approved, verify_log, capsys, monkeypatch):
    """승인 판정은 두 쪽 모두 바꿔 끼운다 — 쓰는 쪽은 승인 목록을, 읽는 쪽은 구현 검사(`impl_problems`, 시험의 대역은 커밋되지 않은 코드다)를.
    엄격 경로가 나머지 내용 검사를 모두 지나는지 보려는 것이다."""
    import evaluation.bundle_v14 as BV

    monkeypatch.setattr(BV, "impl_problems", lambda v: [])
    rp, lst, echo, _ = main_ready(w, approved)
    folder = w.repo / "main_out" / "export"
    assert c_main(w, approved, receipt=rp, list_path=lst, folder=folder) is None
    assert (folder / "uni_local_C1_s1.export_meta.json").is_file()
    # 본실험 에코 — 영수증 커밋의 등록 옆 학습 설정 원문으로 봉인된다(평가 쪽 README 의 자리)
    assert c_main(w, approved, receipt=rp, list_path=echo, mode="echo", folder=folder) is None
    assert (folder / "uni_local_C1_s1.echo.export_meta.json").is_file()
    code, _ = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo, bundles=folder)
    assert verify_log == [("verified", "uni_local_C1_s1.echo", [], []), ("verified", "uni_local_C1_s1", [], [])], verify_log
    assert code == SU.EXIT_ENTRY                         # 에코 묶음 없이 카나리아에서, 그 뒤 채점 단계가 아직 없다


def test_가7_리허설_묶음을_옮겨_엄격_진입점에_넣으면_내용_검사가_거부한다(w, approved, verify_log, capsys):
    assert c_reh(w, folder=w.root / "export") is None
    moved = w.repo / "main_in" / "bundles"
    for q in (w.root / "export").iterdir():
        shutil.copyfile(q, moved / q.name)
    rp, lst, echo, _ = main_registration(w, ids=[W.EVAL_ID])
    code, _ = d_main(w, capsys, registration="strict", purpose="main", receipt=rp, gen=lst, echo=echo, bundles=moved)
    assert code == SU.EXIT_ENTRY
    rej = [x for x in verify_log if x[0] == "rejected"]
    # 경로가 아니라 내용이 막는다 — 먼저 걸리는 것은 줄 수다(2판 §5-2 의 1: 목록과 지문은 원인이 같다)
    assert rej and "line_count_mismatch" in rej[0][2], verify_log


# ---------------------------------------------------------------- 가-8(진단/val) · 진-1 · 진-2 · 진-4
DIAG_REG_REL = "configs/registration/frame_diag-20261001-1.json"


DIAG_RULES_REL = "configs/registration/frame_diag_rules-20261001-1.json"


def _diag_rules_file() -> bytes:
    """진단 규칙 파일 — 평가 쪽의 공개 함수로 정규형을 만든다. 시험용 문턱이다(등록 문턱이 아니다). 평가 쪽 진단 경로가
    등록이 가리킨 커밋에서 이 파일을 읽고 해시를 맞댄다 — 세계에 없으면 그 자리에서 멈춘다."""
    from evaluation import frame_diag as F
    from evaluation.frame_diag_rules import build_rules_file

    rules = F.FrameRules(n_img=10, s_min=0.7, w_s=0.1, w_r=0.0075, w_mix=0.0019, delta=0.0055, trim=0.0,
                         n_boot=40, boot_seed=1, min_pairs=10, min_groups=10, min_pair_groups=10)
    reqs = [{"noise_step": 0.1, "stat": "r", "axis": "y", "width": "w_r", "n0": 40, "n_required": 30, "g0": 20,
             "groups_required": 15, "q0": None, "pairs_required": None}]
    return build_rules_file(rules, sample_requirements=reqs, expected_sha256=W.hx("expected"),
                            cases_sha256=W.hx("cases"), procedure_sha256=W.hx("procedure"), noise_ladder=[0.1],
                            summary={}, sources={})


def diag_ready(w, *, edit=None, commit_to_main=False):
    """진단 루트(`fdiag-*`)와 **로컬 합성** 진단 등록 — 임시 저장소의 본줄기에 커밋한다(실제 본줄기의 조상이 아니다).
    `commit_to_main` 이면 영수증이 실제 저장소의 루트 커밋을 가리킨다(커밋된 진단 영수증).
    규칙 파일(시험용 문턱)을 등록과 같은 커밋에 두고 등록에 그 해시를 적는다."""
    import subprocess

    from evaluation.eval_list import validate_eval_list
    from vlm.exposure import append_row

    rules_raw = _diag_rules_file()
    (w.repo / DIAG_RULES_REL).parent.mkdir(parents=True, exist_ok=True)
    (w.repo / DIAG_RULES_REL).write_bytes(rules_raw)

    froot = w.parent / "fdiag-20261001T000000"
    (froot / "lists").mkdir(parents=True)
    raw = canonical_bytes(list(W.IDS))
    for name in ("gen.txt", "echo.txt"):
        (froot / "lists" / name).write_bytes(raw)
    el = validate_eval_list(raw)
    plan = froot / "plan.yaml"
    plan.write_text("kind: frame_diag\nversion: 1\ngen_limit: {max_new_tokens: 512, provisional: true}\n", encoding="utf-8")
    for kind in ("gen", "echo"):
        append_row(w.exposure, {"event": "planned", "run_id": froot.name, "purpose": "frame_diag", "tag": None,
                                "seed_index": 1, "mode": None, "list_kind": kind,
                                "list_path": (froot / "lists" / f"{kind}.txt").as_posix(), "list_file_sha256": el.file_sha256,
                                "list_set_sha256": el.set_sha256, "n": len(W.IDS), "snapshot_digest": w.digest,
                                "commit": w.head, "at": "2026-10-01T09:00:00.000000+09:00"})
    W.write_adapter(w, purpose="frame_diag", base=froot / "train")

    def fill(r):
        _canary(r)
        g = r.generation
        g.purpose, g.list_split = "frame_diag", "val"
        g.plan_sha256 = hashlib.sha256(plan.read_bytes()).hexdigest()
        g.frame_diag_rules_sha256 = hashlib.sha256(rules_raw).hexdigest()
        g.frame_diag_rules_path = DIAG_RULES_REL
        if edit is not None:
            edit(r)

    reg = W.registration(w, edit=fill)
    raw_reg = P.dump_registration(reg)
    (w.repo / DIAG_REG_REL).parent.mkdir(parents=True, exist_ok=True)
    (w.repo / DIAG_REG_REL).write_bytes(raw_reg)
    W.git(w.repo, "add", DIAG_REG_REL, DIAG_RULES_REL)
    _commit_train_config(w, DIAG_REG_REL)
    W.git(w.repo, "commit", "-q", "-m", "진단 등록")
    head = W.git(w.repo, "rev-parse", "HEAD")
    if commit_to_main:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "refs/heads/main"],
                              capture_output=True, check=True).stdout.decode().split()[-1]
    receipt = {"generation_sha256": reg.generation_sha256(), "scoring_sha256": reg.scoring_sha256(),
               "registered_at": "2026-09-21T09:00:00+09:00", "main_commit": head, "kind": "frame_diag",
               "registration_path": DIAG_REG_REL, "registration_file_sha256": hashlib.sha256(raw_reg).hexdigest()}
    rp = froot / "registration" / "frame_diag-20261001-1.receipt.json"
    rp.parent.mkdir(parents=True)
    rp.write_text(json.dumps(receipt), encoding="utf-8")
    return froot, rp, plan


def c_diag(w, froot, rp, plan, *, mode="model", **kw):
    try:
        W.run(w, purpose="frame_diag", mode=mode, receipt_path=rp, list_path=froot / "lists" / ("echo.txt" if mode == "echo" else "gen.txt"),
              plan_path=plan, run_root=froot, folder=froot / "export",
              adapter_dir=froot / "train" / "uni_local_C1_s1" if mode == "model" else None, standin_allowed=True, **kw)
    except ExportRefused as exc:
        return exc.code
    return None


def d_not_judged(froot) -> dict:
    """평가 쪽 진단 경로가 판정하지 않고 쓴 파일 — 사유와 묶음별 거부 코드."""
    body = json.loads((froot / "score" / "frame_diag_not_judged.json").read_text(encoding="utf-8"))
    return {"kind": body["kind"], "reason": body["reason"], "detail": body["detail"]}


#: 로컬 합성 진단 영수증의 대역 묶음 — 평가 쪽은 진단에서 대역 구현의 묶음을 거부하고(등록 항목 불일치) 판정하지 않는다
STANDIN_NOT_JUDGED = {"kind": "frame_diag_not_judged", "reason": "묶음이 거부됐다",
                      "detail": {"uni_local_C1_s1": ["registration_item_mismatch"]}}


def d_diag(w, capsys, froot, rp, plan):
    return d_main(w, capsys, registration="probe", purpose="frame_diag", receipt=rp, gen=froot / "lists" / "gen.txt",
                  echo=froot / "lists" / "echo.txt", bundles=froot / "export", train=froot / "train", out=froot, plan=plan)


def test_가8_진단_val_로컬_합성_영수증의_모델_묶음을_쓰는_쪽이_봉인한다(w, capsys):
    froot, rp, plan = diag_ready(w)
    assert c_diag(w, froot, rp, plan) is None
    assert (froot / "export" / "uni_local_C1_s1.export_meta.json").is_file()
    assert c_diag(w, froot, rp, plan, mode="echo") is None                          # 영수증 커밋의 학습 설정 원문으로
    assert d_diag(w, capsys, froot, rp, plan) == (SU.EXIT_REJECTED, None)
    # 에코 묶음도 같은 까닭(로컬 합성 영수증의 대역 구현)으로 거부된다 — 판정하지 않는다
    assert d_not_judged(froot) == {**STANDIN_NOT_JUDGED, "detail": {**STANDIN_NOT_JUDGED["detail"],
                                                                   "uni_local_C1_s1.echo": ["registration_item_mismatch"]}}


def test_진1_작업_트리의_진단_등록이_커밋과_달라도_쓰는_쪽은_커밋의_것으로_돈다(w, capsys):
    froot, rp, plan = diag_ready(w)
    p = w.repo / DIAG_REG_REL
    p.write_bytes(p.read_bytes().replace(b'"left"', b'"right"'))
    assert c_diag(w, froot, rp, plan) is None                          # 작업 트리를 읽었다면 파일 해시에서 멈춘다
    assert d_diag(w, capsys, froot, rp, plan) == (SU.EXIT_REJECTED, None)  # 읽는 쪽도 커밋의 등록으로 돌고 대역 묶음을 판정하지 않는다
    assert d_not_judged(froot) == STANDIN_NOT_JUDGED


def test_진2_진단_규칙의_칸이_빈_진단_등록은_시작하지_않는다(w, capsys):
    froot, rp, plan = diag_ready(w, edit=lambda r: setattr(r.generation, "frame_diag_rules_sha256", None))
    loader = W.Loader()
    code = c_diag(w, froot, rp, plan, loader=loader)
    assert code is not None and code.startswith("entry_") and loader.calls == 0, code
    assert not (froot / "export").exists()
    assert d_diag(w, capsys, froot, rp, plan) == (SU.EXIT_ENTRY, "REGISTRATION_INCOMPLETE")


def test_진4_커밋된_진단_영수증과_대역이면_대역을_부르기_전에_거부한다(w, capsys):
    froot, rp, plan = diag_ready(w, commit_to_main=True)
    calls, loader = [], W.Loader()
    code = c_diag(w, froot, rp, plan, loader=loader, measure_fn=lambda spec: calls.append(spec))
    assert code == "seam_on_committed_receipt" and calls == [] and loader.calls == 0
    assert d_diag(w, capsys, froot, rp, plan) == (SU.EXIT_ENTRY, "SEAM_ON_COMMITTED_RECEIPT")


# ================================================================ 커밋된 영수증의 에코 — 학습 설정 원문의 자리
def test_본실험_에코는_영수증_커밋에_없는_학습_설정_원문을_작업_트리에서_읽지_않는다(w, approved):
    """등록 옆의 원문이 커밋에 없고 **작업 트리에만** 있으면 거부한다 — 읽는 자리는 영수증 커밋이다(평가 쪽 README)."""
    from vlm.export_preflight import echo_train_config_rel

    rp, _lst, echo, _ = main_ready(w, approved, train_config=False)
    rel = echo_train_config_rel(MAIN_REG_REL)
    (w.repo / rel).write_bytes(W.train_config_text().encode("utf-8"))         # 커밋하지 않은 같은 바이트
    folder = w.repo / "main_out" / "export"
    assert c_main(w, approved, receipt=rp, list_path=echo, mode="echo", folder=folder) == "echo_train_config"
    assert not (folder / "uni_local_C1_s1.echo.export_meta.json").exists()


def test_본실험_에코는_등록과_다른_학습_설정_원문을_받지_않는다(w, approved):
    """커밋된 원문의 바이트가 등록의 `train_config_sha256` 과 다르면 실측 대조가 거부한다."""
    from vlm.export_preflight import echo_train_config_rel

    rp, _lst, echo, _ = main_ready(w, approved, train_config=False)
    rel = echo_train_config_rel(MAIN_REG_REL)
    (w.repo / rel).write_bytes(W.train_config_text().encode("utf-8") + b"\n")   # 끝 LF 하나 — 등록의 해시와 다르다
    W.git(w.repo, "add", rel)
    W.git(w.repo, "commit", "-q", "-m", "다른 원문")
    rc = json.loads(rp.read_text(encoding="utf-8"))
    rc["main_commit"] = W.git(w.repo, "rev-parse", "HEAD")
    rp.write_text(json.dumps(rc), encoding="utf-8")
    folder = w.repo / "main_out" / "export"
    assert c_main(w, approved, receipt=rp, list_path=echo, mode="echo", folder=folder) == "actuals"


def test_학습_설정_원문의_이름은_등록_줄기에서_낸다():
    from vlm.export_preflight import echo_train_config_rel
    from vlm.export_writer import ExportRefused

    assert echo_train_config_rel("configs/registration/main-20261001-1.json") == \
        "configs/registration/main-20261001-1.train_config.json"
    for bad in ("configs/registration/main-20261001-1.yaml", "configs/registration/x.train_config.json"):
        with pytest.raises(ExportRefused):
            echo_train_config_rel(bad)
