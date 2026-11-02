"""부분 B 결합 시험(평가 쪽 몫) — 생성 쪽의 실제 신판 표 · 로더 · 역할 파생 · 착수 고정 작성기를 평가 쪽 함수와 잇는다(대역 없음).

기대는 시험 전에 고정한 사례 파일 셋에서 읽는다 — 조항 축은 1 단계의 `clause_axis_roles.v1.json`(바이트 불변)의 신판 세계(edition2)와
그 사례들, 판 6 ↔ 계약 ↔ 착수 고정은 `partb_binding_cases.v1.json`, 실제 메타(PAIRS_META)는 `partb_meta_binding_cases.v1.json`.
표 · 영수증 · 실물의 경로와 sha256 은 그 파일들에 고정했다 — 없거나 바이트가 다르면 건너뛰지 않고 실패한다. 영수증은 트리의
정본 파일을 먼저 맞대고, Git 객체는 쓸 수 있을 때만 더 맞댄다 — 그래서 (나) 는 생성 쪽 커밋이 합쳐진 트리에서 선다.
모델 · GPU · 이미지 열기 0. 허용치 표의 수치를 시험에 옮겨 적지 않는다.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from corpus.rules import limits_loader
from corpus.rules.limit_eval import derive_clause_roles
from data import frozen_guard as FG
from evaluation import clause_axis as CA
from evaluation import narrative_registration as NR
from tests.test_prereg_v6 import v6
from vlm import pin as PIN
from vlm import uni_config as UC

REPO = Path(__file__).resolve().parents[1]
CASES_REL = "evaluation/vectors/partb_binding_cases.v1.json"
CASES_SHA256 = "827ca571b9a62bf65ddc1f40eebe9232b7d3977f9efa3026f8a143ebfada11cc"
"""시험 전에 고정한 바이트 — 기대를 바꾸려면 새 판의 파일을 낸다."""
CASES = json.loads((REPO / CASES_REL).read_bytes().decode("utf-8"))
RB = CASES["roles_binding"]
ROLES_CASES = json.loads((REPO / RB["against"]["cases"]).read_bytes().decode("utf-8"))
WORLD = ROLES_CASES["worlds"][RB["against"]["world"]]
V6 = CASES["v6_function_level"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def codes(problems: list[str]) -> list[str]:
    assert all(isinstance(p, str) and ": " in p for p in problems), problems
    return sorted({p.split(":", 1)[0] for p in problems})


# ── 조항 축 — 실제 신판 표 → 역할 → 평가 쪽 이미지 단위 함수 ────────────────────────────────

@pytest.fixture(scope="module")
def real_roles() -> dict:
    path = REPO / RB["table"]["path"]
    assert path.is_file(), f"생성 쪽 신판 표가 없다: {RB['table']['path']}"           # 건너뛰지 않는다
    assert sha(path.read_bytes()) == RB["table"]["sha256"], "신판 표의 바이트가 사례 파일에 고정한 것과 다르다"
    table = limits_loader.load_limits(path, pilot=True)
    derived = derive_clause_roles(table, method="RT")
    return {"table": table, "derived": derived, "roles": CA.roles_from_derived(derived)}


def world_roles() -> dict:
    derived = {(r["material"], r["code"]): {"main": r["main"], "auxiliary": [tuple(a) for a in r["auxiliary"]],
                                            "defect_class": None, "conditions": ()} for r in WORLD["roles"]}
    return CA.roles_from_derived(derived)


def test_사례_파일은_시험_전에_고정한_바이트다() -> None:
    assert sha((REPO / CASES_REL).read_bytes()) == CASES_SHA256
    assert CASES["kind"] == "partb_binding_cases" and CASES["version"] == 1
    for src in ("roles_cases_v1", "inputs_cases_v1"):
        s = CASES["sources"][src]
        assert sha((REPO / s["path"]).read_bytes()) == s["sha256"], src


def test_실제_역할은_사례의_신판_세계와_같다(real_roles: dict) -> None:
    roles, e = real_roles["roles"], RB["expect"]
    nonempty = {k: v for k, v in roles.items() if v.main is not None or v.auxiliary}
    assert (nonempty == world_roles()) is e["nonempty_roles_equal_world"]
    empty_materials = sorted({m for (m, _), v in roles.items() if v.main is None and not v.auxiliary}
                             - {m for (m, _) in nonempty})
    assert empty_materials == e["materials_without_rows"]
    assert sum(v.main is None and bool(v.auxiliary) for v in roles.values()) == e["primary_none_with_auxiliary"]
    index = sorted({r.clause_id for r in real_roles["table"].rows})
    assert (index == sorted(WORLD["index"])) is e["index_equal_world"]


def test_표의_코드_조합마다_정답_쌍_보조_전용_대조선이_사례의_신판_세계와_같다(real_roles: dict) -> None:
    real, world = real_roles["roles"], world_roles()
    by_material: dict[str, list[str]] = {}
    for m, c in sorted(real):
        by_material.setdefault(m, []).append(c)
    n = 0
    for material, codes_of in by_material.items():
        for k in range(1, len(codes_of) + 1):
            for subset in itertools.combinations(codes_of, k):
                mat, gt = {"i": material}, {"i": list(subset)}
                assert CA.gold_pairs_from_roles(real, mat, gt) == CA.gold_pairs_from_roles(world, mat, gt), (material, subset)
                assert CA.auxiliary_only_by_image(real, mat, gt) == CA.auxiliary_only_by_image(world, mat, gt), (material, subset)
                assert CA.citations_from_roles(real, mat, gt) == CA.citations_from_roles(world, mat, gt), (material, subset)
                n += 1
    assert RB["expect"]["every_code_subset_equal_world"] and n == sum(2 ** len(v) - 1 for v in by_material.values())


def counts_with(roles: dict, images: list[dict]) -> dict:
    material_of = {im["image"]: im["material"] for im in images}
    gt = {im["image"]: im["gt_codes"] for im in images}
    cited = {im["image"]: im["cited"] for im in images}
    pairs, skipped = CA.gold_pairs_from_roles(roles, material_of, gt)
    aux = CA.auxiliary_only_by_image(roles, material_of, gt)
    rep = CA.clause_axis_report(cited, pairs, WORLD["index"], auxiliary_only=aux)
    return {"n_gold_pairs": rep["citation_match_rate"]["denominator"], "match_hits": rep["citation_match_rate"]["numerator"],
            "skipped": skipped, "n_citations": rep["n_citations"], "correct": rep["citation_precision"]["numerator"],
            "nonexistent": rep["ungrounded_rate"]["nonexistent"], "irrelevant": rep["ungrounded_rate"]["irrelevant"],
            "auxiliary_only": rep["ungrounded_rate"]["auxiliary_only"]}


EDITION2 = [c for c in ROLES_CASES["cases"] if c["images"]["edition2"] is not None]


@pytest.mark.parametrize("case", EDITION2, ids=[c["id"] for c in EDITION2])
def test_사례의_신판_이미지를_실제_역할로_재도_기대와_같다(case: dict, real_roles: dict) -> None:
    assert RB["expect"]["edition2_cases_equal_expect"]
    assert counts_with(real_roles["roles"], case["images"]["edition2"]) == case["expect"]


def test_대조선_사례를_실제_역할로_재도_기대와_같다(real_roles: dict) -> None:
    case = ROLES_CASES["baseline_case"]
    got = CA.citations_from_roles(real_roles["roles"], case["material_of"], case["codes_by_image"]["edition2"])
    assert RB["expect"]["baseline_edition2_equal_expect"] and got == case["expect"]["edition2"]


# ── 판 6 ↔ 계약 ↔ 착수 고정 — 평가 쪽 작성기 + 생성 쪽 착수 고정 작성기(함수 수준) ───────────────

def contract_of(label_key: str) -> tuple[dict, bytes]:
    c = V6["contract"]
    block = NR.narrative_inputs_block(REPO, limits_rel=c["limits_rel"], pairs_path=c["pairs_path"],
                                      snapshot_digest=sha(c[label_key].encode("utf-8")),
                                      pairs_meta_raw=c["pairs_meta_label"].encode("utf-8"))
    return block, NR.dump_narrative_inputs(block)


def write_start_pin(path: Path, contract_raw: bytes, *, driver: str = "합성") -> str:
    pin = {"schema": PIN.PIN_SCHEMA, "code": {"dirty": [], "commit": "합성"}, "env": {"driver": driver},
           "common": {"registration": {NR.INPUTS_REGISTRATION_KEY: {"version": NR.INPUTS_CONTRACT_VERSION,
                                                                  "sha256": sha(contract_raw)}}},
           "init_adapter": {"init_adapter_digest": "합성"}}
    return PIN.write_pin(path, pin)


def registration_with(contract_path: Path, conf: Path):
    return replace(v6(), narrative=NR.narrative_block(REPO, confirmation_path=conf, checker_vectors_path=conf,
                                                      inputs_contract_path=contract_path),
                   clause_axis=NR.clause_axis_block(REPO))


def align_identity(reg, contract: dict):
    return replace(reg, narrative=replace(reg.narrative, limits_sha256=contract["limits"]["sha256"],
                                          limits_loader_sha256=contract["limits_loader"]["sha256"],
                                          generator_sha256=contract["generator"]["sha256"]),
                   clause_axis=replace(reg.clause_axis, limits_sha256=contract["limits"]["sha256"]))


def build(case: dict, tmp: Path) -> tuple:
    conf = tmp / "synthetic_confirmation_cases.json"
    conf.write_bytes("결합 시험 합성 확인 사례".encode())
    a, a_raw = contract_of("snapshot_label")
    b, b_raw = contract_of("other_snapshot_label")
    (tmp / "a.json").write_bytes(a_raw)
    (tmp / "b.json").write_bytes(b_raw)
    psha = write_start_pin(tmp / "PIN.json", a_raw)                      # 학습은 계약 갑에 묶였다
    _, read_sha = PIN.read_pin(tmp / "PIN.json")
    assert read_sha == psha                                              # 생성 쪽 읽개와 같은 sha
    pin_raw, adapters = (tmp / "PIN.json").read_bytes(), [psha] * 5
    reg, contract, passed = registration_with(tmp / "a.json", conf), a, a_raw
    ops = case["ops"]
    if "reissue_with_other_contract" in ops:
        reg, contract, passed = registration_with(tmp / "b.json", conf), b, b_raw
    if "align_identity" in ops:
        reg = align_identity(reg, contract)
    if "pass_other_contract" in ops:
        passed = b_raw
    if "other_pin" in ops:
        write_start_pin(tmp / "PIN_other.json", a_raw, driver="합성 다른 판")
        pin_raw = (tmp / "PIN_other.json").read_bytes()
    if "no_adapters" in ops:
        adapters = []
    if "one_adapter_other" in ops:
        adapters = [psha] * 4 + [sha("다른 착수 고정".encode())]
    return reg, passed, pin_raw, adapters


@pytest.mark.parametrize("case", V6["cases"], ids=lambda c: c["id"])
def test_판_6_계약_착수_고정은_사례의_사유_코드를_낸다(case: dict, tmp_path: Path) -> None:
    assert set(case["ops"]) <= set(V6["ops_meaning"])
    reg, contract_raw, pin_raw, adapters = build(case, tmp_path)
    got = NR.v6_contract_problems(reg, contract_raw, pin_raw=pin_raw, adapter_pin_sha256=adapters)
    assert codes(got) == case["expect"], got
    for label in case.get("labels", []):                                 # 지금 작성기가 어긋나는 칸이 바로 그 칸인지
        assert any(label in p for p in got), (label, got)
    if case.get("labels"):
        assert len(got) == len(case["labels"]), got


def test_작성기는_계약_파일의_sha_를_판_6_에_적는다(tmp_path: Path) -> None:
    _, a_raw = contract_of("snapshot_label")
    (tmp_path / "a.json").write_bytes(a_raw)
    (tmp_path / "conf.json").write_bytes("합성".encode())
    reg = registration_with(tmp_path / "a.json", tmp_path / "conf.json")
    assert reg.narrative.inputs_contract_sha256 == sha(a_raw)
    assert reg.clause_axis.multi_clause == "primary_one_auxiliary_breakdown"


# ── (나) 실제 메타 — 생성 쪽 봉인 페어 2 판 의 PAIRS_META → 평가 쪽 계약 작성기 → 평가 쪽 학습 전 대조 ──────────────────

META_REL = "evaluation/vectors/partb_meta_binding_cases.v1.json"
META_SHA256 = "ef07ed9a3f26f8a8dc52012b280e9eea11bce3f02d4af1f0dc9095e4fa77b737"
"""시험 전에 고정한 바이트 — 기대를 바꾸려면 새 판의 파일을 낸다."""
META = json.loads((REPO / META_REL).read_bytes().decode("utf-8"))


def git_root_is(repo: Path) -> bool:
    """repo 가 그 자체로 Git 작업 트리의 뿌리인가 — 상위 저장소(사본을 품은 바깥 트리)의 객체는 빌려 쓰지 않는다."""
    try:
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--show-toplevel"], capture_output=True, check=False)
    except OSError:                                                        # Git 실행 파일이 없다
        return False
    if p.returncode != 0:
        return False
    try:
        return Path(p.stdout.decode("utf-8").strip()).samefile(repo)
    except OSError:
        return False


def receipt_bytes(src: dict | None = None, repo: Path = REPO) -> tuple[bytes, bool]:
    """생성 쪽 영수증 — 트리의 정본 파일을 사례 파일에 고정한 sha 로 먼저 맞댄다. 없거나 sha 가 다르면 실패다(건너뜀 · 대체 영수증 · 다른 사본 없음).

    그 트리가 Git 작업 트리의 뿌리이고 사례 파일이 적은 커밋 객체가 있을 때만 그 커밋의 blob 과도 바이트로 맞댄다 — 둘 다 있으면 둘 다
    같아야 한다. git archive 사본처럼 객체가 없으면 그 대조만 하지 않는다. 둘째 값은 blob 대조를 했는지다.
    """
    src = META["sources"]["receipt"] if src is None else src
    tree = repo / src["path"]
    assert tree.is_file(), f"트리에 생성 쪽 영수증이 없다 — 생성 쪽 커밋이 합쳐진 트리(통합 가지)에서 돈다: {src['path']}"
    raw = tree.read_bytes()
    assert sha(raw) == src["sha256"], "트리의 생성 쪽 영수증이 사례 파일에 고정한 sha 와 다르다"
    if not git_root_is(repo):
        return raw, False
    have = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{src['commit']}^{{commit}}"], capture_output=True, check=False)
    if have.returncode != 0:                                               # 이 저장소에 그 커밋 객체가 없다
        return raw, False
    p = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", f"{src['commit']}:{src['path']}"], capture_output=True,
                       check=False)
    assert p.returncode == 0, f"커밋에 생성 쪽 영수증 blob 이 없다 — {src['commit'][:8]}:{src['path']}"
    assert p.stdout == raw, "트리의 생성 쪽 영수증이 커밋의 blob 과 다르다"
    return raw, True


@pytest.fixture(scope="module")
def sealed_pairs() -> dict:
    """최종 봉인 파일을 고정 지문 및 실제 스냅샷 계약과 맞댄다. 명부 상태도 실제 값을 쓴다."""
    want = META["sealed_pairs"]
    folder = REPO / want["path"]
    meta_p, snap_p = folder / "PAIRS_META.json", folder / "SNAPSHOT.sha256"
    assert meta_p.is_file() and snap_p.is_file(), f"봉인 페어 2 판 이 이 트리에 없다 — 자산 부재를 먼저 의심한다: {want['path']}"
    meta_raw, snap_raw = meta_p.read_bytes(), snap_p.read_bytes()
    assert sha(meta_raw) == want["pairs_meta_sha256"], "봉인 페어 2 판 PAIRS_META.json 이 영수증의 sha 와 다르다"
    assert sha(snap_raw) == want["snapshot_file_sha256"], "봉인 페어 2 판 SNAPSHOT.sha256 이 영수증의 sha 와 다르다"
    lines = snap_raw.decode("utf-8").splitlines()
    digest = next(ln.split()[-1] for ln in lines if ln.startswith("# snapshot_digest"))
    members: dict[str, str] = {}
    for ln in lines:
        if ln.strip() and not ln.startswith("#"):
            h, name = ln.split(None, 1)
            members[name.strip()] = h
    assert digest == want["snapshot_digest"]
    assert members["PAIRS_META.json"] == sha(meta_raw)                   # 스냅샷이 적은 메타 sha = 실제 메타
    return {"meta_raw": meta_raw, "digest": digest, "snap_meta_sha": members["PAIRS_META.json"],
            "problems": tuple(UC.pairs_snapshot_problems(folder / "pairs.jsonl", digest))}


def edit_meta(meta: dict, ops: list[dict]) -> dict:
    m = copy.deepcopy(meta)
    for op in ops:
        if "set" in op:
            *parent, key = op["set"]
            node = m
            for k in parent:
                node = node[k]
            node[key] = op["value"] if "value" in op else sha(op["sha_of_text"].encode("utf-8"))
        elif "move" in op:
            *src_parent, src_key = op["move"]
            *dst_parent, dst_key = op["to"]
            s = m
            for k in src_parent:
                s = s[k]
            d = m
            for k in dst_parent:
                d = d[k]
            d[dst_key] = s.pop(src_key)
    return m


def test_메타_사례_파일은_시험_전에_고정한_바이트다() -> None:
    assert sha((REPO / META_REL).read_bytes()) == META_SHA256
    assert META["kind"] == "partb_meta_binding_cases" and META["version"] == 1
    for src in ("inputs_cases_v1", "binding_cases_v1"):
        s = META["sources"][src]
        assert sha((REPO / s["path"]).read_bytes()) == s["sha256"], src


@pytest.mark.parametrize("case", META["cases"], ids=lambda c: c["id"])
def test_실제_sealed_pairs_메타로_지은_계약을_평가_쪽_대조가_사례대로_받고_거부한다(case: dict, sealed_pairs: dict, tmp_path: Path) -> None:
    assert sealed_pairs["problems"] == (), sealed_pairs["problems"]                          # 실제 스냅샷 검사 — 파일 해시를 다시 쟀다
    sealed_status = FG.EXPECTED_SEALED[META["sealed_pairs"]["path"]]["status"]
    assert sealed_status == NR.SEALED_EXPECTED
    meta = json.loads(sealed_pairs["meta_raw"].decode("utf-8"))
    table_rel = meta["limits_table"]["path"]
    root = tmp_path / "world"
    for rel in (*NR.LIMITS_LOADER_FILES, *NR.GENERATOR_FILES_V1, table_rel):  # 표 · 로더 · 생성 경로는 저장소의 실제 바이트 사본
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, root / rel)
    meta_raw, snap_meta_sha = sealed_pairs["meta_raw"], sealed_pairs["snap_meta_sha"]
    meta_ops = [op for op in case["ops"] if "set" in op or "move" in op]
    if meta_ops:  # 호출자가 검증했다고 주는 사실을 합성한다 — 메타 sha · 계약만 새 바이트에 맞추고 digest · 스냅샷 검사는 원래 봉인 페어 2 판 값
        meta_raw = (json.dumps(edit_meta(meta, meta_ops), ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        snap_meta_sha = sha(meta_raw)
    contract_raw = NR.dump_narrative_inputs(NR.narrative_inputs_block(
        root, limits_rel=table_rel, pairs_path=META["sealed_pairs"]["path"], snapshot_digest=sealed_pairs["digest"], pairs_meta_raw=meta_raw))
    for op in case["ops"]:
        if "edit_table_after_contract" in op:
            p = root / table_rel
            text = p.read_bytes().decode("utf-8")
            assert op["token"] in text
            p.write_bytes(text.replace(op["token"], "", 1).encode("utf-8"))
    pairs = NR.VerifiedPairs(path=META["sealed_pairs"]["path"], snapshot_digest=sealed_pairs["digest"], snapshot_problems=sealed_pairs["problems"],
                             sealed_status=sealed_status, pairs_meta_raw=meta_raw, snapshot_pairs_meta_sha256=snap_meta_sha,
                             config_pairs_digest=sealed_pairs["digest"], token_table_pairs_digest=sealed_pairs["digest"])
    got = NR.narrative_inputs_problems(contract_raw, registered_sha256=sha(contract_raw), repo=root, pairs=pairs)
    assert codes(got) == case["expect"], got


def test_실제_sealed_pairs_메타는_I_02_의_꼴이다(sealed_pairs: dict) -> None:
    """status 는 limits_table 제자리에 남고 identity.limits 는 그 두 칸의 닫힌 객체다 — 영수증이 적은 값과 같다."""
    meta = json.loads(sealed_pairs["meta_raw"].decode("utf-8"))
    assert meta["limits_table"].get("status")
    assert meta["identity"]["limits"] == {k: meta["limits_table"][k] for k in ("path", "sha256")} == META["sealed_pairs"]["identity_limits"]


# ── 영수증 읽기의 실행 계약 — 트리의 정본 파일이 먼저, Git 객체는 쓸 수 있을 때만 더한다(git archive 사본 · 상위 저장소 안의 사본) ──

DEMO_RECEIPT = b'{"demo": "receipt"}\n'
"""합성 영수증 — 실제 영수증 · b1′ 는 쓰지 않는다."""


def git_demo(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false",
                           "-c", "core.autocrlf=false", *args], capture_output=True, check=True).stdout.decode("utf-8").strip()


def demo_src(commit: str = "0" * 40, data: bytes = DEMO_RECEIPT) -> dict:
    return {"commit": commit, "path": "docs/receipt.json", "sha256": sha(data)}


def put_receipt(root: Path, data: bytes = DEMO_RECEIPT) -> None:
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs/receipt.json").write_bytes(data)


def committed_demo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    git_demo(repo, "init", "-q")
    put_receipt(repo)
    git_demo(repo, "add", "docs/receipt.json")
    git_demo(repo, "commit", "-q", "-m", "receipt")
    return git_demo(repo, "rev-parse", "HEAD")


def test_영수증은_Git_없는_사본에서_트리의_정본_파일로_읽는다(tmp_path: Path) -> None:
    put_receipt(tmp_path)
    assert receipt_bytes(demo_src(), tmp_path) == (DEMO_RECEIPT, False)


@pytest.mark.parametrize("data", [None, DEMO_RECEIPT + b" "], ids=["없음", "바이트_다름"])
def test_영수증이_트리에_없거나_sha_가_다르면_실패한다(data: bytes | None, tmp_path: Path) -> None:
    if data is not None:
        put_receipt(tmp_path, data)
    with pytest.raises(AssertionError, match="트리에 생성 쪽 영수증이 없다" if data is None else "고정한 sha 와 다르다"):
        receipt_bytes(demo_src(), tmp_path)


def test_상위_저장소의_객체로_영수증을_대신하지_않는다(tmp_path: Path) -> None:
    head = committed_demo(tmp_path)
    copy_root = tmp_path / "copy"                                          # 상위 저장소 안에 푼 git archive 사본의 꼴
    copy_root.mkdir()
    with pytest.raises(AssertionError, match="트리에 생성 쪽 영수증이 없다"):  # 상위의 blob 이 있어도 사본 트리에 없으면 실패
        receipt_bytes(demo_src(head), copy_root)
    put_receipt(copy_root)
    assert receipt_bytes(demo_src(head), copy_root) == (DEMO_RECEIPT, False)  # 상위 저장소의 객체는 쓰지 않는다


def test_저장소의_뿌리이고_커밋이_있으면_blob_과도_맞댄다(tmp_path: Path) -> None:
    head = committed_demo(tmp_path / "repo")
    assert receipt_bytes(demo_src(head), tmp_path / "repo") == (DEMO_RECEIPT, True)
    other = b'{"demo": "other"}\n'
    put_receipt(tmp_path / "repo", other)                                  # 트리를 바꾸고 고정 sha 도 거기에 맞춰도 blob 과 다르면 실패
    with pytest.raises(AssertionError, match="커밋의 blob 과 다르다"):
        receipt_bytes(demo_src(head, other), tmp_path / "repo")


def test_커밋_객체가_없는_저장소에서는_blob_대조를_하지_않는다(tmp_path: Path) -> None:
    committed_demo(tmp_path / "repo")                                      # 사본을 git init 한 교차 검수 꼴 — 그 커밋은 없다
    assert receipt_bytes(demo_src("1" * 40), tmp_path / "repo") == (DEMO_RECEIPT, False)
