"""공통 학습 예산 비율(`vlm.budget`, 등록 `budget_fraction`) — 모델 없이. 1.0 이면 지금과 같고, 1 보다 작으면 참여자마다 고정 몫."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import yaml

from vlm import budget as B
from vlm import registration as R

ROOT = Path(__file__).resolve().parents[1]


def _pairs(tmp: Path, sizes=None) -> Path:
    sizes = sizes or {"C1": 1000, "C2": 300, "C3": 77}
    rows = []
    for c, n in sizes.items():
        for i in range(n):
            rows.append({"image_id": f"synth:{c}:{i:05d}", "client": c, "split": "train"})
        rows.append({"image_id": f"synth:{c}:val", "client": c, "split": "val"})
    p = tmp / "pairs.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _rows(p: Path, client=None):
    from vlm.pilot_vlm import load_pairs

    return load_pairs("train", client=client, pairs_path=p)


def test_비율_1_이면_행_신원_기대값이_지금과_같다(tmp_path):
    p = _pairs(tmp_path)
    rows = _rows(p, "C1")
    assert B.apply_budget(rows, fraction=1.0, seed_value=20260828, pairs_path=p) is rows
    assert B.budget_identity(1.0) == {}
    exp = {"C1": 10, "C2": 20, "C3": 30}
    assert B.expected_tokens(exp, fraction=1.0, seed_value=1, pairs_path=p) == exp
    assert B.expected_tokens(None, fraction=0.5, seed_value=1, pairs_path=p) is None


@pytest.mark.parametrize("f", [0.125, 0.5, 0.25])
def test_참여자마다_갱신_수가_식대로이고_칸끼리_같은_행이다(tmp_path, f):
    p = _pairs(tmp_path)
    w = B._window()                                                       # 등록 창(마이크로 배치 × 누적) — 상수를 직접 읽는다
    picked = {}
    for c, n in (("C1", 1000), ("C2", 300), ("C3", 77)):
        sub = B.apply_budget(_rows(p, c), fraction=f, seed_value=20260828, pairs_path=p)
        k = min(n, math.ceil(f * n / w) * w)
        assert len(sub) == k and math.ceil(len(sub) / w) == math.ceil(f * n / w) == B.budget_steps(n, f)
        assert [r["image_id"] for r in sub] == sorted(r["image_id"] for r in sub)        # 파일 순서로 되돌린다
        picked[c] = {r["image_id"] for r in sub}
    central = B.apply_budget(_rows(p), fraction=f, seed_value=20260828, pairs_path=p)
    assert {r["image_id"] for r in central} == set().union(*picked.values())             # 중앙 = 세 몫의 합집합
    again = B.apply_budget(_rows(p, "C1"), fraction=f, seed_value=20260828, pairs_path=p)
    assert {r["image_id"] for r in again} == picked["C1"]                                 # 결정적
    other = B.apply_budget(_rows(p, "C1"), fraction=f, seed_value=20260829, pairs_path=p)
    assert {r["image_id"] for r in other} != picked["C1"]                                 # 시드가 고른다


def test_일부_행에는_걸지_않는다(tmp_path):
    p = _pairs(tmp_path)
    with pytest.raises(B.BudgetRejected, match="budget_rows"):
        B.apply_budget(_rows(p, "C1")[:500], fraction=0.5, seed_value=1, pairs_path=p)


def test_감독_토큰_기대값은_고른_행의_합이다(tmp_path):
    p = _pairs(tmp_path)
    table = tmp_path / "per_row.csv"
    lines = ["image_id,client,n_defects,supervised_tokens"]
    per = {"C1": 0, "C2": 0, "C3": 0}
    tok = {}
    for r in _rows(p):
        t = 21 + len(r["image_id"]) % 5
        tok[r["image_id"]] = t
        per[r["client"]] += t
        lines.append(f'{r["image_id"]},{r["client"]},0,{t}')
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    got = B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=p, table_path=table)
    for c in per:
        ids = {r["image_id"] for r in B.apply_budget(_rows(p, c), fraction=0.125, seed_value=7, pairs_path=p)}
        assert got[c] == sum(tok[i] for i in ids)
    with pytest.raises(B.BudgetRejected, match="token_table_mismatch"):
        B.expected_tokens({**per, "C1": per["C1"] + 1}, fraction=0.125, seed_value=7, pairs_path=p, table_path=table)


@pytest.mark.parametrize("bad", [0, 0.0, 1.5, -0.1, True, "0.125", float("nan")])
def test_비율의_꼴(bad):
    with pytest.raises(B.BudgetRejected):
        B.check_fraction(bad)


def test_등록은_예산_칸이_없으면_1_이고_있으면_신원에_든다(tmp_path):
    base = yaml.safe_load((Path(R.REPO_ROOT) / "configs/train_registration.yaml").read_text(encoding="utf-8"))
    base.pop("budget_fraction", None)
    p0 = tmp_path / "r0.yaml"
    p0.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    reg0, sha0 = R.load_registration(p0)
    assert R.budget_fraction_of(reg0) == 1.0
    p1 = tmp_path / "r1.yaml"
    p1.write_text(yaml.safe_dump({**base, "budget_fraction": 0.125}, allow_unicode=True), encoding="utf-8")
    reg1, sha1 = R.load_registration(p1)
    assert R.budget_fraction_of(reg1) == 0.125 and sha1 != sha0                          # 등록 지문(착수 고정의 common)이 바뀐다
    p2 = tmp_path / "r2.yaml"
    p2.write_text(yaml.safe_dump({**base, "budget_fractions": 0.125}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(R.RegistrationRejected):
        R.load_registration(p2)


def test_칸_신원에_비율이_들고_1_이면_없다():
    from vlm.train_cell import _run_identity

    a = _run_identity(_spec(1.0), "uni_central", client_idx=0, rows_digest="x", n_rows=3, prompt_sha256="p", init_digest="i")
    b = _run_identity(_spec(0.125), "uni_central", client_idx=0, rows_digest="x", n_rows=3, prompt_sha256="p", init_digest="i")
    assert "budget_fraction" not in a and b["budget_fraction"] == 0.125 and b["budget_policy"] == B.BUDGET_POLICY
    assert {k: v for k, v in b.items() if k not in ("budget_fraction", "budget_policy")} == a


def _spec(f):
    from vlm.train_cell import UniRunSpec

    return UniRunSpec(purpose="main", run_stamp="main_s1", seed_index=1, seed_value=20260828, snapshot_digest="s",
                      model_id="m", model_revision="r", pairs_path="p.jsonl", pairs_digest="d", chat_template_kwargs={},
                      num_rounds=1, local_epochs=1, total_epochs=1, train_root=Path("o"), init_adapter_path=Path("o/i.npz"),
                      budget_fraction=f)


def test_명세는_비율의_꼴을_본다():
    from vlm.train_cell import CellRejected

    with pytest.raises(CellRejected, match="budget_fraction"):
        _spec(0)


def test_연합_설정_키(tmp_path):
    from fl.uni_run_config import UNI_RUN_DEFAULTS, client_run_cfg

    raw = dict(UNI_RUN_DEFAULTS)
    assert client_run_cfg({**raw, "purpose": "rehearsal"})["budget_fraction"] == 1.0
    with pytest.raises(ValueError, match="본실험만"):
        client_run_cfg({**raw, "purpose": "rehearsal", "uni-budget-fraction": "0.5"})
    with pytest.raises(ValueError, match="uni-budget-fraction"):
        client_run_cfg({**raw, "purpose": "rehearsal", "uni-budget-fraction": "1.5"})


# ================================================================ 레시피별 행별 토큰 표(02 2판 §3-5 의 새 시험 9, 지시 1022b)
V2 = "json_line_then_basis"
A02 = ROOT / "outputs/diag_20261027/02a_sealed.json"
A02_SHA256 = "d1d06d7d42cd1e6987f4ec3a0bc5f57b0a578061e80d2fa85f3a7f78f46fe9e3"
SEALED_TABLE = Path("outputs/target_tokens_v3_20261027/per_row.csv")
SEALED_TABLE_SHA256 = "295603f08002e17ae6f3de0387f85fea74968abbf2e9233955792cb82788c8f1"


def test_1차_표_자리와_지문은_그대로이고_레시피가_표를_고른다():
    """1 차 표는 봉인 폴더 안이라 **열지 않는다** — 상수만 맞댄다."""
    assert B.TOKEN_TABLE == Path("outputs/B_seal_20261001/tokens/per_row.csv")
    assert B.TOKEN_TABLE_SHA256 == "c88155c48c7c9a5e3ce226a3ae5441d78373c13c7e98afb6116053ba6371a62b"
    assert B.token_table_of("json_only") == (B.TOKEN_TABLE, B.TOKEN_TABLE_SHA256)
    rel, sha = B.token_table_of(V2)
    assert "B_seal_20261001" not in rel.as_posix() and sha != B.TOKEN_TABLE_SHA256 and len(sha) == 64
    from vlm.registration import TRAIN_RECIPES

    assert tuple(B.TOKEN_TABLES) == TRAIN_RECIPES
    with pytest.raises(B.BudgetRejected) as exc:
        B.token_table_of("two_task_mix")
    assert exc.value.code == "recipe_unknown"


def test_1차_레시피는_비율_1_이면_표를_읽지_않는다(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("1 차 레시피 · f = 1 은 표를 읽지 않는다")

    monkeypatch.setattr(B, "_token_table", boom)
    per = {"C1": 10, "C2": 20}
    assert B.expected_tokens(per, fraction=1.0, seed_value=7) == per
    assert B.expected_tokens(per, fraction=1.0, seed_value=7, recipe="json_only") == per


#: 합성 토크나이저 환경(`tests.uni_fakes.fake_tokenizer_env`)의 모델 — 출처의 신원 · 지금 환경의 신원을 이 모델로 잰다.
MODEL_KW = {"model_id": "tiny", "model_revision": "r1"}


def _v2_table(tmp: Path, monkeypatch, *, prov_over=None, drop=False):
    """합성 새 레시피 표와 출처 파일. 표 sha 칸은 고정값(시험은 표 바이트 지문을 주지 않는다 — 출처 대조만 본다). 출처의 꼴 · 계측
    코드 · 토크나이저 신원은 지금 코드와 합성 토크나이저 환경(`fake_tokenizer_env`, 모델 tiny@r1)의 값이다."""
    from tests.uni_fakes import fake_tokenizer_env
    from vlm import token_table as T
    from vlm.pilot_vlm import target_contract_sha256

    fake_tokenizer_env(monkeypatch, tmp)
    p = _pairs(tmp)
    lines, per = ["image_id,client,n_defects,supervised_tokens"], {}
    for r in _rows(p):
        t = 30 + int(r["image_id"].rsplit(":", 1)[-1]) % 5
        lines.append(f"{r['image_id']},{r['client']},0,{t}")
        per[r["client"]] = per.get(r["client"], 0) + t
    table = tmp / "t2" / "per_row.csv"
    table.parent.mkdir(parents=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prov = {"recipe": V2, "pairs_digest": "d" * 64, "table_sha256": B.token_table_of(V2)[1],
            "contract_sha256": target_contract_sha256(V2), "schema": T.PROVENANCE_SCHEMA,
            "counter": {"module": "vlm.token_table", "sha256_lf": T.source_sha256()},
            "tokenizer": T.tokenizer_identity(MODEL_KW["model_id"], MODEL_KW["model_revision"])}
    prov.update(prov_over or {})
    if not drop:
        (table.parent / ("per_row.csv" + B.PROVENANCE_SUFFIX)).write_text(json.dumps(prov), encoding="utf-8")
    return p, table, per


def test_새_레시피는_비율_1_에서도_표_합과_설정_기대값을_맞댄다(tmp_path, monkeypatch):
    p, table, per = _v2_table(tmp_path, monkeypatch)
    kw = {"seed_value": 7, "pairs_path": p, "table_path": table, "recipe": V2, "pairs_digest": "d" * 64, **MODEL_KW}
    assert B.expected_tokens(per, fraction=1.0, **kw) == per
    wrong = {**per, "C1": per["C1"] + 1}                       # 설정 값이 틀리면 착수 전에 멈춘다
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(wrong, fraction=1.0, **kw)
    assert exc.value.code == "token_table_mismatch"
    got = B.expected_tokens(per, fraction=0.25, **kw)
    assert set(got) == set(per) and all(0 < got[c] <= per[c] for c in per)


@pytest.mark.parametrize("over,drop,needle", [
    ({}, True, "출처 파일이 없다"),
    ({"recipe": "json_only"}, False, "레시피"),
    ({"pairs_digest": "e" * 64}, False, "페어 digest"),
    ({"table_sha256": "0" * 64}, False, "표 sha256"),
    ({"contract_sha256": "0" * 64}, False, "계약"),
    ({"schema": None}, False, "옛 출처"),                                     # 10-22 의 첫 출처 꼴(파일 신원 없음)
    ({"counter": {"module": "vlm.token_table", "sha256_lf": "0" * 64}}, False, "계측 코드"),
    ({"counter": "c" * 64}, False, "계측 코드"),
    ({"tokenizer": None}, False, "토크나이저 신원이 없다"),
])
def test_새_레시피의_표는_출처가_지금_등록과_맞아야_쓴다(tmp_path, monkeypatch, over, drop, needle):
    p, table, per = _v2_table(tmp_path, monkeypatch, prov_over=over, drop=drop)
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=1.0, seed_value=7, pairs_path=p, table_path=table, recipe=V2,
                          pairs_digest="d" * 64, **MODEL_KW)
    assert exc.value.code == "token_table_provenance" and needle in str(exc.value)


@pytest.mark.parametrize(("change", "needle"), [
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("tokenizer_bytes", "tokenizer.json"),
    ("template_bytes", "chat_template.jinja"),
    ("processor_bytes", "preprocessor_config.json"),
    ("file_added", "special_tokens_map.json"),
    ("recorded_extra_file", "vocab.json"),
    ("revision", "revision"),
    ("model_id", "model_id"),
])
def test_같은_페어_레시피_합이어도_토크나이저_신원이_다르면_표를_쓰지_않는다(tmp_path, monkeypatch, change, needle):
    """외부 검토 200 M-01 — 페어 · 레시피 · 표 · 합은 그대로 두고 신원만 바꾼다. 둘 다 거부: 지금 환경이 바뀐 경우(꾸러미 판 · 스냅샷
    파일 바이트 · 파일 하나 더 · 다른 판 · 다른 모델)와 출처에 다른 신원이 적힌 경우(계측 때 없던 파일이 적힘)."""
    from vlm import token_table as T

    p, table, per = _v2_table(tmp_path, monkeypatch)
    kw = {"seed_value": 7, "pairs_path": p, "table_path": table, "recipe": V2, "pairs_digest": "d" * 64, **MODEL_KW}
    assert B.expected_tokens(per, fraction=1.0, **kw) == per                      # 바꾸기 전에는 쓴다
    snap = tmp_path / "_hf_snapshot"
    if change in ("transformers", "tokenizers"):
        versions = {"transformers": "9.9.9", "tokenizers": "0.0.1", change: "0.0.0-other"}
        monkeypatch.setattr(T, "_package_version", lambda name: versions.get(name))
    elif change.endswith("_bytes"):
        name = {"tokenizer": "tokenizer.json", "template": "chat_template.jinja",
                "processor": "preprocessor_config.json"}[change.split("_")[0]]
        (snap / name).write_bytes((snap / name).read_bytes() + b" ")
    elif change == "file_added":
        (snap / "special_tokens_map.json").write_bytes(b"{}\n")
    elif change == "recorded_extra_file":
        prov_p = table.parent / (table.name + B.PROVENANCE_SUFFIX)
        prov = json.loads(prov_p.read_text(encoding="utf-8"))
        prov["tokenizer"]["files"]["vocab.json"] = "0" * 64
        prov_p.write_text(json.dumps(prov), encoding="utf-8")
    else:
        kw = {**kw, {"revision": "model_revision", "model_id": "model_id"}[change]: "other"}
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=1.0, **kw)
    assert exc.value.code == "token_table_provenance" and needle in str(exc.value)


def test_토크나이저_신원을_맞대지_못하면_새_레시피의_표를_쓰지_않는다(tmp_path, monkeypatch):
    """모델을 받지 못했거나 로컬 캐시에 토크나이저 설정이 없으면 — 맞대지 않은 채 쓰지 않는다(닫힌 쪽)."""
    from vlm import token_table as T

    p, table, per = _v2_table(tmp_path, monkeypatch)
    kw = {"seed_value": 7, "pairs_path": p, "table_path": table, "recipe": V2, "pairs_digest": "d" * 64}
    with pytest.raises(B.BudgetRejected, match="모델") as exc:
        B.expected_tokens(per, fraction=1.0, **kw)
    assert exc.value.code == "token_table_provenance"
    monkeypatch.setattr(T, "_cached_file", lambda model_id, revision, name: None)
    with pytest.raises(B.BudgetRejected, match="재지 못했다"):
        B.expected_tokens(per, fraction=1.0, **kw, **MODEL_KW)


def test_1차_레시피는_토크나이저_신원을_재지_않는다(tmp_path, monkeypatch):
    """v1: 신원 미기록 — 1 차 표에는 출처 파일이 없어 지문 고정만 본다. 1 차 경로는 신원을 재지 않고 지금처럼 지난다(회귀)."""
    from vlm import token_table as T

    def boom(*a, **k):
        raise AssertionError("1 차 레시피는 토크나이저 신원을 재지 않는다")

    monkeypatch.setattr(T, "tokenizer_identity", boom)
    monkeypatch.setattr(T, "_cached_file", boom)
    assert B.token_table_problems("json_only", pairs_digest=None) == []
    assert B.token_table_problems("json_only", pairs_digest="d" * 64, **MODEL_KW) == []
    per = {"C1": 10, "C2": 20}
    assert B.expected_tokens(per, fraction=1.0, seed_value=7, recipe="json_only", **MODEL_KW) == per


def test_토크나이저_신원은_스냅샷_파일의_sha256_과_꾸러미_판이다(tmp_path, monkeypatch):
    from tests.uni_fakes import FAKE_TOKENIZER_FILES, fake_tokenizer_env
    from vlm import token_table as T

    snap = fake_tokenizer_env(monkeypatch, tmp_path)
    got = T.tokenizer_identity("tiny", "r1")
    assert set(got) == set(T.IDENTITY_KEYS)
    assert got["files"] == {n: hashlib.sha256(raw).hexdigest() for n, raw in FAKE_TOKENIZER_FILES.items()}
    assert (got["model_id"], got["revision"], got["transformers"], got["tokenizers"]) == ("tiny", "r1", "9.9.9", "0.0.1")
    assert set(got["files"]) <= set(T.IDENTITY_FILES)
    (snap / "tokenizer_config.json").unlink()
    with pytest.raises(FileNotFoundError):                                     # 토크나이저 설정이 없으면 신원이 없다
        T.tokenizer_identity("tiny", "r1")
    assert T.identity_problems(None, got) == ["출처 파일에 토크나이저 신원이 없다"]
    assert T.identity_problems({**got, "revision": None}, got)                   # 자료형까지 본다
    assert T.identity_problems(dict(got), got) == []


def _real_v2():
    """운영 설정과 정본 토큰 표 핀을 그대로 읽어 봉인 계측과 대조한다."""
    from vlm.token_table import tokenizer_identity
    from vlm.uni_config import load_uni_config

    cfg = load_uni_config()
    rel, digest = B.token_table_of(V2)
    assert (rel, digest) == (SEALED_TABLE, SEALED_TABLE_SHA256)
    table = ROOT / rel
    assert table.is_file() and (ROOT / cfg.pairs_path).is_file(), "봉인 계측 자산이 없다"
    assert hashlib.sha256(A02.read_bytes()).hexdigest() == A02_SHA256
    tokenizer_identity(cfg.model_id, cfg.model_revision)
    return cfg, table, digest


def test_새_표는_고정_지문이고_참여자_합이_02a_의_독립_계측과_같다():
    """표 바이트의 지문 = 고정값, 표에서 직접 더한 참여자 합 = 출처 파일의 합 = 02a 의 (나) 계측(`shares_by_fraction.na`, 세 갈래
    대조를 지난 셈) — 같은 입력에서 두 구현이 같은 값을 낸다."""
    import csv
    import hashlib

    cfg, table, sha = _real_v2()
    raw = table.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == sha
    sums: dict[str, int] = {}
    for r in csv.DictReader(raw.decode("utf-8").splitlines()):
        sums[r["client"]] = sums.get(r["client"], 0) + int(r["supervised_tokens"])
    prov = json.loads((table.parent / (table.name + B.PROVENANCE_SUFFIX)).read_text(encoding="utf-8"))
    assert {c: v["supervised_tokens"] for c, v in prov["per_client"].items()} == sums
    a02 = json.loads(A02.read_text(encoding="utf-8"))["shares_by_fraction"]["na"]
    assert {c: v["supervised_tokens"] for c, v in a02["1.000000"].items()} == sums
    assert B.token_table_problems(V2, pairs_digest=str(cfg.pairs_digest), model_id=cfg.model_id,
                                  model_revision=cfg.model_revision) == []         # 출처의 신원 · 계측 코드 = 이 환경 · 이 코드


@pytest.mark.parametrize("key", ["0.031250", "0.125000", "0.500000", "1.000000"])
def test_새_레시피의_예산_몫_기대값이_02a_와_같다(key, monkeypatch):
    """`expected_tokens`(행 고르기 · 표 합) 와 02a 의 따로 짠 몫 계산(같은 시드 20260828)이 같은 값을 낸다.

    02a(10-27)는 창 32(누적 32)로 몫을 계산한 기록이다 — 두 구현의 대조는 같은 창에서 한다(10-31 누적 8 로 바꾼 뒤에도 알고리즘 대조로 유지).
    """
    from vlm import pilot_vlm

    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 32)
    cfg, _table, _sha = _real_v2()
    a02 = json.loads(A02.read_text(encoding="utf-8"))["shares_by_fraction"]["na"]
    per = {c: v["supervised_tokens"] for c, v in a02["1.000000"].items()}
    got = B.expected_tokens(per, fraction=float(key), seed_value=20260828, pairs_path=cfg.pairs_path, recipe=V2,
                            pairs_digest=str(cfg.pairs_digest), model_id=cfg.model_id, model_revision=cfg.model_revision)
    assert got == {c: v["supervised_tokens"] for c, v in a02[key].items()}
