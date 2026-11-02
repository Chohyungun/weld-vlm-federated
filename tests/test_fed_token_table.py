"""연합 서버 앱의 행별 토큰 표 자리(2026-10-17, 06 §6-5) — 서버 앱은 풀어 놓은 FAB 폴더에서 돌아 그 `REPO_ROOT` 아래에
csv 가 없다. 예산 비율이 1 이 아닌 본실험은 실행기가 넘긴 저장소 뿌리(`uni-repo-root`) 아래의 표를 등록 지문과 맞대 읽는다.

모델 · GPU · flwr 없이 본다 — 표 자리 · 지문 대조, FAB 폴더 모사(이 모듈의 `REPO_ROOT` 를 빈 폴더로)에서 기본 자리는 표가 없고
저장소 뿌리 자리는 찾는지, 서버 → `main_run_problems` → `expected_tokens` 의 배선, 뿌리가 없을 때의 거부, 실행기가 넘기는 값.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import vlm.budget as B
from fl.uni_fed import UniFedRejected, UniFedRun, main_token_table

SIZES = {"C1": 40, "C2": 35, "C3": 33}


@pytest.fixture(autouse=True)
def _legacy_registration(monkeypatch, tmp_path):
    """구판 표 배선 시험은 같은 체제의 등록을 읽는다. 새 레시피 시험의 명시 입력은 그대로 받는다."""
    from tests.registration_history import legacy_default

    legacy_default(monkeypatch, tmp_path)


def _repo(tmp: Path) -> tuple[Path, Path, dict[str, int]]:
    """(저장소 뿌리, 페어, 참여자별 epoch 당 기대값) — 뿌리 아래 `TOKEN_TABLE` 자리에 합성 표를 둔다."""
    repo = tmp / "checkout"
    rows, lines, per = [], ["image_id,client,n_defects,supervised_tokens"], {}
    for c, n in SIZES.items():
        per[c] = 0
        for i in range(n):
            iid, t = f"synth:{c}:{i:04d}", 20 + i % 7
            rows.append({"image_id": iid, "client": c, "split": "train"})
            lines.append(f"{iid},{c},0,{t}")
            per[c] += t
    pairs = repo / "pairs.jsonl"
    pairs.parent.mkdir(parents=True)
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    table = repo / B.TOKEN_TABLE
    table.parent.mkdir(parents=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return repo, pairs, per


def test_표를_자리로_주면_지문을_줄_때만_맞댄다(tmp_path):
    repo, pairs, per = _repo(tmp_path)
    table = repo / B.TOKEN_TABLE
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    got = B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=pairs, table_path=table, table_sha256=sha)
    assert set(got) == set(SIZES) and all(0 < got[c] < per[c] for c in SIZES)
    assert B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=pairs, table_path=table) == got   # 지문 없음
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=pairs, table_path=table, table_sha256="0" * 64)
    assert exc.value.code == "token_table_changed"


def test_FAB_폴더_모사에서_기본_자리는_표가_없고_저장소_뿌리_자리는_찾는다(tmp_path, monkeypatch):
    repo, pairs, per = _repo(tmp_path)
    app = tmp_path / "flwr_home" / "apps" / "weld-fl.weld-fl.0.1.0.deadbeef"     # 풀어 놓은 FAB 폴더 — csv 가 없다
    app.mkdir(parents=True)
    monkeypatch.setattr(B, "REPO_ROOT", app)
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=pairs)
    assert exc.value.code == "token_table_missing"
    table = main_token_table("main", 0.125, repo.as_posix())
    assert table == repo / B.TOKEN_TABLE and table.is_file()
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    assert B.expected_tokens(per, fraction=0.125, seed_value=7, pairs_path=pairs, table_path=table, table_sha256=sha)


@pytest.mark.parametrize(("purpose", "fraction", "root", "want"), [
    ("main", 1.0, "", None), ("rehearsal", 0.125, "", None), ("main", 0.125, "", "uni_repo_root_missing"),
    ("main", 0.125, "relative/checkout", "uni_repo_root_missing")])
def test_표를_읽는_자리는_비율이_1_이_아닌_본실험만이고_뿌리가_없으면_거부한다(purpose, fraction, root, want):
    if want is None:
        assert main_token_table(purpose, fraction, root) is None
        return
    with pytest.raises(UniFedRejected) as exc:
        main_token_table(purpose, fraction, root)
    assert exc.value.code == want


def test_main_run_problems_는_받은_표_자리와_등록_지문을_expected_tokens_에_넘긴다(monkeypatch):
    from vlm.uni_config import load_uni_config, main_run_problems

    seen = {}

    def spy(per_epoch, **kw):
        seen.update(kw)
        return per_epoch

    monkeypatch.setattr(B, "expected_tokens", spy)
    cfg = load_uni_config()
    main_run_problems(cfg, model_id=cfg.model_id, model_revision=cfg.model_revision, pairs_path=cfg.pairs_path,
                      pairs_digest=cfg.pairs_digest, chat_template_kwargs=cfg.chat_template_kwargs, num_rounds=1,
                      local_epochs=1, total_epochs=1, seed_index=1, seed_value=int(cfg.seeds[0]),
                      snapshot_digest=cfg.snapshot_digest, prompt_path=cfg.prompt_path, coord_space=cfg.coord_space,
                      output_paths=[], budget_fraction=0.125, token_table=Path("/repo/x.csv"))
    assert seen["table_path"] == Path("/repo/x.csv") and seen["table_sha256"] == B.TOKEN_TABLE_SHA256
    main_run_problems(cfg, model_id=cfg.model_id, model_revision=cfg.model_revision, pairs_path=cfg.pairs_path,
                      pairs_digest=cfg.pairs_digest, chat_template_kwargs=cfg.chat_template_kwargs, num_rounds=1,
                      local_epochs=1, total_epochs=1, seed_index=1, seed_value=int(cfg.seeds[0]),
                      snapshot_digest=cfg.snapshot_digest, prompt_path=cfg.prompt_path, coord_space=cfg.coord_space,
                      output_paths=[], budget_fraction=0.125)
    assert seen["table_path"] is None and seen["table_sha256"] is None                  # 칸 학습 · ECC — 지금처럼 기본 자리


def _main_get(tmp: Path, **over):
    vals = {"uni-model": "Qwen/Qwen3.5-4B", "uni-model-revision": "r" * 40, "uni-pairs": str(tmp / "pairs.jsonl"),
            "uni-pairs-digest": "p" * 64, "uni-prompt": str(ROOT / "vlm/prompts/unified_v2_absorig.txt"),
            "uni-chat-template-kwargs": '{"enable_thinking": false}', "uni-coord-space": "ABS_ORIG",
            "uni-seed-index": "1", "uni-processor-kwargs": '{"max_pixels": 901120}',
            "uni-expected-supervised-tokens": '{"C1": 10, "C2": 10, "C3": 10}', "purpose": "main",
            "uni-budget-fraction": "0.125"}
    vals.update(over)
    return lambda k, d: vals.get(k, d)


def _run(tmp: Path, get, **kw):
    return UniFedRun(get=get, out_dir=tmp / "out" / "uni_fed_s1", run_id="uni_fed_s1_x", base_seed=7,
                     split_hash="s" * 64, num_rounds=1, local_epochs=1, total_epochs=1, client_tags=["C1", "C2", "C3"],
                     non_main_parent=tmp / "non_main", **kw)


def test_연합_서버는_저장소_뿌리_아래의_표_자리를_main_run_problems_에_넘기고_뿌리가_없으면_거부한다(tmp_path, monkeypatch):
    import vlm.uni_config as UC

    seen = {}

    def spy(cfg, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(UC, "main_run_problems", spy)
    repo = tmp_path / "checkout"
    _run(tmp_path, _main_get(tmp_path, **{"uni-repo-root": repo.as_posix()}))
    assert seen["token_table"] == repo / B.TOKEN_TABLE and seen["budget_fraction"] == 0.125
    seen.clear()
    with pytest.raises(UniFedRejected) as exc:
        _run(tmp_path, _main_get(tmp_path))
    assert exc.value.code == "uni_repo_root_missing" and not seen                       # 설정 대조 전에 멈춘다
    _run(tmp_path, _main_get(tmp_path, **{"uni-budget-fraction": ""}))                  # 비율 1.0 — 지금처럼
    assert seen["token_table"] is None


def test_실행기는_저장소_뿌리의_절대_경로를_연합_설정으로_넘기고_신원에는_넣지_않는다(tmp_path, monkeypatch):
    from flwr.common.config import parse_config_args

    import vlm.registration as reg
    from scripts import main_uni as M
    from tests.test_main_uni import _cfg_yaml

    monkeypatch.setattr(reg, "registered_budget_fraction", lambda path=None: 1.0)      # 합성 페어 — 토큰 표와 맞지 않는다

    cfg = M.load_main_config(_cfg_yaml(tmp_path))
    got = parse_config_args([M.fed_run_config(cfg, 1, out=tmp_path / "out")])
    assert got["uni-repo-root"] == Path(os.path.abspath(M.REPO_ROOT)).as_posix()
    import inspect

    from fl.uni_fed import fed_identity

    assert "repo" not in " ".join(inspect.signature(fed_identity).parameters)              # 연합 신원의 칸이 아니다


# ================================================================ 새 레시피 본실험의 연합 서버 경로(지시 1022c 덧붙임 B)
V2 = "json_line_then_basis"
V3_PROMPT = str(ROOT / "vlm/prompts/unified_v3_1_basis.txt")


#: 합성 토크나이저 환경(`tests.uni_fakes.fake_tokenizer_env`)의 모델 — 서버가 설정의 모델로 지금 신원을 잰다.
MODEL_KW = {"model_id": "tiny", "model_revision": "r1"}


def _repo_v2(tmp: Path, monkeypatch, *, prov_over=None):
    """저장소 뿌리 아래 새 레시피의 표 · 출처 파일(합성). 표의 고정 지문은 이 합성 표로 바꿔 끼운다(실물 고정값은 실물 표의 것).
    출처의 꼴 · 계측 코드 · 토크나이저 신원은 지금 코드와 합성 토크나이저 환경(모델 tiny@r1)의 값이다."""
    from tests.uni_fakes import fake_tokenizer_env
    from vlm import token_table as T
    from vlm.pilot_vlm import target_contract_sha256

    fake_tokenizer_env(monkeypatch, tmp)
    repo, pairs, _per_v1 = _repo(tmp)
    rel = B.token_table_of(V2)[0]
    lines, per2 = ["image_id,client,supervised_tokens"], {}
    for r in (json.loads(x) for x in pairs.read_text(encoding="utf-8").splitlines()):
        t = 40 + int(r["image_id"].rsplit(":", 1)[-1]) % 9
        lines.append(f"{r['image_id']},{r['client']},{t}")
        per2[r["client"]] = per2.get(r["client"], 0) + t
    table = repo / rel
    table.parent.mkdir(parents=True)
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sha = hashlib.sha256(table.read_bytes()).hexdigest()
    monkeypatch.setitem(B.TOKEN_TABLES, V2, (rel, sha))
    prov = {"recipe": V2, "pairs_digest": "p" * 64, "table_sha256": sha, "contract_sha256": target_contract_sha256(V2),
            "schema": T.PROVENANCE_SCHEMA, "counter": {"module": "vlm.token_table", "sha256_lf": T.source_sha256()},
            "tokenizer": T.tokenizer_identity(MODEL_KW["model_id"], MODEL_KW["model_revision"]), **(prov_over or {})}
    (table.parent / (table.name + B.PROVENANCE_SUFFIX)).write_text(json.dumps(prov), encoding="utf-8")
    return repo, pairs, per2, table, sha


def _registration(tmp: Path, recipe: str) -> Path:
    """보존된 1차 등록에서 recipe 한 줄만 바꾼 합성 등록(서버의 등록 경로 이음새)."""
    from tests.registration_history import historical_registration

    raw = historical_registration()
    assert raw.count(b"recipe: json_only") == 1
    p = tmp / f"train_registration_{recipe}.yaml"
    p.write_bytes(raw.replace(b"recipe: json_only", f"recipe: {recipe}".encode()))
    return p


def test_FAB_모사에서_새_레시피_본실험은_비율_1_에서도_저장소_뿌리의_표를_돌려주고_지문_출처를_맞댄다(tmp_path, monkeypatch):
    repo, pairs, per, table, sha = _repo_v2(tmp_path, monkeypatch)
    app = tmp_path / "flwr_home" / "apps" / "weld-fl.weld-fl.0.1.0.deadbeef"
    app.mkdir(parents=True)
    monkeypatch.setattr(B, "REPO_ROOT", app)                                    # 풀어 놓은 FAB 폴더 — 표가 없다
    got = main_token_table("main", 1.0, repo.as_posix(), recipe=V2)
    assert got == table and got.is_file()                                       # 옛 꼴(f == 1 이면 None)이면 여기서 떨어진다
    assert main_token_table("main", 1.0, repo.as_posix()) is None               # 1 차 레시피 · f = 1 은 지금처럼 표를 읽지 않는다
    kw = {"seed_value": 7, "pairs_path": pairs, "table_path": got, "recipe": V2, "pairs_digest": "p" * 64, **MODEL_KW}
    assert B.expected_tokens(per, fraction=1.0, table_sha256=sha, **kw) == per
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=1.0, table_sha256="0" * 64, **kw)
    assert exc.value.code == "token_table_changed"
    with pytest.raises(B.BudgetRejected) as exc:
        B.expected_tokens(per, fraction=1.0, table_sha256=sha, **{**kw, "pairs_digest": "e" * 64})
    assert exc.value.code == "token_table_provenance"
    with pytest.raises(B.BudgetRejected) as exc:                                # 기본 자리(FAB 안)에는 표가 없다
        B.expected_tokens(per, fraction=1.0, seed_value=7, pairs_path=pairs, recipe=V2, pairs_digest="p" * 64, **MODEL_KW)
    assert exc.value.code in ("token_table_provenance", "token_table_missing")
    # 서버 쪽 토크나이저가 계측 때와 다르면(같은 판 이름 · 다른 스냅샷 바이트) — 같은 페어 · 레시피 · 합이어도 멈춘다(M-01)
    snap = tmp_path / "_hf_snapshot"
    (snap / "tokenizer.json").write_bytes((snap / "tokenizer.json").read_bytes() + b" ")
    with pytest.raises(B.BudgetRejected, match="tokenizer.json") as exc:
        B.expected_tokens(per, fraction=1.0, table_sha256=sha, **kw)
    assert exc.value.code == "token_table_provenance"


def test_연합_서버는_새_레시피_본실험이면_비율_1_에서도_표_자리를_넘기고_레시피를_등록과_맞댄다(tmp_path, monkeypatch):
    import vlm.uni_config as UC

    repo, _pairs, _per, table, _sha = _repo_v2(tmp_path, monkeypatch)
    seen = {}

    def spy(cfg, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(UC, "main_run_problems", spy)
    get = _main_get(tmp_path, **{"uni-repo-root": repo.as_posix(), "uni-budget-fraction": "", "uni-target-recipe": V2,
                                 "uni-prompt": V3_PROMPT})
    reg2 = _registration(tmp_path, V2)
    _run(tmp_path, get, registration_path=reg2)
    assert (seen["token_table"], seen["recipe"], seen["budget_fraction"]) == (table, V2, 1.0)
    seen.clear()
    for g, reg, code in (
        (get, None, "recipe_vs_registration"),                                  # 실행 설정은 새 레시피 · 등록(기본)은 1 차
        (_main_get(tmp_path, **{"uni-repo-root": repo.as_posix()}), reg2, "recipe_vs_registration"),   # 거꾸로
        (_main_get(tmp_path, **{"uni-repo-root": repo.as_posix(), "uni-budget-fraction": "", "uni-target-recipe": V2}),
         reg2, "regime_mismatch"),                                              # 새 레시피 + v2 프롬프트
    ):
        with pytest.raises(UniFedRejected) as exc:
            _run(tmp_path, g, registration_path=reg)
        assert exc.value.code == code and not seen                              # 표 · 설정 대조 전에 멈춘다


def test_연합_서버는_출력_뿌리의_실행_묶음이_묶은_등록을_읽는다(tmp_path, monkeypatch):
    import vlm.uni_config as UC

    monkeypatch.setattr(UC, "main_run_problems", lambda cfg, **kw: [])
    sens = ROOT / "configs/train_registration_sens025.yaml"                     # 저장소 안의 등록 벌(1 차 레시피)
    out_root = tmp_path / "sens_u025"
    (out_root / "seed1").mkdir(parents=True)
    rec = {"train_registration": "configs/train_registration_sens025.yaml", "run_name": "sens_u025",
           "train_registration_sha256": hashlib.sha256(sens.read_bytes()).hexdigest()}
    (out_root / "RUN_PROFILE.json").write_text(json.dumps(rec), encoding="utf-8")
    get = _main_get(tmp_path, **{"uni-train-root": (out_root / "seed1").as_posix(), "uni-budget-fraction": ""})
    _run(tmp_path, get)                                                         # 묶음의 등록(1 차) = 실행 설정(1 차)
    (out_root / "RUN_PROFILE.json").write_text(json.dumps({**rec, "train_registration_sha256": "0" * 64}),
                                               encoding="utf-8")
    with pytest.raises(UniFedRejected) as exc:
        _run(tmp_path, get)
    assert exc.value.code == "registration"
