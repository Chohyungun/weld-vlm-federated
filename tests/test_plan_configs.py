"""계획 파일 둘과 통합형 설정 키가 명세와 같은지 본다.

**설정을 읽는 코드가 아직 없다.** 그래서 이 시험이 키의 정본을 대신 지킨다 — 키가 명세와 어긋나면 읽는 코드가
생긴 날 조용히 다른 이름을 찾는다.

아래 두 상수는 명세의 표·코드 블록을 **기계로 뽑아** 넣었다(사람이 옮기지 않았다). 명세가 바뀌면 이 상수를
다시 뽑고 이 시험이 먼저 떨어진다.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
PLANS = {"rehearsal": CONFIGS / "rehearsal_uni.yaml", "frame_diag": CONFIGS / "frame_diag_uni.yaml"}

#: (키, rehearsal 에 있다, frame_diag 에 있다). 리허설 명세 반영판 §4-4 의 표에서 기계로 뽑았다.
PLAN_TABLE: tuple[tuple[str, bool, bool], ...] = (
    ('kind', True, True),
    ('version', True, True),
    ('sampling_seed', True, True),
    ('seed_index', True, True),
    ('cells', True, True),
    ('gen.split', True, True),
    ('gen.defect_only', True, True),
    ('gen.target_n', True, False),
    ('gen.target_boxes', False, True),
    ('gen.max_n', True, True),
    ('gen.min_groups_per_stratum', True, True),
    ('gen.max_target_tokens', True, True),
    ('echo.same_as_gen', True, True),
    ('train.split', True, True),
    ('train.rows', True, True),
    ('train.avoid_multiple_of_accum', True, True),
    ('train.central_rows', True, True),
    ('train.fed_rows', True, False),
    ('amount', True, True),
    ('exclude.near_dup_edges', True, True),
    ('exclude.drop_components_touching_eval', True, True),
    ('gen_limit.max_new_tokens', True, True),
    ('gen_limit.provisional', True, True),
    ('faults.train', True, False),
    ('faults.export', True, False),
    ('negatives', True, False),
    ('timeouts_s', True, True),
    ('optional', True, False),
    ('attempts_allowed', False, True),
)

#: 통합형 설정 키. 정정된 통합형 학습 명세 §2-3 의 코드 블록을 읽어 뽑고, 통합형 학습 구현 보고 6절 7 · 8 의
#: 두 키씩을 더했다.
UNI_KEYS: frozenset[str] = frozenset({
    'uni_batch_size',
    'uni_chat_template_kwargs.enable_thinking',
    'uni_coord_space',
    'uni_expected_supervised_tokens',
    'uni_max_new_tokens',
    'uni_model.id',
    'uni_model.revision',
    'uni_pairs.digest',
    'uni_pairs.path',
    'uni_processor_kwargs',
    'uni_prompt_path',
    'uni_prompt_sha256',
    'uni_train_budget.local_epochs',
    'uni_train_budget.num_rounds',
    'uni_train_budget.total_epochs',
})

#: 계획에 있으면 안 되는 것 — 본실험 조건을 계획이 바꾸는 길을 막는다. 경로의 마디 이름으로 본다.
FORBIDDEN_SEGMENTS = frozenset({
    "model", "adapter", "lora", "optimizer", "lr", "preprocess", "prompt", "template",
    "chat_template_kwargs", "coord_space", "decoding", "seed", "seeds", "experiment", "budget",
    "train_budget", "fixed_before_main_runs", "snapshot_digest", "pairs_sha256", "pairs_digest", "expect",
})
#: 값을 정하지 않아 `null` 로 둔 자리. 채우는 것은 의도한 변경이어야 하므로 여기도 함께 고친다.
PLAN_NULLS = {"exclude.near_dup_edges"}
#: 채운 자리 — 리허설 계획의 간선은 판본 neardup_edges_v1 이다(총괄 결정 19 의 R-2). 진단 계획은 아직 비어 있다
FILLED_BY_KIND = {"rehearsal": {"exclude.near_dup_edges"}}
EDGES_DIR = "data/interim/neardup_edges_v1"
REHEARSAL_ONLY_NULLS = {"gen.max_target_tokens"}
#: 시간 제한의 단계 다섯 — 학습 쪽 오케스트레이터가 읽는 키다. 값은 적재 시간을 잰 뒤 채운다.
TIMEOUT_STAGES = ("lists", "register", "train", "export", "score")
#: 값이 있는 통합형 키 — 통합형 학습 구현 보고 6절 8 에서 기계로 뽑았다. `null` 이 아니라 이 값이어야 한다.
UNI_VALUES: dict[str, str] = {'uni_prompt_path': 'vlm/prompts/unified_v2_absorig.txt', 'uni_coord_space': 'ABS_ORIG'}
#: 값이 사전인 통합형 키 — 사전 안은 키 목록 대조에서 펼치지 않는다(안의 꼴은 따로 본다).
UNI_DICT_VALUED = frozenset({"uni_processor_kwargs", "uni_expected_supervised_tokens"})
#: 감독 토큰 기대값의 출처 — 학습 쪽 계측 산출 파일. 설정의 주석이 이 경로와 해시를 적는다.
TOKENS_SOURCE = "outputs/B_seal_20261001/tokens/supervised_tokens.json"
TOKENS_SOURCE_SHA256 = "10e62663b0297557832d290ea04f7a3c9a7d49156b9fe9ef7df2b11071f9c190"
#: 총괄 결정 19 로 채운 키 — 모델 판과 봉인 페어. 값은 결정 원문 · 봉인 계약서에서 읽어 넣었다.
UNI_DECIDED = frozenset({"uni_model.revision", "uni_pairs.path", "uni_pairs.digest"})
#: 결정 19 가 승인한 모델 판(파일럿 · 프로세서 확인 · 감독 토큰 계측이 쓴 캐시 판)
UNI_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
#: 봉인 페어 — 동결 명부의 자리. path 는 그 안의 페어 파일이다(학습 쪽 판독기가 부모 폴더의 계약서를 맞댄다)
UNI_PAIRS_DIR = "data/processed/pairs_main_v1"
UNI_NULLS = (UNI_KEYS - {"uni_model.id", "uni_chat_template_kwargs.enable_thinking", "uni_batch_size"}
             - set(UNI_VALUES) - UNI_DICT_VALUED - UNI_DECIDED)
FAULT_POINTS = {"train": {"after_ckpt", "before_ckpt"}, "export": {"after_lines", "torn_line", "after_all_lines"}}


def _load(p: Path):
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _base():
    return _load(CONFIGS / "base.yaml")


def _get(d, path: str):
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            raise KeyError(path)
        d = d[part]
    return d


def _has(d, path: str) -> bool:
    try:
        _get(d, path)
        return True
    except KeyError:
        return False


def _paths(d, known: set[str], prefix: str = "") -> list[str]:
    """명세의 키에서 멈추는 경로 목록. 명세에 없는 가지는 그 자리의 경로로 낸다."""
    out = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        if p in known:
            out.append(p)
        elif isinstance(v, dict) and any(x.startswith(p + ".") for x in known):
            out.extend(_paths(v, known, p + "."))
        else:
            out.append(p)
    return out


def _flat(d, prefix: str = "", leaves: frozenset[str] = frozenset()) -> list[str]:
    out = []
    for k, v in d.items():
        p = f"{prefix}{k}"
        out.extend(_flat(v, p + ".", leaves) if isinstance(v, dict) and v and p not in leaves else [p])
    return out


# --- 계획 파일 --------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(PLANS))
def test_계획_파일의_키가_명세의_표와_같다(kind):
    """표가 그 목적에 두라는 키는 있고, 두지 말라는 키("—"·"키가 있으면 거부")와 모르는 키는 없다."""
    plan = _load(PLANS[kind])
    col = 1 if kind == "rehearsal" else 2
    want = {row[0] for row in PLAN_TABLE if row[col]}
    known = {row[0] for row in PLAN_TABLE}
    got = set(_paths(plan, known))
    assert got == want, f"더 있는 키 {sorted(got - want)} · 빠진 키 {sorted(want - got)}"


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_계획은_자기_목적만_가리킨다(kind):
    assert _load(PLANS[kind])["kind"] == kind
    assert _load(PLANS[kind])["version"] == 1


#: 본실험 설정 키의 이름. 칸 태그(`uni_central` 등)는 계획에 있어도 된다 — 이름이 아니라 이 목록으로 가른다.
UNI_TOP = frozenset(k.split(".")[0] for k in UNI_KEYS) | {"uni_template_mode"}


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_계획에_본실험_조건이_없다(kind):
    """모델·템플릿·시드 값·예산 같은 본실험 키는 계획에 두지 않는다. 본실험의 `uni_*` 설정 키도 없다."""
    bad = [p for p in _flat(_load(PLANS[kind]))
           if any(seg in FORBIDDEN_SEGMENTS or seg in UNI_TOP for seg in p.split("."))]
    assert not bad, bad


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_시드는_기본_설정에서_가져오고_지문은_두지_않는다(kind):
    """스냅샷·페어의 지문은 계획이 적지 않는다 — 계획이 적은 값을 믿으면 지문만 다른 판을 적은 계획이 대조를 지난다."""
    plan, base = _load(PLANS[kind]), _base()
    assert "expect" not in plan
    text = PLANS[kind].read_text(encoding="utf-8")
    assert base["fixed_before_main_runs"]["snapshot_digest"] not in text
    assert plan["seed_index"] == 1
    assert plan["sampling_seed"] == base["experiment"]["seeds"][plan["seed_index"] - 1], "등록 시드표의 첫 값이 아니다"


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_정하지_않은_값은_null_이다(kind):
    """`null` 을 채우는 것은 의도한 변경이다. 채우면 이 목록도 함께 고친다."""
    plan = _load(PLANS[kind])
    nulls = {p for p in _paths(plan, {row[0] for row in PLAN_TABLE}) if _get(plan, p) is None}
    want = (PLAN_NULLS - FILLED_BY_KIND.get(kind, set())) | (REHEARSAL_ONLY_NULLS if kind == "rehearsal" else set())
    assert nulls == want, f"null 인 자리 {sorted(nulls)} · 기대 {sorted(want)}"


def test_리허설_계획의_크기와_중단():
    p = _load(PLANS["rehearsal"])
    assert p["cells"] == ["uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_fed"]
    assert p["gen"] == {"split": "val", "defect_only": False, "target_n": 48, "max_n": 64,
                        "min_groups_per_stratum": 1, "max_target_tokens": None}
    assert p["echo"]["same_as_gen"] is True
    assert p["train"] == {"split": "train", "rows": {"C1": 100, "C2": 60, "C3": 30},
                          "avoid_multiple_of_accum": True, "central_rows": "union", "fed_rows": "same_as_local"}
    assert p["amount"] == {"n": 2, "r": 2, "e": 1}
    assert p["exclude"]["drop_components_touching_eval"] is True
    assert p["gen_limit"] == {"max_new_tokens": 512, "provisional": True}
    assert p["negatives"] == ["policy_blank", "identity_key_missing", "identity_key_extra", "format_mismatch"]
    assert p["optional"] == {"t_eq_uni": False, "dtype_compare": False, "logprob_invariance": False}
    f = p["faults"]
    assert f["train"] == {"uni_central": "train:after_ckpt:uni_central:ep=0",
                          "uni_local_C3": "train:before_ckpt:uni_local_C3:ep=0"}
    assert f["export"]["uni_central"] == [
        {"fault": "export:after_lines:uni_central:n=12"}, {"fault": "export:torn_line:uni_central:n=21"},
        {"stop_after_lines": 10}, {"to_end": True}]
    assert f["export"]["uni_local_C3"] == [{"fault": "export:after_all_lines:uni_local_C3:-"}, {"to_end": True}]


def test_진단_계획의_크기():
    p = _load(PLANS["frame_diag"])
    assert p["cells"] == ["uni_central"]
    assert p["gen"] == {"split": "val", "defect_only": True, "target_boxes": 1416, "max_n": 480,
                        "min_groups_per_stratum": 1, "max_target_tokens": 512}
    assert p["train"] == {"split": "train", "rows": {"central": 4000},
                          "avoid_multiple_of_accum": False, "central_rows": "direct"}
    assert p["amount"] == {"n": 1, "r": 1, "e": 1}
    assert p["gen_limit"] == {"max_new_tokens": 512, "provisional": True}
    assert p["attempts_allowed"] == 1
    assert "faults" not in p, "진단 계획에 중단 키가 있다"


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_학습량은_n_r_e_셋이고_n_은_r_곱하기_e_다(kind):
    a = _load(PLANS[kind])["amount"]
    assert set(a) == {"n", "r", "e"}
    assert all(type(v) is int and v > 0 for v in a.values()), a
    assert a["n"] == a["r"] * a["e"], a


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_시간_제한은_단계_다섯의_사전이고_값은_비어_있다(kind):
    """값을 채우는 것은 적재 시간을 잰 뒤의 의도한 변경이다. 채우면 이 시험도 함께 고친다."""
    t = _load(PLANS[kind])["timeouts_s"]
    assert isinstance(t, dict) and tuple(t) == TIMEOUT_STAGES, t
    assert all(v is None for v in t.values()), t


@pytest.mark.parametrize("kind", sorted(PLANS))
def test_간선_파일의_꼴을_머리에_적었다(kind):
    """간선 파일의 꼴은 읽는 쪽과 맞물린다 — 머리행과 인코딩을 계획 파일이 말한다."""
    head = PLANS[kind].read_text(encoding="utf-8").split("\nkind:")[0]
    assert "`a_id,b_id`" in head and "UTF-8 CSV" in head and "입력 오류로 거부" in head


def test_중단_값의_꼴이_맞고_칸이_계획_안에_있다():
    p = _load(PLANS["rehearsal"])
    faults = list(p["faults"]["train"].items())
    for tag, attempts in p["faults"]["export"].items():
        faults += [(tag, a["fault"]) for a in attempts if "fault" in a]
    assert len(faults) == 5
    for tag, s in faults:
        stage, point, t, arg = s.split(":")
        assert point in FAULT_POINTS[stage], s
        assert t == tag and t in p["cells"], s
        assert ":" not in arg and arg, s


def test_계획은_기본_설정에_병합되지_않는다():
    """기본 설정은 계획의 키를 갖지 않고 계획 파일을 가리키지 않는다. 설정 폴더를 통째로 읽어 합치는 코드도 없다."""
    base_text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    base = _base()
    top = {row[0].split(".")[0] for row in PLAN_TABLE}
    assert not (top & set(base)), sorted(top & set(base))
    assert not ({"rehearsal_uni", "frame_diag_uni"} & set(re.findall(r"[a-z_]+(?=\.yaml)", base_text)))
    r = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py"],
                       cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        pytest.skip("git 으로 파일 목록을 낼 수 없다")
    files = [f for f in r.stdout.splitlines() if f and not f.startswith("tests/")]
    pat = re.compile(r"configs.{0,40}(glob|rglob|iterdir|listdir)\(|(glob|rglob|iterdir|listdir)\(.{0,40}configs")
    hits = [f for f in files if pat.search((REPO / f).read_text(encoding="utf-8", errors="replace"))]
    assert len(files) > 50, "검사 범위가 비었다"
    assert not hits, f"설정 폴더를 통째로 읽는 코드: {hits}"


# --- 통합형 설정 키 ------------------------------------------------------------------------

def test_통합형_설정_키가_명세와_같다():
    fb = _base()["fixed_before_main_runs"]
    uni = {k: v for k, v in fb.items() if k.startswith("uni_")}
    got = set(_flat(uni, leaves=UNI_DICT_VALUED))
    assert got == UNI_KEYS, f"더 있는 키 {sorted(got - UNI_KEYS)} · 빠진 키 {sorted(UNI_KEYS - got)}"


def test_템플릿_인자는_한_곳이다():
    fb = _base()["fixed_before_main_runs"]
    assert fb["uni_chat_template_kwargs"] == {"enable_thinking": False}
    assert "uni_template_mode" not in fb, "템플릿 표현이 둘이 됐다"
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    assert len(re.findall(r"^\s*uni_chat_template_kwargs:", text, re.M)) == 1


def test_정하지_않은_통합형_값은_null_이고_검출_예산을_가리키지_않는다():
    """통합형 예산을 검출 예산의 앵커로 채우면 두 예산이 같다는 결정이 된다. 그 결정은 아직 없다."""
    fb = _base()["fixed_before_main_runs"]
    for k in UNI_NULLS:
        assert _get(fb, k) is None, k
    assert fb["uni_model"]["id"] == "Qwen/Qwen3.5-4B" and fb["uni_batch_size"] == 1
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    block = text[text.index("  uni_train_budget:"):text.index("  uni_model:")]
    assert "*" not in block and "&" not in block
    assert fb["train_budget"] == {"num_rounds": 50, "local_epochs": 2, "total_epochs": 100}


def test_모델_판과_봉인_페어는_결정_19_의_값이고_계약서와_맞는다():
    """revision 은 감독 토큰 계측이 쓴 캐시 판과 같아야 한다(다르면 기대값을 다시 잰다). 페어는 봉인 계약서와 내용으로 맞댄다."""
    from data.frozen_guard import EXPECTED_SEALED
    from vlm.uni_config import pairs_snapshot_problems

    fb = _base()["fixed_before_main_runs"]
    assert fb["uni_model"]["revision"] == UNI_REVISION
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    tokens = text[text.index("  uni_expected_supervised_tokens:"):text.index("  uni_prompt_path:")]
    assert f"캐시 판 {UNI_REVISION}" in tokens, "감독 토큰을 잰 판과 고정한 판이 다르다"
    path, digest = fb["uni_pairs"]["path"], fb["uni_pairs"]["digest"]
    assert path.startswith(UNI_PAIRS_DIR + "/") and UNI_PAIRS_DIR in EXPECTED_SEALED
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    block = text[text.index("  uni_pairs:"):text.index("  uni_prompt_sha256:")]
    assert "SNAPSHOT.sha256" in block and "결정 19" in text[text.index("  uni_model:"):text.index("  uni_pairs:")]
    if not (REPO / UNI_PAIRS_DIR / "SNAPSHOT.sha256").is_file():
        pytest.fail(f"봉인 페어가 이 트리에 없다 — 자산 부재를 먼저 의심한다: {UNI_PAIRS_DIR}")
    assert pairs_snapshot_problems(REPO / path, digest) == []


def test_감독_토큰_기대값은_참여자_셋의_정수이고_출처를_적었다():
    """값은 학습 쪽 계측 산출 파일에서 읽어 넣는다. 그 파일이 이 트리에 있고 해시가 같으면 값까지 맞댄다."""
    import hashlib
    import json

    from vlm.uni_config import TOKEN_KEYS
    exp = _base()["fixed_before_main_runs"]["uni_expected_supervised_tokens"]
    assert isinstance(exp, dict) and tuple(exp) == tuple(TOKEN_KEYS), exp
    assert all(type(v) is int and v > 0 for v in exp.values()), exp
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    block = text[text.index("  uni_expected_supervised_tokens:"):text.index("  uni_prompt_path:")]
    assert TOKENS_SOURCE.rsplit("/", 1)[1] in block and TOKENS_SOURCE_SHA256 in block, "출처 주석이 없다"
    assert "캐시 판" in block, "모델 판 단서가 없다"
    src = REPO / TOKENS_SOURCE
    if not src.is_file():
        pytest.skip("계측 산출 파일이 이 트리에 없다 — 값 대조를 건너뛴다")
    assert hashlib.sha256(src.read_bytes()).hexdigest() == TOKENS_SOURCE_SHA256
    per = json.loads(src.read_text(encoding="utf-8"))["참여자별"]
    assert exp == {c: per[c]["supervised_tokens_per_epoch"] for c in TOKEN_KEYS}


def test_프로세서_설정은_파일럿의_해상도를_채점_입구가_아는_키로_적는다():
    """파일럿이 받은 크기(캐시 판의 기본값)를 `min_pixels` · `max_pixels` 로 고정한다.

    `size` 꼴은 채점 입구가 모르는 키로 거부하고 영상 처리기 크기까지 덮어쓴다. 입구의 실제 판독기로 읽어 본다 —
    입구가 아는 키가 바뀌면 이 시험이 먼저 떨어진다."""
    from evaluation.entry_gate import PROCESSOR_KWARG_FIELDS, read_processor_kwargs
    raw = (CONFIGS / "base.yaml").read_bytes()
    kw = _base()["fixed_before_main_runs"]["uni_processor_kwargs"]
    assert kw == {"min_pixels": 65536, "max_pixels": 16777216}, kw
    assert all(type(v) is int for v in kw.values()), kw
    assert set(kw) <= set(PROCESSOR_KWARG_FIELDS), sorted(set(kw) - set(PROCESSOR_KWARG_FIELDS))
    assert read_processor_kwargs(raw) == kw
    text = raw.decode("utf-8")
    block = text[text.index("  uni_processor_kwargs:"):text.index("  uni_expected_supervised_tokens:")]
    assert "11-0" in block and "C450D68B5693D7FF" in block, "출처 주석이 없다"


def test_값이_있는_통합형_키는_프롬프트_파일과_좌표_규약이다():
    """`null` 을 막는 것이 아니라 값이 그 둘인지 본다. 프롬프트 파일은 저장소가 추적하는 파일이고, 좌표 규약은
    좌표 모듈이 아는 값이다. 같은 뜻의 키가 둘이 되지 않는다 — YAML 은 겹친 키를 조용히 덮으므로 글자로도 본다."""
    from pathlib import PurePosixPath

    from vlm.coords import COORD_SPACES
    fb = _base()["fixed_before_main_runs"]
    assert {k: fb.get(k) for k in UNI_VALUES} == UNI_VALUES
    path = fb["uni_prompt_path"]
    assert "\\" not in path and not path.startswith("/") and ":" not in path \
        and ".." not in PurePosixPath(path).parts, f"저장소 루트 기준 POSIX 경로가 아니다: {path!r}"
    assert (REPO / path).is_file(), f"프롬프트 파일이 없다: {path}"
    assert fb["uni_coord_space"] in COORD_SPACES
    assert sorted(k for k in fb if "coord" in k) == ["uni_coord_space"]
    assert sorted(k for k in fb if "prompt" in k) == ["uni_prompt_path", "uni_prompt_sha256"]
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
    for k in UNI_VALUES:
        assert len(re.findall(rf"^\s*{k}:", text, re.MULTILINE)) == 1, k
    r = subprocess.run(["git", "ls-files", "--error-unmatch", "--", path], cwd=REPO, capture_output=True, check=False)
    if r.returncode not in (0, 1):
        pytest.skip("git 으로 추적 여부를 볼 수 없다")
    assert r.returncode == 0, f"프롬프트 파일이 추적되지 않는다: {path}"


def test_리허설의_간선은_동결된_판본의_파일이고_계약서와_바이트가_같다():
    """간선 파일은 판본의 계약서에 적힌 해시와 같고, 계획의 sha256 도 그 값이다. 머리행은 a_id,b_id 이고 짝은 정렬돼 있다."""
    import hashlib

    from data.frozen_guard import EXPECTED_SEALED
    from vlm.rehearsal_lists import read_edges

    spec = _load(PLANS["rehearsal"])["exclude"]["near_dup_edges"]
    assert set(spec) == {"path", "sha256"} and spec["path"].startswith(EDGES_DIR + "/")
    assert EDGES_DIR in EXPECTED_SEALED
    d = REPO / EDGES_DIR
    if not (d / "SNAPSHOT.sha256").is_file():
        pytest.fail(f"간선 판본이 이 트리에 없다 — 자산 부재를 먼저 의심한다: {EDGES_DIR}")
    contract = (d / "SNAPSHOT.sha256").read_text(encoding="utf-8")
    raw = (REPO / spec["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == spec["sha256"] and f"{spec['sha256']}  edges.csv" in contract
    edges = read_edges(raw)
    assert edges and all(a < b for a, b in edges) and edges == sorted(edges)
    assert "exclude.near_dup_edges" in PLAN_NULLS and _load(PLANS["frame_diag"])["exclude"]["near_dup_edges"] is None

