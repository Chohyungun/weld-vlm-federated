"""2026-09-21 공개본 표현 정리가 **주석과 독스트링만** 바꿨다는 것을 기계가 확인한다.

## 왜 이 시험이 있나

v3 산출물은 채점 코드의 **파일별 sha256** 을 싣는다. "이 수치가 어느 코드에서 나왔는가" 에 답하는 사슬이 그것이다.
표현 정리가 그 파일들의 바이트를 바꿔 사슬이 끊겼다 — 주석만 고쳤어도 해시는 달라진다.

바이트를 되돌리면 공개본에 지워야 할 표기가 되살아난다. 그래서 되돌리지 않고 **사슬을 다른 방식으로 잇는다.**

## 이 시험이 보증하는 것 — 그리고 보증하지 않는 것

**보증한다**: 정리 전 바이트와 지금 바이트의 **구문 트리가 같다**(독스트링과 주석성 문자열을 뺀 뒤).
곧 실행되는 코드가 같다는 뜻이다. **2026-10-01 정리는 다르다** — 산출 문자열이 바뀐 일곱 파일은 문자열 리터럴까지 가린
트리가 같고 바뀐 리터럴의 자리가 고정돼 있다는 것까지, `provenance.py` 는 목록 구조가 바뀌어 바이트 고정으로 본다(아래 절).

**보증하지 않는다**: 바이트가 같다는 것. 같지 않다 — 그것이 이 시험이 있는 이유다.
`SCRUB_BASELINE` 의 기준점은 정리 직전(`6bde066`)이고, `MATCHED_V3_AT_BASELINE` 의 다섯 파일은 그 바이트가
v3 기록과 **같았다**는 것까지 확인한다. 그래서 사슬이 `v3 해시 → 정리 전 바이트 → 구문 트리 → 지금 바이트` 로 이어진다.

**문자열 상수가 바뀐 파일은 구문 트리도 달라진다.** 그런 파일은 되돌리지 않고 `AST_DIVERGED` 에 사유와 함께 남긴다 —
`evaluation/eval_set.py` 를 "v3 와 같은 파일" 목록에서 뺀 것과 같은 처리다. 갈라진 사실을 지우지 않는 것이 요점이다.

**독스트링만 있는 모듈**(`__init__.py` 셋)은 트리를 벗기면 `Pass` 하나만 남아 이 검사가 사실상 비어 있다.
그 파일들은 `tests/test_d1_baseline_frozen.py` 의 바이트 고정이 지킨다.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SCRUB_BASELINE: dict[str, str] = {
    "evaluation/__init__.py": "5c04595997820c90159baeb6d4411d9a479f4dc26f1ac532dd740dcb011f3e23",
    "evaluation/eval_set.py": "32f6550ca35ce0af4f5cbbd9f4af125bc35ae4b8bc8ce6445c44fe38e9de998b",
    "evaluation/gold.py": "ce25f83306b2d46fbe779cb4295e64d46fb7d78e7f33b15ba7628ed707149dc2",
    "evaluation/metrics/clause.py": "1f157f2175318cb9dae294aab60b8f0c75c584660dd68749a6eb162ddae789e8",
    "evaluation/provenance.py": "68621c50a38baa6e9720bd904c107c136ad5d3d4450ee5be0084bf825ddb7193",
    "evaluation/recovery_ci.py": "a1fa94e85b43c162c4bd11666dd0421dba3bea16a11f2b35ebcff2e7826e2665",
    "rag/__init__.py": "5c04595997820c90159baeb6d4411d9a479f4dc26f1ac532dd740dcb011f3e23",
    "scripts/probe/map50_independent.py": "cdd27fd6b016ab1c66bc7356155d38c9fecc088f8822a0fe3aefcbcde0d2e705",
    "scripts/probe/recovery_bootstrap.py": "8c6a34ee4371f7ae64a6a4fdcff3e04844a57071aa0329f49f957593a8f6eef1",
    "scripts/probe/run_judge_pilot.py": "cbe521c1947a20d0356ee513b9901a5577c74d5d6b8f9b2f67ae6a03df9affb2",
    "scripts/probe/score_cells.py": "626e430ae071e8f7b818e1d49a663aeaddd353b1b27b4687332e4aa68abfb94f",
    "tests/test_aggregate_seeds.py": "ec3fece93e76c2842477f97efbe605d9324513502682df6c406b518726d358f5",
    "tests/test_output_guards.py": "41d288343d96b2aa9370c101167b1c15d3829b70b8b21751f7af7445a78b60aa",
    "tests/test_scorer_provenance.py": "bf7ee190da7bc4c9e64d4c5bdb037ecb258d93d90d4f9f60d9679be382a63fcc",
    "tracking/__init__.py": "5c04595997820c90159baeb6d4411d9a479f4dc26f1ac532dd740dcb011f3e23",
}
"""정리 직전(`6bde066`)의 **구문 트리 지문** — 독스트링·주석성 문자열을 뺀 `ast.dump` 의 sha256."""

AST_DIVERGED: dict[str, str] = {
    "scripts/probe/aggregate_seeds.py":
        "VARIANCE_INTERPRETATION 은 주석이 아니라 집계 산출물에 실리는 값이다(`variance_interpretation` 필드). "
        "표현 정리가 그 문자열의 출처 표기를 고쳤다 — 되돌리면 공개되는 파일에 지워야 할 표기가 되살아나므로 "
        "되돌리지 않고 여기 남긴다. 같은 입력으로 집계를 다시 돌리면 이 한 필드가 옛 산출물과 다르다.",
    "tests/test_d1_baseline_frozen.py":
        "고정 해시 표를 정리 뒤 값으로 다시 박았다. 주석이 아니라 시험의 자료라 구문 트리가 달라지는 것이 정상이다.",
}
"""구문 트리까지 달라진 파일과 그 사유. **비우지 마라** — 갈라진 사실을 남기는 것이 이 표의 일이다."""

# ── 2026-10-01 역할 이름 정리와 채점 지문 목록의 재고정 ─────────────────────────────────────────
# 고정 해시 파일 열한 곳의 역할 이름을 걷고 `SCORER_FILES` 에 진입점을 더했다. 09-21 과 같은 방식이다 — 되돌리지 않고
# 정리 **직전**의 구문 트리 지문을 남겨 사슬을 잇는다: 09-21 기준점 → 10-01 정리 직전 → 지금.

ROLE_SCRUB_BEFORE: dict[str, str] = {
    "evaluation/discrimination.py": "06524073e9aef576d7af239b0fa41f497e7aaa7317d75cae031292eccbfbe7bb",
    "evaluation/gates.py": "d94cae5af2d0b550318a7eedb9e7522089168356e3fcce1060d1830b809828d6",
    "evaluation/metrics/detection.py": "79a77ab5e674ae613cf795236190479fbaf225b4aa5e2678a29292819e294807",
    "evaluation/params.py": "da45c54934185d55cca88262521a506365ec43c005d4288208e947ae1ecac9f4",
    "evaluation/prereg.py": "fe2bb9a62d93706827a0592fc270d3ca7878fc814c779428daea238dd8865abf",
    "evaluation/provenance.py": "68621c50a38baa6e9720bd904c107c136ad5d3d4450ee5be0084bf825ddb7193",
    "evaluation/recovery_ci.py": "a1fa94e85b43c162c4bd11666dd0421dba3bea16a11f2b35ebcff2e7826e2665",
    "evaluation/score.py": "8149c841e8311d0a3f6fb31f275a64cb4ca408a7d15ba8226c4e42566b5d2318",
    "scripts/probe/aggregate_seeds.py": "312bddeb62ea88875a51d69b914f2fa316b80477dca622a22d09afacf06dea6f",
    "scripts/probe/map50_independent.py": "cdd27fd6b016ab1c66bc7356155d38c9fecc088f8822a0fe3aefcbcde0d2e705",
    "scripts/probe/recovery_bootstrap.py": "8c6a34ee4371f7ae64a6a4fdcff3e04844a57071aa0329f49f957593a8f6eef1",
    "scripts/probe/score_cells.py": "626e430ae071e8f7b818e1d49a663aeaddd353b1b27b4687332e4aa68abfb94f",
}
"""2026-10-01 정리 직전(`d609c57`)의 구문 트리 지문 — 독스트링·주석성 문자열을 뺀 `ast.dump` 의 sha256."""

ROLE_SCRUB_DIVERGED: dict[str, str] = {
    "evaluation/gates.py":
        "게이트 사유 문자열 셋(`sweep_curve_recorded` · `macro_ap_baseline_paired` 의 막는 사유)에서 역할 이름을 뺐다. 게이트 결과에 실리는 문구라 사전실험을 다시 채점하면 그 사유 글이 옛 산출물과 다르다. 판정 · 조건은 같다",
    "evaluation/params.py":
        "산출물에 실리는 출처 문자열(`coord_space_source` 의 폴백 표기 · conf 출처 문구 · 운용점 역할 문구)에서 역할 이름을 뺐다. 값을 고르는 규칙은 같고 출처를 적는 글만 다르다",
    "evaluation/prereg.py":
        "`RECOVERY_CI_REGISTRATION['registered']` 의 글에서 역할 이름을 뺐다. 회복률 CI 산출물의 `registration` 블록에 실리는 값이라 다시 내면 그 한 칸의 글이 옛 산출물과 다르다. 판정 규칙의 수는 같다",
    "evaluation/provenance.py":
        "`SCORER_FILES` 에 통합형 본채점 진입점을 더했다. 사전실험 산출물이 지문을 낸 옛 목록은 `SCORER_FILES_AT_V3` 로 병기했다 — 그 산출물의 `rule` · `files` · `combined` 은 고치지 않는다. 목록 이름과 파생 튜플이 생겨 문자열을 가린 트리도 다르다 — 그래서 바이트로 고정한다(`tests/test_d1_baseline_frozen.py` 의 `FROZEN_SHA256`, 검수 16번 I-3)",
    "scripts/probe/aggregate_seeds.py":
        "집계 산출물의 판정 문구 셋 — `3_headline_adoption`(대표 지표 미결 문구) · `v[\"combined\"]` 의 게재 가능 분기 · Δ_AUC 의 `status` — 에서 역할 이름을 뺐다. 같은 입력으로 다시 집계하면 그 글만 옛 산출물과 다르다(2026-10-01 의 사유는 둘째를 `public_status` 라 잘못 적었다 — 그 값은 바뀌지 않았다, 검수 16번 M-10)",
    "scripts/probe/map50_independent.py":
        "검산 산출물의 `decision` · `spec` 글에서 역할 이름을 뺐다. 판정 · 대조는 같다",
    "scripts/probe/recovery_bootstrap.py":
        "CI 산출물의 결론 문구 · `spec` 글에서 역할 이름을 뺐다. 재표집 · 구간 계산은 같다",
    "scripts/probe/score_cells.py":
        "채점 산출물의 `ruling` · 곡선의 note · 운용점 문구 · 보조지표 note 에서 역할 이름을 뺐다. 채점 · 게이트의 계산은 같고 산출에 실리는 글만 다르다",
}
"""2026-10-01 정리로 구문 트리까지 달라진 파일과 그 사유. **비우지 마라.** 여기 없는 파일은 주석·독스트링만 바뀌었다."""

ROLE_SCRUB_BLOBS: dict[str, str] = {
    "evaluation/discrimination.py": "44bb28312a4c0453733dce2764489ac47c13dab7",
    "evaluation/gates.py": "57255bfb5511cbb71323c37d9d297b616755b25c",
    "evaluation/metrics/detection.py": "0dc9fe150b5f8d95be032f169cf727cf5f1678f8",
    "evaluation/params.py": "1437036ed555506d376c56638732058426c1e1cd",
    "evaluation/prereg.py": "b37fbbe3a044fb841f9c27c4dd2b6e378b1d3780",
    "evaluation/provenance.py": "800cd86cde5b086073fef042fee7d44df1f61205",
    "evaluation/recovery_ci.py": "aa91cf67db99a0e1721ab67fb629c65888f449aa",
    "evaluation/score.py": "eef32ecb7eb0e7e0be7225e0f36be4d88fa853ca",
    "scripts/probe/aggregate_seeds.py": "86d4e0aff61651fc242b4675d1c1a0d1b2c54271",
    "scripts/probe/map50_independent.py": "e9fbae75349648b4e0ca5c204254a69a7b05638a",
    "scripts/probe/recovery_bootstrap.py": "a9f28f3133c4fc666f532890a82e327939daa5d8",
    "scripts/probe/score_cells.py": "863eaab14f8516197328f3464f7101332b9fca0f",
}
"""정리 직전(`d609c57`) 바이트의 **git blob id**. 시험이 이 blob 을 읽어 바이트의 당시 값(`BEFORE_ROLE_SCRUB`) ·
구문 트리의 당시 값(`ROLE_SCRUB_BEFORE`) · 문자열을 가린 트리(`ROLE_SCRUB_MASKED`)를 **다시 잰다** — 상수와 상수를 맞대지 않는다(검수 16번 M-9)."""

ROLE_SCRUB_MASKED: dict[str, str] = {
    "evaluation/gates.py": "a5de51beeadd4405c491436c1e8799d8b65966e3b92be22cd17b58c37fde77d0",
    "evaluation/params.py": "64c1b46ed02fca56a4dbd5499a6f5d120e1dc7f75c6da47729f0613999e6cf92",
    "evaluation/prereg.py": "808cf53d99f43dcbba227dab832dbde9d9e40565bc587e560564cef47e228ede",
    "scripts/probe/aggregate_seeds.py": "3b2cb71d888d8c91c0be50187ab22ab14edf49144bc4e584958c5f4a07fb807b",
    "scripts/probe/map50_independent.py": "236c7e4a75adbf364de207ac1f0b2a02a696dc308aec0e9c98410b4ad9d484e2",
    "scripts/probe/recovery_bootstrap.py": "82178e89da5751dacbd99e9f404d193b03d8ab639a3e17bc3ac7e31a50f1006f",
    "scripts/probe/score_cells.py": "49966dc7d4437dc1a383662e20bdfe1f3dc0aed4480d987f47d39ca5cf779edf",
}
"""정리 직전의 **문자열을 가린 구문 트리** 지문 — 독스트링을 뺀 뒤 모든 문자열 리터럴(f-string 의 조각 포함)을 같은 자리표시자로 바꾼
`ast.dump` 의 sha256. 산출 문자열만 바뀐 일곱 파일은 지금도 이 값이어야 한다 — 조건 · 수치 상수 · 제어 흐름이 같다(검수 16번 I-2)."""

ROLE_SCRUB_LITERALS: dict[str, tuple[int, tuple[tuple[int, str, str], ...]]] = {
    "evaluation/gates.py": (250, (
        (143, "afed20b5218fca46", "44aef73e35e3c1f3"),
        (147, "39d0cdf6c581c267", "6f952c4a69b274f7"),
        (180, "62aed82a3ddfa93b", "15b8a152cbbe24d0"),
    )),
    "evaluation/params.py": (115, (
        (30, "1897f789e60df3a2", "f24c5abeb8e89f8e"),
        (33, "cc32e10a4ec8969f", "0e5cf027feddc1ed"),
        (36, "1897f789e60df3a2", "f24c5abeb8e89f8e"),
        (92, "a5a4c325849772b3", "2826e3f062015dad"),
    )),
    "evaluation/prereg.py": (62, (
        (15, "55b40d6e871187ef", "f074b6bfd4a79f29"),
    )),
    "scripts/probe/aggregate_seeds.py": (952, (
        (144, "7854274d91570b05", "d5d84a798b25ad0f"),
        (334, "73cb4297f0e126df", "2d52c6b392dab973"),
        (720, "54767f57bfa646cb", "b0a325b7aa252113"),
    )),
    "scripts/probe/map50_independent.py": (326, (
        (69, "7938713d114366bb", "992b8a7be306e5ed"),
        (105, "5fdea895af3d7b31", "6acd1de390e4f0a4"),
    )),
    "scripts/probe/recovery_bootstrap.py": (254, (
        (26, "40660a25c36b07da", "6da1665f5047bc06"),
        (41, "80935c07b16d9956", "993ee9e90a84061e"),
    )),
    "scripts/probe/score_cells.py": (717, (
        (8, "97add07038ef99a9", "0f6b6d9e56f7c2e2"),
        (64, "72aba70d8d94152d", "97f26cdfbecf59f0"),
        (85, "cb53628c75148444", "102ba1a6ff646925"),
        (314, "9860ab0a9bdb6f63", "be29fab2e08f7b45"),
    )),
}
"""파일 → (문자열 리터럴 수, 바뀐 리터럴의 `(자리, 정리 전 sha256 앞 16자, 지금 sha256 앞 16자)`). 자리는 독스트링을 뺀 트리를
`ast.walk` 로 돈 순서다. **바뀐 문자열의 목록**이다 — 사유는 `ROLE_SCRUB_DIVERGED` 의 그 파일 글이다. 글 자체는 싣지 않는다
(정리 전 글에 걷어 낸 표기가 있다)."""

MATCHED_V3_AT_BASELINE: dict[str, str] = {
    "evaluation/__init__.py": "6c126c2f52b08e89af54ce7133320d7b6c50ecab040e0ff5c6dd99a55380a8cc",
    "evaluation/eval_set.py": "eda6b2ca3bb2142a7be663cfe69f314e69e6df0585644eb4624078c76487bbea",
    "evaluation/gold.py": "26ec65a2e0f38c7a76a89780712d492894f493a022fc1d7909cb0749c9f82002",
    "evaluation/metrics/clause.py": "b3bcaa6860f0d9141000a093edc6541c0ce78ab0bf7a17f3b5376333013a2bdb",
    "evaluation/provenance.py": "59f3e140bbe5fb80118ea78e6b3f67c8bd4aa9a73e91e979d61b4188cf416c2c",
}
"""정리 직전 바이트의 sha256 — v3 산출물이 기록한 값과 **같았던** 다섯 파일.

이 표가 사슬의 첫 칸이다. 나머지 셋(`recovery_ci.py`·`recovery_bootstrap.py`·`score_cells.py`)은 정리 전에 이미
v3 와 달랐고 그 사실이 09-17 독립 검산 산출물의 `code.drift.changed` 에 기록돼 있다 — 표현 정리가 만든 차이가 아니다.
"""


def _strip(tree: ast.AST) -> ast.AST:
    """독스트링과 주석성 문자열(값으로 쓰이지 않는 바깥 문자열)을 뺀다. 주석은 애초에 트리에 없다."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.body = [
                n for n in node.body
                if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, str))
            ] or [ast.Pass()]
    return tree


def ast_digest(source: str) -> str:
    return hashlib.sha256(ast.dump(_strip(ast.parse(source))).encode()).hexdigest()


def _literals(source: str) -> list[str]:
    tree = _strip(ast.parse(source))
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def masked_ast_digest(source: str) -> str:
    """문자열 리터럴을 모두 같은 자리표시자로 바꾼 구문 트리의 지문 — 문자열 밖이 같은지만 본다."""
    tree = _strip(ast.parse(source))
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = "<s>"
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _h16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("rel", sorted(SCRUB_BASELINE))
def test_정리는_주석과_독스트링만_바꿨다(rel: str) -> None:
    """실행되는 코드가 정리 전과 같다. **바이트가 같다는 말이 아니다**(모듈 독스트링).

    2026-10-01 정리로 구문 트리가 달라진 파일은 **그 정리 직전의 지문**으로 본다 — 09-21 기준점과 10-01 정리 직전이
    같았다는 것까지가 이 시험의 몫이고, 그 뒤의 갈림은 `ROLE_SCRUB_DIVERGED` 가 사유와 함께 든다."""
    now = ROLE_SCRUB_BEFORE[rel] if rel in ROLE_SCRUB_DIVERGED else ast_digest((REPO / rel).read_text(encoding="utf-8"))
    assert now == SCRUB_BASELINE[rel], (
        f"{rel} 의 구문 트리가 2026-09-21 정리 직전과 다르다. 주석 밖을 고쳤거나 그 뒤 코드가 바뀐 것이다. "
        "의도한 변경이면 기준점을 다시 박되, v3 산출물과의 사슬이 어떻게 되는지 먼저 보고한다"
    )


def test_구문_트리가_갈린_파일은_목록에_적혀_있다() -> None:
    """정리 대상인데 여기도 저기도 없는 파일이 생기면 실패한다 — 조용히 빠지는 자리를 없앤다."""
    assert not (SCRUB_BASELINE.keys() & AST_DIVERGED.keys())
    for rel, reason in AST_DIVERGED.items():
        assert (REPO / rel).exists(), f"{rel} 이 없다"
        assert len(reason) > 40, f"{rel} 의 사유가 너무 짧다"


def test_기준점_바이트가_v3_를_채점한_코드였다() -> None:
    """사슬의 첫 칸. 이것이 서야 '구문 트리가 같다' 가 v3 까지 이어진다."""
    art = REPO / "outputs/main_d/seed1/score_cells_v3.json"
    if not art.exists():
        pytest.skip(f"공유 산출물 없음: {art.relative_to(REPO)} — 대형 자산이 있는 체크아웃에서 돈다")
    files = json.loads(art.read_text(encoding="utf-8"))["scorer_code"]["files"]
    assert {rel: files[rel] for rel in MATCHED_V3_AT_BASELINE} == MATCHED_V3_AT_BASELINE


def test_검사가_물_수_있다() -> None:
    """이빨 시험 — 코드를 한 줄 바꾸면 잡히고, 독스트링만 바꾸면 안 잡힌다."""
    src = (REPO / "evaluation/eval_set.py").read_text(encoding="utf-8")
    base = ast_digest(src)
    assert base == SCRUB_BASELINE["evaluation/eval_set.py"]
    assert ast_digest(src.replace('ISO_SEP = ";"', 'ISO_SEP = "|"', 1)) != base
    assert ast_digest(src.replace('"""매니페스트', '"""(주석만 바꾼다) 매니페스트', 1)) == base


@pytest.mark.parametrize("rel", sorted(set(ROLE_SCRUB_BEFORE) - set(ROLE_SCRUB_DIVERGED)))
def test_역할_이름_정리는_주석과_독스트링만_바꿨다(rel: str) -> None:
    """2026-10-01 — 이 파일들은 실행되는 코드가 정리 직전과 같다."""
    assert ast_digest((REPO / rel).read_text(encoding="utf-8")) == ROLE_SCRUB_BEFORE[rel]


@pytest.mark.parametrize("rel", sorted(ROLE_SCRUB_DIVERGED))
def test_역할_이름_정리로_갈린_파일은_실제로_갈렸고_사유가_있다(rel: str) -> None:
    """목록이 참이어야 한다 — 갈리지 않은 파일을 갈렸다고 적으면 주석만 고친 파일을 검사 밖으로 빼는 길이 된다."""
    assert ast_digest((REPO / rel).read_text(encoding="utf-8")) != ROLE_SCRUB_BEFORE[rel]
    assert len(ROLE_SCRUB_DIVERGED[rel]) > 40


def test_역할_이름_정리의_두_표는_맞물린다() -> None:
    assert set(ROLE_SCRUB_DIVERGED) <= set(ROLE_SCRUB_BEFORE)
    assert len(ROLE_SCRUB_BEFORE) == 12
    assert set(ROLE_SCRUB_MASKED) == set(ROLE_SCRUB_LITERALS) == set(ROLE_SCRUB_DIVERGED) - {"evaluation/provenance.py"}
    assert set(ROLE_SCRUB_BLOBS) == set(ROLE_SCRUB_BEFORE)


# ---------------------------------------------------------------- 산출 문자열이 바뀐 파일 — 문자열 밖은 같다 (검수 16번 I-2)

@pytest.mark.parametrize("rel", sorted(ROLE_SCRUB_MASKED))
def test_역할_이름_정리로_바뀐_것은_문자열뿐이다(rel: str) -> None:
    """조건을 뒤집거나 수치 상수를 바꾸면 여기서 떨어진다 — "구문 트리가 다르다(!=)" 만으로는 그것을 막지 못했다."""
    assert masked_ast_digest((REPO / rel).read_text(encoding="utf-8")) == ROLE_SCRUB_MASKED[rel]


@pytest.mark.parametrize("rel", sorted(ROLE_SCRUB_LITERALS))
def test_바뀐_문자열의_자리와_새_글이_고정돼_있다(rel: str) -> None:
    """바뀐 리터럴은 고정한 자리뿐이고, 그 자리의 지금 글은 고정한 글이다. 다른 리터럴을 고치면 떨어진다."""
    n, changed = ROLE_SCRUB_LITERALS[rel]
    now = _literals((REPO / rel).read_text(encoding="utf-8"))
    assert len(now) == n
    assert [(i, _h16(now[i])) for i, _, _ in changed] == [(i, new) for i, _, new in changed]


def test_문자열을_가린_트리는_문자열_밖의_변경을_잡는다() -> None:
    """이빨 시험 — 문자열을 바꾸면 같고, 조건 · 수 · 이름을 바꾸면 다르다."""
    src = (REPO / "evaluation/gates.py").read_text(encoding="utf-8")
    base = masked_ast_digest(src)
    assert base == ROLE_SCRUB_MASKED["evaluation/gates.py"]
    lit = next(x for x in _literals(src) if len(x) > 20)
    assert masked_ast_digest(src.replace(repr(lit)[1:-1], repr(lit)[1:-1] + "x", 1)) == base
    assert masked_ast_digest(src.replace(" == ", " != ", 1)) != base
    assert masked_ast_digest(src.replace("return ", "return not ", 1)) != base


# ---------------------------------------------------------------- 당시 값의 사슬 — 커밋된 blob 에서 다시 잰다 (검수 16번 M-9)

def _blob(blob_id: str) -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", blob_id], capture_output=True, check=False)
    except OSError:
        pytest.skip("git 을 부를 수 없다 — 이 사슬은 저장소 체크아웃에서 돈다")
    if out.returncode != 0 and subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-dir"],
                                              capture_output=True, check=False).returncode != 0:
        pytest.skip("git 저장소가 아니다 — 이 사슬은 저장소 체크아웃에서 돈다")
    assert out.returncode == 0, f"정리 직전 blob {blob_id} 이 저장소에 없다 — 사슬의 고리가 끊겼다"
    return out.stdout.decode("utf-8").replace("\r\n", "\n")


@pytest.mark.parametrize("rel", sorted(ROLE_SCRUB_BLOBS))
def test_당시_값은_정리_직전_blob_에서_다시_잰_값이다(rel: str) -> None:
    """바이트의 당시 값 → 구문 트리의 당시 값 → (문자열을 가린 트리) 를 **같은 바이트**에서 잰다."""
    from tests.test_d1_baseline_frozen import BEFORE_ROLE_SCRUB
    old = _blob(ROLE_SCRUB_BLOBS[rel])
    assert hashlib.sha256(old.encode("utf-8")).hexdigest() == BEFORE_ROLE_SCRUB[rel]
    assert ast_digest(old) == ROLE_SCRUB_BEFORE[rel]
    if rel in ROLE_SCRUB_MASKED:
        assert masked_ast_digest(old) == ROLE_SCRUB_MASKED[rel]
        n, changed = ROLE_SCRUB_LITERALS[rel]
        lits = _literals(old)
        assert len(lits) == n and [(i, _h16(lits[i])) for i, _, _ in changed] == [(i, o) for i, o, _ in changed]
