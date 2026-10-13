"""채팅 템플릿 모드 고정 시험 — 2026-09-21 총괄 판정 ①.

감독 마스킹은 `_encode` 가 낸 `prompt_len` **길이로** 자른다. 이 산술은 "생성 접두가 학습
렌더의 토큰 접두와 같다"는 전제 위에 있다. Qwen3.5 계열의 채팅 템플릿은 `enable_thinking`
**미지정 시의 기본값이 모델마다 반대**여서(0.8B·2B 는 비생각, 4B 는 생각) 4B 에서만 그 전제가
깨진다. 실제로 4B 의 감독 토큰이 행당 2개 많았고, 길이로 자른 자리가 `</think>` 앞이었다.

여기서 고정하는 것은 넷이다(11번 §1-3).
1. 세 모델 모두 `enable_thinking=False` 에서 **접두 등식**이 성립한다 — 토큰 id 비교다.
2. **미지정이 4B 에서 깨진다** — 상류 판이 바뀌어 기본값이 또 뒤집히면 이 시험이 먼저 깨진다.
3. 코드가 두 호출에 모두 인자를 명시한다(미지정에 기대지 않는다). 2026-10-01 부터는 **한 원본 객체**를
   두 호출이 같이 받는다 — 본실험은 설정의 `uni_chat_template_kwargs` 를 넘긴다(리허설 2판 §2-5).
4. 생성 경로(export)도 같은 모드를 쓴다.

1·2 는 가중치를 받지 않고 토크나이저·프로세서만 쓴다. 로컬 캐시가 없으면 건너뛴다 —
망을 타는 시험을 회귀에 넣지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAIN = "Qwen/Qwen3.5-4B"
PILOT = "Qwen/Qwen3.5-0.8B"


def _prompt() -> str:
    from vlm.pilot_vlm import PROMPT_PATH

    return PROMPT_PATH.read_text(encoding="utf-8")


#: 프로세서를 로컬에서 만드는 데 드는 파일. 셋이 모두 HF 캐시에 있을 때만 시험을 돌린다.
_CACHE_FILES = ("tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja")


def _cached(model_id: str) -> bool:
    """캐시에 프로세서 파일이 있는가 — **캐시 부재만** 건너뛰는 근거다. 망을 타지 않는다."""
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(model_id, f), str) for f in _CACHE_FILES)


def _checked(model_id: str) -> dict:
    """프로세서 경로의 접두 등식. **캐시가 없을 때만** 건너뛴다.

    앞 판은 프로세서를 만들고 템플릿을 부르는 동안의 모든 예외를 캐시 부재처럼 건너뛰었다 — 캐시가 있는데
    템플릿 호출이 깨져도 실패 대신 건너뜀으로 보였다. 이제 캐시가 있으면 예외는 그대로 시험 실패다.
    """
    pytest.importorskip("transformers")
    if not _cached(model_id):
        pytest.skip(f"{model_id} 의 프로세서 파일이 로컬 HF 캐시에 없다: {_CACHE_FILES}")
    from scripts.probe.template_prefix import check_processor

    return check_processor(model_id, _prompt())


def test_캐시가_있으면_템플릿_호출의_결함은_건너뛰지_않고_실패한다(monkeypatch):
    """캐시 부재만 좁게 건너뛴다 — 앞 판에서는 이 예외가 건너뜀으로 접혔다."""
    import scripts.probe.template_prefix as tp

    mod = sys.modules[__name__]
    monkeypatch.setattr(mod, "_cached", lambda m: True, raising=False)

    def boom(*a, **k):
        raise RuntimeError("템플릿 호출 결함")

    monkeypatch.setattr(tp, "check_processor", boom)
    try:
        _checked(MAIN)
    except BaseException as e:                 # noqa: BLE001 - 건너뜀(Skipped)도 잡아 가른다
        assert type(e) is RuntimeError, f"결함이 {type(e).__name__} 로 접혔다"
    else:
        raise AssertionError("예외가 나지 않았다")


@pytest.mark.parametrize("model_id", [PILOT, MAIN])
def test_비생각_모드에서_접두_등식이_성립한다(model_id):
    out = _checked(model_id)
    row = out["enable_thinking=False"]
    assert row["생성접두가_학습렌더의_토큰접두인가"], (
        f"{model_id}: 비생각 모드인데 생성 접두가 학습 렌더의 토큰 접두가 아니다 — "
        "감독 마스킹의 전제가 깨진다"
    )
    assert row["n_supervised"] > 0


def test_미지정은_4B_에서_깨진다__반례():
    """기본값에 기대면 안 된다는 근거. 상류 기본값이 또 바뀌면 여기가 먼저 깨진다."""
    out = _checked(MAIN)
    assert not out["미지정"]["생성접두가_학습렌더의_토큰접두인가"], (
        "4B 의 미지정 기본값이 비생각으로 바뀐 것 같다 — 사실이면 11번 §1 을 갱신한다. "
        "그래도 인자 명시는 유지한다(판마다 갈리는 값에 기대지 않는다)"
    )
    assert out["미지정"]["n_supervised"] == out["enable_thinking=False"]["n_supervised"] + 2, (
        "미지정과 비생각의 감독 토큰 차이가 2 가 아니다 — 템플릿 구조가 달라졌다"
    )


def _template_calls(path: str, func: str | None = None) -> list[dict]:
    """`apply_chat_template` 호출의 키워드 인자를 AST 로 모은다.

    문자열 세기로 검사하면 **주석에 적힌 인자까지 센다** — 실제로 그렇게 헛실패했다.
    호출의 인자만 본다.
    """
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    scope: ast.AST = tree
    if func is not None:
        scope = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == func)
    out = []
    for node in ast.walk(scope):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply_chat_template"):
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            v = kw.get("enable_thinking")
            out.append({"line": node.lineno,
                        "enable_thinking": (v.value if isinstance(v, ast.Constant)
                                            else ("<없음>" if v is None else "<상수아님>"))})
    return out


def _double_star_names(path: str, func: str) -> list[tuple[int, str]]:
    """`apply_chat_template(..., **X)` 의 X 를 모은다. X 가 이름이 아니면 그 식의 꼴을 적는다."""
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func)
    out = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "apply_chat_template"):
            stars = [k.value for k in node.keywords if k.arg is None]
            out.append((node.lineno, ast.unparse(stars[0]) if stars else "<없음>"))
    return out


def test_학습_코드의_두_호출이_같은_인자_객체를_받는다():
    """템플릿 인자의 **원본은 하나다**(리허설 2판 §2-5). 두 호출이 같은 이름의 객체를 풀어 받고,
    글자로 적은 `enable_thinking` 을 따로 두지 않는다 — 둘을 섞으면 원본이 둘이 된다.

    이 시험의 앞 판(두 호출에 `enable_thinking=False` 를 글자로 요구)은 설정의 인자를 넘기는 길을 막았다.
    """
    calls = _template_calls("vlm/pilot_vlm.py", "_encode")
    assert len(calls) == 2, f"호출 수가 달라졌다 — 시험을 다시 본다: {calls}"
    for c in calls:
        assert c["enable_thinking"] == "<없음>", (
            f"{c['line']}행이 enable_thinking 을 따로 적었다 — 원본이 둘이 된다")
    stars = _double_star_names("vlm/pilot_vlm.py", "_encode")
    assert [s for _, s in stars] == ["kwargs", "kwargs"], f"두 호출의 인자 객체가 같지 않다: {stars}"


def test_기본_인자는_비생각_모드다():
    """설정을 넘기지 않은 경로(파일럿)의 값. 미지정에 기대지 않는다는 판정 ① 을 지킨다."""
    from vlm import pilot_vlm

    assert pilot_vlm.DEFAULT_CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
    assert pilot_vlm._template_kwargs(None) == {"enable_thinking": False}
    assert pilot_vlm.TEMPLATE_MODE == pilot_vlm.template_mode_of({"enable_thinking": False})
    with pytest.raises(TypeError):
        pilot_vlm._template_kwargs("enable_thinking=False")


def test_접두_지문도_같은_원본을_받는다():
    stars = _double_star_names("vlm/pilot_vlm.py", "gen_prefix_digest")
    assert [s for _, s in stars] == ["_template_kwargs(chat_template_kwargs)"], stars


def test_생성_경로도_같은_모드다():
    calls = _template_calls("scripts/pilot_export_vlm.py")
    assert calls, "생성 경로에서 apply_chat_template 호출을 찾지 못했다"
    for c in calls:
        assert c["enable_thinking"] is False, f"{c['line']}행: {c['enable_thinking']!r}"
