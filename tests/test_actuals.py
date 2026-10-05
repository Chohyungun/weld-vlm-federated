"""곁 파일의 실제 값 ↔ 등록의 생성 쪽 값 — 쓰는 쪽과 읽는 쪽이 부르는 **같은 함수**(진입점 미니스펙 3판 1-8).

지키는 것.

1. 자료형까지 본다 — `256` 과 `"256"` 은 다르다, 참·거짓은 수가 아니다. **사전 · 목록의 안쪽까지**. 등록의 튜플은 JSON 목록과 같다.
2. 출발점 두 필드는 둘 다 싣고 `null` 도 같음으로 본다.
3. 에코 묶음은 에코 목록 해시와, 모델 묶음은 생성 목록 해시와 맞댄다.
4. `plan_sha256` 은 등록에 값이 있을 때만 맞대고, `train_rows_digest` 는 곁 파일에서 맞대지 않는다(어댑터 meta).
5. 어긋난 항목의 이름과 자료형 이름만 싣고 **값은 싣지 않는다.**
6. 묶음 검증이 **이 함수를** 쓴다 — 실제로 부른다.
7. 구현 식별자는 쓰는 쪽의 꼴(식별자 사전)에서 열쇠를 내 등록의 열쇠와 맞댄다. 승인 여부는 따로 본다.
8. 생성 쪽 칸마다 맞대는 자리가 정해져 있다.
"""

from __future__ import annotations

import hashlib

import pytest

from dataclasses import fields

from evaluation import actuals as A
from evaluation import bundle_v14
from evaluation.prereg_unified import GenerationSpec, UnifiedRegistration

IMPL = {"seam": "export_generator", "approved": True, "module": "vlm.export_run", "qualname": "load_generator",
        "source_path": "vlm/export_run.py", "in_repo": True, "blob_sha1": "a" * 40, "head_blob_sha1": "a" * 40}
KEY = "export_generator=vlm.export_run:load_generator@vlm/export_run.py"


def hx(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def reg(purpose: str = "main") -> UnifiedRegistration:
    r = UnifiedRegistration()
    g = r.generation
    g.purpose, g.list_split, g.snapshot_digest = purpose, "eval" if purpose == "main" else "val", hx("snap")
    g.eval_list_file_sha256, g.eval_list_set_sha256 = hx("gen-file"), hx("gen-set")
    g.echo_list_file_sha256, g.echo_list_set_sha256 = hx("echo-file"), hx("echo-set")
    g.prompt_sha256, g.chat_template_kwargs, g.gen_prefix_sha256 = hx("p"), {"enable_thinking": False}, hx("pre")
    g.max_new_tokens, g.decoding, g.batch_size, g.padding_side = 512, {"do_sample": False}, 8, "left"
    g.processor_config_sha256, g.processor_min_pixels, g.processor_max_pixels = hx("proc"), 3136, 921600
    g.patch_size, g.merge_size, g.coord_space, g.coord_cfg_hash = 16, 2, "ABS_ORIG", hx("coord")
    g.base_model_id, g.base_model_revision = "local/qwen3.5-4b", "frozen"
    g.start_checkpoint_sha256, g.init_adapter_digest = None, hx("init")
    g.train_config_sha256, g.budget_n, g.budget_r, g.budget_e = hx("cfg"), 30, 3, 10
    g.transformers_version, g.impl_ids = "5.15.0", (KEY,)
    return r


def side_of(r: UnifiedRegistration, mode: str = "model") -> dict:
    """등록값을 그대로 옮긴 곁 파일 — JSON 을 한 번 지난 꼴(튜플이 목록이 된다)."""
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in A.registered_values(r, mode=mode).items()}
    out["impl_ids"] = {"export_generator": dict(IMPL)}      # 쓰는 쪽은 식별자 사전으로 쓴다
    out["unrelated"] = 1
    return out


def test_같으면_빈_목록이다() -> None:
    r = reg()
    assert A.check_actuals(r, side_of(r), mode="model") == []


@pytest.mark.parametrize(("key", "bad"), [("max_new_tokens", "512"), ("batch_size", True),
                                          ("processor_min_pixels", 3136.0), ("patch_size", 15)])
def test_자료형까지_본다(key: str, bad) -> None:
    r = reg()
    side = side_of(r) | {key: bad}
    [m] = A.check_actuals(r, side, mode="model")
    assert m.item == key and m.reason == "등록값과 다르다"


def test_값은_싣지_않는다() -> None:
    r = reg()
    [m] = A.check_actuals(r, side_of(r) | {"max_new_tokens": "512"}, mode="model")
    d = m.as_detail()
    assert d == {"item": "max_new_tokens", "reason": "등록값과 다르다", "registered_type": "int",
                 "sidecar_type": "str", "equal": False}


def test_곁_파일에_없으면_이름을_준다() -> None:
    r = reg()
    side = side_of(r)
    del side["impl_ids"]
    assert [m.item for m in A.check_actuals(r, side, mode="model")] == ["impl_ids"]


def test_출발점_두_필드는_둘_다_싣고_null_도_같다() -> None:
    r = reg()
    side = side_of(r)
    assert side["start_checkpoint_sha256"] is None
    assert A.check_actuals(r, side, mode="model") == []
    del side["start_checkpoint_sha256"]
    assert [m.item for m in A.check_actuals(r, side, mode="model")] == ["start_checkpoint_sha256"]


def test_에코_묶음은_에코_목록_해시와_맞댄다() -> None:
    r = reg()
    echo = A.registered_values(r, mode="echo")
    assert echo["eval_list_file_sha256"] == r.generation.echo_list_file_sha256
    assert A.registered_values(r, mode="model")["eval_list_file_sha256"] == r.generation.eval_list_file_sha256
    assert echo["eval_list_set_sha256"] == r.generation.echo_list_set_sha256
    assert [m.item for m in A.check_actuals(r, side_of(r, "model"), mode="echo")] == \
        ["eval_list_file_sha256", "eval_list_set_sha256"], "모델 목록을 실은 곁 파일은 에코 모드에서 목록 해시 둘이 어긋난다"


def test_목적마다_비어도_되는_칸은_값이_있을_때만_맞댄다() -> None:
    r = reg("rehearsal")
    assert "plan_sha256" not in A.registered_values(r, mode="model")
    r.generation.plan_sha256 = hx("plan")
    assert A.registered_values(r, mode="model")["plan_sha256"] == hx("plan")


def test_학습_행_digest_는_곁_파일에서_맞대지_않는다() -> None:
    """쓰는 쪽 2판 반영판 §2-5 의 곁 파일 표에 없다 — 어댑터 meta 에만 있다(§2-7). 묶음 검증의 학습 단계가 본다."""
    r = reg("frame_diag")
    r.generation.train_rows_digest = hx("rows")
    assert "train_rows_digest" not in A.registered_values(r, mode="model")
    assert A.check_actuals(r, side_of(r), mode="model") == []
    assert A.GENERATION_ROUTE["train_rows_digest"].startswith("어댑터 meta")


def test_생성_쪽_칸마다_맞대는_자리가_있다() -> None:
    """칸이 늘면 여기서 떨어진다 — 대조 밖의 칸은 까닭을 적어야 한다(검수 16번 M-2)."""
    assert set(A.GENERATION_ROUTE) == {f.name for f in fields(GenerationSpec)}
    assert all(isinstance(v, str) and len(v) > 8 for v in A.GENERATION_ROUTE.values())
    r = reg("frame_diag")
    r.generation.plan_sha256 = hx("plan")
    compared = set(A.registered_values(r, mode="model")) | set(A.registered_values(r, mode="echo"))
    for name, how in A.GENERATION_ROUTE.items():
        if how.startswith("곁 파일") and not name.startswith("echo_list"):
            assert name in compared, name
        if how.startswith(("맞대지 않는다", "어댑터 meta")):
            assert name not in compared, name


@pytest.mark.parametrize(("key", "want", "got"), [
    ("decoding", {"do_sample": False}, {"do_sample": 0}),
    ("chat_template_kwargs", {"enable_thinking": False}, {"enable_thinking": 0}),
    ("decoding", {"do_sample": False, "num_beams": 1}, {"do_sample": False, "num_beams": 1.0}),
    ("decoding", {"a": [1, 2]}, {"a": [1, 2.0]}),
    ("decoding", {"a": {"b": True}}, {"a": {"b": 1}}),
    ("decoding", {"do_sample": False}, {"do_sample": False, "extra": 1}),
])
def test_자료형은_안쪽까지_본다(key: str, want, got) -> None:
    """검수 16번 M-1 — 겉층만 보면 `{"do_sample": 0}` 이 `False` 와 같다고 지난다."""
    [m] = A.compare_values({key: want}, {key: got})
    assert m.item == key


def test_안쪽의_튜플과_목록은_같다() -> None:
    assert A.compare_values({"x": {"a": (1, (2, 3))}}, {"x": {"a": [1, [2, 3]]}}) == []


# ---------------------------------------------------------------- 구현 식별자 (검수 16번 I-4)

def test_구현_식별자는_열쇠의_집합으로_맞댄다() -> None:
    r = reg()
    assert A.check_actuals(r, side_of(r) | {"impl_ids": [dict(IMPL)]}, mode="model") == [], "목록 꼴(어댑터 meta)도 받는다"
    other = dict(IMPL, qualname="other_generator")
    [m] = A.check_actuals(r, side_of(r) | {"impl_ids": [other]}, mode="model")
    assert m.item == "impl_ids" and "열쇠" in m.reason
    [m] = A.check_actuals(r, side_of(r) | {"impl_ids": [KEY]}, mode="model")
    assert m.item == "impl_ids" and "꼴" in m.reason, "이름 문자열은 쓰는 쪽의 꼴이 아니다"


@pytest.mark.parametrize(("over", "why"), [
    ({"approved": False}, "승인 목록 밖"), ({"in_repo": False}, "저장소 밖"),
    ({"blob_sha1": "b" * 40}, "커밋되지 않은 코드"), ({"blob_sha1": None, "head_blob_sha1": None}, "커밋되지 않은 코드"),
])
def test_승인된_실제_구현이_아닌_까닭(over: dict, why: str) -> None:
    assert A.impl_problems([dict(IMPL)]) == []
    [p] = A.impl_problems([dict(IMPL, **over)])
    assert why in p


@pytest.mark.parametrize("bad", [
    [], "x", [KEY], [dict(IMPL, extra=1)], [{k: v for k, v in IMPL.items() if k != "approved"}],
    [dict(IMPL, approved="yes")], [dict(IMPL, blob_sha1="A" * 40)], [dict(IMPL), dict(IMPL)],
    {"export_preflight": dict(IMPL)}, [dict(IMPL, seam="")],
])
def test_구현_식별자의_꼴(bad) -> None:
    assert A.impl_identifiers(bad) is None and A.impl_keys(bad) is None
    assert A.impl_problems(bad) == ["꼴이 아니다"]


def test_모드는_둘뿐이다() -> None:
    with pytest.raises(ValueError):
        A.registered_values(reg(), mode="probe")


def test_묶음_검증이_이_함수를_쓴다(tmp_path, monkeypatch) -> None:
    """쓰는 쪽의 시작 전 점검과 읽는 쪽의 묶음 검증이 **같은 함수**를 부른다(3판 1-8). 이름이 같은지가 아니라 **실제로 불리는지** 본다(M-11)."""
    from tests.test_bundle_v14 import make
    assert bundle_v14.compare_values is A.compare_values
    calls = []

    def spy(registered, actual):
        calls.append(set(registered))
        return A.compare_values(registered, actual)
    monkeypatch.setattr(bundle_v14, "compare_values", spy)
    make(tmp_path).verify()
    assert calls and "impl_ids" in calls[0]
