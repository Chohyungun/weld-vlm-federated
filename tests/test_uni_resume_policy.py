"""통합형 재개 신원에 결속한 **정책 판** — 같은 판의 재개와 구판 거부.

## 무엇을 막는가

재개 신원은 `run_id`·라운드·참여자·시드·예산·모델·페어 경로였다. 여기에 **셔플 정책 판·
채팅 템플릿 모드·생성 접두 판본**을 더했다. 이 셋이 없으면, 키와 모양이 같고
`supervised_tokens` 도 갖춘 **구판 체크포인트가 같은 `run_id` 로 새 경로에 들어간다.**
그러면 회계(`R × E = N`)는 맞는데 궤적이 두 프로토콜에 걸치고, 그 어긋남은 지표에 흔적을
남기지 않는다.

**기존 방어를 없는 것으로 보지 않는다.** 신원 불일치·텐서 구조 불일치·`supervised_tokens`
누락은 이전에도 막혔다. 이번에 더한 것은 *정책이 다른데 나머지가 같은* 경우이고, 옮긴 것은
필드 누락 검사를 **모델 상태 적용 앞으로** 보낸 것이다.

## 보증 범위

여기서 보는 것은 **신원과 거부 순서**다. 4B 전 구간의 수치적 재개 동치는 이 시험의 범위가
아니고 아직 시험되지 않았다 — 옵티마이저·스케줄 복원, 누적 gradient, 커널 비결정성이 모두
걸린다. 모델을 올리지 않고 CPU 로만 돈다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="체크포인트 저장·적재가 torch 를 쓴다")

from detection.resume import (ResumeCheckpointer, ResumeIdentity,  # noqa: E402
                              latest_resume)

#: 현행 정책. `vlm/pilot_vlm.py` 의 상수와 같은 값이어야 한다.
POLICY = {"shuffle_policy": "str-seed-v1",
          "template_mode": "enable_thinking=False",
          "gen_prefix_digest": "0123456789abcdef"}
BASE = {"run_id": "uni-t", "round_idx": 0, "client_idx": 0, "seed": 20260828,
        "total_epochs": 2, "local_epochs": 2, "model": "m", "data": "pairs.jsonl"}


def _ident(**over) -> ResumeIdentity:
    return ResumeIdentity(**{**BASE, **POLICY, **over})


class _Net:
    def state_dict(self):
        # 값은 **텐서**여야 한다 — `save()` 가 `serialize.state_dict_to_ndarrays` 로 넘기고 그 함수가
        # `.detach()` 를 부른다. 실제 경로의 `get_peft_model_state_dict` 도 텐서를 낸다.
        return {"w": torch.zeros(2)}

    def load_state_dict(self, sd, strict=True):
        pass


class _Opt:
    def state_dict(self):
        return {"state": {}, "param_groups": []}


class _Trainer:
    """`ResumeCheckpointer.save()` 가 읽는 모양만 갖춘 최소 대역.

    상태 사전은 텐서이고 optimizer 는 `state_dict()` 를 가진다 — `save()` 가 둘을 그대로 부른다.
    실제 통합형 경로(`vlm/pilot_vlm.py` 의 `_AdapterTrainerView`)가 그 모양이다.
    """

    def __init__(self, epoch: int) -> None:
        self.model = _Net()
        self.epoch = epoch
        self.start_epoch = 0
        self.n_optimizer_updates = 2 * (epoch + 1)
        self.optimizer = _Opt()


class _Counter:
    def __init__(self, n: int) -> None:
        self.n = n


def _save(d: Path, ident: ResumeIdentity, *, epoch: int, tokens: int | None) -> None:
    """실제 경로와 같은 방식으로 저장한다 — `vlm/pilot_vlm.py:423-425` 가 하는 것.

    `save()` 는 `**extra` 를 받지 않는다. 부가 항목은 `ckpt.extra` 딕셔너리에 담고
    호출 가능 객체로 부른다.
    """
    ck = ResumeCheckpointer(d, identity=ident, step_counter=_Counter(4),
                            state_dict_fn=lambda tr: tr.model.state_dict())
    if tokens is not None:
        ck.extra = {"supervised_tokens": tokens}
    ck.save(_Trainer(epoch))


# ---------------------------------------------------------------- ① 같은 판의 재개
def test_같은_판은_중단_전후로_이어진다(tmp_path):
    d = tmp_path / "r"
    ident = _ident()
    _save(d, ident, epoch=0, tokens=12345)

    state = latest_resume(d, identity=ident)
    assert state is not None, "같은 판인데 재개 상태를 못 찾는다"
    assert state.identity.mismatch(ident) == []
    assert state.payload["supervised_tokens"] == 12345
    assert state.epoch_done == 0


def test_정책_판이_신원에_실제로_저장된다(tmp_path):
    """저장·적재 왕복에서 세 값이 살아남아야 거부가 성립한다."""
    d = tmp_path / "r"
    ident = _ident()
    _save(d, ident, epoch=0, tokens=1)
    got = latest_resume(d, identity=ident).identity
    for k, v in POLICY.items():
        assert getattr(got, k) == v, f"{k} 가 왕복에서 사라졌다"


# ---------------------------------------------------------------- ② 구판 거부
@pytest.mark.parametrize("field", sorted(POLICY))
def test_정책이_다른_상태는_거부한다(tmp_path, field):
    """세 필드 **각각**이 거부 사유가 된다 — 하나만 걸려 있으면 나머지가 샌다."""
    d = tmp_path / "r"
    _save(d, _ident(**{field: "다른-판"}), epoch=0, tokens=1)
    with pytest.raises(ValueError, match="신원이 현재 실행과 다르다") as e:
        latest_resume(d, identity=_ident())
    assert field in str(e.value), f"거부 사유에 {field} 가 적히지 않았다"


def test_정책_필드가_없던_구판은_빈값으로_읽혀_거부된다(tmp_path):
    """뒤호환의 요점 — 필드가 없던 체크포인트는 `""` 가 되고, 값을 채운 새 실행과 어긋난다."""
    d = tmp_path / "r"
    old = ResumeIdentity(**BASE)          # 정책 세 필드를 주지 않는다 = 구판
    assert (old.shuffle_policy, old.template_mode, old.gen_prefix_digest) == ("", "", "")
    _save(d, old, epoch=0, tokens=1)
    with pytest.raises(ValueError, match="신원이 현재 실행과 다르다"):
        latest_resume(d, identity=_ident())


def test_검출_경로는_정책_필드를_비워_영향받지_않는다(tmp_path):
    """검출은 세 값을 쓰지 않는다. 비운 신원끼리는 그대로 이어져야 한다."""
    d = tmp_path / "r"
    det = ResumeIdentity(**BASE, profile="main")
    _save(d, det, epoch=0, tokens=1)
    assert latest_resume(d, identity=ResumeIdentity(**BASE, profile="main")) is not None


# ---------------------------------------------------------------- 거부 순서
def test_필드_누락_검사가_모델_적용_앞에_있다():
    """소스 순서를 본다 — 적용 뒤에 거부하면 거부할 가중치가 이미 얹힌다.

    실행으로 보려면 모델과 GPU 가 필요하므로 순서만 고정한다. 이전 판은 `apply_resume` 뒤에
    `supervised_tokens` 를 검사했다.
    """
    src = (Path(__file__).resolve().parents[1] / "vlm/pilot_vlm.py").read_text(encoding="utf-8")
    body = src[src.index("state = latest_resume("):src.index("start_ep = state.next_epoch")]
    i_check = body.index('"supervised_tokens" not in state.payload')
    i_apply = body.index("apply_resume(")
    assert i_check < i_apply, "필드 누락 검사가 모델 상태 적용보다 뒤에 있다"


def test_정책_상수가_시험과_같은_값이다():
    """상수를 바꾸면 이 시험이 먼저 깨진다 — 판을 바꾸는 것은 프로토콜을 바꾸는 것이다."""
    from vlm import pilot_vlm

    assert pilot_vlm.SHUFFLE_POLICY == POLICY["shuffle_policy"]
    assert pilot_vlm.TEMPLATE_MODE == POLICY["template_mode"]
    assert 'random.Random(f"{seed}:{ep}")' in (
        Path(__file__).resolve().parents[1] / "vlm/pilot_vlm.py").read_text(encoding="utf-8")


def test_접두_판본은_템플릿이_바뀌면_달라진다():
    """프롬프트 내용이 아니라 템플릿 구조의 판을 잡는지 본다."""
    from vlm.pilot_vlm import gen_prefix_digest

    class _Proc:
        def __init__(self, ids):
            self._ids = ids

        def apply_chat_template(self, *a, **kw):
            import torch
            return {"input_ids": torch.tensor([self._ids])}

    a = gen_prefix_digest(_Proc([1, 2, 3]))
    b = gen_prefix_digest(_Proc([1, 2, 3]))
    c = gen_prefix_digest(_Proc([1, 2, 4]))
    assert a == b and a != c
    assert len(a) == 16 and set(a) <= set("0123456789abcdef")
