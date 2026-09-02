"""좌표 규약 고정 시험 — 총괄 판정 1 (2026-09-02) 의 이행을 코드에 못 박는다.

카나리아-1 실측(75번 §5)이 사전학습 모델의 네이티브 규약을 `ABS_ORIG` 로 판별했고,
총괄이 전환을 확정했다. 이 시험은 그 값이 조용히 되돌아가지 않게 한다.

**왜 시험까지 두는가.** 규약 불일치는 함정 #4 이고, 그 사고의 성질은 "틀려도 아무것도
멈추지 않는다"는 것이다. 파일럿에서 `NORM_1000` 으로 돌던 내내 전 시험이 초록이었다.
값이 상수 한 줄이라 되돌리기도 쉽다 — 되돌리면 여기서 죽어야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlm.coords import CoordCfg, ImageGeom, to_model, to_px  # noqa: E402


def test_통합형_타깃_규약은_ABS_ORIG_다():
    from vlm.pilot_vlm import COORD_CFG

    assert COORD_CFG.coord_space == "ABS_ORIG", (
        "총괄 판정 1(2026-09-02)로 ABS_ORIG 가 확정됐다. 되돌리려면 판정을 먼저 바꿔라."
    )


def test_ABS_ORIG_에서_정변환은_항등이다():
    """항등이라는 사실이 곧 라벨측 계통 오차가 0 이라는 뜻이다."""
    cfg = CoordCfg(coord_space="ABS_ORIG")
    geom = ImageGeom(orig_w=1280, orig_h=720)
    box = (13.0, 407.0, 1279.0, 719.0)
    assert to_model(box, geom, cfg) == box
    assert to_px(to_model(box, geom, cfg), geom, cfg) == box


def test_NORM_1000_은_왕복에서_손실이_난다__전환의_근거():
    """이빨 — ABS_ORIG 가 왜 나은지를 수치로 남긴다. 이 시험이 통과해야 위 항등이 값을 한다."""
    from vlm.coords import quantize

    geom = ImageGeom(orig_w=1280, orig_h=720)
    box = (13.0, 407.0, 1279.0, 719.0)

    norm = CoordCfg(coord_space="NORM_1000")
    back_norm = to_px(quantize(to_model(box, geom, norm)), geom, norm)
    absorig = CoordCfg(coord_space="ABS_ORIG")
    back_abs = to_px(quantize(to_model(box, geom, absorig)), geom, absorig)

    assert back_abs == box, "ABS_ORIG 는 양자화를 거쳐도 원본 픽셀 그대로다"
    assert back_norm != box, "NORM_1000 은 양자화에서 원본을 잃는다 — 그것이 전환 사유다"


def test_프롬프트가_좌표_규약을_ABS_ORIG_로_지시한다():
    """타깃 규약과 프롬프트 문장이 갈리면 모델은 두 지시를 동시에 받는다."""
    from vlm.pilot_vlm import PROMPT_PATH

    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "절대 픽셀" in text
    assert "0에서 1000" not in text, "정규화 지시가 남아 있으면 타깃과 프롬프트가 어긋난다"


def test_파일럿_프롬프트_v1_은_보존되어_있다():
    """v1 로 돈 파일럿 산출물의 `prompt_sha256` 을 사후에 재현할 수 있어야 한다."""
    v1 = Path("vlm/prompts/unified_pilot_v1.txt")
    assert v1.exists(), "덮어쓰지 않고 새 파일을 만든 것이 요점이다"
    assert "0에서 1000" in v1.read_text(encoding="utf-8")


def test_두_프롬프트는_좌표_문장만_다르다():
    """전환의 범위를 고정한다 — 좌표 말고 다른 것이 함께 바뀌면 여기서 잡힌다."""
    from vlm.pilot_vlm import PROMPT_PATH

    v1 = Path("vlm/prompts/unified_pilot_v1.txt").read_text(encoding="utf-8").splitlines()
    v2 = PROMPT_PATH.read_text(encoding="utf-8").splitlines()
    assert len(v1) == len(v2) == 3
    assert v1[0] == v2[0] and v1[1] == v2[1], "지시문·스키마 행은 그대로여야 한다"
    assert v1[2] != v2[2]
