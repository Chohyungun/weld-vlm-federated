"""합성 벡터 측정이 **다시 돌려도 같은 파일**을 내는지 본다.

측정값은 백분위 규약을 고르는 근거로 인용된다. 돌릴 때마다 달라지면 인용한 값이 무엇이었는지
알 수 없다. 합성 벡터만 쓰므로 자산이 필요 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.probe import percentile_inf_vectors as piv


def _no_bare_constant(name: str):
    raise AssertionError(f"표준 JSON 밖의 값이 맨값으로 나갔다: {name}")


def test_다시_돌려도_같은_JSON_을_낸다(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert piv.main(["--out", str(a)]) == 0
    assert piv.main(["--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()
    # 표준 JSON 이어야 한다 — `NaN`·`Infinity` 가 맨값으로 나가면 읽는 쪽이 갈린다.
    got = json.loads(a.read_text(encoding="utf-8"), parse_constant=_no_bare_constant)
    # 측정 구간에서 파일을 하나도 열지 않았다는 기록이 산출물에 있다.
    assert got["io_guard"] == {"file_opens_during_measurement": 0}
