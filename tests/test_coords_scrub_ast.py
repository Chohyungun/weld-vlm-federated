"""`vlm/coords.py` 의 09-21 주석 정리가 **주석만** 바꿨다는 것을 기계가 확인한다.

## 왜 이 파일에 따로 필요한가

`coords_source_sha256()` 은 이 파일의 **바이트 전체**를 해시해 `coord_cfg_hash` 의 입력에 넣는다.
그래서 주석 한 줄만 고쳐도 `coord_cfg_hash` 가 달라진다. 실제로 달라졌다.

| | 값 |
|---|---|
| 정리 전 바이트 | `43d1821f…` (09-02 `dffd3db` 이후) |
| 지금 바이트 | `06eb8289…` |
| `coord_cfg_hash(ABS_ORIG)` 전 → 후 | `c598d549…` → `0cc62b38…` |

**동결 산출물이 옛 값을 기록하고 있다.** 검출 세 시드의 export 15 벌이 `coord_cfg_hash c598d549`
다. 바이트를 되돌리면 공개본에 지워야 할 표기가 되살아나므로 되돌리지 않고, 평가 쪽이
`tests/test_scrub_was_comment_only.py` 에서 한 것과 **같은 방식으로 사슬을 잇는다** — 구문
트리가 같다는 것을 고정한다. 한쪽 소관에만 있는 방어는 없는 것과 같다.

사슬: `동결 export 의 c598d549` → `정리 전 바이트 43d1821f` → **구문 트리** → `지금 바이트 06eb8289`.

## 보증하는 것과 하지 않는 것

**보증한다**: 독스트링·주석을 벗긴 구문 트리가 정리 전과 같다. 곧 실행되는 코드가 같다.
문자열 상수가 바뀌면 이 검사가 잡는다(규약 이름·경계 상수가 주석 정리에 섞여 들어가는 것을 막는다).

**보증하지 않는다**: 바이트가 같다는 것. 같지 않고, 그것이 이 시험이 있는 이유다.
값의 이력은 `vlm/coords_fixtures/HASH_RECORD.yaml` 에 병기했다 — 형식은
`data/mock/FROZEN_RECORD.yaml` 을 따르고(당시 값 · 지금 값 · 바뀐 것 · 결과 불변의 근거),
한쪽만 남으면 아래 시험이 실패한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_scrub_was_comment_only import ast_digest  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TARGET = "vlm/coords.py"

#: 정리 **직전**(`83199bd` 시점, 바이트 `43d1821f…`) 소스의 구문 트리 다이제스트.
#: 갱신하려면 그 시점의 트리와 실제로 같은지 확인한 값이어야 한다 — 지금 파일에서 다시
#: 뽑아 넣으면 이 시험이 아무것도 보증하지 않게 된다.
AST_BASELINE = "a8245ff82daf86ad7bbae55f7297fb56e4e3b75733d798572f57ceb0de1508d2"

#: 사슬에 적히는 값들. 시험이 아니라 기록이지만, 문서와 어긋나면 눈에 띄게 여기 둔다.
BYTES_BEFORE = "43d1821fa8a0482f4e9140741968b77169beeed9cbf554bd616986ad1eb9693f"
CFG_HASH_BEFORE = "c598d5490e781c1add7fe54cf265dff59ff353a5f9bf8c1bf65f7989c1bc3fac"


def test_정리는_주석만_바꿨다():
    src = (REPO / TARGET).read_text(encoding="utf-8")
    assert ast_digest(src) == AST_BASELINE, (
        f"{TARGET} 의 구문 트리가 정리 전과 다르다 — 주석 정리에 코드 변경이 섞였다. "
        "섞인 것이 의도한 변경이면 기준값을 갱신하고 이유를 적는다"
    )


def _record() -> dict:
    import yaml

    d = yaml.safe_load((REPO / "vlm/coords_fixtures/HASH_RECORD.yaml").read_text(encoding="utf-8"))
    return d["files"][TARGET]


def _now_sha256() -> str:
    import hashlib

    return hashlib.sha256((REPO / TARGET).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_기록이_당시_값과_지금_값을_함께_적는다():
    """**한쪽만 있으면 실패한다.**

    당시 값을 지우면 "동결 export 가 어느 바이트로 돌았는가" 에 답할 수 없고, 지금 값을 적지
    않으면 기록이 현행 파일을 가리키지 못한다. 형식은 `data/mock/FROZEN_RECORD.yaml` 을 따른다.
    """
    v = _record()
    after = v.get("after_public_cleanup")
    assert after is not None, "정리 뒤 값이 없다 — 당시 값만 남으면 기록이 현행 파일을 가리키지 못한다"
    assert v["sha256"] == BYTES_BEFORE, "정리 전 바이트가 기록에서 사라졌다 — 덮어쓰지 않는다"
    assert after["sha256"] != v["sha256"] and after["git_blob"] != v["git_blob"]
    for key, n in (("sha256", 64), ("git_blob", 40)):
        for src in (v, after):
            assert len(src[key]) == n and set(src[key]) <= set("0123456789abcdef")
    assert after.get("changed") and after.get("evidence"), "무엇이 바뀌었는지와 그 근거가 없다"


def test_기록의_지금_값이_실제_파일과_같다():
    assert _record()["after_public_cleanup"]["sha256"] == _now_sha256(), (
        "기록이 낡았다 — 파일을 고치고 HASH_RECORD.yaml 을 갱신하지 않은 것이다"
    )


def test_갈라진_파생값이_기록에_남아_있다():
    """결과가 불변이라는 것이 해시가 같다는 뜻은 아니다. 갈라진 값을 지우지 않는다."""
    v = _record()
    assert v["derived"]["coord_cfg_hash_ABS_ORIG"] == CFG_HASH_BEFORE, (
        "동결 export 가 기록한 coord_cfg_hash 가 이력에서 빠졌다"
    )
    div = v["after_public_cleanup"]["diverged"]
    assert div["coord_cfg_hash_ABS_ORIG"] != CFG_HASH_BEFORE
    assert div.get("consequence"), "갈라진 값이 소비처에 무엇을 뜻하는지 적혀 있지 않다"


def test_근거에_적은_구문트리_값이_실제와_같다():
    """`evidence` 가 값을 적는데 그 값이 실제와 다르면 근거가 아니다."""
    src = (REPO / TARGET).read_text(encoding="utf-8")
    assert ast_digest(src) in _record()["after_public_cleanup"]["evidence"]


def test_SHA256SUMS_본문이_지금_파일을_가리킨다():
    sums = (REPO / "vlm/coords_fixtures/SHA256SUMS").read_text(encoding="utf-8")
    body = [ln for ln in sums.splitlines() if ln and not ln.startswith("#")]
    line = next(ln for ln in body if ln.endswith(TARGET))
    assert line.split()[0] == _now_sha256(), f"SHA256SUMS 의 {TARGET} 항이 낡았다"
    assert "HASH_RECORD.yaml" in sums, "이력의 단일 소스를 가리키는 줄이 없다"


def test_검사가_물_수_있다():
    """기준값이 우연히 맞는 것이 아님을 보인다 — 평가 쪽 시험과 같은 이빨 검사."""
    src = (REPO / TARGET).read_text(encoding="utf-8")
    base = ast_digest(src)
    assert ast_digest(src.replace('"ABS_ORIG"', '"ABS_RESIZED"', 1)) != base
    assert ast_digest(src.replace("return 0.5", "return 0.25", 1)) != base
    assert ast_digest(src.replace('"""좌표 규약 단일 모듈', '"""(주석만) 좌표 규약 단일 모듈', 1)) == base
