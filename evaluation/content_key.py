"""생성 파일의 **내용 해시** — 판정 한 번 규칙의 열쇠. 07번 §32-2 · 진입점 미니스펙 3판 8-1 의 2.

원시 바이트 해시(`generations_sha256`)는 줄마다 `latency_ms` 를 품는다. 같은 모델 출력을 다시 내보내면 지연이 달라
해시가 달라지고, 그 해시를 열쇠로 쓰면 "같은 묶음의 판정은 한 번" 이 복사한 파일에만 걸린다. 그래서 열쇠는
**실행마다 바뀌는 칸을 뺀 내용**의 해시다. 원시 바이트 해시는 묶음 검증과 곁 파일 대조에 그대로 쓴다.

| | 키 |
|---|---|
| 내용 | `image_id` · `image_sha256` · `text` · `bbox_px_parsed` · `parse_error` · `gen_stop` · `n_new_tokens` · `n_bad_items_dropped` · `model_input_wh` |
| 내용 — 있으면 | `image_grid_thw`(프로세서가 그 이미지에 낸 격자 `[t, h, w]` — 같은 이미지 · 같은 프로세서 설정이면 같은 값이고 실행마다 바뀌지 않는다. `model_input_wh` 가 그 격자에서 나온다). 줄에 격자가 없으면 해시는 내지만, **격자 없는 묶음은 묶음 검증이 모든 목적에서 거부한다** — 격자는 생성 시점에만 나오고 사후 보충을 받지 않는다(07번 §37) |
| 제외 | `latency_ms`(실행마다 다르다) · `raw_output_ref`(경로다) · `coord_space` · `coord_cfg_hash`(코드의 표지와 작업 트리 지문이다 — 출력이 같아도 주석 한 줄에 바뀐다) |

**분류는 닫혀 있다.** 줄에 세 목록 밖의 키가 있으면 해시를 내지 않는다 — 새 키가 조용히 열쇠 밖으로 새지 않게.
내용 키가 빠진 줄도 멈춘다(값이 null 이면 null 로 남긴다). "있으면" 의 키는 있을 때만 내용에 들고, 없으면 없는 대로 — 있는 줄과 없는 줄은 열쇠가 다르다.
"""

from __future__ import annotations

import hashlib
import json

CONTENT_KEYS: tuple[str, ...] = (
    "image_id", "image_sha256", "text", "bbox_px_parsed", "parse_error",
    "gen_stop", "n_new_tokens", "n_bad_items_dropped", "model_input_wh",
)
OPTIONAL_CONTENT_KEYS: tuple[str, ...] = ("image_grid_thw",)
"""있으면 내용에 드는 키. 쓰는 쪽은 생성기가 격자를 낼 때만 줄에 싣는다(실제 생성기와 에코는 늘 낸다)."""
EXCLUDED_KEYS: tuple[str, ...] = ("latency_ms", "raw_output_ref", "coord_space", "coord_cfg_hash")


class ContentKeyError(ValueError):
    """생성 파일의 줄이 정규화 규칙으로 해시를 낼 수 없는 꼴이다. 몇째 줄인지와 사유를 든다."""


def generations_content_sha256(raw: bytes) -> str:
    """생성 파일의 바이트 → 내용 해시(소문자 64자리).

    줄마다 JSON 으로 읽어 내용 키만 남긴 객체를 정규 JSON(키 정렬 · UTF-8 · 구분자 `,` `:` · 공백 없음)으로 적고,
    파일의 줄 순서대로 LF 로 이어 끝에 LF 하나를 둔 바이트의 sha256 이다. 빈 파일은 받지 않는다.

    Raises:
        ContentKeyError: 줄이 JSON 객체가 아니거나, 분류 밖 키가 있거나, 내용 키가 빠졌으면.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ContentKeyError("UTF-8 이 아니다") from None
    if not text.endswith("\n") or text == "\n" or "\r" in text:
        raise ContentKeyError("줄이 LF 로 끝나는 비지 않은 파일이 아니다")
    known = set(CONTENT_KEYS) | set(OPTIONAL_CONTENT_KEYS) | set(EXCLUDED_KEYS)
    out: list[str] = []
    for n, line in enumerate(text[:-1].split("\n"), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ContentKeyError(f"{n}번째 줄이 JSON 이 아니다") from None
        if not isinstance(row, dict):
            raise ContentKeyError(f"{n}번째 줄이 객체가 아니다")
        unknown = sorted(set(row) - known)
        if unknown:
            raise ContentKeyError(f"{n}번째 줄에 분류되지 않은 키가 있다: {unknown} — 정규화 규칙을 먼저 개정한다")
        lacking = [k for k in CONTENT_KEYS if k not in row]
        if lacking:
            raise ContentKeyError(f"{n}번째 줄에 내용 키가 없다: {lacking}")
        body = {k: row[k] for k in CONTENT_KEYS} | {k: row[k] for k in OPTIONAL_CONTENT_KEYS if k in row}
        out.append(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    return hashlib.sha256(("\n".join(out) + "\n").encode("utf-8")).hexdigest()
