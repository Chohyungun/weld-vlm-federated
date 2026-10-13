"""본실험 D4 페어의 독립 재구현 대조 — 06번 미니스펙 §5 관점 C (2026-09-21).

    python scripts/probe/pairs_independent.py \\
        --manifest-dir data/interim/manifest_v1 --limits corpus/rules/limits_v0_pilot.csv \\
        --pairs <빌드 산출 디렉터리> --out outputs/pairs_independent/<새 이름>

빌더를 **import 하지도 읽지도 않고** 매니페스트·허용치 표·사상표에서 이미지마다 기대 레코드의 구조화 부분을 다시 내고,
빌드 산출(`pairs.jsonl`·`discarded.jsonl`)의 전 레코드와 맞댄다.

**종료 코드가 상태를 가른다 — 봉인해도 되는 것은 `0` 하나뿐이다.**
`0` 전량 일치 · `1` 불일치 · `2` 대조 범위 제한 · `3` 허용 범주만 남음.
필수 파일 부재와 **`--pairs` 를 주지 않은 실행**(기대 레코드만 내는 용도)이 `2` 에 들어간다.
`summary.json` 의 `sealable`·`coverage`·`verdict`·`not_compared` 는 **어느 경로로 끝나든 늘 실린다** —
그 넷이 없는 산출을 봉인 증거로 받을 수 없기 때문이다(교차 검수 I-2: 대조를 하나도 안 한 실행이 0 을 내고 넷을 빼먹었다).

09-17 의 교훈(57번): 알고리즘만 다시 짜면 입력 규약의 오류를 못 잡는다. 그래서 이 모듈은 **입력 규약까지 직접 읽는다** —
열 이름, 재질·코드 표기, 불리언 표기, `ann_id` 문자열 정렬, 좌표 자료형. 프로젝트의 적재기(`data.manifest_io`)도 쓰지 않는다.
매니페스트 README 는 직접 읽기를 금지하지만(계약 #2), 이 직접 읽기는 **검산에 한한 예외**다(2026-09-21 결정).
예외가 막으려던 사고(빈 문자열 → NaN)는 `csv` 모듈로 전 열을 문자열 그대로 읽어 피한다.

평가 격리: eval 행은 **식별자 네 축(`image_id`·`group_id`·`rel_path`·`sha256`)만** 집합으로 남긴다. 교차 0 을 세는 데 쓴다.
id 만 보면 같은 파일이 다른 id 로 양쪽에 있어도 겹침 0 이 나오므로 경로와 내용 해시까지 본다. eval 의 라벨·결함 코드·
크기는 보관하지 않는다. 무는 자리는 셋이다 — eval 행이 `images` 에 들어가지 않고, 색인의 이미지가 `images` 에 없으면
`InputContractError` 이며, 적재가 끝난 뒤와 기대 레코드를 만든 뒤 `assert_isolated()` 가 네 축의 교집합을 본다.

레코드 규약의 출처는 B 의 입력 규약 문서 **10번**(`10_페어입력규약_B.md`, 2026-09-21)과 06번 미니스펙이다. 코드는 읽지 않았다.
규약이 정하지 않은 경우를 만나면 **추측하지 않고 멈춘다**(`AmbiguousContract`) — 모호함 자체가 발견이기 때문이다.
표준 라이브러리와 PyYAML 만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import operator
import re
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import yaml

SELECTED_METHODS = frozenset({"RT", "ALL"})
"""행 선택 기본 필터 — 허용치 표 안내문 '유의 사항'(스펙 §1-2 5a). VT 행은 RT 주 실험에서 고르지 않는다."""

KNOWN_OPS = frozenset({"le", "lt"})
"""06번 §5 표 3번: `ge` 는 스키마에 없다. 모르는 부등호는 조용히 '이하' 로 떨어뜨리지 않고 예외를 낸다."""

GRID = Decimal("0.01")
"""허용치 표 안내문 '전사 규약': 원문 폐구간 (a, b] 를 반열림 [a+0.01, b+0.01) 로 환산했다."""

DISCARD_NO_VALID_GEOMETRY = "defect_label_no_valid_geometry"
"""봉인본 `pairs_pilot_v1/discarded.jsonl` 에서 읽은 사유 문자열. 결함 라벨인데 유효 기하가 하나도 없는 이미지."""

DISCARD_BBOX_BOUNDS = "bbox_out_of_bounds"
"""이 모듈의 어휘다. 빌더의 사유 문자열은 모른다 — 폐기 **여부**만 맞대고 사유는 참고로 싣는다."""

MANIFEST_COLS = ("image_id", "rel_path", "sha256", "width_px", "height_px", "material", "has_defect", "n_defects",
                 "iso_codes", "group_id", "split", "client")
ANN_COLS = ("ann_id", "image_id", "iso_code", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
            "major_axis_px", "equiv_diameter_px", "geom_valid")
LIMIT_COLS = ("rule_id", "canonical", "scope", "defect_code", "material", "inspection_method", "thickness_min",
              "thickness_max", "quality_scheme", "limit_rule", "limit_value", "limit_factor", "limit_cap",
              "ratio_basis", "limit_op", "unit", "clause_id")

SPLITS = frozenset({"train", "val", "eval"})
"""분할 어휘를 닫는다. `Train`·`test`·` eval` 은 어느 쪽에도 들지 않아 조용히 빠진다."""

CLIENTS = ("C1", "C2", "C3")
"""참여자 어휘. 다른 값이 오면 회계가 조용히 갈라지므로 멈춘다."""

ISOLATION_AXES = ("image_id", "group_id", "rel_path", "sha256")
"""평가 격리를 보는 네 축. 식별자만 본다 — 같은 파일이 다른 id 로 양쪽에 있으면 id 축만으로는 겹침 0 이 나온다."""

MATERIAL_ANY = "ALL"
"""B 10번 §5: 행 선택의 재질 축은 `{레코드 재질, ALL}` 이다. 지금 표에 `ALL` 행은 없다."""

COORD_SPACE = "ABS_ORIG"
"""B 10번 §1: 레코드가 좌표 규약을 최상위에 들고 다닌다. 상자는 원본 픽셀이다(06번 §5 회귀 표 9번)."""

SKIP_COUNT_FIELD = "n_annotations_skipped_geom_invalid"
"""기하 무효 어노테이션 수. 레코드에서는 **그 이미지 하나의 수**이고 0 이면 키가 없다(B 10번 §1).
같은 이름이 `PAIRS_META.json` 최상위에도 있는데 그것은 **빌드 총계**다 — 층을 섞지 않는다."""

META_PATHS_CHECKED = "image_paths_checked"
META_SNAPSHOT = "input_snapshot"
META_LIMITS = "limits_table"
META_VALIDATED_BY = "validated_by"
NOT_COMPARED_ALL = "pairs.*"
"""`--pairs` 를 주지 않아 **아무것도 대조하지 않은** 상태. 축 하나가 빠진 것과 전부 빠진 것을 같은 목록에서 가른다."""

KNOWN_VALIDATED_BY = frozenset({"rule"})
"""검증 주체 선언 가운데 **내가 확인할 수 있는 것**. 내가 한 것도 규칙 재생성이므로 `rule` 만 받는다.
사람 검수 주장은 빌드 산출물 안에 뒷받침이 없어 통과시키지 않는다 — 문자열을 본 것이 검수의 증거는 아니므로,
받아 주는 대신 **거부**한다(검토 §27-19 의 `metadata_validated_by`: 그때 이 대조기는 통과시켰다)."""
"""빌드가 경로 실재를 **몇 개 보았는지** 싣는 자리. 참·거짓만 싣던 때에는 검사를 꺼도 \"확인했다\" 가 남았다.
나는 파일시스템을 다시 보지 않고, 이 주장의 건수를 내가 따로 센 모집단과 맞댄다."""

META_SKIP_IN_RECORDS = "n_annotations_skipped_in_records"
"""메타 총계 가운데 채택 레코드에 실린 몫. `총계 = Σ(레코드별 값) + 폐기된 이미지 몫`."""

TOP_KEY_ORDER = ("image_id", "image_path", "client", "split", "material", "width_px", "height_px", "coord_space",
                 "skeleton", "target_text", SKIP_COUNT_FIELD)
SKELETON_KEY_ORDER = ("defects", "verdict", "verdict_mode", "clauses", "candidate_rules", "uncited_codes")
"""B 10번 §1 의 고정 키 순서. 결정론 계약(06번 §5 관점 G)의 일부라 **있는 키의 상대 순서**를 대조한다."""

TOP_FIELDS = ("material", "width_px", "height_px", "coord_space")
SKELETON_FIELDS = ("clauses", "candidate_rules", "uncited_codes")
"""자리가 정해진 필드. 다른 자리에서 발견되면 `field_place:<이름>` 으로 센다."""

DEFECT_KEY_ORDER = ("type", "bbox_px", "size_px", "size_mm")
"""B 10번 — `defects` 항목의 키와 순서. `size_mm` 은 늘 `null` 이다(두께·화소당 실치수가 전량 결측 — 함정 10)."""

EXIT_MATCH = 0
EXIT_MISMATCH = 1
EXIT_PARTIAL = 2
EXIT_ALLOWED = 3
"""종료 코드가 상태를 가른다. **`0` 은 봉인해도 되는 상태 하나뿐이다** — 부분 대조(2)와 허용 불일치(3)에는 주지 않는다.
`0` 하나로 넷을 읽던 때에는 \"필수 파일이 없어 못 봤다\" 도 0 이었다."""

UNCITED_MATERIAL = "material_not_covered"
UNCITED_CODE = "code_not_covered"
"""B 10번 §1 의 미특정 사유 두 가지 — 재질을 덮는 행이 아예 없는 경우와 그 코드 행만 없는 경우."""


REQUIRED_MEMBERS = ("manifest.csv", "annotations.csv", "data_capabilities.yaml")
"""계약서(`SNAPSHOT.sha256`)에 **반드시** 있어야 하는 구성원. 없으면 멈춘다.

계약서가 검증할 파일을 스스로 고르면 빠진 입력은 파일별 해시 대조에서도 빠진다. 받는 입력의 집합은
정해 둔다 — 필수 셋, 선택 하나(`OPTIONAL_MEMBERS`), **그 밖의 구성원은 거부**한다(검토 E-1)."""

OPTIONAL_MEMBERS = ("tiles.csv",)
"""있으면 파일별 해시를 맞대고 지문에 넣는 구성원. 없어도 받는다."""

MEMBER_ORDER = REQUIRED_MEMBERS + OPTIONAL_MEMBERS
"""지문을 계산하는 **정해진 순서**. 계약서에 줄이 적힌 순서와 무관하다 — 줄 순서만 바꾼 계약서는 같은
지문을 낸다. 있는 구성원만 이 순서로 넣는다."""

DIGEST_LINE_PREFIX = "# snapshot_digest "
"""지문 줄의 머리. 이것으로 시작하는 줄만 지문 줄이다 — 다른 `#` 줄은 받지 않는다."""

DIGEST_RULE = ("구성원을 정해진 순서(manifest.csv · annotations.csv · data_capabilities.yaml · tiles.csv 가 "
               "있으면)로 두고, 줄마다 '<sha256>  <이름>\\n' 을 이은 바이트의 sha256")
"""합산 지문의 규칙. 동결본 두 곳(본 매니페스트·파일럿)의 기록값과 같다 —
시험 `test_지문을_다시_계산하면_동결본의_기록과_같다`."""


class InputContractError(ValueError):
    """입력이 규약과 다르다 — 열 누락, 표기 이탈, 자료형 이탈."""


class AmbiguousContract(ValueError):
    """규약이 정하지 않은 경우를 만났다. 추측하지 않고 멈춘다."""


@dataclass(frozen=True)
class ImageRow:
    image_id: str
    split: str
    client: str
    material: str
    rel_path: str
    path_key: str
    sha256: str
    width_px: int
    height_px: int
    has_defect: bool
    n_defects: int
    iso_codes: tuple[str, ...]
    group_id: str


@dataclass(frozen=True)
class Ann:
    ann_id: str
    iso_code: str
    geom_valid: bool
    bbox: tuple[int, int, int, int] | None
    major_axis_px: str
    equiv_diameter_px: str


@dataclass(frozen=True)
class LimitRow:
    rule_id: str
    clause_id: str
    defect_code: str
    material: str
    inspection_method: str
    canonical: str
    quality_scheme: str
    thickness_min: Decimal
    thickness_max: Decimal | None
    limit_rule: str
    limit_value: Decimal | None
    limit_factor: Decimal | None
    limit_cap: Decimal | None
    ratio_basis: str
    limit_op: str
    unit: str


# ---------------------------------------------------------------- 입력 규약


def _bool(text: str, where: str) -> bool:
    if text == "True":
        return True
    if text == "False":
        return False
    raise InputContractError(f"{where}: 불리언 표기는 'True'/'False' 여야 한다 — {text!r}")


def _int(text: str, where: str) -> int:
    if not re.fullmatch(r"-?\d+", text):
        raise InputContractError(f"{where}: 정수 표기가 아니다 — {text!r}")
    return int(text)


def _dec(text: str, where: str) -> Decimal | None:
    if text == "":
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise InputContractError(f"{where}: 수 표기가 아니다 — {text!r}") from exc


def _token(text: str, where: str) -> str:
    if text == "" or text != text.strip():
        raise InputContractError(f"{where}: 빈 값이거나 앞뒤 공백이 있다 — {text!r}")
    return text


def _path(text: str, where: str) -> str:
    """상대 경로만 받는다. **표기는 그대로 돌려준다** — 레코드 대조는 매니페스트 표기와 글자까지 같아야 한다."""
    value = _token(text, where)
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise InputContractError(f"{where}: 절대 경로다 — {value!r}")
    if ".." in value.replace("\\", "/").split("/"):
        raise InputContractError(f"{where}: 상위 경로 참조가 있다 — {value!r}")
    return value


def _path_key(value: str) -> str:
    """격리 비교용 열쇠. 구분자와 대소문자만 맞춘다 — 같은 파일을 다른 표기로 적어도 같은 것으로 본다."""
    return value.replace("\\", "/").lower()


def _code(text: str, where: str) -> str:
    """결함 코드는 **문자열**이다. `2011.0`·`02011` 처럼 수로 읽혔다 돌아온 흔적을 막는다."""
    if not re.fullmatch(r"[1-9]\d*", text):
        raise InputContractError(f"{where}: 결함 코드 표기가 아니다 — {text!r}")
    return text


def _reader(path: Path, required: tuple[str, ...]):
    handle = open(path, encoding="utf-8", newline="")  # noqa: SIM115 — 호출자가 닫는다
    reader = csv.DictReader(handle)
    missing = [c for c in required if c not in (reader.fieldnames or [])]
    if missing:
        handle.close()
        raise InputContractError(f"{path.name}: 열이 없다 — {missing}")
    return handle, reader


def verify_contract(manifest_dir: Path) -> dict[str, str]:
    """`SNAPSHOT.sha256` 의 파일별 해시를 다시 계산해 맞댄다. 바이트만 읽는다(내용 해석 없음).

    줄은 셋 가운데 하나여야 한다 — 빈 줄, 지문 줄(`DIGEST_LINE_PREFIX` 로 시작, 한 번만), 구성원 줄
    (`<sha256>  <이름>`, 이름은 `MEMBER_ORDER` 안, 한 번만). **지문 줄이 아닌 주석 줄은 받지 않는다** —
    받으면 그 줄이 지문 줄을 가리거나 뒤에 붙어 지문 대조를 건너뛰게 만든다(검토 E-1). 적힌 지문을
    **읽지 못한 것**(머리는 있는데 값이 64자리 소문자 16진이 아님)은 멈추고, 지문 줄이 **없는 것**은
    받아서 메타의 지문을 미대조로 남긴다 — 둘을 같은 칸에 넣지 않는다.
    """
    snap = manifest_dir / "SNAPSHOT.sha256"
    out: dict[str, str] = {}
    for number, line in enumerate(snap.read_text(encoding="utf-8").splitlines(), 1):
        where = f"SNAPSHOT.sha256:{number}"
        if not line.strip():
            continue
        if line.startswith("#"):
            if not line.startswith(DIGEST_LINE_PREFIX.rstrip()):
                raise InputContractError(f"{where}: 지문 줄이 아닌 주석 줄이다 — {line!r}")
            if "#digest_line" in out:
                raise InputContractError(f"{where}: 지문 줄이 둘 이상이다")
            value = line[len(DIGEST_LINE_PREFIX):].strip() if line.startswith(DIGEST_LINE_PREFIX) else ""
            if not _is_sha256(value):
                raise InputContractError(f"{where}: 적힌 지문을 읽지 못했다 — {line!r}")
            out["#digest_line"] = line.strip()
            continue
        parts = line.split()
        if len(parts) != 2 or not _is_sha256(parts[0]):
            raise InputContractError(f"{where}: 구성원 줄의 꼴이 '<sha256>  <이름>' 이 아니다 — {line!r}")
        recorded, name = parts
        if name not in MEMBER_ORDER:
            raise InputContractError(f"{where}: 받는 구성원이 아니다 — {name!r} (받는 것 {list(MEMBER_ORDER)})")
        if name in out:
            raise InputContractError(f"{where}: 구성원이 두 번 적혔다 — {name!r}")
        h = hashlib.sha256()
        with open(manifest_dir / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != recorded:
            raise InputContractError(f"{name}: SNAPSHOT.sha256 과 다르다 — 동결본이 아니다")
        out[name] = recorded
    missing = [m for m in REQUIRED_MEMBERS if m not in out]
    if missing:
        raise InputContractError(f"SNAPSHOT.sha256 에 필수 구성원이 없다 — {missing}")
    # 적힌 지문을 **다시 계산한 값**과 맞댄다. 적힌 문자열끼리만 맞대면 계약서의 지문과 메타의 지문을
    # 같은 오기로 바꿔도 통과한다(검토 E-1). 파일별 해시는 위에서 이미 바이트로 확인했다.
    declared = snapshot_digest(out)
    if declared is not None and declared != recompute_snapshot_digest(out):
        raise InputContractError("SNAPSHOT.sha256 의 snapshot_digest 가 같은 파일의 파일별 해시에서 다시 계산한 "
                                 "값과 다르다 — 계약서가 스스로 어긋난다")
    return out


def _is_sha256(text: str) -> bool:
    """64자리 소문자 16진. 대문자·자릿수 이탈은 읽지 못한 것으로 본다."""
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def snapshot_digest(contract: dict[str, str]) -> str | None:
    """`SNAPSHOT.sha256` 의 `# snapshot_digest <값>` 줄에서 **적힌 값**을 뽑는다. 없으면 None.

    이 값을 그대로 믿지 않는다 — `verify_contract()` 가 `recompute_snapshot_digest()` 와 맞대 확인한 뒤에만
    이 함수의 반환이 입력의 지문이다. 적힌 문자열이 같다는 것만으로 무결성을 확인한 것이 아니다.
    """
    line = contract.get("#digest_line") or ""
    parts = line.lstrip("#").split()
    return parts[1] if len(parts) >= 2 and parts[0] == "snapshot_digest" else None


def recompute_snapshot_digest(contract: dict[str, str]) -> str:
    """계약서의 파일별 해시에서 합산 지문을 **다시 계산**한다(`DIGEST_RULE`).

    **정해진 순서**(`MEMBER_ORDER`)를 따르고 계약서에 줄이 적힌 순서는 보지 않는다. `contract` 는
    `verify_contract()` 의 반환이다 — 파일별 해시가 이미 바이트로 확인된 값이다.
    """
    body = "".join(f"{contract[name]}  {name}\n" for name in MEMBER_ORDER if name in contract)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_label_codes(label_map: Path) -> frozenset[str]:
    """사상표의 L2 결함 유형에서 ISO 코드 집합을 읽는다. 코드 문자열을 이 모듈에 적어 두지 않는다(불변조건 1-8)."""
    doc = yaml.safe_load(label_map.read_text(encoding="utf-8"))
    codes: set[str] = set()
    for key, spec in doc["defect_types"].items():
        codes.add(_code(str(spec["iso_code"]), f"label_map.{key}.iso_code"))
        for alt in spec.get("iso_code_alt") or []:
            codes.add(_code(str(alt), f"label_map.{key}.iso_code_alt"))
    return frozenset(codes)


def load_manifest(manifest_dir: Path) -> tuple[dict[str, ImageRow], dict[str, frozenset[str]]]:
    """train·val 행과, eval 의 **식별자 네 축** 집합을 돌려준다. eval 의 라벨·코드·크기는 보관하지 않는다."""
    handle, reader = _reader(manifest_dir / "manifest.csv", MANIFEST_COLS)
    rows: dict[str, ImageRow] = {}
    eval_keys: dict[str, set[str]] = {axis: set() for axis in ISOLATION_AXES}
    n_split: Counter[str] = Counter()
    with handle:
        for n, r in enumerate(reader, start=2):
            where = f"manifest.csv:{n}"
            image_id = _token(r["image_id"], where + " image_id")
            split = r["split"]
            if split not in SPLITS:
                raise InputContractError(f"{where}: split 표기 — {split!r}")
            n_split[split] += 1
            keys = {"image_id": image_id, "group_id": _token(r["group_id"], where + " group_id"),
                    "rel_path": _path_key(_path(r["rel_path"], where + " rel_path")),
                    "sha256": _token(r["sha256"], where + " sha256")}
            if split == "eval":
                if r["client"] != "":
                    raise InputContractError(f"{where}: eval 행에 client 가 붙어 있다")
                for axis, value in keys.items():
                    eval_keys[axis].add(value)
                continue
            if image_id in rows:
                raise InputContractError(f"{where}: image_id 중복")
            if r["client"] not in CLIENTS:
                raise InputContractError(f"{where}: 참여자 어휘 — {r['client']!r}")
            codes = tuple(_code(c, where + " iso_codes") for c in r["iso_codes"].split(";")) if r["iso_codes"] else ()
            if _bool(r["has_defect"], where + " has_defect") != (_int(r["n_defects"], where + " n_defects") > 0):
                raise InputContractError(f"{where}: has_defect 와 n_defects 가 어긋난다")
            rows[image_id] = ImageRow(
                image_id=image_id, split=split, client=r["client"],
                material=_token(r["material"], where + " material"),
                rel_path=_path(r["rel_path"], where + " rel_path"), path_key=keys["rel_path"],
                sha256=keys["sha256"],
                width_px=_int(r["width_px"], where + " width_px"), height_px=_int(r["height_px"], where + " height_px"),
                has_defect=_bool(r["has_defect"], where + " has_defect"),
                n_defects=_int(r["n_defects"], where + " n_defects"), iso_codes=codes,
                group_id=keys["group_id"])
    if not n_split["eval"]:  # 겹칠 것이 없으면 "겹침 0" 이 아무것도 말하지 않는다
        raise InputContractError("eval 행이 0건이다 — 격리 검사가 공허해진다")
    if not rows:
        raise InputContractError("train·val 행이 0건이다")
    frozen = {axis: frozenset(values) for axis, values in eval_keys.items()}
    assert_isolated(rows.values(), frozen)
    return rows, frozen


def load_annotations(manifest_dir: Path, images: dict[str, ImageRow], eval_keys: dict[str, frozenset[str]],
                     label_codes: frozenset[str]) -> tuple[dict[str, list[Ann]], int]:
    """train·val 이미지의 어노테이션만 색인한다. eval 행은 셈만 하고 버린다."""
    handle, reader = _reader(manifest_dir / "annotations.csv", ANN_COLS)
    index: dict[str, list[Ann]] = {}
    dropped = 0
    seen: set[str] = set()
    with handle:
        for n, r in enumerate(reader, start=2):
            image_id = r["image_id"]
            if image_id in eval_keys["image_id"]:
                dropped += 1
                continue
            where = f"annotations.csv:{n}"
            if image_id not in images:
                raise InputContractError(f"{where}: 매니페스트에 없는 image_id")
            ann_id = _token(r["ann_id"], where + " ann_id")
            if ann_id in seen:
                raise InputContractError(f"{where}: ann_id 중복")
            seen.add(ann_id)
            if not ann_id.startswith(image_id + "#"):
                raise InputContractError(f"{where}: ann_id 가 '<image_id>#<번호>' 꼴이 아니다")
            code = _code(r["iso_code"], where + " iso_code")
            if code not in label_codes:
                raise InputContractError(f"{where}: 사상표에 없는 결함 코드 {code!r}")
            valid = _bool(r["geom_valid"], where + " geom_valid")
            corners = [r[c] for c in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")]
            if valid:
                bbox = tuple(_int(v, where + " bbox") for v in corners)
            elif any(corners):
                raise AmbiguousContract(f"{where}: 기하 무효인데 bbox 가 있다 — 빌더가 이것을 싣는지 규약에 없다")
            else:
                bbox = None
            index.setdefault(image_id, []).append(Ann(ann_id, code, valid, bbox, r["major_axis_px"],
                                                      r["equiv_diameter_px"]))
    return index, dropped


def assert_isolated(rows, eval_keys: dict[str, frozenset[str]]) -> None:
    """train·val 행이 eval 과 **네 축 어디서도** 겹치지 않는지 본다.

    id 만 보면 같은 파일이 다른 id 로 양쪽에 있어도 겹침 0 이 나온다. 그래서 묶음·경로·내용 해시까지 본다.
    같은 함수 안에서 방금 걸러낸 것을 다시 세면 실패할 수 없는 검사가 되므로(80번 G2), 호출은 적재가 끝난 뒤다.
    """
    seen = {"image_id": set(), "group_id": set(), "rel_path": set(), "sha256": set()}
    for row in rows:
        seen["image_id"].add(row.image_id)
        seen["group_id"].add(row.group_id)
        seen["rel_path"].add(row.path_key)
        seen["sha256"].add(row.sha256)
    for axis in ISOLATION_AXES:
        leaked = seen[axis] & eval_keys[axis]
        if leaked:
            raise AssertionError(f"train·val 과 eval 이 {axis} 축에서 {len(leaked)}건 겹친다")


def load_limits(path: Path) -> list[LimitRow]:
    handle, reader = _reader(path, LIMIT_COLS)
    rows: list[LimitRow] = []
    with handle:
        for n, r in enumerate(reader, start=2):
            where = f"{path.name}:{n}"
            if r["scope"] != "active":
                continue  # B 10번 §5 — scope 는 선택 필터다(오류가 아니다). 지금 표에 active 아닌 행은 없다
            if r["limit_op"] not in KNOWN_OPS:
                raise InputContractError(f"{where}: 모르는 부등호 {r['limit_op']!r}")
            t_min = _dec(r["thickness_min"], where + " thickness_min")
            if t_min is None:
                raise AmbiguousContract(f"{where}: thickness_min 이 비어 있다 — 하한 공란의 뜻이 규약에 없다")
            rows.append(LimitRow(
                rule_id=_token(r["rule_id"], where + " rule_id"), clause_id=_token(r["clause_id"], where + " clause_id"),
                defect_code=_code(r["defect_code"], where + " defect_code"),
                material=_token(r["material"], where + " material"),
                inspection_method=_token(r["inspection_method"], where + " inspection_method"),
                canonical=r["canonical"], quality_scheme=r["quality_scheme"],
                thickness_min=t_min, thickness_max=_dec(r["thickness_max"], where + " thickness_max"),
                limit_rule=_token(r["limit_rule"], where + " limit_rule"),
                limit_value=_dec(r["limit_value"], where + " limit_value"),
                limit_factor=_dec(r["limit_factor"], where + " limit_factor"),
                limit_cap=_dec(r["limit_cap"], where + " limit_cap"), ratio_basis=r["ratio_basis"],
                limit_op=r["limit_op"], unit=r["unit"]))
    if len({x.rule_id for x in rows}) != len(rows):
        raise InputContractError(f"{path.name}: rule_id 중복")
    return rows


# ---------------------------------------------------------------- 조항 선택과 기대 레코드


def selectable(limits: list[LimitRow], material: str) -> list[LimitRow]:
    """검사 방식·재질 축만 지난 행. 코드 축은 아직 보지 않는다 — 미특정 사유 두 가지를 가르는 데 쓴다."""
    return [x for x in limits if x.inspection_method in SELECTED_METHODS
            and x.material in (material, MATERIAL_ANY)]


def rules_for(limits: list[LimitRow], material: str, code: str) -> list[LimitRow]:
    """(재질, 결함 코드)에 걸리는 행 — B 10번 §5 의 선택 축 넷. 표의 행 순서로 돌려준다."""
    picked = [x for x in selectable(limits, material) if x.defect_code == code]
    if len({x.clause_id for x in picked}) > 1:
        raise AmbiguousContract(f"({material}, {code}) 가 두 조항에 걸린다 — 06번 §2-다: 빌더는 여기서 종료한다")
    for a, b in itertools.combinations(picked, 2):  # 한 두께에 기준이 둘이면 서술이 모순된 기준을 나열한다
        lo_a, hi_a = a.thickness_min, a.thickness_max
        lo_b, hi_b = b.thickness_min, b.thickness_max
        if (hi_a is None or lo_b < hi_a) and (hi_b is None or lo_a < hi_b):
            raise AmbiguousContract(f"{a.rule_id} 와 {b.rule_id} 의 두께 구간이 겹친다")
    for row in picked:  # 선택 축은 아니지만 선택된 행에 있으면 빌드가 멈춘다(B 10번 §5)
        if row.canonical != "true":
            raise AmbiguousContract(f"{row.rule_id}: canonical={row.canonical!r} 행이 선택됐다")
        if row.quality_scheme != "none":
            raise AmbiguousContract(f"{row.rule_id}: quality_scheme={row.quality_scheme!r} 행이 선택됐다")
    return picked


def _num(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value.normalize(), "f")
    return text if text != "-0" else "0"


def _thick(value: Decimal | None) -> str | None:
    """`candidate_rules` 의 두께 표기 — 소수 둘째 자리 문자열, 상한 공란은 null(B 10번 §3)."""
    return None if value is None else format(value.quantize(GRID), "f")


def narrative_values(row: LimitRow) -> dict:
    """서술에 들어갈 **값**의 기대. 문장은 만들지 않는다. 구간은 원문 폐구간 (a, b] 로 되돌린다."""
    for bound in (row.thickness_min, row.thickness_max):
        if bound is not None and bound != bound.quantize(GRID):
            raise AmbiguousContract(f"{row.rule_id}: 두께 구간이 0.01 그리드가 아니다 — 폐구간 환산이 정의되지 않는다")
    lower = None if row.thickness_min == 0 else row.thickness_min - GRID
    upper = None if row.thickness_max is None else row.thickness_max - GRID
    return {"rule_id": row.rule_id, "thickness_gt": _num(lower), "thickness_le": _num(upper),
            "limit_rule": row.limit_rule, "limit_value": _num(row.limit_value), "limit_factor": _num(row.limit_factor),
            "limit_cap": _num(row.limit_cap), "ratio_basis": row.ratio_basis, "limit_op": row.limit_op,
            "unit": row.unit}


def expected_record(img: ImageRow, anns: list[Ann], limits: list[LimitRow]) -> dict:
    """이미지 한 장의 기대 레코드(구조화 부분). 폐기 대상이면 `discard` 에 사유가 든다."""
    ordered = sorted(anns, key=operator.attrgetter("ann_id"))  # 문자열 정렬 — "#10" < "#2" (06번 §2-나 '정렬은 학습 계약이다')
    valid = [a for a in ordered if a.geom_valid]
    if len(anns) != img.n_defects:  # B 10번 §6 — 폐기가 아니라 빌드 중단. 본 매니페스트 49,847장 전부 성립한다
        raise InputContractError(f"{img.image_id}: 유효 {len(valid)} + 무효 {len(anns) - len(valid)} "
                                 f"≠ 매니페스트 n_defects {img.n_defects}")
    flags: list[str] = []
    if not img.has_defect and anns:
        flags.append("normal_with_annotations")
    if tuple(sorted({a.iso_code for a in anns})) != tuple(sorted(img.iso_codes)):
        flags.append("iso_codes_ne_annotation_codes")

    discard: list[str] = []
    if img.has_defect and not valid:
        discard.append(DISCARD_NO_VALID_GEOMETRY)
    for a in valid:
        x1, y1, x2, y2 = a.bbox
        if not (0 <= x1 < x2 <= img.width_px and 0 <= y1 < y2 <= img.height_px):
            discard.append(DISCARD_BBOX_BOUNDS)
            break

    codes: list[str] = []  # 첫 등장 순 — `uncited_codes` 의 순서다(B 10번 §1)
    for a in valid:
        if a.iso_code not in codes:
            codes.append(a.iso_code)
    picked: list[LimitRow] = []
    for code in codes:
        picked.extend(rules_for(limits, img.material, code))
    picked.sort(key=operator.attrgetter("rule_id"))  # rule_id 문자열 오름차순(B 10번 §3 — 표의 행 순서가 아니다)
    pool = selectable(limits, img.material)
    uncited = [{"code": c, "reason": UNCITED_CODE if pool else UNCITED_MATERIAL}
               for c in codes if not any(r.defect_code == c for r in picked)]

    return {
        "image_id": img.image_id, "split": img.split, "client": img.client, "material": img.material,
        "image_path": img.rel_path, "width_px": img.width_px, "height_px": img.height_px,
        "coord_space": COORD_SPACE,
        "defects": [{"type": a.iso_code, "bbox_px": list(a.bbox),
                     "size_px": {"major_axis": float(a.major_axis_px), "equiv_diameter": float(a.equiv_diameter_px)}}
                    for a in valid],
        SKIP_COUNT_FIELD: len(ordered) - len(valid),  # 빌더는 0 이면 키를 뺀다 — 대조기가 그 규칙을 적용한다
        "clauses": sorted({r.clause_id for r in picked}),
        "candidate_rules": [{"rule_id": r.rule_id, "clause_id": r.clause_id, "defect_code": r.defect_code,
                             "thickness_min": _thick(r.thickness_min), "thickness_max": _thick(r.thickness_max)}
                            for r in picked],
        "narrative_values": [narrative_values(r) for r in picked],
        "uncited_codes": uncited,
        "input_flags": flags,
        "discard": discard,
    }


def build_expected(manifest_dir: Path, limits_path: Path, label_map: Path) -> dict:
    contract = verify_contract(manifest_dir)
    images, eval_keys = load_manifest(manifest_dir)
    index, dropped = load_annotations(manifest_dir, images, eval_keys, load_label_codes(label_map))
    limits = load_limits(limits_path)
    records = [expected_record(images[i], index.get(i, []), limits) for i in sorted(images)]
    assert_isolated([images[r["image_id"]] for r in records], eval_keys)
    return {"records": records, "eval_ids": eval_keys["image_id"], "eval_groups": eval_keys["group_id"],
            "group_of": {i: images[i].group_id for i in images}, "contract": contract,
            "input_digest": {"computed": recompute_snapshot_digest(contract),
                             "declared": snapshot_digest(contract), "rule": DIGEST_RULE},
            "n_eval_annotation_rows_dropped": dropped, "limits_sha256": hashlib.sha256(limits_path.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- 대조기


def _find(rec: dict, name: str):
    """필드를 최상위와 `skeleton` 두 자리에서 찾는다. 자리는 B 10번 §1 이 정했고, 다른 자리에서 찾으면 함께 알린다."""
    if name in rec:
        return "top", rec[name]
    sk = rec.get("skeleton")
    if isinstance(sk, dict) and name in sk:
        return "skeleton", sk[name]
    return None, None


def _key_order_ok(keys, contract: tuple[str, ...]) -> bool:
    """**있는 키의 상대 순서**가 계약과 같은지 — 있는 키가 계약 순서의 부분열이면 참이다."""
    present = [k for k in keys if k in contract]
    return present == [k for k in contract if k in set(present)]


def _as_num(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise InputContractError("두께 값이 불리언이다")
    return _num(Decimal(str(value)))


def compare(expected: dict, pairs_dir: Path) -> dict:
    mism: Counter[str] = Counter()
    detail: list[dict] = []
    notes: Counter[str] = Counter()

    def hit(kind: str, image_id: str | None = None, **more):
        mism[kind] += 1
        detail.append({"kind": kind, "image_id": image_id, **more})

    raw = (pairs_dir / "pairs.jsonl").read_bytes()
    if b"\r" in raw:
        hit("crlf_in_pairs", n_cr=raw.count(b"\r"))
    got_order: list[str] = []
    got: dict[str, dict] = {}
    for n, line in enumerate(raw.decode("utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        image_id = rec["image_id"]
        got_order.append(image_id)
        if image_id in got:
            hit("duplicate_image_id", image_id, line=n)
        got[image_id] = rec

    exp_keep = {r["image_id"]: r for r in expected["records"] if not r["discard"]}
    exp_drop = {r["image_id"]: r for r in expected["records"] if r["discard"]}

    if got_order != sorted(got_order):
        hit("file_order", n_inversions=sum(a > b for a, b in itertools.pairwise(got_order)))
    for image_id in got.keys() & expected["eval_ids"]:
        hit("eval_image_in_pairs", image_id)
    groups = {expected["group_of"][i] for i in got if i in expected["group_of"]}
    for _ in groups & expected["eval_groups"]:
        hit("eval_group_in_pairs")
    for image_id in exp_keep.keys() - got.keys():
        hit("missing_record", image_id)
    for image_id in got.keys() - exp_keep.keys():
        hit("unexpected_record", image_id, expected_discard=image_id in exp_drop)

    for image_id in sorted(exp_keep.keys() & got.keys()):
        e, g = exp_keep[image_id], got[image_id]
        sk = g.get("skeleton", {})
        for name in ("split", "client", "image_path"):
            if g.get(name) != e[name]:
                hit(name, image_id, expected=e[name], got=g.get(name))
        for name in TOP_FIELDS:
            where, value = _find(g, name)
            if where is None:
                mism[f"field_absent:{name}"] += 1
            else:
                notes[f"{name}@{where}"] += 1
                if where != "top":
                    mism[f"field_place:{name}"] += 1
                if value != e[name]:
                    hit(name, image_id, expected=e[name], got=value)
        for name in SKELETON_FIELDS:
            where, _ = _find(g, name)
            if where is not None and where != "skeleton":
                mism[f"field_place:{name}"] += 1
        if not _key_order_ok(g.keys(), TOP_KEY_ORDER):
            hit("key_order", image_id, got=[k for k in g if k in TOP_KEY_ORDER])
        if isinstance(sk, dict) and not _key_order_ok(sk.keys(), SKELETON_KEY_ORDER):
            hit("key_order_skeleton", image_id, got=[k for k in sk if k in SKELETON_KEY_ORDER])

        g_def = sk.get("defects")
        if not isinstance(g_def, list):
            hit("defects_not_list", image_id)
            g_def = []
        e_core = [(d["type"], tuple(d["bbox_px"])) for d in e["defects"]]
        try:
            g_core = [(d["type"], tuple(d["bbox_px"])) for d in g_def]
        except (KeyError, TypeError):
            hit("defect_item_shape", image_id)
            g_core = []
        if any(not isinstance(t, str) or any(type(v) is not int for v in b) for t, b in g_core):
            hit("defect_value_type", image_id)
        if g_core != e_core:
            if len(g_core) != len(e_core):
                hit("defect_count", image_id, expected=len(e_core), got=len(g_core))
            elif sorted(g_core) == sorted(e_core):
                hit("defect_order", image_id)
            elif [t for t, _ in g_core] != [t for t, _ in e_core]:
                hit("defect_type", image_id)
            else:
                hit("defect_bbox", image_id)
        elif g_def:
            for ed, gd in zip(e["defects"], g_def):
                if gd.get("size_px") != ed["size_px"]:
                    hit("size_px", image_id)
                    break
        for gd in g_def:  # B 10번 — 키 넷과 그 순서, `size_mm` 은 늘 null
            if isinstance(gd, dict) and tuple(gd) != DEFECT_KEY_ORDER:
                hit("defect_item_keys", image_id, got=list(gd))
                break
        if any(isinstance(gd, dict) and gd.get("size_mm", "absent") is not None for gd in g_def):
            hit("defect_size_mm_not_null", image_id)

        g_cl = sk.get("clauses")
        if g_cl != e["clauses"]:
            kind = "clauses_order" if isinstance(g_cl, list) and sorted(g_cl) == e["clauses"] else "clauses_set"
            hit(kind, image_id, expected=e["clauses"], got=g_cl, material=e["material"])

        where, g_rules = _find(g, "candidate_rules")
        if where is None:
            mism["field_absent:candidate_rules"] += 1
        else:
            notes[f"candidate_rules@{where}"] += 1
            try:
                g_norm = [{"rule_id": x["rule_id"], "clause_id": x["clause_id"], "defect_code": x["defect_code"],
                           "thickness_min": _as_num(x["thickness_min"]), "thickness_max": _as_num(x["thickness_max"])}
                          for x in g_rules]
                for x in g_rules:  # B 10번 §3 은 소수 둘째 자리 **문자열**이다. 수로 오면 값과 따로 센다
                    notes["candidate_rules.thickness 자료형=" + type(x["thickness_min"]).__name__] += 1
                    if not isinstance(x["thickness_min"], str) or (
                            x["thickness_max"] is not None and not isinstance(x["thickness_max"], str)):
                        mism["candidate_rules_thickness_type"] += 1
                        break
            except (KeyError, TypeError, InvalidOperation):
                hit("candidate_rules_shape", image_id)
                g_norm = None
            e_norm = [{**x, "thickness_min": _as_num(x["thickness_min"]),
                       "thickness_max": _as_num(x["thickness_max"])} for x in e["candidate_rules"]]
            if g_norm is not None and g_norm != e_norm:
                key = operator.itemgetter("rule_id")
                if sorted(g_norm, key=key) == sorted(e_norm, key=key):
                    hit("candidate_rules_order", image_id)
                elif {x["rule_id"] for x in g_norm} == {x["rule_id"] for x in e_norm}:
                    hit("candidate_rules_value", image_id)
                else:
                    hit("candidate_rules_set", image_id)

        if sk.get("verdict", "absent") is not None:
            hit("verdict_not_null", image_id)
        if sk.get("verdict_mode") != "clause_only":
            hit("verdict_mode", image_id, got=sk.get("verdict_mode"))

        where, g_unc = _find(g, "uncited_codes")
        if where is None:
            mism["field_absent:uncited_codes"] += 1
        elif g_unc != e["uncited_codes"]:
            kind = "uncited_codes_order" if isinstance(g_unc, list) and len(g_unc) == len(e["uncited_codes"]) \
                and sorted(map(repr, g_unc)) == sorted(map(repr, e["uncited_codes"])) else "uncited_codes_set"
            hit(kind, image_id, expected=e["uncited_codes"], got=g_unc, material=e["material"])

        expected_skip = e[SKIP_COUNT_FIELD]
        where, value = _find(g, SKIP_COUNT_FIELD)
        if where is None:  # B 10번 §1 — 0 이면 키가 없다
            if expected_skip:
                mism["skipped_count_unrecorded"] += 1
        else:
            notes[f"{SKIP_COUNT_FIELD}@{where}"] += 1
            if where != "top":
                mism[f"field_place:{SKIP_COUNT_FIELD}"] += 1
            if value != expected_skip:
                hit("skipped_count", image_id, expected=expected_skip, got=value)
            elif not expected_skip:
                mism["skipped_count_zero_key_present"] += 1

        if not isinstance(g.get("target_text"), str):  # 문장은 보지 않지만 **필드의 존재**는 본다(검토 12번 I-1)
            hit("field_absent:target_text", image_id, got=type(g.get("target_text")).__name__)
        nv = narrative_check(e, g.get("target_text"))
        for kind in nv:
            mism[kind] += 1
            detail.append({"kind": kind, "image_id": image_id})

    disc_path = pairs_dir / "discarded.jsonl"
    got_disc: dict[str, list] = {}
    if disc_path.exists():
        for line in disc_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                got_disc[d["image_id"]] = d.get("reasons", [])
    else:
        mism["discarded_file_absent"] += 1  # 폐기가 기대되는데 목록이 없으면 불일치다(미대조가 아니다)
    for image_id in exp_drop.keys() - got_disc.keys():
        hit("discard_missing", image_id, expected=exp_drop[image_id]["discard"])
    for image_id in got_disc.keys() - exp_drop.keys():
        hit("discard_unexpected", image_id, got=got_disc[image_id])
    for image_id in exp_drop.keys() & got_disc.keys():
        if DISCARD_NO_VALID_GEOMETRY in exp_drop[image_id]["discard"] and \
                DISCARD_NO_VALID_GEOMETRY not in got_disc[image_id]:
            hit("discard_reason", image_id, expected=exp_drop[image_id]["discard"], got=got_disc[image_id])

    def table(records) -> dict[str, int]:
        c: Counter[str] = Counter()
        for r in records:
            kind = "defect" if (r.get("defects") if "defects" in r else r["skeleton"]["defects"]) else "normal"
            c[f"{r['split']}|{r['client']}|{kind}"] += 1
        return dict(sorted(c.items()))

    exp_table, got_table = table(exp_keep.values()), table(got.values())
    if exp_table != got_table:
        mism["count_table"] += 1

    per_split: dict[str, dict[str, dict[str, int]]] = {}
    for key, n in exp_table.items():
        split, client, kind = key.split("|")
        cell = per_split.setdefault(split, {}).setdefault(client, {"n_total": 0, "n_defect": 0, "n_normal": 0})
        cell["n_total"] += n
        cell[f"n_{kind}"] += n
    per_client: dict[str, dict[str, int]] = {}  # B 10번 §7 — counts.json 의 clients 는 train·val 합산 회계다
    for clients in per_split.values():
        for client, cell in clients.items():
            got_cell = per_client.setdefault(client, {"n_total": 0, "n_defect": 0, "n_normal": 0})
            for k, v in cell.items():
                got_cell[k] += v

    counts_note = None
    skipped: list[str] = []
    """대조하지 못한 축. **산문이 아니라 목록이다** — 사람이 훑고 지나가지 않게, 기계가 읽는 자리에 둔다.
    못 본 것을 통과로 세지 않되 실패로도 세지 않는다. 둘은 다르다."""
    counts_path = pairs_dir / "counts.json"
    if not counts_path.exists():  # 필수 파일이 없는 것도 "대조하지 못한 축" 이다 — 조용히 넘어가지 않는다
        counts_note = "counts.json 이 없다 — 참여자별 회계와 폐기 수를 대조하지 않았다"
        skipped.extend(["counts.clients", "counts.discarded.quarantine"])
    if counts_path.exists():
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        clients = counts.get("clients")
        if isinstance(clients, dict) and all(isinstance(v, dict) and "n_defect" in v for v in clients.values()):
            seen = {c: {k: v[k] for k in ("n_total", "n_defect", "n_normal")} for c, v in clients.items()}
            if seen != per_client:
                hit("counts_json_clients", expected=per_client, got=seen)
        else:
            counts_note = "counts.json 의 clients 구조를 모른다 — 대조하지 않았다"
            skipped.append("counts.clients")
        quarantine = (counts.get("discarded") or {}).get("quarantine")
        if quarantine is None:
            counts_note = (counts_note or "") + " / discarded.quarantine 이 없다"
            skipped.append("counts.discarded.quarantine")
        elif quarantine != len(exp_drop):
            hit("counts_json_quarantine", expected=len(exp_drop), got=quarantine)

    meta_path = pairs_dir / "PAIRS_META.json"
    meta_note = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        by_split = meta.get("counts_by_split")
        if isinstance(by_split, dict) and set(by_split) == set(per_split):
            seen_split = {s: {c: {k: v[k] for k in ("n_total", "n_defect", "n_normal")} for c, v in cs.items()}
                          for s, cs in by_split.items()}
            if seen_split != per_split:
                hit("meta_counts_by_split", expected=per_split, got=seen_split)
        else:
            meta_note = "PAIRS_META.json 의 counts_by_split 구조를 모른다 — 대조하지 않았다"
            skipped.append("meta.counts_by_split")
        # 이름 겹침 주의: `SKIP_COUNT_FIELD` 는 레코드에서는 **그 이미지 하나의 수**, 메타에서는 **빌드 총계**다.
        # 총계 = Σ(레코드별 값) + 폐기된 이미지 몫. 층을 섞지 않도록 여기서만 메타를 읽는다.
        for key, want in ((SKIP_COUNT_FIELD, sum(r[SKIP_COUNT_FIELD] for r in expected["records"])),
                          (META_SKIP_IN_RECORDS, sum(r[SKIP_COUNT_FIELD] for r in exp_keep.values()))):
            got_value = meta.get(key)
            if got_value is None:
                meta_note = (meta_note or "") + f" / 메타에 {key} 가 없다"
                skipped.append(f"meta.{key}")
            elif got_value != want:
                hit(f"meta:{key}", expected=want, got=got_value)
        # 경로 실재 검사는 겹쳐 두지 않는다 — 같은 술어를 같은 파일시스템에 두 번 실행하면 유도가 늘지 않는다.
        # 대신 **빌드가 하는 주장**을 내가 따로 센 모집단과 맞댄다. 검사를 끄고 빌드하면 `n` 이 0 이라 여기서 걸린다.
        # 검토 §27-19 의 메타 세 변이(입력 digest·검증 주체·허용치 해시)를 이 대조기는 전부 통과시켰다.
        # 세 값 모두 **내가 따로 읽은 원천**과 맞댈 수 있다 — 빌드의 선언을 빌드의 다른 자리로 확인하지 않는다.
        # 적힌 값이지만 `verify_contract()` 가 다시 계산한 값과 같음을 이미 확인했다. 없으면 미대조로 남긴다.
        want_digest = snapshot_digest(expected["contract"])
        declared = meta.get(META_SNAPSHOT)
        got_digest = declared.get("snapshot_digest") if isinstance(declared, dict) else None
        if want_digest is None:  # 원천에 그 줄이 없으면 대조가 성립하지 않는다 — 통과로 세지 않는다
            meta_note = (meta_note or "") + " / SNAPSHOT.sha256 에 snapshot_digest 줄이 없다 — 선언을 맞댈 원천이 없다"
            skipped.append(f"meta.{META_SNAPSHOT}.snapshot_digest")
        elif got_digest != want_digest:  # 없는 것도 어긋난 것으로 센다 — 입력 판본 선언은 봉인의 전제다
            hit(f"meta:{META_SNAPSHOT}.snapshot_digest", expected=want_digest, got=got_digest)

        limits = meta.get(META_LIMITS)
        got_limits = limits.get("sha256") if isinstance(limits, dict) else None
        if got_limits != expected["limits_sha256"]:
            hit(f"meta:{META_LIMITS}.sha256", expected=expected["limits_sha256"], got=got_limits)

        got_by = meta.get(META_VALIDATED_BY)
        if got_by not in KNOWN_VALIDATED_BY:
            hit(f"meta:{META_VALIDATED_BY}", expected=sorted(KNOWN_VALIDATED_BY), got=got_by)

        claim = meta.get(META_PATHS_CHECKED)
        if isinstance(claim, dict) and {"n", "of"} <= claim.keys():
            want_of = len(expected["records"])
            if claim["of"] != want_of or claim["n"] != claim["of"]:
                hit(f"meta:{META_PATHS_CHECKED}", expected={"n": want_of, "of": want_of}, got=claim)
        else:
            meta_note = (meta_note or "") + f" / 메타의 {META_PATHS_CHECKED} 가 건수를 싣지 않는다"
            skipped.append(f"meta.{META_PATHS_CHECKED}")
    else:
        meta_note = "PAIRS_META.json 이 없다 — 분할 축 회계와 기하 무효 총계를 대조하지 않았다(파일럿에는 없다)"
        # 메타가 없으면 위 블록 전체가 돌지 않는다 — **일곱 축이 미대조다.** 넷만 적으면 보고서가
        # 못 본 범위를 실제보다 좁게 말한다(교차 검수 I-4, 이 저장소가 세 번 낸 "0건의 범위" 와 같은 형태다).
        skipped.extend(["meta.counts_by_split", f"meta.{SKIP_COUNT_FIELD}", f"meta.{META_SKIP_IN_RECORDS}",
                        f"meta.{META_PATHS_CHECKED}", f"meta.{META_SNAPSHOT}.snapshot_digest",
                        f"meta.{META_LIMITS}.sha256", f"meta.{META_VALIDATED_BY}"])

    return {"mismatch": dict(sorted(mism.items())), "detail": detail, "notes": dict(sorted(notes.items())),
            "expected_table": exp_table, "got_table": got_table, "n_expected_keep": len(exp_keep),
            "n_expected_discard": len(exp_drop), "n_got": len(got), "counts_note": counts_note,
            "meta_note": meta_note, "not_compared": sorted(set(skipped)), "expected_counts_by_split": per_split,
            "expected_discard_reasons": dict(Counter(x for r in exp_drop.values() for x in r["discard"])),
            "input_flags": dict(Counter(f for r in expected["records"] for f in r["input_flags"])),
            "uncited_reasons": dict(Counter(x["reason"] for r in exp_keep.values() for x in r["uncited_codes"])),
            "n_uncited_defect_records": sum(1 for r in exp_keep.values() if r["uncited_codes"]),
            "n_defect_order_not_numeric": sum(1 for r in exp_keep.values() if len(r["defects"]) > 10)}


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def narrative_check(e: dict, text) -> list[str]:
    """서술의 **문장은 대조하지 않는다.** 기대 값이 서술 안에 수로 나타나는지만 본다(참고 범주, `narrative_*`).

    조항 id 와 'ISO 6520-1' 을 지운 뒤 남은 수의 다중집합이 기대 값(두께 구간 양끝·한계값·배수·상한)의 다중집합을
    품어야 하고, '미만' 의 수가 `lt` 행의 수와 같아야 한다. 값 한 칸이 바뀌거나 기준 문장이 통째로 빠지면 걸린다.
    """
    if not isinstance(text, str) or not e["narrative_values"]:
        return []
    stripped = text
    for clause in e["clauses"]:
        stripped = stripped.replace(clause, " ")
    stripped = stripped.replace("ISO 6520-1", " ")
    have = Counter(_num(Decimal(t)) for t in _NUMBER.findall(stripped))
    want: Counter[str] = Counter()
    clause_rows: dict[tuple, dict] = {}
    for v in e["narrative_values"]:  # 301·401 이 같은 조항을 가리키면 같은 문장이 한 번만 나올 수 있다 — 값 묶음으로 접는다
        key = tuple(v[k] for k in ("thickness_gt", "thickness_le", "limit_rule", "limit_value", "limit_factor",
                                   "limit_cap", "limit_op"))
        clause_rows[key] = v
    for v in clause_rows.values():
        for k in ("thickness_gt", "thickness_le", "limit_value", "limit_factor", "limit_cap"):
            if v[k] is not None:
                want[v[k]] += 1
    out = []
    if any(have[k] < n for k, n in want.items()):
        out.append("narrative_value_missing")
    n_lt = sum(1 for v in clause_rows.values() if v["limit_op"] == "lt")
    if stripped.count("미만") != n_lt:
        out.append("narrative_op_lt_count")
    return out


# ---------------------------------------------------------------- 실행


def _refuse_out(out: Path, *inputs: Path) -> None:
    for p in inputs:  # 입력(봉인본일 수 있다)을 먼저 본다 — 비어 있지 않다는 말보다 이쪽이 사고를 설명한다
        if out.resolve() == p.resolve() or p.resolve() in out.resolve().parents:
            raise SystemExit(f"--out 을 입력 디렉터리 안에 둘 수 없다: {out}")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out 이 비어 있지 않다: {out}")


def _write(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--limits", type=Path, required=True)
    ap.add_argument("--label-map", type=Path, default=Path("configs/label_map.yaml"))
    ap.add_argument("--pairs", type=Path, help="pairs.jsonl 이 든 디렉터리. 없으면 기대 레코드만 낸다")
    ap.add_argument("--out", type=Path, required=True, help="새 경로(미추적). 식별자가 든 파일이 여기 쓰인다")
    ap.add_argument("--allow", default="", help="쉼표로 나눈 불일치 범주 — 이 범주만 남으면 차단은 풀리지만 종료 코드는 3(EXIT_ALLOWED)이고 "
                         "봉인 가능이 아니다 (파일럿 대조용)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    _refuse_out(args.out, args.manifest_dir, *([args.pairs] if args.pairs else []))
    expected = build_expected(args.manifest_dir, args.limits, args.label_map)
    args.out.mkdir(parents=True, exist_ok=True)
    _write(args.out / "expected.jsonl",
           "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in expected["records"]))

    summary = {"manifest_contract": expected["contract"], "input_digest": expected["input_digest"],
               "limits_sha256": expected["limits_sha256"],
               "n_trainval_images": len(expected["records"]), "n_eval_ids": len(expected["eval_ids"]),
               "n_eval_annotation_rows_dropped": expected["n_eval_annotation_rows_dropped"]}
    if args.pairs:
        result = compare(expected, args.pairs)
        _write(args.out / "mismatches.jsonl",
               "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in result.pop("detail")))
        summary.update(result)
        allowed = {a for a in args.allow.split(",") if a}
        blocking = {k: v for k, v in summary["mismatch"].items() if k not in allowed}
        summary["blocking"] = blocking
        # **"불일치 0" 만으로는 두 가지가 구별되지 않는다** — 전부 보고 같았는지, 볼 수 있는 것만 보고 같았는지.
        # `coverage` 가 그것을 가른다. 봉인 조건은 `verdict == "일치"` 이며, 범위가 좁으면 그 이름을 주지 않는다.
        summary["coverage"] = "full" if not summary["not_compared"] else "partial"
        if blocking:
            summary["verdict"], code = "불일치", EXIT_MISMATCH
        elif summary["mismatch"]:
            summary["verdict"], code = "허용 범주만 남음", EXIT_ALLOWED
        elif summary["not_compared"]:
            summary["verdict"], code = "대조 범위 제한 — 본 것은 전부 같았다", EXIT_PARTIAL
        else:
            summary["verdict"], code = "일치", EXIT_MATCH
    else:
        # 대조 대상을 주지 않은 실행이다(기대 레코드만 내는 용도). **대조를 하나도 하지 않았으므로 0 이 아니다.**
        summary.update({"mismatch": {}, "blocking": {}, "not_compared": [NOT_COMPARED_ALL],
                        "coverage": "none", "verdict": "대조하지 않았다 — 기대 레코드만 냈다"})
        code = EXIT_PARTIAL
    summary["sealable"] = code == EXIT_MATCH
    """봉인 소비 경로가 읽는 자리. 종료 코드와 같은 것을 말하며, 둘 중 하나만 보고 판단해도 같은 답이 나온다.
    `verdict`·`coverage`·`not_compared` 와 함께 **모든 경로에서** 실린다."""
    _write(args.out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
