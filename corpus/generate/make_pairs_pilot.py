"""D4 이미지-판정문 페어 축소본 — 파일럿 통합형(⑥⑦) 학습 입력.

파일럿 표본 스냅샷(train·val 분할만)의 어노테이션에서 페어를 만든다. eval 분할은
만들지 않는다 — 평가셋에 학습 자산이 파생되는 순간 격리가 깨진다(개발규약 1-4).

판정 축은 좁힌다: **조항 검색 + 기준 서술.** 치수 합부는 넣지 않는다 — 두께·픽셀→mm
스케일이 전부 결측이라(함정 #10) 측정 크기 기반 합부가 원리적으로 성립하지 않고,
합성하면 근거 없는 수치를 지어내게 된다. verdict 는 null 이다.

target_text 는 골격 텍스트 그대로다(윤문 없음). GPU 는 학습 담당이 점유 중이고,
결정론 텍스트는 같은 입력에서 같은 바이트가 나와 재현 검증이 쉽다. 윤문은 본실험
페어에서 한다.

스키마는 학습 담당 인계 계약을 따른다:
  {image_id, image_path, client, split, skeleton{defects[{type, bbox_px, size_px,
   size_mm}], verdict, clauses[]}, target_text}
bbox 는 원본 픽셀이다(불변조건 8) — 네이티브 좌표 변환은 학습 쪽 collator 가 한다.

검증 게이트(위반분은 재생성하지 않고 폐기, 통과율 기록):
  스키마 · 조항 실재(limits 파생 근거와 대조) · 부등식 방향(기준 서술의 한계·"이하"가
  limits 행과 일치) · 합부 단어 금지 · 정상 페어 결함 어휘 금지 · bbox 경계.

`--out` 은 **필수**다. 기본값이 `pairs_pilot_v2` 이던 시절, v2 가 봉인된 뒤로는 인자 없는
실행 한 번이 봉인본을 지웠다. 옛 가드는 경로가 v1 인지만 봤기 때문에 그 실행을 통과시켰다 —
이름을 열거하는 가드는 자산이 느는 속도를 못 따라간다. 지금은 계약(`SNAPSHOT.sha256`)을 본다.

실행: uv run python -m corpus.generate.make_pairs_pilot --out data/processed/pairs_pilot_v3
산출: 지정한 디렉터리 (pairs.jsonl + counts.json + SNAPSHOT.sha256)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# 봉인 계약의 정본은 `data.frozen_guard` 다 — corpus 쪽 문안만 frozen_out 이 감싼다.
from corpus.generate.frozen_out import CONTRACT_NAME, assert_not_frozen

# 어휘 락·판정어·아티팩트 검출의 정본 (§4-6-1 ④ · 80번 G2-6·G3-1).
# **여기에 같은 성격의 정규식을 다시 두지 않는다** — 배선 시험이 강제한다.
from corpus.generate.numeric_lock import (
    find_artifacts,
    find_defect_tokens,
    find_verdict_implying,
)

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "data/processed/aihub71761_rt_v1_pilot3000"
#: v1 은 동결이다 (규약 1-6). 알려진 결함 421건의 회계가 v1 에 붙어 있고, 같은 경로에
#: 덮어쓰면 그 회계가 가리키는 실물이 사라진다. v2 도 09-02 에 봉인됐다.
#: **둘 다 기본값이 아니다** — `--out` 은 필수이고, 봉인 여부는 계약 파일로 판정한다.
FROZEN_V1 = REPO / "data/processed/pairs_pilot_v1"
LEGACY_OUT_DIRS = (FROZEN_V1, REPO / "data/processed/pairs_pilot_v2")
LIMITS_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"

#: 좌표 규약 (판정 1, main 47c4dbc). 페어의 `bbox_px` 는 **원본 절대 픽셀**이고
#: 모델 좌표 변환은 학습 쪽 `vlm/coords.py` 가 한다. ABS_ORIG 에서 그 변환은 항등이지만
#: 규약을 산출물에 **명시**한다 — 함정 #4 는 규약이 암묵일 때 터진다.
COORD_SPACE = "ABS_ORIG"

#: 재질을 덮는 허용치 행이 없을 때의 서술 (§4 안 B — 09-21 판정으로 확정).
#: 결함은 서술하되 조항을 특정하지 않는다. 근거가 없다는 것은 지어낸 사실이 아니라 사실이다.
NO_CLAUSE_TAIL = ("이 재질에 적용할 허용치 조항이 허용치 표에 없어"
                  " 적용 조항을 특정하지 않는다.")
#: 재질은 덮이는데 **그 결함 코드**의 행이 없을 때. 재질 단위 문장을 쓰면 거짓이 된다 —
#: AL 행이 일부 코드만 확보되는 순간 "이 재질에 적용할 조항이 없어" 는 틀린 말이다(06번 §2-나).
NO_CLAUSE_CODE_TAIL = "에 적용할 허용치 조항이 허용치 표에 없어 그 결함의 적용 조항을 특정하지 않는다."
#: 값 한계에 미표현 단서가 있을 때의 부기 (80번 B12).
NOTE_UNEXPRESSED = ("이 조항에는 기계 표현으로 옮기지 못한 단서가 있어"
                    " 위 기준만으로 합부를 결정하지 않는다.")
#: 맺음 문장. 치수 기준이 하나라도 걸렸거나 조항을 특정하지 못한 결함이 있으면 앞의 것,
#: 크기와 무관한 기준(전량 불허)이 걸렸으면 뒤의 것. 균열만 있는 강재 이미지에는 뒤의 것만 붙는다 —
#: "두께와 실치수가 없어 판정하지 않는다" 는 크기와 무관한 기준에는 맞지 않는 이유다(06번 §3-사).
#: 판정 축은 열지 않는다(`verdict=null`, 09-21 판정 6).
CLOSE_DIMENSIONAL = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."
CLOSE_SIZE_INDEPENDENT = ("크기와 무관한 기준은 치수 정보 없이도 적용되지만,"
                          " 이 자료는 판정 축을 조항 검색과 기준 서술로 한정해 합부 판정을 싣지 않는다.")

#: 조항을 특정하지 못한 사유 — 레코드의 `skeleton.uncited_codes[].reason`.
UNCITED_MATERIAL = "material_not_covered"
UNCITED_CODE = "code_not_covered"

# ------------------------------------------------------------------ 근거 구축

def clause_basis(table, material: str) -> dict[str, dict]:
    """결함 코드 → 적용 조항과 두께 구간별 기준 서술 (RT 축, **재질 축 포함**).

    v1 에서 이 함수가 재질을 보지 않았다. 파일럿 허용치 표 12행이 전부 `material=ST`
    라서, 알루미늄 이미지에 강재 전용 조항(KRA27-T15/T16/3D)이 그대로 붙었다 —
    C3 결함 페어 219건 **전량**이다 (74번 P4 · 80번 B11).

    정본 행 선택기 `limit_eval.applicable_row` 는 재질 축을 보고 해당 행이 없으면
    fallback 없이 거절한다. 여기서도 같은 축을 본다. 덮는 행이 없으면 빈 사전을 내고,
    호출부가 "적용 조항을 특정하지 않는다"로 닫는다 (§4 안 B).

    두께가 결측이라 행 하나를 특정할 수 없다(함정 #10). 조항 수준으로 묶고 구간별
    기준을 나열한다 — 합부를 말하지 않으므로 행 특정이 필요 없다.

    문장화는 `clause_text` 정본을 쓴다. 부등호는 `limit_op`, 비례 분모는 `ratio_basis`
    에서 온다 — v1 은 둘 다 f-string 상수라 부등식 방향 게이트가 자기참조였다 (M9).
    """
    from corpus.rules.clause_text import criterion_ko, thickness_ko, val

    by_code: dict[str, dict] = {}
    rows = [r for r in table.rows
            if getattr(r, "scope", "active") == "active"
            and val(r.inspection_method) in ("RT", "ALL")
            and val(r.material) in (material, "ALL")]
    rows.sort(key=lambda r: r.rule_id)
    for r in rows:
        ent = by_code.setdefault(r.defect_code, {"clause_id": r.clause_id, "criteria": [],
                                                 "rules": [], "rule_kinds": []})
        if ent["clause_id"] != r.clause_id:
            # 같은 코드가 두 조항에 걸리면 조항 특정이 모호해진다. 파일럿 표에는 없고,
            # 생기면 페어 축을 다시 정해야 하므로 시끄럽게 실패한다.
            raise SystemExit(f"결함 {r.defect_code} 가 복수 조항에 걸린다: "
                             f"{ent['clause_id']} vs {r.clause_id}")
        band = thickness_ko(r.thickness_min, r.thickness_max)
        head = band if band == "모든 두께" else f"두께 {band}"
        ent["criteria"].append(f"{head}: {criterion_ko(r)}")
        # 행별 두께 구간 — 조항 입도를 **레코드만으로** 바꿀 수 있게 한다(06번 §3-다). 두께를 모르면
        # 조항 id 가 정답이고, 두께가 생기면 이 구간으로 행을 고른다. 반열림 [min, max), 0.01 그리드.
        ent["rules"].append({"rule_id": r.rule_id, "clause_id": r.clause_id,
                             "defect_code": str(r.defect_code),
                             "thickness_min": _grid(r.thickness_min),
                             "thickness_max": _grid(r.thickness_max)})
        kind = val(r.limit_rule)
        if kind not in ent["rule_kinds"]:
            ent["rule_kinds"].append(kind)
        # 부기는 **값 한계**에만 건다. 전량 불허 행에는 단서가 걸릴 값이 없고 그 행의 `note` 는
        # 전사 메모다 — 붙이면 "허용하지 않는다" 뒤에 "위 기준만으로 결정하지 않는다" 가 온다(06번 §3-사).
        if kind != "none_permitted" and (getattr(r, "note", None) or "").strip():
            ent["has_unexpressed"] = True
    return by_code


def _grid(x) -> str | None:
    """두께 경계를 표의 부호화 그대로(소수 둘째 자리 문자열) 옮긴다. 상한 공란은 None."""
    from decimal import Decimal

    if x is None or x == "":
        return None
    return str(Decimal(str(x)).quantize(Decimal("0.01")))


def covered_materials(table) -> set[str]:
    """`clause_basis` 가 고르는 행 묶음이 덮는 재질 집합.

    `material=ALL` 행은 **모든 재질을 덮는다.** 표기 그대로 `{"ALL"}` 을 돌려주면 메타의 `uncovered` 가
    "AL 은 덮이지 않았다" 고 적는데 정작 AL 레코드에는 그 조항이 실린다 — 산출물이 스스로와 어긋난다(검토 M-8).
    """
    from corpus.rules.clause_text import val
    from corpus.rules.schema import Material

    known = {m.value for m in Material} - {"ALL"}
    out: set[str] = set()
    for r in table.rows:
        if getattr(r, "scope", "active") == "active" and val(r.inspection_method) in ("RT", "ALL"):
            mat = val(r.material)
            out |= known if mat == "ALL" else {mat}
    return out


# ------------------------------------------------------------------ 텍스트

def observation_line(defects: list[dict], names: dict[str, str]) -> tuple[str, list[str]]:
    """관찰 문장 + 등장 순서 코드 목록."""
    seen: list[str] = []
    for d in defects:
        if d["type"] not in seen:
            seen.append(d["type"])
    obs = [f"{names.get(c, '결함')}(ISO 6520-1 코드 {c}) "
           f"{sum(1 for d in defects if d['type'] == c)}개" for c in seen]
    return "방사선투과 영상에서 " + ", ".join(obs) + "가 관찰된다.", seen


def defect_target_text(defects: list[dict], names: dict[str, str],
                       basis: dict[str, dict]) -> str:
    """결함 페어의 골격 텍스트. 관찰 → 조항 → 기준. 합부는 말하지 않는다.

    `basis` 에 그 코드가 없으면 조항을 지어내지 않고 **없다고 적는다.** v1 은 재질을
    안 보고 다른 재질의 조항을 붙였다 — 그것이 틀린 근거를 학습 타깃에 넣은 경로다.
    """
    head, codes = observation_line(defects, names)
    lines = [head]
    missing = [c for c in codes if c not in basis]
    kinds: set[str] = set()
    for code in codes:
        b = basis.get(code)
        if b is None:
            continue
        lines.append(f"{names.get(code, '결함')}에 적용되는 조항은 {b['clause_id']} 이다.")
        # 기준의 끝말은 명사("4 mm 이하")일 수도 서술어("허용하지 않는다")일 수도 있다. "…이다" 로 받으면
        # 서술어 끝에서 "허용하지 않는다이다" 가 된다 — 균열 페어 전부가 그랬다(06번 §2-나). 끝말을 가리지 않는 틀로 쓴다.
        lines.append("이 조항의 기준은 다음과 같다. " + " / ".join(b["criteria"]) + ".")
        kinds.update(b.get("rule_kinds", ()))
        if b.get("has_unexpressed"):
            # 원천 표에 기계 표현으로 옮기지 못한 단서가 있다 (집계 단위·무시 하한 등).
            # KRA27-T16 은 **합계 길이** 기준인데 스키마에는 길이 한계로만 실린다 —
            # v1 은 그것을 개별 결함 한계처럼 서술해 286건에 실었다 (80번 B12).
            lines.append(NOTE_UNEXPRESSED)
    if missing:
        # `basis` 가 비었으면 그 재질을 덮는 행이 하나도 없는 것이다. 일부 코드만 빠졌으면 그 결함만 말한다.
        # 코드를 함께 적는다. 사상표에는 명칭이 같은 코드가 있어(2011·2012 둘 다 "기공") 명칭만 쓰면
        # 한 문장이 "기공에 적용되는 조항은 X 이다 … 기공에 적용할 조항이 없어" 로 스스로 부딪힌다(검토 F7).
        lines.append(NO_CLAUSE_TAIL if not basis else
                     ", ".join(f"{names.get(c, '결함')}(ISO 6520-1 코드 {c})" for c in missing) + NO_CLAUSE_CODE_TAIL)
    if missing or kinds - {"none_permitted"}:
        lines.append(CLOSE_DIMENSIONAL)
    if "none_permitted" in kinds:
        lines.append(CLOSE_SIZE_INDEPENDENT)
    return " ".join(lines)


NORMAL_TEXT = ("방사선투과 영상에서 검출 한계 내 특기할 지시가 관찰되지 않는다. "
               "인용할 허용치 조항이 없다.")


# ------------------------------------------------------------------ 검증

#: 레코드가 반드시 실어야 하는 키. 값이 비면 `schema:<키>` 로 폐기한다.
REQUIRED_KEYS = ("image_id", "image_path", "client", "split", "material",
                 "width_px", "height_px", "coord_space", "skeleton", "target_text")
#: 레코드·골격·결함에 실릴 수 있는 키 **전부**. 여분 키는 폐기 사유다 — 열린 스키마로 두면 지어낸 판정이
#: `judgment` 같은 이름으로 조용히 실려 나간다(검토 F9).
ALLOWED_KEYS = frozenset({*REQUIRED_KEYS, "n_annotations_skipped_geom_invalid"})
SKELETON_KEYS = ("defects", "verdict", "verdict_mode", "clauses", "candidate_rules", "uncited_codes")
DEFECT_KEYS = ("type", "bbox_px", "size_px", "size_mm")
#: 분할·참여자 어휘. 페어는 train·val 만이고 참여자는 D1 분할의 셋뿐이다.
SPLITS = frozenset({"train", "val"})
CLIENTS = ("C1", "C2", "C3")


def _first_seen_types(sk: dict) -> list[str]:
    out: list[str] = []
    for d in sk.get("defects", []) or []:
        t = d.get("type") if isinstance(d, dict) else None
        if t is not None and t not in out:
            out.append(t)
    return out


def _outside(text: str, spans) -> str:
    """`spans` 구간을 뺀 나머지. 기준 문장은 허용치 행과 등치로 따로 대조하므로 판정 함의 검사에서 뺀다."""
    out, i = [], 0
    for a, b in spans:
        out.append(text[i:a])
        i = b
    out.append(text[i:])
    return " ".join(out)


def check_pair(rec: dict, names: dict[str, str], rules, lexicon: frozenset[str],
               wh: tuple[int, int] | None = None) -> list[str]:
    """검증 게이트. 폐기 사유 목록을 낸다(비면 통과). 위반분은 재생성하지 않고 폐기한다(규약 3-6).

    `rules` 는 `corpus.validate.pair_criteria_gate.PairRules` — 허용치 행에서 **독립 경로로** 만든 기대값이다.
    예전 게이트 ③은 생성기가 만든 기준 문장의 조각이 서술에 부분문자열로 들어 있는지만 봤고, 재질도 받지 않았다.
    기준 문장을 지운 서술과 AL 에 강재 조항을 단 레코드가 통과했다(06번 §2-나). 지금은 서술을 파싱해 등치로 본다.

    `wh` 를 주면 레코드가 들고 있는 `width_px`·`height_px` 가 그 값과 같은지도 본다(빌더는 매니페스트 값을 준다).
    """
    from corpus.validate.pair_criteria_gate import check_text

    bad: list[str] = []
    for k in REQUIRED_KEYS:
        if not rec.get(k):
            bad.append(f"schema:{k}")
    if set(rec) - ALLOWED_KEYS:
        bad.append("schema:extra_keys")
    if rec.get("coord_space") not in (None, "") and rec.get("coord_space") != COORD_SPACE:
        bad.append("coord_space_unknown")     # 규약이 암묵이면 함정 #4 가 터진다 — 모르는 규약도 같다
    if rec.get("split") not in SPLITS or rec.get("client") not in CLIENTS:
        bad.append("split_or_client_unknown")
    sk = rec.get("skeleton") or {}
    if tuple(sk) != SKELETON_KEYS:
        bad.append("schema:skeleton_keys")    # 순서까지 본다 — 산출 바이트가 키 순서에 걸려 있다
    if sk.get("verdict") is not None:
        bad.append("verdict_present")
    if sk.get("verdict_mode") != "clause_only":
        bad.append("verdict_mode_unknown")    # 아무도 안 보던 값이다(검토 F9). 축을 바꾸는 표지다
    text = rec.get("target_text", "")
    W, H = rec.get("width_px"), rec.get("height_px")
    if type(W) is not int or type(H) is not int:
        bad.append("image_size_type")         # 10.0 은 10 과 다른 토큰이다 — 참·거짓도 int 로 읽히니 type 으로 본다
        W, H = int(W or 0), int(H or 0)
    if wh is not None and (W, H) != tuple(wh):
        bad.append("image_size_mismatch")
    for d in sk.get("defects", []):
        if tuple(d) != DEFECT_KEYS:
            bad.append("schema:defect_keys")
            continue
        if d["type"] not in names:
            bad.append("unknown_code")
        if len(d["bbox_px"]) != 4 or any(type(v) is not int for v in d["bbox_px"]):
            bad.append("bbox_type")
            continue
        x1, y1, x2, y2 = d["bbox_px"]
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            bad.append("bbox_out_of_bounds")
        if d.get("size_mm") is not None:
            bad.append("mm_present")          # 스케일 부재 — mm 가 있으면 지어낸 값이다
    for c in sk.get("clauses", []):
        if c not in rules.valid_clause_ids:
            bad.append("clause_unknown")

    scan = text
    if sk.get("defects"):
        # 게이트 ③ — 관찰·조항·기준·부기·맺음을 **전부** 허용치 행과 골격에서 다시 계산해 등치로 본다.
        found, parsed = check_text(rec, rules)
        bad.extend(found)
        if parsed is not None:
            scan = _outside(text, parsed.criteria_spans)
    else:
        # 정상 페어 — 결함 어휘 금지 (부정 문맥 포함, §4-6-1 ③④).
        # 검출은 **정본 하나**만 쓴다 — `numeric_lock.find_defect_tokens`
        # (`skeleton_gen` 이 재수출). 여기서 맨 부분문자열 루프로 재구현했더니
        # NFKC 정규화·대소문자 불문·숫자 경계가 빠졌고, 그것은 skeleton_gen.py:34-38
        # 이 명시적으로 금지한 것이다 (74번 감사 P6).
        if find_defect_tokens(text, lexicon):
            bad.append("defect_word_in_normal")
        if sk.get("clauses") or sk.get("candidate_rules") or sk.get("uncited_codes"):
            bad.append("normal_has_clauses")
        if text != NORMAL_TEXT:
            bad.append("normal_text_mismatch")
    # 판정어·판정 함의 표현은 정본 하나로 본다 (G2-6). 자체 정규식을 두면 "적합하다"
    # 처럼 낱말을 피한 표현이 통째로 새고, 그 구멍은 한쪽에만 생긴다.
    # **기준 문장은 뺀다** — 그 구간은 위에서 허용치 행과 등치로 닫혔다. 빼지 않으면 "허용하지 않는다"(전량 불허
    # 조항의 기준 자체)가 검출기 목록에 들어가는 순간 균열 페어가 재생성 없이 전량 폐기된다(06번 §3-사).
    if find_verdict_implying(scan):
        bad.append("verdict_word")
    # 내부 표현이 문장으로 샌 흔적 (G3-1). 골격 텍스트라 지금은 나올 수 없지만,
    # 윤문을 붙이는 본실험 페어에서 이 검사가 실검사가 된다.
    # **기준은 허용치 행에서 만든 토큰이다.** `basis=""` 로 두면 표에서 정당하게 온 값·조항 id 까지
    # "새어 나온 내부 표기" 로 읽혀, 표를 개정하는 순간 그 조항의 페어가 통째로 폐기된다(검토 F4).
    if find_artifacts(text, basis=rules.artifact_basis(rec.get("material"), _first_seen_types(sk))):
        bad.append("artifact_violation")
    return sorted(set(bad))


def write_jsonl(path: Path, rows) -> None:
    """LF 고정 (.gitattributes eol=lf). win32 텍스트 모드는 개행을 CRLF 로 바꾼다 —
    v1 의 pairs.jsonl 이 CRLF 였고 SNAPSHOT 해시가 그 바이트 위에 섰다."""
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body + ("\n" if rows else ""))


def write_json(path: Path, doc: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


# ------------------------------------------------------------------ 조립

class PairInputError(ValueError):
    """입력(매니페스트·어노테이션)이 계약을 어겼다. **폐기가 아니라 빌드 중단**이다 —
    폐기는 생성물의 결함에 쓰고, 입력 결함을 폐기로 세면 통과율이 입력 문제를 가린다."""


#: 결함 라벨인데 유효 기하가 하나도 없는 이미지의 폐기 사유.
NO_VALID_GEOMETRY = "defect_label_no_valid_geometry"

#: **빌더가 낼 수 있는 폐기 사유 전부.** 소비처가 어휘 밖 사유를 만나면 지어낸 것이다 —
#: 회계를 맞춰 둬도 여기서 걸린다(검토 §27-19 의 부당 폐기). `schema:<키>` 는 접두로 본다.
DISCARD_REASONS: frozenset[str] = frozenset({
    NO_VALID_GEOMETRY,
    "coord_space_unknown", "split_or_client_unknown", "verdict_present", "verdict_mode_unknown",
    "image_size_type", "image_size_mismatch", "unknown_code", "bbox_type", "bbox_out_of_bounds",
    "mm_present", "clause_unknown", "verdict_word", "artifact_violation",
    "defect_word_in_normal", "normal_has_clauses", "normal_text_mismatch",
    # 게이트 ③ (`pair_criteria_gate.check_text`)
    "material_missing", "material_unknown", "clause_set_mismatch", "candidate_rules_mismatch",
    "uncited_codes_mismatch", "text_unparsable", "observation_mismatch", "clause_text_mismatch",
    "criterion_mismatch", "note_mismatch", "uncited_text_mismatch", "closing_mismatch",
})
#: 키 이름이 뒤에 붙는 사유의 접두.
DISCARD_PREFIXES: tuple[str, ...] = ("schema:",)


def known_reason(reason: str) -> bool:
    return reason in DISCARD_REASONS or reason.startswith(DISCARD_PREFIXES)


def truth(v, what: str) -> bool:
    """참·거짓을 **엄격히** 읽는다. `bool()` 을 쓰면 문자열 `"False"` 가 참이 되어 기하 무효 어노테이션이
    유효로 실린다(검토 15). 적재기를 지나면 진짜 bool 이지만, CSV 를 직접 읽는 검산 경로는 문자열을 준다."""
    if isinstance(v, bool):
        return v
    if hasattr(v, "item") and isinstance(v.item(), bool):      # numpy.bool_
        return bool(v.item())
    if v in ("True", "False"):
        return v == "True"
    raise PairInputError(f"{what}: 참·거짓이 아니다 — {v!r}")


def index_annotations(annotations) -> dict[str, list[dict]]:
    """어노테이션 표 → `image_id` 별 행 목록. `iterrows` 를 쓰지 않는다 — 행마다 Series 를 만들어
    느리고, 열 자료형을 object 로 끌어올린다. 값은 아래에서 명시적으로 변환하므로 출력은 같다."""
    out: dict[str, list[dict]] = defaultdict(list)
    for tup in annotations.itertuples(index=False):
        a = tup._asdict()
        out[a["image_id"]].append(a)
    return out


def assemble_record(row, anns, names: dict[str, str], base: dict[str, dict]) -> tuple[dict | None, int]:
    """매니페스트 행 하나와 그 이미지의 어노테이션 → `(레코드, 기하 무효로 뺀 어노테이션 수)`.

    빌더·계측·독립 검산이 **이 함수 하나**를 부른다. `main()` 안에 있을 때는 import 할 수 없어서
    계측 스크립트가 조립을 복제해야 했다 — "판정 논리는 한 곳" 은 이것이 돼야 성립한다(06번 §2-나).

    - 어노테이션은 `ann_id` **문자열**로 정렬한다("#10" < "#2"). 정렬은 학습 계약이다 — 생성이 한도에서
      잘릴 때 어느 결함이 잘리는지가 이 순서로 정해진다.
    - 결함 라벨인데 유효 기하가 0개면 레코드가 `None` 이다. 정상 서술을 붙이면 라벨과 텍스트가
      모순되는 조용한 오염이고, 결함 서술을 붙이면 위치 근거가 없다. 호출부가 폐기로 적는다.
    - 조항·행별 두께 구간·미특정 사유는 **유효 기하 결함의 코드**에서만 낸다.
    - 검증(`check_pair`)은 하지 않는다. 조립과 검증을 가른다.
    """
    defects = []
    skipped = 0
    for a in sorted(anns, key=lambda x: x["ann_id"]):
        if "geom_valid" not in a:
            raise PairInputError(f"{a.get('ann_id')}: geom_valid 열이 없다 — 없으면 전부 유효로 열린다")
        if not truth(a["geom_valid"], f"{a['ann_id']}.geom_valid"):
            skipped += 1          # 무기록 스킵이 manifest n_defects 와 4건 어긋났다 (M10)
            continue
        defects.append({
            "type": str(a["iso_code"]),
            "bbox_px": [int(a["bbox_x1_px"]), int(a["bbox_y1_px"]),
                        int(a["bbox_x2_px"]), int(a["bbox_y2_px"])],
            "size_px": {"major_axis": float(a["major_axis_px"]),
                        "equiv_diameter": float(a["equiv_diameter_px"])},
            "size_mm": None,   # 스케일 부재 (함정 #10) — mm 는 원리적으로 불가
        })
    if truth(row["has_defect"], f"{row['image_id']}.has_defect") and not defects:
        return None, skipped
    codes: list[str] = []
    for d in defects:
        if d["type"] not in codes:
            codes.append(d["type"])
    clauses = sorted({base[c]["clause_id"] for c in codes if c in base})
    candidate_rules = sorted((r for c in codes if c in base for r in base[c]["rules"]),
                             key=lambda r: r["rule_id"])
    why = UNCITED_CODE if base else UNCITED_MATERIAL
    rec = {
        "image_id": row["image_id"],
        "image_path": str(row["rel_path"]).replace("\\", "/"),
        "client": row["client"],
        "split": row["split"],
        "material": str(row["material"]),
        # bbox 경계 검사의 근거를 레코드가 들고 다닌다 — 파일을 다시 열지 않아도 상자를 검증할 수 있다.
        "width_px": int(row["width_px"]),
        "height_px": int(row["height_px"]),
        "coord_space": COORD_SPACE,     # 판정 1 — 규약을 산출물에 명시한다
        "skeleton": {
            "defects": defects,
            "verdict": None,          # 축 좁힘: 조항 검색 + 기준 서술 (합부 없음)
            "verdict_mode": "clause_only",
            "clauses": clauses,
            "candidate_rules": [dict(r) for r in candidate_rules],
            "uncited_codes": [{"code": c, "reason": why} for c in codes if c not in base],
        },
        "target_text": (defect_target_text(defects, names, base)
                        if defects else NORMAL_TEXT),
    }
    if skipped:
        # 기하 무효를 조용히 깎지 않는다. 개수 서술은 유효 기하 기준이고, 몇 개를
        # 뺐는지 레코드가 들고 다닌다.
        rec["n_annotations_skipped_geom_invalid"] = skipped
    return rec, skipped


class BuildResult:
    """`build_pairs` 의 산출. 파일은 쓰지 않는다."""

    def __init__(self):
        self.made: list[dict] = []
        self.discarded: list[dict] = []
        self.reasons: Counter = Counter()       # 사유별 건수 — 한 레코드가 둘 이상을 낼 수 있다
        self.n_input = 0
        self.n_geom_skipped = 0
        self.materials: list[str] = []
        self.covered: set[str] = set()
        self.uncovered: list[str] = []
        self.isolation: dict = {}
        #: 이미지 경로 실재를 **실제로** 몇 개 확인했나. 메타의 주장은 이 값에서 나온다 —
        #: 상수로 적으면 검사를 꺼도 메타가 계속 '확인했다' 고 말한다(검토 M-9).
        self.n_paths_checked: int = 0


def build_pairs(manifest, annotations, table, names: dict[str, str], lexicon: frozenset[str], *,
                root: Path | None = None, check_paths: bool = False) -> BuildResult:
    """train·val 행에서 페어를 조립하고 검증한다. eval 은 만들지 않는다(개발규약 1-4).

    입력 계약을 먼저 본다 — 어기면 `PairInputError` 로 멈춘다.
      · `split` 어휘가 train·val·eval 뿐이고 eval 이 비어 있지 않다 (표기가 어긋난 행이 조용히 빠지면
        그 이미지가 eval 과 같은 묶음이어도 격리 검사에 안 걸린다 — 검토 2·3)
      · train·val 의 `image_id` 유일 · `ann_id` 유일 · `client`·`material` 어휘와 표기
      · `has_defect` 와 `n_defects` 가 서로 맞는다 (한쪽만 보면 정상 라벨에 결함 페어가 붙는다 — 검토 7)
      · train·val 이 eval 과 겹치지 않는다 — `image_id`·`group_id` 에 더해 **`rel_path`·`sha256`**
        (id 만 보면 같은 파일이 다른 id 로 양쪽에 있어도 "겹침 0" 이 나온다 — 검토 1)
      · `group_id` 가 비어 있지 않다 (빈 값끼리는 겹침을 볼 수 없다 — 검토 2)
      · (`check_paths`) `rel_path` 가 전부 실재한다
      · 이미지마다 `유효 기하 결함 수 + 기하 무효 수 = 매니페스트 n_defects`
      · 허용치 표가 이 자료의 (재질 × 코드)에서 행을 고를 수 있다 — 두께 구간이 겹치면 멈춘다
    """
    from corpus.rules.schema import Material
    from corpus.validate.pair_criteria_gate import PairRules

    res = BuildResult()
    m = manifest
    odd = sorted(set(map(str, m["split"])) - {"train", "val", "eval"})
    if odd:
        raise PairInputError(f"모르는 split 표기: {odd[:5]} — 어느 쪽에도 안 들어가 조용히 빠진다")
    tv = m[m["split"].isin(["train", "val"])]
    ev = m[m["split"] == "eval"]
    res.n_input = len(tv)
    if not len(ev):
        raise PairInputError("eval 행이 0건이다 — 격리를 확인할 수 없는 상태를 '겹침 0' 으로 적지 않는다")
    if not res.n_input:
        raise PairInputError("train·val 행이 0건이다")
    if tv["image_id"].duplicated().any():
        raise PairInputError(f"train·val 에 image_id 중복 {int(tv['image_id'].duplicated().sum())}건")
    known = {x.value for x in Material} - {"ALL"}
    def _blanks(frame, col):
        return sorted({repr(v) for v in frame[col].tolist()
                       if not isinstance(v, str) or not v or v != v.strip()})

    for frame, cols in ((tv, ("client", "material")),
                        # 격리에 쓰는 열은 eval 쪽도 비어 있으면 안 된다 — 빈 값끼리는 겹침을 볼 수 없다.
                        (m, ("group_id", "rel_path", "sha256"))):
        for col in cols:
            if col not in frame.columns:
                continue
            odd = _blanks(frame, col)
            if odd:
                raise PairInputError(f"{col} 표기 이상: {odd[:5]}")
    odd = sorted(set(tv["material"]) - known)
    if odd:
        raise PairInputError(f"모르는 재질 표기: {odd[:5]} (아는 값 {sorted(known)})")
    odd = sorted(set(map(str, tv["client"])) - set(CLIENTS))
    if odd:
        raise PairInputError(f"모르는 참여자 표기: {odd[:5]} (아는 값 {list(CLIENTS)})")
    # 경로는 저장소 기준 상대경로여야 한다. 절대경로가 실리면 산출물에 로컬 위치가 들어간다(규약 2-6).
    odd = sorted({str(v) for v in tv["rel_path"]
                  if Path(str(v)).is_absolute() or ".." in Path(str(v).replace(chr(92), "/")).parts})[:3]
    if odd:
        raise PairInputError(f"rel_path 가 상대경로가 아니다: {odd}")
    # has_defect 와 n_defects 는 서로를 보증한다. 한쪽만 보면 정상 라벨 이미지에 결함 페어가 붙거나,
    # 결함 라벨인데 어노테이션이 아예 없는 입력 결함이 폐기로 세어진다(검토 7).
    bad_rows = [r.image_id for r in tv.itertuples()
                if truth(r.has_defect, f"{r.image_id}.has_defect") != (int(r.n_defects) > 0)]
    if bad_rows:
        raise PairInputError(f"has_defect 와 n_defects 가 어긋난 행 {len(bad_rows)}개 — 예: {bad_rows[:3]}")
    eval_ids, eval_groups = set(ev["image_id"]), set(ev["group_id"])
    hits = {"image_id": set(tv["image_id"]) & eval_ids, "group_id": set(tv["group_id"]) & eval_groups}
    for col in ("rel_path", "sha256"):        # 같은 파일이 다른 id 로 양쪽에 있는 경우(검토 1)
        if col in m.columns:
            hits[col] = {str(v).replace(chr(92), "/").lower() for v in tv[col]} &                         {str(v).replace(chr(92), "/").lower() for v in ev[col]}
    if any(hits.values()):
        raise PairInputError("평가 격리 위반 — eval 과 겹치는 " + " · ".join(f"{k} {len(v)}" for k, v in hits.items()))
    if check_paths:
        base_dir = Path(root) if root is not None else REPO
        missing = [str(p) for p in tv["rel_path"] if not (base_dir / str(p)).is_file()]
        if missing:
            raise PairInputError(f"이미지 경로 {len(missing)}개가 없다 — 예: {missing[:3]}")
        res.n_paths_checked = len(tv)

    res.materials = sorted({str(x) for x in tv["material"].unique()})
    # 재질별 근거를 따로 만든다. 덮는 행이 없는 재질은 빈 사전이고, 그 경우 텍스트가
    # 조항을 특정하지 않는다 (§4 안 B — 격리하면 C3 결함 페어가 0건이 되어
    # RQ3 이 인위적으로 바뀐다).
    bases = {mat: clause_basis(table, mat) for mat in res.materials}
    res.covered = covered_materials(table)
    res.uncovered = sorted(set(res.materials) - res.covered)
    rules = PairRules(table.rows, names)      # 게이트의 기대값 — 생성 경로(`bases`)와 따로 만든다
    ann_tv = annotations[annotations["image_id"].isin(set(tv["image_id"]))]
    if ann_tv["ann_id"].duplicated().any():
        # 정렬 키가 겹치면 행 순서가 산출 바이트를 정한다 — 같은 입력에서 결함 순서가 갈린다(검토 11).
        raise PairInputError(f"train·val 어노테이션에 ann_id 중복 {int(ann_tv['ann_id'].duplicated().sum())}건")
    # 표가 이 자료의 (재질 × 코드) 전부에서 행을 고를 수 있는지 **먼저** 본다. 조립 도중에 죽으면
    # 5만 건을 돌린 뒤에 멈춘다.
    for mat in res.materials:
        for code in sorted({str(c) for c in ann_tv["iso_code"]}):
            rules.rows_for(mat, code)

    anns = index_annotations(ann_tv)
    if set(anns) & eval_ids:
        raise PairInputError("색인에 eval 이미지의 어노테이션이 있다")
    group_of = dict(zip(m["image_id"], m["group_id"]))      # 재집계는 **전 행**의 사상으로 — 산출물에 무엇이 실렸든 센다
    for tup in tv.sort_values("image_id").itertuples(index=False):
        row = tup._asdict()
        iid = row["image_id"]
        rec, skipped = assemble_record(row, anns.get(iid, []), names, bases[str(row["material"])])
        res.n_geom_skipped += skipped
        n_valid = len(rec["skeleton"]["defects"]) if rec is not None else 0
        if n_valid + skipped != int(row["n_defects"]):
            raise PairInputError(f"{iid}: 유효 {n_valid} + 기하 무효 {skipped} ≠ 매니페스트 n_defects "
                                 f"{int(row['n_defects'])} — 어노테이션과 매니페스트가 어긋난다")
        if rec is None:
            bad = [NO_VALID_GEOMETRY]
        else:
            bad = check_pair(rec, names, rules, lexicon,
                             wh=(int(row["width_px"]), int(row["height_px"])))
        if bad:
            res.discarded.append({"image_id": iid, "reasons": bad})
            res.reasons.update(bad)
        else:
            res.made.append(rec)

    # 산출물에서 **다시** 센다 — 단언이 돈 것과 산출물이 깨끗한 것은 다른 사실이다.
    res.isolation = {
        "eval_image_id_overlap": len({r["image_id"] for r in res.made} & eval_ids),
        "eval_group_id_overlap": len({group_of[r["image_id"]] for r in res.made} & eval_groups),
        "n_eval_images": len(eval_ids),
        "n_eval_groups": len(eval_groups),
    }
    if res.isolation["eval_image_id_overlap"] or res.isolation["eval_group_id_overlap"]:
        raise PairInputError(f"산출물이 eval 과 겹친다: {res.isolation}")
    return res


def snapshot_digest(out_dir: Path, members) -> str:
    """`SNAPSHOT.sha256` 을 쓰고 결합 다이제스트를 돌려준다.

    줄의 꼴(`<해시>  <이름>`)은 매니페스트 스냅샷과 같지만 **다이제스트 산식은 다르다** — 여기서는 파일 해시
    문자열을 이어 붙여 해싱하고, `data/manifest_io` 는 줄 전체를 이어 붙여 해싱한다. 파일럿 v1 부터 그렇고,
    바꾸면 기존 자산의 digest 가 전부 달라진다. 소비처는 산식을 섞어 쓰지 마라.

    **digest 는 입력만의 함수가 아니다.** `counts.json` 이 실행 신원(`git_commit`·빌드 날짜)을 싣기 때문에,
    같은 입력이라도 커밋이 다르면 digest 가 달라진다. "같은 입력이면 같은 바이트" 는 `pairs.jsonl` 과
    `discarded.jsonl` 에 대한 주장이다.
    """
    entries = [(hashlib.sha256((out_dir / f).read_bytes()).hexdigest(), f) for f in members]
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    with (out_dir / CONTRACT_NAME).open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(f"{h}  {f}" for h, f in entries)
                 + f"\n# snapshot_digest {digest}\n")
    return digest


def write_outputs(out_dir: Path, res: BuildResult, *, asset: str, version: str, date: str,
                  limits_csv: Path, manifest_csv: Path, meta: dict) -> str:
    """페어·폐기·회계·메타·계약서를 쓴다. 전부 LF 다."""
    from corpus.generate.counts_builder import build_counts, write_counts

    # counts.json — 회계다(§7-5). 폐기는 **레코드 수**로 센다. 사유 수로 세면 한 레코드가 사유 둘을 낼 때
    # "폐기 합계 = 투입 − 채택" 정합이 깨진다(06번 §2-나).
    # **쓰기 전에** 만든다. 뒤에 두면 정합이 깨질 때 반쯤 쓴 디렉터리가 남는다(검토 4).
    class _Rec:
        def __init__(self, r):
            self.image_id = r["image_id"]
            self.defects = r["skeleton"]["defects"]
    client_of = {r["image_id"]: r["client"] for r in res.made}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO)
    if head.returncode != 0:
        # 실행 신원이 빈 문자열로 실리면 어느 코드가 만든 자산인지 산출물만으로 말할 수 없다.
        raise PairInputError(f"git rev-parse HEAD 실패 — 실행 신원을 적을 수 없다: {head.stderr.strip()[:80]}")
    counts = build_counts(
        [_Rec(r) for r in res.made], client_of,
        n_generated=res.n_input,
        discarded={"quarantine": len(res.discarded)} if res.discarded else {},
        limits_sha256=hashlib.sha256(limits_csv.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_csv.read_bytes()).hexdigest(),
        git_commit=head.stdout.strip(), date=date, asset=asset, version=version)

    if out_dir.exists() and any(out_dir.iterdir()):
        # 비어 있지 않은 곳에는 쓰지 않는다. 계약서가 없다는 이유로 남의 산출물을 덮던 경로다(검토 8).
        raise PairInputError(f"{out_dir} 가 비어 있지 않다 — 새 경로를 지정하라")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pairs.jsonl", res.made)
    write_jsonl(out_dir / "discarded.jsonl", res.discarded)
    write_counts(out_dir / "counts.json", counts)
    write_json(out_dir / "PAIRS_META.json", meta)
    return snapshot_digest(out_dir, ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"))


def counts_by_split(made: list[dict]) -> dict:
    """분할 × 참여자의 결함·정상 건수. `counts.json` 의 `clients` 는 train·val 을 합친 회계이고
    (스키마 동결), 학습기는 train 만 읽는다 — 그 구분을 메타가 싣는다."""
    out: dict = {}
    for r in made:
        cell = out.setdefault(r["split"], {}).setdefault(
            r["client"], {"n_total": 0, "n_defect": 0, "n_normal": 0})
        cell["n_total"] += 1
        cell["n_defect" if r["skeleton"]["defects"] else "n_normal"] += 1
    return {s: dict(sorted(v.items())) for s, v in sorted(out.items())}


def report(res: BuildResult, digest: str, out_dir: Path) -> None:
    made = res.made
    n_def = sum(1 for r in made if r["skeleton"]["defects"])
    n_noclause = sum(1 for r in made
                     if r["skeleton"]["defects"] and not r["skeleton"]["clauses"])
    by_split = Counter((r["split"], bool(r["skeleton"]["defects"])) for r in made)
    print(f"페어 {len(made)} (결함 {n_def} / 정상 {len(made)-n_def}) | 폐기 {len(res.discarded)}")
    print("분할별:", {f"{s}_{'결함' if d else '정상'}": n for (s, d), n in sorted(by_split.items())})
    print(f"조항 미특정 결함 페어: {n_noclause} (덮지 않는 재질 {res.uncovered})")
    print(f"기하 무효 스킵 어노테이션: {res.n_geom_skipped}")
    print("폐기 사유:", dict(res.reasons) or "없음")
    print(f"통과율: {len(made)}/{res.n_input} = {len(made)/res.n_input:.6f} (형식 통과율)")
    print(f"eval 교차: {res.isolation}")
    print(f"snapshot_digest {digest}")
    print("산출:", out_dir)


# ------------------------------------------------------------------ 본체

def v1_accounting() -> dict:
    """동결본 v1 의 알려진 결함 회계 — **"421건, 두 종류"** (80번 체크리스트 20).

    v1 은 규약 1-6 에 따라 재생성하지 않는다. 대신 무엇이 틀렸는지를 v2 산출물이 들고
    다니게 한다. RQ2·RQ3 을 v1 파일럿 결과 위에서 읽으려면 이 숫자가 함께 가야 한다.
    """
    return {
        "asset": "d4_pairs_pilot_v1",
        "status": "frozen_with_known_defects",
        "snapshot_digest": "63fc6b6e2d89e6fa5fba077de62afd4db43eb8f5f38ddda096cb9be78363efcc",
        "n_defect_pairs": 1495,
        "n_defective_citations": 421,
        "share_of_defect_pairs": 0.2816,
        "kinds": [
            {"kind": "material_axis",
             "n": 219,
             "detail": "알루미늄 이미지에 강재 전용 조항(KRA27-T15/T16/3D) 인용. "
                       "C3 결함 페어의 100% 다 — 전체 대비 8.3% 로 읽으면 RQ3 이 재는 "
                       "단위를 놓친다.",
             "fixed_in_v2": "재질 축을 보고, 덮는 행이 없으면 조항을 특정하지 않는다"},
            {"kind": "aggregate_length",
             "n": 286,
             "detail": "KRA27-T16 은 합계 길이 기준인데 개별 결함 한계로 서술됐다. "
                       "286건 중 '합계'·'총' 언급 0건.",
             "fixed_in_v2": "미표현 단서 부기 — 위 기준만으로 합부를 결정하지 않는다고 "
                            "명시. 집계 단위를 기계로 말하려면 limits CSV 스키마 축이 "
                            "필요하고 그것은 단일 소스 계약 변경이라 게이트 사항이다"},
        ],
        "note": "두 종류는 겹치지 않는다 (219 + 286 = 505 가 아니라 421 인 것은 "
                "같은 페어가 두 종류를 함께 맞은 84건이 있기 때문이다 — 실측으로 확인).",
    }


def base_meta(res: BuildResult, *, asset: str) -> dict:
    """페어 자산이 스스로 말해야 하는 것들 (G6-2·G6-3). counts 스키마는 단계명이 고정이라
    여기 따로 싣는다 — 무엇을 잰 값인지·무엇이 빠졌는지를 산출물이 들고 다닌다."""
    return {
        "asset": asset,
        "coord_space": COORD_SPACE,
        "coord_note": "bbox_px 는 원본 절대 픽셀이다 (불변조건 8). 모델 좌표 변환은 "
                      "vlm/coords.py 가 하고 ABS_ORIG 에서 그 변환은 항등이다. "
                      "판정 1 (main 47c4dbc).",
        "verdict_mode": "clause_only",
        "verdict_axis_policy": (
            "verdict 는 전건 null 이다. 두께·화소당 실치수가 표본 전체에서 결측이라"
            " (비결측 0건) 치수 기준 합부가 원리적으로 성립하지 않는다(함정 #10)."
            " 조건부 승급은 가정값 3종(두께·스케일·품질수준)을 B·D 가 같은 키로 읽는"
            " 인터페이스가 선 뒤에만 가능하고, 지표명에 '(조건부)' 와 가정값 ± 민감도"
            " 병기가 따라온다 — 의사결정로그 미정 항목."),
        "validated_by": "rule",
        "measures": ["format", "schema"],
        "materials": {"present": res.materials, "covered_by_limits": sorted(res.covered),
                      "uncovered": res.uncovered},
        "uncovered_policy": (
            "덮는 행이 없는 재질은 조항을 특정하지 않고 그 사실을 문장에 적는다"
            " (§4 안 B). 격리(안 A)하면 C3 결함 페어가 0건이 되어 RQ3 이"
            " 인위적으로 바뀐다."),
        "ordering": ("레코드는 image_id 오름차순, 결함은 ann_id **문자열** 오름차순이다('#10' < '#2')."
                     " 정렬은 학습 계약이다 — 학습기의 셔플이 행 위치에 시드를 걸고, 생성이 한도에서 잘릴 때"
                     " 어느 결함이 잘리는지가 이 순서로 정해진다."),
        # 두 수는 다르다: 빌드 전체에서 기하 무효로 뺀 어노테이션 수와, 그중 **채택된 레코드에** 실린 수.
        # 차이는 폐기된 이미지의 것이다. 한 수만 적으면 다른 수를 검산할 수 없다.
        "n_annotations_skipped_geom_invalid": res.n_geom_skipped,
        "n_annotations_skipped_in_records": sum(r.get("n_annotations_skipped_geom_invalid", 0) for r in res.made),
        "discard_reasons": dict(sorted(res.reasons.items())),
        "eval_isolation": res.isolation,
        "counts_by_split": counts_by_split(res.made),
    }


def main() -> None:
    import argparse

    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon
    from data.manifest_io import load_snapshot

    from corpus.generate.run_cycle_corpus import defect_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="산출 디렉터리. **기본값 없음** — 봉인본을 겨누는 사고를 막는다."
                         " 관례대로 판본 접미를 붙여라 (pairs_pilot_v3 …)")
    args = ap.parse_args()
    out_dir = Path(args.out)
    # 스냅샷 적재·페어 생성 전에 막는다. 뒤에서 막으면 몇 분을 버리고, 그 사이 산출물이
    # 봉인본 위에 떨어진다. 이름이 아니라 계약(`SNAPSHOT.sha256`)으로 판정한다.
    assert_not_frozen(out_dir, what="--out 대상", flag="--out")

    snap = load_snapshot(SNAP)
    table = limits_loader.load_limits(str(LIMITS_CSV), pilot=True)
    res = build_pairs(snap.manifest, snap.annotations, table, defect_names(), load_defect_lexicon())

    meta = base_meta(res, asset="d4_pairs_pilot_v2")
    meta["known_defects_in_v1"] = v1_accounting()
    digest = write_outputs(out_dir, res, asset="d4_pairs_pilot", version="v2", date="pilot",
                           limits_csv=LIMITS_CSV, manifest_csv=SNAP / "manifest.csv", meta=meta)
    report(res, digest, out_dir)


if __name__ == "__main__":
    main()
