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

실행: uv run python -m corpus.generate.make_pairs_pilot
산출: data/processed/pairs_pilot_v1/ (pairs.jsonl + counts.json + SNAPSHOT.sha256)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# 어휘 락·판정어·아티팩트 검출의 정본 (§4-6-1 ④ · 80번 G2-6·G3-1).
# **여기에 같은 성격의 정규식을 다시 두지 않는다** — 배선 시험이 강제한다.
from corpus.generate.numeric_lock import (
    find_artifacts,
    find_defect_tokens,
    find_verdict_implying,
)

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "data/processed/aihub71761_rt_v1_pilot3000"
#: v1 은 동결이다 (규약 1-6). 정정본은 v2 로 가른다 — 알려진 결함 421건의 회계가
#: v1 에 붙어 있고, 같은 경로에 덮어쓰면 그 회계가 가리키는 실물이 사라진다.
OUT = REPO / "data/processed/pairs_pilot_v2"
FROZEN_V1 = REPO / "data/processed/pairs_pilot_v1"
LIMITS_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"

#: 좌표 규약 (총괄 판정 1, main 47c4dbc). 페어의 `bbox_px` 는 **원본 절대 픽셀**이고
#: 모델 좌표 변환은 학습 쪽 `vlm/coords.py` 가 한다. ABS_ORIG 에서 그 변환은 항등이지만
#: 규약을 산출물에 **명시**한다 — 함정 #4 는 규약이 암묵일 때 터진다.
COORD_SPACE = "ABS_ORIG"

#: 재질을 덮는 허용치 행이 없을 때의 서술 (미니스펙 §4 안 B).
#: 결함은 서술하되 조항을 특정하지 않는다. 근거가 없다는 것은 지어낸 사실이 아니라 사실이다.
NO_CLAUSE_TAIL = ("이 재질에 적용할 허용치 조항이 허용치 표에 없어"
                  " 적용 조항을 특정하지 않는다.")

# ------------------------------------------------------------------ 근거 구축

def clause_basis(table, material: str) -> dict[str, dict]:
    """결함 코드 → 적용 조항과 두께 구간별 기준 서술 (RT 축, **재질 축 포함**).

    v1 에서 이 함수가 재질을 보지 않았다. 파일럿 허용치 표 12행이 전부 `material=ST`
    라서, 알루미늄 이미지에 강재 전용 조항(KRA27-T15/T16/3D)이 그대로 붙었다 —
    C3 결함 페어 219건 **전량**이다 (74번 P4 · 80번 B11).

    정본 행 선택기 `limit_eval.applicable_row` 는 재질 축을 보고 해당 행이 없으면
    fallback 없이 거절한다. 여기서도 같은 축을 본다. 덮는 행이 없으면 빈 사전을 내고,
    호출부가 "적용 조항을 특정하지 않는다"로 닫는다 (미니스펙 §4 안 B).

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
        ent = by_code.setdefault(r.defect_code, {"clause_id": r.clause_id, "criteria": []})
        if ent["clause_id"] != r.clause_id:
            # 같은 코드가 두 조항에 걸리면 조항 특정이 모호해진다. 파일럿 표에는 없고,
            # 생기면 페어 축을 다시 정해야 하므로 시끄럽게 실패한다.
            raise SystemExit(f"결함 {r.defect_code} 가 복수 조항에 걸린다: "
                             f"{ent['clause_id']} vs {r.clause_id}")
        band = thickness_ko(r.thickness_min, r.thickness_max)
        head = band if band == "모든 두께" else f"두께 {band}"
        ent["criteria"].append(f"{head}: {criterion_ko(r)}")
        if (getattr(r, "note", None) or "").strip():
            ent["has_unexpressed"] = True
    return by_code


def covered_materials(table) -> set[str]:
    """`clause_basis` 가 고르는 행 묶음이 덮는 재질 집합."""
    from corpus.rules.clause_text import val

    return {val(r.material) for r in table.rows
            if getattr(r, "scope", "active") == "active"
            and val(r.inspection_method) in ("RT", "ALL")}


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
    for code in codes:
        b = basis.get(code)
        if b is None:
            continue
        lines.append(f"{names.get(code, '결함')}에 적용되는 조항은 {b['clause_id']} 이다.")
        crit = b["criteria"]
        lines.append(f"이 조항의 기준은 {crit[0]}이다." if len(crit) == 1
                     else "이 조항의 기준은 " + " / ".join(crit) + " 이다.")
        if b.get("has_unexpressed"):
            # 원천 표에 기계 표현으로 옮기지 못한 단서가 있다 (집계 단위·무시 하한 등).
            # KRA27-T16 은 **합계 길이** 기준인데 스키마에는 길이 한계로만 실린다 —
            # v1 은 그것을 개별 결함 한계처럼 서술해 286건에 실었다 (80번 B12).
            lines.append("이 조항에는 기계 표현으로 옮기지 못한 단서가 있어"
                         " 위 기준만으로 합부를 결정하지 않는다.")
    if missing:
        lines.append(NO_CLAUSE_TAIL)
    lines.append("모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다.")
    return " ".join(lines)


NORMAL_TEXT = ("방사선투과 영상에서 검출 한계 내 특기할 지시가 관찰되지 않는다. "
               "인용할 허용치 조항이 없다.")


# ------------------------------------------------------------------ 검증

def check_pair(rec: dict, names: dict[str, str], valid_clauses: set[str],
               limit_texts: dict[str, list[str]], wh: tuple[int, int],
               lexicon: frozenset[str]) -> list[str]:
    bad: list[str] = []
    for k in ("image_id", "image_path", "client", "split", "skeleton", "target_text"):
        if not rec.get(k):
            bad.append(f"schema:{k}")
    sk = rec.get("skeleton") or {}
    if sk.get("verdict") is not None:
        bad.append("verdict_present")
    text = rec.get("target_text", "")
    # 판정어·판정 함의 표현은 정본 하나로 본다 (G2-6). 자체 정규식을 두면 "적합하다"
    # 처럼 낱말을 피한 표현이 통째로 새고, 그 구멍은 한쪽에만 생긴다.
    if find_verdict_implying(text):
        bad.append("verdict_word")
    # 내부 표현이 문장으로 샌 흔적 (G3-1). 골격 텍스트라 지금은 나올 수 없지만,
    # 윤문을 붙이는 본실험 페어에서 이 검사가 실검사가 된다.
    if find_artifacts(text, basis=""):
        bad.append("artifact_violation")
    W, H = wh
    for d in sk.get("defects", []):
        if d["type"] not in names:
            bad.append("unknown_code")
        x1, y1, x2, y2 = d["bbox_px"]
        if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
            bad.append("bbox_out_of_bounds")
        if d.get("size_mm") is not None:
            bad.append("mm_present")          # 스케일 부재 — mm 가 있으면 지어낸 값이다
    for c in sk.get("clauses", []):
        if c not in valid_clauses:
            bad.append("clause_unknown")
        else:
            # 부등식 방향: 기준 서술의 한계 표현이 limits 행(전부 le="이하")과 일치하는가
            for frag in limit_texts[c]:
                if frag not in rec["target_text"] and sk.get("defects"):
                    bad.append("criterion_mismatch")
                    break
    if not sk.get("defects"):
        # 정상 페어 — 결함 어휘 금지 (부정 문맥 포함, §4-6-1 ③④).
        # 검출은 **정본 하나**만 쓴다 — `numeric_lock.find_defect_tokens`
        # (`skeleton_gen` 이 재수출). 여기서 맨 부분문자열 루프로 재구현했더니
        # NFKC 정규화·대소문자 불문·숫자 경계가 빠졌고, 그것은 skeleton_gen.py:34-38
        # 이 명시적으로 금지한 것이다 (74번 감사 P6).
        if find_defect_tokens(rec["target_text"], lexicon):
            bad.append("defect_word_in_normal")
        if sk.get("clauses"):
            bad.append("normal_has_clauses")
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


def main() -> None:
    import argparse

    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon
    from data.manifest_io import load_snapshot

    from corpus.generate.run_cycle_corpus import defect_names

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    if out_dir.resolve() == FROZEN_V1.resolve():
        raise SystemExit("v1 은 동결이다 (규약 1-6) — 덮어쓰지 않는다. --out 을 바꿔라.")

    snap = load_snapshot(SNAP)
    table = limits_loader.load_limits(str(LIMITS_CSV), pilot=True)
    names = defect_names()
    lexicon = load_defect_lexicon()

    m = snap.manifest
    tv = m[m["split"].isin(["train", "val"])]
    materials = sorted({str(x) for x in tv["material"].unique()})
    # 재질별 근거를 따로 만든다. 덮는 행이 없는 재질은 빈 사전이고, 그 경우 텍스트가
    # 조항을 특정하지 않는다 (미니스펙 §4 안 B — 격리하면 C3 결함 페어가 0건이 되어
    # RQ3 이 인위적으로 바뀐다).
    bases = {mat: clause_basis(table, mat) for mat in materials}
    covered = covered_materials(table)
    uncovered = sorted(set(materials) - covered)
    valid_clauses = {b["clause_id"] for base in bases.values() for b in base.values()}
    limit_texts = {b["clause_id"]: [c.split(": ", 1)[-1] for c in b["criteria"]]
                   for base in bases.values() for b in base.values()}

    eval_ids = set(m[m["split"] == "eval"]["image_id"])
    anns = defaultdict(list)
    for _, a in snap.annotations.iterrows():
        anns[a["image_id"]].append(a)

    made, discarded = [], []
    reasons = Counter()
    n_geom_skipped = 0
    for _, row in tv.sort_values("image_id").iterrows():
        iid = row["image_id"]
        assert iid not in eval_ids
        mat = str(row["material"])
        base = bases[mat]
        defects = []
        skipped = 0
        for a in sorted(anns.get(iid, []), key=lambda x: x["ann_id"]):
            if not bool(a.get("geom_valid", True)):
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
        n_geom_skipped += skipped
        # 결함 라벨인데 유효 기하가 0개면 페어를 만들지 않는다. 정상 서술을 붙이면
        # 라벨과 텍스트가 모순되는 조용한 오염이고, 결함 서술을 붙이면 위치 근거가 없다.
        if bool(row["has_defect"]) and not defects:
            discarded.append({"image_id": iid, "reasons": ["defect_label_no_valid_geometry"]})
            reasons["defect_label_no_valid_geometry"] += 1
            continue
        clauses = sorted({base[d["type"]]["clause_id"] for d in defects
                          if d["type"] in base})
        rec = {
            "image_id": iid,
            "image_path": str(row["rel_path"]).replace("\\", "/"),
            "client": row["client"],
            "split": row["split"],
            "material": mat,
            "coord_space": COORD_SPACE,     # 총괄 판정 1 — 규약을 산출물에 명시한다
            "skeleton": {
                "defects": defects,
                "verdict": None,          # 축 좁힘: 조항 검색 + 기준 서술 (합부 없음)
                "verdict_mode": "clause_only",
                "clauses": clauses,
            },
            "target_text": (defect_target_text(defects, names, base)
                            if defects else NORMAL_TEXT),
        }
        if skipped:
            # 기하 무효를 조용히 깎지 않는다. 개수 서술은 유효 기하 기준이고, 몇 개를
            # 뺐는지 레코드가 들고 다닌다.
            rec["n_annotations_skipped_geom_invalid"] = skipped
        bad = check_pair(rec, names, valid_clauses, limit_texts,
                         (int(row["width_px"]), int(row["height_px"])), lexicon)
        if bad:
            discarded.append({"image_id": iid, "reasons": bad})
            for b in bad:
                reasons[b] += 1
        else:
            made.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pairs.jsonl", made)
    write_jsonl(out_dir / "discarded.jsonl", discarded)

    # counts.json — 통합·연합 n_k 의 단일 소스 (스펙 §7-5). C 가 이것만 읽는다.
    from corpus.generate.counts_builder import build_counts, write_counts

    class _Rec:
        def __init__(self, r):
            self.image_id = r["image_id"]
            self.defects = r["skeleton"]["defects"]
    client_of = {r["image_id"]: r["client"] for r in made}
    counts = build_counts(
        [_Rec(r) for r in made], client_of,
        n_generated=len(tv),
        discarded={"quarantine": sum(reasons.values())} if reasons else {},
        limits_sha256=hashlib.sha256(LIMITS_CSV.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((SNAP / "manifest.csv").read_bytes()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=REPO).stdout.strip(),
        date="pilot", asset="d4_pairs_pilot", version="v2")
    write_counts(out_dir / "counts.json", counts)

    # 페어 자산이 스스로 말해야 하는 것들 (G6-2·G6-3). counts 스키마는 단계명이 고정이라
    # 여기 따로 싣는다 — 무엇을 잰 값인지·무엇이 빠졌는지를 산출물이 들고 다닌다.
    meta = {
        "asset": "d4_pairs_pilot_v2",
        "coord_space": COORD_SPACE,
        "coord_note": "bbox_px 는 원본 절대 픽셀이다 (불변조건 8). 모델 좌표 변환은 "
                      "vlm/coords.py 가 하고 ABS_ORIG 에서 그 변환은 항등이다. "
                      "총괄 판정 1 (main 47c4dbc).",
        "verdict_mode": "clause_only",
        "verdict_axis_policy": (
            "verdict 는 전건 null 이다. 두께·화소당 실치수가 표본 전체에서 결측이라"
            " (비결측 0건) 치수 기준 합부가 원리적으로 성립하지 않는다(함정 #10)."
            " 조건부 승급은 가정값 3종(두께·스케일·품질수준)을 B·D 가 같은 키로 읽는"
            " 인터페이스가 선 뒤에만 가능하고, 지표명에 '(조건부)' 와 가정값 ± 민감도"
            " 병기가 따라온다 — 의사결정로그 미정 항목."),
        "validated_by": "rule",
        "measures": ["format", "schema"],
        "materials": {"present": materials, "covered_by_limits": sorted(covered),
                      "uncovered": uncovered},
        "uncovered_policy": (
            "덮는 행이 없는 재질은 조항을 특정하지 않고 그 사실을 문장에 적는다"
            " (미니스펙 §4 안 B). 격리(안 A)하면 C3 결함 페어가 0건이 되어 RQ3 이"
            " 인위적으로 바뀐다."),
        "n_annotations_skipped_geom_invalid": n_geom_skipped,
        "known_defects_in_v1": v1_accounting(),
    }
    write_json(out_dir / "PAIRS_META.json", meta)

    # SNAPSHOT.sha256 — A 스냅샷과 같은 형식 (파일별 해시 + 결합 다이제스트)
    entries = []
    for f in ("pairs.jsonl", "counts.json", "discarded.jsonl", "PAIRS_META.json"):
        entries.append((hashlib.sha256((out_dir / f).read_bytes()).hexdigest(), f))
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    with (out_dir / "SNAPSHOT.sha256").open("w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(f"{h}  {f}" for h, f in entries)
                 + f"\n# snapshot_digest {digest}\n")

    n_def = sum(1 for r in made if r["skeleton"]["defects"])
    n_noclause = sum(1 for r in made
                     if r["skeleton"]["defects"] and not r["skeleton"]["clauses"])
    by_split = Counter((r["split"], bool(r["skeleton"]["defects"])) for r in made)
    print(f"페어 {len(made)} (결함 {n_def} / 정상 {len(made)-n_def}) | 폐기 {len(discarded)}")
    print("분할별:", {f"{s}_{'결함' if d else '정상'}": n for (s, d), n in sorted(by_split.items())})
    print(f"조항 미특정 결함 페어: {n_noclause} (덮지 않는 재질 {uncovered})")
    print(f"기하 무효 스킵 어노테이션: {n_geom_skipped}")
    print("폐기 사유:", dict(reasons) or "없음")
    print(f"통과율: {len(made)/len(tv):.4f}")
    print(f"snapshot_digest {digest}")
    print("산출:", out_dir)


if __name__ == "__main__":
    main()
