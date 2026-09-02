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
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

# 결함 어휘 락의 정본 (§4-6-1 ④). 재구현 금지 — skeleton_gen.py:34-38.
from corpus.generate.numeric_lock import find_defect_tokens

REPO = Path(__file__).resolve().parents[2]
SNAP = REPO / "data/processed/aihub71761_rt_v1_pilot3000"
OUT = REPO / "data/processed/pairs_pilot_v1"
LIMITS_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"

VERDICT_WORD = re.compile(r"합격|불합격|적합\s*판정|부적합\s*판정")


# ------------------------------------------------------------------ 근거 구축

def clause_basis(table) -> dict[str, dict]:
    """결함 코드 → 적용 조항과 두께 구간별 기준 서술 (RT 축).

    두께가 결측이라 행 하나를 특정할 수 없다. 조항 수준으로 묶고 구간별 기준을
    나열한다 — 파일럿 축(조항 검색 + 기준 서술)에는 이것으로 충분하며, 합부를
    말하지 않으므로 행 특정이 필요 없다.
    """
    from corpus.generate.run_cycle_corpus import thickness_ko, unit_ko, val

    by_code: dict[str, dict] = {}
    rows = [r for r in table.rows
            if getattr(r, "scope", "active") == "active"
            and val(r.inspection_method) in ("RT", "ALL")]
    rows.sort(key=lambda r: r.rule_id)
    for r in rows:
        ent = by_code.setdefault(r.defect_code, {"clause_id": r.clause_id, "criteria": []})
        if ent["clause_id"] != r.clause_id:
            # 같은 코드가 두 조항에 걸리면 조항 특정이 모호해진다. 파일럿 표에는 없고,
            # 생기면 페어 축을 다시 정해야 하므로 시끄럽게 실패한다.
            raise SystemExit(f"결함 {r.defect_code} 가 복수 조항에 걸린다: "
                             f"{ent['clause_id']} vs {r.clause_id}")
        if r.limit_rule == "none_permitted":
            ent["criteria"].append("크기와 무관하게 허용하지 않는다")
        else:
            unit = unit_ko(r.unit)
            if r.limit_rule == "const":
                c = f"{val(r.limit_value)} {unit} 이하"
            elif r.limit_rule == "prop_t":
                c = f"모재 두께의 {val(r.limit_factor)} 배 이하"
            else:
                c = (f"모재 두께의 {val(r.limit_factor)} 배 이하이고"
                     f" 최대 {val(r.limit_cap)} {unit}")
            ent["criteria"].append(f"두께 {thickness_ko(r.thickness_min, r.thickness_max)}: {c}")
    return by_code


# ------------------------------------------------------------------ 텍스트

def defect_target_text(defects: list[dict], names: dict[str, str],
                       basis: dict[str, dict]) -> str:
    """결함 페어의 골격 텍스트. 관찰 → 조항 → 기준. 합부는 말하지 않는다."""
    lines = []
    seen_codes = []
    for d in defects:
        code = d["type"]
        if code not in seen_codes:
            seen_codes.append(code)
    obs = []
    for code in seen_codes:
        n = sum(1 for d in defects if d["type"] == code)
        obs.append(f"{names.get(code, '결함')}(ISO 6520-1 코드 {code}) {n}개")
    lines.append("방사선투과 영상에서 " + ", ".join(obs) + "가 관찰된다.")
    for code in seen_codes:
        b = basis[code]
        lines.append(f"{names.get(code, '결함')}에 적용되는 조항은 {b['clause_id']} 이다.")
        if len(b["criteria"]) == 1:
            lines.append(f"이 조항의 기준은 {b['criteria'][0]}이다.")
        else:
            lines.append("이 조항의 기준은 " + " / ".join(b["criteria"]) + " 이다.")
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
    if VERDICT_WORD.search(rec.get("target_text", "")):
        bad.append("verdict_word")
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


# ------------------------------------------------------------------ 본체

def covered_materials(table) -> set[str]:
    """`clause_basis` 가 고르는 행 묶음이 덮는 재질 집합."""
    from corpus.generate.run_cycle_corpus import val

    return {val(r.material) for r in table.rows
            if getattr(r, "scope", "active") == "active"
            and val(r.inspection_method) in ("RT", "ALL")}


def main() -> None:
    from corpus.generate.run_cycle_corpus import defect_names
    from corpus.rules import limits_loader
    from corpus.rules.skeleton_gen import load_defect_lexicon
    from data.manifest_io import load_snapshot

    snap = load_snapshot(SNAP)
    table = limits_loader.load_limits(str(LIMITS_CSV), pilot=True)
    names = defect_names()
    lexicon = load_defect_lexicon()
    basis = clause_basis(table)
    valid_clauses = {b["clause_id"] for b in basis.values()}
    limit_texts = {b["clause_id"]: [c.split(": ", 1)[-1] for c in b["criteria"]]
                   for b in basis.values()}

    m = snap.manifest
    tv = m[m["split"].isin(["train", "val"])]

    # 재질 축 fail-closed (74번 감사 P4 소급 검토 결과).
    #
    # clause_basis 는 scope 와 검사 방식만 보고 **재질을 보지 않는다.** 정본 행 선택기
    # `limit_eval.applicable_row` 는 재질 축을 보고 해당 행이 없으면 fallback 없이 거절한다.
    # 파일럿 limits CSV 는 12행 전부 material=ST 라, 이 생성기를 그대로 돌리면 알루미늄
    # 이미지에 강재 조항이 붙는다 — v1 산출물에서 실제로 219건이 그렇게 나갔다.
    # 알루미늄 허용치 근거(KS-AL)는 아직 미확보다(sources.yaml status=pending).
    #
    # 어떻게 처리할지 — 격리 / "적용 가능한 조항 없음" 서술 / KS-AL 확보 대기 — 는
    # C3 이질성과 RQ3 에 걸리는 판단이라 함정 #6 미니스펙의 게이트 사항이다. 여기서
    # 정하지 않는다. 다만 **조용히 틀린 것을 다시 만들지는 않는다.**
    covered = covered_materials(table)
    seen = {str(x) for x in tv["material"].unique()}
    missing = sorted(seen - covered)
    if missing:
        raise SystemExit(
            f"재질 {missing} 을 덮는 허용치 행이 없다 (CSV 재질 축: {sorted(covered)}).\n"
            "그대로 진행하면 그 재질 이미지에 다른 재질의 조항이 붙는다 — v1 에서 219건이"
            " 그렇게 나갔다(74번 감사 P4 소급 검토).\n"
            "처리 방침은 함정 #6 미니스펙(docs/dev_log/2026-08-22-데이터확정/"
            "minispec_B_D4페어_판정논리통합.md)의 게이트 사항이다."
        )
    eval_ids = set(m[m["split"] == "eval"]["image_id"])
    anns = defaultdict(list)
    for _, a in snap.annotations.iterrows():
        anns[a["image_id"]].append(a)

    made, discarded = [], []
    reasons = Counter()
    for _, row in tv.sort_values("image_id").iterrows():
        iid = row["image_id"]
        assert iid not in eval_ids
        defects = []
        for a in sorted(anns.get(iid, []), key=lambda x: x["ann_id"]):
            if not bool(a.get("geom_valid", True)):
                continue
            defects.append({
                "type": str(a["iso_code"]),
                "bbox_px": [int(a["bbox_x1_px"]), int(a["bbox_y1_px"]),
                            int(a["bbox_x2_px"]), int(a["bbox_y2_px"])],
                "size_px": {"major_axis": float(a["major_axis_px"]),
                            "equiv_diameter": float(a["equiv_diameter_px"])},
                "size_mm": None,   # 스케일 부재 (함정 #10) — mm 는 원리적으로 불가
            })
        # 결함 라벨인데 유효 기하가 0개면 페어를 만들지 않는다. 정상 서술을 붙이면
        # 라벨과 텍스트가 모순되는 조용한 오염이고, 결함 서술을 붙이면 위치 근거가 없다.
        if bool(row["has_defect"]) and not defects:
            discarded.append({"image_id": iid, "reasons": ["defect_label_no_valid_geometry"]})
            reasons["defect_label_no_valid_geometry"] += 1
            continue
        clauses = sorted({basis[d["type"]]["clause_id"] for d in defects
                          if d["type"] in basis})
        rec = {
            "image_id": iid,
            "image_path": str(row["rel_path"]).replace("\\", "/"),
            "client": row["client"],
            "split": row["split"],
            "skeleton": {
                "defects": defects,
                "verdict": None,          # 축 좁힘: 조항 검색 + 기준 서술 (합부 없음)
                "verdict_mode": "clause_only",
                "clauses": clauses,
            },
            "target_text": (defect_target_text(defects, names, basis)
                            if defects else NORMAL_TEXT),
        }
        bad = check_pair(rec, names, valid_clauses, limit_texts,
                         (int(row["width_px"]), int(row["height_px"])), lexicon)
        if bad:
            discarded.append({"image_id": iid, "reasons": bad})
            for b in bad:
                reasons[b] += 1
        else:
            made.append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / "pairs.jsonl"
    pairs_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in made) + "\n",
                          encoding="utf-8")
    (OUT / "discarded.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in discarded) + ("\n" if discarded else ""),
        encoding="utf-8")

    # counts.json — 통합·연합 n_k 의 단일 소스 (스펙 §7-5). C 가 이것만 읽는다.
    from corpus.generate.counts_builder import build_counts, write_counts

    class _Rec:
        def __init__(self, r):
            self.image_id = r["image_id"]
            self.defects = r["skeleton"]["defects"]
    client_of = {r["image_id"]: r["client"] for r in made}
    counts = build_counts(
        [_Rec(r) for r in made], client_of,
        # counts 스키마(§7-5)는 단계명이 고정이다. 라벨-기하 모순은 외부 입력 이상이라
        # quarantine 으로 집계하고, 상세 사유는 discarded.jsonl 에 남는다.
        n_generated=len(tv),
        discarded={"quarantine": sum(reasons.values())} if reasons else {},
        limits_sha256=hashlib.sha256(LIMITS_CSV.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((SNAP / "manifest.csv").read_bytes()).hexdigest(),
        git_commit=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, cwd=REPO).stdout.strip(),
        date="pilot", asset="d4_pairs_pilot", version="v1")
    write_counts(OUT / "counts.json", counts)

    # SNAPSHOT.sha256 — A 스냅샷과 같은 형식 (파일별 해시 + 결합 다이제스트)
    entries = []
    for f in ("pairs.jsonl", "counts.json", "discarded.jsonl"):
        entries.append((hashlib.sha256((OUT / f).read_bytes()).hexdigest(), f))
    digest = hashlib.sha256("".join(h for h, _ in entries).encode()).hexdigest()
    (OUT / "SNAPSHOT.sha256").write_text(
        "\n".join(f"{h}  {f}" for h, f in entries) + f"\n# snapshot_digest {digest}\n",
        encoding="utf-8")

    n_def = sum(1 for r in made if r["skeleton"]["defects"])
    by_split = Counter((r["split"], bool(r["skeleton"]["defects"])) for r in made)
    print(f"페어 {len(made)} (결함 {n_def} / 정상 {len(made)-n_def}) | 폐기 {len(discarded)}")
    print("분할별:", {f"{s}_{'결함' if d else '정상'}": n for (s, d), n in sorted(by_split.items())})
    print("폐기 사유:", dict(reasons) or "없음")
    print(f"통과율: {len(made)/len(tv):.4f}")
    print("산출:", OUT)


if __name__ == "__main__":
    main()
