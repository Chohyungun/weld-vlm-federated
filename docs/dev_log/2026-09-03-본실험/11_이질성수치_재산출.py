"""동결 스냅샷에서 이질성 수치 재산출 (79번 C8 폐쇄).

논문 §3.2 가 싣는 이질성 수치(TVD·슬래그비·규모비·클래스 구성비)를
동결 매니페스트에서 직접 다시 계산한다. 승인된 경로(load_snapshot, 해시 검증 포함)만
쓰고 원본 폴더를 직접 읽지 않는다.

다중라벨 이미지의 클래스 배정 규칙 두 가지를 모두 계산한다.

  규칙 A (이미지-유형 발생): 이미지가 가진 서로 다른 결함 유형마다 1씩 센다.
    정상 이미지는 정상 범주에 1. 동결 문서(58번)의 클래스×client 표와 같은 기준.
  규칙 B (인스턴스): annotations 의 결함 인스턴스 행을 센다. 정상 이미지는
    인스턴스가 없으므로 4결함 분포로만 계산한다.

실행:  .venv/Scripts/python docs/dev_log/2026-09-03-본실험/11_이질성수치_재산출.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data.manifest_io import load_snapshot  # noqa: E402

SNAPSHOT_DIR = ROOT / "data" / "interim" / "manifest_v1"
CLASSES = ["crack", "porosity", "lack_of_fusion", "slag_inclusion"]
KO = {
    "crack": "균열",
    "porosity": "기공",
    "lack_of_fusion": "융합불량",
    "slag_inclusion": "슬래그혼입",
    "normal": "정상",
}


def tvd(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def dist(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def main() -> None:
    snap = load_snapshot(SNAPSHOT_DIR)  # verify=True 기본 — 해시 어긋나면 여기서 죽는다
    m = snap.manifest
    print(f"snapshot_id={snap.snapshot_id}  rows={len(m)}")

    # ---- 규모비 (이미지 수, 재질별) -------------------------------------------
    n_by_mat = m.groupby("material").size().to_dict()
    st, al = n_by_mat["ST"], n_by_mat["AL"]
    print(f"\n[규모] ST {st:,} : AL {al:,}  = 1 : {st / al:.2f} (AL 기준)")

    # ---- 다중라벨 실태 --------------------------------------------------------
    def types_of(row) -> list[str]:
        if not row or (isinstance(row, float)):
            return []
        return sorted({t for t in str(row).split(";") if t})

    type_lists = m["defect_types"].map(types_of)
    n_multi = int((type_lists.map(len) > 1).sum())
    n_defect = int(m["has_defect"].sum())
    print(f"\n[다중라벨] 결함 이미지 {n_defect:,}장 중 유형 2종 이상 {n_multi:,}장 "
          f"({n_multi / n_defect:.4%})")

    # ---- 규칙 A: 이미지-유형 발생 (5범주) -------------------------------------
    counts_a: dict[str, dict[str, int]] = {"ST": {}, "AL": {}}
    for mat, tl, has in zip(m["material"], type_lists, m["has_defect"]):
        cats = tl if has else ["normal"]
        for c in cats:
            counts_a[mat][c] = counts_a[mat].get(c, 0) + 1

    print("\n[규칙 A] 이미지-유형 발생 (정상 포함 5범주)")
    pa = {mat: dist(counts_a[mat]) for mat in ("ST", "AL")}
    for c in CLASSES + ["normal"]:
        print(f"  {KO[c]:<6} ST {counts_a['ST'].get(c, 0):>7,} ({pa['ST'].get(c, 0):6.2%})"
              f"   AL {counts_a['AL'].get(c, 0):>6,} ({pa['AL'].get(c, 0):6.2%})"
              f"   AL/ST 비율배수 {pa['AL'].get(c, 0) / pa['ST'].get(c, 1e-12):.2f}")
    print(f"  TVD(A) = {tvd(pa['ST'], pa['AL']):.4f}")
    slag_a = pa["AL"]["slag_inclusion"] / pa["ST"]["slag_inclusion"]
    print(f"  슬래그비(A) = {slag_a:.2f}")

    # ---- 규칙 B: 인스턴스 수준 (4결함) ----------------------------------------
    ann = snap.annotations.merge(m[["image_id", "material"]], on="image_id", how="left")
    counts_b = {
        mat: g["defect_type"].value_counts().to_dict()
        for mat, g in ann.groupby("material")
    }
    print("\n[규칙 B] 결함 인스턴스 (4결함)")
    pb = {mat: dist(counts_b[mat]) for mat in ("ST", "AL")}
    for c in CLASSES:
        print(f"  {KO[c]:<6} ST {counts_b['ST'].get(c, 0):>7,} ({pb['ST'].get(c, 0):6.2%})"
              f"   AL {counts_b['AL'].get(c, 0):>6,} ({pb['AL'].get(c, 0):6.2%})")
    print(f"  TVD(B, 4결함) = {tvd(pb['ST'], pb['AL']):.4f}")
    slag_b = pb["AL"]["slag_inclusion"] / pb["ST"]["slag_inclusion"]
    print(f"  슬래그비(B) = {slag_b:.2f}")

    # 규칙 A 를 4결함으로 좁힌 값 — 규칙 간 차이를 같은 범주에서 비교하기 위한 참고
    pa4 = {mat: dist({c: counts_a[mat].get(c, 0) for c in CLASSES}) for mat in ("ST", "AL")}
    print(f"\n[참고] 규칙 A 를 4결함으로 좁힌 TVD = {tvd(pa4['ST'], pa4['AL']):.4f}"
          f" / 슬래그비 = {pa4['AL']['slag_inclusion'] / pa4['ST']['slag_inclusion']:.2f}")

    # ---- 클라이언트 축 (73번이 인용하는 값 검산, 규칙 A) ----------------------
    print("\n[클라이언트] 학습 풀 (eval 제외), 규칙 A")
    pool = m[m["client"].notna()]
    sizes = pool.groupby("client").size()
    print("  규모:", {k: f"{v:,}" for k, v in sizes.to_dict().items()},
          f" 최대/최소 {sizes.max() / sizes.min():.2f}")
    for cl in sorted(sizes.index):
        sub = pool[pool["client"] == cl]
        tl = sub["defect_types"].map(types_of)
        inc: dict[str, int] = {}
        for lst, has in zip(tl, sub["has_defect"]):
            for c in (lst if has else ["normal"]):
                inc[c] = inc.get(c, 0) + 1
        n = len(sub)
        parts = "  ".join(
            f"{KO[c]} {inc.get(c, 0):,}({inc.get(c, 0) / n:.1%})" for c in CLASSES
        )
        print(f"  {cl}: {parts}")


if __name__ == "__main__":
    main()
