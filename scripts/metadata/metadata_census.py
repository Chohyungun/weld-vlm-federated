"""함정 10 재확인 — 라벨 원본의 **키를 전수로 세어** 치수 정보가 정말 결측인지 본다.

왜. 매니페스트의 `thickness_mm`·`px_per_mm` 가 비어 있다는 것은 **우리 변환기가 채우지 않았다** 는 뜻이지
원본에 없다는 뜻이 아니다. 변환기는 자기가 아는 키만 읽는다. 그래서 라벨 JSON 을 열어 나타나는 키를 전부 센다.

## 이 조사가 지키는 것

1. **배열은 첫 항목만 보지 않는다.** 리스트 안의 모든 원소를 돌아 키의 합집합을 만든다.
   원소마다 키가 다른 자료에서 첫 항목만 보면 "전 키" 라고 말할 수 없다.
2. **분모와 실패를 따로 적는다.** zip 열기 · 구성원 읽기 · JSON 파싱 · 순회를 단계로 나누고 각 단계의 시도 수와 실패 수를 남긴다.
   하나의 성공률로 합치지 않는다. 실패를 조용히 건너뛰면 "전수" 가 거짓이 된다.
3. **기본 산출에 값을 내보내지 않는다.** 키 이름 · 자료형 · 건수만 낸다.
   원시 값에는 파일명·식별자·자유 텍스트가 섞여 있어 공개 문서로 새어 나갈 수 있다.
   값의 예가 필요하면 `--with-values` 로 **비공개 경로에만** 따로 뽑는다.
4. **덮어쓰지 않는다.** 출력 경로는 실행마다 배타로 만든다. 이미 있으면 멈춘다.

영상 안에 인쇄된 정보(필름 문자열·IQI·눈금)는 이 스크립트가 보지 않는다 — `film_text_read.py` 가 본다.

사용: python scripts/metadata/metadata_census.py [--with-values] [--out <경로>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data/interim/aihub_labels"
OUT_ROOT = ROOT / "_workspace" / "2026-09-25-metadata"

#: 치수 기준 판정에 쓸 수 있는 정보로 읽힐 만한 **키 이름** — 넓게 잡고 사람이 나중에 거른다.
DIM_HINT = re.compile(
    r"thick|두께|mm|pixel|픽셀|해상도|resolution|spacing|pitch|scale|배율|눈금|iqi|penetrameter|투과도|"
    r"wire|와이어|magnif|sod|sfd|focal|distance|거리|kv|ma|expos|노출",
    re.IGNORECASE)


def type_name(v) -> str:
    if v is None:
        return "null"
    return {bool: "bool", int: "int", float: "float", str: "str"}.get(type(v), type(v).__name__)


def walk(o, prefix: str, keys: dict[str, Counter], lens: dict[str, list],
         values: dict[str, Counter] | None = None) -> None:
    """중첩 구조를 평평한 키 경로로 편다. **리스트는 모든 원소를 돈다** — 원소마다 키가 다를 수 있다.

    `values` 가 주어지면 값의 예도 모은다. 그 산출은 **비공개 경로 전용**이다(식별자가 섞인다).
    """
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, f"{prefix}.{k}" if prefix else str(k), keys, lens, values)
    elif isinstance(o, list):
        lens.setdefault(prefix + "[]", []).append(len(o))
        for item in o:
            walk(item, prefix + "[]", keys, lens, values)
    else:
        keys.setdefault(prefix, Counter())[type_name(o)] += 1
        if values is not None and len(values[prefix]) < 20:
            values[prefix][str(o)[:60]] += 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-values", action="store_true",
                    help="키마다 값의 예를 함께 뽑는다. 식별자가 섞일 수 있어 **비공개 경로에만** 쓴다")
    ap.add_argument("--out", type=Path, default=None, help="출력 경로. 비우면 실행 시각으로 새로 만든다")
    a = ap.parse_args(argv)

    stamp = datetime.now(UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    out = a.out or (OUT_ROOT / f"census_{stamp}.json")
    if out.exists():
        print(f"!! 이미 있다: {out} — 덮지 않는다", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)

    # 단계별 회계 — 하나의 성공률로 합치지 않는다
    acct = Counter()
    fail_examples: dict[str, Counter] = defaultdict(Counter)
    keys: dict[str, Counter] = {}
    lens: dict[str, list] = {}
    files_with_key: Counter = Counter()
    values: dict[str, Counter] = defaultdict(Counter)

    zips = sorted(LABELS.glob("*.zip"))
    acct["zip_발견"] = len(zips)
    for zp in zips:
        try:
            z = zipfile.ZipFile(zp)
        except (OSError, zipfile.BadZipFile) as e:
            acct["zip_열기_실패"] += 1
            fail_examples["zip_열기"][type(e).__name__] += 1
            continue
        with z:
            # 확장자로 거르되 **전체 구성원 수와 배제 수를 남긴다.** 허용목록은 조용히 자료를 버릴 수 있다.
            members = z.namelist()
            names = [n for n in members if n.lower().endswith(".json")]
            acct["구성원_전체"] += len(members)
            acct["json_구성원"] += len(names)
            acct["확장자로_배제"] += len(members) - len(names)
            for n in members:
                if not n.lower().endswith(".json"):
                    ext = Path(n).suffix.lower() or "(확장자 없음)"
                    fail_examples["확장자_배제"][ext] += 1
            for name in names:
                acct["읽기_시도"] += 1
                try:
                    raw = z.read(name)
                except (OSError, zipfile.BadZipFile) as e:
                    acct["읽기_실패"] += 1
                    fail_examples["읽기"][type(e).__name__] += 1
                    continue
                acct["파싱_시도"] += 1
                try:
                    o = json.loads(raw.decode("utf-8-sig"))
                except (ValueError, UnicodeDecodeError) as e:
                    acct["파싱_실패"] += 1
                    fail_examples["파싱"][type(e).__name__] += 1
                    continue
                acct["순회_시도"] += 1
                k_local: dict[str, Counter] = {}
                l_local: dict[str, list] = {}
                try:
                    walk(o, "", k_local, l_local, values if a.with_values else None)
                except RecursionError as e:
                    acct["순회_실패"] += 1
                    fail_examples["순회"][type(e).__name__] += 1
                    continue
                acct["순회_성공"] += 1
                for k, tc in k_local.items():
                    keys.setdefault(k, Counter()).update(tc)
                for k, ls in l_local.items():
                    lens.setdefault(k, []).extend(ls)
                # **파일 하나에 키 하나씩만 센다.** 배열 안이 스칼라면 같은 키가 두 표에 다 들어와
                # 따로 더하면 한 파일을 두 번 센다(예: {"a": [1, 2]} 의 `a[]`).
                for k in set(k_local) | set(l_local):
                    files_with_key[k] += 1

    n_ok = acct["순회_성공"]
    print(f"zip {acct['zip_발견']} · 구성원 전체 {acct['구성원_전체']:,} "
          f"(json {acct['json_구성원']:,} · 확장자로 배제 {acct['확장자로_배제']:,}) · 순회 성공 {n_ok:,}")
    print("단계별 회계 (분모와 실패를 따로 적는다)")
    for stage in ("zip_발견", "zip_열기_실패", "구성원_전체", "확장자로_배제", "json_구성원",
                  "읽기_시도", "읽기_실패", "파싱_시도", "파싱_실패", "순회_시도", "순회_실패", "순회_성공"):
        print(f"  {stage:12s} {acct[stage]:>8,}")
    if any(acct[k] for k in ("zip_열기_실패", "읽기_실패", "파싱_실패", "순회_실패")):
        print("  ! 실패가 있다 — 아래 키 집계는 순회 성공분만의 것이다")

    print(f"\n서로 다른 키 {len(keys) + len(lens):,}개 (스칼라 {len(keys):,} · 배열 {len(lens):,})")
    print(f"{'키':44s} {'있는 파일':>9} {'자료형'}")
    for k in sorted(keys, key=lambda x: -files_with_key[x]):
        t = " ".join(f"{tn}:{c:,}" for tn, c in keys[k].most_common())
        print(f"  {k:42s} {files_with_key[k]:>9,} {t}")
    for k in sorted(lens, key=lambda x: -files_with_key[x]):
        ls = lens[k]
        print(f"  {k:42s} {files_with_key[k]:>9,} list(길이 중앙 {sorted(ls)[len(ls)//2] if ls else 0})")

    hints = sorted([k for k in list(keys) + list(lens) if DIM_HINT.search(k)])
    print(f"\n치수·촬영 조건으로 읽힐 만한 **키 이름**: {len(hints)}개")
    for k in hints:
        print(f"  ★ {k}  (있는 파일 {files_with_key[k]:,})")
    if not hints:
        print("  (없다 — 라벨 원본에 두께·배율·IQI·촬영 조건 키가 존재하지 않는다)")

    payload = {
        "생성": stamp,
        "회계": dict(acct),
        "실패_유형": {k: dict(v) for k, v in fail_examples.items()},
        "주의": ("키 집계는 순회 성공분만의 것이다. 실패 수와 확장자로 배제한 수를 함께 읽는다. "
                 "값은 담지 않는다 — 식별자가 섞일 수 있다."),
        "키": {k: {"있는_파일": files_with_key[k], "자료형": dict(keys[k])} for k in sorted(keys)},
        "배열키": {k: {"있는_파일": files_with_key[k], "길이_중앙": sorted(lens[k])[len(lens[k]) // 2] if lens[k] else 0}
                   for k in sorted(lens)},
        "치수_후보_키이름": hints,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ {out}  (키·자료형·건수만. 값은 담지 않는다)")
    if a.with_values:
        vout = out.with_name(out.stem + "_values_비공개.json")
        vout.write_text(json.dumps(
            {"경고": "값에 식별자·파일명·자유 텍스트가 섞여 있다. 공개 문서에 옮기지 않는다.",
             "값": {k: dict(v) for k, v in sorted(values.items())}}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"→ {vout.name}  **비공개**. 공개본에는 집계만 낸다", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
