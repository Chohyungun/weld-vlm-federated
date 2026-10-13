"""근사 중복 13단계 — **2차 자동 판독용 묶음을 만든다.** 사람 검수가 아니다.

왜 하나. 1차 판독은 A 한 사람(한 판독 단위)이 했다. 그 라벨 자체의 불확실성을 재려면
같은 쌍을 독립으로 다시 읽어야 한다. **이것은 두 번째 자동 판독이고 사람 검수를 대신하지 않는다** —
계약은 D 의 `evaluation/second_read.py` 에 있고, 그 모듈이 읽는 쪽에서 표시를 요구한다.

넘기는 것에 **식별자를 넣지 않는다.** 쌍을 `SR-00`~`SR-59` 로만 부른다.
몽타주에도 식별자가 찍히지 않으므로(행 번호만 찍는다) 넘기는 묶음 전체에 실물을 가리키는 것이 없다.
되찾는 사상표는 이 트리의 추적하지 않는 경로에만 둔다.

**내 판정을 먼저 봉인한다.** 2차 판독을 받은 뒤에 내 판정을 고치면 일치율이 뜻을 잃는다.
넘기기 전에 내 60쌍 판정의 지문을 남기고, 일치율은 그 지문이 그대로일 때만 낸다.

표본은 개발 180쌍에서 뽑는다. **보류 120쌍은 넘기지 않는다** — 보류는 한 번만 보고, 그 한 번은 문턱을 얼린 뒤다.

사용: python nd_stage13_handoff.py [씨앗]
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "_workspace" / "2026-09-21-neardup"
OUT = ROOT / "outputs" / "neardup_second_read"        # 워크트리 사이에 정션으로 공유된다. 추적하지 않는다
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nd_stage12_fit import load_dev  # noqa: E402

SEED = 20260925
N = 60
csv.field_size_limit(10_000_000)

CRITERIA = """# 근사 중복 판독 기준 v1 (1차 판독자 A 가 정함 · 2026-09-25)

읽는 것은 **두 장이 같은 방사선 사진인가** 하나다. 라벨·분할·참여자는 보지 않는다(주어지지도 않는다).

몽타주 한 줄은 [영상 A 안쪽 | 영상 B 안쪽 | 두 영상의 절대차 ×8] 이다. B 는 기록된 이동만큼 밀어 맞춘 뒤 뺐다.
머리글의 수치는 참고이고 **판정 근거로 쓰지 않는다** — 그 수치를 쓰면 독립 판독이 아니다.

## 어휘 셋

- **같음** — 같은 방사선 사진이다. 같은 자리에 같은 결함·표식·실루엣이 있고, 다른 것은 밝기·잡음·수십 px 이동뿐이다.
- **다름** — 다른 영상이다. **같은 필름의 다른 프레임을 포함한다** — 하단에 인쇄된 문자열이 같아도
  본문(용접부)의 구조가 다르면 다름이다. 이 계열이 이 자료에서 가장 흔한 오판이다.
- **불확실** — 위 둘 중 하나로 정할 수 없다. 아래 둘은 **반드시** 불확실이다.
  - 양쪽 안쪽 상자에 잴 구조가 없다(거의 무늬 없는 평탄면). 무늬 없는 면은 무엇과도 맞으므로 차이가 0 이어도 같음이 아니다.
  - 일부 구조는 맞고 일부는 어긋난다(예: 한쪽 가장자리 형상은 같은데 본문 덩어리의 수가 다르다).

## 무엇을 보고 정하나

1. **결함·표식의 자리** — 어두운 기공·균열·밝은 조각이 같은 자리에 같은 모양으로 있는가. 이것이 가장 무겁다.
2. **경계·실루엣의 형상** — 밝기 경계선의 굽은 모양이 겹치는가.
3. **하단 인쇄 문자** — 노출 번호가 다르면 다른 프레임이다. **다만 같다고 같음이 되지는 않는다.**
4. **차이 판** — 정합이 맞았을 때만 뜻이 있다. 이동이 크면 차이 판은 아무것도 말해 주지 않는다.

## 하지 않을 것

- 머리글 수치로 판정하지 않는다.
- 애매하면 한쪽으로 몰지 않는다. **불확실은 정상적인 답이다** — 일치율에서 따로 센다.
- 다른 판독자의 판정을 보지 않는다. 받은 것은 몽타주와 이 기준뿐이다.
"""


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    dev = load_dev()
    by = {}
    for r in dev:
        by.setdefault(r["층"], []).append(r)
    picked = []
    for s in sorted(by):
        rnd = random.Random(f"{seed}:sr:{s}")
        rows = sorted(by[s], key=lambda r: r["행"])
        rnd.shuffle(rows)
        picked += rows[:round(len(rows) * N / len(dev))]
    rnd = random.Random(f"{seed}:sr:order")
    rnd.shuffle(picked)                      # 층이 순서로 드러나지 않게 섞는다
    picked = picked[:N]

    OUT.mkdir(parents=True, exist_ok=True)
    # **줄끝을 LF 로 고정해 쓴다.** write_text 는 윈도에서 CRLF 로 바꿔 놓는데, 해시는 메모리의
    # LF 문자열에서 계산되므로 파일 바이트와 통보한 해시가 어긋난다. 2026-09-25 실제로 그랬다
    # (바이트 068026bb… · LF 98cb81a8…). 받는 쪽이 바이트로 재검증하면 거기서 멈춘다.
    (OUT / "판독기준_v1.md").write_bytes(CRITERIA.encode("utf-8"))
    crit_sha = hashlib.sha256(CRITERIA.encode("utf-8")).hexdigest()

    # 몽타주용 입력(식별자 포함) — 추적하지 않는 경로에만 둔다
    feat_rows = []
    sample = {i: s for i, s in enumerate(_read_sample())}
    for k, r in enumerate(picked):
        s = sample[r["행"]]
        feat_rows.append({"image_id_a": s["image_id_a"], "image_id_b": s["image_id_b"],
                          "rd_dx": s["rd_dx"], "rd_dy": s["rd_dy"]})
    src = W / "sr60_input.csv"
    with src.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id_a", "image_id_b", "rd_dx", "rd_dy"])
        w.writeheader()
        w.writerows(feat_rows)

    # 되찾는 사상표 — 식별자와 내 판정이 들어간다. 넘기지 않는다
    with (W / "sr60_map_비공개.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "표본_행", "층", "구간", "A_판정"])
        for k, r in enumerate(picked):
            w.writerow([f"SR-{k:02d}", r["행"], r["층"], r["구간"], r["판정"]])

    # **내 판정을 먼저 봉인한다.** 2차 판독을 받은 뒤에 고치면 일치율이 뜻을 잃는다
    mine = [{"pair_id": f"SR-{k:02d}", "verdict": r["판정"]} for k, r in enumerate(picked)]
    seal = hashlib.sha256(json.dumps(mine, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    # 넘기는 것 — 식별자가 하나도 없다
    with (OUT / "판독할_쌍.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id", "몽타주", "몽타주_안_줄", "두번째_자동판독_사람검수아님"])
        for k in range(len(picked)):
            w.writerow([f"SR-{k:02d}", f"montage/sr_{k // 6:02d}.png", k % 6, ""])
    meta = {
        "무엇인가": "2차 자동 판독용 묶음. **사람 검수가 아니다.**",
        "계약": "evaluation/second_read.py (D 소관). read_type=automated_second_read · human_review=False 상수 · 한계에 '사람 판독 미측정'",
        "기준": {"criteria_id": "neardup_read_v1", "criteria_sha256": crit_sha,
               "allowed_verdicts": ["같음", "다름", "불확실"], "source": "A(1차 판독자)"},
        "쌍_수": len(picked),
        "표본_출처": "sample_v1.csv 의 개발 180쌍에서 층 비율대로. **보류 120쌍은 넘기지 않았다**",
        "씨앗": seed,
        "A_판정_봉인_sha256": seal,
        "봉인의_뜻": "A 는 이 지문을 남긴 뒤 자기 판정을 고치지 않는다. 일치율은 지문이 그대로일 때만 낸다",
        "넘기는_것에_식별자가_없다": "쌍은 SR-00~SR-59 로만 부르고 몽타주에도 행 번호만 찍힌다",
        "판정을_적는_법": "판독할_쌍.csv 의 마지막 열에 같음·다름·불확실 중 하나를 적는다. 열 이름이 곧 표시다",
    }
    (OUT / "안내.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (W / "sr60_seal.json").write_text(json.dumps(
        {"seal": seal, "mine": mine}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**meta, "층별": {s: sum(1 for r in picked if r["층"] == s) for s in sorted(by)},
                      "몽타주_할_일": f"python nd_montage.py {src} {OUT / 'montage'} sr 0,1,..,{len(picked)-1} --align"},
                     ensure_ascii=False, indent=2))
    return 0


def _read_sample() -> list[dict]:
    with (W / "sample_v1.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


if __name__ == "__main__":
    sys.exit(main())
