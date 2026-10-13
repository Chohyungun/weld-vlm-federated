"""통합형 epoch 셔플 파생 시험 — 2026-09-21 총괄 판정 ③.

`random.Random(seed + ep)` 는 **등록 시드가 연속 정수라서 인접 시드가 순열을 공유한다.**
시드 3개 × epoch 3개의 9조합이 만드는 순열이 9개가 아니라 5개이고, 그중 하나는 세 시드가
함께 쓴다. 겹침이 조건마다 다르다 — 중앙·로컬은 epoch 오프셋만으로 파생되어 겹치고, 연합은
라운드 오프셋(10007)이 갈라 주어 겹치지 않는다. 그래서 시드 간 산포가 조건마다 다르게 잡히고,
그 산포가 회복률 분모의 표준편차로 들어가 채택 관문을 헐겁게 만든다.

여기서 고정하는 것은 넷이다.
1. 고친 파생이 9조합에서 **9개의 서로 다른 순열**을 낸다.
2. `PYTHONHASHSEED` 가 달라도 같은 순열이 나온다 — 재현성이 프로세스 환경에 묶이지 않는다.
3. 옛 파생이 5/9 로 겹친다(반례). 파생을 되돌리면 1번과 함께 이 시험이 깨진다.
4. 학습 루프가 실제로 고친 파생을 쓴다.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.round_runner import derive_seed  # noqa: E402

#: 본실험 등록 시드 3세트. **연속 정수인 것이 이 시험의 전제다.**
SEEDS = (20260828, 20260829, 20260830)
EPOCHS = (0, 1, 2)
N_ROWS = 50


def _perm_new(seed: int, ep: int) -> tuple[int, ...]:
    order = list(range(N_ROWS))
    random.Random(f"{seed}:{ep}").shuffle(order)
    return tuple(order)


def _perm_old(seed: int, ep: int) -> tuple[int, ...]:
    order = list(range(N_ROWS))
    random.Random(seed + ep).shuffle(order)
    return tuple(order)


def test_고친_파생은_아홉_조합이_아홉_순열을_낸다():
    perms = {_perm_new(s, ep) for s in SEEDS for ep in EPOCHS}
    assert len(perms) == len(SEEDS) * len(EPOCHS) == 9


def test_옛_파생은_아홉_조합이_다섯_순열로_줄어든다__반례():
    """되돌리면 깨지는 시험. 겹침의 **구조**까지 적어 둔다."""
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for s in SEEDS:
        for ep in EPOCHS:
            groups.setdefault(_perm_old(s, ep), []).append((s, ep))
    assert len(groups) == 5, "옛 파생의 겹침 수가 달라졌다 — 등록 시드가 바뀌었는지 본다"
    shared = sorted((sorted(v) for v in groups.values() if len(v) > 1), key=len, reverse=True)
    # 세 시드가 함께 쓰는 순열이 하나 있다: (828,2)·(829,1)·(830,0)
    assert [len(g) for g in shared] == [3, 2, 2]
    assert {s for s, _ in shared[0]} == set(SEEDS)


def test_라운드_클라이언트_파생과_합쳐도_겹치지_않는다():
    """연합 경로. `seed` 는 이미 derive_seed 를 지난 값이다."""
    perms = set()
    for base in SEEDS:
        for r in range(2):
            for c in range(3):
                for ep in EPOCHS:
                    perms.add(_perm_new(derive_seed(base, r, c), ep))
    assert len(perms) == len(SEEDS) * 2 * 3 * len(EPOCHS)


def test_PYTHONHASHSEED_에_의존하지_않는다():
    """문자열 시드는 sha512 를 거친다 — 해시 무작위화에 묶이면 재현이 깨진다."""
    code = (
        "import random;"
        "o=list(range(50));"
        "random.Random('20260828:1').shuffle(o);"
        "print(','.join(map(str,o)))"
    )
    outs = []
    for hs in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                   text=True, env=env, check=True).stdout.strip())
    assert outs[0] == outs[1] != ""
    assert outs[0] == ",".join(map(str, _perm_new(20260828, 1)))


def test_학습_루프가_고친_파생을_쓴다():
    src = Path("vlm/pilot_vlm.py").read_text(encoding="utf-8")
    assert 'random.Random(f"{seed}:{ep}")' in src
    assert "random.Random(seed + ep)" not in src, "옛 파생이 남아 있다"
