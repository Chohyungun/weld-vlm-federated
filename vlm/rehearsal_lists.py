"""리허설 목록 — 생성 · 에코 목록과 학습 행을 뽑는다(리허설 2판 §4-1 · §4-4 의 "뽑는 법" · §5-3 의 제외 · §5-4 의 노출 기록).

## 읽는 것과 읽지 않는 것(2판 §5-3 의 표)

동결 매니페스트는 **열을 고르는 읽기 함수**(`data.manifest_view.read_manifest_columns`)로만 읽는다 — train · val 행은
`image_id` · `split` · `group_id` · `strata_key` · `client` · `sha256`, 평가 행은 `image_id` · `split` · `sha256` · `group_id` 넷이다.
평가 행의 값은 교집합 수를 세고 성분을 잇는 데만 쓰고 산출에 싣지 않는다. 본실험 페어의 train · val 행과 근사 중복 후보 간선(id 쌍)을 읽는다.
평가 이미지 · 평가 정답 · 주석 파일은 열지 않는다.

## 뽑는 법

두 구현이 같은 목록을 내는지 맞대는 자리다(2판 §6-1 의 40). 갈렸던 세 자리는 2판 §13(3판 개정)이 정했다 — 3 · 6 · 8 이 그 자리다.
나머지 번호는 명세 문장이 정하지 않아 **이 구현이 고른 읽기**다.

1. 생성 목록의 후보는 매니페스트에서 분할이 `gen.split` 인 장 가운데 **페어에 같은 분할의 행이 있는 장**이다(에코 타깃이 페어 val 행에서 나온다).
   제외(아래 8)를 먼저 걷는다. 진단이면 페어 행의 결함이 하나 이상이고, `gen.max_target_tokens` 가 값이면 타깃 길이가 그 이하인 장만 남긴다.
2. 층은 `strata_key`, 묶음은 `group_id` 다. 층 안에서 묶음을 `sha256(f"{group_id}|{sampling_seed}")` 의 16진 문자열 오름차순으로 세운다.
   묶음을 넣으면 그 묶음의 남은 후보 장을 모두 넣는다.
3. 3~5 는 **2판 §13-2(3판 개정 — 총괄 결정 04 의 2절)** 를 따른다 — 최소 단계(상한 검사 없이, 끝까지) → 나머지를 최대 잔여 방식으로 나눈 정수 몫
   (잔여가 같으면 층 키 오름차순, 정확한 유리수) → 1차 채우기(층 순서대로, 층마다 이 단계에서 넣은 양이 몫에 닿을 때까지, 합이 목표에 닿으면 모두 멈춘다,
   장 수가 상한을 넘는 묶음은 버린다) → 2차 채우기(몫을 보지 않고 층 순서 → 층 안 순서) → 그래도 미달이면 거부.
6. 학습 행은 참여자마다 페어의 train 행 가운데 매니페스트의 train 행인 것(제외를 걷은 뒤)에서 뽑는다. 묶음을 2 의 순서로 넣어 행 수가
   목표 이상이 되면 멈추고, `train.avoid_multiple_of_accum` 이면 행 수가 누적 창(32)으로 나누어 떨어지는 동안 다음 묶음을 더 넣는다(§13-1 의 ②).
   묶음이 떨어져도 목표에 못 미치면 쓰지 않고 멈춘다. 중앙은 세 참여자 행의 합집합이다.
   진단 계획(`train.central_rows: direct`)은 참여자 행 없이 중앙 풀에서 바로 뽑는다 — 아래 10.
7. 목록 파일의 줄은 `evaluation.eval_list.canonical_bytes` 의 꼴이다. 학습 행 목록은 **페어 파일의 순서**를 지킨다(학습 행 digest 가 수열의 해시다).
8. 제외 — 매니페스트의 모든 id 를 꼭짓점으로, 근사 중복 후보 간선과 **같은 `group_id` 간선**으로 이은 연결 성분 가운데 평가 id 를 품은 성분의
   train · val 장을 모두 뺀다(묶음 간선을 함께 이으므로 성분은 묶음을 쪼개지 않는다). 매니페스트에 없는 id 가 간선에 있으면 입력 오류로 거부한다(§13-1 의 ①).
   기록하는 수는 train · val 장을 품은 그런 성분의 수, 그 안의 train · val 묶음 수와 장 수다.
9. 낸 목록(생성 · 학습)과 평가 id 의 교집합이 0 이 아니면 쓰지 않는다.
10. **중앙 직접 뽑기 — 총괄 결정 20(층화).** 후보는 세 참여자의 학습 행 전부(6 의 거름과 제외를 지난 것)다. 층은 `strata_key`,
    묶음은 `group_id` 다. 층의 몫은 목표 T 를 **층의 행 수**에 비례해 최대 잔여 방식으로 나눈 정수다(동률은 층 키 오름차순 — 3 과 같은 함수).
    층 순서(키 오름차순)대로, 층 안에서는 묶음을 `sha256(f"{group_id}|{sampling_seed}|central_direct")` 의 16진 문자열 오름차순
    (같으면 `group_id` 오름차순)으로 넣는다 — 생성 · 참여자 목록과 **다른 흐름**이다. 그 층에서 넣은 행이 몫 이상이 되면 다음 층으로 간다.
    마지막 묶음이 몫을 넘으면 그 묶음까지 넣고 다음 층의 몫에서 빼지 않는다(넘은 수는 기록에 싣는다). 몫이 0 인 층은 넣지 않는다.
    T 가 후보 행 수보다 크면 거부한다(몫은 층의 행 수를 넘지 않으므로 그 밖에는 모자랄 수 없다). 한 묶음의 행이 두 층에 걸치면 입력 오류로
    거부한다 — 층은 묶음 대표 클래스라 묶음 안에서 같아야 한다. `train.avoid_multiple_of_accum` 은 이 뽑기에서 정하지 않았다 — 참이면 거부한다.
11. **빈 층 — 총괄 결정 21 의 3 · 4 와 그 덧붙임(2판 §14 · §14-1 · §14-2).** 제외(8) **전에는** 후보가 있었는데(1 의 내용 거름 — 결함 · 타깃 길이 —
    을 지난 장) 제외 **뒤에** 후보 묶음이 0 이 된 층이다. 리허설의 생성 목록은 그 층을 **다른 곳에서도, 남은 층으로도 채우지 않는다** —
    **층별 몫은 빈 층이 있기 전의 배분 그대로다.** 빈 층에만 제외 전 후보를 돌려 둔 세계(그 층이 비지 않았을 세계)에서 3 의 절차를 목표 T 로
    한 번 돌리고, 빈 층에서 뽑힌 장만 걷어 낸다 — 남은 층의 선택은 그 세계와 같고(최소 단계 · 몫 · 상한 · 2차 채우기 모두), 빈 층의 몫만 0 이다.
    걷어 낸 장은 목록에 들지 않는다(셈에만 쓴다 — 평가와 이어진 장이다). T 미달을 허용하고 기록한다 —
    `unverified_strata`(층 키 · 사유 · 그 층의 몫 — 걷어 낸 양) · `gen_target`(목표 · 빈 몫의 합 · 남은 층이 낸 양 · 단위).
    묶음 · 장 수는 싣지 않는다(평가와 이어진 묶음을 특정할 수 있다).
    **진단 목적의 목록은 보류다**(결정 21 의 4 가 풀릴 때까지) — 목적 `frame_diag` 이거나 `train.central_rows: direct` 인 계획은 목록 단계에서
    `lists_diag_on_hold` 하나로 거부한다. 보류 아래의 두 거부(생성 목록의 빈 층 · 학습 풀(10)의 빈 층 — `lists_stratum_empty`)는 진단 재개 조건의 자리다.

간선 파일의 꼴은 아직 정해지지 않았다 — 이 모듈은 머리행 `a_id,b_id` 의 UTF-8 CSV 를 받는다(계획의 `exclude.near_dup_edges.sha256` 으로 바이트를 묶는다).
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

__all__ = [
    "ACCUM",
    "DIAG_LISTS_ON_HOLD",
    "DIRECT_STREAM",
    "TRAIN_TAGS",
    "ListsRefused",
    "ListsResult",
    "build_lists",
    "diag_hold",
    "direct_order_key",
    "exclusion",
    "is_diag_plan",
    "order_key",
    "read_edges",
    "sample_central_direct",
    "sample_gen",
    "sample_train",
]

ACCUM = 32
TRAIN_TAGS = {"C1": "uni_local_C1", "C2": "uni_local_C2", "C3": "uni_local_C3"}
CENTRAL_TAG = "uni_central"


class ListsRefused(ValueError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


@dataclass
class ListsResult:
    gen_ids: list[str]
    echo_ids: list[str]
    train_rows: dict[str, list[dict]]
    """태그 → 페어 행(페어 파일의 순서). `uni_local_C1` · `uni_local_C2` · `uni_local_C3` · `uni_central`."""
    record: dict[str, Any] = field(default_factory=dict)


def order_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{group_id}|{seed}".encode()).hexdigest()


#: 중앙 직접 뽑기의 순서 흐름 — 생성 · 참여자 목록의 순서(`order_key`)와 다른 흐름이다(총괄 결정 20 의 시드 행).
DIRECT_STREAM = "central_direct"


def direct_order_key(group_id: str, seed: int) -> tuple[str, str]:
    return hashlib.sha256(f"{group_id}|{seed}|{DIRECT_STREAM}".encode()).hexdigest(), group_id


def read_edges(raw: bytes) -> list[tuple[str, str]]:
    """머리행 `a_id,b_id` 의 UTF-8 CSV — id 쌍의 목록."""
    text = raw.decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or [c.strip() for c in rows[0]] != ["a_id", "b_id"]:
        raise ListsRefused("lists_edges_form", "간선 파일의 머리행이 a_id,b_id 가 아니다")
    out = []
    for n, r in enumerate(rows[1:], 2):
        if len(r) != 2 or not r[0] or not r[1]:
            raise ListsRefused("lists_edges_form", f"간선 파일 {n} 행의 꼴이 틀리다")
        out.append((r[0], r[1]))
    return out


def exclusion(manifest: Sequence[Mapping[str, Any]], edges: Iterable[tuple[str, str]]) -> tuple[set[str], dict]:
    """읽기 8 — 평가 id 를 품은 성분의 train · val 장."""
    split_of = {str(r["image_id"]): str(r["split"]) for r in manifest}
    group_of = {str(r["image_id"]): str(r["group_id"]) for r in manifest}
    parent: dict[str, str] = {i: i for i in split_of}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    first_of_group: dict[str, str] = {}
    for i in sorted(split_of):
        g = group_of[i]
        if g in first_of_group:
            union(first_of_group[g], i)
        else:
            first_of_group[g] = i
    for a, b in edges:
        if a not in split_of or b not in split_of:
            raise ListsRefused("lists_edge_unknown_id", "간선에 매니페스트에 없는 id 가 있다")
        union(a, b)
    comps: dict[str, list[str]] = defaultdict(list)
    for i in split_of:
        comps[find(i)].append(i)
    drop: set[str] = set()
    n_comp = 0
    for members in comps.values():
        if any(split_of[i] == "eval" for i in members):
            tv = [i for i in members if split_of[i] != "eval"]
            if tv:
                n_comp += 1
                drop.update(tv)
    rec = {"n_components": n_comp, "n_groups": len({group_of[i] for i in drop}), "n_images": len(drop)}
    return drop, rec


def largest_remainder(total: int, weights: Mapping[str, int]) -> dict[str, int]:
    """최대 잔여 방식의 정수 몫 — `⌊total·w/W⌋` 를 주고 남은 것을 잔여가 큰 층부터 하나씩. 잔여가 같으면 층 키 오름차순(2판 §13-2 의 2)."""
    W = sum(weights.values())
    if total <= 0 or W == 0:
        return {k: 0 for k in weights}
    exact = {k: Fraction(total * w, W) for k, w in weights.items()}
    q = {k: int(v) for k, v in exact.items()}           # Fraction 의 int 는 내림이다(값이 0 이상)
    left = total - sum(q.values())
    for k in sorted(weights, key=lambda k: (-(exact[k] - q[k]), k))[:left]:
        q[k] += 1
    return q


def sample_gen(cands: Sequence[Mapping[str, Any]], *, seed: int, target: int, by_boxes: bool, max_n: int,
               min_groups: int) -> list[str]:
    """생성 목록 — 2판 §13-2 의 절차(3판 개정). `cands` 의 행은 `image_id` · `group_id` · `strata_key` · `n_defects`."""
    unit = (lambda r: int(r["n_defects"])) if by_boxes else (lambda r: 1)
    strata: dict[str, dict[str, list[Mapping]]] = defaultdict(lambda: defaultdict(list))
    for r in cands:
        strata[str(r["strata_key"])][str(r["group_id"])].append(r)
    order = sorted(strata)
    queues = {s: [strata[s][g] for g in sorted(strata[s], key=lambda g: order_key(g, seed))] for s in order}
    weight = {s: sum(len(x) for x in queues[s]) for s in order}
    taken: list[Mapping] = []
    n_img = got = 0
    # 1 최소 단계 — 상한 검사 없이, 끝까지
    for s in order:
        for _ in range(min_groups):
            if queues[s]:
                g = queues[s].pop(0)
                taken.extend(g)
                n_img += len(g)
                got += sum(unit(r) for r in g)
    if n_img > max_n:
        raise ListsRefused("lists_min_groups_exceed_max", f"층마다 최소 묶음을 넣은 장 수 {n_img} 가 상한 {max_n} 을 넘는다")
    # 2 몫 — 나머지를 최대 잔여 방식으로
    quota = largest_remainder(max(0, target - got), weight)

    def put(s: str) -> int | None:
        """층 s 의 다음 묶음을 넣는다. 상한을 넘기는 묶음은 버리고 다음을 본다. 넣은 양(없으면 None)."""
        nonlocal n_img, got
        while queues[s]:
            g = queues[s].pop(0)
            if n_img + len(g) > max_n:
                continue
            taken.extend(g)
            n_img += len(g)
            v = sum(unit(r) for r in g)
            got += v
            return v
        return None

    # 3 1차 채우기 — 층마다 이 단계에서 넣은 양이 몫에 닿을 때까지, 합이 목표에 닿으면 모두 멈춘다
    for s in order:
        if got >= target:
            break
        here = 0
        while here < quota[s] and got < target:
            v = put(s)
            if v is None:
                break
            here += v
    # 4 2차 채우기 — 몫을 보지 않고 층 순서 → 층 안 순서
    for s in order:
        while got < target:
            if put(s) is None:
                break
    if got < target:
        raise ListsRefused("lists_target_not_reached", f"모든 층을 돌아도 목표 {target} 에 못 미친다({got})")
    return sorted(str(r["image_id"]) for r in taken)


def sample_train(rows: Sequence[Mapping[str, Any]], *, group_of: Mapping[str, str], seed: int, target: int,
                 avoid_multiple: bool, accum: int = ACCUM) -> set[str]:
    """읽기 6 — 한 참여자의 행에서 묶음 단위로. 돌려주는 것은 뽑은 `image_id` 집합이다."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        groups[group_of[str(r["image_id"])]].append(str(r["image_id"]))
    order = sorted(groups, key=lambda g: order_key(g, seed))
    picked: list[str] = []
    k = 0
    while len(picked) < target and k < len(order):
        picked.extend(groups[order[k]])
        k += 1
    if len(picked) < target:
        raise ListsRefused("lists_train_short", f"학습 행 {len(picked)} 이 목표 {target} 에 못 미친다")
    while avoid_multiple and len(picked) % accum == 0:
        if k >= len(order):
            raise ListsRefused("lists_train_multiple", f"학습 행 {len(picked)} 이 누적 창 {accum} 으로 나누어 떨어지고 더 넣을 묶음이 없다")
        picked.extend(groups[order[k]])
        k += 1
    return set(picked)


def sample_central_direct(rows: Sequence[Mapping[str, Any]], *, group_of: Mapping[str, str],
                          stratum_of: Mapping[str, str], seed: int, target: int) -> tuple[set[str], dict]:
    """읽기 10 — 중앙 풀에서 층 · 묶음 단위로 T 행(총괄 결정 20). 돌려주는 것은 뽑은 `image_id` 집합과 층별 기록이다."""
    strata: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    seen_stratum: dict[str, str] = {}
    for r in rows:
        i = str(r["image_id"])
        g, s = group_of[i], stratum_of[i]
        if seen_stratum.setdefault(g, s) != s:
            raise ListsRefused("lists_group_spans_strata", f"묶음 {g} 의 행이 두 층에 걸친다 — 층은 묶음 안에서 같아야 한다")
        strata[s][g].append(i)
    weight = {s: sum(len(v) for v in gs.values()) for s, gs in strata.items()}
    total = sum(weight.values())
    if target < 1 or target > total:
        raise ListsRefused("lists_train_short", f"중앙 직접 뽑기의 목표 {target} 이 후보 행 {total} 밖이다")
    quota = largest_remainder(target, weight)
    picked: list[str] = []
    by_stratum: dict[str, dict[str, int]] = {}
    for s in sorted(strata):
        here = 0
        for g in sorted(strata[s], key=lambda g: direct_order_key(g, seed)):
            if here >= quota[s]:
                break
            picked.extend(strata[s][g])
            here += len(strata[s][g])
        by_stratum[s] = {"rows": weight[s], "quota": quota[s], "taken": here, "over": here - quota[s]}
    rec = {"target": target, "n_rows": len(picked), "over": len(picked) - target, "by_stratum": by_stratum}
    return set(picked), rec


#: 진단 목적 목록의 보류(총괄 결정 21 의 4 와 그 덧붙임의 2) — 풀릴 때까지 참이다. 계획 · 명령줄 · 환경으로는 바꾸지 않는다.
DIAG_LISTS_ON_HOLD = True


def is_diag_plan(plan: Any) -> bool:
    return getattr(plan, "purpose", None) == "frame_diag" or plan.get("train.central_rows") == "direct"


def diag_hold(plan: Any) -> None:
    """진단 목적의 계획이면 목록 단계에서 거부한다(읽기 11) — 빈 층과 무관하다."""
    if DIAG_LISTS_ON_HOLD and is_diag_plan(plan):
        raise ListsRefused("lists_diag_on_hold", "진단은 보류다(총괄 결정 21 의 4) — 재개 조건이 닫힐 때까지 진단 목록을 쓰지 않는다")


def build_lists(manifest: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
                edges: Iterable[tuple[str, str]], plan: Any, *, target_tokens: Mapping[str, int] | None = None,
                accum: int = ACCUM) -> ListsResult:
    """매니페스트(허용 열) · 페어 행(train · val) · 간선 · 계획 → 목록. 계획은 `vlm.rehearsal_plan.Plan`."""
    diag_hold(plan)
    seed = int(plan.get("sampling_seed"))
    split_of = {str(r["image_id"]): str(r["split"]) for r in manifest}
    eval_ids = {i for i, s in split_of.items() if s == "eval"}
    drop, rec = (exclusion(manifest, edges) if plan.get("exclude.drop_components_touching_eval")
                 else (set(), {"n_components": 0, "n_groups": 0, "n_images": 0}))
    meta = {str(r["image_id"]): r for r in manifest if r["split"] != "eval"}
    gsplit = plan.get("gen.split")
    pair_of = {str(r["image_id"]): r for r in pairs if r.get("split") == gsplit}
    defect_only = bool(plan.get("gen.defect_only"))
    max_tok = plan.get("gen.max_target_tokens")
    cands = []
    for i, m in meta.items():
        if m["split"] != gsplit or i in drop or i not in pair_of:
            continue
        n_def = len((pair_of[i].get("skeleton") or {}).get("defects") or [])
        if defect_only and n_def < 1:
            continue
        if max_tok is not None:
            if target_tokens is None or i not in target_tokens:
                raise ListsRefused("lists_target_tokens_missing", "타깃 길이로 거르려면 장마다 길이가 있어야 한다")
            if target_tokens[i] > int(max_tok):
                continue
        cands.append({"image_id": i, "group_id": m["group_id"], "strata_key": m["strata_key"], "n_defects": n_def})
    # 읽기 11 — 제외 전에는 후보가 있었는데 제외 뒤에 0 묶음이 된 층
    before: dict[str, set[str]] = defaultdict(set)
    pre_cands: list[dict] = []
    for i, m in meta.items():
        if m["split"] != gsplit or i not in pair_of:
            continue
        n_def = len((pair_of[i].get("skeleton") or {}).get("defects") or [])
        if defect_only and n_def < 1:
            continue
        if max_tok is not None and (target_tokens is None or i not in target_tokens or target_tokens[i] > int(max_tok)):
            continue
        before[str(m["strata_key"])].add(str(m["group_id"]))
        pre_cands.append({"image_id": i, "group_id": m["group_id"], "strata_key": m["strata_key"], "n_defects": n_def})
    after = {str(c["strata_key"]) for c in cands}
    empty = sorted(s for s in before if s not in after)
    diag = is_diag_plan(plan)
    if empty and diag:
        raise ListsRefused("lists_stratum_empty", f"제외 뒤 후보 묶음이 0 인 층이 있다 {empty} — 진단은 빈 층을 미검증으로 넘기지 않는다(결정 21 의 4)")
    by_boxes = plan.get("gen.target_boxes") is not None
    target = int(plan.get("gen.target_boxes") if by_boxes else plan.get("gen.target_n"))
    # 빈 층의 몫은 비운다(결정 21 덧붙임의 1) — 층별 몫은 빈 층이 있기 전의 배분 그대로: 빈 층에만 제외 전 후보를 돌려 둔 세계에서
    # 목표 T 로 뽑고 빈 층의 장만 걷는다(남은 층은 다시 나누지 않는다 — 외부 검토 회신 emptystrata2 의 1)
    phantom = [c for c in pre_cands if str(c["strata_key"]) in set(empty)]
    phantom_ids = {str(c["image_id"]) for c in phantom}
    picked = sample_gen(cands + phantom, seed=seed, target=target, by_boxes=by_boxes, max_n=int(plan.get("gen.max_n")),
                        min_groups=int(plan.get("gen.min_groups_per_stratum")))
    units = {str(c["image_id"]): (int(c["n_defects"]) if by_boxes else 1) for c in cands + phantom}
    share = {s: 0 for s in empty}
    for c in phantom:
        if str(c["image_id"]) in picked:
            share[str(c["strata_key"])] += units[str(c["image_id"])]
    gen = [i for i in picked if i not in phantom_ids]
    empty_share = sum(share.values())
    unverified = [{"stratum": s, "reason": "제외(R0) 뒤 후보 묶음 0 — 다른 곳에서도 남은 층으로도 채우지 않는다", "share": share[s]}
                  for s in empty]
    gen_target = {"target": target, "empty_share": empty_share, "filled": sum(units[i] for i in gen),
                  "unit": "boxes" if by_boxes else "images"}
    if not gen:
        raise ListsRefused("lists_gen_empty", "생성 목록이 비었다 — 후보가 남은 층이 없다")
    group_of = {i: str(m["group_id"]) for i, m in meta.items()}
    tsplit = plan.get("train.split")
    train_rows: dict[str, list[dict]] = {}
    picked_all: set[str] = set()
    rows_target = plan.get("train.rows") or {}
    central_rows = plan.get("train.central_rows")
    direct_rec = None

    def pool(client: str | None) -> list[Mapping[str, Any]]:
        return [r for r in pairs if r.get("split") == tsplit and (client is None or r.get("client") == client)
                and r.get("client") in TRAIN_TAGS
                and str(r["image_id"]) in meta and meta[str(r["image_id"])]["split"] == tsplit
                and str(r["image_id"]) not in drop]

    if central_rows == "direct":
        if plan.get("train.avoid_multiple_of_accum"):
            raise ListsRefused("lists_direct_accum", "중앙 직접 뽑기에서 누적 창의 배수 피하기는 정하지 않았다")
        rows = pool(None)
        train_before = {str(meta[str(r["image_id"])]["strata_key"]) for r in pairs
                        if r.get("split") == tsplit and r.get("client") in TRAIN_TAGS and str(r["image_id"]) in meta
                        and meta[str(r["image_id"])]["split"] == tsplit}
        train_empty = sorted(train_before - {str(meta[str(r["image_id"])]["strata_key"]) for r in rows})
        if train_empty:                                          # 읽기 11 — 진단 재개 조건의 자리(학습 풀의 층별 지지량)
            raise ListsRefused("lists_stratum_empty", f"제외 뒤 학습 풀의 후보가 0 인 층이 있다 {train_empty} — 진단은 빈 층을 넘기지 않는다")
        picked_all, direct_rec = sample_central_direct(
            rows, group_of=group_of, stratum_of={i: str(m["strata_key"]) for i, m in meta.items()}, seed=seed,
            target=int(rows_target["central"]))
    elif central_rows == "union":
        for client, tag in TRAIN_TAGS.items():
            rows = pool(client)
            ids = sample_train(rows, group_of=group_of, seed=seed, target=int(rows_target[client]),
                               avoid_multiple=bool(plan.get("train.avoid_multiple_of_accum")), accum=accum)
            train_rows[tag] = [dict(r) for r in rows if str(r["image_id"]) in ids]
            picked_all |= ids
    else:
        raise ListsRefused("lists_central_rows", f"모르는 train.central_rows: {central_rows!r}")
    train_rows[CENTRAL_TAG] = [dict(r) for r in pairs if r.get("split") == tsplit and str(r["image_id"]) in picked_all]
    inter = len((set(gen) | picked_all) & eval_ids)
    if inter:
        raise ListsRefused("lists_eval_intersection", f"목록과 평가셋의 교집합이 {inter} 이다")
    by_stratum: dict[str, int] = defaultdict(int)
    by_material: dict[str, int] = defaultdict(int)
    for i in gen:
        sk = str(meta[i]["strata_key"])
        by_stratum[sk] += 1
        by_material[sk.split("|", 1)[0]] += 1
    record = {"excluded": rec, "eval_intersection": inter, "n_gen": len(gen), "unverified_strata": unverified,
              "gen_target": gen_target,
              "n_train": {t: len(v) for t, v in train_rows.items()},
              "exposure_by_stratum": dict(sorted(by_stratum.items())),
              "exposure_by_material": dict(sorted(by_material.items()))}
    if direct_rec is not None:
        record["central_direct"] = direct_rec
    return ListsResult(gen_ids=gen, echo_ids=list(gen), train_rows=train_rows, record=record)
