"""D16의 사전 고정 합성 입력 작성. 실물 명부/영상/결과를 읽지 않는다.

N+는 규칙 작성자가 아닌 별도 작성자의 파일을 필수로 받는다. 이 도구는
양성 기대를 정책 통과 여부에 맞춰 고치지 않는다. 모델 호출도 하지 않는다.
"""

from __future__ import annotations

import base64
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
import hashlib
from pathlib import Path
import re

from data import label_map

from .common import assert_gate_closed, canonical_bytes, parse_json, read_json, sha256_file, write_json_new

LIMITATIONS = (
    "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다.",
    "두께와 실치수 정보가 없어 합격 · 불합격은 판단하지 않는다.",
    "모재 두께 정보가 주어지지 않아 치수 기준 합부는 판단하지 않는다.",
)
POSITIONS = ("위-왼쪽", "위-가운데", "위-오른쪽", "가운데-왼쪽", "가운데-가운데",
             "가운데-오른쪽", "아래-왼쪽", "아래-가운데", "아래-오른쪽")


def _label_names(path=None) -> dict[str, str]:
    loaded = label_map.load_label_map(label_map.DEFAULT_LABEL_MAP if path is None else path)
    names = {key: loaded.defect_types[key].name_ko for key in ("porosity", "crack")}
    if any(not isinstance(name, str) or not name.strip() for name in names.values()):
        raise ValueError("synthetic case label names must be nonempty strings")
    if len(set(names.values())) != len(names):
        raise ValueError("synthetic case defect names must remain distinct")
    return names


def _assert_inputs_unchanged(paths, hashes) -> None:
    for key, path in paths.items():
        if sha256_file(path) != hashes[key]:
            raise ValueError(f"synthetic case input changed during construction: {key}")


def _task(index, defect, *, label_names=None):
    count, position = index % 5 + 1, POSITIONS[index % len(POSITIONS)]
    name = (label_names if label_names is not None else _label_names())["porosity"] if defect else None
    return {"kind": "image_report", "source_ids": ["synthetic-annotation"],
            "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.",
            "evidence": {"defect_counts_by_name": {name: count} if defect else {},
                         "defect_positions": [{"name": name, "where": position}] * count if defect else [],
                         "size_mm_available": False, "limitation": LIMITATIONS[0]},
            "locked_target": {"defects": [{"name": name} for _ in range(count)] if defect else [],
                              "clauses": [], "verdict": None}}


def _case(group, index, text, expected, *, deterministic=None, label_names=None):
    return {"id": f"synthetic-{group}-{index:03d}", "group": group,
            "task": _task(index, group.startswith("D"), label_names=label_names), "candidate": text,
            "expected_model_accept": expected,
            "expected_deterministic_accept": expected if deterministic is None else deterministic}


def build_cases(nplus_path) -> dict:
    """읽은 입력의 지문을 결속해 메모리에서 구성한다. 실행 결과로 기대를 바꾸지 않는다."""
    paths = {"independent_nplus_sha256": Path(nplus_path).resolve(),
             "builder_sha256": Path(__file__).resolve(),
             "label_map_sha256": Path(label_map.DEFAULT_LABEL_MAP).resolve(),
             "label_map_loader_sha256": Path(label_map.__file__).resolve()}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    names = _label_names(paths["label_map_sha256"])
    porosity, crack = names["porosity"], names["crack"]
    make_case = partial(_case, label_names=names)
    source = read_json(paths["independent_nplus_sha256"])
    if not isinstance(source, dict):
        raise ValueError("independent N+ file must be an object")
    metadata = source.get("metadata", {})
    if (not isinstance(metadata, dict) or metadata.get("independent_expected_cases") is not True or
            metadata.get("policy_source_not_read") is not True):
        raise ValueError("independent N+ authorship metadata is required; it is not external approval")
    positives = source.get("cases")
    if source.get("group") != "N+" or not isinstance(positives, list) or len(positives) != 50:
        raise ValueError("independent N+ file must contain exactly 50 cases")
    cases = []
    for value in positives:
        if not isinstance(value, dict):
            raise ValueError("N+ case must be an object")
        if value.get("expected_model_accept") is not True or value.get("expected_deterministic_accept") is not True:
            raise ValueError("N+ expectations must not be changed to fit the implemented grammar")
        if not isinstance(value.get("candidate"), str) or not value["candidate"].strip():
            raise ValueError("N+ candidate missing")
        if (value.get("task", {}).get("kind") != "image_report" or
                value["task"].get("locked_target", {}).get("defects") != []):
            raise ValueError("N+ must be a synthetic defect-free image_report")
        cases.append({**value, "group": "N+"})
    for i in range(50):
        count, position = i % 5 + 1, POSITIONS[i % len(POSITIONS)]
        fact = f"{porosity} {count}개가 {position}에 주석으로 표시되어 있다."
        limit = LIMITATIONS[(i // 9) % len(LIMITATIONS)]
        cases.extend((
            make_case("N-invented", i, f"{crack} {i + 1}개가 {position}에 있다.", False),
            make_case("N-number", i, f"표시된 결함은 {i}개다.", False),
            make_case("N-assertion", i,
                  ("표시된 결함이 없으므로 합격이다." if i % 2 == 0 else
                   "표시된 결함이 없다. 결함이 표시되지 않았다는 것은 합격과 안전을 뜻한다."), False),
            make_case("N-leak", i, "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.", False),
            make_case("D+", i, fact + " " + limit, True),
            make_case("D-missing", i, fact, False),
            make_case("D-half", i, fact + (" 두께 정보가 없다." if i % 2 == 0 else
                                      " 치수 기준 합부는 판단하지 않는다."), False),
        ))
        traps = (
            "모재 두께 정보는 없다. 합격 판정은 어렵지 않다.",
            "모재 두께 정보가 없지 않다. 치수 기준 합부는 판단하지 않는다.",
            f'"{limit}" 이 영상은 검사에 통과한다.',
            limit + " 이 영상은 검사에 통과한다.",
            "두께 정보는 없고 합격 판정은 가능하지만 원인은 판단하지 않는다.",
        )
        cases.append(make_case("D-trap", i, fact + " " + traps[i % len(traps)], False))
        # D1~D7만으로 개수/위치/종류/형상의 허위를 알아냈다고 주장하지 않는다.
        wrong = (
            f"{porosity} {count + 1}개가 {position}에 있다.",
            f"{porosity} {count}개가 {POSITIONS[(i + 1) % len(POSITIONS)]}에 있다.",
            f"{crack} {count}개가 {position}에 있다.",
            f"{porosity}은 긴 띠 모양으로 이어져 있으며 가스 때문에 생겼다.",
            f"{porosity} {count}개가 {position}에 있고 길이는 3 mm 이다.",
        )
        cases.append(make_case("D-ungrounded", i, wrong[i % len(wrong)] + " " + limit, False,
                           deterministic=i % len(wrong) != 4))
    # D16 §8-2/§9-2에 명시된 반례도 원문 그대로 넣고 같은 묶음 크기를 유지한다.
    replacements = {
        "synthetic-N-invented-000": "표시된 결함이 없다는 말은 거짓이다.",
        "synthetic-N-assertion-001": "표시된 결함이 없으므로 이 용접부는 안전하다.",
        "synthetic-N-assertion-002": "표시된 결함이 없다. 합격이나 안전을 뜻하지 않는 것은 아니다.",
    }
    for case in cases:
        if case["id"] in replacements:
            case["candidate"] = replacements[case["id"]]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate synthetic case id")
    uniqueness = {group: {"cases": count,
                          "unique_candidate_texts": len({c["candidate"] for c in cases if c["group"] == group})}
                  for group, count in sorted(Counter(c["group"] for c in cases).items())}
    _assert_inputs_unchanged(paths, hashes)
    return {"schema_version": 1, "cases": cases,
            "provenance": {**hashes,
                           "label_names": names,
                           "real_images_or_ids_used": False,
                           "expectations_fixed_before_model_execution": True,
                           "case_reuse": uniqueness,
                           "statistical_generalization_claimed": False},
            "thresholds": {"cases_per_group": 50, "negative_false_accept": 0,
                           "nplus_deterministic_false_reject": 0,
                           "positive_combined_false_reject_maximum": 2,
                           "actual_independent_calls": 2},
            "model_discrimination_measured": False}


def write_cases(nplus_path, output_path, *, repo_root=None) -> dict:
    assert_gate_closed(repo_root)
    result = build_cases(nplus_path)
    write_json_new(output_path, result)
    return {"path": str(output_path), "sha256": sha256_file(output_path), "cases": len(result["cases"])}


V5_PROTOCOL = "judge_v5_g150"
V5_GROUP_COUNTS = {"N+": 50, "D+": 50, "ADV1": 25, "ADV2": 25}
V6_PROTOCOL = "judge_v6_g150"
V6_EVALUATION_SCOPE = {
    "previous_model_protocol": V5_PROTOCOL,
    "prior_model_output_exposed": True,
    "new_independent_discrimination_claimed": False,
    "role": "fixed-source regression and preregistered operational gate",
}
_V5_SOURCE_SETS = {"N+": "judge_nplus_g1", "D+": "judge_dplus_g1",
                   "ADV1": "judge_adv_negative_g1", "ADV2": "judge_adv_negative_g2"}
_V5_ZONES = frozenset(position.replace("-", " ") for position in POSITIONS) - {"가운데 가운데"} | {"가운데"}


def _v5_task(source: dict, case: dict, names: dict[str, str]) -> dict:
    """원본 필드만 옮긴다. 정답·작성 사유·제한 문장의 예시는 모델 입력에 넣지 않는다."""
    root_kind, case_kind = source.get("row_kind"), case.get("row_kind")
    if root_kind is not None and case_kind is not None and root_kind != case_kind:
        raise ValueError("v5 row_kind conflicts with the source")
    kind = case_kind if case_kind is not None else root_kind
    if kind not in ("defect_free", "defect"):
        raise ValueError("v5 explicit row_kind is required")
    fields = case.get("fields")
    if "fields" not in case and kind == "defect_free":
        fields = {"defect_counts_by_name": {}, "defect_positions": {}}
    if not isinstance(fields, dict) or set(fields) != {"defect_counts_by_name", "defect_positions"}:
        raise ValueError("v5 fields must contain counts and per-name positions")
    counts, positions = fields["defect_counts_by_name"], fields["defect_positions"]
    if not isinstance(counts, dict) or not isinstance(positions, dict) or set(counts) != set(positions):
        raise ValueError("v5 count/position names differ")
    if bool(counts) != (kind == "defect"):
        raise ValueError("v5 row_kind and fields disagree")
    facts, locked = [], []
    for name, count in counts.items():
        if name not in names or type(count) is not int or count < 1:
            raise ValueError("v5 unknown label or nonpositive integer count")
        zones = positions[name]
        if not isinstance(zones, list) or len(zones) != count:
            raise ValueError("v5 position count differs from the locked count")
        for zone in zones:
            if not isinstance(zone, str) or zone not in _V5_ZONES:
                raise ValueError("v5 unknown coarse position")
            facts.append({"name": name, "where": zone})
            # 원본에 없는 ISO 코드나 bbox를 합성하지 않는다.
            locked.append({"name": name, "where": zone})
    return {"kind": "image_report", "source_ids": ["synthetic-annotation"],
            "prompt": "제공한 주석의 이름·개수·거친 위치와 한계만 한국어로 설명하라.",
            "evidence": {"row_class": kind, "defect_counts_by_name": deepcopy(counts),
                         "defect_positions": facts, "size_mm_available": False},
            "locked_target": {"defects": locked, "clauses": [], "verdict": None}}


def task_from_v5_source(source: dict, case: dict, *, label_names: dict[str, str]) -> dict:
    """출처 대조용 순수 변환. 호출자는 원파일 SHA와 사상표 지문을 먼저 검증해야 한다."""
    return _v5_task(source, case, label_names)


def build_v5_cases(sources: Mapping, *, expected_source_sha256: Mapping) -> dict:
    """독립 작성된 네 원본을 새 v5 입력으로 변환한다. 500개 구판 생성기는 그대로 둔다.

    호출자가 별도로 기록한 원파일 지문 네 개가 필수다. 원문 문자열과 expected는
    고치지 않으며 source_case_sha256은 원객체의 canonical_bytes SHA256이다.
    원파일의 바이트 표현은 source_file_sha256으로 결속한다. 기대·이유·묶음 이름은
    task에 넣지 않는다. 이 변환이나 결정적 음성 미검출은 모델 승인 증거가 아니다.
    """
    groups = set(V5_GROUP_COUNTS)
    if (not isinstance(sources, Mapping) or set(sources) != groups or
            not isinstance(expected_source_sha256, Mapping) or set(expected_source_sha256) != groups):
        raise ValueError("v5 requires all four sources and externally expected SHA256 pins")
    pins = dict(expected_source_sha256)
    if any(not isinstance(pin, str) or re.fullmatch(r"[0-9a-f]{64}", pin) is None for pin in pins.values()):
        raise ValueError("v5 source pins must be lowercase SHA256 strings")
    source_paths = {group: Path(sources[group]).resolve() for group in V5_GROUP_COUNTS}
    if len(set(source_paths.values())) != len(source_paths):
        raise ValueError("v5 sources must be four distinct files")
    paths = {"adapter_sha256": Path(__file__).resolve(),
             "label_map_sha256": Path(label_map.DEFAULT_LABEL_MAP).resolve(),
             "label_map_loader_sha256": Path(label_map.__file__).resolve()}
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    loaded = label_map.load_label_map(paths["label_map_sha256"])
    names = {value.name_ko: key for key, value in loaded.defect_types.items()}
    if (len(names) != len(loaded.defect_types) or
            any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("v5 label names must be nonempty and unique")
    cases, metadata, source_files, seen_ids = [], {}, {}, set()
    for group, expected_count in V5_GROUP_COUNTS.items():
        raw = source_paths[group].read_bytes()
        if hashlib.sha256(raw).hexdigest() != pins[group]:
            raise ValueError(f"v5 source SHA256 differs from caller pin: {group}")
        source_files[group] = base64.b64encode(raw).decode("ascii")
        source = parse_json(raw)
        if (not isinstance(source, dict) or source.get("set") != _V5_SOURCE_SETS[group] or
                not isinstance(source.get("cases"), list) or len(source["cases"]) != expected_count):
            raise ValueError(f"v5 wrong source set or case count: {group}")
        metadata[group] = deepcopy({key: value for key, value in source.items() if key != "cases"})
        positive = group in ("N+", "D+")
        for index, original in enumerate(source["cases"]):
            if not isinstance(original, dict):
                raise ValueError("v5 case must be an object")
            cid, text = original.get("id"), original.get("text")
            if not isinstance(cid, str) or not cid.strip() or cid in seen_ids:
                raise ValueError("v5 empty or duplicate case id")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("v5 original text must be nonempty")
            if original.get("expected") != ("accept" if positive else "reject"):
                raise ValueError("v5 original expectation differs from its group; do not relabel")
            task = task_from_v5_source(source, original, label_names=names)
            expected_kind = {"N+": "defect_free", "D+": "defect", "ADV1": "defect_free"}.get(group)
            if expected_kind is not None and task["evidence"]["row_class"] != expected_kind:
                raise ValueError("v5 group and original row_kind disagree")
            seen_ids.add(cid)
            cases.append({"id": cid, "group": group, "task": task, "candidate": text,
                          "expected_model_accept": positive,
                          "expected_deterministic_accept": True if positive else None,
                          "source_case": deepcopy(original),
                          "source_case_sha256": hashlib.sha256(canonical_bytes(original)).hexdigest(),
                          "source_file_sha256": pins[group], "source_group": group, "source_index": index})
    _assert_inputs_unchanged(source_paths, pins)
    _assert_inputs_unchanged(paths, hashes)
    return {"schema_version": 2, "protocol": V5_PROTOCOL, "cases": cases,
            "provenance": {"source_sha256": pins, **hashes, "label_names": names,
                           "source_files_base64": source_files,
                           "source_metadata": metadata, "real_images_or_ids_used": False,
                           "expectations_fixed_before_model_execution": True,
                           "statistical_generalization_claimed": False,
                           "deterministic_results_are_not_model_approval": True},
            "model_discrimination_measured": False}


def write_v5_cases(sources: Mapping, output_path, *, expected_source_sha256: Mapping, repo_root=None) -> dict:
    """원본과 다른 새 경로에만 쓴다. 입력/게이트 재검사가 끝나기 전에는 출력하지 않는다."""
    assert_gate_closed(repo_root)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise ValueError("v5 output must be a new path")
    result = build_v5_cases(sources, expected_source_sha256=expected_source_sha256)
    assert_gate_closed(repo_root)
    write_json_new(output, result)
    return {"path": str(output), "sha256": sha256_file(output), "cases": len(result["cases"]),
            "protocol": V5_PROTOCOL}


def build_v6_cases(sources: Mapping, *, expected_source_sha256: Mapping) -> dict:
    """네 G 원본은 그대로 재검증하되 v5 실행/승인과 섞이지 않는 새 입력을 만든다.

    후보·기대·원본 SHA·task의 근거는 바꾸지 않는다. 위치 표의 정규화는 정책
    렌더러의 책임이다. 이미 읽은 v5 모델 실패로 고친 정책에 대해 이 150건을
    새로운 독립 표본이라고 주장하지 않는다. 저장된 v5 탐침은 입력으로 받지 않는다.
    """
    result = build_v5_cases(sources, expected_source_sha256=expected_source_sha256)
    result["schema_version"] = 3
    result["protocol"] = V6_PROTOCOL
    result["provenance"]["evaluation_scope"] = deepcopy(V6_EVALUATION_SCOPE)
    for case in result["cases"]:
        case["protocol"] = V6_PROTOCOL
    return result


def write_v6_cases(sources: Mapping, output_path, *, expected_source_sha256: Mapping, repo_root=None) -> dict:
    """새 경로에만 v6 입력을 쓴다. v5 사례·모델 원문·실패 영수증은 덮지 않는다."""
    assert_gate_closed(repo_root)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise ValueError("v6 output must be a new path")
    result = build_v6_cases(sources, expected_source_sha256=expected_source_sha256)
    assert_gate_closed(repo_root)
    write_json_new(output, result)
    return {"path": str(output), "sha256": sha256_file(output), "cases": len(result["cases"]),
            "protocol": V6_PROTOCOL}
