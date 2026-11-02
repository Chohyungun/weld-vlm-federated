"""1차 평가 설정에서 2차 후보를 만든다. 등록 발행·기본 경로 전환·영수증 발급은 하지 않는다.

작성(`build_candidate`)·대조(`candidate_problems`)는 `evaluation.eval_registration` 의 순수 함수(`build_regime2_eval`·
`regime2_eval_problems`)를 감싼다 — 정식 등록의 네 소비자가 같은 함수로 같은 여섯 칸·보조 이름을 대조한다(판 6 연결 2 판 §2-3).
앞 판 후보(`CANDIDATE_REL_V1`)는 바이트 그대로 보존한다. 설정 경로를 고르는 명령줄 인자는 없다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REL = "configs/eval_registration.yaml"
CANDIDATE_REL_V1 = "configs/registration/candidates/main-20261027-1.eval.yaml"
CANDIDATE_REL = "configs/registration/candidates/main-20261028-1.eval.yaml"
HEADER = (
    "# 2차 평가 설정 후보 — 미발행. scripts/prepare_eval_regime2.py로 생성한다.\n"
    "# 1차 결정값은 원본에서 계승한다. decided는 계승한 결정일이며 2차 승인일이 아니다.\n"
    "# YAML 설정의 version=1과 정식 등록 JSON의 판 6은 서로 다른 스키마다.\n"
    "# 이 파일은 정식 등록 JSON·영수증을 대신하지 않으며 기본 채점 경로에서 자동 소비하지 않는다.\n"
    "# 정식 발행 때 같은 바이트를 등록 줄기의 .eval.yaml 로 복사한다. 보조 이름 다섯은 기존 이름 뒤에 붙인다.\n"
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def changed_fields(old: dict, new: dict, prefix: str = "") -> list[str]:
    from evaluation.eval_registration import changed_fields as impl

    return impl(old, new, prefix)


def build_candidate(source: dict, base: dict, train: dict, prompt_raw: bytes) -> dict:
    from evaluation.eval_registration import build_regime2_eval

    return build_regime2_eval(source, base_doc=base, train_doc=train, prompt_raw=prompt_raw)


def candidate_problems(candidate: dict, *, source: dict, base: dict, train: dict,
                       prompt_raw: bytes) -> list[str]:
    """기존 설정 판독기가 보지 않는 후보의 체제·페어·보조 이름 칸까지 대조한다."""
    from evaluation.eval_registration import regime2_eval_problems

    return regime2_eval_problems(candidate, source_doc=source, base_doc=base, train_doc=train, prompt_raw=prompt_raw)


def candidate_bytes(candidate: dict) -> bytes:
    return (HEADER + yaml.safe_dump(candidate, allow_unicode=True, sort_keys=False, width=120)).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="새 후보 파일을 쓴다. 기존 파일 덮어쓰기 금지")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from evaluation import eval_registration as ER

    source_raw = (ROOT / SOURCE_REL).read_bytes()
    base_raw = (ROOT / "configs/base.yaml").read_bytes()
    train_raw = (ROOT / "configs/train_registration.yaml").read_bytes()
    source, base = ER.load_eval_registration(source_raw), ER.load_base(base_raw)
    train = yaml.safe_load(train_raw)
    prompt_path = base["fixed_before_main_runs"]["uni_prompt_path"]
    prompt_raw = (ROOT / prompt_path).read_bytes()
    try:
        candidate = build_candidate(source, base, train, prompt_raw)
        expected = candidate_bytes(candidate)
        output = ROOT / CANDIDATE_REL
        if args.write:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(expected)
        if output.read_bytes() != expected:
            raise ValueError("후보 바이트가 작성기의 재생성 결과와 다르다")
    except (OSError, ValueError) as exc:
        print(f"후보 거부: {exc}")
        return 1
    print(json.dumps({"candidate": CANDIDATE_REL, "sha256": _sha(expected), "issued": False,
                      "changes": changed_fields(source, candidate),
                      "source_sha256": {SOURCE_REL: _sha(source_raw), "configs/base.yaml": _sha(base_raw),
                                        "configs/train_registration.yaml": _sha(train_raw), prompt_path: _sha(prompt_raw)}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
