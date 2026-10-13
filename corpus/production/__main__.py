"""Explicit single-stage CLI. Importing or asking for help cannot load a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import read_json


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="로컬 오픈 가중치 코퍼스 생성·검증·학습")
    commands = parser.add_subparsers(dest="stage", required=True)
    prepare = commands.add_parser("prepare", help="봉인 입력 검사와 작업 명부 작성 (CPU)")
    prepare.add_argument("--pairs", required=True, type=Path)
    prepare.add_argument("--sources", required=True, type=Path)
    prepare.add_argument("--config", required=True, type=Path)
    prepare.add_argument("--run", required=True, type=Path)
    prepare.add_argument("--source-text", type=Path, default=None)
    prepare.add_argument("--source-registry", type=Path, default=None)
    prepare.add_argument("--source-rights", type=Path, default=None)
    prepare.add_argument("--source-review", type=Path, default=None)
    regenerate = commands.add_parser("prepare-regeneration", help="부모 명부를 보존한 새 생성 신원 작성 (CPU)")
    regenerate.add_argument("--parent", required=True, type=Path)
    regenerate.add_argument("--run", required=True, type=Path)
    regenerate.add_argument("--config", required=True, type=Path)
    for name in ("generate", "judge"):
        command = commands.add_parser(name, help="로컬 캐시 모델 실행 (GPU; 자동 서버 시작 없음)")
        command.add_argument("--run", required=True, type=Path)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--max-shards", type=int, default=None)
    separate = commands.add_parser("judge-separate", help="명시한 별도 판정 경로 실행; 옛 판정 보충 없음")
    separate.add_argument("--run", required=True, type=Path)
    separate.add_argument("--config", required=True, type=Path)
    separate.add_argument("--judge-dir", required=True, type=Path)
    separate.add_argument("--judge-config", type=Path, default=None, help="별도 모델 spec JSON; RUN config를 바꾸지 않음")
    separate.add_argument("--max-shards", type=int, default=None)
    separate.add_argument("--probe-approval", type=Path, default=None, help="봉인된 두 탐침 승인 폴더")
    separate.add_argument("--probe-approval-sha256", default=None, help="별도로 기록한 APPROVAL.json SHA256")
    cases = commands.add_parser("judge-cases", help="독립 N+ 기대를 포함한 합성 입력 고정 (CPU)")
    cases.add_argument("--nplus", required=True, type=Path)
    cases.add_argument("--output", required=True, type=Path)
    for version in ("v5", "v6"):
        fixed_cases = commands.add_parser("judge-cases-" + version,
                                         help=f"고정된 G 네 원파일을 보존한 {version} 150개 입력 작성 (CPU)")
        for name in ("nplus", "dplus", "adv1", "adv2"):
            fixed_cases.add_argument("--" + name, required=True, type=Path)
            fixed_cases.add_argument("--" + name + "-sha256", required=True,
                                     help="별도로 고정한 해당 원파일 SHA256")
        fixed_cases.add_argument("--output", required=True, type=Path)
    rules = commands.add_parser("judge-rules-check", help="합성 입력의 규칙 계수만 확인; 모델 검증 아님 (CPU)")
    rules.add_argument("--cases", required=True, type=Path)
    probe = commands.add_parser("judge-probe", help="고정 합성 입력에 실제 판정 모델 1회 호출; 재시도 없음 (GPU)")
    probe.add_argument("--cases", required=True, type=Path)
    probe.add_argument("--config", required=True, type=Path)
    probe.add_argument("--output", required=True, type=Path)
    compare = commands.add_parser("compare-judge-probes", help="서로 다른 두 호출의 원문·신원·판별력 대조 (CPU)")
    compare.add_argument("--first", required=True, type=Path)
    compare.add_argument("--second", required=True, type=Path)
    seal = commands.add_parser("seal-judge-probe-approval", help="원 탐침 둘을 재검증하고 이동 가능한 승인 증빙 봉인 (CPU)")
    seal.add_argument("--first", required=True, type=Path)
    seal.add_argument("--second", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit", help="전체 출력 품질 검사와 사람 검수 자료 생성")
    audit.add_argument("--run", required=True, type=Path)
    audit.add_argument("--config", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    audit.add_argument("--judge-dir", type=Path, default=None, help="선택한 새 판정만 읽음; 누락은 미완료")
    audit.add_argument("--expected-judge-identity-sha256", default=None, help="별도로 기록한 JUDGE 신원 SHA256")
    audit.add_argument("--expected-probe-approval-sha256", default=None, help="별도로 기록한 탐침 승인 SHA256")
    release = commands.add_parser("release", help="검수 증빙을 확인하고 학습 릴리스 발행")
    release.add_argument("--audit", required=True, type=Path)
    release.add_argument("--review", required=True, type=Path)
    release.add_argument("--rights", required=True, type=Path)
    release.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify-release", help="릴리스 증빙 재검사 (CPU)")
    verify.add_argument("--release", required=True, type=Path)
    train = commands.add_parser("train", help="검수 릴리스로 SFT; --dry-run은 모델 적재 없음")
    train.add_argument("--release", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--scope", choices=("central", "C1", "C2", "C3", "public_text"), required=True)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = read_json(args.config) if hasattr(args, "config") else None
        if args.stage == "prepare":
            from .prepare import prepare_run
            result = prepare_run(args.pairs, args.sources, config, args.run,
                                 source_text_path=args.source_text, source_registry_path=args.source_registry,
                                 source_rights_path=args.source_rights, source_review_path=args.source_review)
        elif args.stage == "prepare-regeneration":
            from .generation_policy import prepare_regeneration
            result = prepare_regeneration(args.parent, args.run, config)
        elif args.stage == "judge-separate":
            from .infer import run_judge
            judge_config = read_json(args.judge_config) if args.judge_config else None
            result = run_judge(args.run, config, args.judge_dir,
                               judge_config=judge_config, max_shards=args.max_shards,
                               probe_approval=args.probe_approval,
                               probe_approval_sha256=args.probe_approval_sha256)
        elif args.stage == "judge-cases":
            from .judge_cases import write_cases
            result = write_cases(args.nplus, args.output)
        elif args.stage in ("judge-cases-v5", "judge-cases-v6"):
            from .judge_cases import write_v5_cases, write_v6_cases
            write_fixed = write_v6_cases if args.stage == "judge-cases-v6" else write_v5_cases
            groups = {"N+": "nplus", "D+": "dplus", "ADV1": "adv1", "ADV2": "adv2"}
            result = write_fixed({g: getattr(args, name) for g, name in groups.items()}, args.output,
                                    expected_source_sha256={g: getattr(args, name + "_sha256")
                                                            for g, name in groups.items()})
        elif args.stage == "judge-rules-check":
            from .judge_probe import check_rules
            result = check_rules(args.cases)
        elif args.stage == "judge-probe":
            from .judge_probe import run_probe
            result = run_probe(args.cases, config, args.output)
        elif args.stage == "compare-judge-probes":
            from .judge_probe import compare_probes
            result = compare_probes(args.first, args.second)
        elif args.stage == "seal-judge-probe-approval":
            from .judge_probe import seal_probe_approval
            result = seal_probe_approval(args.first, args.second, args.output)
        elif args.stage in ("generate", "judge"):
            from .infer import run_stage
            result = run_stage(args.run, args.stage, config, max_shards=args.max_shards)
        elif args.stage == "audit":
            from .quality import audit_run
            if args.judge_dir is None:
                if args.expected_judge_identity_sha256 is not None or args.expected_probe_approval_sha256 is not None:
                    raise ValueError("external judge pins require an explicit --judge-dir; no legacy fallback")
                result = audit_run(args.run, config, args.output)
            else:
                from inspect import Parameter, signature
                from .infer import read_judge_source
                parameters = signature(audit_run).parameters
                if ("judge_source" not in parameters and
                        not any(p.kind == Parameter.VAR_KEYWORD for p in parameters.values())):
                    raise ValueError("separate judge audit consumer is not integrated; no legacy fallback")
                result = audit_run(args.run, config, args.output,
                                   judge_source=read_judge_source(args.run, config, args.judge_dir,
                                       expected_judge_identity_sha256=args.expected_judge_identity_sha256,
                                       expected_probe_approval_sha256=args.expected_probe_approval_sha256))
        elif args.stage == "release":
            from .release import release_run
            result = release_run(args.audit, args.review, args.rights, args.output)
        elif args.stage == "verify-release":
            from .common import validate_release
            result = validate_release(args.release)
        else:
            from .train import train_from_release
            result = train_from_release(args.release, config, args.scope, args.output,
                                        resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return int(result.get("exit_code", 0)) if isinstance(result, dict) else 0
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        print(f"{args.stage} failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
