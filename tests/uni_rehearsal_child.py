"""리허설 합성 시험의 자식 진입 — 세계(`tests/uni_rehearsal_world.py`)에 다시 붙어 실제 진입점의 `main()` 을 대역 이음새와 함께 부른다.

    python -X utf8 tests/uni_rehearsal_child.py <세계 폴더> <진입> <인자...>

진입은 `lists` · `register` · `train`(오케스트레이터의 하위 명령) · `export`(`scripts/export_uni.py`) · `score`(평가 쪽 본채점 진입점, 그 시험의 이음새로).
고의 중단 변수는 오케스트레이터가 넣은 환경에서 자식이 그대로 읽는다. CUDA 를 쓰지 않는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from tests import uni_rehearsal_world as RW

    tmp, entry, argv = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    w = RW.attach(tmp)
    import vlm.rehearsal_lists as RL

    RL.DIAG_LISTS_ON_HOLD = False        # 시험 세계의 자식만 — 진단 경로 시험은 보류가 풀린 뒤의 길을 본다(리허설 경로는 상관없다)
    if entry in ("lists", "register", "train"):
        from scripts import rehearsal_uni

        return rehearsal_uni.main([entry, *argv], measure_env=RW.measure, model_loader=RW.fake_model_loader,
                                  generator_loader=RW.generator_loader, checkout=w.repo,
                                  exposure_ledger=w.exposure, non_main_parent=w.parent, config_path=w.config,
                                  standin_allowed=True, fed_runner=RW.fed_runner(w.parent),
                                  flwr_procs=list, flwr_stop=lambda found: None, token_counter=RW.token_counter)
    if entry == "export":
        from scripts import export_uni

        return export_uni.main(argv, measure_env=RW.measure, load_generator=RW.generator_loader,
                               repo=w.repo, code_repo=w.code, exposure_ledger=w.exposure, non_main_parent=w.parent,
                               standin_allowed=True)
    if entry == "score_echo_noop":                       # 에코 전용 호출이 아무것도 쓰지 않고 끝난 꼴(있던 산출을 받은 재개처럼)
        if "--echo-only" in argv:
            return 0
        entry = "score"
    if entry == "score_echo_pass":
        return RW.fake_echo_gate(w, argv, status="pass", reasons=[])
    if entry == "score_echo_mismatch":
        return RW.fake_echo_gate(w, argv, status="fail", reasons=["echo_mismatch"])
    if entry == "score_echo_other_list":
        return RW.fake_echo_gate(w, argv, status="pass", reasons=[], other_echo_list=True)
    if entry == "score_echo_other_registration":
        return RW.fake_echo_gate(w, argv, status="pass", reasons=[], generation_sha256="0" * 64)
    if entry == "score_verdict_then_crash":
        return RW.fake_d_verdict_then_crash(w, argv)
    if entry == "score":
        import scripts.probe.score_unified as SU

        return SU.main(argv, _seam=SU.Seam(checkout=w.repo, repo=w.repo))
    print(f"모르는 진입: {entry}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
