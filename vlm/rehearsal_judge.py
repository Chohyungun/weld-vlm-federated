"""리허설의 판정기 — 통과 조건 다섯(리허설 2판 §3)을 실행 기록과 산출물에서 읽는다. 결과는 `<루트>/judge.json` 한 파일이다.

**지표 값을 싣지 않는다**(판정 03). ①②④ 는 값이 아니라 자리 · 수 · 같음을 보고, ③⑤ 는 평가 쪽 산출물의 상태를 옮긴다. 입력 파일마다 sha256 을 싣는다.
먼저 묶음 곁 파일과 어댑터 meta 의 `impl_ids` 를 본다 — **승인된 실제 구현이 아니면 조건마다 관측만 싣고 판정 상태를 `stand_in`(대역 실행)으로 적는다.**
대역으로 돈 리허설이 배관을 검증했다고 적히는 길을 막는다(2판 §1-4).

| 조건 | 읽는 것 | 관측 |
|---|---|---|
| ① 재개 · 구판 | 줄 A 의 어댑터 meta `resume_segments` · 고의 중단의 활성화 기록 · 2′ 넷의 실행 기록 · stdout · stderr · 원장 · 합성 체크포인트 | (가) 세그먼트 둘 이상 · 신원 해시가 모두 같다 (나) 둘째 세그먼트의 시작 epoch = 중단 epoch + 1 (다) 넷 다 비영 · 75 아님, 정책 셋만 비운 경우 stderr 의 불일치 필드 집합 = 정책 셋 (라) 원장에 데이터 행이 없다 · 적용 줄이 없다 · 합성 체크포인트 바이트가 그대로다 |
| ② 좌표 | 에코 관문과 본채점의 `canaries` · 어댑터 meta 와 곁 파일 | (가) 에코 · 문자 일치 카나리아 상태 (다) 칸마다 학습 관측 격자 = export 관측 격자(키 집합), 프로세서 설정 해시 · 좌표 해시가 같다. (나) 축 배율은 **결과가 계약의 꼴로 다섯 모델 묶음 모두에 있는가**만 본다(`axis_ratio_problems`) — 상태로 판정하지 않는다 |
| ③ 스키마 | 평가 쪽 본채점 산출물 | 거부 묶음 수 · 검증 묶음 · 관문 ③ 의 상태 |
| ④ 덮지 않음 | 실행 기록의 2″ · 6′ 줄과 앞뒤 해시 목록 · 끝 줄(루트 밖 목록 · 두 원장의 접두) · 스냅샷 · 페어 재검증 | 같음 · 종료 코드 |
| ⑤ 지연 · 길이 | 평가 쪽 관문 ⑤ · 모델 묶음의 줄 | 관문 상태 · 줄마다 `latency_ms` · `n_new_tokens` · `gen_stop` 이 있다 · 한도 도달 비율이 5 % 를 넘는가(경보 — 불통과가 아니다) |

**보고만 하는 것** — 찢어진 꼬리를 다시 만든 줄의 앞부분이 잘라 낸 바이트와 같은가(2판 §3 의 끝), 켜졌고 발화 기록이 없는 중단.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["AXIS_COUNTS", "AXIS_STATUSES", "JUDGE_FILE", "JudgeRefused", "axis_observation", "axis_ratio_problems",
           "judge"]

JUDGE_FILE = "judge.json"
POLICY_FIELDS = frozenset({"shuffle_policy", "template_mode", "gen_prefix_digest"})
APPLY_LINE = "까지의 상태를 적용한다"
_MISMATCH = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*): 체크포인트 ")


#: 생성 한도에 닿아 끝난 줄의 값 — 계약의 어휘에서 가져온다(`evaluation.schema_v14.GenStop`, 외부 검토 9).
def _limit_stop() -> str:
    from typing import get_args

    from evaluation.schema_v14 import GenStop

    vocab = get_args(GenStop)
    if "length" not in vocab:
        raise RuntimeError(f"계약의 멈춤 어휘에 'length' 가 없다: {vocab}")
    return "length"


_HEX64 = re.compile(r"[0-9a-f]{64}")

#: 축 배율 카나리아의 소비 계약(외부 검토 §39 · §40 · 후속 확인 §2). 평가 쪽 `evaluation.axis_ratio_canary` 가 모델 묶음마다 내는 꼴이다 —
#: `{"bundle", "status", "reason", "n_defect_images", "n_pairs", "n_pair_groups", "n_pairs_unusable"}` 의 목록.
#: 상태는 평가 쪽 프레임 진단의 상수(`evaluation.frame_diag` 의 `PASS` · `FAIL` · `INDETERMINATE`)와 **같은 문자열**이다 — 변환하지 않는다.
#: 그 모듈이 있으면 시험이 둘을 맞댄다. 표본 수 넷은 모두 필수이고 bool 이 아닌 0 이상 정수다(0 과 누락을 가른다).
AXIS_STATUSES = ("통과", "불통과", "판정 불가")
AXIS_COUNTS = ("n_defect_images", "n_pairs", "n_pair_groups", "n_pairs_unusable")


def axis_ratio_problems(ax: Any, expected: set[str]) -> list[str]:
    """`canaries.axis_ratio` 가 계약을 지켰는가 — 어긋난 자리의 목록(비면 지켰다).

    묶음 집합 = 그 시드의 기대 모델 묶음 집합(중복 없음) · 상태는 세 어휘 가운데 하나 · 표본 수 넷. **통계적 상태는 보지 않는다** —
    올바른 표본 수를 가진 '판정 불가' 도 호출이다(2판 §3 ② 나). 집계 객체 하나는 받지 않는다 — 어느 묶음의 결과인지 대조할 수 없다."""
    if not isinstance(ax, list):
        return ["블록이 없거나 묶음별 목록이 아니다"]
    out: list[str] = []
    stems: list[str] = []
    for k, e in enumerate(ax):
        if not isinstance(e, Mapping) or not isinstance(e.get("bundle"), str):
            out.append(f"{k} 번째 항목에 묶음 이름이 없다")
            continue
        b = e["bundle"]
        stems.append(b)
        if e.get("status") not in AXIS_STATUSES:
            out.append(f"{b}: 상태가 {list(AXIS_STATUSES)} 가 아니다 — {e.get('status')!r}")
        for c in AXIS_COUNTS:
            if c not in e:
                out.append(f"{b}: {c} 가 없다")
            elif type(e[c]) is not int or e[c] < 0:
                out.append(f"{b}: {c} 가 0 이상 정수가 아니다 — {e[c]!r}")
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        out.append(f"같은 묶음이 둘 이상이다: {dup}")
    if set(stems) != set(expected):
        out.append(f"묶음 집합이 기대와 다르다 — 빠짐 {sorted(set(expected) - set(stems))} · 남음 {sorted(set(stems) - set(expected))}")
    return out


def echo_identity_ok(e: Mapping[str, Any]) -> bool:
    """에코 한 묶음이 항등 검사를 지났는가 — 상태 통과 · **기대 박스 수 > 0** · TP = 기대 수 · FP = FN = 0 · 좌표 최대 차 0.0
    (2판 §3 ② 가 · 07번 §14-2 의 8). 수는 싣지 않고 이 불리언만 적는다(외부 검토 6)."""
    try:
        n = int(e["n_expected"])
        return (e.get("status") == "pass" and n > 0 and int(e["tp"]) == n and int(e["fp"]) == 0
                and int(e["fn"]) == 0 and float(e["max_coord_diff"]) == 0.0)
    except (KeyError, TypeError, ValueError):
        return False


def axis_observation(canaries: Mapping[str, Any], model_stems: set[str]) -> dict[str, Any]:
    """평가 쪽 본채점 산출물의 `canaries` → 판정 파일에 싣는 축 배율 관측. `judge` 가 이 함수 하나로 읽는다.
    `called` 는 계약을 지킨 결과가 기대 묶음 모두에 있는가다 — 상태로 판정하지 않는다."""
    ax = canaries.get("axis_ratio")
    problems = axis_ratio_problems(ax, model_stems)
    return {"called": not problems,
            "entries": [{k: e.get(k) for k in ("bundle", "status", "reason", *AXIS_COUNTS)}
                        for e in (ax if isinstance(ax, list) else []) if isinstance(e, Mapping)],
            "problems": problems}


class JudgeRefused(ValueError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"[{code}] {msg}")


class _Inputs:
    """읽은 파일의 sha256 을 모은다 — 판정기가 무엇을 보고 판정했는지."""

    def __init__(self, root: Path):
        self.root, self.seen = Path(root), {}

    def raw(self, p: Path) -> bytes | None:
        p = Path(p)
        if not p.is_file():
            return None
        b = p.read_bytes()
        try:
            key = p.relative_to(self.root).as_posix()
        except ValueError:
            key = p.as_posix()
        self.seen[key] = hashlib.sha256(b).hexdigest()
        return b

    def json(self, p: Path) -> Any:
        b = self.raw(p)
        return None if b is None else json.loads(b)

    def text(self, p: Path) -> str:
        b = self.raw(p)
        return "" if b is None else b.decode("utf-8", errors="replace")


def _all(xs) -> bool:
    xs = list(xs)
    return bool(xs) and all(xs)


def _status(checks: Mapping[str, bool], stand_in: bool) -> str:
    if stand_in:
        return "stand_in"
    return "pass" if all(checks.values()) else "fail"


def judge(root: Path, *, checkout: Path | None = None, exposure_ledger: Path | None = None) -> dict[str, Any]:
    """판정을 내고 `judge.json` 을 배타 생성한다. 돌려주는 값은 그 내용이다."""
    from data.manifest_io import verify_snapshot
    from evaluation.actuals import impl_problems
    from vlm.rehearsal_plan import load_plan
    from vlm.uni_config import load_uni_config, pairs_snapshot_problems

    root = Path(root)
    out_p = root / JUDGE_FILE
    if out_p.exists():
        raise JudgeRefused("judge_exists", f"판정 파일이 이미 있다 — 덮지 않는다: {out_p}")
    I = _Inputs(root)
    log_raw = I.raw(root / "rehearsal_log.jsonl")
    if log_raw is None:
        raise JudgeRefused("log_missing", "실행 기록이 없다")
    rows = [json.loads(x) for x in log_raw.decode("utf-8").splitlines() if x]
    stages = [r for r in rows if "seq" in r]
    by_name = {r["name"]: r for r in stages}
    kinds = {}
    for r in rows:
        if "kind" in r:
            kinds.setdefault(r["kind"], []).append(r)
    plan = load_plan(root / "plan.yaml", purpose="rehearsal")
    I.raw(root / "plan.yaml")
    idx = int(plan.get("seed_index"))
    from vlm.rehearsal_orch import REQUIRED_TAGS, expected_stages

    cells = list(REQUIRED_TAGS)                       # 필수 칸 — 계획의 cells 와 무관하다(외부 검토 3)
    export = root / "export"

    # ---------------------------------------------------------------- 구현 식별자
    metas = {t: I.json(root / "train" / f"{t}_s{idx}" / "adapter_last.meta.json") for t in cells}
    sidecars = {t: I.json(export / f"{t}_s{idx}.export_meta.json") for t in cells}
    echo_side = I.json(export / f"uni_central_s{idx}.echo.export_meta.json")
    impl_bad: dict[str, list[str]] = {}
    missing: list[str] = []
    for name, obj in [*(("meta:" + t, m) for t, m in metas.items()), *(("export:" + t, s) for t, s in sidecars.items()),
                      ("export:uni_central.echo", echo_side)]:
        if obj is None:
            missing.append(name)                      # 필수 산출물이 없다 — 대역 실행이 아니라 불통과다
            continue
        probs = impl_problems(obj.get("impl_ids"))
        if probs:
            impl_bad[name] = probs
    stand_in = bool(impl_bad)

    # ---------------------------------------------------------------- ①
    a_meta = metas.get("uni_central") or {}
    segs = a_meta.get("resume_segments") or []
    t_fault = (plan.get("faults") or {}).get("train", {}).get("uni_central")
    fault_row = next((r for r in stages if r.get("fault") == t_fault), None) if t_fault else None
    fault_ep = None
    if fault_row and fault_row.get("activated"):
        act = I.json(root / "faults" / fault_row["activated"][0]["name"]) or {}
        m = re.fullmatch(r"ep=(\d+)", str(act.get("arg", "")))
        fault_ep = int(m.group(1)) if m else None
    made = (kinds.get("negatives_made") or [{}])[0].get("cases", {})
    neg: dict[str, dict] = {}
    for case in plan.get("negatives") or []:
        r = by_name.get(f"negative_{case}")
        if r is None:
            neg[case] = {"ran": False}
            continue
        err, out = I.text(root / r["stderr_path"]), I.text(root / r["stdout_path"])
        led = root / "negative" / case / "train" / f"uni_central_s{idx}" / "train_ledger.csv"
        led_rows = [x for x in I.text(led).splitlines()[1:] if x.strip()] if led.exists() else []
        info = made.get(case) or {}
        ck = root / info["path"] if info.get("path") else None
        fields = {m.group(1) for ln in err.splitlines() if (m := _MISMATCH.match(ln))}
        neg[case] = {"ran": True, "exit_code": r["exit_code"], "refused": r["exit_code"] not in (0, 75),
                     "apply_line": APPLY_LINE in out or APPLY_LINE in err, "ledger_data_rows": len(led_rows),
                     "checkpoint_unchanged": bool(ck and ck.is_file()
                                                  and hashlib.sha256(I.raw(ck)).hexdigest() == info.get("sha256")),
                     "identity_mismatch_reported": "신원이 현재 실행과 다르다" in err,
                     "mismatch_fields": sorted(fields)}
    planned_rows = [r for r in stages if r.get("fault")]
    stages_missing = [n for n in expected_stages(plan) if n not in by_name]
    c1 = {
        "stage_table_ok": bool(stages) and all(r.get("ok") for r in stages) and not stages_missing,
        "planned_faults_all_intended": bool(planned_rows) and all(r.get("intended_fault") for r in planned_rows),
        "segments_ge_2": len(segs) >= 2,
        # 세그먼트마다 유효한 신원 해시가 있어야 같음을 본다 — None 은 검증된 재개 사슬 밖이다(외부 검토 5)
        "segments_same_identity": bool(segs) and all(isinstance(s.get("identity_sha256"), str)
                                                     and _HEX64.fullmatch(s["identity_sha256"]) for s in segs)
        and len({s["identity_sha256"] for s in segs}) == 1,
        "second_segment_after_fault": (len(segs) >= 2 and fault_ep is not None
                                       and segs[1].get("start_epoch") == fault_ep + 1),
        "negatives_all_refused": _all(n.get("refused") for n in neg.values()),
        "policy_blank_fields_are_policy": set(neg.get("policy_blank", {}).get("mismatch_fields", [])) == POLICY_FIELDS
        and neg.get("policy_blank", {}).get("identity_mismatch_reported", False),
        "negatives_no_data_rows": _all(n.get("ledger_data_rows") == 0 for n in neg.values()),
        "negatives_no_apply_line": _all(n.get("ran") and not n.get("apply_line") for n in neg.values()),
        "negatives_checkpoint_unchanged": _all(n.get("checkpoint_unchanged") for n in neg.values()),
    }

    # ---------------------------------------------------------------- ② · ③ · ⑤ (평가 쪽 산출물)
    gate = I.json(root / "echo_gate" / "score" / "score_unified_rehearsal.json") or {}
    score = I.json(root / "score" / "score_unified_rehearsal.json") or {}
    gates = I.json(root / "score" / "rehearsal_gates.json") or {}
    can = score.get("canaries") or {}
    d_stand_in = score.get("status") == "stand_in" or bool((score.get("bundles") or {}).get("stand_in"))
    stand_in = stand_in or d_stand_in
    model_stems = {f"{t}_s{idx}" for t in cells}
    lit = [e for e in can.get("literal", []) if e.get("bundle") in model_stems]
    env_rows = {}
    for t in cells:
        m, s = metas.get(t) or {}, sidecars.get(t) or {}
        env_rows[t] = {
            "grid_keys_same": bool(m.get("train_input_grid_observed")) and
            set(m.get("train_input_grid_observed") or {}) == set(s.get("image_grid_thw_observed") or {}),
            "processor_same": m.get("processor_config_sha256") is not None
            and m.get("processor_config_sha256") == s.get("processor_config_sha256"),
            "coord_same": m.get("coord_cfg_hash") is not None and m.get("coord_cfg_hash") == s.get("coord_cfg_hash"),
        }
    c2 = {
        "echo_gate_identity": _all(echo_identity_ok(e) for e in gate.get("canaries", {}).get("echo", [])),
        "echo_identity": _all(echo_identity_ok(e) for e in can.get("echo", [])),
        "literal_pass_all_models": len(lit) == len(model_stems) and _all(e.get("status") == "pass" for e in lit),
        "env_same_all_cells": _all(all(v.values()) for v in env_rows.values()),
    }
    verified = set((score.get("bundles") or {}).get("verified") or [])
    # 축 배율 — 평가 쪽 본채점 산출물의 `canaries.axis_ratio`. **호출의 누락 · 계약에 어긋난 결과와 통계적 판정 불가를 가른다**:
    # 다섯 모델 묶음 모두의 결과가 계약의 꼴(`axis_ratio_problems`)로 있어야 호출된 것이고, 그 상태가 '판정 불가' 여도 경로가 돈 것이다 —
    # 상태와 표본 수만 싣고 판정하지 않는다(2판 §3 ② 나 · 외부 검토 §39 · §40).
    axis = axis_observation(can, model_stems)
    c2["axis_ratio_called"] = axis["called"]
    rejected = (score.get("bundles") or {}).get("rejected") or {}
    c3 = {
        "no_rejected_bundle": score != {} and not rejected,
        "all_bundles_verified": model_stems | {f"uni_central_s{idx}.echo"} <= verified,
        "gate3_pass": (gates.get("gates") or {}).get("③", {}).get("status") == "pass",
    }
    missing_keys = 0
    limit_hits = n_lines = 0
    for t in cells:
        raw = I.raw(export / f"{t}_s{idx}.generations.jsonl")
        for ln in (raw or b"").decode("utf-8").splitlines():
            row = json.loads(ln)
            n_lines += 1
            missing_keys += any(row.get(k) is None for k in ("latency_ms", "n_new_tokens", "gen_stop"))
            limit_hits += row.get("gen_stop") == _limit_stop()
    c5 = {
        "gate5_pass": (gates.get("gates") or {}).get("⑤", {}).get("status") == "pass",
        "lines_have_latency_length_stop": n_lines > 0 and missing_keys == 0,
        "sidecars_have_batch_and_device": _all(s and s.get("batch_size") is not None and s.get("device")
                                               for s in sidecars.values()),
    }
    alarm_limit = n_lines > 0 and limit_hits / n_lines > 0.05

    # ---------------------------------------------------------------- ④
    hashes = {h["around"]: h["before"] == h["after"] for h in kinds.get("hashes", [])}
    end = (kinds.get("end") or [{}])[0]
    start = (kinds.get("start") or [{}])[0]
    snap_ok = pairs_ok = None
    cfg = None
    try:
        cfg = load_uni_config(None if checkout is None else Path(checkout) / "configs" / "base.yaml")
    except Exception:  # noqa: BLE001 - 설정을 읽지 못하면 재검증을 관측하지 못한 것으로 둔다
        cfg = None
    if start.get("snapshot") and cfg is not None:
        try:
            snap_ok = verify_snapshot(Path(start["snapshot"])) == cfg.snapshot_digest
        except Exception:  # noqa: BLE001 - 재검증 실패는 False
            snap_ok = False
        from vlm.rehearsal_run import _abs

        pairs_ok = not pairs_snapshot_problems(_abs(cfg.pairs_path), cfg.pairs_digest)
    c4 = {
        "same_stamp_skipped": by_name.get("overwrite_same_stamp", {}).get("exit_code") == 0,
        "other_stamp_refused": by_name.get("overwrite_other_stamp", {}).get("exit_code") not in (None, 0, 75),
        "rerun_refused": by_name.get("rerun_uni_central", {}).get("exit_code") == 2,
        "root_hashes_same_2pp": hashes.get("2''") is True,
        "root_hashes_same_6p": hashes.get("6'") is True,
        "outputs_preexisting_preserved": end.get("outputs_preexisting_preserved") is True,
        "ledgers_prefix_kept": bool(end.get("ledger_prefix_kept")) and all(end["ledger_prefix_kept"].values()),
        "snapshot_reverified": snap_ok is True,
        "pairs_reverified": pairs_ok is True,
    }

    # ---------------------------------------------------------------- 보고만
    torn = []
    for t in cells:
        for b in sorted(export.glob(f"{t}_s{idx}.truncated-*.bin")):
            cut = I.raw(b) or b""
            gen = (I.raw(export / f"{t}_s{idx}.generations.jsonl") or b"").split(b"\n")
            iid = re.search(rb'"image_id": "([^"]+)"', cut)
            line = next((x for x in gen if iid and iid.group(0) in x), b"")
            torn.append({"bundle": f"{t}_s{idx}", "file": b.name, "prefix_same": bool(line) and line.startswith(cut)})
    fd = root / "faults"
    no_fire = sorted(p.name for p in fd.glob("*.activated.json")
                     if not (fd / p.name.replace(".activated.json", ".fired.json")).exists()) if fd.is_dir() else []
    lists_rec = I.json(root / "lists" / "lists_record.json") or {}
    I.raw(exposure_ledger) if exposure_ledger is not None else None

    planned = planned_rows
    conditions = {
        "①": {"status": _status(c1, stand_in), "checks": c1, "negatives": neg},
        "②": {"status": _status(c2, stand_in), "checks": c2, "cells": env_rows, "axis_ratio": axis},
        "③": {"status": _status(c3, stand_in), "checks": c3, "rejected": {k: sorted(v) for k, v in rejected.items()}},
        "④": {"status": _status(c4, stand_in), "checks": c4,
              "outputs_preexisting_changed": end.get("outputs_preexisting_changed", []),
              "outputs_new_n": end.get("outputs_new_n")},
        "⑤": {"status": _status(c5, stand_in), "checks": c5, "alarm_limit_hit_over_5pct": alarm_limit,
              "alarm_length": "보지 않았다 — 타깃 길이의 토크나이저 길이가 판정기 입력에 없다"},
    }
    # 필수 칸 · 단계가 빠졌으면 대역 여부와 무관하게 불통과다(외부 검토 3). 대역 실행은 판정을 내지 않는다.
    incomplete = sorted(set(missing) | {f"stage:{n}" for n in stages_missing})
    if incomplete:
        status = "fail"
    elif stand_in:
        status = "stand_in"
    else:
        status = "pass" if all(c["status"] == "pass" for c in conditions.values()) else "fail"
    out = {
        "status": status, "run_id": root.name, "purpose": "rehearsal",
        "impl": {"approved": not stand_in, "problems": impl_bad, "scorer_stand_in": d_stand_in},
        "incomplete": incomplete,
        "stages": {"n": len(stages), "all_ok": all(r.get("ok") for r in stages),
                   "planned_faults": len(planned), "intended_faults": sum(bool(r.get("intended_fault")) for r in planned),
                   "not_run": [r.get("tag") for r in kinds.get("not_run", [])]},
        "conditions": conditions,
        "report_only": {"torn_tail_prefix": torn, "activated_without_fired": no_fire},
        "lists": {"before_3c": lists_rec.get("before_3c"), "edges_file_sha256": lists_rec.get("edges_file_sha256"),
                  "excluded": lists_rec.get("excluded"), "eval_intersection": lists_rec.get("eval_intersection"),
                  "files": lists_rec.get("files"), "exposure_by_material": lists_rec.get("exposure_by_material"),
                  "exposure_by_stratum": lists_rec.get("exposure_by_stratum"),
                  "unverified_strata": lists_rec.get("unverified_strata")},
        "inputs": dict(sorted(I.seen.items())),
        "note": "지표 값을 싣지 않는다 — 조건은 자리 · 수 · 같음과 평가 쪽 산출물의 상태로 본다(리허설 2판 §3)",
    }
    raw = (json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with open(out_p, "xb") as fh:
        fh.write(raw)
    return out
