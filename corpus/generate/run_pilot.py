"""소표본 관통 시험 실행기 (스펙 §7-6 · 게이트 #8 결정 I).

(c) 판정추론 400 + (b) QA 100 을 전 구간에 1회 통과시키고 통과율을 실측한다.
본 생성이 아니다 — 입력이 limits_v0_pilot.csv(미검수)이므로 산출물은 스냅샷 비대상이다.

핵심 산출: corpus/validate/validation_report_pilot.json
  단계별 통과율 + **난이도 구간(sample_bucket)별 폐기 교차표**.
  경계 사례(허용치 ±10%)를 의도적으로 20% 넣는데 그 구간이 검증에서 더 걸러지면
  학습 데이터에 쉬운 것만 남는 선택 편향이 생긴다. 통과율 총계만으로는 보이지 않는다.

실행: uv run python -m corpus.generate.run_pilot [--n-c 400] [--n-b 100] [--batch 8]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
PILOT_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"
GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW = 200


# ---------------------------------------------------------------- 프롬프트

def prompt_c(sk: dict) -> str:
    """(c) 판정추론 — 골격의 수치·조항·판정을 그대로 두고 서술만 만든다 (§5-3)."""
    size = sk.get("size_mm")
    meas = f"{size} mm" if size is not None else f"{sk.get('measured_value')} {sk.get('measured_unit')}"
    code, clause, verdict = sk["defect_code"], sk["clause_id"], sk["verdict"]

    if sk.get("limit_value") is None:  # 불허(none_permitted) — 수치 한계가 없다
        limit_line = "- 허용 기준: 이 결함은 크기와 무관하게 허용하지 않는다\n"
        limit_rule = ""
    else:
        limit_line = f"- 허용 한계: {sk['limit_value']} {sk.get('unit') or 'mm'}\n"
        limit_rule = f" 허용 한계 {sk['limit_value']} 도 그대로 적는다."

    return (
        "다음 용접 검사 판정 골격을 한국어 서술로 옮긴다.\n"
        f"- 결함 코드(ISO 6520-1): {code}\n"
        f"- 실측 크기: {meas}\n"
        f"- 모재 두께: {sk['thickness_mm']} mm\n"
        f"- 적용 조항: {clause}\n"
        f"{limit_line}"
        f"- 판정: {verdict}\n\n"
        "반드시 지킬 것:\n"
        f"1. 본문에 결함 코드 {code} 와 조항 번호 {clause} 를 문자 그대로 적는다.\n"
        f"2. 실측 크기 {meas} 를 숫자 그대로 적는다.{limit_rule}\n"
        f"3. 판정 결론은 '{verdict}' 이라는 낱말을 그대로 쓴다. 반대 결론을 쓰지 않는다.\n"
        "4. 위에 없는 수치를 새로 만들지 않는다. 비율·분수·환산값·다른 표준 이름을 쓰지 않는다.\n\n"
        "형식: 관찰 → 대조 → 판정 순서로 3문장 이내."
    )


def parse_qa(text: str) -> tuple[str, str]:
    """생성문에서 질문/답을 뽑는다. 형식 위반은 빈 문자열로 두어 1단계가 폐기한다."""
    q = a = ""
    mq = re.search(r"질문\s*[:：]\s*(.+)", text)
    ma = re.search(r"답\s*[:：]\s*(.+)", text, re.DOTALL)
    if mq:
        q = mq.group(1).strip().splitlines()[0].strip()
    if ma:
        a = ma.group(1).strip()
    return q, a


def prompt_b(passage: str) -> str:
    return (
        "다음 규정 구절만을 근거로 질문 1개와 답 1개를 만든다.\n"
        "구절에 없는 사실을 답에 넣지 않는다. 수치를 지어내지 않는다.\n"
        "형식은 정확히 다음 두 줄이다.\n"
        "질문: ...\n답: ...\n\n"
        f"[구절]\n{passage}"
    )


# ---------------------------------------------------------------- 생성

def generate(prompts: Sequence[str], batch: int) -> list[str]:
    """transformers greedy 생성. 배치 크기는 재현 조건의 일부다 (실측: 배치가 다르면
    같은 프롬프트도 다른 문장이 나온다 — 좌측 패딩 영향)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(GEN_MODEL, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    out: list[str] = []
    gen_kw = dict(max_new_tokens=MAX_NEW, do_sample=False, temperature=None,
                  top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in chunk]
        ids = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            o = model.generate(**ids, **gen_kw)
        for j in range(len(chunk)):
            out.append(tok.decode(o[j][ids.input_ids.shape[1]:], skip_special_tokens=True).strip())
        print(f"  생성 {min(i+batch, len(prompts))}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- 구절

def load_passages(n: int, **bank_kwargs) -> list[dict]:
    """QA 원문 구절 — 구절 은행 소비 경계(`run_cycle_corpus.load_passages`)를 지난 것만.

    예전에는 조사 마크다운을 길이 · 표제로 골랐다. 그 길은 원문 지문 · 권리 · 사람 검수를 보지 않아 막았다
    (검토 §82). 본문 은행 · 권리 · 검수 기록을 주지 않으면 0 구절이다.
    """
    from corpus.generate.run_cycle_corpus import load_passages as _bank_passages

    full_bank = bank_kwargs.pop("full_bank", None)
    passages, gate = _bank_passages(n, full_bank=full_bank, **bank_kwargs)
    print(f"[구절] 은행 경계 통과 {gate['n_selected']} (거른 사유 {gate['reasons']})")
    return passages


# ---------------------------------------------------------------- 교차표

def _reasons(res) -> list[str]:
    """폐기 사유 코드 추출 — LockResult 는 reasons, 검사 결과는 codes 를 쓴다."""
    for attr in ("reasons", "codes"):
        v = getattr(res, attr, None)
        if v:
            return [str(x) for x in v]
    return []


def bucket_crosstab(records: Sequence[dict], results: Sequence, stage: str) -> dict:
    """버킷 × 통과여부 × 폐기사유. 지시 사항: 경계 구간이 더 걸러지는지 보이게 한다."""
    per: dict[str, dict] = defaultdict(lambda: {"n_in": 0, "n_pass": 0, "n_fail": 0,
                                                "fail_reasons": Counter()})
    for rec, res in zip(records, results):
        b = rec.get("sample_bucket") or "n/a"
        cell = per[b]
        cell["n_in"] += 1
        ok = getattr(res, "ok", None)
        if ok is None:
            ok = bool(res)
        if ok:
            cell["n_pass"] += 1
        else:
            cell["n_fail"] += 1
            for code in sorted(set(_reasons(res))):
                cell["fail_reasons"][code] += 1
    table = {}
    for b, c in sorted(per.items()):
        rate = c["n_pass"] / c["n_in"] if c["n_in"] else 0.0
        table[b] = {"n_in": c["n_in"], "n_pass": c["n_pass"], "n_fail": c["n_fail"],
                    "pass_rate": round(rate, 4), "fail_reasons": dict(c["fail_reasons"])}
    return {"stage": stage, "by_bucket": table}


def bias_note(before: Counter, after: Counter) -> dict:
    """생성 시점 버킷 비율과 검증 통과 후 비율을 나란히 둔다 (선택 편향 가시화)."""
    nb, na = sum(before.values()), sum(after.values())
    rows = {}
    for b in sorted(set(before) | set(after)):
        pb = before[b] / nb if nb else 0.0
        pa = after[b] / na if na else 0.0
        rows[b] = {"before_n": before[b], "before_share": round(pb, 4),
                   "after_n": after[b], "after_share": round(pa, 4),
                   "share_delta_pp": round((pa - pb) * 100, 2)}
    boundary = [b for b in rows if b.startswith("boundary")]
    bb = sum(before[b] for b in boundary) / nb if nb else 0.0
    ba = sum(after[b] for b in boundary) / na if na else 0.0
    return {"by_bucket": rows,
            "boundary_share_before": round(bb, 4),
            "boundary_share_after": round(ba, 4),
            "boundary_share_delta_pp": round((ba - bb) * 100, 2)}


# ---------------------------------------------------------------- 본체

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-c", type=int, default=400)
    ap.add_argument("--n-b", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--judge-model", default=None, help="2단계 Judge (다른 계열). 미지정 시 2단계 미실행")
    args = ap.parse_args()

    from corpus.generate import numeric_lock as nl
    from corpus.rules import limits_loader, skeleton_gen as sg
    from corpus.validate import report as rp, stage1_rules as s1

    table = limits_loader.load_limits(str(PILOT_CSV), pilot=True)
    label_codes = {r.defect_code for r in table.rows}
    clause_reg = {r.clause_id for r in table.rows}

    notes: list[str] = [
        "입력 limits_v0_pilot.csv 는 미검수 파일럿 전용이다 — 스냅샷 비대상.",
        f"생성 {GEN_MODEL}, greedy, 배치 {args.batch}. 배치 크기는 재현 조건의 일부다"
        " (실측: 배치가 다르면 같은 프롬프트도 다른 문장이 나온다).",
    ]

    # ---- (c) 판정추론 --------------------------------------------------
    print(f"[1/4] (c) 골격 {args.n_c}건 생성")
    sks = sg.generate_corpus_skeletons(table, seed=0, total=args.n_c, cap=40,
                                       inspection_method="RT")
    recs_c = [s.model_dump(mode="json") for s in sks]
    before_c = Counter(r.get("sample_bucket") or "n/a" for r in recs_c)

    print(f"[2/4] (c) 표현 변주 생성 ({len(recs_c)}건)")
    texts = generate([prompt_c(r) for r in recs_c], args.batch)
    for r, t in zip(recs_c, texts):
        r["text"] = t

    print("[3/4] (c) 0단계 수치 잠금 → 1단계 규칙 검사")
    lock_res = [nl.check_numeric_lock(r["text"], r) for r in recs_c]
    stage0_c = bucket_crosstab(recs_c, lock_res, "stage0_numeric_lock")
    survivors = [r for r, res in zip(recs_c, lock_res) if getattr(res, "ok", False)]

    st1 = s1.run_stage1(survivors, asset="c", table=table, label_codes=label_codes,
                        clause_registry=clause_reg, inspection_method="RT",
                        verdict_mode="full")
    stage1_c = bucket_crosstab(survivors, st1.results, "stage1_rule")
    passed_c = [r for r, res in zip(survivors, st1.results) if res.ok]
    after_c = Counter(r.get("sample_bucket") or "n/a" for r in passed_c)

    # ---- (b) QA -------------------------------------------------------
    print(f"[4/4] (b) QA {args.n_b}건")
    passages = load_passages(args.n_b)
    recs_b: list[dict] = []
    if passages:
        qa_texts = generate([prompt_b(p["text"]) for p in passages], args.batch)
        for p, t in zip(passages, qa_texts):
            q, a = parse_qa(t)
            recs_b.append({"sample_id": p["passage_id"], "passage_id": p["passage_id"],
                           "doc": p["doc"], "text": t, "question": q, "answer": a,
                           "evidence_passage_ids": [p["passage_id"]],
                           "sample_bucket": "n/a"})
        st1b = s1.run_stage1(recs_b, asset="b",
                             passage_ids={p["passage_id"] for p in passages})
        stage1_b = bucket_crosstab(recs_b, st1b.results, "stage1_rule")
        b_report = rp.build_validation_report(
            asset="b_qa", version="v0-pilot", pilot=True,
            stages={"stage1_rule": rp.stage_summary(
                len(recs_b), st1b.n_pass, dict(st1b.fail_reasons))},
        )
        b_report["bucket_crosstab"] = [stage1_b]
        b_report["bucket_crosstab_note"] = (
            "(b) QA 에는 난이도 구간(sample_bucket)이 없다 — 버킷은 허용치 대비 수치 위치"
            "(합격 / 불합격 / 경계 ±10%)로 정의되는데, QA 는 규정 구절에서 만든 질의응답이라"
            " 대조할 허용치가 없다. 따라서 n/a 한 칸이 정상이며 경계 사례 선택 편향 점검은"
            " (c) 판정추론에만 적용된다 (게이트 #13 지시 사항)."
        )
        b_report["passage_screen"] = {"paid_standard_excluded": True,
                                      "n_passages": len(passages)}
    else:
        b_report = rp.build_validation_report(asset="b_qa", version="v0-pilot",
                                              pilot=True, stages={})
        b_report["blocked"] = "원문 구절 없음 — 파싱 산출물 부재"

    # ---- 2단계 --------------------------------------------------------
    if args.judge_model:
        notes.append(f"2단계 Judge: {args.judge_model}")
    else:
        notes.append("2단계(다른 계열 Judge) 미실행 — 로컬에 다른 계열 모델이 없다."
                     " 자기검증 금지 조항상 Qwen 계열로 대체할 수 없다.")

    lock_reasons = Counter(c for r in lock_res if not getattr(r, "ok", False)
                           for c in sorted(set(_reasons(r))))
    c_report = rp.build_validation_report(
        asset="c_reasoning", version="v0-pilot", pilot=True,
        stages={
            "stage0_numeric_lock": rp.stage_summary(
                len(recs_c), sum(1 for r in lock_res if getattr(r, "ok", False)),
                dict(lock_reasons)),
            "stage1_rule": rp.stage_summary(
                len(survivors), st1.n_pass, dict(st1.fail_reasons)),
        },
    )
    c_report["bucket_crosstab"] = [stage0_c, stage1_c]
    c_report["bucket_crosstab_note"] = (
        "난이도 구간별 폐기 분포 (게이트 #13 지시). 경계 사례(허용치 ±10%)를 의도적으로"
        " 20% 넣으므로, 그 구간이 다른 구간보다 많이 걸러지면 학습 데이터에 쉬운 것만 남는"
        " 선택 편향이 생긴다. 통과율 총계로는 보이지 않아 구간별로 나눠 기록한다."
        " boundary_eq/low/high 는 경계 구간의 3층(정확 일치 / 하경계 / 상경계)이다."
    )
    c_report["selection_bias"] = bias_note(before_c, after_c)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, cwd=REPO).stdout.strip()
    pilot = rp.build_pilot_report([c_report, b_report], git_commit=commit, notes=notes)
    pilot["generation"] = {"model": GEN_MODEL, "decoding": "greedy",
                           "batch_size": args.batch, "max_new_tokens": MAX_NEW}
    out = rp.write_pilot_report(pilot, REPO / "corpus/validate")
    print(f"\n보고서: {out}")
    print(json.dumps(c_report["selection_bias"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
