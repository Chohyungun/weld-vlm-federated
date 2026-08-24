"""한 사이클 파일럿용 합성 corpus 축소본 — QA 200 + 판정추론 200.

판정 축을 좁혀서 만든다. 치수 합부(측정 크기와 허용치를 비교해 합격·불합격을 단정하는 것)는
넣지 않는다. 픽셀을 mm로 바꿀 수 없는 상태에서 치수 판정문을 합성하면 근거 없는 수치를
지어내게 된다. 대신 두 축으로 만든다.

  (a) 조항 검색 + 기준 서술 — 결함 종류와 검사 방식으로 적용 조항을 찾고 그 조항이 정한
      기준을 서술한다. 특정 이미지의 합부는 말하지 않는다.
  (b) 조치 서술 — 결함에 대한 처리 방법을 서술한다. 근거는 IACS Rec.47의 보수 지침이며,
      치수 임계값이 붙은 항목은 제외한다.

규칙: 생성은 greedy 1회, 실패분은 재생성하지 않고 폐기하며 통과율을 기록한다.
검증은 생성 모델과 다른 계열 모델이 맡는다.

실행: uv run python -m corpus.generate.run_cycle_corpus [--n-qa 200] [--n-reason 200]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO = Path(__file__).resolve().parents[2]
PILOT_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"
OUT_DIR = REPO / "corpus/generate/cycle_pilot"
PASSAGE_DOCS = [
    ("KR-RULES-P2", REPO / "corpus/parse/survey/KR-RULES-P2/KR-RULES-P2_p316-336.md"),
    ("IACS47", REPO / "corpus/parse/survey/IACS47/IACS47_full.md"),
]
GEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "microsoft/Phi-4-mini-instruct"   # 생성과 다른 계열 (자기검증 금지)
MAX_NEW = 200
PAID_STD = re.compile(r"ISO\s*5817|ISO\s*10042|ISO\s*10675|AWS\s+Welding\s+Handbook")

# 결함 코드 → 한국어 명칭. 사상표(계약 #1)에서 읽어 하드코딩을 피한다.
def defect_names() -> dict[str, str]:
    from data.label_map import load_label_map
    lm = load_label_map()
    out: dict[str, str] = {}
    for dt in lm.defect_types.values():
        out[dt.iso_code] = dt.name_ko
        for alt in getattr(dt, "iso_code_alt", []) or []:
            out.setdefault(alt, dt.name_ko)
    return out


# (b) 조치 축의 근거. IACS Rec.47(무료 공개)의 보수 지침 중 **치수 임계값이 붙지 않은**
# 항목만 옮겼다. 치수가 붙은 항목(언더컷 D값 등)은 치수 판정을 부르므로 제외했다.
REMEDY_TABLE = [
    {"topic": "아크 스트라이크", "clause_id": "IACS47-Table", "inspection_method": "VT",
     "remedy_ko": "경화된 부위를 그라인딩으로 제거한다",
     "source_en": "Remove the hardened zone by grinding or other measures"},
    {"topic": "슬래그·유분·부착물", "clause_id": "IACS47-Prep", "inspection_method": "VT",
     "remedy_ko": "용접 전에 제거한다",
     "source_en": "Slag, grease, loose mill scale, rust and paint to be removed"},
    {"topic": "균열 보수", "clause_id": "IACS47-Crack", "inspection_method": "RT",
     "remedy_ko": "보수 용접이 가능하다고 판단되면 정해진 보수 기법을 따른다",
     "source_en": "In the event that a crack is considered weldable, the following techniques should be adopted"},
    {"topic": "균열 종단부", "clause_id": "IACS47-Crack", "inspection_method": "VT",
     "remedy_ko": "모서리에서 끝나는 균열은 탭재 위에서 용접을 종료한다",
     "source_en": "For cracks ending on edges weld to be terminated on a tab"},
]


# ------------------------------------------------------------------ 골격

def build_clause_skeletons(n: int) -> list[dict]:
    """(a) 조항 검색 + 기준 서술. limits.csv 행이 곧 근거다. 합부는 말하지 않는다."""
    from corpus.rules import limits_loader
    names = defect_names()
    table = limits_loader.load_limits(str(PILOT_CSV), pilot=True)
    rows = [r for r in table.rows if getattr(r, "scope", "active") == "active"]
    rows.sort(key=lambda r: r.rule_id)

    # 조항이 정한 기준을 문장 재료로 편다 (수치는 표에서 그대로 가져온다)
    def criterion(r) -> str:
        if r.limit_rule == "none_permitted":
            return "크기와 무관하게 허용하지 않는다"
        unit = r.unit or "mm"
        if r.limit_rule == "const":
            return f"{r.limit_value} {unit} 이하"
        if r.limit_rule == "prop_t":
            return f"모재 두께의 {r.limit_factor} 배 이하"
        if r.limit_rule == "prop_t_cap":
            return f"모재 두께의 {r.limit_factor} 배 이하이고 최대 {r.limit_cap} {unit}"
        return "표에 정한 값 이하"

    framings = [
        ("적용 조항", "이 결함에 적용되는 조항과 그 조항이 정한 기준을 밝힌다"),
        ("기준 서술", "조항이 정한 허용 기준을 서술한다"),
        ("검사 방식", "이 기준이 어느 검사 방식에 해당하는지 밝힌다"),
        ("두께 구간", "이 기준이 적용되는 모재 두께 구간을 밝힌다"),
    ]
    out: list[dict] = []
    i = 0
    while len(out) < n:
        r = rows[i % len(rows)]
        fr, ask = framings[(i // len(rows)) % len(framings)]
        tmax = "상한 없음" if r.thickness_max is None else f"{r.thickness_max} mm 미만"
        out.append({
            "sample_id": f"clause-{i:04d}",
            "axis": "조항검색_기준서술",
            "framing": fr, "ask": ask,
            "rule_id": r.rule_id,
            "defect_code": r.defect_code,
            "defect_name": names.get(r.defect_code, "해당 결함"),
            "inspection_method": r.inspection_method,
            "clause_id": r.clause_id,
            "thickness_min": str(r.thickness_min),
            "thickness_max": tmax,
            "criterion": criterion(r),
            "source_doc": r.source_doc,
        })
        i += 1
    return out


def build_remedy_skeletons(n: int) -> list[dict]:
    """(b) 조치 서술. IACS Rec.47 보수 지침 중 치수 임계값이 없는 항목만 쓴다."""
    framings = [
        ("조치", "이 상황에서 취할 조치를 서술한다"),
        ("근거", "그 조치의 근거가 되는 지침을 밝힌다"),
    ]
    out = []
    for i in range(n):
        t = REMEDY_TABLE[i % len(REMEDY_TABLE)]
        fr, ask = framings[(i // len(REMEDY_TABLE)) % len(framings)]
        out.append({"sample_id": f"remedy-{i:04d}", "axis": "조치서술",
                    "framing": fr, "ask": ask, **t})
    return out


# ------------------------------------------------------------------ 프롬프트

def prompt_clause(sk: dict) -> str:
    return (
        "다음 자료만 사용해 한국어 2~3문장으로 서술한다.\n"
        f"- 결함: {sk['defect_name']} (ISO 6520-1 코드 {sk['defect_code']})\n"
        f"- 검사 방식: {sk['inspection_method']}\n"
        f"- 적용 조항: {sk['clause_id']}\n"
        f"- 조항이 정한 기준: {sk['criterion']}\n"
        f"- 적용 두께 구간: {sk['thickness_min']} mm 이상 {sk['thickness_max']}\n\n"
        f"요구: {sk['ask']}.\n"
        "반드시 지킬 것:\n"
        f"1. 조항 번호 {sk['clause_id']} 와 결함 코드 {sk['defect_code']} 를 그대로 적는다.\n"
        "2. 위에 없는 수치를 새로 만들지 않는다. 비율·분수·환산값을 만들지 않는다.\n"
        "3. **특정 용접부가 합격인지 불합격인지 판단하지 않는다.** 조항이 정한 기준만 서술한다.\n"
        "4. 다른 표준 이름을 인용하지 않는다."
    )


def prompt_remedy(sk: dict) -> str:
    return (
        "다음 자료만 사용해 한국어 2~3문장으로 서술한다.\n"
        f"- 상황: {sk['topic']}\n"
        f"- 조치: {sk['remedy_ko']}\n"
        f"- 근거 지침: {sk['clause_id']} ({sk['source_en']})\n\n"
        f"요구: {sk['ask']}.\n"
        "반드시 지킬 것:\n"
        f"1. 근거로 {sk['clause_id']} 를 그대로 적는다.\n"
        "2. 수치를 만들지 않는다. 치수·온도·시간 같은 값을 지어내지 않는다.\n"
        "3. 합격·불합격을 판단하지 않는다.\n"
        "4. 다른 표준 이름을 인용하지 않는다."
    )


# 구절당 최대 3문항까지 만든다(스펙 §7-1). greedy 라 같은 프롬프트는 같은 출력을 내므로,
# 문항을 늘리려면 묻는 각도를 달리해야 한다.
QA_ASPECTS = [
    ("정의", "구절이 정의하거나 규정하는 대상이 무엇인지 묻는다"),
    ("조건", "구절이 정한 조건·범위·적용 대상을 묻는다"),
    ("절차", "구절이 요구하는 절차나 방법을 묻는다"),
]


def prompt_qa(passage: str, aspect: str = "정의", ask: str = "") -> str:
    return (
        "다음 규정 구절만을 근거로 질문 1개와 답 1개를 만든다.\n"
        f"질문은 '{aspect}' 관점으로 만든다: {ask}\n"
        "구절에 없는 사실을 답에 넣지 않는다. 수치를 지어내지 않는다.\n"
        "형식은 정확히 다음 두 줄이다.\n"
        "질문: ...\n답: ...\n\n"
        f"[구절]\n{passage}"
    )


def parse_qa(text: str) -> tuple[str, str]:
    q = a = ""
    mq = re.search(r"질문\s*[:：]\s*(.+)", text)
    ma = re.search(r"답\s*[:：]\s*(.+)", text, re.DOTALL)
    if mq:
        q = mq.group(1).strip().splitlines()[0].strip()
    if ma:
        a = ma.group(1).strip()
    return q, a


# ------------------------------------------------------------------ 검사

VERDICT_WORD = re.compile(r"합격|불합격|적합 판정|부적합 판정")
NUM = re.compile(r"\d+(?:\.\d+)?")
STD_REF = re.compile(r"(ISO|KS|EN|IEC|API|AWS|ASME)\s?\d+", re.I)


def check_record(text: str, sk: dict) -> tuple[bool, list[str]]:
    """0단계 규칙 검사. 실패분은 폐기한다(재생성 금지).

    치수 합부를 금지했으므로 판정어 출현 자체가 위반이다.
    """
    reasons: list[str] = []
    if not text.strip():
        return False, ["empty"]

    if VERDICT_WORD.search(text):
        reasons.append("verdict_asserted")          # 합부 단정 금지

    if sk["clause_id"] not in text:
        reasons.append("missing_clause")
    if sk["axis"] == "조항검색_기준서술" and sk["defect_code"] not in text:
        reasons.append("missing_defect_code")

    # 허용 수치: 골격이 준 값 + 조항 번호에 포함된 숫자
    allowed = set()
    for k in ("defect_code", "thickness_min", "thickness_max", "criterion", "clause_id"):
        v = str(sk.get(k, ""))
        allowed |= set(NUM.findall(v))
    masked = text
    for tok in sorted({sk["clause_id"], *STD_REF.findall(text)}, key=len, reverse=True):
        masked = masked.replace(tok, " ")
    masked = re.sub(r"ISO\s*6520-1", " ", masked)
    extra = [n for n in NUM.findall(masked) if n not in allowed and n.rstrip("0.") not in
             {a.rstrip("0.") for a in allowed}]
    if extra:
        reasons.append("extra_value")

    other = [s for s in STD_REF.findall(text)]
    if any(o.upper().startswith(("API", "AWS", "ASME")) for o in other):
        reasons.append("foreign_standard")
    return (not reasons), reasons


# ------------------------------------------------------------------ 생성·검증

def generate(prompts, batch: int, model_id: str, max_new: int = MAX_NEW) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
              top_k=None, pad_token_id=tok.eos_token_id)
    out: list[str] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in chunk]
        ids = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            o = model.generate(**ids, **kw)
        for j in range(len(chunk)):
            out.append(tok.decode(o[j][ids.input_ids.shape[1]:], skip_special_tokens=True).strip())
        print(f"    {min(i+batch, len(prompts))}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def judge(records: list[dict], batch: int) -> list[dict]:
    """다른 계열 모델의 이진 판정. 생성 모델이 자기 결과를 검증하지 않는다."""
    prompts = []
    for r in records:
        sk = r["skeleton"]
        basis = (f"결함 {sk.get('defect_name','')} / 조항 {sk['clause_id']} / "
                 f"기준 {sk.get('criterion', sk.get('remedy_ko',''))}")
        prompts.append(
            "아래 [자료]와 [문장]을 비교한다.\n"
            "문장이 자료에 없는 사실이나 수치를 담고 있으면 NG, 자료 범위 안이면 OK.\n"
            "문장이 특정 용접부의 합격·불합격을 단정하면 NG.\n"
            "첫 줄에 OK 또는 NG만 쓰고, 둘째 줄에 이유를 한 줄로 쓴다.\n\n"
            f"[자료]\n{basis}\n\n[문장]\n{r['text']}")
    outs = generate(prompts, batch, JUDGE_MODEL, max_new=64)
    for r, o in zip(records, outs):
        head = o.strip().splitlines()[0].upper() if o.strip() else ""
        r["judge_pass"] = head.startswith("OK")
        r["judge_note"] = " ".join(o.split())[:160]
    return records


# ------------------------------------------------------------------ QA 구절

def load_passages(n: int) -> list[dict]:
    out = []
    for doc_id, path in PASSAGE_DOCS:
        if not path.exists():
            continue
        blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n")]
        for k, b in enumerate(blocks):
            if not (200 <= len(b) <= 800) or b.lstrip().startswith(("|", "#", "<!--")):
                continue
            if PAID_STD.search(b):
                continue
            out.append({"passage_id": f"{doc_id}#b{k:04d}", "doc": doc_id, "text": b})
    out.sort(key=lambda r: r["passage_id"])
    return out[:n]


# ------------------------------------------------------------------ 본체


def write_report(args, recs, survivors, accepted, qa_recs, qa_accepted, stage0_fail):
    """보고서와 산출 jsonl 을 쓴다. 생성·판정 두 단계에서 공통으로 쓴다."""
    def rate(n_pass, n_in):
        return round(n_pass / n_in, 4) if n_in else None

    report = {
        "purpose": "한 사이클 파일럿 판정부 입력용 축소본",
        "axis_note": ("치수 합부는 넣지 않았다. 픽셀을 mm로 바꿀 수 없어 측정 크기 기반 "
                      "합부를 합성하면 근거 없는 수치를 지어내게 된다. "
                      "조항 검색 + 기준 서술과 조치 서술 두 축으로만 만들었다."),
        "generation": {"model": GEN_MODEL, "decoding": "greedy", "batch_size": args.batch,
                       "retry": "없음 (실패분 폐기)"},
        "validation": {"model": JUDGE_MODEL,
                       "different_family": True,
                       "note": "생성 Qwen 계열, 검증 Phi 계열"},
        "reasoning": {
            "n_in": len(recs),
            "axis_split": dict(Counter(r["axis"] for r in recs)),
            "stage0": {"n_pass": len(survivors), "n_fail": len(recs) - len(survivors),
                       "pass_rate": rate(len(survivors), len(recs)),
                       "fail_reasons": dict(stage0_fail)},
            "stage2_judge": {"n_in": len(survivors), "n_pass": len(accepted),
                             "n_fail": len(survivors) - len(accepted),
                             "pass_rate": rate(len(accepted), len(survivors))},
            "n_accepted": len(accepted),
            "end_to_end_rate": rate(len(accepted), len(recs)),
        },
        "qa": {
            "n_in": len(qa_recs), "n_accepted": len(qa_accepted),
            "pass_rate": rate(len(qa_accepted), len(qa_recs)),
            "fail_reasons": dict(Counter(w for r in qa_recs for w in r["stage0_reasons"])),
            "passage_screen": "유료 표준 전재 의심 구절 제외",
        },
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd=REPO).stdout.strip(),
    }
    (OUT_DIR / "reasoning_accepted.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in accepted), encoding="utf-8")
    (OUT_DIR / "qa_accepted.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in qa_accepted), encoding="utf-8")
    (OUT_DIR / "discarded.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False)
                  for r in recs + qa_recs if not r.get("stage0_pass") or
                  (r in survivors and not r.get("judge_pass"))), encoding="utf-8")
    (OUT_DIR / "cycle_corpus_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps({k: report[k] for k in ("reasoning", "qa")},
                            ensure_ascii=False, indent=1))
    print("산출:", OUT_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qa", type=int, default=200)
    ap.add_argument("--n-reason", type=int, default=200)
    ap.add_argument("--clause-share", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stage", choices=["generate", "judge", "all"], default="all",
                    help="생성과 검증을 다른 프로세스로 나눈다. 큰 모델 둘을 한 프로세스에 "
                         "올리면 적재 도중 죽는다(실측).")
    args = ap.parse_args()
    mid = OUT_DIR / "_raw_generated.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage == "judge":
        d = json.loads(mid.read_text(encoding="utf-8"))
        recs, qa_recs = d["recs"], d["qa_recs"]
        stage0_fail = Counter(d["stage0_fail"])
        base = [None] * d["n_base_passages"]
        survivors = [r for r in recs if r["stage0_pass"]]
        qa_accepted = [r for r in qa_recs if r["stage0_pass"]]
        print(f"[판정] 중간 파일 로드: 판정추론 {len(recs)} (통과 {len(survivors)}), QA {len(qa_recs)}")
        if survivors:
            judge(survivors, args.batch)
        accepted = [r for r in survivors if r.get("judge_pass")]
        write_report(args, recs, survivors, accepted, qa_recs, qa_accepted, stage0_fail)
        return

    n_clause = int(args.n_reason * args.clause_share)
    n_remedy = args.n_reason - n_clause
    print(f"[1/5] 골격: 조항검색·기준서술 {n_clause} + 조치서술 {n_remedy}")
    sks = build_clause_skeletons(n_clause) + build_remedy_skeletons(n_remedy)

    # 생성 모델은 한 번만 올린다. 같은 프로세스에서 두 번 재적재하면 세그폴트가 난다(실측).
    base = load_passages(args.n_qa)
    passages = []
    for asp, ask in QA_ASPECTS:
        for pp in base:
            if len(passages) >= args.n_qa:
                break
            passages.append({**pp, "aspect": asp, "ask": ask,
                             "qa_id": f"{pp['passage_id']}#{asp}"})
    print(f"[2/5] 생성 {len(sks)}건 + QA {len(passages)}건 "
          f"(구절 {len(base)}개 x 관점, greedy 1회, 모델 1회 적재)")
    prompts = [prompt_clause(s) if s["axis"] == "조항검색_기준서술" else prompt_remedy(s)
               for s in sks]
    prompts += [prompt_qa(p["text"], p["aspect"], p["ask"]) for p in passages]
    all_texts = generate(prompts, args.batch, GEN_MODEL)
    texts, qa_texts = all_texts[:len(sks)], all_texts[len(sks):]
    recs = [{"sample_id": s["sample_id"], "axis": s["axis"], "framing": s["framing"],
             "skeleton": s, "text": t} for s, t in zip(sks, texts)]

    print("[3/5] 0단계 규칙 검사")
    stage0_fail = Counter()
    for r in recs:
        ok, why = check_record(r["text"], r["skeleton"])
        r["stage0_pass"] = ok
        r["stage0_reasons"] = why
        for w in why:
            stage0_fail[w] += 1
    survivors = [r for r in recs if r["stage0_pass"]]

    print(f"[4/5] QA 0단계 검사 {len(passages)}건")
    qa_recs = []
    if passages:
        for p, t in zip(passages, qa_texts):
            q, a = parse_qa(t)
            ok = bool(q and a)
            qa_recs.append({"sample_id": p["qa_id"], "passage_id": p["passage_id"],
                            "aspect": p["aspect"],
                            "doc": p["doc"], "question": q, "answer": a, "text": t,
                            "stage0_pass": ok,
                            "stage0_reasons": [] if ok else ["schema_violation"]})
        qa_sur = [r for r in qa_recs if r["stage0_pass"]]
        seen: set[str] = set()
        for r in qa_sur:
            key = re.sub(r"\s+", "", r["question"])
            if key in seen:
                r["stage0_pass"] = False
                r["stage0_reasons"] = ["duplicate_question"]
            seen.add(key)
        qa_accepted = [r for r in qa_recs if r["stage0_pass"]]
    else:
        qa_accepted = []

    # 큰 모델 둘을 한 프로세스에 올리면 판정 모델 적재 도중 죽는다(실측). 단계를 나눈다.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mid.write_text(json.dumps({"recs": recs, "qa_recs": qa_recs,
                               "stage0_fail": dict(stage0_fail),
                               "n_base_passages": len(base)},
                              ensure_ascii=False), encoding="utf-8")
    if args.stage == "generate":
        print(f"생성분 저장: {mid} (판정추론 {len(recs)}, QA {len(qa_recs)})")
        return

    print(f"[5/5] 2단계 다른 계열 검증 {len(survivors)}건 ({JUDGE_MODEL})")
    if survivors:
        judge(survivors, args.batch)
    accepted = [r for r in survivors if r.get("judge_pass")]

    write_report(args, recs, survivors, accepted, qa_recs, qa_accepted, stage0_fail)


if __name__ == "__main__":
    main()
