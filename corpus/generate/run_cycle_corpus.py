"""합성 corpus 사이클 실행기 — 생성 → 0단계 수치 잠금 → 1단계 규칙 → 2단계 다른 계열 판정.

체크리스트 6·7·8·9·19. 재작성 이유는 80번 §4-1 이 적은 사고 열둘이다. 요약하면 셋이다.

**(1) 정본이 있는데 아무도 안 불렀다 (B7·B10, G1-1).** 이 파일에 `check_record` 라는
자체 규칙 검사가 있었고, 그 안의 후행 0 정규화가 `n.rstrip("0.")` 이라 4 ≡ 40 ≡ 400 이
동치가 됐다. 10배·100배 뻥튀기가 통과했고 stage0 통과율 0.89 를 만든 것이 그 검사다.
지금은 `numeric_lock.check_numeric_lock` 과 `stage1_rules.run_stage1` 만 부른다.
**이 파일에는 수치·판정어·표준 정규식이 하나도 없다** — 배선 시험이 그것을 강제한다.

**(2) 생성기와 검증기가 다른 것을 봤다 (B2·B4, G4-1).** 생성 프롬프트는 결함 코드를
의무화했는데 판정 [자료]에는 결함 코드가 없었고, 판정 지시는 "자료에 없는 사실은 NG"
였다. 지시대로 쓴 문장이 구조적으로 기각됐다(178건 중 98건). 지금은 두 프롬프트가
`basis.render_basis()` 가 낸 **같은 문자열**을 싣는다.

**(3) 골격이 스펙 밖이었다 (B1·B3, G3-1).** 자체 골격 dict 를 f-string 으로 폈더니
`Unit.MM`·개구간 인코딩 `25.01`·후행 0 `4.00` 이 자료로 새고 생성문에 박혔다 —
채택 69건 중 49건(71%)이 그 산물이다. 지금 (c) 축 골격은
`skeleton_gen.generate_corpus_skeletons` 정본이 만들고, 자료 직렬화는 `clause_text`
표기 정본을 지난다.

축은 둘이다.
  (c) 조항검색·기준서술 — 정본 골격. `--verdict-mode full` 이 스펙 정본(§4-7)이고,
      `clause_only` 면 verdict·margin 이 자료에서 빠지고 판정어·판정 함의 표현이 금지된다.
  (b2) 조치서술 — IACS Rec.47 보수 지침 중 치수 임계값이 없는 항목. 허용치 행이 없어
      골격 검사 대상이 아니고, `check_normal_lock`(수치 일절 금지) 으로 받는다.
  (b) QA — 규정 구절 기반 질의응답. `run_stage1(asset="b")`.

검증기 설정은 `configs/corpus_validation.yaml` 사전 등록본을 읽는다. 두 후보를 같은
예산·같은 사고 정책으로 돌린다 (G12-4) — 파일럿의 56.75pp 는 계열 차이가 아니라
계열+설정 차이였다.

실행:
  uv run python -m corpus.generate.run_cycle_corpus --stage generate
  uv run python -m corpus.generate.run_cycle_corpus --stage judge [--judge-id deepseek]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import yaml

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

REPO = Path(__file__).resolve().parents[2]
PILOT_CSV = REPO / "corpus/rules/limits_v0_pilot.csv"
OUT_DIR = REPO / "corpus/generate/cycle_pilot"
CONFIG_PATH = REPO / "configs/corpus_validation.yaml"
PASSAGE_DOCS = [
    ("KR-RULES-P2", REPO / "corpus/parse/survey/KR-RULES-P2/KR-RULES-P2_p316-336.md"),
    ("IACS47", REPO / "corpus/parse/survey/IACS47/IACS47_full.md"),
]
#: 골격 표집 시드. 같은 (limits sha, seed, config)면 골격 jsonl 이 바이트 동일하다.
SKELETON_SEED = 0

# 유료 표준 전재 스크린 (§3-6) — 이 식별자가 있는 구절은 QA 원문으로 쓰지 않는다.
# 수치·판정 정규식이 아니라 문서 식별자 스크린이라 정본 위임 대상이 아니다.
PAID_STD = re.compile(r"ISO\s*5817|ISO\s*10042|ISO\s*10675|AWS\s+Welding\s+Handbook")

# 표현 헬퍼의 정본은 `corpus.rules.clause_text` 하나다 — 여기서는 가져다 쓰기만 한다.
from corpus.generate.basis import (  # noqa: E402
    AXIS_CLAUSE,
    AXIS_REMEDY,
    render_basis,
    required_mentions,
)
# 재수출 — 시험과 `make_pairs_pilot` 이 여기서 가져간다. 표현 정본은 clause_text 다.
from corpus.rules.clause_text import (  # noqa: E402,F401
    method_ko,
    num_ko,
    thickness_ko,
    unit_ko,
    val,
)


def load_config(path: Path = CONFIG_PATH) -> dict:
    """검증기 사전 등록본. 모델·예산·사고 정책·파싱 규칙이 여기서만 온다 (G12-3)."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def defect_names() -> dict[str, str]:
    """결함 코드 → 한국어 명칭. 사상표(계약 #1)에서 읽어 하드코딩을 피한다."""
    from data.label_map import load_label_map

    lm = load_label_map()
    out: dict[str, str] = {}
    for dt in lm.defect_types.values():
        out[dt.iso_code] = dt.name_ko
        for alt in getattr(dt, "iso_code_alt", []) or []:
            out.setdefault(alt, dt.name_ko)
    return out


# (b2) 조치 축의 근거. IACS Rec.47(무료 공개)의 보수 지침 중 **치수 임계값이 붙지 않은**
# 항목만 옮겼다. 치수가 붙은 항목(언더컷 D값 등)은 치수 판정을 부르므로 제외했다.
REMEDY_TABLE = [
    {"topic": "아크 스트라이크", "source_ref": "IACS Rec.47", "inspection_method": "VT",
     "remedy_ko": "경화된 부위를 그라인딩으로 제거한다",
     "source_en": "Remove the hardened zone by grinding or other measures"},
    {"topic": "슬래그·유분·부착물", "source_ref": "IACS Rec.47", "inspection_method": "VT",
     "remedy_ko": "용접 전에 제거한다",
     "source_en": "Slag, grease, loose mill scale, rust and paint to be removed"},
    {"topic": "균열 보수", "source_ref": "IACS Rec.47", "inspection_method": "RT",
     "remedy_ko": "보수 용접이 가능하다고 판단되면 정해진 보수 기법을 따른다",
     "source_en": "In the event that a crack is considered weldable, the following techniques should be adopted"},
    {"topic": "균열 종단부", "source_ref": "IACS Rec.47", "inspection_method": "VT",
     "remedy_ko": "모서리에서 끝나는 균열은 탭재 위에서 용접을 종료한다",
     "source_en": "For cracks ending on edges weld to be terminated on a tab"},
]
#: 조치 축이 인용해도 되는 표준. 자료가 지시하는 문서 하나뿐이다.
REMEDY_STANDARDS = ("IACS Rec.47",)

QA_ASPECTS = [
    ("정의", "구절이 정의하거나 규정하는 대상이 무엇인지 묻는다"),
    ("조건", "구절이 정한 조건·범위·적용 대상을 묻는다"),
    ("절차", "구절이 요구하는 절차나 방법을 묻는다"),
]


# ------------------------------------------------------------------ 골격

def build_clause_records(n: int, verdict_mode: str, seed: int = SKELETON_SEED) -> list[dict]:
    """(c) 축 레코드. 골격은 **정본 생성기**가 만든다 (자체 dict 금지 — B1·B6).

    `generate_corpus_skeletons` 는 조합 열거·난이도 층화·두께 표집·행 재조회 불변식을
    모두 지고 있고, 같은 (limits sha, seed, config)면 바이트 동일하다.
    """
    from corpus.rules import limits_loader
    from corpus.rules import skeleton_gen as sg
    from corpus.rules.schema import VerdictMode

    table = limits_loader.load_limits(str(PILOT_CSV), pilot=True)
    sks = sg.generate_corpus_skeletons(table, seed=seed, total=n, cap=40,
                                       inspection_method="RT")
    names = defect_names()
    gate = VerdictMode(verdict_mode) is VerdictMode.CLAUSE_ONLY
    out: list[dict] = []
    for sk in sks:
        rec = sk.model_dump(mode="json")
        if gate:
            # §4-7 게이트는 **직렬화**에서 건다. judge 계산은 이미 끝났고 값만 내린다.
            rec["verdict"] = None
            rec["margin"] = None
            rec["verdict_type"] = None
        rec["axis"] = AXIS_CLAUSE
        rec["defect_name"] = names.get(str(rec["defect_code"]), "해당 결함")
        rec["sample_id"] = f"clause-{rec['sample_id']}"
        out.append(rec)
    return out


def build_remedy_records(n: int) -> list[dict]:
    """(b2) 조치 축. 허용치 행이 없어 골격 검사 대상이 아니다 — 수치 금지로 받는다."""
    out = []
    for i in range(n):
        t = REMEDY_TABLE[i % len(REMEDY_TABLE)]
        out.append({"sample_id": f"remedy-{i:04d}", "axis": AXIS_REMEDY, **t})
    return out


# ------------------------------------------------------------------ 프롬프트
#
# 생성도 판정도 `render_basis` 가 낸 **같은 [자료]** 를 싣는다 (G4-1). 필드를 빼거나
# 더하려면 `corpus/generate/basis.py` 하나를 고쳐야 하고, 고치면 양쪽이 함께 움직인다.

def _obligations(rec: dict) -> list[str]:
    """생성문이 문자 그대로 담아야 하는 요소. **자료 안에서만 고른다.**

    자료 밖 요소를 의무화하면 지시대로 쓴 문장이 판정에서 구조적으로 기각된다 (B4).
    `basis.required_mentions` 가 그 목록의 정본이고, 시험이 자료와의 포함 관계를 강제한다.
    """
    return [f"[{k}] 의 값 '{v}' 를 문자 그대로 적는다." for k, v in required_mentions(rec).items()]


def prompt_generate(rec: dict) -> str:
    """생성 프롬프트. [자료]는 판정 프롬프트와 **같은 문자열**이다 (G4-1)."""
    basis = render_basis(rec)
    rules = list(_obligations(rec))
    if rec["axis"] == AXIS_REMEDY:
        ask = "이 상황에서 취할 조치와 그 근거를 서술한다"
        rules += [
            "수치를 쓰지 않는다. 치수·온도·시간·개수를 지어내지 않는다.",
            "합격·불합격을 판단하지 않는다.",
            "자료에 없는 표준 이름을 인용하지 않는다.",
        ]
    else:
        ask = "적용 조항과 그 조항이 정한 기준을 밝히고, 실측값을 기준과 대조해 서술한다"
        rules += ["자료에 없는 수치를 새로 만들지 않는다. 비율·분수·환산값을 만들지 않는다."]
        if not rec.get("verdict"):
            rules += ["**특정 용접부가 합격인지 불합격인지 말하지 않는다.** 적합·부적합·"
                      "만족·허용 같은 판정을 함의하는 표현도 쓰지 않는다."]
        rules += [
            "자료에 없는 표준 이름을 인용하지 않는다.",
            "수치는 자료에 적힌 표기 그대로 쓴다. 자릿수를 늘리거나 낱말로 바꾸지 않는다.",
        ]
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    return (
        "다음 [자료]만 사용해 한국어 2~3문장으로 서술한다.\n\n"
        f"[자료]\n{basis}\n\n"
        f"요구: {ask}.\n"
        "반드시 지킬 것:\n"
        f"{numbered}"
    )


def prompt_judge(rec: dict) -> str:
    """판정 프롬프트. [자료]는 생성 프롬프트와 **같은 문자열**이다 (G4-1·G4-3)."""
    gate = "" if rec.get("verdict") else (
        "문장이 특정 용접부의 합격·불합격을 단정하거나 함의하면 NG.\n")
    return (
        "아래 [자료]와 [문장]을 비교한다.\n"
        "문장이 자료에 없는 사실이나 수치를 담고 있으면 NG, 자료 범위 안이면 OK.\n"
        f"{gate}"
        "판정을 OK 또는 NG 한 낱말로만 답한다.\n\n"
        f"[자료]\n{render_basis(rec)}\n\n[문장]\n{rec['text']}"
    )


def prompt_reason(rec: dict) -> str:
    return (
        "아래 [문장]은 [자료] 범위를 벗어난다고 판정됐다.\n"
        "그 이유를 한국어 한 문장으로 쓴다. 문장을 그대로 옮겨 적지 않는다.\n\n"
        f"[자료]\n{render_basis(rec)}\n\n[문장]\n{rec['text']}"
    )


def prompt_qa(passage: str, aspect: str, ask: str) -> str:
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


# ------------------------------------------------------------------ 판정 파싱

_THINK = re.compile("<think>.*?</think>", re.S)
_TAG = re.compile("</?think>")


def parse_judgement(out: str, anchors: list[str]) -> tuple[bool, bool]:
    """판정문에서 OK/NG 를 **앵커**로 뽑는다 (G3-4).

    부분문자열 매칭은 LOOKS·TOKEN·WRONG 같은 낱말을 판정으로 읽는다 (B16). 앵커에
    걸리지 않으면 통과로 세지 않되 '형식 위반'으로 따로 기록해, 검증기 성능과 하네스
    결함을 구분한다.
    """
    if not out or not out.strip():
        return False, False
    body = _TAG.sub(" ", _THINK.sub(" ", out)).strip()
    for pat in anchors:
        m = re.search(pat, body, re.MULTILINE | re.IGNORECASE)
        if m:
            tok = next(g for g in m.groups() if g)
            return tok.upper() == "OK", True
    return False, False


def echo_ratio(reason: str, source: str) -> float:
    """사유가 입력 문장을 얼마나 되풀이했는가 (0~1). 최장 공통 부분열 / 사유 길이."""
    r = "".join(reason.split())
    s = "".join(source.split())
    if not r:
        return 0.0
    match = difflib.SequenceMatcher(None, r, s, autojunk=False).find_longest_match(
        0, len(r), 0, len(s))
    return match.size / len(r)


def clean_reason(out: str, source: str, cfg: dict) -> tuple[str, bool]:
    """기각 사유 정리 + 되풀이 강등 (G3-3).

    파일럿에서 기각 사유 109건 중 97건(89%)이 입력 문장의 되풀이였고 실질 사유는 12건
    이었다. 되풀이를 사유로 세면 "사유가 있다"는 회계가 거짓이 된다.
    """
    body = _TAG.sub(" ", _THINK.sub(" ", out or "")).strip()
    lines = [x.strip() for x in body.splitlines() if x.strip()]
    txt = ""
    for line in lines:
        cleaned = re.sub(r"^\s*(OK|NG)\b[\s.:\-]*", "", line, flags=re.I).strip(" .:-")
        if cleaned:
            txt = cleaned[:200]
            break
    fallback = str(cfg["parsing"]["reason_echo_downgrade_to"])
    if not txt:
        return fallback, True
    if echo_ratio(txt, source) >= float(cfg["parsing"]["reason_echo_threshold"]):
        return fallback, True
    return txt, False

# ------------------------------------------------------------------ 생성

class LoadedModel:
    """모델 1회 적재 컨텍스트.

    **같은 프로세스에서 같은 모델을 두 번 올리면 죽는다** (원 코드 주석의 실측, 재현:
    판정 통과분 25건을 낸 직후 사유 생성용 재적재에서 종료코드 139). 판정과 사유는
    한 번 올린 모델에서 이어서 낸다.
    """

    def __init__(self, model_id: str, four_bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        load_kw: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "cuda:0"}
        if four_bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4")
            load_kw.pop("dtype")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        self.model.eval()

    def run(self, prompts, batch: int, max_new: int, prefill: str = "") -> list[str]:
        import torch

        kw = dict(max_new_tokens=max_new, do_sample=False, temperature=None,
                  top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id)
        out: list[str] = []
        t0 = time.time()
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                                                  tokenize=False,
                                                  add_generation_prompt=True) + prefill
                     for p in chunk]
            ids = self.tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.inference_mode():
                o = self.model.generate(**ids, **kw)
            for j in range(len(chunk)):
                out.append(self.tok.decode(o[j][ids.input_ids.shape[1]:],
                                           skip_special_tokens=True).strip())
            print(f"    {min(i+batch, len(prompts))}/{len(prompts)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return out

    def close(self) -> None:
        import torch

        del self.model
        torch.cuda.empty_cache()

    def __enter__(self) -> "LoadedModel":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def generate(prompts, batch: int, model_id: str, max_new: int,
             four_bit: bool = False, prefill: str = "") -> list[str]:
    """1회성 생성. 같은 모델을 두 번 부를 일이 있으면 `LoadedModel` 을 직접 써라."""
    with LoadedModel(model_id, four_bit) as m:
        return m.run(prompts, batch, max_new, prefill)


def run_judge(records: list[dict], batch: int, cand: dict, cfg: dict) -> None:
    """한 후보 검증기의 이진 판정. **두 후보가 같은 예산·같은 사고 정책을 받는다** (G12-4).

    결과는 `judge_<id>_*` 키로 레코드에 붙는다 — 후보끼리 덮어쓰지 않아야 일치도를
    같은 파일에서 잴 수 있다. 판정과 사유를 한 번 올린 모델에서 이어 낸다.
    """
    budget = cfg["judges"]["budget"]
    four_bit = str(cfg["judges"].get("quantization", "")).lower() == "4bit"
    # 사고 정책은 후보 공통이다. 한쪽만 억제하면 예산이 사실상 달라진다 (파일럿 사고).
    prefill = "<think>\n\n</think>\n\n" if cfg["judges"]["thinking_policy"] == "suppress" else ""
    cid = cand["id"]
    anchors = list(cfg["parsing"]["verdict_anchors"])

    with LoadedModel(cand["model"], four_bit) as m:
        outs = m.run([prompt_judge(r) for r in records], batch,
                     int(budget["max_new_tokens"]), prefill)
        for r, o in zip(records, outs):
            ok, parsed = parse_judgement(o, anchors)
            r[f"judge_{cid}_pass"] = ok
            r[f"judge_{cid}_parse_ok"] = parsed
            r[f"judge_{cid}_note"] = " ".join(o.split())[:200]

        rejected = [r for r in records if not r[f"judge_{cid}_pass"]]
        if rejected:
            outs2 = m.run([prompt_reason(r) for r in rejected], batch,
                          int(budget["reason_max_new_tokens"]), prefill)
            for r, o in zip(rejected, outs2):
                reason, is_echo = clean_reason(o, r["text"], cfg)
                r[f"judge_{cid}_reason"] = reason
                r[f"judge_{cid}_reason_is_echo"] = is_echo


# ------------------------------------------------------------------ 검증
#
# **이 파일에 수치·판정어·표준 정규식은 없다.** 전부 정본 호출이다 (G1-1·G1-2).

def validate_clause(recs: list[dict], verdict_mode: str) -> Any:
    """(c) 축 0단계(수치 잠금) + 1단계(규칙). 자료는 생성 프롬프트와 같은 것을 넘긴다."""
    from corpus.generate import numeric_lock as nl
    from corpus.rules import limits_loader
    from corpus.validate import stage1_rules as s1

    table = limits_loader.load_limits(str(PILOT_CSV), pilot=True)
    label_codes = {r.defect_code for r in table.rows}
    clause_reg = {r.clause_id for r in table.rows}
    for r in recs:
        lock = nl.check_numeric_lock(r["text"], r)
        r["stage0_pass"] = lock.ok
        r["stage0_reasons"] = list(lock.reasons)
        r["stage0_detail"] = list(lock.detail)[:3]
    survivors = [r for r in recs if r["stage0_pass"]]
    return survivors, s1.run_stage1(
        survivors, asset="c", table=table, basis_fn=render_basis,
        label_codes=label_codes, clause_registry=clause_reg,
        inspection_method="RT", verdict_mode=verdict_mode,
    )


def validate_remedy(recs: list[dict]) -> None:
    """(b2) 조치 축. 허용치 행이 없으므로 **수치 일절 금지** + 판정어 금지로 받는다.

    `check_normal_lock` 이 그 검사의 정본이다 — 정상 페어와 요구가 같다(인용할 수치가
    없다). 표준 식별자는 자료가 지시하는 IACS 만 허용한다.
    """
    from corpus.generate import numeric_lock as nl
    from corpus.validate import stage1_rules as s1

    for r in recs:
        lock = nl.check_normal_lock(
            r["text"], defect_lexicon=(), expected_verdict=None,
            allowed_standards=REMEDY_STANDARDS,
        )
        reasons = list(lock.reasons)
        detail = list(lock.detail)
        arts = nl.find_artifacts(r["text"], basis=render_basis(r))
        if arts:
            reasons.append(s1.R_ARTIFACT)
            detail.append(f"생성문 아티팩트: {list(arts)}")
        imply = nl.find_verdict_implying(r["text"])
        if imply:
            reasons.append(s1.R_ARTIFACT)
            detail.append(f"판정 함의 표현: {list(imply)}")
        r["stage0_pass"] = not reasons
        r["stage0_reasons"] = sorted(set(reasons))
        r["stage0_detail"] = detail[:3]


def validate_qa(recs: list[dict], passage_ids) -> Any:
    from corpus.validate import stage1_rules as s1

    return s1.run_stage1(recs, asset="b", passage_ids=passage_ids)


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


# ------------------------------------------------------------------ 보고

def axis_block(name: str, n_in: int, stages: dict, *, validated_by: str,
               measures: list[str], extra: Optional[dict] = None) -> dict:
    """축 보고 블록 (G6-1).

    `validated_by=none` 인 축은 키가 `pass_rate` 가 아니라 **`format_rate`** 다.
    QA 0.94(형식만 잼)와 판정추론 0.345(다른 계열 검증)가 같은 이름으로 나란히 실려
    교차검증을 거친 값처럼 읽힌 것이 파일럿의 사고였다 (B8).
    """
    key = "pass_rate" if validated_by != "none" else "format_rate"
    # 마지막으로 **실제로 돈** 단계의 통과 수가 채택 수다. 미실행 단계(`not_run`)를
    # 그대로 집으면 None 이 종단 통과율의 분자가 된다 (G6-2 의 반대 방향 사고).
    ran = [v for v in stages.values() if v.get("status") == "ran"]
    n_out = ran[-1]["n_pass"] if ran else 0
    block = {
        "n_in": n_in,
        "validated_by": validated_by,
        "measures": measures,
        "stages": stages,
        "n_accepted": n_out,
        f"end_to_end_{key}": round(n_out / n_in, 4) if n_in else None,
    }
    if extra:
        block.update(extra)
    return block


def stage(n_in: int, n_pass: int, reasons: dict, *, ran: bool = True) -> dict:
    """단계 요약. 돌지 않은 단계를 '폐기 0건'으로 렌더하지 않는다 (G6-2)."""
    if not ran:
        return {"n_in": n_in, "n_pass": None, "n_fail": None, "status": "not_run",
                "fail_reasons": {}}
    return {"n_in": n_in, "n_pass": n_pass, "n_fail": n_in - n_pass,
            "pass_rate": round(n_pass / n_in, 4) if n_in else None,
            "fail_reasons": dict(sorted(reasons.items())), "status": "ran"}


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=REPO).stdout.strip()


def write_jsonl(path: Path, rows) -> None:
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(body + ("\n" if rows else ""))


def write_json(path: Path, doc: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")


# ------------------------------------------------------------------ 본체

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-qa", type=int, default=200)
    ap.add_argument("--n-reason", type=int, default=200)
    ap.add_argument("--clause-share", type=float, default=0.6)
    ap.add_argument("--batch", type=int, default=None, help="미지정 시 사전 등록본 값")
    ap.add_argument("--verdict-mode", choices=["full", "clause_only"], default="full",
                    help="(c) 축 판정 모드. full 이 스펙 정본이다 (§4-7)")
    ap.add_argument("--judge-id", default=None,
                    help="사전 등록본의 후보 id. 미지정 시 등록된 후보를 전부 돌린다")
    ap.add_argument("--stage", choices=["generate", "judge", "all"], default="all")
    ap.add_argument("--out", default=None, help="산출 디렉터리 (기본 cycle_pilot)")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(args.out) if args.out else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = args.batch or int(cfg["generation"]["batch_size"])
    mid = out_dir / "_raw_generated.json"

    if args.stage != "judge":
        n_clause = int(args.n_reason * args.clause_share)
        n_remedy = args.n_reason - n_clause
        print(f"[1/5] 골격: (c) {n_clause} + 조치 {n_remedy}  (verdict_mode={args.verdict_mode})")
        recs = build_clause_records(n_clause, args.verdict_mode) + build_remedy_records(n_remedy)

        base = load_passages(args.n_qa)
        passages = []
        for asp, ask in QA_ASPECTS:
            for pp in base:
                if len(passages) >= args.n_qa:
                    break
                passages.append({**pp, "aspect": asp, "ask": ask,
                                 "qa_id": f"{pp['passage_id']}#{asp}"})
        # G11-6: 요청 n ≤ 가용 구절 × 관점 수. 넘으면 같은 구절을 되풀이해 쓴 것이다.
        capacity = len(base) * len(QA_ASPECTS)
        print(f"[2/5] 생성 {len(recs)} + QA {len(passages)} "
              f"(구절 {len(base)} × 관점 {len(QA_ASPECTS)} = 가용 {capacity})")
        prompts = [prompt_generate(r) for r in recs]
        prompts += [prompt_qa(p["text"], p["aspect"], p["ask"]) for p in passages]
        texts = generate(prompts, batch, cfg["generation"]["model"],
                         max_new=int(cfg["generation"]["max_new_tokens"]))
        for r, t in zip(recs, texts[:len(recs)]):
            r["text"] = t

        print("[3/5] (c)·조치 축 0·1단계 (정본 검사기)")
        clause = [r for r in recs if r["axis"] == AXIS_CLAUSE]
        remedy = [r for r in recs if r["axis"] == AXIS_REMEDY]
        clause_survivors, st1 = validate_clause(clause, args.verdict_mode)
        for r, res in zip(clause_survivors, st1.results):
            r["stage1_pass"] = res.ok
            r["stage1_reasons"] = list(res.reasons)
        validate_remedy(remedy)

        print(f"[4/5] QA {len(passages)}건 1단계")
        qa_recs = []
        for p, t in zip(passages, texts[len(recs):]):
            q, a = parse_qa(t)
            qa_recs.append({"sample_id": p["qa_id"], "passage_id": p["passage_id"],
                            "aspect": p["aspect"], "doc": p["doc"], "question": q,
                            "answer": a, "text": t,
                            "evidence_passage_ids": [p["passage_id"]]})
        st1b = validate_qa(qa_recs, {p["passage_id"] for p in base}) if qa_recs else None
        if st1b is not None:
            for r, res in zip(qa_recs, st1b.results):
                r["stage0_pass"] = res.ok
                r["stage0_reasons"] = list(res.reasons)

        payload = {"recs": recs, "qa_recs": qa_recs, "n_base_passages": len(base),
                   "capacity": capacity, "verdict_mode": args.verdict_mode,
                   "batch": batch}
        write_json(mid, payload)
        if args.stage == "generate":
            print(f"생성분 저장: {mid}")
            return
    else:
        payload = json.loads(mid.read_text(encoding="utf-8"))
        recs, qa_recs = payload["recs"], payload["qa_recs"]
        batch = payload.get("batch", batch)

    # ---- 2단계: 다른 계열 판정 (후보 전부, 같은 예산·같은 사고 정책) ----
    clause = [r for r in recs if r["axis"] == AXIS_CLAUSE]
    survivors = [r for r in clause if r.get("stage0_pass") and r.get("stage1_pass")]
    cands = [c for c in cfg["judges"]["candidates"]
             if args.judge_id is None or c["id"] == args.judge_id]
    print(f"[5/5] 2단계 판정 {len(survivors)}건 × 후보 {[c['id'] for c in cands]}")
    for cand in cands:
        if survivors:
            run_judge(survivors, batch, cand, cfg)

    write_report(recs, qa_recs, cands, cfg, payload, out_dir)


def write_report(recs, qa_recs, cands, cfg, payload, out_dir: Path) -> None:
    clause = [r for r in recs if r["axis"] == AXIS_CLAUSE]
    remedy = [r for r in recs if r["axis"] == AXIS_REMEDY]
    s0 = [r for r in clause if r.get("stage0_pass")]
    s1 = [r for r in s0 if r.get("stage1_pass")]

    axes: dict[str, dict] = {}
    stages = {
        "stage0_numeric_lock": stage(
            len(clause), len(s0),
            Counter(x for r in clause for x in r.get("stage0_reasons", []))),
        "stage1_rule": stage(
            len(s0), len(s1),
            Counter(x for r in s0 for x in r.get("stage1_reasons", []))),
    }
    judged: dict[str, dict] = {}
    for cand in cands:
        cid = cand["id"]
        acc = [r for r in s1 if r.get(f"judge_{cid}_pass")]
        n_fmt = sum(1 for r in s1 if r.get(f"judge_{cid}_parse_ok") is False)
        n_echo = sum(1 for r in s1 if r.get(f"judge_{cid}_reason_is_echo"))
        judged[cid] = {
            "model": cand["model"], "family": cand["family"],
            "n_in": len(s1), "n_pass": len(acc), "n_fail": len(s1) - len(acc),
            "pass_rate": round(len(acc) / len(s1), 4) if s1 else None,
            "n_format_violation": n_fmt,
            "n_reason_is_echo": n_echo,
            "budget": dict(cfg["judges"]["budget"]),
            "thinking_policy": cfg["judges"]["thinking_policy"],
            "quantization": cfg["judges"].get("quantization"),
        }
    canonical = cfg["judges"].get("canonical")
    axes[AXIS_CLAUSE] = axis_block(
        AXIS_CLAUSE, len(clause), stages,
        validated_by="cross_family" if judged else "rule",
        measures=["format", "groundedness"],
        extra={"stage2_judges": judged,
               "canonical_judge": canonical,
               "canonical_note": (
                   "정본 미정 — 사람 라벨 100건 비교 전까지 어느 후보도 정본이 아니다."
                   " 후보별 통과율은 judge_agreement 로만 읽는다 (G12-3)."
                   if not canonical else None)})

    axes[AXIS_REMEDY] = axis_block(
        AXIS_REMEDY, len(remedy),
        {"stage0_numeric_lock": stage(
            len(remedy), sum(1 for r in remedy if r.get("stage0_pass")),
            Counter(x for r in remedy for x in r.get("stage0_reasons", []))),
         "stage2_judge": stage(0, 0, {}, ran=False)},
        validated_by="rule", measures=["format"])

    n_qa_pass = sum(1 for r in qa_recs if r.get("stage0_pass"))
    axes["QA"] = axis_block(
        "QA", len(qa_recs),
        {"stage1_rule": stage(len(qa_recs), n_qa_pass,
                              Counter(x for r in qa_recs for x in r.get("stage0_reasons", []))),
         "stage2_judge": stage(0, 0, {}, ran=False)},
        validated_by="none", measures=["format"],
        extra={"n_source_passages": payload.get("n_base_passages"),
               "n_aspects": len(QA_ASPECTS),
               "reuse_factor": round(len(qa_recs) / payload["n_base_passages"], 3)
               if payload.get("n_base_passages") else None,
               "capacity": payload.get("capacity"),
               "note": ("근거성·사실성 검증 없음 — 형식(질문/답 두 줄 + 중복 제거)만 잰 값이다."
                        " pass_rate 로 읽으면 안 된다 (B8)."),
               "passage_screen": "유료 표준 전재 의심 구절 제외"})

    report = {
        "purpose": "합성 corpus 사이클 — 정본 검사기 배선 후 재실행 (체크리스트 19)",
        "verdict_mode": payload.get("verdict_mode"),
        "generation": {**{k: cfg["generation"][k] for k in ("model", "decoding", "max_new_tokens")},
                       "batch_size": payload.get("batch"),
                       "retry": "없음 (실패분 폐기)",
                       "batch_note": "배치는 재현 조건의 일부다 — 좌측 패딩 때문에 배치가"
                                     " 다르면 같은 프롬프트도 다른 문장이 나온다 (B22)."},
        "validator_prereg": {"path": str(CONFIG_PATH.relative_to(REPO)),
                             "version": cfg["version"]},
        "axes": axes,
        "git_commit": git_commit(),
    }
    write_jsonl(out_dir / "reasoning_accepted.jsonl",
                [r for r in clause if r.get("stage1_pass")])
    write_jsonl(out_dir / "remedy_accepted.jsonl",
                [r for r in remedy if r.get("stage0_pass")])
    write_jsonl(out_dir / "qa_accepted.jsonl",
                [r for r in qa_recs if r.get("stage0_pass")])
    write_jsonl(out_dir / "discarded.jsonl",
                [r for r in recs + qa_recs
                 if not (r.get("stage0_pass") and r.get("stage1_pass", True))])
    write_json(out_dir / "cycle_corpus_report.json", report)
    print("\n" + json.dumps(axes, ensure_ascii=False, indent=1))
    print("산출:", out_dir)


if __name__ == "__main__":
    main()
