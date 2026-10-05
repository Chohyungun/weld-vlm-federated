"""통합형 설정 키 읽기 — `configs/base.yaml` 의 `fixed_before_main_runs.uni_*` 와 등록 시드.

통합형 학습·생성 코드는 모델 판 · 페어 · 프롬프트 · 템플릿 인자 · 예산 · 시드를 **여기서만** 읽는다.
모듈 상수(`vlm/pilot_vlm.py` 의 `MODEL_ID`·`PAIRS_PATH`)는 파일럿 재현용 기본값이고 본실험은 쓰지 않는다.
예산이나 시드 수가 바뀌어도 코드를 다시 짜지 않게 하려는 것이다.

## `null` 은 아직 정하지 않은 값이다

설정 파일은 정하지 않은 값을 `null` 로 둔다. 그 값이 필요한 단계는 **`null` 인 채로 시작하지 않는다** —
`UniConfig.require(...)` 가 빠진 키의 이름을 전부 모아 멈춘다. 짐작한 기본값으로 채우지 않는다.

## 학습량의 출처는 목적마다 하나다

본실험(`main`)은 `uni_train_budget` 을 읽는다. 리허설과 진단은 계획 파일의 `amount` 를 읽는다 —
그 경로는 계획 파일을 읽는 쪽이 따로 만든다. 이 모듈은 본실험 키만 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

# 좌표 규약의 값 공간은 `vlm/coords.py` 의 것을 그대로 쓴다 — 목록을 두 곳에 두지 않는다.
from vlm.coords import COORD_SPACES

__all__ = ["BASE_CONFIG", "UniConfig", "UniConfigIncomplete", "load_uni_config", "uni_config_from",
           "main_run_problems", "pairs_snapshot_problems", "NON_MAIN_PARENTS", "REPO_ROOT"]

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/base.yaml")

#: 본실험이 쓰거나 읽지 않는 부모 — 리허설 · 진단 루트의 부모(리허설 2판 §5-1)와 파일럿 출력 루트
#: (`pyproject.toml` 의 `project` 기본값). 작업 트리 루트 기준이다.
NON_MAIN_PARENTS: tuple[Path, ...] = (Path("outputs/rehearsal_u"), Path("outputs/pilot_c"))
#: 감독 토큰 기대값의 키 — 참여자 셋이 **정확히** 있어야 한다. 중앙은 셋의 합이다.
TOKEN_KEYS: tuple[str, ...] = ("C1", "C2", "C3")


class UniConfigIncomplete(ValueError):
    """필요한 설정 키가 비어 있거나 꼴이 틀렸다. 빠진 키의 이름을 전부 든다."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("통합형 설정이 이 단계에 쓸 수 없다: " + " · ".join(self.problems))


@dataclass(frozen=True)
class UniConfig:
    """통합형 본실험 설정. 값이 `None` 이면 설정 파일에서 아직 정하지 않았다는 뜻이다."""

    model_id: str | None
    model_revision: str | None
    pairs_path: str | None
    pairs_digest: str | None
    chat_template_kwargs: dict[str, Any] | None
    max_new_tokens: int | None
    batch_size: int | None
    num_rounds: int | None
    local_epochs: int | None
    total_epochs: int | None
    seeds: tuple[int, ...]
    snapshot_digest: str | None
    processor_kwargs: dict[str, Any] | None = None
    """학습 쪽 프로세서를 여는 설정(`min_pixels` · `max_pixels` 등). 본실험은 이 값으로만 연다 — 전달값과 맞댄다.
    키 이름 `fixed_before_main_runs.uni_processor_kwargs` 는 학습 쪽의 제안이다. 설정에 없으면 본실험은 시작하지 않는다."""
    expected_supervised_tokens: dict[str, int] | None = None
    """참여자별 epoch 당 감독 토큰 기대값. 키 이름 `fixed_before_main_runs.uni_expected_supervised_tokens` 는 제안이다.
    키는 `TOKEN_KEYS` 셋이 정확히 있어야 한다 — 빠진 참여자가 학습을 다 돈 뒤에야 거부되지 않게 읽을 때 본다."""
    prompt_path: str | None = None
    """학습 프롬프트 파일(작업 트리 루트 기준). 키 이름 `fixed_before_main_runs.uni_prompt_path` 는 제안이다.
    프롬프트의 해시는 설정에 두지 않는다 — 등록 단계가 이 파일의 문자열로 낸다(12번 정정 155행)."""
    coord_space: str | None = None
    """학습 타깃 · 생성의 좌표 규약. 키 이름 `fixed_before_main_runs.uni_coord_space` 는 제안이다."""
    source: str = ""
    """읽은 파일의 경로. 산출물이 어느 설정에서 나왔는지 적는 자리다."""

    #: 단계마다 비어 있으면 안 되는 칸. 학습은 모델 · 페어 · 템플릿 · 예산 · 시드 · 스냅샷이 선 뒤에 돈다.
    NEEDS = {
        "train": ("model_id", "model_revision", "pairs_path", "pairs_digest", "chat_template_kwargs",
                  "num_rounds", "local_epochs", "total_epochs", "seeds", "snapshot_digest",
                  "processor_kwargs", "expected_supervised_tokens", "prompt_path", "coord_space"),
    }

    def problems(self, stage: str = "train") -> list[str]:
        """이 단계에 쓸 수 없는 까닭. 비었으면 쓸 수 있다."""
        if stage not in self.NEEDS:
            raise ValueError(f"모르는 단계: {stage!r} — {sorted(self.NEEDS)}")
        out = [f"{name} 이 비어 있다" for name in self.NEEDS[stage]
               if getattr(self, name) in (None, (), "")]
        n, r, e = self.total_epochs, self.num_rounds, self.local_epochs
        if None not in (n, r, e) and n != r * e:
            out.append(f"total_epochs {n} ≠ num_rounds {r} × local_epochs {e}")
        return out

    def require(self, stage: str = "train") -> "UniConfig":
        bad = self.problems(stage)
        if bad:
            raise UniConfigIncomplete(bad)
        return self


def _int_or_none(v: Any, name: str, problems: list[str]) -> int | None:
    if v is None:
        return None
    # 참·거짓은 정수가 아니다 — YAML 의 `yes` 가 조용히 1 이 되는 길을 막는다.
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        problems.append(f"{name} 이 0 이상의 정수가 아니다: {v!r}")
        return None
    return int(v)


def _str_or_none(v: Any, name: str, problems: list[str]) -> str | None:
    if v is None:
        return None
    if not isinstance(v, str) or not v:
        problems.append(f"{name} 이 비지 않은 문자열이 아니다: {v!r}")
        return None
    return v


def uni_config_from(doc: Mapping[str, Any], *, source: str = "") -> UniConfig:
    """읽은 YAML 문서에서 설정을 만든다. **꼴이 틀린 값은 여기서 멈춘다**(빈 값은 멈추지 않는다)."""
    fixed = doc.get("fixed_before_main_runs") or {}
    problems: list[str] = []
    budget = fixed.get("uni_train_budget") or {}
    model = fixed.get("uni_model") or {}
    pairs = fixed.get("uni_pairs") or {}
    kwargs = fixed.get("uni_chat_template_kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        problems.append(f"uni_chat_template_kwargs 가 사전이 아니다: {kwargs!r}")
        kwargs = None
    proc_kw = fixed.get("uni_processor_kwargs")
    if proc_kw is not None and not isinstance(proc_kw, dict):
        problems.append(f"uni_processor_kwargs 가 사전이 아니다: {proc_kw!r}")
        proc_kw = None
    exp_tok = fixed.get("uni_expected_supervised_tokens")
    if exp_tok is not None and (not isinstance(exp_tok, dict) or any(
            not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int) or v < 0
            for k, v in exp_tok.items())):
        problems.append(f"uni_expected_supervised_tokens 가 참여자 → 0 이상 정수의 사전이 아니다: {exp_tok!r}")
        exp_tok = None
    if exp_tok is not None and sorted(exp_tok) != sorted(TOKEN_KEYS):
        # 빈 사전도 여기서 멈춘다 — 빠진 참여자는 그 칸의 학습이 끝난 뒤에야 드러난다.
        problems.append(f"uni_expected_supervised_tokens 의 참여자 {sorted(exp_tok)} ≠ {list(TOKEN_KEYS)}")
        exp_tok = None
    if fixed.get("uni_prompt_sha256") is not None:
        # 12번 정정(155행)이 정본이다 — 두지 않는 키다. 채워 두면 등록 단계의 값과 두 출처가 된다.
        problems.append("uni_prompt_sha256 은 두지 않는다 — 등록 단계가 프롬프트 문자열의 UTF-8 해시로 낸다")
    coord = _str_or_none(fixed.get("uni_coord_space"), "uni_coord_space", problems)
    if coord is not None and coord not in COORD_SPACES:
        problems.append(f"uni_coord_space 가 {list(COORD_SPACES)} 가운데 하나가 아니다: {coord!r}")
        coord = None
    seeds_raw = (doc.get("experiment") or {}).get("seeds") or ()
    if not isinstance(seeds_raw, (list, tuple)) or any(
            isinstance(s, bool) or not isinstance(s, int) for s in seeds_raw):
        problems.append(f"experiment.seeds 가 정수 목록이 아니다: {seeds_raw!r}")
        seeds_raw = ()
    if len(set(seeds_raw)) != len(seeds_raw):
        problems.append(f"experiment.seeds 에 같은 값이 있다: {list(seeds_raw)}")
    cfg = UniConfig(
        model_id=_str_or_none(model.get("id"), "uni_model.id", problems),
        model_revision=_str_or_none(model.get("revision"), "uni_model.revision", problems),
        pairs_path=_str_or_none(pairs.get("path"), "uni_pairs.path", problems),
        pairs_digest=_str_or_none(pairs.get("digest"), "uni_pairs.digest", problems),
        chat_template_kwargs=dict(kwargs) if kwargs is not None else None,
        max_new_tokens=_int_or_none(fixed.get("uni_max_new_tokens"), "uni_max_new_tokens", problems),
        batch_size=_int_or_none(fixed.get("uni_batch_size"), "uni_batch_size", problems),
        num_rounds=_int_or_none(budget.get("num_rounds"), "uni_train_budget.num_rounds", problems),
        local_epochs=_int_or_none(budget.get("local_epochs"), "uni_train_budget.local_epochs", problems),
        total_epochs=_int_or_none(budget.get("total_epochs"), "uni_train_budget.total_epochs", problems),
        seeds=tuple(int(s) for s in seeds_raw),
        snapshot_digest=_str_or_none(fixed.get("snapshot_digest"), "snapshot_digest", problems),
        processor_kwargs=None if proc_kw is None else dict(proc_kw),
        expected_supervised_tokens=None if exp_tok is None else {str(k): int(v) for k, v in exp_tok.items()},
        prompt_path=_str_or_none(fixed.get("uni_prompt_path"), "uni_prompt_path", problems),
        coord_space=coord,
        source=source,
    )
    if problems:
        raise UniConfigIncomplete(problems)
    return cfg


def load_uni_config(path: str | Path | None = None) -> UniConfig:
    """설정 파일을 읽는다. 빈 값은 그대로 두고, 쓰는 단계가 `require()` 로 거른다.

    경로가 없으면 **작업 트리 루트의** `configs/base.yaml` 이다 — 작업 폴더에 기대지 않는다.
    """
    p = Path(path) if path is not None else REPO_ROOT / BASE_CONFIG
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return uni_config_from(doc, source=str(p))


def pairs_snapshot_problems(pairs_path: str | Path, want_digest: str | None) -> list[str]:
    """페어 스냅샷의 계약(`SNAPSHOT.sha256`)을 **페어 파일과 함께** 맞댄다 — 경로가 아니라 내용이다.

    계약의 꼴은 `<sha256>  <이름>` 줄들과 `# snapshot_digest <digest>` 한 줄이고, digest 는 적힌 파일 해시 문자열을
    이어 붙인 것의 sha256 이다(`corpus/generate/make_pairs_pilot.snapshot_digest`). 셋을 본다 — 계약의 digest 가
    설정의 값과 같다, 적힌 파일마다 해시가 실물과 같다(페어 파일이 그 안에 있다), digest 를 다시 내면 같다.
    """
    import hashlib

    pp = Path(pairs_path)
    contract = pp.parent / "SNAPSHOT.sha256"
    if not contract.exists():
        return [f"페어 스냅샷의 계약이 없다: {contract}"]
    lines = contract.read_text(encoding="utf-8").splitlines()
    digest = next((ln.split()[-1] for ln in lines if ln.startswith("# snapshot_digest")), None)
    entries = [ln.split(None, 1) for ln in lines if ln.strip() and not ln.startswith("#")]
    out: list[str] = []
    if digest is None:
        out.append(f"계약에 snapshot_digest 줄이 없다: {contract}")
    elif digest != want_digest:
        out.append(f"페어 스냅샷 digest {digest} ≠ 설정 {want_digest}")
    names = [e[1].strip() for e in entries if len(e) == 2]
    if pp.name not in names:
        out.append(f"계약에 페어 파일 {pp.name} 이 없다")
    for e in entries:
        if len(e) != 2:
            out.append(f"계약의 줄 꼴이 틀리다: {e}")
            continue
        h, name = e[0], e[1].strip()
        f = pp.parent / name
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            out.append(f"계약의 파일 해시가 실물과 다르다: {name}")
    if digest is not None and hashlib.sha256("".join(e[0] for e in entries).encode()).hexdigest() != digest:
        out.append("계약의 digest 를 다시 내면 적힌 값과 다르다")
    return out


def main_run_problems(cfg: UniConfig, *, model_id: str | None, model_revision: str | None,
                      pairs_path: str | None, pairs_digest: str | None,
                      chat_template_kwargs: Mapping[str, Any] | None,
                      num_rounds: int, local_epochs: int, total_epochs: int,
                      seed_index: int | None, seed_value: int, snapshot_digest: str,
                      prompt_path: str | None, coord_space: str | None, output_paths: list[Path],
                      processor_kwargs: Mapping[str, Any] | None = None,
                      expected_supervised_tokens: Mapping[str, int] | None = None,
                      non_main_parents: tuple[Path, ...] | None = None) -> list[str]:
    """본실험 실행의 값이 설정 파일과 같은지 — **모델을 올리기 전에** 부른다. 빈 목록이면 시작해도 된다.

    실행 설정에 남은 옛 기본값(파일럿의 R/E/N · 시드 · 출력 루트)이 본실험 목적과 함께 쓰이는 길을 막는다.
    설정 파일 자체가 비어 있으면(`require("train")`) 그 까닭을 먼저 낸다.
    """
    out = list(cfg.problems("train"))
    if out:
        return out

    def same(name: str, got: Any, want: Any) -> None:
        if got != want:
            out.append(f"{name} {got!r} ≠ 설정 {want!r}")

    same("model_id", model_id, cfg.model_id)
    same("model_revision", model_revision, cfg.model_revision)
    same("pairs_path", None if pairs_path is None else Path(pairs_path).as_posix(), Path(cfg.pairs_path).as_posix())
    same("pairs_digest", pairs_digest, cfg.pairs_digest)
    if pairs_path is not None:
        out.extend(pairs_snapshot_problems(pairs_path, cfg.pairs_digest))
    same("chat_template_kwargs", None if chat_template_kwargs is None else dict(chat_template_kwargs),
         cfg.chat_template_kwargs)
    same("budget", (num_rounds, local_epochs, total_epochs), (cfg.num_rounds, cfg.local_epochs, cfg.total_epochs))
    same("snapshot_digest", snapshot_digest, cfg.snapshot_digest)
    # 프로세서 설정과 감독 토큰 기대값도 **등록된 값**이어야 한다 — 전달받은 값이 설정 파일의 것인지 맞댄다.
    same("processor_kwargs", None if processor_kwargs is None else dict(processor_kwargs), cfg.processor_kwargs)
    same("expected_supervised_tokens",
         None if expected_supervised_tokens is None else {str(k): int(v) for k, v in expected_supervised_tokens.items()},
         cfg.expected_supervised_tokens)
    if seed_index is None or not 1 <= seed_index <= len(cfg.seeds):
        out.append(f"seed_index {seed_index!r} 가 등록 시드표 1..{len(cfg.seeds)} 밖이다")
    else:
        same("seed_value", seed_value, cfg.seeds[seed_index - 1])
    # 프롬프트 · 좌표 규약도 설정에서만 온다 — 모듈 상수 · 기본값으로 내려가지 않는다(검수 14번 M-1).
    same("prompt_path", None if prompt_path is None else Path(prompt_path).as_posix(), Path(cfg.prompt_path).as_posix())
    same("coord_space", coord_space, cfg.coord_space)
    parents = [(REPO_ROOT / q).resolve() for q in (non_main_parents or NON_MAIN_PARENTS)]
    for p in output_paths:
        rp = Path(p).resolve()
        for q in parents:
            if rp == q or rp.is_relative_to(q):
                out.append(f"본실험 출력 {p} 이 비본실험 부모 {q} 아래다")
    return out
