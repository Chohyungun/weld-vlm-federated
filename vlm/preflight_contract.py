"""진단 실행 계약의 닫힌 꼴. 칸 이름·필수 여부·자료형과 고정 경로는 이곳에서만 정한다."""

from enum import StrEnum

KIND = "run_preflight"
SCHEMA = 1
VERSION = "1"
# C 08 §6-2: 답 길이는 이 값 이하, 실행 생성 한도는 이 값으로 고정한다.
MAX_NEW_TOKENS = 1280


class Code(StrEnum):
    INPUT_SET = "INPUT_SET"
    INPUT_BYTES = "INPUT_BYTES"
    CONTRACT_FORM = "CONTRACT_FORM"
    TABLE_READ = "TABLE_READ"
    TABLE_EMPTY = "TABLE_EMPTY"
    ENV_MISMATCH = "ENV_MISMATCH"
    APPROVAL_BINDING = "APPROVAL_BINDING"
    HELDOUT_BINDING = "HELDOUT_BINDING"
    INIT_MISMATCH = "INIT_MISMATCH"
    PROMPT_MISMATCH = "PROMPT_MISMATCH"
    PAIRS_BINDING = "PAIRS_BINDING"
    C08_BINDING = "C08_BINDING"
    CODE_MISMATCH = "CODE_MISMATCH"
    PATH_UNSAFE = "PATH_UNSAFE"
    STALE_RECEIPT = "STALE_RECEIPT"
    IO_ERROR = "IO_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"


# dict = 정확한 필수 키 집합, list = 행 꼴, tuple = 허용 자료형/꼴의 합집합.
REF = {"path": "path", "sha256": "sha256"}
MEMBER = {"path": "path", "bytes": "uint", "sha256": "sha256"}
MODEL_ROW = {"name": "path", "bytes": "uint", "sha256": "sha256"}
REGIME = {"recipe": "text", "prompt_file": "path", "prompt_sha256": "sha256",
          "max_new_tokens": "positive_int", "vectors_sha256": "sha256"}
CAPS = {"a2": "positive_number", "a3_05": "positive_number", "recover": "positive_number"}
CONTRACT = {
    "sha": "commit", "bundle": {"name": "path", "sha256": "sha256", "ref": "text"},
    "inputs": {"tar_sha256": "sha256", "members_sha256": "sha256", "count": "uint"},
    "pairs_v2": {"folder": "path", "digest": "sha256"},
    "init_adapter": {"npz_sha256": "sha256", "proof_sha256": "sha256", "digest": "sha256"},
    "a2_rows": "sha256", "regime": REGIME,
    "c08": {k: REF for k in ("measure", "token_table", "token_provenance", "narrative_inputs")},
    "model_manifest": REF,
    "a3": {"rows_sha256": "sha256", "threshold": "number", "unexposed_path": "text",
           "heldout_list": ("sha256", "null")},
    "caps_s": CAPS, "caps_cu": ("positive_number", "null"), "gpu_expect": "text",
    "env_expect": {"uv_lock_sha256": "sha256", "pyproject_sha256": "sha256", "torch": "text",
                   "transformers": "text", "bitsandbytes": "text", "gpu_name_has": "text"},
    "approved_by": {"record": REF, "decision_log": {"commit": "commit", "path": "path",
                   "section": "text", "section_sha256": "sha256"}},
}
# 계약의 외부 칸 이름만 바꿀 때는 이 매핑 한 곳을 고친다.
# 키는 위 내부 경로, 값은 외부 JSON 키다. 검사 구현·승인 target은 정규화된 내부 이름을 쓴다.
# 예: {("c08", "measure"): "sealed_measure"}. 필수 여부와 자료형은 CONTRACT 한 곳에 있다.
WIRE_NAMES: dict[tuple[str, ...], str] = {}


def contract_form(form=None, prefix=()):
    """외부 이름을 반영한 정확한 계약 스키마. 이름 충돌도 거부한다."""
    form = CONTRACT if form is None else form
    if not isinstance(form, dict):
        return form
    result = {}
    for key, value in form.items():
        path = (*prefix, key)
        wire = WIRE_NAMES.get(path, key)
        if not isinstance(wire, str) or not wire or wire in result:
            raise ValueError("계약 외부 이름이 비었거나 충돌한다")
        result[wire] = contract_form(value, path) if isinstance(value, dict) else value
    return result


def canonical_contract(value, form=None, prefix=()):
    """닫힌 외부 스키마 검증 뒤 내부 이름으로 읽는다. 원본 바이트·해시는 고치지 않는다."""
    form = CONTRACT if form is None else form
    return {key: canonical_contract(value[WIRE_NAMES.get((*prefix, key), key)], shape, (*prefix, key))
            if isinstance(shape, dict) else value[WIRE_NAMES.get((*prefix, key), key)]
            for key, shape in form.items()}


APPROVAL_TARGET = {k: CONTRACT[k] for k in ("sha", "regime", "pairs_v2", "a3", "caps_s", "caps_cu")}
APPROVAL = {"original": "text", "at": "text", "target": APPROVAL_TARGET}
APPROVAL_SECTION = {"record_sha256": "sha256", "approval": APPROVAL}
CHECK_NAMES = ("inputs", "tables", "nonempty", "environment", "approval", "heldout", "init", "prompt", "pairs")
CHECK_KEYS = frozenset({"ok", "codes", "evidence"})
RECEIPT_KEYS = frozenset({"kind", "schema", "checker_version", "contract_sha256", "inputs", "checks", "all_ok", "at"})
CONTEXT_KEYS = frozenset({"inputs", "checkout", "model_cache", "environment"})
ENV_REQUIRED = ("session", "cuda_ready", "uv_lock_sha256", "torch", "transformers", "bitsandbytes", "gpu_name")
PACKAGES = ("torch", "transformers", "tokenizers", "peft", "bitsandbytes", "accelerate", "flwr")
MEASURE_CHECKS = (
    "all_ok", "box_close_same_across_forms", "encode_C_all_match", "ga_json_close_is_last_token",
    "na_json_close_equals_v1_length", "rows_equal_pairs_meta", "template_B_all_match",
    "v1_f_1_8_cells_equal_07_recount", "v1_f_1_clients_equal_base_yaml_expected", "v1_json_close_is_last_token",
)
PATHS = {
    "model_manifest": "docs/dev_log/2026-10-16-본실험실행/transfer_manifest_C.json",
    "archive": "inputs.tar", "members": "MEMBERS.json", "adapter": "initial.npz",
    "proof": "initial.proof.json", "a2": "g1a1_confirmed.txt", "heldout": "dev_slice_v2_tune.list",
    "manifest": "data/interim/manifest_v1/manifest.csv",
    "train_config": "configs/registration/main-20261017-1.train_config.json",
    "registration": "configs/train_registration.yaml", "config": "configs/base.yaml",
    "vectors": "evaluation/vectors/narrative_extract.v1.json",
    "lock": "uv.lock", "project": "pyproject.toml",
}


def references(c):
    """계약 참조를 (뿌리, 상대 경로, 원시 바이트 SHA)로 펼친다. 합성 digest는 파일 SHA가 아니다."""
    result = {
        "bundle": ("inputs", c["bundle"]["name"], c["bundle"]["sha256"]),
        "archive": ("inputs", PATHS["archive"], c["inputs"]["tar_sha256"]),
        "members": ("inputs", PATHS["members"], c["inputs"]["members_sha256"]),
        "adapter": ("inputs", PATHS["adapter"], c["init_adapter"]["npz_sha256"]),
        "proof": ("inputs", PATHS["proof"], c["init_adapter"]["proof_sha256"]),
        "a2": ("inputs", PATHS["a2"], c["a2_rows"]),
        "prompt": ("checkout", c["regime"]["prompt_file"], c["regime"]["prompt_sha256"]),
        "vectors": ("checkout", PATHS["vectors"], c["regime"]["vectors_sha256"]),
        "lock": ("checkout", PATHS["lock"], c["env_expect"]["uv_lock_sha256"]),
        "project": ("checkout", PATHS["project"], c["env_expect"]["pyproject_sha256"]),
    }
    for key, ref in {**c["c08"], "model_manifest": c["model_manifest"]}.items():
        result[key] = ("checkout", ref["path"], ref["sha256"])
    ref = c["approved_by"]["record"]
    result["approval"] = ("inputs", ref["path"], ref["sha256"])
    if c["a3"]["heldout_list"] is not None:
        result["heldout"] = ("inputs", PATHS["heldout"], c["a3"]["heldout_list"])
    return result
