"""진단 실행 직전의 읽기 전용 대조. GPU·학습·생성·전송·봉인·승인 발급은 하지 않는다."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from vlm import preflight_contract as S
from vlm.preflight_contract import KIND, SCHEMA, VERSION, Code

__all__ = ["contract_view", "inside", "read", "sha", "validate"]

ROOT = Path(__file__).resolve().parents[1]
# scripts는 __init__.py가 없는 이름공간 패키지다.
REPO_PACKAGES = ("vlm", "evaluation", "data", "scripts", "fl", "rag", "tracking", "corpus", "detection")


class Rejected(ValueError):
    def __init__(self, code, detail):
        self.code = Code(code)
        super().__init__(str(detail))


def require(condition, code, detail):
    if not condition:
        raise Rejected(code, detail)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"중복 JSON 키: {key}")
        out[key] = value
    return out


def loads(raw):
    def constant(value):
        raise ValueError(f"JSON 유한수 아님: {value}")

    return json.loads(raw, object_pairs_hook=_object, parse_constant=constant)


def read(path):
    return loads(Path(path).read_bytes())


def relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name
            and not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).is_absolute()
            and all(p not in ("", ".", "..") for p in name.split("/")), Code.PATH_UNSAFE, f"상대 경로 위반: {name!r}")
    return name


def inside(root, name):
    relative(name)
    root = Path(root).resolve()
    p = root
    for part in name.split("/"):
        p /= part
        require(not p.is_symlink(), Code.PATH_UNSAFE, f"링크 입력: {p}")
    require(p.resolve().is_relative_to(root), Code.PATH_UNSAFE, f"뿌리 이탈: {name}")
    return p


def validate(value, form, where="계약"):
    if isinstance(form, dict):
        require(type(value) is dict and value.keys() == form.keys(), Code.CONTRACT_FORM, f"{where}: 정확한 키 집합 필요")
        for key, shape in form.items():
            validate(value[key], shape, f"{where}.{key}")
        return
    if isinstance(form, list):
        require(type(value) is list, Code.CONTRACT_FORM, f"{where}: 목록 필요")
        for i, row in enumerate(value):
            validate(row, form[0], f"{where}[{i}]")
        return
    if isinstance(form, tuple):
        for shape in form:
            try:
                validate(value, shape, where)
                return
            except Rejected as exc:
                if exc.code == Code.PATH_UNSAFE:
                    raise
        raise Rejected(Code.CONTRACT_FORM, f"{where}: 허용 자료형 아님")
    predicates = {
        "text": lambda: isinstance(value, str) and bool(value.strip()),
        "path": lambda: bool(relative(value)),
        "sha256": lambda: isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value),
        "commit": lambda: isinstance(value, str) and re.fullmatch("[0-9a-f]{40}", value),
        "uint": lambda: type(value) is int and value >= 0,
        "positive_int": lambda: type(value) is int and value > 0,
        "number": lambda: type(value) in (int, float),
        "positive_number": lambda: type(value) in (int, float) and math.isfinite(value) and value > 0,
        "null": lambda: value is None,
    }
    require(predicates[form](), Code.CONTRACT_FORM, f"{where}: {form} 필요")


def contract_view(contract):
    """외부 이름을 한 스키마로 검증한 뒤 판정용 이름으로 정규화한다."""
    validate(contract, S.contract_form())
    return S.canonical_contract(contract)


def expected_inputs(contract):
    return {key: row[2] for key, row in S.references(contract_view(contract)).items()}


def _git(root, *args, input_bytes=None):
    r = subprocess.run(["git", "-c", f"safe.directory={Path(root).as_posix()}", "-C", str(root), *args],
                       input=input_bytes, capture_output=True, check=False)
    require(r.returncode == 0, Code.CODE_MISMATCH, f"git {args[0]} 대조 실패: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout


def _literal(path, name):
    """GPU 모듈을 import/실행하지 않고 정본의 리터럴 상수만 판독한다. 알 수 없는 식은 거부한다."""
    nodes = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node.value
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node.value

    def resolve(node, seen):
        if isinstance(node, ast.Name):
            if node.id in seen:
                raise ValueError("순환 상수")
            return resolve(nodes[node.id], seen | {node.id})
        if isinstance(node, ast.Dict):
            return {resolve(k, seen): resolve(v, seen) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            return [resolve(v, seen) for v in node.elts]
        return ast.literal_eval(node)

    return resolve(nodes[name], {name})


def current_session():
    """코랩 런타임의 boot_id. 호스트 이름만 같은 새 세션을 옛 세션으로 받지 않는다."""
    value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    require(bool(value), Code.ENV_MISMATCH, "세션 식별자가 비었다")
    return value


def _section(raw, heading):
    text = raw.decode("utf-8").replace("\r\n", "\n")
    lines = text.splitlines(keepends=True)
    indexes = [i for i, line in enumerate(lines) if line.rstrip("\n") == heading]
    require(len(indexes) == 1 and re.match(r"^#{1,6} ", heading), Code.APPROVAL_BINDING, "승인 절이 없거나 중복이다")
    level = len(heading.split(" ", 1)[0])
    start, end = indexes[0], len(lines)
    for i in range(start + 1, len(lines)):
        if re.match(r"^#{" + str(level) + r"} ", lines[i]):
            end = i
            break
    return "".join(lines[start:end])


class Inspector:
    def __init__(self, contract, *, inputs, checkout, model_cache, environment):
        self.cp = Path(contract).resolve()
        self.roots = {"inputs": Path(inputs).resolve(), "checkout": Path(checkout).resolve(),
                      "model_cache": Path(model_cache).resolve(), "environment": Path(environment).resolve()}
        self.raw = self.cp.read_bytes()
        self.c = contract_view(loads(self.raw))
        self.refs = S.references(self.c)
        self.evidence = {}

    def file(self, root, name):
        return inside(self.roots[root], name)

    def ref(self, key):
        root, name, expected = self.refs[key]
        p = self.file(root, name)
        measured = sha(p)
        self.evidence[key] = {"path": str(p), "sha256": measured}
        require(measured == expected, Code.INPUT_BYTES, f"{key}: 계약 바이트 불일치")
        return p

    def records(self):
        members = read(self.ref("members"))
        validate(members, [S.MEMBER], "MEMBERS")
        require(self.c["model_manifest"]["path"] == S.PATHS["model_manifest"], Code.TABLE_READ, "모델 명부는 정본 transfer_manifest의 model 절이어야 한다")
        model = read(self.ref("model_manifest"))["model"]
        validate({k: model[k] for k in ("id", "revision", "files")},
                 {"id": "text", "revision": "text", "files": [S.MODEL_ROW]}, "model")
        for rows, key in ((members, "path"), (model["files"], "name")):
            require(len({r[key] for r in rows}) == len(rows), Code.TABLE_READ, "명부의 중복 행")
        return members, model

    def code(self):
        root = self.roots["checkout"]
        require(ROOT.resolve() == root, Code.CODE_MISMATCH, "검사기를 지정 체크아웃에서 불러오지 않았다")
        c = self.c
        head = _git(root, "rev-parse", "HEAD").decode().strip()
        require(head == c["sha"], Code.CODE_MISMATCH, "체크아웃 HEAD가 실행 계약과 다르다")
        files = _git(root, "ls-tree", "-rz", "--full-tree", "HEAD").split(b"\0")
        require(len(files) > 1, Code.CODE_MISMATCH, "코드 트리가 비었다")
        _git(root, "diff", "--no-ext-diff", "--exit-code", "HEAD", "--")
        tracked = []
        for record in filter(None, files):
            attrs, name = record.split(b"\t", 1)
            mode, kind, blob = attrs.split()
            rel = name.decode("utf-8")
            require(mode in (b"100644", b"100755") and kind == b"blob", Code.PATH_UNSAFE, "코드 링크/서브모듈 불허")
            require(self.file("checkout", rel).is_file(), Code.CODE_MISMATCH, f"추적 파일 부재: {rel}")
            tracked.append((rel, blob))
        # assume-unchanged/skip-worktree에 기대어 바이트 대조를 건너뛰지 않는다.
        raw = _git(root, "cat-file", "--batch", input_bytes=b"".join(blob + b"\n" for _, blob in tracked))
        offset = 0
        for rel, blob in tracked:
            end = raw.index(b"\n", offset)
            got_blob, kind, size = raw[offset:end].split()
            require(got_blob == blob and kind == b"blob", Code.CODE_MISMATCH, "커밋 blob 판독 불일치")
            expected = raw[end + 1:end + 1 + int(size)]
            offset = end + int(size) + 2
            actual = self.file("checkout", rel).read_bytes()
            require(actual.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n"), Code.CODE_MISMATCH, f"추적 파일 바이트 변경: {rel}")
        # 다른 체크아웃에서 불러온 검사기가 지정 체크아웃의 코드인 척하지 않는다.
        for name, module in tuple(sys.modules.items()):
            p = getattr(module, "__file__", None)
            if name.split(".")[0] not in REPO_PACKAGES or not p or not str(p).endswith(".py"):
                continue
            source = Path(p).resolve()
            require(source.is_relative_to(root), Code.CODE_MISMATCH, f"다른 뿌리에서 적재된 저장소 코드: {name}")
            rel = source.relative_to(root).as_posix()
            require(rel in {path for path, _ in tracked}, Code.CODE_MISMATCH, f"추적되지 않은 저장소 코드: {name}")
            q = self.file("checkout", rel)
            require(q.read_bytes().replace(b"\r\n", b"\n") == source.read_bytes().replace(b"\r\n", b"\n"),
                    Code.CODE_MISMATCH, f"적재된 검사 코드가 체크아웃과 다르다: {name}")
        self.evidence["code"] = {"head": head, "tracked_files": len(files) - 1}

    def inputs(self):
        self.code()
        for key in self.refs:
            self.ref(key)
        members, model = self.records()
        heads = _git(self.roots["checkout"], "bundle", "list-heads", str(self.ref("bundle"))).decode("utf-8").splitlines()
        ref = self.c["bundle"]["ref"]
        allowed = {ref, "refs/heads/" + ref, "refs/tags/" + ref}
        require(any(line.split() == [self.c["sha"], name] for line in heads for name in allowed),
                Code.CODE_MISMATCH, "꾸러미 ref가 계약의 실행 SHA를 가리키지 않는다")
        require(len(members) == self.c["inputs"]["count"], Code.INPUT_SET, "명부 개수 불일치")
        wanted = {r["path"]: r for r in members}
        seen = set()
        with tarfile.open(self.ref("archive"), "r:*") as archive:
            for member in archive:
                name = relative(member.name)
                require(member.isfile() and not member.issym() and not member.islnk(), Code.PATH_UNSAFE, "압축본 링크/특수 구성원")
                require(name not in seen, Code.PATH_UNSAFE, f"압축본 중복 구성원: {name}")
                seen.add(name)
                require(name in wanted, Code.INPUT_SET, f"압축본 미등록 구성원: {name}")
                row = wanted[name]
                with archive.extractfile(member) as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                p = self.file("checkout", name)
                require(member.size == row["bytes"] == p.stat().st_size and digest == row["sha256"] == sha(p),
                        Code.INPUT_BYTES, f"압축본/풀린 파일 대조 실패: {name}")
        require(seen == set(wanted), Code.INPUT_SET, "압축본 정확한 집합 불일치")
        # 명부에 포함된 자료 폴더의 파일도 빠짐없이 맞댄다. 체크아웃의 코드/승인 산출은 이 집합에 섞지 않는다.
        for parent in {str(PurePosixPath(n).parent) for n in wanted}:
            folder = self.file("checkout", parent)
            actual = {p.relative_to(self.roots["checkout"]).as_posix() for p in folder.iterdir() if p.is_file() or p.is_symlink()}
            require(actual == {n for n in wanted if str(PurePosixPath(n).parent) == parent}, Code.INPUT_SET,
                    f"풀린 입력 폴더의 여분/누락 파일: {parent}")
        manifest = self.file("checkout", S.PATHS["manifest"])
        require(S.PATHS["manifest"] in wanted, Code.INPUT_SET, "매니페스트가 명부에 없다")
        rows = list(csv.DictReader(io.StringIO(manifest.read_text(encoding="utf-8"))))
        split = {r["rel_path"]: r["split"] for r in rows}
        require(len(split) == len(rows), Code.TABLE_READ, "매니페스트 중복 경로")
        for name in wanted:
            if "/tiles_v1_histmatch/" in name or name in split:
                require(split.get(name) in ("train", "val"), Code.INPUT_SET, f"평가/미등록 영상 유입: {name}")
        cache = self.roots["model_cache"]
        actual = {p.relative_to(cache).as_posix() for p in cache.rglob("*") if p.is_file() or p.is_symlink()}
        require(actual == {r["name"] for r in model["files"]}, Code.INPUT_SET, "모델 캐시 정확한 집합 불일치")
        for row in model["files"]:
            p = self.file("model_cache", row["name"])
            require(p.stat().st_size == row["bytes"] and sha(p) == row["sha256"], Code.INPUT_BYTES, f"모델 파일 불일치: {row['name']}")
        self.evidence["members"] = {"count": len(members), "sha256": sha(self.ref("members"))}

    def tables(self):
        self.records()
        read(self.ref("proof"))
        read(self.ref("measure"))
        read(self.ref("token_provenance"))
        read(self.ref("narrative_inputs"))

    def nonempty(self):
        members, model = self.records()
        require(bool(members) and bool(model["files"]), Code.TABLE_EMPTY, "빈 명부/모델 표")
        for key in ("a2", "heldout"):
            if key in self.refs:
                ids = self.ref(key).read_text(encoding="utf-8").splitlines()
                require(bool(ids) and all(s.strip() for s in ids) and len(set(ids)) == len(ids), Code.TABLE_EMPTY, f"빈/중복 행 목록: {key}")
        for key in ("token_table",):
            rows = list(csv.DictReader(io.StringIO(self.ref(key).read_text(encoding="utf-8"))))
            require(bool(rows), Code.TABLE_EMPTY, f"빈 표: {key}")

    def environment(self):
        c = self.c
        e = read(self.roots["environment"])
        require(type(e) is dict and all(k in e for k in S.ENV_REQUIRED), Code.ENV_MISMATCH, "환경 영수증 필수 근거 누락")
        session = current_session()
        require(e["session"] == session and e["cuda_ready"] is True, Code.ENV_MISMATCH, "현재 세션 CUDA 준비 기록이 아니다")
        locked = tomllib.loads(self.ref("lock").read_text(encoding="utf-8"))
        tomllib.loads(self.ref("project").read_text(encoding="utf-8"))
        versions = {name: importlib.metadata.version(name) for name in S.PACKAGES}
        for name, v in versions.items():
            candidates = {r["version"] for r in locked["package"] if r["name"] == name}
            require(v in candidates, Code.ENV_MISMATCH, f"잠금과 설치 판 불일치: {name}")
        for name in ("torch", "transformers", "bitsandbytes"):
            require(versions[name] == c["env_expect"][name] == e[name], Code.ENV_MISMATCH, f"환경 계약 불일치: {name}")
        require(e["uv_lock_sha256"] == c["env_expect"]["uv_lock_sha256"], Code.ENV_MISMATCH, "환경 잠금 지문 불일치")
        gpu = e["gpu_name"]
        require(isinstance(gpu, str) and c["gpu_expect"] in gpu and c["env_expect"]["gpu_name_has"] in gpu,
                Code.ENV_MISMATCH, "GPU 이름 기록 불일치")
        self.evidence.update(context={k: str(v) for k, v in self.roots.items()}, session=session,
                             environment_sha256=sha(self.roots["environment"]), packages=versions)

    def approval(self):
        c = self.c
        v = c["a3"]["threshold"]
        require(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1, Code.APPROVAL_BINDING, "문턱은 유한한 [0, 1] 수여야 한다")
        record = read(self.ref("approval"))
        validate(record, S.APPROVAL, "승인")
        at = datetime.fromisoformat(record["at"])
        require(at.utcoffset() is not None, Code.APPROVAL_BINDING, "승인 시각에 시간대가 없다")
        target = {k: c[k] for k in S.APPROVAL_TARGET}
        require(record["target"] == target, Code.APPROVAL_BINDING, "다른 코드/체제/페어/문턱/갈래/누적 상한의 승인")
        log = c["approved_by"]["decision_log"]
        section = _section(_git(self.roots["checkout"], "show", f"{log['commit']}:{relative(log['path'])}"), log["section"])
        require(sha_bytes(section.encode()) == log["section_sha256"], Code.APPROVAL_BINDING, "승인 절 지문 불일치")
        blocks = re.findall(r"^```json\n(.*?)^```\s*$", section, re.MULTILINE | re.DOTALL)
        require(len(blocks) == 1, Code.APPROVAL_BINDING, "승인 절의 구조화 JSON 한 개 필요")
        bound = loads(blocks[0])
        validate(bound, S.APPROVAL_SECTION, "승인 절")
        require(bound == {"record_sha256": c["approved_by"]["record"]["sha256"], "approval": record},
                Code.APPROVAL_BINDING, "승인 원문과 로그의 구조/해시 불일치")
        self.evidence["decision_log"] = log

    def heldout(self):
        a3 = self.c["a3"]
        require(a3["unexposed_path"] in ("나", "다"), Code.HELDOUT_BINDING, "미등록 비노출 갈래")
        if a3["unexposed_path"] == "나":
            require(a3["heldout_list"] is not None, Code.HELDOUT_BINDING, "나 갈래의 tune 목록 부재")
        if a3["heldout_list"] is not None:
            ids = self.ref("heldout").read_text(encoding="utf-8").splitlines()
            pairs = self.file("checkout", self.c["pairs_v2"]["folder"] + "/pairs.jsonl")
            rows = [loads(line) for line in pairs.read_bytes().splitlines() if line.strip()]
            val_ids = {r["image_id"] for r in rows if r["split"] == "val"}
            require(bool(ids) and len(ids) == len(set(ids)) and set(ids) <= val_ids, Code.HELDOUT_BINDING, "tune 목록이 val의 고유 행 집합이 아니다")
            require(not set(ids) & set(self.ref("a2").read_text(encoding="utf-8").splitlines()), Code.HELDOUT_BINDING, "tune/A2 중복")

    def init(self):
        import numpy as np

        from vlm.init_adapter import init_adapter_digest

        with np.load(self.ref("adapter"), allow_pickle=False) as z:
            require(bool(z.files) and len(z.files) == len(set(z.files)), Code.INIT_MISMATCH, "빈/중복 초기 배열")
            arrays = [z[k] for k in z.files]
            bs = [a for k, a in zip(z.files, arrays) if "lora_B" in k]
            require(bool(bs) and all(a.size and np.isfinite(a).all() for a in arrays)
                    and all(not np.any(a) for a in bs), Code.INIT_MISMATCH, "초기 B가 없거나 영이 아니다/비유한 배열")
            digest = init_adapter_digest(arrays)
        tc = self.file("checkout", S.PATHS["train_config"])
        require(digest == read(self.ref("proof"))["init_adapter_digest"] == read(tc)["init_adapter_digest"]
                == self.c["init_adapter"]["digest"], Code.INIT_MISMATCH, "배열/proof/보존 학습 설정/계약 digest 불일치")
        self.evidence["train_config"] = {"path": str(tc), "sha256": sha(tc)}

    def prompt(self):
        import yaml

        from vlm import registration, token_table

        c = self.c
        r = c["regime"]
        require(r["prompt_file"] == registration.REGIME_PROMPTS[r["recipe"]], Code.PROMPT_MISMATCH, "레시피의 프롬프트 경로 불일치")
        require(not registration.regime_problems(r["recipe"], prompt_sha256=r["prompt_sha256"], gen_limit=r["max_new_tokens"]),
                Code.PROMPT_MISMATCH, "레시피/프롬프트/한도 짝 불일치")
        prompt = self.ref("prompt")
        self.ref("vectors")
        measure = read(self.ref("measure"))
        rc = measure["input_contract"]["render_check"]
        require(rc["encode_C_all_match"] is True and rc["template_B_all_match"] is True
                and rc["prompt_sha256_text"] == sha_bytes(prompt.read_text(encoding="utf-8").encode()), Code.PROMPT_MISMATCH, "학습/생성 렌더 불일치")
        require(measure["inputs"]["prompt"] == {"path": r["prompt_file"], "sha256": r["prompt_sha256"]}, Code.PROMPT_MISMATCH, "계측 프롬프트 참조 불일치")
        _, model = self.records()
        config = yaml.safe_load(self.file("checkout", S.PATHS["config"]).read_bytes())["fixed_before_main_runs"]
        require(config["uni_model"]["id"] == model["id"] and config["uni_model"]["revision"] == model["revision"],
                Code.PROMPT_MISMATCH, "학습 설정과 캐시 명부의 모델/판 불일치")
        render = {"model_id": model["id"], "revision": model["revision"], "processor_kwargs": config["uni_processor_kwargs"],
                  "chat_template_kwargs": config["uni_chat_template_kwargs"]}
        require(measure["input_contract"]["render_config"] == render, Code.PROMPT_MISMATCH, "명시 렌더 설정 불일치")
        prov = read(self.ref("token_provenance"))
        targets = _literal(self.file("checkout", "vlm/pilot_vlm.py"), "TARGET_CONTRACTS")
        digest = sha_bytes(json.dumps(targets[r["recipe"]], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
        require(prov["contract_sha256"] == digest, Code.PROMPT_MISMATCH, "타깃 계약 지문 불일치")
        identity = {"model_id": model["id"], "revision": model["revision"],
                    "files": {row["name"]: sha(self.file("model_cache", row["name"])) for row in model["files"] if row["name"] in token_table.IDENTITY_FILES},
                    **{n: importlib.metadata.version(n) for n in ("transformers", "tokenizers")}}
        require("tokenizer_config.json" in identity["files"] and not token_table.identity_problems(prov["tokenizer"], identity),
                Code.PROMPT_MISMATCH, "토크나이저 파일/설치 판 신원 불일치")
        self.evidence["target_contract_sha256"] = digest

    def pairs(self):
        import yaml

        from vlm import budget, narrative_inputs, token_table
        from vlm.probe_inputs import verify_probe_pairs

        c = self.c
        folder = self.file("checkout", c["pairs_v2"]["folder"])
        facts = verify_probe_pairs(folder / "pairs.jsonl", expected_digest=c["pairs_v2"]["digest"], require_digest=True)
        self.evidence["snapshot"] = facts
        nc = self.ref("narrative_inputs")
        entry = yaml.safe_load(self.file("checkout", S.PATHS["registration"]).read_bytes())["narrative_inputs"]
        require(nc == self.file("checkout", narrative_inputs.contract_rel(entry["version"]))
                and entry["sha256"] == c["c08"]["narrative_inputs"]["sha256"], Code.PAIRS_BINDING, "학습 등록의 계약 참조 불일치")
        problems = narrative_inputs.problems(entry, repo=self.roots["checkout"], pairs_path=folder / "pairs.jsonl",
                                           config_pairs_digest=c["pairs_v2"]["digest"], recipe=c["regime"]["recipe"])
        require(not problems, Code.PAIRS_BINDING, problems)
        measure = read(self.ref("measure"))
        ic = measure["input_contract"]
        require(ic["pairs_digest"] == c["pairs_v2"]["digest"] and ic["identity_failures"] == []
                and ic["snapshot_problems"] == [], Code.C08_BINDING, "최종 페어 계측의 신원/실패 목록 불일치")
        require(all(measure["checks"].get(k) is True for k in S.MEASURE_CHECKS), Code.C08_BINDING, "필수 개별 계측 검사가 빠지거나 실패했다")
        require(c["regime"]["max_new_tokens"] == S.MAX_NEW_TOKENS
                and type(measure["recommended_limit"]["value"]) is int
                and measure["recommended_limit"]["value"] <= S.MAX_NEW_TOKENS,
                Code.C08_BINDING, "실행 생성 한도 또는 권고 답 길이가 승인 규칙과 다르다")
        for key, name in (("pairs", "pairs.jsonl"), ("pairs_meta", "PAIRS_META.json"), ("pairs_snapshot", "SNAPSHOT.sha256")):
            require(measure["inputs"][key] == {"path": c["pairs_v2"]["folder"] + "/" + name, "sha256": sha(folder / name)},
                    Code.C08_BINDING, f"계측의 {key} 참조 불일치")
        table = self.ref("token_table")
        provpath = self.ref("token_provenance")
        pinned_path, pinned_sha = budget.token_table_of(c["regime"]["recipe"])
        require(table == self.file("checkout", str(pinned_path).replace("\\", "/")) and sha(table) == pinned_sha
                and provpath == table.with_name(table.name + budget.PROVENANCE_SUFFIX), Code.C08_BINDING, "정식 토큰 표/곁 출처가 아니다")
        prov = read(provpath)
        require(prov["schema"] == token_table.PROVENANCE_SCHEMA and type(prov["schema"]) is int
                and prov["recipe"] == c["regime"]["recipe"] and prov["pairs_digest"] == c["pairs_v2"]["digest"]
                and prov["pairs"] == measure["inputs"]["pairs"] and prov["table_sha256"] == sha(table)
                and prov["counter"] == {"module": "vlm.token_table", "sha256_lf": token_table.source_sha256()},
                Code.C08_BINDING, "토큰 출처의 페어/표/계측 코드 신원 불일치")
        rows = list(csv.DictReader(io.StringIO(table.read_text(encoding="utf-8"))))
        require(bool(rows) and set(rows[0]) == {"image_id", "client", "supervised_tokens"}, Code.C08_BINDING, "토큰 표 꼴")
        train = {r["image_id"]: r["client"] for r in (loads(line) for line in (folder / "pairs.jsonl").read_bytes().splitlines()) if r["split"] == "train"}
        require(len(rows) == len(train) == len({r["image_id"] for r in rows}) == prov["n_rows"]
                and {r["image_id"]: r["client"] for r in rows} == train and all(int(r["supervised_tokens"]) > 0 for r in rows),
                Code.C08_BINDING, "토큰 표의 학습 행 집합/중복/빈 토큰")


def inspect(contract, *, inputs, checkout, model_cache, environment):
    """아홉 항목을 끝까지 판정한다. 실패나 판독 불능을 성공으로 채우지 않는다."""
    cp = Path(contract)
    receipt = {"kind": KIND, "schema": SCHEMA, "checker_version": VERSION, "contract_sha256": None,
               "inputs": {}, "checks": {}, "all_ok": False, "at": datetime.now(UTC).isoformat()}
    inspector, prerequisite = None, None
    try:
        receipt["contract_sha256"] = sha(cp)
        inspector = Inspector(cp, inputs=inputs, checkout=checkout, model_cache=model_cache, environment=environment)
        for key, (root, name, expected) in inspector.refs.items():
            try:
                measured = sha(inspector.file(root, name))
            except (OSError, ValueError):
                measured = None
            receipt["inputs"][key] = {"contract": expected, "measured": measured}
    except (OSError, ValueError, TypeError) as exc:
        prerequisite = exc
    for name in S.CHECK_NAMES:
        evidence = {}
        result = {"ok": False, "codes": [], "evidence": evidence}
        receipt["checks"][name] = result
        try:
            if prerequisite:
                code = prerequisite.code if isinstance(prerequisite, Rejected) else Code.CONTRACT_FORM
                raise Rejected(code if name == "tables" else Code.PRECONDITION_FAILED, prerequisite)
            inspector.evidence = evidence
            getattr(inspector, name)()
            result["ok"] = True
        except Rejected as exc:
            result["codes"].append(exc.code.value)
            evidence["error"] = str(exc)
        except OSError as exc:
            result["codes"].append(Code.IO_ERROR.value)
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        except (KeyError, TypeError, ValueError, SystemExit, tarfile.TarError) as exc:
            result["codes"].append((Code.TABLE_READ if name == "tables" else Code.PRECONDITION_FAILED).value)
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 — 예기치 않은 실패도 거부 영수증에 남긴다.
            result["codes"].append(Code.INTERNAL_ERROR.value)
            evidence["error"] = f"{type(exc).__name__}: {exc}"
    if inspector is not None and all(r["ok"] for r in receipt["checks"].values()):
        try:
            inspector.evidence = receipt["checks"]["inputs"]["evidence"]
            inspector.code()
            require(sha(cp) == receipt["contract_sha256"], Code.STALE_RECEIPT, "검사 중 계약 변경")
            for key in inspector.refs:
                inspector.ref(key)
            require(sha(inspector.roots["environment"]) == receipt["checks"]["environment"]["evidence"]["environment_sha256"],
                    Code.STALE_RECEIPT, "검사 중 환경 영수증 변경")
        except (OSError, Rejected) as exc:
            r = receipt["checks"]["inputs"]
            r["ok"] = False
            r["codes"].append(getattr(exc, "code", Code.IO_ERROR).value)
            r["evidence"]["error"] = str(exc)
    receipt["all_ok"] = bool(receipt["inputs"]) and all(r["ok"] for r in receipt["checks"].values()) and all(
        r["contract"] == r["measured"] for r in receipt["inputs"].values())
    return receipt


def write_receipt(path, receipt):
    """같은 디렉터리의 완성 파일을 hard-link로 공개한다. 경쟁 시에도 기존 출력을 덮지 않는다."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise Rejected(Code.IO_ERROR, "출력 영수증이 이미 있다 — 덮어쓰지 않는다")
    fd, temporary = tempfile.mkstemp(prefix=".preflight-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write((json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=1, allow_nan=False) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify(receipt_path, contract_path):
    """기존 성공 영수증과 현재의 계약·입력·코드·환경을 같은 검사로 다시 맞댄다. 새 영수증은 쓰지 않는다."""
    try:
        raw = Path(receipt_path).read_bytes()
        r = loads(raw)
        require(type(r) is dict and r.keys() == S.RECEIPT_KEYS, Code.STALE_RECEIPT, "영수증 최상위 정확한 여덟 칸 필요")
        require(r["kind"] == KIND and type(r["schema"]) is int and r["schema"] == SCHEMA
                and r["checker_version"] == VERSION and r["all_ok"] is True, Code.STALE_RECEIPT, "성공/검사기 판 불일치")
        require(type(r["checks"]) is dict and set(r["checks"]) == set(S.CHECK_NAMES), Code.STALE_RECEIPT, "아홉 검사 집합 불일치")
        for check in r["checks"].values():
            require(type(check) is dict and check.keys() == S.CHECK_KEYS and check["ok"] is True and check["codes"] == []
                    and type(check["evidence"]) is dict, Code.STALE_RECEIPT, "통과 검사 꼴 불일치")
        at = datetime.fromisoformat(r["at"])
        require(at.utcoffset() is not None and at <= datetime.now(UTC), Code.STALE_RECEIPT, "영수증 시각 불일치")
        c = read(contract_path)
        expected = expected_inputs(c)
        require(r["contract_sha256"] == sha(contract_path) and type(r["inputs"]) is dict and r["inputs"].keys() == expected.keys(),
                Code.STALE_RECEIPT, "계약/기대 입력 집합 불일치")
        for key, digest in expected.items():
            require(r["inputs"][key] == {"contract": digest, "measured": digest}, Code.STALE_RECEIPT, f"입력 영수증 꼴/해시: {key}")
        context = r["checks"]["environment"]["evidence"]["context"]
        require(type(context) is dict and context.keys() == S.CONTEXT_KEYS and all(isinstance(p, str) and Path(p).is_absolute() for p in context.values()),
                Code.STALE_RECEIPT, "실측 경로 근거 누락")
        now = inspect(contract_path, **context)
        require(now["all_ok"] and now["inputs"] == r["inputs"] and now["checks"] == r["checks"],
                Code.STALE_RECEIPT, "입력/코드/세션/증거가 바뀌었다")
        require(Path(receipt_path).read_bytes() == raw, Code.STALE_RECEIPT, "소비 중 영수증 변경")
        return r
    except Rejected:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise Rejected(Code.STALE_RECEIPT, exc) from exc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", nargs=2, metavar=("PREFLIGHT", "CONTRACT"))
    for name in ("contract", "inputs", "checkout", "out", "model-cache", "environment"):
        p.add_argument("--" + name)
    args = p.parse_args(argv)
    try:
        if args.verify:
            if any(getattr(args, n) is not None for n in ("contract", "inputs", "checkout", "out", "model_cache", "environment")):
                p.error("--verify는 다른 인자와 섞지 않는다")
            verify(*args.verify)
            return 0
        if not all(getattr(args, n) for n in ("contract", "inputs", "checkout", "out", "model_cache", "environment")):
            p.error("계약·입력 뿌리·체크아웃·출력·모델 캐시·환경 영수증 경로가 모두 필요하다")
        receipt = inspect(args.contract, inputs=args.inputs, checkout=args.checkout,
                          model_cache=args.model_cache, environment=args.environment)
        write_receipt(args.out, receipt)
        return 0 if receipt["all_ok"] else 1
    except (OSError, Rejected) as exc:
        print(f"{getattr(exc, 'code', Code.IO_ERROR)}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
