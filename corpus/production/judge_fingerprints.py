"""판정 파서의 실제 정적 호출 폐포만 결속한다. 감사 보고 코드는 결속하지 않는다.

소스를 실행하지 않고 AST와 code object를 비교한다. 지원하지 않는 동적 전역
조회/바인딩은 추측하지 않고 거부한다. Python 표준 라이브러리 자체의 공급망
검증이나 임의의 악성 프로세스에 대한 인증 도구는 아니다.
"""

from __future__ import annotations

import ast
import builtins
import dis
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from types import CodeType, FunctionType, ModuleType

_JSON = {"loads": json.loads, "dumps": json.dumps}
_RE_SUB = re.sub
_NORMALIZE = unicodedata.normalize
_MAPPING = Mapping
_DYNAMIC = {"eval", "exec", "globals", "locals", "vars", "getattr", "setattr",
            "delattr", "__import__", "compile"}
_MISSING = object()
_V5_SCHEMA = "judge_structured_v5"
_V6_SCHEMA = "judge_structured_v6"


class FingerprintRefused(ValueError):
    """실제 실행 함수와 확인 가능한 정적 의존성이 일치하지 않는다."""


def _canonical(value):
    return _JSON["dumps"](value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _code_value(code):
    """무관한 행 삽입/경로 차이는 무시하고 실행 code object의 구조를 비교한다."""
    def constant(value):
        if isinstance(value, CodeType):
            return ("code", _code_value(value))
        if isinstance(value, tuple):
            return ("tuple", tuple(constant(v) for v in value))
        if isinstance(value, frozenset):
            return ("frozenset", frozenset(constant(v) for v in value))
        return (type(value).__name__, value)

    return (code.co_code, tuple(constant(v) for v in code.co_consts), code.co_names,
            code.co_varnames, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_freevars, code.co_cellvars,
            getattr(code, "co_exceptiontable", b""))


def _global_names(code):
    names = {i.argval for i in dis.get_instructions(code)
             if i.opname in {"LOAD_GLOBAL", "LOAD_NAME"}}
    for value in code.co_consts:
        if isinstance(value, CodeType):
            names.update(_global_names(value))
    return names


class _Closure:
    def __init__(self, modules):
        self.modules = {m.__name__: m for m in modules}
        if len(self.modules) != len(modules):
            raise FingerprintRefused("duplicate parser module names")
        self.sources = {}
        self.selected = {}
        self.visiting = set()
        for module in modules:
            path = Path(module.__file__)
            raw = path.read_bytes()
            tree = ast.parse(raw, filename=str(path))
            # compile은 code object만 만들며 import나 함수 본문을 실행하지 않는다.
            compiled = compile(tree, str(path), "exec", dont_inherit=True,
                               optimize=sys.flags.optimize)
            definitions = {}
            futures = []
            for node in tree.body:
                entries = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    entries = [(node.name, node)]
                elif isinstance(node, ast.Assign):
                    entries = [(t.id, node) for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    entries = [(node.target.id, node)]
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                        futures.append(ast.dump(node, include_attributes=False))
                        continue
                    for alias in node.names:
                        if alias.name == "*":
                            raise FingerprintRefused("wildcard import prevents a closed parser identity")
                        name = alias.asname or alias.name.split(".")[0]
                        if isinstance(node, ast.Import):
                            part = ast.Import(names=[alias])
                        else:
                            part = ast.ImportFrom(module=node.module, names=[alias], level=node.level)
                        entries.append((name, part))
                for name, binding in entries:
                    definitions.setdefault(name, []).append(binding)
            self.sources[module.__name__] = {
                "module": module, "path": path, "raw": raw, "tree": tree,
                "definitions": definitions, "futures": futures,
                "codes": {c.co_name: c for c in compiled.co_consts if isinstance(c, CodeType)},
            }

    def _entry(self, module_name, name, node):
        self.selected[f"{module_name}:{name}"] = {
            "kind": type(node).__name__,
            "ast_sha256": hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")).hexdigest(),
        }

    def _import_target(self, module, node):
        alias = node.names[0]
        if isinstance(node, ast.Import):
            target = alias.name if alias.asname else alias.name.split(".")[0]
            return target, None
        package = "." * node.level + (node.module or "")
        target = (importlib.util.resolve_name(package, module.__package__)
                  if node.level else package)
        return target, alias.name

    def _external_import(self, target, symbol):
        if target != "json" or symbol not in {None, *_JSON}:
            raise FingerprintRefused(f"unsupported parser import: {target}.{symbol}")
        if any(getattr(json, k) is not v for k, v in _JSON.items()):
            raise FingerprintRefused("loaded standard JSON parser functions changed")
        return json if symbol is None else _JSON[symbol]

    def _module_attribute(self, module, attribute):
        return (module is json and attribute in _JSON
                and getattr(json, attribute) is _JSON[attribute])

    def _computed_constant(self, source, name, node, actual):
        return False

    def visit(self, module_name, name):
        key = (module_name, name)
        if key in self.visiting:
            return
        source = self.sources[module_name]
        module = source["module"]
        bindings = source["definitions"].get(name, [])
        if len(bindings) != 1:
            raise FingerprintRefused(f"ambiguous or dynamic parser binding: {module_name}.{name}")
        node = bindings[0]
        self.visiting.add(key)
        self._entry(module_name, name, node)
        actual = vars(module).get(name, _MISSING)
        if isinstance(node, ast.FunctionDef):
            self._function(source, name, node, actual)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if self._computed_constant(source, name, node, actual):
                return
            if isinstance(node.value, ast.Name):
                self.visit(module_name, node.value.id)
                if actual is not vars(module).get(node.value.id, _MISSING):
                    raise FingerprintRefused(f"parser alias changed: {module_name}.{name}")
            else:
                try:
                    expected = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise FingerprintRefused(f"nonliteral parser constant: {module_name}.{name}") from exc
                if type(actual) is not type(expected) or actual != expected:
                    raise FingerprintRefused(f"loaded parser constant differs: {module_name}.{name}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            target, symbol = self._import_target(module, node)
            if target in self.modules and symbol is not None:
                self.visit(target, symbol)
                expected = vars(self.modules[target]).get(symbol, _MISSING)
            else:
                expected = self._external_import(target, symbol)
            if actual is not expected:
                raise FingerprintRefused(f"loaded parser import differs: {module_name}.{name}")
        else:
            raise FingerprintRefused(f"unsupported parser binding: {module_name}.{name}")

    def _function(self, source, name, node, actual):
        module = source["module"]
        if (not isinstance(actual, FunctionType) or actual.__globals__ is not vars(module)
                or actual.__closure__ or node.decorator_list):
            raise FingerprintRefused(f"wrapped or replaced parser function: {module.__name__}.{name}")
        expected_code = source["codes"].get(name)
        if expected_code is None or _code_value(actual.__code__) != _code_value(expected_code):
            raise FingerprintRefused(f"loaded parser code differs from source: {module.__name__}.{name}")
        try:
            defaults = tuple(ast.literal_eval(v) for v in node.args.defaults) or None
            kwdefaults = {a.arg: ast.literal_eval(v) for a, v in
                          zip(node.args.kwonlyargs, node.args.kw_defaults) if v is not None} or None
        except (ValueError, TypeError, SyntaxError) as exc:
            raise FingerprintRefused("nonliteral parser defaults") from exc
        if actual.__defaults__ != defaults or actual.__kwdefaults__ != kwdefaults:
            raise FingerprintRefused("loaded parser defaults differ")
        if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
               for n in ast.walk(node)):
            raise FingerprintRefused("dynamic/local parser imports or global writes are unsupported")
        names = _global_names(expected_code)
        for dependency in sorted(names):
            if dependency in _DYNAMIC:
                raise FingerprintRefused(f"dynamic parser lookup is unsupported: {dependency}")
            if dependency in source["definitions"]:
                self.visit(module.__name__, dependency)
            else:
                expected = getattr(builtins, dependency, _MISSING)
                runtime_builtins = actual.__builtins__
                loaded = actual.__globals__.get(dependency, runtime_builtins.get(dependency, _MISSING))
                if expected is _MISSING or loaded is not expected:
                    raise FingerprintRefused(f"unbound or replaced parser global: {dependency}")
        # 모듈 의존성은 명시적인 json.loads/json.dumps만 허용한다.
        # 동적 속성 조회가 의존성 검증을 우회하면 안 된다.
        for child in ast.walk(node):
            if (isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name)
                    and child.value.id in names
                    and isinstance(vars(module).get(child.value.id), ModuleType)):
                if not self._module_attribute(vars(module)[child.value.id], child.attr):
                    raise FingerprintRefused("unsupported parser module attribute")

    def result(self, roots):
        for module, name in roots:
            self.visit(module.__name__, name)
        for source in self.sources.values():
            if source["path"].read_bytes() != source["raw"]:
                raise FingerprintRefused("parser source changed during fingerprinting")
        payload = {
            "schema_version": 1, "method": "selected-parser-ast-closure-v1",
            "roots": [f"{m.__name__}:{n}" for m, n in roots],
            "symbols": dict(sorted(self.selected.items())),
            "future_imports": {name: source["futures"]
                               for name, source in sorted(self.sources.items())},
            "stdlib_contract": "Python json.loads/json.dumps and unmodified builtin bindings",
        }
        return {**payload, "closure_sha256": _digest(payload),
                "fingerprinter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _parser_identity(quality_module, common_module, schema=_V5_SCHEMA):
    """선택한 출력 계약의 실제 파서 폐포. 기본값은 기존 v5 호출자와 호환된다."""
    if schema == _V6_SCHEMA:
        return _parser_identity_v6(quality_module, common_module)
    if schema != _V5_SCHEMA:
        raise FingerprintRefused(f"unsupported parser schema: {schema!r}")
    namespace = vars(quality_module)
    marker = namespace.get("JUDGE_V5_SCHEMA_VERSION", _MISSING)
    parser = namespace.get("parse_judge_v5", _MISSING)
    has_v5 = marker is not _MISSING or parser is not _MISSING
    if has_v5 and (type(marker) is not str or marker != _V5_SCHEMA
                   or not isinstance(parser, FunctionType)):
        raise FingerprintRefused("incomplete or unsupported structured-v5 parser capability")
    roots = [(quality_module, "_judge"), (common_module, "parse_json"),
             (common_module, "canonical_bytes")]
    if has_v5:
        roots.extend(((quality_module, "JUDGE_V5_SCHEMA_VERSION"),
                      (quality_module, "parse_judge_v5")))
    report = _Closure((quality_module, common_module)).result(roots)
    if (namespace.get("JUDGE_V5_SCHEMA_VERSION", _MISSING) is not marker
            or namespace.get("parse_judge_v5", _MISSING) is not parser):
        raise FingerprintRefused("structured-v5 parser capability changed during fingerprinting")
    report["capabilities"] = {"legacy_four_keys": True, _V5_SCHEMA: has_v5}
    return _rehash_report(report)


def _parser_identity_v6(quality_module, common_module):
    """v6 키·식·형식 정규화에 닿는 선택 폐포를 독립적으로 결속한다.

    v5 파서를 루트에 넣지 않는다. 이미 남긴 v5 원문을 새 규칙으로 재해석하거나
    v6 능력이 빠졌을 때 v5 능력을 대신 승인하는 일이 없어야 한다.
    """
    namespace = vars(quality_module)
    marker = namespace.get("JUDGE_V6_SCHEMA_VERSION", _MISSING)
    parser = namespace.get("parse_judge_v6", _MISSING)
    has_v6 = marker is not _MISSING or parser is not _MISSING
    if has_v6 and (type(marker) is not str or marker != _V6_SCHEMA
                   or not isinstance(parser, FunctionType)):
        raise FingerprintRefused("incomplete or unsupported structured-v6 parser capability")
    roots = [(common_module, "parse_json"), (common_module, "canonical_bytes")]
    if has_v6:
        roots.extend(((quality_module, "JUDGE_V6_SCHEMA_VERSION"),
                      (quality_module, "parse_judge_v6")))
    report = _Closure((quality_module, common_module)).result(roots)
    if has_v6:
        symbols = report["symbols"]
        if (symbols[f"{quality_module.__name__}:parse_judge_v6"]["kind"] != "FunctionDef" or
                f"{quality_module.__name__}:parse_judge_v5" in symbols):
            raise FingerprintRefused("structured-v6 parser aliases or falls back to v5")
    if (namespace.get("JUDGE_V6_SCHEMA_VERSION", _MISSING) is not marker
            or namespace.get("parse_judge_v6", _MISSING) is not parser):
        raise FingerprintRefused("structured-v6 parser capability changed during fingerprinting")
    report["selected_schema"] = _V6_SCHEMA
    report["capabilities"] = {_V6_SCHEMA: has_v6}
    return _rehash_report(report)


def parser_identity(schema=_V5_SCHEMA):
    """선택한 실제 판정 JSON 파서와 직렬화 폐포를 읽기 전용으로 확인한다."""
    from . import common, quality

    return _parser_identity(quality, common, schema)


def _require_v5_parser(quality_module, common_module):
    report = _parser_identity(quality_module, common_module)
    if report["capabilities"][_V5_SCHEMA] is not True:
        raise FingerprintRefused("B structured-v5 parser is not implemented and fingerprinted")
    return report


def require_v5_parser():
    """B 소유 구조화 파서가 실제 선택 폐쇄에 결속된 경우에만 그 지문을 낸다.

    호출자는 이 결과를 정책 신원의 parser_implementation 과 대조해야 한다.
    이 준비 확인은 판별력 승인이나 B 파서의 의미 정확도 승인을 대신하지 않는다.
    """
    from . import common, quality

    return _require_v5_parser(quality, common)


def _require_v6_parser(quality_module, common_module):
    report = _parser_identity(quality_module, common_module, _V6_SCHEMA)
    if report.get("selected_schema") != _V6_SCHEMA or report["capabilities"].get(_V6_SCHEMA) is not True:
        raise FingerprintRefused("B structured-v6 parser is not implemented and fingerprinted")
    return report


def require_v6_parser():
    """현재 v6 파서의 선택 폐포를 검증한다. 판별력/운영 승인은 하지 않는다."""
    from . import common, quality

    return _require_v6_parser(quality, common)


def _rehash_report(report):
    payload = {key: value for key, value in report.items()
               if key not in {"closure_sha256", "fingerprinter_sha256"}}
    return {**report, "closure_sha256": _digest(payload)}


_WINDOW_BODY = '''t = re.sub(r"\\s+", "", unicodedata.normalize("NFC", text))
return {t[i:i + size] for i in range(len(t) - size + 1)}
'''


class _LeakClosure(_Closure):
    """현재 유출 검사의 명시된 의존성만 추가한다. 일반 AST 실행기는 아니다."""

    def _external_import(self, target, symbol):
        if target == "re" and symbol in {None, "sub"}:
            if re.sub is not _RE_SUB:
                raise FingerprintRefused("loaded standard re.sub changed")
            return re if symbol is None else _RE_SUB
        if target == "unicodedata" and symbol in {None, "normalize"}:
            if unicodedata.normalize is not _NORMALIZE:
                raise FingerprintRefused("loaded standard Unicode normalizer changed")
            return unicodedata if symbol is None else _NORMALIZE
        if target == "collections.abc" and symbol == "Mapping":
            return _MAPPING
        return super()._external_import(target, symbol)

    def _module_attribute(self, module, attribute):
        return (super()._module_attribute(module, attribute)
                or (module is re and attribute == "sub" and re.sub is _RE_SUB)
                or (module is unicodedata and attribute == "normalize"
                    and unicodedata.normalize is _NORMALIZE))

    def _computed_constant(self, source, name, node, actual):
        if name != "_SYSTEM_WINDOWS":
            return False
        expected_call = ast.parse("_windows(GENERATE_SYSTEM, SYSTEM_WINDOW)", mode="eval").body
        if ast.dump(node.value, include_attributes=False) != ast.dump(expected_call, include_attributes=False):
            raise FingerprintRefused("unsupported system-window initializer")
        module_name = source["module"].__name__
        for dependency in ("_windows", "GENERATE_SYSTEM", "SYSTEM_WINDOW"):
            self.visit(module_name, dependency)
        window_node = source["definitions"]["_windows"][0]
        if not isinstance(window_node, ast.FunctionDef):
            raise FingerprintRefused("unsupported system-window function binding")
        expected_body = ast.parse(_WINDOW_BODY).body
        actual_body = window_node.body
        if (actual_body and isinstance(actual_body[0], ast.Expr)
                and isinstance(actual_body[0].value, ast.Constant)
                and isinstance(actual_body[0].value.value, str)):
            actual_body = actual_body[1:]
        args = window_node.args
        if (args.posonlyargs or [a.arg for a in args.args] != ["text", "size"]
                or args.vararg or args.kwarg or args.kwonlyargs or args.defaults
                or [ast.dump(n, include_attributes=False) for n in actual_body]
                != [ast.dump(n, include_attributes=False) for n in expected_body]):
            raise FingerprintRefused("unsupported system-window function body")
        namespace = vars(source["module"])
        text, size = namespace["GENERATE_SYSTEM"], namespace["SYSTEM_WINDOW"]
        if type(text) is not str or type(size) is not int or size < 1 or len(text) > 1_000_000:
            raise FingerprintRefused("unsupported system-window inputs")
        # 검증한 좁은 계산만 독립적으로 재현한다. 적재 함수나 initializer는 호출하지 않는다.
        compact = _RE_SUB(r"\s+", "", _NORMALIZE("NFC", text))
        expected = {compact[i:i + size] for i in range(len(compact) - size + 1)}
        if type(actual) is not set or actual != expected:
            raise FingerprintRefused("loaded system-window cache differs")
        return True


def _leak_identity(quality_module, common_module, infer_module):
    report = _LeakClosure((quality_module, common_module, infer_module)).result(
        ((quality_module, "leak_problems"), (common_module, "canonical_bytes")))
    report["dependency_contract"] = (
        "selected leak AST; standard re.sub, Unicode normalize, Mapping and JSON; "
        "only the explicit system-window initializer is independently recomputed")
    # 추가 계약도 최종 폐쇄 지문에 포함한다.
    return _rehash_report(report)


def leak_identity():
    """유출 검사 선택 폐쇄를 읽기 전용으로 검증한다. 모델은 적재하지 않는다."""
    from . import common, infer, quality

    return _leak_identity(quality, common, infer)
