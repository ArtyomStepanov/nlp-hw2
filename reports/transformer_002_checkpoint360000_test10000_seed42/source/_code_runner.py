"""Отдельный процесс для ограниченного выполнения арифметического solve().

Внутренний модуль evaluation.py; не импортирует PyTorch и не запускает код
в процессе ноутбука. Это ограниченный Python evaluator, не универсальная
песочница для произвольных недоверенных программ.
"""

import ast
import builtins
from contextlib import redirect_stdout, redirect_stderr
import json
import math
import os
import sys


ALLOWED_NODES = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign", "AnnAssign", "AugAssign",
    "Expr", "For", "While", "If", "Break", "Continue", "Pass", "Name", "Load", "Store", "Constant",
    "BinOp", "UnaryOp", "BoolOp", "Compare", "IfExp", "Call", "keyword", "Attribute", "Subscript", "Slice",
    "List", "Tuple", "Dict", "Set", "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    "JoinedStr", "FormattedValue", "Import", "ImportFrom", "alias",
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "UAdd", "USub", "Not", "And", "Or",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "In", "NotIn", "Is", "IsNot",
}
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int", "isinstance",
    "len", "list", "map", "max", "min", "pow", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
)}
# Общий доступ к атрибутам опасен даже без dunder: gi_frame/f_back/f_globals
# раскрывают globals процесса. format/format_map также умеют обходить цепочки
# атрибутов внутри строк, поэтому они намеренно не входят в этот список.
SAFE_ATTRIBUTES = {name for name in dir(math) if not name.startswith("_")} | {
    "append", "extend", "insert", "remove", "pop", "clear", "copy", "count", "index", "reverse", "sort",
    "keys", "values", "items", "get", "setdefault", "update", "fromkeys", "popitem",
    "add", "discard", "union", "intersection", "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint",
    "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "join", "replace", "lower", "upper", "casefold",
    "capitalize", "title", "swapcase", "startswith", "endswith", "find", "rfind", "rindex", "partition", "rpartition",
    "removeprefix", "removesuffix", "isdigit", "isnumeric", "isdecimal", "isalpha", "isalnum", "isspace",
    "encode", "decode", "zfill", "ljust", "rjust", "center", "real", "imag", "conjugate", "as_integer_ratio", "bit_length", "bit_count",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name != "math" or level:
        raise ImportError("Разрешён только модуль math.")
    return math


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if type(node).__name__ not in ALLOWED_NODES:
            raise ValueError(f"Не поддерживается {type(node).__name__}.")
        if isinstance(node, (ast.Name, ast.arg)) and (node.id if isinstance(node, ast.Name) else node.arg).startswith("_"):
            raise ValueError("Служебные имена недоступны.")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            raise ValueError(f"Атрибут {node.attr!r} не разрешён.")
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("_") or node.decorator_list):
            raise ValueError("Служебные функции и декораторы недоступны.")
        if isinstance(node, ast.Import) and any(alias.name != "math" or (alias.asname or "").startswith("_") for alias in node.names):
            raise ValueError("Разрешён только import math.")
        if isinstance(node, ast.ImportFrom) and (node.module != "math" or node.level or any(alias.name.startswith("_") or (alias.asname or "").startswith("_") for alias in node.names)):
            raise ValueError("Разрешён только from math import ... .")


def _json_value(value, depth=0):
    if depth > 20:
        raise ValueError("Слишком глубокий результат.")
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Ответ должен быть конечным.")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, depth + 1) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    raise ValueError("solve() должен вернуть число, строку или JSON-совместимую коллекцию; None недопустим.")


def _run(code: str) -> dict:
    if not code.strip():
        return {"status": "empty_code", "error": "Пустой код."}
    try:
        tree = ast.parse(code)
        compiled = compile(tree, "<generated>", "exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "syntax_error", "error": str(exc)[:500]}
    try:
        _validate(tree)
    except ValueError as exc:
        return {"status": "unsupported_code", "error": str(exc)}
    namespace = {"__builtins__": {**SAFE_BUILTINS, "__import__": _safe_import}, "__name__": "generated", "math": math}
    # print допускается, но весь вывод отправляется в /dev/null, без буфера в RAM.
    namespace["__builtins__"]["print"] = builtins.print
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            exec(compiled, namespace)
            solve = namespace.get("solve")
            if not callable(solve):
                return {"status": "missing_solve", "error": "Нет функции solve()."}
            value = solve()
    except BaseException as exc:
        return {"status": "runtime_error", "error": f"{type(exc).__name__}: {exc}"[:500]}
    try:
        value = _json_value(value)
        if len(json.dumps(value, ensure_ascii=True, allow_nan=False)) > 16384:
            raise ValueError("Слишком большой результат (>16 KiB).")
    except (ValueError, TypeError, RecursionError) as exc:
        return {"status": "invalid_output", "error": str(exc)[:500]}
    return {"status": "ok", "value": value, "error": None}


def main() -> None:
    request = json.load(sys.stdin)
    # На POSIX ограничиваем CPU, адресное пространство, размер файлов и fd.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (request["cpu_seconds"], request["cpu_seconds"]))
        resource.setrlimit(resource.RLIMIT_AS, (request["memory_bytes"], request["memory_bytes"]))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ImportError, OSError, ValueError):
        # Родитель всегда контролирует wall-clock timeout; лимиты зависят от ОС.
        pass
    try:
        result = _run(request["code"])
        serialized = json.dumps(result, ensure_ascii=True, allow_nan=False)
    except BaseException as exc:
        serialized = json.dumps({"status": "runtime_error", "error": type(exc).__name__})
    sys.stdout.write(serialized)


if __name__ == "__main__":
    main()
