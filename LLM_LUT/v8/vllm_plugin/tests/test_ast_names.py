#!/usr/bin/env python3
"""Static undefined-name scan (no torch/vllm needed): ast.parse every
vllm_plugin/*.py and report Name loads that are covered by neither the
module level, the enclosing top-level scope's stores, nor builtins. This
is the local defense for the "missing parameter" bug class (the
_attend_only v_new class) — py_compile cannot see it and
test_integration only runs where torch exists.

Deliberate over-approximation: stores from ALL scopes inside the
enclosing top-level definition count as visible (closures, sibling
branches). That can miss a residual hidden by an unrelated same-named
store, but never false-positives one.

Run anywhere:
    python vllm_plugin/tests/test_ast_names.py
"""
import ast
import builtins
import os
import sys

ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

BUILTIN_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__"}


def module_names(tree):
    """Names bound at module level: imports, assignments, defs, classes."""
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) \
                else [node.target]
            for t in targets:
                names.update(n.id for n in ast.walk(t)
                             if isinstance(n, ast.Name))
    return names


def subtree_stores(node):
    """Every name bound anywhere inside node (incl. nested scopes):
    assignment targets, import aliases, def/class names, except-as names."""
    stores = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(
                n.ctx, (ast.Store, ast.Del)):
            stores.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                            ast.ClassDef)):
            stores.add(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            stores.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                stores.add((a.asname or a.name).split(".")[0])
    return stores


def arg_names(args):
    out = set()
    for x in args.posonlyargs + args.args + args.kwonlyargs + \
            [args.vararg, args.kwarg]:
        if x is not None:
            out.add(x.arg)
    return out


def scope_header_nodes(child):
    """Parts of a nested def that evaluate in the ENCLOSING scope:
    decorators, defaults, annotations (the body is checked separately at
    the child's own node)."""
    out = []
    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out.extend(child.decorator_list)
        if child.returns is not None:
            out.append(child.returns)
    a = child.args
    for d in a.defaults:
        out.append(d)
    for d in a.kw_defaults:
        if d is not None:
            out.append(d)
    for x in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
        if x is not None and x.annotation is not None:
            out.append(x.annotation)
    return out


def scope_own_loads(body):
    """Load Names evaluated in THIS scope's body — nested def/lambda
    bodies are excluded (checked at their own nodes); nested headers are
    included. Class bodies are transparent (executed at class creation,
    still able to read enclosing names)."""
    loads = []
    stack = []

    def push(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            stack.extend(scope_header_nodes(node))
        else:
            stack.append(node)

    for stmt in body:
        push(stmt)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.append(node)
        for child in ast.iter_child_nodes(node):
            push(child)
    return loads


def all_scopes(tree):
    """Every scope-bearing node: module body, functions, lambdas.
    Yields (node, name, body, local_names)."""
    scopes = [(None, "module", tree.body, set())]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            body = ([node.body] if isinstance(node, ast.Lambda)
                    else node.body)
            name = node.name if hasattr(node, "name") else "<lambda>"
            scopes.append((node, name, body, arg_names(node.args)))
    return scopes


def check_file(path, rel):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    mod = module_names(tree)
    # module-level scope can also see names bound at module level outside
    # defs (incl. comprehension targets in module-level expressions)
    mod_allowed = mod | BUILTIN_NAMES
    for stmt in tree.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            mod_allowed |= subtree_stores(stmt)
    fails = []
    reported = set()
    tops = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef))]
    top_stores = {}
    for top in tops:
        # allowed = module level + everything bound anywhere inside this
        # top-level scope (closures and sibling branches included on
        # purpose — over-approximation, see module docstring)
        top_stores[id(top)] = mod | BUILTIN_NAMES | subtree_stores(top)

    def allowed_for(node):
        if node is None:
            return mod_allowed  # module-level scope
        for top in tops:
            if any(n is node for n in ast.walk(top)):
                return top_stores[id(top)]
        raise AssertionError("scope node outside top-level scopes")

    for node, name, body, args in all_scopes(tree):
        ok = allowed_for(node) | args
        for n in scope_own_loads(body):
            if n.id in ok or (n.id, n.lineno) in reported:
                continue
            reported.add((n.id, n.lineno))
            fails.append(f"{rel}:{n.lineno}: {name}: "
                         f"undefined name {n.id!r}")
    return fails


def scan():
    fails = []
    pkg = os.path.join(ROOT, "vllm_plugin")
    for fname in sorted(os.listdir(pkg)):
        if not fname.endswith(".py"):
            continue
        fails += check_file(os.path.join(pkg, fname), f"vllm_plugin/{fname}")
    return fails


def main():
    fails = scan()
    if fails:
        print("[ast] FAIL")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("[ast] ALL PASS (no undefined names in vllm_plugin/*.py)")


if __name__ == "__main__":
    main()
