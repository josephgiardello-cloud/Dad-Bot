import ast
from collections import deque
from pathlib import Path

ROOT = Path(r"C:\Users\Josep\OneDrive\Desktop\Dad-Bot")
SRC_DIRS = [ROOT / "dadbot", ROOT / "dadbot_system"]


def read_safe(p):
    raw = p.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    return raw.decode("utf-8", errors="replace")


def module_to_path(mod):
    parts = mod.split(".")
    for base in SRC_DIRS:
        pkg = base.name
        rel = parts[1:] if parts[0] == pkg else parts[:]
        if not rel:
            init = base / "__init__.py"
            if init.exists():
                return init
            continue
        p = base.joinpath(*rel)
        if (p / "__init__.py").exists():
            return p / "__init__.py"
        py = p.parent / (rel[-1] + ".py")
        if py.exists():
            return py
    return None


def imports_from(path):
    try:
        tree = ast.parse(read_safe(path))
    except:
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(("dadbot", "dadbot_system")):
                mods.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("dadbot", "dadbot_system")):
                    mods.append(alias.name)
    return mods


seeds = [
    ROOT / "Dad.py",
    ROOT / "dadbot" / "core" / "orchestrator.py",
    ROOT / "dadbot" / "core" / "dadbot.py",
    ROOT / "dadbot" / "registry.py",
]
visited = set()
queue = deque(seeds)
while queue:
    p = queue.popleft()
    if not p.exists() or p in visited:
        continue
    visited.add(p)
    for mod in imports_from(p):
        dep = module_to_path(mod)
        if dep and dep not in visited:
            queue.append(dep)

all_src = set()
for d in SRC_DIRS:
    all_src.update(d.rglob("*.py"))
dead_src = all_src - visited


# Map module path -> dead?
def path_to_mod(p):
    for base in SRC_DIRS:
        try:
            rel = p.relative_to(base)
            parts = list(rel.parts)
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]  # strip .py
            return base.name + ("." if parts else "") + ".".join(parts)
        except ValueError:
            continue
    return None


dead_mods = {path_to_mod(p) for p in dead_src if path_to_mod(p)}

# Check test files
test_dir = ROOT / "tests"
tests = list(test_dir.rglob("*.py"))

# Find tests with top-level dead imports
broken_tests = {}
for tf in tests:
    top_dead = []
    try:
        src = read_safe(tf)
        tree = ast.parse(src)
    except:
        continue
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(("dadbot", "dadbot_system")):
                for dm in dead_mods:
                    if dm and (node.module == dm or node.module.startswith(dm + ".")):
                        top_dead.append(node.module)
                        break
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(dm and (alias.name == dm or alias.name.startswith(dm + ".")) for dm in dead_mods):
                    top_dead.append(alias.name)
    if top_dead:
        broken_tests[tf.name] = top_dead

print("=== DEAD SOURCE FILES ===")
for p in sorted(dead_src):
    print(p.relative_to(ROOT))

print()
print("=== TESTS WITH TOP-LEVEL DEAD IMPORTS ===")
for name, mods in sorted(broken_tests.items()):
    print(f"{name}: {', '.join(set(mods))}")
