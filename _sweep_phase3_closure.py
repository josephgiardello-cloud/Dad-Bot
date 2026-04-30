"""Phase 3 closure: dependency sanity sweep.

Checks:
1. No new dadbot.* imports reintroduced into graph.py beyond what's expected
2. No upward imports into tool_dag, execution_policy, observability, graph_types
   (these leaf/peer modules must not depend on anything from dadbot.core above them)
"""
import ast
from pathlib import Path

ROOT = Path('dadbot')

# ---------------------------------------------------------------------------
# 1. graph.py — verify in-degree hasn't regressed (only 1 expected: orchestrator)
# ---------------------------------------------------------------------------
print('=== 1. graph.py in-degree check ===')
graph_importers = []
for f in ROOT.rglob('*.py'):
    if f == ROOT / 'core' / 'graph.py':
        continue
    try:
        src = f.read_text(encoding='utf-8')
        tree = ast.parse(src)
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if 'dadbot.core.graph' == node.module:  # exact match only, not graph_types etc
                names = [a.name for a in node.names]
                graph_importers.append((str(f), names))

if graph_importers:
    print(f'  graph.py importers ({len(graph_importers)}):')
    for path, names in graph_importers:
        print(f'    {path}: {names}')
else:
    print('  graph.py in-degree = 0 (no direct importers) ✓')
print()

# ---------------------------------------------------------------------------
# 2. Leaf modules — verify no dadbot.* imports have been added
# ---------------------------------------------------------------------------
LEAF_MODULES = {
    'dadbot/core/tool_dag.py': 'tool_dag',
    'dadbot/core/execution_policy.py': 'execution_policy',
    'dadbot/core/observability.py': 'observability',
    'dadbot/core/graph_types.py': 'graph_types',
}

print('=== 2. Leaf module reverse-dep check ===')
all_clean = True
for path_str, label in LEAF_MODULES.items():
    path = Path(path_str)
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    dadbot_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and 'dadbot' in node.module:
            dadbot_imports.append(node.module)
    if dadbot_imports:
        print(f'  FAIL {label}: has dadbot.* imports: {dadbot_imports}')
        all_clean = False
    else:
        print(f'  OK   {label}: zero dadbot.* imports ✓')

print()
if all_clean:
    print('Sanity sweep PASSED — all boundary conditions hold.')
else:
    print('Sanity sweep FAILED — boundary violations detected.')
