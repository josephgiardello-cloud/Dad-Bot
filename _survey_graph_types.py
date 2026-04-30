"""Temporary survey script — graph_types importer analysis."""
import ast
from pathlib import Path

importers = {
    'dadbot/core/contracts/mutation.py': ['GoalMutationOp', 'LedgerMutationOp', 'MemoryMutationOp', 'MutationKind', 'RelationshipMutationOp'],
    'dadbot/core/graph.py': ['GoalMutationOp', 'LedgerMutationOp', 'MemoryMutationOp', 'MutationKind', 'MutationTransactionRecord', 'MutationTransactionStatus', 'NodeType', 'RelationshipMutationOp', 'StageTrace', '_json_safe'],
    'dadbot/core/graph_context.py': ['StageTrace', '_json_safe'],
    'dadbot/core/graph_mutation.py': ['GoalMutationOp', 'LedgerMutationOp', 'MemoryMutationOp', 'MutationKind', 'MutationTransactionRecord', 'MutationTransactionStatus', 'RelationshipMutationOp'],
    'dadbot/core/graph_pipeline_nodes.py': ['NodeType'],
    'dadbot/core/nodes.py': ['NodeType'],
    'dadbot/core/system_identity.py': ['GoalMutationOp', 'LedgerMutationOp', 'MemoryMutationOp', 'MutationKind', 'RelationshipMutationOp'],
}

for path, names in importers.items():
    src = Path(path).read_text(encoding='utf-8')
    lines = src.splitlines()
    tree = ast.parse(src)

    type_check_only = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and 'TYPE_CHECKING' in ast.dump(node.test):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom) and sub.module == 'dadbot.core.graph_types':
                    type_check_only = True

    tag = ' [TYPE_CHECKING]' if type_check_only else ''
    print(f'=== {path}{tag} ===')
    for name in names:
        usages = []
        for i, l in enumerate(lines):
            stripped = l.strip()
            if (name in l
                    and not stripped.startswith('#')
                    and not stripped.startswith('from ')
                    and not stripped.startswith('import ')):
                usages.append((i+1, stripped[:80]))
        if usages:
            print(f'  {name}: {len(usages)} runtime uses')
            for ln, text in usages[:3]:
                print(f'    line {ln}: {text}')
        else:
            print(f'  {name}: annotation/import-only')
    print()
