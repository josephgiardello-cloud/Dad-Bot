import re

filename = "tests/test_phase4a.py"

with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

quote_lines = []
for i, line in enumerate(lines, start=1):
    if '"""' in line:
        quote_lines.append((i, line.strip()))

print(f"Found {len(quote_lines)} lines containing triple quotes.")
if len(quote_lines) % 2 != 0:
    print("WARNING: Odd number of lines with triple quotes. Unmatched delimiter likely.")
for i, (line_no, content) in enumerate(quote_lines):
    print(f"{line_no}: {content[:80]}")
