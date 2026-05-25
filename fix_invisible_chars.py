import re

with open('tests/test_phase4a.py', 'rb') as f:
    data = f.read()

# Remove UTF-8 BOM, null bytes, and other non‑printable control characters (keep \n, \r, \t)
cleaned = re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f\xef\xbb\bf]', b'', data)

with open('tests/test_phase4a_fixed.py', 'wb') as f:
    f.write(cleaned)

print("Wrote tests/test_phase4a_fixed.py – review and then rename.")
