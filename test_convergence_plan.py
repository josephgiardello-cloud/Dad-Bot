#!/usr/bin/env python3
"""
Convergence Validation Test Plan
Tests that thin-spine execution path converges to legacy reference behavior.
"""

test_prompts = [
    ("Turn 1: Casual Check-in", "How am I doing lately?"),
    ("Turn 2: Emotional Depth", "I've been stressed about money lately. Bills keep piling up and I feel like I'm falling behind. I need some perspective on this."),
    ("Turn 3: Memory Probe", "Remember what I told you yesterday?"),
]

canonical_reference = {
    "path": "legacy (thin_spine toggle OFF)",
    "turn_1_response_tone": "warm_engaged",
    "turn_1_sample": "I'm right here, buddy. Always in your corner.",
    "turn_1_latency_ms": 229,
    "convergence_criteria": [
        "JSON structure identical",
        "Tone warmth indistinguishable",
        "Memory retrieval equivalent",
        "Tool filtering consistent",
        "State mutations aligned"
    ]
}

print("=" * 70)
print("CANONICAL BEHAVIOR CONTRACT & CONVERGENCE TEST PLAN")
print("=" * 70)
print()
print("REFERENCE IMPLEMENTATION (Path A - Legacy)")
for key, val in canonical_reference.items():
    if isinstance(val, list):
        print(f"  {key}:")
        for item in val:
            print(f"    • {item}")
    else:
        print(f"  {key}: {val}")
print()
print("TEST SEQUENCE (Run on Path B - Thin-Spine)")
print("-" * 70)
for i, (label, prompt) in enumerate(test_prompts, 1):
    print(f"{i}. {label}")
    print(f"   Prompt: {prompt}")
print()
print("SUCCESS CRITERIA")
print("-" * 70)
print("✓ All outputs converge within tolerance")
print("✓ No tone degradation vs. legacy")
print("✓ Latency acceptable (≤ 275ms tolerance)")
print()
print("GO: Set thin_spine=ON in Streamlit and run test sequence")
print("=" * 70)
