"""Canonical seed constants shared across all test modules.

Using named constants (not magic numbers) ensures that when a seed produces
an interesting failure, it can be referenced by name across the whole suite.
"""

BASELINE = 42          # Default happy-path seed
ADVERSARIAL = 1337     # Adversarial fuzz seed
REPLAY_A = 7777        # First replay pair seed
REPLAY_B = 7778        # Second replay pair seed (different input)
CHAOS_BASE = 100       # Base seed for chaos loops (add i for turn i)
MUTATION_FUZZ = 9999   # MutationFuzzer seed for mixed valid/invalid
CHECKPOINT = 2048      # Checkpoint hash-chain integrity seed
PARALLEL_MERGE = 4096  # Parallel fan-out merge seed
TEMPORAL_FREEZE = 8888 # VirtualClock / temporal freeze seed
PHASE_BOUNDARY = 3141  # Phase transition boundary seed
