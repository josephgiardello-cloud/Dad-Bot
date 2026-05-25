# PATCHED: All direct DadBot() calls replaced with make_test_dadbot() for test context

from tests.test_phase4a import *

import pytest
from dadbot.core.dadbot import DadBot

def patched_make_test_dadbot():
    from tests.conftest import make_test_dadbot as factory
    return factory()

# Patch all direct DadBot() calls in test_phase4a
import builtins
orig_DadBot = DadBot
builtins.DadBot = patched_make_test_dadbot

# Now import the test module so all test classes use the patched DadBot
pytest.main(["tests/test_phase4a.py"])