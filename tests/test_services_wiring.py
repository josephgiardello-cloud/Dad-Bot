"""
Test DadBot service wiring and integration of new core modules.
"""
import pytest
from dadbot.core.dadbot import DadBot
from dadbot.testing import make_test_dadbot

def test_tool_registry_wired():
    bot = make_test_dadbot()
    assert hasattr(bot, 'tool_registry')
    assert bot.tool_registry is not None

def test_metrics_wired():
    bot = make_test_dadbot()
    assert hasattr(bot, 'metrics')
    assert bot.metrics is not None

def test_smart_home_wired():
    bot = make_test_dadbot()
    assert hasattr(bot, 'smart_home')
    assert bot.smart_home is not None

def test_asr_tts_wired():
    bot = make_test_dadbot()
    assert hasattr(bot, 'asr')
    assert hasattr(bot, 'tts')
    assert bot.asr is not None
    assert bot.tts is not None
