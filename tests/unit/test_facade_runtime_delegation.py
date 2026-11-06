from types import SimpleNamespace

from dadbot.core.dadbot import DadBot


def test_conversation_history_delegates_to_runtime_state_manager():
    history = [{"role": "user", "content": "hello"}]
    bot = SimpleNamespace(
        runtime_state_manager=SimpleNamespace(conversation_history=lambda: history),
    )

    assert DadBot.conversation_history(bot) == history


def test_missing_facade_method_resolves_from_service_registry():
    provider = SimpleNamespace(select_active_consolidated_memories=lambda *args: [])
    registry = SimpleNamespace(get_provider=lambda name: provider if name == "select_active_consolidated_memories" else None)
    bot = SimpleNamespace(services=SimpleNamespace(registry=registry))

    resolved = DadBot.__getattr__(bot, "select_active_consolidated_memories")

    assert resolved == provider.select_active_consolidated_memories


def test_memory_store_facade_reads_and_replaces_canonical_projection():
    store = {"memories": []}
    replacements = []
    memory_manager = SimpleNamespace(
        memory_store=store,
        _storage=SimpleNamespace(
            replace_projection_via_canonical_event=lambda value: replacements.append(value),
        ),
    )
    bot = SimpleNamespace(memory_manager=memory_manager)

    assert DadBot.MEMORY_STORE.__get__(bot, DadBot) is store
    DadBot.MEMORY_STORE.__set__(bot, {"memories": [{"summary": "remembered"}]})

    assert replacements == [{"memories": [{"summary": "remembered"}]}]


def test_legacy_history_window_aliases_proxy_runtime_config():
    config = SimpleNamespace(
        recent_history_window=8,
        max_history_messages_scan=48,
    )
    bot = SimpleNamespace(config=config)

    assert DadBot.RECENT_HISTORY_WINDOW.__get__(bot, DadBot) == 8
    assert DadBot.MAX_HISTORY_MESSAGES_SCAN.__get__(bot, DadBot) == 48
    DadBot.RECENT_HISTORY_WINDOW.__set__(bot, 3)
    DadBot.MAX_HISTORY_MESSAGES_SCAN.__set__(bot, 24)

    assert config.recent_history_window == 3
    assert config.max_history_messages_scan == 24
