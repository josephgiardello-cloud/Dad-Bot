import logging
from dadbot.core.dadbot import DadBot

logging.basicConfig(level=logging.DEBUG)

bot = DadBot(
    memory_manager=None,
    relationship_manager=None,
    mood_manager=None,
    profile_runtime=None,
    event_bus=None,
)
print("Boot OK")

# Should not see: "EventTap not found", "Checkpoint load failed", "Replay mode enabled"
reply, should_end = bot.process_user_message("What can you do?")
print("Turn OK")
print("Reply:", reply)
