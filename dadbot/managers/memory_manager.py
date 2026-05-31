from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from datetime import date

class MemoryManager:
    """Manages bot memory storage, retrieval, and projection."""
    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._memory = {}

    def memory_projection(self):
        catalog = getattr(self.bot, "memory_catalog", None)
        if callable(catalog):
            return list(catalog() or [])
        return list(self._memory.keys())

    def store(self, key, value):
        key_text = str(key or "").strip()
        if not key_text:
            return False
        self._memory[key_text] = value

        add_memory = getattr(self.bot, "add_memory", None)
        if callable(add_memory):
            summary = f"{key_text}: {value}" if value is not None else key_text
            add_memory(summary)
        return True

    def delete(self, key):
        key_text = str(key or "").strip()
        if key_text in self._memory:
            del self._memory[key_text]

        forget = getattr(self.bot, "forget_memories", None)
        if callable(forget) and key_text:
            forget(key_text)
            return True
        return False

    def should_do_daily_checkin(self):
        today = date.today().isoformat()
        profile = getattr(self.bot, "PROFILE", None)
        if not isinstance(profile, dict):
            return True
        memory_meta = profile.setdefault("memory_manager", {})
        if not isinstance(memory_meta, dict):
            return True
        last_checkin = str(memory_meta.get("last_daily_checkin") or "")
        if last_checkin == today:
            return False
        memory_meta["last_daily_checkin"] = today
        save_profile = getattr(self.bot, "save_profile", None)
        if callable(save_profile):
            save_profile()
        return True
