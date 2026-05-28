from dadbot.contracts import DadBotContext, SupportsDadBotAccess

class MemoryManager:
    """Manages bot memory storage, retrieval, and projection."""
    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._memory = {}

    def memory_projection(self):
        # Return a projection of current memory (stub: return all keys)
        return list(self._memory.keys())

    def store(self, key, value):
        self._memory[key] = value
        return True

    def delete(self, key):
        if key in self._memory:
            del self._memory[key]
            return True
        return False

    def should_do_daily_checkin(self):
        # Example: always True for now
        return True
