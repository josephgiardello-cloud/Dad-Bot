from dadbot.contracts import DadBotContext, SupportsDadBotAccess

class MoodManager:
    """Tracks and analyzes bot/user mood trends."""
    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._mood_log = []

    def log_mood(self, mood):
        self._mood_log.append(mood)

    def recent_moods(self, limit=5):
        return self._mood_log[-limit:]

    def current_mood(self):
        return self._mood_log[-1] if self._mood_log else None
