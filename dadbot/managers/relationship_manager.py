from dadbot.contracts import DadBotContext, SupportsDadBotAccess

class RelationshipManager:
    """Manages user relationship state and history."""
    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        self._relationships = {}

    def get_relationship(self, user_id):
        return self._relationships.get(user_id, {})

    def set_relationship(self, user_id, data):
        self._relationships[user_id] = data
        return True

    def all_relationships(self):
        return self._relationships.copy()
