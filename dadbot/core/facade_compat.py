import warnings


class DadBotFacadeCompat:
    """Compatibility helper for deprecated facade aliases."""

    def __init__(self, deprecated_aliases: dict[str, str]):
        self._deprecated_aliases = dict(deprecated_aliases)

    def replacement_for(self, name: str) -> str | None:
        return self._deprecated_aliases.get(name)

    def warn_if_deprecated(self, name: str) -> None:
        replacement = self.replacement_for(name)
        if not replacement:
            return
        warnings.warn(
            f"DadBot.{name} is deprecated; prefer DadBot.{replacement}.",
            DeprecationWarning,
            stacklevel=3,
        )
