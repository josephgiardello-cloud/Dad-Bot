"""Voice interaction component wrappers.

This module exposes stable function-level helpers while delegating
implementation to the media service.
"""

from dadbot.ui.media.service import MediaService


def render_voice_controls(bot):
    return MediaService().render_voice_controls(bot)


def render_realtime_voice_call(bot):
    return MediaService().render_realtime_voice_call(bot)


def render_reply_tts(bot, reply_text):
    return MediaService().render_reply_tts(bot, reply_text)


__all__ = ["render_realtime_voice_call", "render_reply_tts", "render_voice_controls"]
