from __future__ import annotations

from dadbot.contracts import AttachmentList, ChunkCallback, DadBotContext, SupportsTurnProcessingRuntime


class ReplyGenerationManager:
	"""Owns Ollama reply generation, supervision, validation, and finalization before turn persistence."""

	def __init__(self, bot: DadBotContext | SupportsTurnProcessingRuntime):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	def generate_validated_reply(
		self,
		stripped_input: str,
		turn_text: str,
		current_mood: str,
		normalized_attachments: AttachmentList,
		stream: bool = False,
		chunk_callback: ChunkCallback | None = None,
	) -> str:
		validation_input = stripped_input or turn_text
		if stream:
			raw_reply = self.bot.call_ollama_chat_stream(
				messages=self.bot.build_chat_request_messages(turn_text, current_mood, normalized_attachments),
				purpose="chat response",
				chunk_callback=chunk_callback,
			)
			reviewed_reply = raw_reply if self.bot.LIGHT_MODE else self.bot.critique_reply(stripped_input, raw_reply, current_mood)
		else:
			response = self.bot.call_ollama_chat(
				messages=self.bot.build_chat_request_messages(turn_text, current_mood, normalized_attachments),
				purpose="chat response",
			)
			reply_text = response["message"]["content"]
			reviewed_reply = reply_text if self.bot.LIGHT_MODE else self.bot.critique_reply(stripped_input, reply_text, current_mood)

		validated_reply = self.bot.validate_reply(validation_input, reviewed_reply)
		return self.bot.reply_finalization.finalize(validated_reply, current_mood, validation_input)

	async def generate_validated_reply_async(
		self,
		stripped_input: str,
		turn_text: str,
		current_mood: str,
		normalized_attachments: AttachmentList,
		stream: bool = False,
		chunk_callback: ChunkCallback | None = None,
	) -> str:
		validation_input = stripped_input or turn_text
		if stream:
			raw_reply = await self.bot.call_ollama_chat_stream_async(
				messages=self.bot.build_chat_request_messages(turn_text, current_mood, normalized_attachments),
				purpose="chat response",
				chunk_callback=chunk_callback,
			)
		else:
			response = await self.bot.call_ollama_chat_async(
				messages=self.bot.build_chat_request_messages(turn_text, current_mood, normalized_attachments),
				purpose="chat response",
			)
			raw_reply = response["message"]["content"]
		reviewed_reply = raw_reply if self.bot.LIGHT_MODE else await self.bot.critique_reply_async(stripped_input, raw_reply, current_mood)
		validated_reply = self.bot.validate_reply(validation_input, reviewed_reply)
		return await self.bot.reply_finalization.finalize_async(validated_reply, current_mood, validation_input)


__all__ = ["ReplyGenerationManager"]