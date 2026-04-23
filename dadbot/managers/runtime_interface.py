from __future__ import annotations

import uuid

from dadbot.contracts import DadBotContext, SupportsDadBotAccess


class RuntimeInterfaceManager:
	"""Owns interactive CLI chat loops for the DadBot facade."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	def chat_loop(self):
		self.bot.print_system_message("--- Dad is online a\u2764\ufe0f ---")
		self.bot.print_system_message("Type 'bye' or 'goodnight' to head out.")
		self.bot.print_system_message(self.bot.command_help_text())
		print()

		if not self.bot.ensure_ollama_ready():
			return

		self.bot.reset_session_state()
		opening = self.bot.opening_message(default_message="")
		if opening:
			self.bot.print_speaker_message("Dad", opening)

		while True:
			try:
				user_input = input("Tony: ").strip()
				if not user_input:
					continue

				dad_reply, should_end = self.bot.process_user_message(user_input)
				if dad_reply:
					self.bot.print_speaker_message("Dad", dad_reply)

				if should_end:
					break
			except KeyboardInterrupt:
				self.bot.persist_conversation()
				self.bot.print_speaker_message("Dad", self.bot.reply_finalization.append_signoff("Hey Tony, you okay? I'm right here."))
				break
			except EOFError:
				self.bot.persist_conversation()
				break
			except Exception:
				self.bot.print_speaker_message("Dad", self.bot.reply_finalization.append_signoff("Sorry buddy, something went a bit wonky there. Try again?"))

	def chat_loop_via_service(self, service_client, session_id=None):
		self.bot.print_system_message("--- Dad is online via Dad Bot API a\u2764\ufe0f ---")
		self.bot.print_system_message("Type 'bye' or 'goodnight' to head out.")
		self.bot.print_system_message(self.bot.command_help_text())
		print()

		session_key = str(session_id or f"cli-{uuid.uuid4().hex}")
		service_client.ensure_service_running(preferred_model=self.bot.MODEL_NAME)
		self.bot.reset_session_state()
		opening = self.bot.opening_message(default_message="")
		if opening:
			self.bot.print_speaker_message("Dad", opening)

		while True:
			try:
				user_input = input("Tony: ").strip()
				if not user_input:
					continue

				result = service_client.chat(
					session_key,
					user_input=user_input,
					requested_model=self.bot.MODEL_NAME,
				)
				if result.session_state:
					self.bot.load_session_state_snapshot(result.session_state)
				if result.active_model:
					self.bot.ACTIVE_MODEL = result.active_model

				if result.reply:
					self.bot.print_speaker_message("Dad", result.reply)

				if result.should_end:
					self.bot.persist_conversation()
					break
			except KeyboardInterrupt:
				self.bot.persist_conversation()
				self.bot.print_speaker_message("Dad", self.bot.reply_finalization.append_signoff("Hey Tony, you okay? I'm right here."))
				break
			except EOFError:
				self.bot.persist_conversation()
				break
			except Exception:
				self.bot.print_speaker_message("Dad", self.bot.reply_finalization.append_signoff("Sorry buddy, something went a bit wonky there. Try again?"))


__all__ = ["RuntimeInterfaceManager"]