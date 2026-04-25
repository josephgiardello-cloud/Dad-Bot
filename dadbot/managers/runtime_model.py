from __future__ import annotations

import importlib
import logging
import re

import ollama

from dadbot.contracts import DadBotContext, SupportsDadBotAccess

tiktoken = importlib.import_module("tiktoken") if importlib.util.find_spec("tiktoken") else None

logger = logging.getLogger(__name__)


class RuntimeModelManager:
	"""Owns model metadata lookup, tokenizer setup, and runtime model selection helpers."""

	def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
		self.context = DadBotContext.from_runtime(bot)
		self.bot = self.context.bot

	def ollama_show_payload(self, model_name=None):
		candidate = str(model_name or self.bot.ACTIVE_MODEL or self.bot.MODEL_NAME or "").strip().lower()
		if not candidate:
			return {}
		if candidate in self.bot._model_metadata_cache:
			return self.bot._model_metadata_cache[candidate]

		payload = {}
		try:
			response = ollama.show(candidate)
			if hasattr(response, "model_dump"):
				payload = response.model_dump()
			elif isinstance(response, dict):
				payload = dict(response)
		except self.bot.ollama_retryable_errors() as exc:
			logger.debug("Could not inspect Ollama model metadata for %s: %s", candidate, self.bot.ollama_error_summary(exc))
		except Exception as exc:
			logger.debug("Unexpected model metadata error for %s: %s", candidate, exc)

		self.bot._model_metadata_cache[candidate] = payload if isinstance(payload, dict) else {}
		return self.bot._model_metadata_cache[candidate]

	def model_context_length(self, model_name=None):
		payload = self.ollama_show_payload(model_name)
		modelinfo = payload.get("modelinfo") or payload.get("model_info") or {}
		context_lengths = []
		if isinstance(modelinfo, dict):
			for key, value in modelinfo.items():
				if "context_length" not in str(key or ""):
					continue
				try:
					parsed = int(value)
				except (TypeError, ValueError):
					continue
				if parsed > 0:
					context_lengths.append(parsed)

		parameters = str(payload.get("parameters") or "")
		num_ctx_match = re.search(r"\bnum_ctx\s+(\d+)\b", parameters)
		if num_ctx_match:
			context_lengths.append(int(num_ctx_match.group(1)))

		if context_lengths:
			return min(context_lengths)

		details = payload.get("details") or {}
		family = str((details.get("family") or "")).strip().lower()
		model_key = str(model_name or self.bot.ACTIVE_MODEL or self.bot.MODEL_NAME or "").strip().lower()
		heuristic_context = {
			"llama": 8192,
			"gemma": 8192,
			"mistral": 8192,
			"phi": 4096,
			"qwen": 32768,
		}
		for hint, estimate in heuristic_context.items():
			if family.startswith(hint) or model_key.startswith(hint):
				return estimate
		return None

	def effective_context_token_budget(self, model_name=None):
		configured_budget = max(1, int(self.bot.CONTEXT_TOKEN_BUDGET or 0))
		candidate_names = []
		if model_name is not None:
			candidate_names = [str(model_name or "").strip().lower()]
		else:
			for candidate in [self.bot.ACTIVE_MODEL, *self.model_candidates()]:
				normalized = str(candidate or "").strip().lower()
				if normalized and normalized not in candidate_names:
					candidate_names.append(normalized)

		known_contexts = [self.model_context_length(candidate) for candidate in candidate_names]
		known_contexts = [length for length in known_contexts if isinstance(length, int) and length > 0]
		if known_contexts:
			configured_budget = min(configured_budget, min(known_contexts))
		return max(1, configured_budget)

	def model_chars_per_token_estimate(self, model_name=None):
		payload = self.ollama_show_payload(model_name)
		modelinfo = payload.get("modelinfo") or payload.get("model_info") or {}
		details = payload.get("details") or {}
		tokenizer_pre = str(modelinfo.get("tokenizer.ggml.pre") or "").strip().lower()
		architecture = str(modelinfo.get("general.architecture") or details.get("family") or "").strip().lower()
		model_key = str(model_name or self.bot.ACTIVE_MODEL or self.bot.MODEL_NAME or "").strip().lower()

		if tokenizer_pre == "llama-bpe":
			return 3.25
		if "sentencepiece" in tokenizer_pre:
			return 3.0

		family_estimates = {
			"qwen": 2.9,
			"phi": 3.1,
			"gemma": 3.2,
			"llama": 3.4,
			"mistral": 3.4,
		}
		for hint, estimate in family_estimates.items():
			if architecture.startswith(hint) or model_key.startswith(hint):
				return estimate

		return max(1.0, float(self.bot.APPROX_CHARS_PER_TOKEN or 4))

	def resolve_tiktoken_encoding_name(self, model_name=None):
		if tiktoken is None:
			return None

		payload = self.ollama_show_payload(model_name)
		modelinfo = payload.get("modelinfo") or payload.get("model_info") or {}
		details = payload.get("details") or {}
		tokenizer_pre = str(modelinfo.get("tokenizer.ggml.pre") or "").strip().lower()
		tokenizer_model = str(modelinfo.get("tokenizer.ggml.model") or "").strip().lower()
		architecture = str(modelinfo.get("general.architecture") or details.get("family") or "").strip().lower()

		direct_map = {
			"o200k_base": "o200k_base",
			"cl100k_base": "cl100k_base",
			"p50k_base": "p50k_base",
			"r50k_base": "r50k_base",
		}
		if tokenizer_pre in direct_map:
			return direct_map[tokenizer_pre]
		if tokenizer_pre == "llama-bpe":
			return "o200k_base"
		if architecture.startswith(("llama", "qwen", "gemma", "mistral", "phi")):
			return "o200k_base"
		if tokenizer_model == "gpt2":
			return "cl100k_base"
		return None

	def initialize_tokenizer(self, model_name=None):
		if tiktoken is None:
			return None

		encoding_name = self.resolve_tiktoken_encoding_name(model_name)
		if not encoding_name:
			return None
		if encoding_name in self.bot._tokenizer_cache:
			return self.bot._tokenizer_cache[encoding_name]

		try:
			tokenizer = tiktoken.get_encoding(encoding_name)
		except Exception:
			return None

		self.bot._tokenizer_cache[encoding_name] = tokenizer
		return tokenizer

	def current_tokenizer(self, model_name=None):
		tokenizer = self.initialize_tokenizer(model_name or self.bot.ACTIVE_MODEL)
		if tokenizer is not None:
			self.bot._tokenizer = tokenizer
		return tokenizer

	def normalized_llm_provider(self) -> str:
		provider = str(getattr(self.bot, "LLM_PROVIDER", "ollama") or "ollama").strip().lower()
		return provider or "ollama"

	def normalized_llm_model(self, model_name: str | None = None) -> str:
		model = str(model_name or getattr(self.bot, "LLM_MODEL", "") or self.bot.MODEL_NAME).strip()
		return model or self.bot.MODEL_NAME

	@staticmethod
	def resolve_temperature(options: dict | None = None, default: float = 0.7) -> float:
		if isinstance(options, dict):
			try:
				return float(options.get("temperature", default))
			except (TypeError, ValueError):
				pass
		return float(default)

	def litellm_model_identifier(self, model_name: str | None = None) -> str:
		provider = self.normalized_llm_provider()
		model = self.normalized_llm_model(model_name)

		if "/" in model:
			return model

		provider_map = {
			"google": "gemini",
			"anthropic": "anthropic",
			"openai": "openai",
			"groq": "groq",
			"xai": "xai",
		}
		litellm_provider = provider_map.get(provider, provider)
		return f"{litellm_provider}/{model}"

	@staticmethod
	def extract_stream_chunk_content(chunk) -> str:
		if isinstance(chunk, dict):
			choices = chunk.get("choices")
			if isinstance(choices, list) and choices:
				delta = choices[0].get("delta") or choices[0].get("message") or {}
				return str(delta.get("content") or "")
			return ""

		choices = getattr(chunk, "choices", None)
		if isinstance(choices, list) and choices:
			delta = getattr(choices[0], "delta", None) or getattr(choices[0], "message", None)
			if delta:
				return str(getattr(delta, "content", "") or "")
		return ""

	@staticmethod
	def model_is_available(models, model_name):
		available_names = {
			model.get("model") or model.get("name")
			for model in models
			if model.get("model") or model.get("name")
		}

		if model_name in available_names:
			return True

		if ":" not in model_name:
			return any(name.startswith(f"{model_name}:") for name in available_names)

		return False

	def model_candidates(self):
		return [self.bot.MODEL_NAME, *self.bot.FALLBACK_MODELS]

	def dedicated_embedding_model_candidates(self):
		candidates = []
		for model_name in [self.bot.ACTIVE_EMBEDDING_MODEL, *self.bot.PREFERRED_EMBEDDING_MODELS]:
			if model_name and model_name not in candidates:
				candidates.append(model_name)
		return candidates

	def fallback_embedding_model_candidates(self):
		candidates = []
		for model_name in [self.bot.ACTIVE_MODEL, *self.model_candidates()]:
			if model_name and model_name not in candidates:
				candidates.append(model_name)
		return candidates

	def embedding_model_candidates(self):
		dedicated = self.dedicated_embedding_model_candidates()
		if dedicated:
			return dedicated
		return self.fallback_embedding_model_candidates()


__all__ = ["RuntimeModelManager"]
