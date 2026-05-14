"""Prompt-budget and token-counting mixin for RuntimeStateManager.

Extracted from dadbot/state.py to reduce that module's size and Halstead
complexity.  RuntimeStateManager inherits from this mixin so all public
interfaces remain identical.
"""
from __future__ import annotations

import hashlib


class _StatePromptBudgetMixin:
    """Token counting, prompt history trimming, and cache helpers.

    Requires that the concrete class provides:
    - self.bot
    - self._token_count_cache, self._message_token_cost_cache, self._prompt_history_cache
    - self.conversation_history()
    """

    def estimate_token_count_cached(self, text):
        normalized = str(text or "")
        cache_key = (str(self.bot.ACTIVE_MODEL or "").strip().lower(), normalized)
        cached = self._token_count_cache.get(cache_key)
        if cached is not None:
            return cached

        estimate = self.bot.estimate_token_count(normalized)
        if len(self._token_count_cache) >= 8192:
            self._token_count_cache.pop(next(iter(self._token_count_cache)))
        self._token_count_cache[cache_key] = estimate
        return estimate

    def message_token_cost(self, message):
        role = str((message or {}).get("role", ""))
        content = str((message or {}).get("content", ""))
        cache_key = (str(self.bot.ACTIVE_MODEL or "").strip().lower(), role, content)
        cached = self._message_token_cost_cache.get(cache_key)
        if cached is not None:
            return cached

        total_cost = 4 + self.estimate_token_count_cached(role) + self.estimate_token_count_cached(content)
        if len(self._message_token_cost_cache) >= 4096:
            self._message_token_cost_cache.pop(
                next(iter(self._message_token_cost_cache)),
            )
        self._message_token_cost_cache[cache_key] = total_cost
        return total_cost

    def prompt_history_cache_key(self, system_prompt, user_input, recent_history):
        signature = hashlib.sha1(
            "\n".join(
                f"{message.get('role', '')!s}\x1f{message.get('content', '')!s}"
                for message in recent_history
                if isinstance(message, dict)
            ).encode("utf-8"),
        ).hexdigest()
        return (
            str(self.bot.ACTIVE_MODEL or "").strip().lower(),
            self.bot.effective_context_token_budget(),
            int(self.bot.RESERVED_RESPONSE_TOKENS or 0),
            int(self.bot.MAX_HISTORY_MESSAGES_SCAN or 0),
            str(system_prompt or ""),
            str(user_input or ""),
            signature,
        )

    def get_cached_prompt_history(self, cache_key):
        cached = self._prompt_history_cache.get(cache_key)
        if cached is None:
            return None
        return [dict(message) for message in cached]

    def remember_prompt_history(self, cache_key, messages):
        snapshot = [dict(message) for message in messages if isinstance(message, dict)]
        if len(self._prompt_history_cache) >= 512:
            self._prompt_history_cache.pop(next(iter(self._prompt_history_cache)))
        self._prompt_history_cache[cache_key] = snapshot

    def prompt_history_token_budget(self, system_prompt, user_input):
        return max(
            0,
            self.bot.effective_context_token_budget()
            - self.bot.RESERVED_RESPONSE_TOKENS
            - self.bot.message_token_cost({"role": "system", "content": system_prompt})
            - self.bot.message_token_cost({"role": "user", "content": user_input}),
        )

    def trim_text_to_token_budget(self, text, token_budget):
        normalized = str(text or "")
        if token_budget <= 0:
            return ""
        tokenizer = self.bot.current_tokenizer(model_name=self.bot.ACTIVE_MODEL)
        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(normalized)
                if len(encoded) <= token_budget:
                    return normalized
                trimmed = tokenizer.decode(encoded[:token_budget]).rstrip()
                if trimmed and trimmed != normalized:
                    trimmed = trimmed.rstrip(" ,.;:") + "..."
                return trimmed
            except Exception:  # noqa: BLE001
                pass

        estimated_tokens = self.bot.estimate_token_count(normalized)
        if estimated_tokens <= token_budget:
            return normalized

        ratio = token_budget / max(1, estimated_tokens)
        cutoff = max(1, min(len(normalized), int(len(normalized) * ratio)))
        trimmed = normalized[:cutoff].rstrip()
        if trimmed and trimmed != normalized:
            trimmed = trimmed.rstrip(" ,.;:") + "..."
        return trimmed

    def trim_message_to_token_budget(self, message, token_budget):
        role = str(message.get("role", ""))
        overhead = 4 + self.bot.estimate_token_count(role)
        content_budget = max(1, token_budget - overhead)
        trimmed_content = self.bot.trim_text_to_token_budget(
            message.get("content", ""),
            content_budget,
        )
        return {
            **message,
            "content": trimmed_content,
        }

    def token_budgeted_prompt_history(self, system_prompt, user_input):
        recent_history = self.conversation_history()[-self.bot.MAX_HISTORY_MESSAGES_SCAN :]
        token_budget = self.prompt_history_token_budget(system_prompt, user_input)
        if token_budget <= 0 or not recent_history:
            return []

        cache_key = self.prompt_history_cache_key(
            system_prompt,
            user_input,
            recent_history,
        )
        cached = self.get_cached_prompt_history(cache_key)
        if cached is not None:
            return cached

        selected_messages = []
        remaining_budget = token_budget
        total_messages = len(recent_history)

        for index, message in enumerate(recent_history):
            remaining_messages = total_messages - index - 1
            reserve_for_rest = remaining_messages * 24
            allowed_budget = max(24, remaining_budget - reserve_for_rest)
            candidate = message
            message_cost = self.bot.message_token_cost(candidate)

            if message_cost > allowed_budget:
                candidate = self.bot.trim_message_to_token_budget(
                    candidate,
                    allowed_budget,
                )
                if not candidate.get("content"):
                    continue
                message_cost = self.bot.message_token_cost(candidate)

            if message_cost > remaining_budget:
                candidate = self.bot.trim_message_to_token_budget(
                    candidate,
                    remaining_budget,
                )
                if not candidate.get("content"):
                    break
                message_cost = self.bot.message_token_cost(candidate)
                if message_cost > remaining_budget:
                    break

            selected_messages.append(candidate)
            remaining_budget -= message_cost
            if remaining_budget <= 0:
                break

        self.remember_prompt_history(cache_key, selected_messages)
        return selected_messages
