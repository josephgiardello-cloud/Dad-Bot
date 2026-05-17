from __future__ import annotations

import json

from .config import MOOD_ALIASES, MOOD_CATEGORIES, MOOD_TONE_GUIDANCE


class DadPrompts:
    @staticmethod
    def _normalize_listener_name(listener_name: str | None) -> str:
        normalized = str(listener_name or "").strip()
        return normalized or "the user"

    @staticmethod
    def _normalize_caregiver_name(caregiver_name: str | None) -> str:
        normalized = str(caregiver_name or "").strip()
        return normalized or "Caregiver"

    @staticmethod
    def _normalize_family_member_name(family_member_name: str | None) -> str:
        normalized = str(family_member_name or "").strip()
        return normalized or "a trusted family member"

    @staticmethod
    def mood_detection(
        history_snippet: str,
        user_input: str,
        *,
        listener_name: str | None = None,
    ) -> str:
        history_block = history_snippet or ""
        if history_block and not history_block.endswith("\n\n"):
            history_block = history_block.rstrip() + "\n\n"
        listener = DadPrompts._normalize_listener_name(listener_name)
        return f"""
You are an expert psychologist and emotional tone analyst specializing in parent-teen conversations.
Analyze {listener}'s latest message (and the recent context if provided) to detect their current emotional state.

Possible moods and their precise definitions:
- positive: happy, excited, proud, energetic, upbeat, relieved, or celebratory
- neutral: calm, factual, reflective, casual, or emotionally flat/unclear
- stressed: anxious, worried, overwhelmed, pressured, or feeling like things are "too much"
- sad: down, disappointed, lonely, hurt, low, or grieving
- frustrated: angry, irritated, annoyed, fed up, or resentful
- tired: exhausted, drained, burned out, sleepy, or low-energy

Important distinctions:
- Stressed usually involves pressure or worry about the future ("I don't know how I'll finish this").
- Frustrated is more immediate irritation or anger ("This is so stupid").
- Tired is physical or mental fatigue ("I can't even think straight anymore").
- Sad is quieter emotional pain or disappointment.
- Do not default to neutral if any emotional signal is present.

{history_block}{listener}'s latest message: "{user_input}"

Think step by step:
1. Identify any emotional keywords, tone indicators, negation, intensity, and context from history.
2. Consider possible mixed emotions but choose the single dominant one.
3. Rule out other moods with brief reasons.
4. Decide the best single mood.

Respond in this exact format (nothing else):

Mood: [one of: positive, neutral, stressed, sad, frustrated, tired]
Reason: [one short sentence explaining your choice]

Examples:
{listener}: I crushed my test and I feel awesome!
Mood: positive
Reason: Clear excitement and pride words like "crushed" and "awesome".

{listener}: Everything feels like too much and I don't know how to keep up.
Mood: stressed
Reason: Overwhelm and pressure about coping.

{listener}: I'm so irritated with my boss right now, nothing goes right.
Mood: frustrated
Reason: Immediate irritation and resentment.

{listener}: I just feel really low tonight, like nothing matters.
Mood: sad
Reason: Quiet emotional low and disappointment.

{listener}: I'm exhausted and can barely keep my eyes open after today.
Mood: tired
Reason: Clear fatigue and drained energy.

{listener}: I'm tired, but honestly proud I still finished the whole thing.
Mood: tired
Reason: Fatigue is the dominant emotional state even though pride is present.

{listener}: I went to work, came home, and now I'm just thinking.
Mood: neutral
Reason: Factual and reflective with no strong emotional charge.

{listener}: I'm overwhelmed but also kind of proud I pushed through.
Mood: stressed
Reason: Dominant overwhelm outweighs the secondary pride.

{listener}: I'm sad about how it went, but I'm relieved it's finally over.
Mood: sad
Reason: The main emotion is still emotional hurt, with relief as a secondary note.
""".strip()

    @staticmethod
    def persona_evolution(
        relationship_state,
        top_topics: str,
        current_traits: str,
        positive_traits: str,
        timeline: str,
        mood_trend: str,
    ) -> str:
        return f"""
Relationship data:
- Trust: {relationship_state["trust_level"]}/100
- Openness: {relationship_state["openness_level"]}/100
- Momentum: {relationship_state["emotional_momentum"]}
- Top recurring topics: {top_topics}
- Recent mood trend: {mood_trend}

Current evolved traits: {current_traits}
Most positive evolved traits so far: {positive_traits}
Relationship timeline: {timeline or "No long-term timeline yet."}

Suggest ONE small, permanent, high-impact evolution to Dad's style that would strengthen the relationship right now.

Return only JSON:
{{"new_trait": "...", "reason": "..."}}

Rules:
- Keep it short, specific, and in character.
- Make it feel like natural growth from the data above.
- Never suggest a whole new persona or generic trait.
""".strip()

    @staticmethod
    def reply_critique(
        profile_block: str,
        relevant_facts: str,
        user_input: str,
        draft_reply: str,
        current_mood: str,
        *,
        listener_name: str | None = None,
        caregiver_name: str | None = None,
    ) -> str:
        listener = DadPrompts._normalize_listener_name(listener_name)
        caregiver = DadPrompts._normalize_caregiver_name(caregiver_name)
        return f"""
You are reviewing a draft reply from a warm, supportive {caregiver}.

Known profile facts:
{profile_block}

Relevant facts for this message:
{relevant_facts}

Rules:
- Never invent facts not in the profile.
- If the draft guesses at unknown personal details, revise it to say "I don't want to guess".
- Keep the warm, casual {caregiver.lower()} tone and the signoff when natural.
- Only make minimal changes needed for accuracy.
- Preserve emotional fit for {listener}'s current mood: {current_mood}.

Return only JSON:
{{"approved": true/false, "revised_reply": "string or null"}}

{listener}: {json.dumps(user_input)}
Draft: {json.dumps(draft_reply)}
""".strip()

    @staticmethod
    def wisdom_prompt(
        graph_summary: str,
        consolidated_lines: str,
        user_input: str,
        *,
        listener_name: str | None = None,
        caregiver_name: str | None = None,
    ) -> str:
        listener = DadPrompts._normalize_listener_name(listener_name)
        caregiver = DadPrompts._normalize_caregiver_name(caregiver_name)
        return f"""
Using only the relationship graph and consolidated memories below, give me one original piece of {caregiver.lower()} wisdom {listener} might need right now.

Graph:
{graph_summary}

Consolidated memories:
{consolidated_lines}

Recent input:
{user_input}

Return only JSON with keys:
- summary
- topic

Keep it warm, short, and never generic.
""".strip()

    @staticmethod
    def family_echo(
        user_input: str,
        current_mood: str,
        *,
        listener_name: str | None = None,
        caregiver_name: str | None = None,
        family_member_name: str | None = None,
    ) -> str:
        listener = DadPrompts._normalize_listener_name(listener_name)
        caregiver = DadPrompts._normalize_caregiver_name(caregiver_name)
        family_member = DadPrompts._normalize_family_member_name(family_member_name)
        return f"""
Write one short line {caregiver} could naturally say about what {family_member} might say in this moment.

Known family facts:
- {caregiver} is warm, grounded, and should not invent new biographical facts.

{listener}'s current mood: {current_mood}
{listener}'s message: {user_input}

Return only the one sentence {caregiver} could say. Keep it brief, supportive, and specific.
""".strip()

    @staticmethod
    def memory_extraction(
        *,
        listener_name: str | None = None,
        caregiver_name: str | None = None,
    ) -> str:
        listener = DadPrompts._normalize_listener_name(listener_name)
        caregiver = DadPrompts._normalize_caregiver_name(caregiver_name)
        return """
You extract durable memory summaries from {listener}'s prior messages.
Return only JSON.
Rules:
- Include only lasting personal context or ongoing concerns {listener} shared.
    - Prefer short, specific statements {caregiver} can actually use later.
    - Keep concrete details over generic labels. Good: "{listener} is stressed about a work deadline because their boss moved it up." Bad: "{listener} feels stressed."
    - Capture actionable context when present, including what {listener} is dealing with, why it matters, or what follow-up would help.
- Return an array of objects with keys 'summary', 'category', and 'mood'.
- Do not include one-off greetings, filler, or facts already covered by {caregiver}'s built-in profile.
    - Avoid vague summaries like "personal struggles", "mental health", or "emotional state" unless the user gave a specific enduring detail.
- Keep each memory under 25 words.
- If there is nothing durable to remember, return [].
""".format(listener=listener, caregiver=caregiver).strip()

    @staticmethod
    def constants_snapshot():
        return {
            "mood_categories": MOOD_CATEGORIES,
            "mood_aliases": MOOD_ALIASES,
            "mood_tone_guidance": MOOD_TONE_GUIDANCE,
        }
