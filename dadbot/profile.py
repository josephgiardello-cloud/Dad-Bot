from __future__ import annotations

from datetime import date


class ProfileContextManager:
    """Owns profile templates, fact lookup, and fact-based reply validation."""

    def __init__(self, bot):
        self.bot = bot

    def age_on_date(self, birthdate, today=None):
        if today is None:
            today = date.today()

        years = today.year - birthdate.year
        if (today.month, today.day) < (birthdate.month, birthdate.day):
            years -= 1
        return years

    @staticmethod
    def ordinal(day):
        if 10 <= day % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        return f"{day}{suffix}"

    def format_long_date(self, value):
        return f"{value.strftime('%B')} {self.ordinal(value.day)}, {value.year}"

    @staticmethod
    def natural_list(items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"

    def template_context(self):
        return {
            "listener_name": self.bot.STYLE["listener_name"],
            "signoff": self.bot.STYLE["signoff"],
            "dad_birthdate_long": self.format_long_date(self.bot.DAD_BIRTHDATE),
            "dad_age": self.age_on_date(self.bot.DAD_BIRTHDATE),
            "dad_birthplace": self.bot.FAMILY["dad"]["birthplace"],
            "dad_moved_to": self.bot.FAMILY["dad"]["moved_to"],
            "dad_move_age": self.bot.FAMILY["dad"]["move_age"],
            "dad_childhood": self.bot.FAMILY["dad"]["childhood"],
            "dad_father_absence": self.bot.FAMILY["dad"]["father_absence"],
            "carrie_full_name": self.bot.FAMILY["carrie"]["full_name"],
            "carrie_nee": self.bot.FAMILY["carrie"]["nee"],
            "carrie_birthdate_long": self.format_long_date(self.bot.CARRIE_BIRTHDATE),
            "carrie_age": self.age_on_date(self.bot.CARRIE_BIRTHDATE),
            "tony_birthdate_long": self.format_long_date(self.bot.TONY_BIRTHDATE),
            "tony_age": self.age_on_date(self.bot.TONY_BIRTHDATE),
            "marriage_date_long": self.format_long_date(self.bot.MARRIAGE_DATE),
            "boarding_school": self.bot.EDUCATION["boarding_school"],
            "university": self.bot.EDUCATION["university"],
            "years_attended": self.bot.EDUCATION["years_attended"],
            "degree": self.bot.EDUCATION["degree"],
            "other_subjects_list": self.natural_list(self.bot.EDUCATION["other_subjects"]),
        }

    def render_template(self, template):
        return template.format_map(self.template_context())

    def profile_fact_catalog(self):
        return [
            {
                "id": fact_definition["id"],
                "text": self.render_template(fact_definition["text_template"]),
                "reply": self.render_template(fact_definition["reply_template"]),
            }
            for fact_definition in self.bot.FACT_DEFINITIONS
        ]

    def fact_map(self):
        return {fact["id"]: fact for fact in self.profile_fact_catalog()}

    def build_profile_block(self, fact_ids=None):
        facts = self.fact_map()
        selected_ids = fact_ids or [fact["id"] for fact in self.profile_fact_catalog()]
        unique_ids = []

        for fact_id in selected_ids:
            if fact_id not in unique_ids:
                unique_ids.append(fact_id)

        return "\n".join(f"- {facts[fact_id]['text']}" for fact_id in unique_ids)

    def matching_topics(self, user_input):
        message = str(user_input or "").lower()
        matches = []

        for rule in self.bot.TOPIC_RULES:
            if any(keyword in message for keyword in rule["keywords_any"]):
                matches.append(rule)

        if "born" in message and "where" in message and not any(rule["name"] == "dad_birthplace" for rule in matches):
            matches.append(next(rule for rule in self.bot.TOPIC_RULES if rule["name"] == "dad_birthplace"))

        return matches

    def relevant_fact_ids_for_input(self, user_input):
        fact_ids = list(self.bot.CORE_FACT_IDS)

        for rule in self.matching_topics(user_input):
            for fact_id in rule["fact_ids"]:
                if fact_id not in fact_ids:
                    fact_ids.append(fact_id)

        return fact_ids

    def get_fact_reply(self, user_input):
        facts = self.bot.fact_map()

        for rule in self.bot.matching_topics(user_input):
            if rule["direct"]:
                primary_fact = facts[rule["fact_ids"][0]]
                return primary_fact["reply"]

        return None

    def expected_tokens_for_fact_ids(self, fact_ids):
        facts = self.bot.fact_map()
        combined_tokens = set()

        for fact_id in fact_ids:
            fact = facts.get(fact_id)
            if fact is None:
                continue
            combined_tokens.update(self.bot.significant_tokens(fact.get("text", "")))
            combined_tokens.update(self.bot.significant_tokens(fact.get("reply", "")))

        return combined_tokens

    def response_has_expected_anchor(self, rule, reply):
        reply_lower = reply.lower()
        primary_fact_id = rule["fact_ids"][0] if rule.get("fact_ids") else None

        if primary_fact_id == "bio_summary":
            anchors = self.bot.expected_tokens_for_fact_ids(rule.get("fact_ids", []))
            overlap = anchors & self.bot.significant_tokens(reply)
            return len(overlap) >= 4

        if primary_fact_id == "dad_birthplace":
            birthplace = str(self.bot.FAMILY["dad"].get("birthplace", "")).lower()
            birthplace_tokens = self.bot.significant_tokens(birthplace)
            return "providence" in reply_lower and "rhode island" in reply_lower or len(birthplace_tokens & self.bot.significant_tokens(reply)) >= 2

        if primary_fact_id == "dad_age":
            return str(self.bot.DAD_BIRTHDATE.year) in reply_lower or str(self.bot.age_on_date(self.bot.DAD_BIRTHDATE)) in reply_lower

        if primary_fact_id == "carrie_age":
            return str(self.bot.CARRIE_BIRTHDATE.year) in reply_lower or str(self.bot.age_on_date(self.bot.CARRIE_BIRTHDATE)) in reply_lower

        if primary_fact_id == "tony_age":
            return str(self.bot.TONY_BIRTHDATE.year) in reply_lower or str(self.bot.age_on_date(self.bot.TONY_BIRTHDATE)) in reply_lower

        if primary_fact_id == "marriage":
            marriage_month = self.bot.MARRIAGE_DATE.strftime("%B").lower()
            return str(self.bot.MARRIAGE_DATE.year) in reply_lower and marriage_month in reply_lower

        if primary_fact_id == "boarding_school":
            school_tokens = self.bot.significant_tokens(str(self.bot.EDUCATION.get("boarding_school", "")))
            return len(school_tokens & self.bot.significant_tokens(reply)) >= 2

        if primary_fact_id == "university":
            university_tokens = self.bot.significant_tokens(str(self.bot.EDUCATION.get("university", "")))
            degree_tokens = self.bot.significant_tokens(str(self.bot.EDUCATION.get("degree", "")))
            overlap = (university_tokens | degree_tokens) & self.bot.significant_tokens(reply)
            return len(overlap) >= 2

        expected_tokens = self.bot.expected_tokens_for_fact_ids(rule.get("fact_ids", []))
        if not expected_tokens:
            return True

        overlap = expected_tokens & self.bot.significant_tokens(reply)
        minimum_overlap = 2 if len(expected_tokens) >= 4 else 1
        return len(overlap) >= minimum_overlap

    def validate_reply(self, user_input, reply):
        memory_reply = self.bot.get_memory_reply(user_input)
        if memory_reply is not None:
            return memory_reply

        fact_reply = self.bot.get_fact_reply(user_input)
        if fact_reply is not None:
            return fact_reply

        for rule in self.bot.matching_topics(user_input):
            if not self.bot.response_has_expected_anchor(rule, reply):
                facts = self.bot.fact_map()
                return facts[rule["fact_ids"][0]]["reply"]

        return reply


__all__ = ["ProfileContextManager"]
