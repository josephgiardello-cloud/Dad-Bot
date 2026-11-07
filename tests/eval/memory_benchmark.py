from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBenchmarkFact:
    topic: str
    summary: str
    category: str


@dataclass(frozen=True)
class MemoryBenchmarkCase:
    question: str
    expected_summary: str


MEMORY_BENCHMARK_FACTS = (
    MemoryBenchmarkFact("my dog's name", "My dog's name is Juniper.", "pets"),
    MemoryBenchmarkFact("my dog's breed", "My dog's breed is a beagle.", "pets"),
    MemoryBenchmarkFact("my cat's name", "My cat's name is Luna.", "pets"),
    MemoryBenchmarkFact("my cat's nap spot", "My cat's favorite nap spot is the sunny window.", "pets"),
    MemoryBenchmarkFact("my favorite color", "My favorite color is forest green.", "preferences"),
    MemoryBenchmarkFact("my favorite snack", "My favorite snack is roasted almonds.", "preferences"),
    MemoryBenchmarkFact("my coffee order", "My coffee order is a double espresso.", "preferences"),
    MemoryBenchmarkFact("my tea preference", "My tea preference is mint tea.", "preferences"),
    MemoryBenchmarkFact("my preferred season", "My preferred season is autumn.", "preferences"),
    MemoryBenchmarkFact("my favorite music genre", "My favorite music genre is jazz.", "preferences"),
    MemoryBenchmarkFact("my favorite book genre", "My favorite book genre is historical fiction.", "preferences"),
    MemoryBenchmarkFact("my movie preference", "My movie preference is nature documentaries.", "preferences"),
    MemoryBenchmarkFact("my weekend hobby", "My weekend hobby is woodworking.", "hobbies"),
    MemoryBenchmarkFact("my exercise routine", "My exercise routine is cycling three mornings a week.", "routines"),
    MemoryBenchmarkFact("my weekday wake-up time", "My weekday wake-up time is 6:30 AM.", "routines"),
    MemoryBenchmarkFact("my bedtime routine", "My bedtime routine includes reading for ten minutes.", "routines"),
    MemoryBenchmarkFact("my family dinner day", "My family dinner day is Thursday.", "family"),
    MemoryBenchmarkFact("my daughter's name", "My daughter's name is Maya.", "family"),
    MemoryBenchmarkFact("my son's favorite sport", "My son's favorite sport is baseball.", "family"),
    MemoryBenchmarkFact("my spouse's favorite dessert", "My spouse's favorite dessert is lemon pie.", "family"),
    MemoryBenchmarkFact("my anniversary month", "My anniversary month is June.", "family"),
    MemoryBenchmarkFact("my hometown", "My hometown is Madison.", "background"),
    MemoryBenchmarkFact("my home project", "My home project is repainting the kitchen cabinets.", "home"),
    MemoryBenchmarkFact("my garden plant", "My garden plant is basil.", "home"),
    MemoryBenchmarkFact("my car color", "My car color is silver.", "background"),
    MemoryBenchmarkFact("my commute method", "My commute method is the train.", "work"),
    MemoryBenchmarkFact("my work team", "My work team is platform engineering.", "work"),
    MemoryBenchmarkFact("my work project", "My work project is migrating the billing service.", "work"),
    MemoryBenchmarkFact("my career goal", "My career goal is to become a staff engineer.", "goals"),
    MemoryBenchmarkFact("my learning goal", "My learning goal is conversational Spanish.", "goals"),
    MemoryBenchmarkFact("my monthly savings target", "My monthly savings target is 200 dollars.", "goals"),
    MemoryBenchmarkFact("my vacation destination", "My vacation destination is Vancouver.", "travel"),
    MemoryBenchmarkFact("my packing preference", "My packing preference is to travel with one carry-on.", "travel"),
    MemoryBenchmarkFact(
        "my preferred cuisine",
        "My preferred cuisine is vegetarian Italian food.",
        "preferences",
    ),
    MemoryBenchmarkFact(
        "my board game preference",
        "My board game preference is cooperative strategy games.",
        "hobbies",
    ),
    MemoryBenchmarkFact("my musical instrument", "My musical instrument is the guitar.", "hobbies"),
    MemoryBenchmarkFact("the language I am learning", "The language I am learning is Japanese.", "goals"),
    MemoryBenchmarkFact("my charity cause", "My charity cause is the local food bank.", "community"),
    MemoryBenchmarkFact("my local walking route", "My local walking route goes through Riverside Park.", "routines"),
    MemoryBenchmarkFact("my weather preference", "My weather preference is cool, rainy mornings.", "preferences"),
    MemoryBenchmarkFact("my birthday month", "My birthday month is October.", "background"),
    MemoryBenchmarkFact("my reminder review day", "My reminder review day is Sunday.", "routines"),
    MemoryBenchmarkFact("my family tradition", "My family tradition is a pancake breakfast on holidays.", "family"),
    MemoryBenchmarkFact("my school pickup time", "My school pickup time is 3:15 PM.", "family"),
    MemoryBenchmarkFact("my morning drink", "My morning drink is black tea.", "routines"),
    MemoryBenchmarkFact("my notification time", "My preferred notification time is 8 PM.", "preferences"),
    MemoryBenchmarkFact("my appointment weekday", "My important appointment weekday is Tuesday.", "routines"),
    MemoryBenchmarkFact("my home repair task", "My home repair task is fixing the hallway light.", "home"),
    MemoryBenchmarkFact("my weekend chore", "My weekend chore is watering the plants on Saturday.", "home"),
    MemoryBenchmarkFact("my creative project", "My current creative project is building a birdhouse.", "hobbies"),
)

_QUESTION_FORMS = (
    "What did I tell you about {topic}?",
    "Can you recall {topic}?",
    "I mentioned {topic} earlier. What was it?",
    "Please remind me of {topic}.",
)


def memory_benchmark_cases() -> tuple[MemoryBenchmarkCase, ...]:
    return tuple(
        MemoryBenchmarkCase(
            question=question_form.format(topic=fact.topic),
            expected_summary=fact.summary,
        )
        for fact in MEMORY_BENCHMARK_FACTS
        for question_form in _QUESTION_FORMS
    )


# Engineering-authored paraphrases of synthetic facts, not family/user history.
_PARAPHRASE_TOPICS = (
    ("What do we call our dog?", "my dog's name"),
    ("What kind of dog do I have?", "my dog's breed"),
    ("What's the kitty called?", "my cat's name"),
    ("Where does the cat like to snooze?", "my cat's nap spot"),
    ("Which shade would I pick?", "my favorite color"),
    ("What do I like nibbling between meals?", "my favorite snack"),
    ("What should the barista make for me?", "my coffee order"),
    ("Which tea would I choose?", "my tea preference"),
    ("Which time of year do I enjoy most?", "my preferred season"),
    ("What kind of tunes am I into?", "my favorite music genre"),
    ("What sort of novels do I enjoy?", "my favorite book genre"),
    ("How do I spend my free weekends?", "my weekend hobby"),
    ("How do I stay physically active?", "my exercise routine"),
    ("When do I get out of bed on workdays?", "my weekday wake-up time"),
    ("What do I do before going to sleep?", "my bedtime routine"),
    ("When do we sit down to eat as a family?", "my family dinner day"),
    ("What's my little girl's name?", "my daughter's name"),
    ("Which sport does my boy like best?", "my son's favorite sport"),
    ("What sweet treat does my partner love?", "my spouse's favorite dessert"),
    ("Where did I grow up?", "my hometown"),
    ("What am I renovating at home?", "my home project"),
    ("How do I get to the office?", "my commute method"),
    ("Which group am I part of at my job?", "my work team"),
    ("What am I trying to achieve professionally?", "my career goal"),
    ("Where am I planning to go on holiday?", "my vacation destination"),
    ("What sort of films would I rather watch?", "my movie preference"),
    ("In which month do we celebrate our anniversary?", "my anniversary month"),
    ("Which herb am I growing?", "my garden plant"),
    ("What shade is my vehicle?", "my car color"),
    ("What are we delivering at my job right now?", "my work project"),
    ("Which language do I want to become conversational in?", "my learning goal"),
    ("How much money am I aiming to put aside each month?", "my monthly savings target"),
    ("How much luggage do I like to take with me?", "my packing preference"),
    ("What kind of meals would I pick at a restaurant?", "my preferred cuisine"),
    ("What sort of tabletop games do I like playing?", "my board game preference"),
    ("What do I play when I make music?", "my musical instrument"),
    ("Which language am I studying at the moment?", "the language I am learning"),
    ("Which community effort do I donate to?", "my charity cause"),
    ("Where do I go for a stroll nearby?", "my local walking route"),
    ("What kind of forecast makes me happy?", "my weather preference"),
    ("In which month was I born?", "my birthday month"),
    ("On what day do I go through my reminders?", "my reminder review day"),
    ("What do we always do together on special holidays?", "my family tradition"),
    ("When do I collect the children after class?", "my school pickup time"),
    ("What do I sip first thing after waking up?", "my morning drink"),
    ("At what hour do I prefer to receive alerts?", "my notification time"),
    ("Which day of the week is my important appointment?", "my appointment weekday"),
    ("What needs fixing around the house?", "my home repair task"),
    ("What household job do I do at the weekend?", "my weekend chore"),
    ("What am I making as my current craft project?", "my creative project"),
)


def memory_paraphrase_cases() -> tuple[MemoryBenchmarkCase, ...]:
    facts = {fact.topic: fact for fact in MEMORY_BENCHMARK_FACTS}
    return tuple(MemoryBenchmarkCase(question, facts[topic].summary) for question, topic in _PARAPHRASE_TOPICS)
