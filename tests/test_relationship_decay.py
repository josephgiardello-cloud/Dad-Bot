from datetime import date, timedelta


def test_decay_relationship_level_returns_clamped_score_without_date(bot):
    assert bot.decay_relationship_level(140, None) == 100


def test_decay_relationship_level_moves_high_score_toward_midpoint(bot):
    last_updated = (date.today() - timedelta(days=10)).isoformat()

    decayed = bot.decay_relationship_level(90, last_updated)

    assert 50 < decayed < 90


def test_decay_relationship_level_moves_low_score_toward_midpoint(bot):
    last_updated = (date.today() - timedelta(days=10)).isoformat()

    decayed = bot.decay_relationship_level(10, last_updated)

    assert 10 < decayed < 50


def test_recency_weight_prefers_fresher_dates(bot):
    freshest = bot.recency_weight(date.today().isoformat())
    older = bot.recency_weight((date.today() - timedelta(days=6)).isoformat())

    assert freshest > older
    assert older == 1
