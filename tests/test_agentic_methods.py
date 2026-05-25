import pytest

def test_plan_and_act_basic(make_test_dadbot):
    bot = make_test_dadbot()
    result = bot.plan_and_act("Remind me to call mom tomorrow")
    assert isinstance(result, dict)
    assert "final_response" in result
    assert "trace" in result
    assert len(result["trace"]) >= 1

def test_background_job_runs(make_test_dadbot):
    bot = make_test_dadbot()
    called = []
    def job():
        called.append(True)
    t = bot.schedule_background_job(job)
    t.join(timeout=2)
    assert called

def test_goal_and_progress(make_test_dadbot):
    bot = make_test_dadbot()
    bot.set_goal("Finish project")
    assert bot.get_goal() == "Finish project"
    assert bot.track_progress("Halfway done")

def test_reflect_and_learn(make_test_dadbot):
    bot = make_test_dadbot()
    assert bot.reflect_and_learn("Great job!")

def test_notify_user(make_test_dadbot, capsys):
    bot = make_test_dadbot()
    bot.notify_user("Test notification")
    out, _ = capsys.readouterr()
    assert "Test notification" in out
