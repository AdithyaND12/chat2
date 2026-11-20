from datetime import datetime

from assistant.scheduler.jobs import SchedulerManager, parse_when


def test_parse_relative():
    result = parse_when("in 5 minutes")
    start = datetime.now(result.tzinfo).replace(second=0, microsecond=0)
    assert result >= start


def test_parse_time_only_rolls_to_next_day_when_past():
    result = parse_when("12:00am")
    assert result > datetime.now(result.tzinfo).replace(second=0, microsecond=0)


def test_timer_lifecycle(tmp_path):
    manager = SchedulerManager(str(tmp_path / "jobs.db"), lambda *args: None)
    manager.start()
    try:
        job = manager.add_timer(60, "pasta")
        assert job.kind == "timer"
        assert job.label == "pasta"
        jobs = manager.list_jobs()
        assert [item.id for item in jobs] == [job.id]
        assert manager.cancel(job.id) is True
        assert manager.list_jobs() == []
    finally:
        manager.shutdown()


def test_jobs_persist_across_restart(tmp_path):
    db = str(tmp_path / "jobs.db")
    first = SchedulerManager(db, lambda *args: None)
    first.start()
    job = first.add_timer(120, "tea")
    first.shutdown()

    second = SchedulerManager(db, lambda *args: None)
    second.start()
    try:
        assert [item.id for item in second.list_jobs()] == [job.id]
    finally:
        second.shutdown()
