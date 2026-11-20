from langchain_core.tools import BaseTool

from assistant.scheduler.jobs import SchedulerManager
from assistant.tools.time_tools import create_time_tools


def _tool(name: str, scheduler: SchedulerManager) -> BaseTool:
    return {tool.name: tool for tool in create_time_tools(scheduler)}[name]


def test_set_list_and_cancel_timer(tmp_path):
    manager = SchedulerManager(str(tmp_path / "jobs.db"), lambda *args: None)
    manager.start()
    try:
        set_timer = _tool("set_timer", manager)
        list_timers = _tool("list_timers", manager)
        cancel_timer = _tool("cancel_timer", manager)

        confirmation = set_timer.invoke({"seconds": 90, "label": "pasta"})
        assert "Timer set" in confirmation
        assert "pasta" in confirmation

        listed = list_timers.invoke({})
        assert "pasta" in listed

        job_id = manager.list_jobs()[0].id
        assert cancel_timer.invoke({"timer_id": job_id}).startswith("Cancelled")
        assert manager.list_jobs() == []
    finally:
        manager.shutdown()


def test_set_alarm_and_reminder(tmp_path):
    manager = SchedulerManager(str(tmp_path / "jobs.db"), lambda *args: None)
    manager.start()
    try:
        set_alarm = _tool("set_alarm", manager)
        set_reminder = _tool("set_reminder", manager)
        assert "Alarm" in set_alarm.invoke({"time": "in 30 minutes", "label": "stand up"})
        assert "Reminder" in set_reminder.invoke({"time": "in 10 minutes", "text": "drink water"})
        assert len(manager.list_jobs()) == 2
    finally:
        manager.shutdown()
