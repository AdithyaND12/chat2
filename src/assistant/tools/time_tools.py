"""Timer, alarm, and reminder tools backed by the persistent scheduler."""

from __future__ import annotations

from datetime import datetime

from langchain_core.tools import BaseTool, tool

from assistant.logging import get_logger
from assistant.scheduler.jobs import JobInfo, SchedulerManager, parse_when

log = get_logger(__name__)


def _format_jobs(jobs: list[JobInfo]) -> str:
    if not jobs:
        return "You have no timers, alarms, or reminders right now."
    lines = ["Here are your scheduled items:"]
    for index, job in enumerate(jobs, start=1):
        repeat = " (repeats daily)" if job.repeat else ""
        label = job.label or job.kind
        lines.append(f"{index}. [{job.kind}] {label} at {job.fire_at}{repeat} (id {job.id})")
    return "\n".join(lines)


def create_time_tools(scheduler: SchedulerManager) -> list[BaseTool]:
    @tool
    def set_timer(seconds: int, label: str = "") -> str:
        """Set a countdown timer. `seconds` is the duration from now; `label` names it."""
        info = scheduler.add_timer(seconds, label)
        minutes = max(1, int(seconds)) // 60
        seconds_left = max(1, int(seconds)) % 60
        if minutes:
            duration = f"{minutes} minute{'s' if minutes != 1 else ''}"
            if seconds_left:
                duration += f" and {seconds_left} seconds"
        else:
            duration = f"{seconds_left} seconds"
        name = f" for {label}" if label.strip() else ""
        return f"Timer set{name} for {duration}. I'll let you know at {info.fire_at}."

    @tool
    def set_alarm(time: str, label: str = "", repeat_daily: bool = False) -> str:
        """Set an alarm for a time like '7am' or 'tomorrow at 6:45pm'. Repeat daily optionally."""
        try:
            when = parse_when(time)
        except Exception:
            return f"I couldn't understand {time!r}. Try '7am' or 'in 20 minutes'."
        info = scheduler.add_alarm(when, label, repeat=repeat_daily, kind="alarm")
        repeat = " daily" if repeat_daily else ""
        name = f" {label}" if label.strip() else ""
        return f"Alarm{name} set for {info.fire_at}{repeat}."

    @tool
    def set_reminder(time: str, text: str) -> str:
        """Set a spoken reminder for a time like 'in 10 minutes' or '6pm'."""
        try:
            when = parse_when(time)
        except Exception:
            return f"I couldn't understand {time!r}. Try 'in 10 minutes'."
        info = scheduler.add_alarm(when, text, repeat=False, kind="reminder")
        return f"Reminder set for {info.fire_at}: {text}."

    @tool
    def list_timers() -> str:
        """List all active timers, alarms, and reminders."""
        return _format_jobs(scheduler.list_jobs())

    @tool
    def cancel_timer(timer_id: str) -> str:
        """Cancel a timer, alarm, or reminder by its id from list_timers."""
        if scheduler.cancel(timer_id):
            return f"Cancelled {timer_id}."
        return f"I couldn't find a scheduled item with id {timer_id}."

    @tool
    def current_time() -> str:
        """Get the current date and time in the assistant's timezone."""
        zone = scheduler.tz
        now = datetime.now(zone)
        return now.strftime("It is %I:%M %p on %A, %B %d, %Y.")

    return [set_timer, set_alarm, set_reminder, list_timers, cancel_timer, current_time]
