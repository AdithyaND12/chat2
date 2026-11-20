"""APScheduler-backed timers, alarms, and reminders with SQLite persistence."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from assistant.config import settings
from assistant.logging import get_logger

log = get_logger(__name__)

FireCallback = Callable[[str, str, str, str], None]

# APScheduler's SQLAlchemy job store pickles job functions by reference. A
# module-level dispatcher keeps the job picklable while still routing fires to
# the live runtime callback.
_FIRE_CALLBACK: FireCallback | None = None


def _dispatch_job_fire(kind: str, job_id: str, label: str, text: str) -> None:
    if _FIRE_CALLBACK is not None:
        _FIRE_CALLBACK(kind, job_id, label, text)


_RELATIVE_RE = re.compile(
    r"^\s*in\s+(\d+)\s*(second|seconds|minute|minutes|hour|hours|day|days)\s*$",
    re.IGNORECASE,
)


@dataclass
class JobInfo:
    id: str
    kind: str
    label: str
    fire_at: str
    repeat: bool
    status: str


def _timezone() -> ZoneInfo:
    try:
        return ZoneInfo(settings.timezone)
    except ZoneInfoNotFoundError:
        log.warning("Unknown timezone %s; using UTC", settings.timezone)
        return ZoneInfo("UTC")


def parse_when(text: str) -> datetime:
    """Parse natural-language times like '7am', 'in 5 minutes', 'tomorrow at 6pm'."""
    tz = _timezone()
    now = datetime.now(tz)
    relative = _RELATIVE_RE.match(text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2).lower().rstrip("s")
        delta = {
            "second": timedelta(seconds=amount),
            "minute": timedelta(minutes=amount),
            "hour": timedelta(hours=amount),
            "day": timedelta(days=amount),
        }[unit]
        return now + delta

    from dateutil import parser as dateparser

    parsed = dateparser.parse(text, default=now, fuzzy=True)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tz)
    else:
        parsed = parsed.astimezone(tz)
    parsed = parsed.replace(second=0, microsecond=0)
    if parsed < now.replace(second=0, microsecond=0):
        parsed += timedelta(days=1)
    return parsed


class SchedulerManager:
    def __init__(self, db_path: str, on_fire: FireCallback) -> None:
        global _FIRE_CALLBACK
        self.tz = _timezone()
        self.on_fire = on_fire
        _FIRE_CALLBACK = on_fire
        stores = {"default": SQLAlchemyJobStore(url=f"sqlite:///{db_path}")}
        self.scheduler = BackgroundScheduler(jobstores=stores, timezone=self.tz)

    def start(self) -> None:
        self.scheduler.start()
        log.info("Scheduler started in %s", self.tz)

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def add_timer(self, seconds: int, label: str = "") -> JobInfo:
        job_id = uuid.uuid4().hex[:10]
        run_at = datetime.now(self.tz) + timedelta(seconds=max(1, int(seconds)))
        display = label.strip() or "timer"
        self.scheduler.add_job(
            _dispatch_job_fire,
            DateTrigger(run_date=run_at),
            args=["timer", job_id, display, f"Your {display} is done."],
            id=job_id,
            replace_existing=True,
            misfire_grace_time=120,
        )
        log.info("Timer %s set for %s from now", job_id, run_at.isoformat(timespec="seconds"))
        fire_at = run_at.isoformat(timespec="seconds")
        return JobInfo(job_id, "timer", display, fire_at, False, "pending")

    def add_alarm(
        self,
        when: datetime,
        label: str = "",
        repeat: bool = False,
        kind: str = "alarm",
    ) -> JobInfo:
        job_id = uuid.uuid4().hex[:10]
        display = label.strip() or kind
        if kind == "reminder":
            text = f"Reminder: {display}"
        else:
            text = f"Alarm: {display}" if display else "Your alarm is ringing."
        if repeat:
            trigger = CronTrigger(
                hour=when.hour,
                minute=when.minute,
                second=when.second,
                timezone=self.tz,
            )
            self.scheduler.add_job(
                _dispatch_job_fire,
                trigger,
                args=[kind, job_id, display, text],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300,
            )
        else:
            self.scheduler.add_job(
                _dispatch_job_fire,
                DateTrigger(run_date=when),
                args=[kind, job_id, display, text],
                id=job_id,
                replace_existing=True,
                misfire_grace_time=300,
            )
        next_run = self.scheduler.get_job(job_id).next_run_time
        fallback = when.isoformat(timespec="seconds")
        fire_at = next_run.isoformat(timespec="seconds") if next_run else fallback
        log.info("Scheduled %s %s at %s", kind, job_id, fire_at)
        return JobInfo(job_id, kind, display, fire_at, repeat, "pending")

    def list_jobs(self) -> list[JobInfo]:
        jobs: list[JobInfo] = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append(
                JobInfo(
                    id=job.id,
                    kind=str(job.args[0]) if job.args else "job",
                    label=str(job.args[2]) if len(job.args) > 2 else "",
                    fire_at=next_run.isoformat(timespec="seconds") if next_run else "",
                    repeat=job.trigger.__class__.__name__ == "CronTrigger",
                    status="pending" if next_run else "paused",
                )
            )
        return sorted(jobs, key=lambda item: item.fire_at)

    def cancel(self, job_id: str) -> bool:
        try:
            self.scheduler.remove_job(job_id)
            log.info("Cancelled job %s", job_id)
            return True
        except Exception as exc:
            log.warning("Could not cancel job %s: %s", job_id, exc)
            return False
