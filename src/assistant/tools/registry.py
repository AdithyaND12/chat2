"""Assemble the full tool set for the assistant."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from assistant.scheduler.jobs import SchedulerManager
from assistant.tools.news import get_news
from assistant.tools.search import web_search
from assistant.tools.time_tools import create_time_tools
from assistant.tools.weather import get_weather


def build_tools(scheduler: SchedulerManager) -> list[BaseTool]:
    return [
        get_weather,
        get_news,
        web_search,
        *create_time_tools(scheduler),
    ]
