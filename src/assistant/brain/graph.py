"""LangGraph tool-calling agent graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from assistant.brain.prompts import build_system_prompt
from assistant.logging import get_logger

log = get_logger(__name__)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_trace: Annotated[list[dict[str, Any]], lambda left, right: (left or []) + (right or [])]
    final: str


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = "default"


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "respond"


class AssistantGraph:
    def __init__(
        self,
        llm,
        tools: list[BaseTool],
        checkpoint_path: str,
        system_prompt: str | None = None,
    ) -> None:
        self.tools = tools
        self.tool_node = ToolNode(tools)
        self.system_prompt = system_prompt or build_system_prompt()
        self.llm = llm.bind_tools(tools)

        self._checkpointer_cm = SqliteSaver.from_conn_string(checkpoint_path)
        self.checkpointer = self._checkpointer_cm.__enter__()

        builder = StateGraph(AgentState)
        builder.add_node("agent", self._call_model)
        builder.add_node("tools", self._execute_tools)
        builder.add_node("respond", self._respond)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", _route, {"tools": "tools", "respond": "respond"})
        builder.add_edge("tools", "agent")
        builder.add_edge("respond", END)
        self.graph = builder.compile(checkpointer=self.checkpointer)

    def _call_model(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt), *messages]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _execute_tools(self, state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        trace = [
            {"name": call.get("name"), "args": call.get("args", {})}
            for call in (getattr(last, "tool_calls", None) or [])
        ]
        result = self.tool_node.invoke(state)
        return {"messages": result.get("messages", []), "tool_trace": trace}

    def _respond(self, state: AgentState) -> dict[str, Any]:
        content = state["messages"][-1].content
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        return {"final": content or ""}

    def run(self, text: str, session_id: str = "default") -> AgentResult:
        config = {"configurable": {"thread_id": session_id}}
        try:
            before = len(self.graph.get_state(config).values.get("tool_trace") or [])
        except Exception:
            before = 0
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=text)]},
            config=config,
        )
        trace = list((result.get("tool_trace") or [])[before:]) if isinstance(result, dict) else []
        answer = result.get("final", "") if isinstance(result, dict) else ""
        return AgentResult(answer=str(answer), tool_calls=trace, session_id=session_id)

    def close(self) -> None:
        try:
            self._checkpointer_cm.__exit__(None, None, None)
        except Exception:
            log.debug("Failed to close checkpointer", exc_info=True)
