from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from assistant.brain.graph import AssistantGraph


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}."


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls % 2 == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Austin"}, "id": "call_1"}],
            )
        return AIMessage(content="It is sunny in Austin.")


def test_graph_executes_tool_and_returns_answer(tmp_path):
    graph = AssistantGraph(FakeLLM(), [get_weather], str(tmp_path / "ckpt.sqlite"))
    try:
        result = graph.run("what is the weather in austin")
        assert result.answer == "It is sunny in Austin."
        assert result.tool_calls == [{"name": "get_weather", "args": {"city": "Austin"}}]
    finally:
        graph.close()


def test_graph_conversation_persists(tmp_path):
    llm = FakeLLM()
    graph = AssistantGraph(llm, [get_weather], str(tmp_path / "ckpt.sqlite"))
    try:
        graph.run("what is the weather in austin")
        config = {"configurable": {"thread_id": "default"}}
        state = graph.graph.get_state(config)
        assert len(state.values["messages"]) >= 2
    finally:
        graph.close()


def test_tool_trace_is_scoped_to_current_turn(tmp_path):
    llm = FakeLLM()
    graph = AssistantGraph(llm, [get_weather], str(tmp_path / "ckpt.sqlite"))
    try:
        first = graph.run("weather in austin")
        second = graph.run("weather in dallas")
        assert len(first.tool_calls) == 1
        assert len(second.tool_calls) == 1
    finally:
        graph.close()
