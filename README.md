# 🎙️ Nova — a LangChain/LangGraph voice assistant

Nova is a local-first, always-listening voice assistant for macOS, in the spirit of
Alexa. It runs on your computer, listens for a wake word, transcribes what you say,
and answers through an OpenRouter-hosted LLM that can call real tools — weather, news,
web search, timers, alarms, and reminders.

Built with **LangChain**, **LangGraph**, and **Python 3.11+**.

## Features

- **Always-on wake word** — openWakeWord runs locally on CPU (ONNX, no cloud), with a
  configurable word (`hey_jarvis`, `alexa`, `hey_mycroft`, `hey_rhasspy`) and tunable
  sensitivity.
- **Voice pipeline** — energy-based adaptive VAD for end-of-speech, Deepgram Nova-2
  for speech-to-text (Whisper fallback), edge-tts neural voices for replies.
- **LangGraph agent** — a `StateGraph` tool-calling loop with a SQLite checkpointer,
  so conversation history survives restarts and multi-step tool chains work naturally.
- **Tools** — weather (Open-Meteo, no key), news (GNews → Hacker News fallback),
  web search (Tavily → SerpAPI → DuckDuckGo fallback), timers, alarms, reminders,
  current time.
- **Persistent scheduler** — APScheduler with an SQLAlchemy job store; timers and
  alarms survive restarts and fire spoken notifications (plus macOS notifications).
- **Streamlit dashboard** — live transcript, scheduled items, activity log, and
  controls for wake word, sensitivity, voice, and model.
- **Terminal mode** — `--cli` for a plain typed/voice session.

## Architecture

```
mic (sounddevice) ──► openWakeWord ──► energy VAD ──► Deepgram STT
                                                          │
                                              transcript ─┘
                                                   │
                                    LangGraph StateGraph (OpenRouter LLM)
                                    ┌──────────────┴───────────────┐
                                    │  agent ⇄ ToolNode (loop)     │
                                    │  respond ──► final text      │
                                    └──────────────┬───────────────┘
                                                   │
              ┌────────────┬────────────┬──────────┴─────┬──────────────┐
              ▼            ▼            ▼                ▼              ▼
          weather        news       web search     timer/alarm/     conversation
         (Open-Meteo)   (GNews)    (Tavily/DDG)    reminder (APScheduler)
                                                   │
                                    edge-tts ──► speakers + Streamlit UI
```

## Requirements

- macOS (tested on Apple Silicon) with a microphone and speakers
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` on your `PATH` for TTS playback (optional — falls back to `afplay`)
- API keys: `OPENROUTER_API_KEY` (required for the brain), `DEEPGRAM_API_KEY`
  (required for voice input). Everything else is optional with keyless fallbacks.

## Setup

```bash
git clone https://github.com/AdithyaND12/chat2.git
cd chat2
uv sync                       # install dependencies
cp .env.example .env          # then fill in your keys
uv run python -m assistant --setup-models   # download wake-word models (~9 MB)
```

`.env`:

```ini
OPENROUTER_API_KEY=sk-or-...        # required — the conversational brain
DEEPGRAM_API_KEY=...                # required — voice input
OPENROUTER_MODEL=anthropic/claude-sonnet-4-20250514
EDGE_VOICE=en-US-AriaNeural
WAKE_WORD=hey_jarvis
ASSISTANT_NAME=Nova
TIMEZONE=UTC
```

All settings, defaults, and optional keys are documented in
[`.env.example`](.env.example).

## Usage

```bash
uv run python -m assistant --dashboard   # Streamlit dashboard (default UI)
uv run python -m assistant --cli         # terminal session
uv run python -m assistant --setup-models
```

Or use the console scripts after `uv sync`:

```bash
uv run nova-dashboard
uv run nova
```

In the dashboard: toggle **Listening**, say *"Hey Jarvis, what's the weather in
London?"*, or type into the chat box. Use the **Scheduled** tab to add timers and
cancel jobs; the **Activity** tab shows every event and tool call.

CLI commands: `listen on` / `listen off` / `quit`.

## Tools available to the agent

| Tool | Backend | Key needed |
|---|---|---|
| `get_weather` | Open-Meteo geocoding + forecast | no |
| `get_news` | GNews, falling back to Hacker News | optional |
| `web_search` | Tavily → SerpAPI → DuckDuckGo HTML | optional |
| `set_timer` | APScheduler (SQLite-persisted) | no |
| `set_alarm` | natural-time parsing, optional daily repeat | no |
| `set_reminder` | spoken reminder at a time | no |
| `list_timers` / `cancel_timer` | job inspection | no |
| `current_time` | configured timezone | no |

## Project structure

```
src/assistant/
├── agent.py            # AssistantRuntime: wires audio, speech, brain, tools, scheduler
├── config.py           # pydantic-settings (reads .env)
├── cli.py              # terminal entrypoint
├── dashboard.py        # Streamlit app
├── audio/              # capture, openWakeWord, VAD, player, model downloader
├── speech/             # Deepgram/Whisper STT, edge-tts TTS
├── brain/              # LangGraph graph, OpenRouter client, prompts, memory
├── tools/              # tool registry + implementations
└── scheduler/          # APScheduler manager + SQLite store
tests/                  # 11 tests: graph, scheduler, time tools, VAD
```

## Development

```bash
uv run pytest                 # run tests
uvx ruff check src tests      # lint
uvx ruff format src tests     # format
```

## How the agent loop works

`brain/graph.py` compiles a `StateGraph(AgentState)`:

1. `agent` — the OpenRouter model is invoked with the conversation plus tool schemas.
2. If the response contains `tool_calls`, control moves to `tools` (a LangGraph
   `ToolNode`) which executes them and appends `ToolMessage`s.
3. The graph loops back to `agent` until the model returns plain text.
4. `respond` extracts the final text, which is spoken via edge-tts and stored in the
   transcript log.

State is checkpointed with `SqliteSaver` keyed by `thread_id`, so history and
in-flight context persist across restarts.

## Troubleshooting

- **Microphone permission** — macOS will prompt on first launch; allow it in
  *System Settings → Privacy & Security → Microphone*.
- **No voice input** — confirm `DEEPGRAM_API_KEY` is set; the dashboard shows
  `STT configured: ✅/❌`.
- **No spoken answers** — confirm `OPENROUTER_API_KEY` is set
  (`LLM configured: ✅/❌`).
- **Wake word never fires** — lower the sensitivity slider (or `WAKE_WORD_SENSITIVITY`)
  and check the model downloaded via `--setup-models`.
- **TTS silent** — install `ffmpeg` (`brew install ffmpeg`); without it Nova falls
  back to `afplay` for MP3 playback.

## License

MIT
