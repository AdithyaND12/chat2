"""High-level assistant runtime tying audio, speech, brain, tools, and scheduler together."""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from assistant.audio.engine import AudioEngine
from assistant.audio.player import AudioPlayer
from assistant.brain.graph import AgentResult, AssistantGraph
from assistant.brain.llm import build_llm
from assistant.brain.memory import ConversationStore
from assistant.config import settings
from assistant.logging import get_logger, setup_logging
from assistant.scheduler.jobs import SchedulerManager
from assistant.scheduler.store import Database
from assistant.speech.stt import CompositeSTT
from assistant.speech.tts import TextToSpeech
from assistant.tools.registry import build_tools

log = get_logger(__name__)


class AssistantRuntime:
    def __init__(self) -> None:
        setup_logging(settings.log_level)
        settings.ensure_paths()

        self.db = Database(settings.db_path)
        self.memory = ConversationStore(self.db, session_id="default")
        self.scheduler = SchedulerManager(settings.db_path, self._on_job_fire)

        self.stt = None
        try:
            self.stt = CompositeSTT()
        except Exception as exc:
            log.warning("Speech-to-text unavailable: %s", exc)

        self.tts = TextToSpeech()
        self.player = AudioPlayer(self.tts)
        self.player.start()

        self.tools = build_tools(self.scheduler)
        self.graph: AssistantGraph | None = None
        if settings.has_llm:
            try:
                self.graph = AssistantGraph(
                    build_llm(),
                    self.tools,
                    settings.checkpoint_path,
                )
            except Exception:
                log.exception("Failed to initialize the language model")
        else:
            log.warning("No OPENROUTER_API_KEY set; conversational brain is disabled.")

        try:
            self.audio = AudioEngine(
                on_wake=self._on_wake,
                on_utterance=self._on_utterance,
                on_status=self._on_audio_status,
            )
        except Exception:
            log.exception("Wake-word model failed to load; using VAD-only listening")
            self.audio = AudioEngine(
                on_wake=self._on_wake,
                on_utterance=self._on_utterance,
                on_status=self._on_audio_status,
                wake_word="none",
            )

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="assistant")
        self._state: dict[str, Any] = {
            "mic_state": "idle",
            "listening": True,
            "last_transcript": "",
            "last_response": "",
            "last_tool_calls": [],
        }
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self.scheduler.start()
        self.audio.start()
        self._started = True
        self._emit("system", {"message": "Assistant started"})
        log.info("Assistant runtime started")

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self.audio.stop()
        self.scheduler.shutdown()
        self.player.shutdown()
        if self.graph is not None:
            self.graph.close()
        self._executor.shutdown(wait=False)
        self.db.close()
        log.info("Assistant runtime stopped")

    # ------------------------------------------------------------------- audio
    def _on_wake(self, label: str) -> None:
        self.player.stop()
        self._set_state(mic_state="recording")
        self._emit("wake", {"label": label})

    def _on_audio_status(self, state: str) -> None:
        self._set_state(mic_state=state)

    def _on_utterance(self, audio) -> None:
        self._executor.submit(self._transcribe_and_handle, audio)

    def _transcribe_and_handle(self, audio) -> None:
        if self.stt is None:
            self._emit("error", {"message": "Speech-to-text is not configured."})
            return
        try:
            text = self.stt.transcribe(audio, settings.sample_rate)
        except Exception as exc:
            log.exception("Transcription failed: %s", exc)
            self._emit("error", {"message": f"Transcription failed: {exc}"})
            return
        if not text:
            self._emit("info", {"message": "No speech detected."})
            return
        self._set_state(last_transcript=text)
        self.process_text(text, source="voice")

    # ------------------------------------------------------------------- brain
    def process_text(self, text: str, source: str = "text") -> AgentResult | None:
        with self._lock:
            self._set_state(last_transcript=text)
            self._emit("transcript", {"text": text, "source": source})

            if self.graph is None:
                answer = (
                    "I'm not connected to a language model yet. "
                    "Add OPENROUTER_API_KEY to your .env file and restart."
                )
                tool_calls: list[dict] = []
            else:
                try:
                    result = self.graph.run(text, session_id="default")
                    answer = result.answer
                    tool_calls = result.tool_calls
                except Exception as exc:
                    log.exception("Agent run failed: %s", exc)
                    answer = "Sorry, something went wrong while I was thinking."
                    tool_calls = []

            self.memory.log(text, response=answer, source=source)
            self._set_state(last_response=answer, last_tool_calls=tool_calls)
            self._emit("response", {"text": answer, "tool_calls": tool_calls, "source": source})
            self.player.enqueue(answer)
            return AgentResult(answer=answer, tool_calls=tool_calls)

    def submit_text(self, text: str) -> None:
        if not text.strip():
            return
        self._executor.submit(self.process_text, text, "text")

    # --------------------------------------------------------------- scheduler
    def _on_job_fire(self, kind: str, job_id: str, label: str, text: str) -> None:
        self.player.enqueue(text)
        self._emit("job_fired", {"kind": kind, "id": job_id, "label": label, "text": text})
        self._notify(settings.assistant_name, text)

    def add_timer(self, seconds: int, label: str = "") -> dict:
        return asdict(self.scheduler.add_timer(seconds, label))

    def cancel_job(self, job_id: str) -> bool:
        return self.scheduler.cancel(job_id)

    def list_jobs(self) -> list[dict]:
        return [asdict(job) for job in self.scheduler.list_jobs()]

    # --------------------------------------------------------------------- ui
    def set_listening(self, enabled: bool) -> None:
        self.audio.set_listening(enabled)
        self._set_state(listening=enabled)

    def set_wake_word(self, word: str) -> None:
        settings.wake_word = word
        try:
            self.audio.stop()
            self.audio = AudioEngine(
                on_wake=self._on_wake,
                on_utterance=self._on_utterance,
                on_status=self._on_audio_status,
                wake_word=word,
            )
            if self._started:
                self.audio.start()
        except Exception:
            log.exception("Could not switch wake word to %s", word)
            self.audio = AudioEngine(
                on_wake=self._on_wake,
                on_utterance=self._on_utterance,
                on_status=self._on_audio_status,
                wake_word="none",
            )
            if self._started:
                self.audio.start()
        self._emit("config", {"wake_word": word})

    def reload_model(self, model: str) -> None:
        settings.openrouter_model = model
        if self.graph is not None:
            self.graph.close()
        self.graph = AssistantGraph(build_llm(), self.tools, settings.checkpoint_path)
        self._emit("config", {"model": model})

    def set_voice(self, voice: str) -> None:
        settings.edge_voice = voice
        self.tts = TextToSpeech(voice=voice)
        self.player.tts_engine = self.tts
        self._emit("config", {"voice": voice})

    def set_sensitivity(self, value: float) -> None:
        settings.wake_word_sensitivity = value
        if self.audio.detector is not None:
            self.audio.detector.threshold = value

    def set_silence_seconds(self, value: float) -> None:
        settings.silence_seconds = value
        self.audio.vad.silence_frames = max(1, int(value * 1000 / settings.vad_frame_ms))

    def test_tts(self, text: str = "Hi, I'm Nova, your assistant.") -> None:
        self.player.enqueue(text)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mic_state": self.audio.state,
                "listening": self.audio.listening,
                "wake_word_enabled": self.audio.wake_word_enabled,
                "wake_word": settings.wake_word,
                "sensitivity": settings.wake_word_sensitivity,
                "silence_seconds": settings.silence_seconds,
                "model": settings.openrouter_model,
                "voice": settings.edge_voice,
                "timezone": settings.timezone,
                "llm_configured": settings.has_llm,
                "stt_configured": self.stt is not None,
                "started": self._started,
                **self._state,
                "jobs": self.list_jobs(),
                "events": self.db.events(limit=50),
                "transcripts": self.db.transcripts(limit=50),
            }

    def _set_state(self, **updates: Any) -> None:
        with self._lock:
            self._state.update(updates)

    def _emit(self, kind: str, payload: dict) -> None:
        self.db.add_event(kind, payload)

    def _notify(self, title: str, message: str) -> None:
        if not settings.notifications_enabled:
            return
        try:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=False,
                timeout=5,
            )
        except Exception:
            log.debug("Could not show notification", exc_info=True)
