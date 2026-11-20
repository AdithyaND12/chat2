"""Streamlit dashboard for the Nova assistant."""

from __future__ import annotations

import os
import sys

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from assistant.agent import AssistantRuntime
from assistant.audio.models import WAKE_WORDS

st.set_page_config(page_title="Nova Assistant", page_icon="🎙️", layout="wide")
st_autorefresh(interval=3000, limit=None, key="nova-refresh")


@st.cache_resource
def get_runtime() -> AssistantRuntime:
    runtime = AssistantRuntime()
    runtime.start()
    return runtime


VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
]


def main() -> None:
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        os.path.abspath(__file__),
        "--server.headless=true",
        "--server.runOnSave=false",
    ]
    stcli.main()


runtime = get_runtime()
snap = runtime.snapshot()

st.title("🎙️ Nova Assistant")
st.caption(f"Model: `{snap['model']}` · Voice: `{snap['voice']}` · TZ: `{snap['timezone']}`")

with st.sidebar:
    st.header("Controls")
    st.write(f"**Mic:** {snap['mic_state']}")
    if snap["wake_word_enabled"]:
        st.write(f"**Wake word:** `{snap['wake_word']}`")
    else:
        st.write("**Wake word:** disabled (VAD only)")
    listening = st.checkbox("Listening", value=snap["listening"], key="listening_toggle")
    if listening != snap["listening"]:
        runtime.set_listening(listening)

    st.divider()
    st.subheader("Voice")
    default_voice = VOICES.index(snap["voice"]) if snap["voice"] in VOICES else 0
    voice = st.selectbox("TTS voice", VOICES, index=default_voice)
    if voice != snap["voice"]:
        runtime.set_voice(voice)
    if st.button("▶ Test voice"):
        runtime.test_tts()

    st.divider()
    st.subheader("Wake word")
    words = list(WAKE_WORDS)
    default_wake = words.index(snap["wake_word"]) if snap["wake_word"] in words else 0
    wake = st.selectbox("Wake word", words, index=default_wake)
    if wake != snap["wake_word"]:
        runtime.set_wake_word(wake)
    sensitivity = st.slider("Sensitivity", 0.10, 0.90, float(snap["sensitivity"]), step=0.05)
    if abs(sensitivity - float(snap["sensitivity"])) > 1e-6:
        runtime.set_sensitivity(sensitivity)
    silence = st.slider("Silence seconds", 0.5, 3.0, float(snap["silence_seconds"]), step=0.1)
    if abs(silence - float(snap["silence_seconds"])) > 1e-6:
        runtime.set_silence_seconds(silence)

    st.divider()
    st.subheader("LLM")
    model = st.text_input("OpenRouter model", value=snap["model"])
    if st.button("Apply model") and model != snap["model"]:
        runtime.reload_model(model)

    st.divider()
    st.subheader("Status")
    st.write(f"LLM configured: {'✅' if snap['llm_configured'] else '❌'}")
    st.write(f"STT configured: {'✅' if snap['stt_configured'] else '❌'}")

tab_chat, tab_jobs, tab_activity = st.tabs(["💬 Conversation", "⏰ Scheduled", "📋 Activity"])

with tab_chat:
    with st.form("say_something", clear_on_submit=True):
        typed = st.text_input("Type a message", placeholder="Ask Nova anything...")
        submitted = st.form_submit_button("Send")
    if submitted and typed.strip():
        runtime.submit_text(typed)

    if snap["last_tool_calls"]:
        with st.expander("Last tool calls", expanded=False):
            for call in snap["last_tool_calls"]:
                st.write(f"**{call.get('name')}** → `{call.get('args')}`")

    for item in snap["transcripts"]:
        with st.container():
            st.markdown(f"**You ({item['source']}):** {item['text']}")
            st.markdown(f"**Nova:** {item['response'] or '…'}")
            st.divider()

with tab_jobs:
    with st.form("add_timer", clear_on_submit=True):
        col1, col2, col3 = st.columns([2, 3, 2])
        seconds = col1.number_input("Seconds", min_value=1, max_value=86400, value=60, step=30)
        label = col2.text_input("Label", placeholder="pasta, meeting, ...")
        add = col3.form_submit_button("Add timer")
    if add:
        info = runtime.add_timer(int(seconds), label)
        st.toast(f"Timer {info['id']} scheduled")

    if not snap["jobs"]:
        st.info("No timers, alarms, or reminders scheduled.")
    for job in snap["jobs"]:
        col1, col2, col3, col4 = st.columns([2, 3, 2, 1])
        col1.write(f"**{job['label'] or job['kind']}**")
        col2.write(f"{job['kind']} · {job['fire_at']}{' · daily' if job['repeat'] else ''}")
        col3.code(job["id"])
        if col4.button("Cancel", key=f"cancel-{job['id']}"):
            runtime.cancel_job(job["id"])
            st.rerun()

with tab_activity:
    for event in reversed(snap["events"]):
        payload = event["payload"]
        text = payload.get("text") or payload.get("message") or payload.get("label") or str(payload)
        st.markdown(f"`{event['ts']}` **{event['kind']}** — {text}")
