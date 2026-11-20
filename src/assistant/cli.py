"""Command-line entrypoint for Nova."""

from __future__ import annotations

import argparse


def _setup_models() -> int:
    from assistant.audio.models import WAKE_WORDS, ensure_core_models, ensure_wake_word

    ensure_core_models()
    for name in WAKE_WORDS:
        ensure_wake_word(name)
    print("All wake-word models downloaded to data/models.")
    return 0


def _run_cli() -> int:
    from assistant.agent import AssistantRuntime
    from assistant.config import settings
    from assistant.logging import get_logger, setup_logging

    setup_logging(settings.log_level)
    log = get_logger("assistant.cli")
    runtime = AssistantRuntime()
    runtime.start()

    print(f"\n{settings.assistant_name} is listening. Say '{settings.wake_word}' or type below.")
    print("Commands: 'quit' to exit, 'listen on/off' to toggle the mic.\n")
    try:
        while True:
            try:
                text = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit", "q"}:
                break
            if text.lower() in {"listen on", "listen"}:
                runtime.set_listening(True)
                continue
            if text.lower() in {"listen off", "mute"}:
                runtime.set_listening(False)
                continue
            result = runtime.process_text(text, source="text")
            if result is not None:
                print(f"{settings.assistant_name}: {result.answer}\n")
                log.debug("Tool calls: %s", result.tool_calls)
    finally:
        runtime.stop()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="nova", description="Nova voice assistant")
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode")
    parser.add_argument("--dashboard", action="store_true", help="Launch the Streamlit dashboard")
    parser.add_argument("--setup-models", action="store_true", help="Download wake-word models")
    args = parser.parse_args()

    if args.setup_models:
        raise SystemExit(_setup_models())
    if args.dashboard:
        from assistant.dashboard import main as dashboard_main

        dashboard_main()
        return
    raise SystemExit(_run_cli())


if __name__ == "__main__":
    main()
