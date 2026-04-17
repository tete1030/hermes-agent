"""Tests for gateway turn navigation commands: /turns, /undo N, /resume-turn."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/turns", platform=Platform.TELEGRAM):
    source = SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
    )
    return MessageEvent(text=text, source=source)


def _sample_history():
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]


def _make_runner(history):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)

    session_entry = MagicMock()
    session_entry.session_id = "session_123"
    session_entry.last_prompt_tokens = 99

    store = MagicMock()
    store.get_or_create_session.return_value = session_entry
    store.load_transcript.return_value = history
    runner.session_store = store

    return runner, session_entry


@pytest.mark.asyncio
async def test_turns_lists_numbered_user_turns():
    runner, _ = _make_runner(_sample_history())
    event = _make_event("/turns")

    result = await runner._handle_turns_command(event)

    assert "User Turns" in result
    assert "1. first question" in result
    assert "2. second question" in result
    assert "3. third question" in result


@pytest.mark.asyncio
async def test_undo_with_count_truncates_multiple_turns():
    history = _sample_history()
    runner, session_entry = _make_runner(history)
    event = _make_event("/undo 2")

    result = await runner._handle_undo_command(event)

    assert "across 2 turns" in result
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "session_123",
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
    )
    assert session_entry.last_prompt_tokens == 0


@pytest.mark.asyncio
async def test_resume_turn_truncates_before_selected_turn():
    history = _sample_history()
    runner, session_entry = _make_runner(history)
    event = _make_event("/resume-turn 2")

    result = await runner._handle_resume_turn_command(event)

    assert "before turn `2`" in result
    runner.session_store.rewrite_transcript.assert_called_once_with(
        "session_123",
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ],
    )
    assert session_entry.last_prompt_tokens == 0


@pytest.mark.asyncio
async def test_resume_turn_without_arg_shows_usage_and_turns():
    runner, _ = _make_runner(_sample_history())
    event = _make_event("/resume-turn")

    result = await runner._handle_resume_turn_command(event)

    assert "Usage: `/resume-turn <turn_number>`" in result
    assert "User Turns" in result
