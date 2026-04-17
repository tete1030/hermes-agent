"""Tests for CLI turn navigation commands: /turns, /undo N, /resume-turn."""

from tests.cli.test_cli_init import _make_cli


def _sample_history():
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
        {"role": "user", "content": "third question"},
        {"role": "assistant", "content": "third answer"},
    ]


def test_process_command_undo_with_count_removes_multiple_turns():
    cli = _make_cli()
    cli.conversation_history = _sample_history()

    cli.process_command("/undo 2")

    assert cli.conversation_history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_process_command_resume_turn_truncates_before_selected_turn():
    cli = _make_cli()
    cli.conversation_history = _sample_history()

    cli.process_command("/resume-turn 2")

    assert cli.conversation_history == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_process_command_turns_lists_numbered_user_turns(capsys):
    cli = _make_cli()
    cli.conversation_history = _sample_history()

    cli.process_command("/turns")
    output = capsys.readouterr().out

    assert "User Turns" in output
    assert "#  1  first question" in output
    assert "#  2  second question" in output
    assert "#  3  third question" in output


def test_resume_turn_without_arg_shows_usage_and_turns(capsys):
    cli = _make_cli()
    cli.conversation_history = _sample_history()

    cli.process_command("/resume-turn")
    output = capsys.readouterr().out

    assert "Usage: /resume-turn <turn_number>" in output
    assert "User Turns" in output
