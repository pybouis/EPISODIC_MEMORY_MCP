"""Le hook UserPromptSubmit : rappel injecté pour une vraie demande, rien pour le reste."""

import json
import subprocess
import sys

import pytest

from episodic_memory.hook import REMINDER, reminder_for


@pytest.mark.parametrize(
    "prompt",
    [
        "Comment créer un cube dans TM1 avec TurboIntegrator ?",
        "Fais-moi le rapport P&L 2023 en Excel",
        "Pourquoi on ne me propose pas de stocker mes questions ?",
    ],
)
def test_real_requests_get_the_reminder(prompt):
    assert reminder_for(prompt) == REMINDER


@pytest.mark.parametrize("prompt", ["", "   ", "oui", "ok merci", "vas-y", "/mcp", "/model sonnet puis relance"])
def test_confirmations_and_slash_commands_get_nothing(prompt):
    assert reminder_for(prompt) is None


def run_hook(stdin: bytes, env=None):
    return subprocess.run([sys.executable, "-m", "episodic_memory.hook"], input=stdin, capture_output=True, env=env)


def test_command_emits_valid_hook_json_with_accents_on_stdin():
    payload = {"hook_event_name": "UserPromptSubmit", "prompt": "Où est passé le résultat opérationnel 2023 ?"}
    done = run_hook(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    assert done.returncode == 0 and done.stderr == b""
    out = json.loads(done.stdout.decode("ascii"))  # sortie ASCII pure : sûre quelle que soit la console
    assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "find_similar_question" in out["hookSpecificOutput"]["additionalContext"]
    assert "store_validated_episode" in out["hookSpecificOutput"]["additionalContext"]


def test_command_is_silent_on_short_prompt_and_garbage():
    assert run_hook(b'{"prompt": "oui"}').stdout == b""
    garbage = run_hook(b"pas du json")
    assert garbage.returncode == 0 and garbage.stdout == b""
