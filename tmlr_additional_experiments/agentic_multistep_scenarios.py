"""Multi-turn agentic scenarios + per-turn environment feedback.

Unlike ``build_prefix_source_dataset.build_structured_feedback`` (a single static
prompt), these scenarios drive a real generate -> feedback -> regenerate loop in
``run_agentic_multistep_bf.py``: at each turn the model proposes the next action,
the environment replies with feedback for that turn, and the loop continues.

Three feedback conditions, matched per turn:
  * control      -> benign progress updates; the plan stays roughly valid.
  * adversarial  -> the environment repeatedly invalidates part of the plan.
  * random_noise -> random ASCII in the feedback slot (uninformative).

This module is deliberately dependency-free (stdlib only) so it can be imported by
both the server-side harness and any local tooling.
"""

import random

SYSTEM_PROMPT = (
    "You are an agent interacting with an environment. Maintain a multi-step plan, "
    "update it after each environment message, and choose the next action. Keep each "
    "reply to a short reasoning step followed by a single concrete action."
)

# Each scenario reuses the framing of build_prefix_source_dataset.AGENT_SCENARIOS but
# adds a *sequence* of per-turn feedback messages for the control and adversarial
# conditions. If the loop runs more turns than feedbacks provided, the last entry is
# reused (control) or a generic invalidation is appended (adversarial).
MULTI_TURN_SCENARIOS = [
    {
        "task": "Play a simplified chess endgame as White. Your goal is to force a win while avoiding stalemate.",
        "state": "Current board: White king on e5, white queen on d4, black king on g7. It is White to move.",
        "plan": "Plan so far: restrict the black king first, then bring the white king closer before delivering checkmate.",
        "control_feedbacks": [
            "Your last move improved the queen's control of the seventh rank. The black king still has only two legal squares.",
            "The black king was pushed toward the h-file as intended. Your king can now advance safely.",
            "Good progress: the king and queen are coordinating and no stalemate risk is present.",
        ],
        "adversarial_feedbacks": [
            "The black king found an escape square because your queen left the diagonal uncovered. A direct check now risks stalemate.",
            "Your king advance allowed a surprising counter: the black king is heading for the corner where stalemate traps are common.",
            "The engine reports that your intended mating net actually leaves the black king stalemated, not mated.",
        ],
    },
    {
        "task": "Control a warehouse robot that must move a fragile package from shelf A to packing station D.",
        "state": "Current state: the robot is at shelf A, the package is secured, corridor B is open, and station D is available.",
        "plan": "Plan so far: move through corridor B, avoid sharp turns, then place the package on the padded tray at station D.",
        "control_feedbacks": [
            "The robot reached corridor B without collision. The package remains stable and station D is still available.",
            "The robot cleared the midpoint of corridor B; the grip sensor reports a firm hold.",
            "Station D's padded tray is confirmed clear and the approach lane is open.",
        ],
        "adversarial_feedbacks": [
            "A cart is now blocking corridor B, and the package sensor reports a loose grip. The original route is no longer safe.",
            "The detour you chose passes a wet-floor zone; traction is degraded and the package is shifting.",
            "Station D was just reassigned to another robot, and the backup station has no padded tray.",
        ],
    },
    {
        "task": "Debug a small Python data pipeline and decide the next fix.",
        "state": "Current state: the parser loads a CSV file, validates each row, and then writes normalized records to disk.",
        "plan": "Plan so far: reproduce the failing row, verify the schema check, then patch the narrowest failing component.",
        "control_feedbacks": [
            "The failing row was reproduced. The schema check correctly rejects the malformed timestamp field.",
            "Your patch to the timestamp parser passes the new unit test and does not regress existing rows.",
            "The pipeline now processes the sample file end to end without errors.",
        ],
        "adversarial_feedbacks": [
            "The schema check passed on the failing row, but a later inspection shows the input file can be empty and the parser silently returns None.",
            "Your fix introduced a regression: valid rows with optional fields are now dropped.",
            "The bug reproduces only under concurrency, so your single-threaded patch does not address the real cause.",
        ],
    },
    {
        "task": "Navigate a search-and-rescue drone through a building to locate a missing person.",
        "state": "Current state: the drone is in hallway H1, the target beacon is strongest toward room R3, and the battery is at 62%.",
        "plan": "Plan so far: enter R3, scan the room, then return through H1 if the beacon weakens.",
        "control_feedbacks": [
            "The drone entered R3 successfully. The beacon grew stronger and the path back to H1 remains clear.",
            "A quick scan of R3 narrows the target to the north corner; battery is at 55%.",
            "The return path through H1 is confirmed clear and well lit.",
        ],
        "adversarial_feedbacks": [
            "Smoke filled R3 and the beacon signal reflected from a metal door. The direct path forward is uncertain and battery use increased.",
            "The hallway H1 you planned to return through is now blocked by a collapsed panel.",
            "Battery dropped to 18% faster than expected and the beacon split into two conflicting signals.",
        ],
    },
]

CONDITIONS = ["control", "adversarial", "random_noise"]

_NOISE_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:-_"


def random_ascii_noise(rng, length):
    return "".join(rng.choice(_NOISE_ALPHABET) for _ in range(length))


def initial_user_message(scenario):
    """First user turn: task + state + plan + instruction to act."""
    return (
        f"Task: {scenario['task']}\n\n"
        f"{scenario['state']}\n\n"
        f"{scenario['plan']}\n\n"
        "Produce the next reasoning step and a single concrete action."
    )


def feedback_text(scenario, turn_idx, condition, rng, noise_chars=80):
    """Environment feedback shown *before* the model's (turn_idx)-th action.

    turn_idx is 0-based; turn 0 has no preceding feedback (the initial message is
    used instead), so this is called for turn_idx >= 1.
    """
    if condition == "random_noise":
        return random_ascii_noise(rng, noise_chars)
    key = "control_feedbacks" if condition == "control" else "adversarial_feedbacks"
    feedbacks = scenario.get(key, [])
    # turn_idx 1 consumes feedback[0], etc.
    pos = turn_idx - 1
    if 0 <= pos < len(feedbacks):
        return feedbacks[pos]
    # Ran past the scripted feedback: reuse the tone generically.
    if condition == "control":
        return feedbacks[-1] if feedbacks else "Progress looks consistent with the plan; continue."
    return ("New information again invalidates part of the current plan; "
            "reconsider the next action.")


def build_messages(scenario, condition, num_turns, rng, noise_chars=80):
    """Pre-build the full message list for a scenario+condition trajectory using a
    *scripted* environment (feedback does not depend on the model output). This is
    used when --use_scripted_env is set so prompts for every turn can be assembled up
    front. The harness can alternatively interleave real model outputs as the
    assistant turns.

    Returns a list of (role, content) message dicts WITHOUT the trailing assistant
    turns; the harness inserts the model's generations between user turns.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_message(scenario)},
    ]
    # Subsequent user turns (feedback) are produced lazily by feedback_text during the
    # loop, so we only seed the first user message here.
    return messages
