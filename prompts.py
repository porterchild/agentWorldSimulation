"""
Prompt builders for each stage of the simulation.

Structured outputs (spawning) ask for JSON.
Prose outputs (acting, synthesizing) are free-form narrative.

The thinking parameter adds a one-sentence reasoning scaffold for non-thinking models.
Native thinking models should run with thinking=False to avoid duplicating tokens.
"""

THINKING_PREAMBLE = "Think through your reasoning carefully before responding.\n\n"


def _thinking(thinking: bool) -> str:
    return THINKING_PREAMBLE if thinking else ""


def scenario_to_entities(scenario: str) -> list[dict]:
    """Return messages asking the model to break a scenario into top-level entities."""
    return [
        {
            "role": "system",
            "content": (
                "You are a scenario analyst. Given a scenario, identify the key entities "
                "whose behavior would most shape how the scenario unfolds. "
                "Focus on actors with agency: organizations, individuals, markets, or institutions. "
                "Return a JSON array of objects with 'name' and 'description' fields. "
                "Aim for 3-6 top-level entities. Return only the JSON array, no other text."
            ),
        },
        {
            "role": "user",
            "content": f"Scenario: {scenario}",
        },
    ]


def entity_spawn_decision(
    entity_name: str,
    entity_desc: str,
    scenario: str,
    budget_remaining: int,
    thinking: bool = False,
) -> list[dict]:
    """Return messages asking whether this entity should spawn sub-entities."""
    return [
        {
            "role": "system",
            "content": (
                f"{_thinking(thinking)}"
                "You are deciding whether a simulated entity should be broken into sub-entities "
                "to improve simulation fidelity. "
                "Spawn sub-entities when the entity is complex, where "
                "different internal actors have meaningfully different perspectives, incentives, "
                "or information relevant to the scenario — and where those internal tensions would "
                "affect the entity's behavior. "
                "For example: a tech company might spawn an executive, a legal team, and a procurement team. Whatever is relevant to the scenario. "
                "Do NOT spawn sub-entities whose internal dynamics don't materially affect the scenario. "
                f"The remaining spawn budget is {budget_remaining}. Be proportionate to complexity. "
                "Return a JSON array of sub-entity objects with 'name' and 'description', "
                "or an empty array [] if no spawning is needed. Return only the JSON array, no other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Entity: {entity_name}\n"
                f"Description: {entity_desc}\n"
                f"Scenario: {scenario}"
            ),
        },
    ]


def entity_act(
    entity_name: str,
    entity_desc: str,
    world_thread: str,
    history: list[str],
    thinking: bool = False,
) -> list[dict]:
    """Return messages asking a leaf entity to act given the current world state."""
    history_text = (
        "\n\nYour recent actions:\n" + "\n---\n".join(history[-3:])
        if history
        else ""
    )
    world_text = f"\n\nCurrent world state:\n{world_thread}" if world_thread else ""

    return [
        {
            "role": "system",
            "content": (
                f"{_thinking(thinking)}"
                f"You are {entity_name}. {entity_desc} "
                "Respond in the first person. "
                "Describe your assessment of the situation and the actions or decisions "
                "you are taking. "
                "Your output is your public action in the world; internal deliberation stays private."
            ),
        },
        {
            "role": "user",
            "content": f"{world_text}{history_text}\n\nWhat are you doing and deciding right now?",
        },
    ]


def entity_coherentize(
    entity_name: str,
    entity_desc: str,
    sub_actions: list[tuple[str, str]],
    world_thread: str,
    thinking: bool = False,
) -> list[dict]:
    """Return messages asking an entity to synthesize its sub-entities' actions into a coherent voice."""
    sub_text = "\n\n".join(f"[{name}]\n{action}" for name, action in sub_actions)
    world_text = f"\n\nCurrent world state:\n{world_thread}" if world_thread else ""

    return [
        {
            "role": "system",
            "content": (
                f"{_thinking(thinking)}"
                f"You are {entity_name}. {entity_desc} "
                "The following are reports from your internal components. "
                "Synthesize them into a single coherent account of what you as an entity are doing — "
                "your decisions, actions, and posture toward the current situation. "
                f"Speak in the first person as {entity_name}, not as any individual sub-component. "
                "This output is your observable presence in the world."
            ),
        },
        {
            "role": "user",
            "content": f"{world_text}\n\nInternal reports:\n\n{sub_text}\n\nWhat is {entity_name} doing?",
        },
    ]


WORLD_THREAD_CONTEXT_CHARS = 2000  # how much of prior world thread to pass as context


def world_synthesize(
    entity_actions: list[tuple[str, str]],
    world_thread: str,
    turn_num: int,
    thinking: bool = False,
) -> list[dict]:
    """Return messages asking the world synthesizer to produce an updated world thread."""
    actions_text = "\n\n".join(f"[{name}]\n{action}" for name, action in entity_actions)

    # Truncate prior thread to avoid context blowup over long runs
    truncated = world_thread[-WORLD_THREAD_CONTEXT_CHARS:] if world_thread else ""
    prior_text = f"Prior world state:\n{truncated}\n\n" if truncated else ""

    return [
        {
            "role": "system",
            "content": (
                f"{_thinking(thinking)}"
                "You are the narrator of a world simulation. "
                "Write a focused narrative of what happened THIS turn — not a recap of all prior history. "
                "Write in the third person, past tense. Label the time period this turn covers. "
                "Describe how entities acted, how they interacted, and what changed. "
                "End with 2-3 sentences on the world's current state entering the next turn. "
                "Aim for 3-5 paragraphs. Do not pad or repeat prior events already in the world state."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Turn {turn_num}\n\n"
                f"{prior_text}"
                f"Entity actions this turn:\n\n{actions_text}\n\n"
                "Write the world narrative for this turn."
            ),
        },
    ]
