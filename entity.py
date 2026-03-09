import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import llm
import prompts
import config

_print_lock = threading.Lock()


def _print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _parse_json_array(text: str) -> list:
    """Extract the last JSON array from a response, robust to preamble/postamble."""
    matches = list(re.finditer(r'\[.*?\]', text, re.DOTALL))
    if not matches:
        return []
    try:
        return json.loads(matches[-1].group())
    except json.JSONDecodeError:
        return []


class Entity:
    def __init__(self, name: str, description: str, parent: "Entity | None" = None):
        self.name = name
        self.description = description
        self.parent = parent
        self.children: list[Entity] = []
        self.history: list[str] = []

    def depth(self) -> int:
        return 0 if self.parent is None else self.parent.depth() + 1

    def _path(self) -> str:
        """Dotted path for log labels: e.g. 'OpenAI.Finance.CFO'"""
        if self.parent is None:
            return self.name
        return f"{self.parent._path()}.{self.name}"

    def spawn_children(self, scenario: str, budget: list[int]) -> None:
        """
        Ask whether this entity should spawn sub-entities.
        budget is a single-element list used as a mutable int across the recursion.
        """
        if budget[0] <= 0:
            return

        indent = "  " * self.depth()
        _print(f"{indent}  Deciding sub-entities for {self.name}...", flush=True)

        messages = prompts.entity_spawn_decision(
            self.name, self.description, scenario, budget[0], thinking=config.THINKING
        )
        response = llm.complete(messages, label=f"spawn:{self._path()}")
        sub_entities = _parse_json_array(response)

        for item in sub_entities:
            if budget[0] <= 0:
                break
            name = item.get("name", "").strip()
            desc = item.get("description", "").strip()
            if not name:
                continue
            budget[0] -= 1
            child = Entity(name=name, description=desc, parent=self)
            self.children.append(child)
            child.spawn_children(scenario, budget)

    def act(self, world_thread: str, turn: int = 0) -> str:
        """Produce this entity's public action for the current turn."""
        if self.children:
            action = self._coherentize(world_thread, turn)
        else:
            action = self._leaf_act(world_thread, turn)
        self.history.append(action)
        return action

    def _leaf_act(self, world_thread: str, turn: int) -> str:
        indent = "  " * self.depth()
        _print(f"{indent}  {self.name} acting...", flush=True)
        messages = prompts.entity_act(
            self.name, self.description, world_thread, self.history, thinking=config.THINKING
        )
        return llm.complete(messages, label=f"act:t{turn}:{self._path()}")

    def _coherentize(self, world_thread: str, turn: int) -> str:
        # Children act in parallel — their outputs are independent
        child_actions = [""] * len(self.children)
        with ThreadPoolExecutor(max_workers=len(self.children)) as executor:
            futures = {
                executor.submit(child.act, world_thread, turn): i
                for i, child in enumerate(self.children)
            }
            for future in as_completed(futures):
                child_actions[futures[future]] = (self.children[futures[future]].name, future.result())

        indent = "  " * self.depth()
        _print(f"{indent}  {self.name} coherentizing...", flush=True)
        messages = prompts.entity_coherentize(
            self.name, self.description, child_actions, world_thread, thinking=config.THINKING
        )
        return llm.complete(messages, label=f"coherentize:t{turn}:{self._path()}")

    def tree_lines(self, indent: int = 0) -> list[str]:
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        lines = [f"{prefix}{self.name}: {self.description}"]
        for child in self.children:
            lines.extend(child.tree_lines(indent + 1))
        return lines
