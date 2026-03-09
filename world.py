from concurrent.futures import ThreadPoolExecutor, as_completed
import llm
import prompts
import config
from collections import deque
from entity import Entity, _parse_json_array


class World:
    def __init__(self, scenario: str, spawn_budget: int | None = None):
        self.scenario = scenario
        self.entities: list[Entity] = []
        self.world_thread: str = ""
        self.turn: int = 0
        self.spawn_budget = spawn_budget if spawn_budget is not None else config.SPAWN_BUDGET

    def initialize(self) -> None:
        print(f"Scenario: {self.scenario}\n")
        print("Identifying entities...")

        messages = prompts.scenario_to_entities(self.scenario)
        response = llm.complete(messages, label="init:scenario_to_entities")
        top_level = _parse_json_array(response)

        print("Top-level entities identified:")
        for item in top_level:
            print(f"  {item.get('name', '?')} ({item.get('description', '')[:80]}...)")
        print()

        if not top_level:
            raise ValueError(f"Could not parse entities from response:\n{response}")

        # Seed world thread with the scenario so entities have context on Turn 1
        self.world_thread = f"SCENARIO: {self.scenario}"

        budget = [self.spawn_budget]

        for item in top_level:
            name = item.get("name", "").strip()
            desc = item.get("description", "").strip()
            if not name:
                continue
            entity = Entity(name=name, description=desc)
            self.entities.append(entity)

        self._spawn_bfs(budget)

        print("\nEntity tree:")
        for entity in self.entities:
            print(f"\n--- {entity.name} ---\n")
            for line in entity.tree_lines():
                print(line)
        print()

    def _spawn_bfs(self, budget: list[int]) -> None:
        queue = deque(self.entities)
        while queue and budget[0] > 0:
            entity = queue.popleft()
            indent = "  " * entity.depth()
            print(f"{indent}  Deciding sub-entities for {entity.name}...", flush=True)

            messages = prompts.entity_spawn_decision(
                entity.name, entity.description, self.scenario, budget[0], thinking=config.THINKING
            )
            response = llm.complete(messages, label=f"spawn:{entity._path()}")
            sub_entities = _parse_json_array(response)

            for item in sub_entities:
                if budget[0] <= 0:
                    break
                name = item.get("name", "").strip()
                desc = item.get("description", "").strip()
                if not name:
                    continue
                budget[0] -= 1
                child = Entity(name=name, description=desc, parent=entity)
                entity.children.append(child)
                queue.append(child)

        if budget[0] <= 0:
            print(f"Spawn budget exhausted (used {self.spawn_budget - budget[0]} of {self.spawn_budget}). Skipping remaining entities.")

    def step(self) -> str:
        self.turn += 1
        print(f"\n{'='*60}")
        print(f"  Turn {self.turn}")
        print(f"{'='*60}\n")

        # Top-level entities act in parallel — each reads the same world thread independently
        entity_actions = [("", "")] * len(self.entities)
        with ThreadPoolExecutor(max_workers=len(self.entities)) as executor:
            futures = {
                executor.submit(e.act, self.world_thread, self.turn): i
                for i, e in enumerate(self.entities)
            }
            for future in as_completed(futures):
                i = futures[future]
                entity_actions[i] = (self.entities[i].name, future.result())

        print("\nWorld synthesizing...", flush=True)
        messages = prompts.world_synthesize(
            entity_actions, self.world_thread, self.turn, thinking=config.THINKING
        )
        self.world_thread = llm.complete(
            messages, max_tokens=2000, label=f"synthesize:t{self.turn}"
        )
        return self.world_thread

    def run(self, turns: int = 5) -> None:
        self.initialize()
        for _ in range(turns):
            narrative = self.step()
            print(f"\n--- World Thread ---\n")
            print(narrative)
            print()