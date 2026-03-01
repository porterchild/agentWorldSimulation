from concurrent.futures import ThreadPoolExecutor, as_completed
import llm
import prompts
import config
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
            entity.spawn_children(self.scenario, budget)
            self.entities.append(entity)

        print("\nEntity tree:")
        for entity in self.entities:
            for line in entity.tree_lines():
                print(line)
        print()

    def step(self) -> str:
        self.turn += 1
        print(f"\n{'='*60}")
        print(f"  Turn {self.turn}")
        print(f"{'='*60}\n")

        # Top-level entities act in parallel — each reads the same world thread independently
        entity_actions = [None] * len(self.entities)
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
