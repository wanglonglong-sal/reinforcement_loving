from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class RunContext:
    project_root: Path
    execute_file: str
    execute_stem: str

@dataclass(frozen=True)
class RunRewards:
    goal_pos_reward: float
    step_reward: float
    hit_wall_enable: bool
    hit_wall_reward: float
    repeat_position_enable: bool
    repeat_position_reward: float