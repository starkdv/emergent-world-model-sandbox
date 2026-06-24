"""
Utility classes for reinforcement learning.

Contains helper classes that support the core learning system:
- Experience: Single experience tuple for RL
- ReplayBuffer: Experience replay buffer
- RewardShaper: Reward shaping for survival behaviors
- BestAgentTracker: Tracks and saves best performing agents

Author: Karan Vasa
Date: November 15, 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
from collections import deque
import os
import json

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World
    from agents.actions import Action, ActionResult


# ---------------------------------------------------------------------------
# Reward-shaping diet (World upgrade W6c)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """
    Selects the reward-shaping "diet" and its magnitudes.

    ``preset``:
      * ``legacy`` (default) — the full dense shaping that has been tuned
        across the Brain v2/v3 work (exploration, anti-loop, anti-spin,
        turn-toward-food, eat bonuses, …). Bit-identical to before W6c.
      * ``minimal`` — strips shaping down to **eat / death / energy-delta
        only**: a successful net-positive EAT is rewarded, dying is penalised,
        and per-step metabolism loss is a small penalty. Nothing else. This is
        the world-side analogue of fading the brain's instincts — it tests
        whether the world's own pressures (and curiosity) are enough signal to
        learn from, instead of hand-built dense rewards.

    The magnitudes are the headline terms the ``minimal`` diet uses; the
    ``legacy`` path keeps its own (extensively tuned) inline constants so it is
    provably unchanged.
    """

    preset: str = "legacy"
    eat_base: float = 5.0
    eat_energy_gain_coef: float = 0.2
    metabolism_penalty_coef: float = 0.01
    death_penalty: float = 10.0

    @classmethod
    def from_dict(cls, cfg: Optional[dict]) -> "RewardConfig":
        cfg = cfg or {}
        preset = str(cfg.get("preset", "legacy")).lower()
        if preset not in ("legacy", "minimal"):
            preset = "legacy"
        return cls(
            preset=preset,
            eat_base=float(cfg.get("eat_base", 5.0)),
            eat_energy_gain_coef=float(cfg.get("eat_energy_gain_coef", 0.2)),
            metabolism_penalty_coef=float(cfg.get("metabolism_penalty_coef", 0.01)),
            death_penalty=float(cfg.get("death_penalty", 10.0)),
        )


# Module-level active reward config (mirrors the observation active-spec
# pattern): main.py sets it once at startup from config['reward']; it defaults
# to the legacy diet so existing runs are unchanged.
_ACTIVE_REWARD_CONFIG = RewardConfig()


def get_active_reward_config() -> RewardConfig:
    """Return the reward config the simulation is currently using."""
    return _ACTIVE_REWARD_CONFIG


def set_active_reward_config(config: RewardConfig) -> None:
    """Set the active reward config (call once at startup)."""
    global _ACTIVE_REWARD_CONFIG
    _ACTIVE_REWARD_CONFIG = config


class Experience:
    """
    A single experience tuple for reinforcement learning with GRU hidden states.

    Stores the observation, hidden state, action, reward, and next observation/hidden state
    for learning from past decisions in a recurrent policy.
    """

    def __init__(
        self,
        observation: np.ndarray,
        hidden_state: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        next_hidden_state: np.ndarray,
        done: bool,
    ):
        self.observation = observation
        self.hidden_state = hidden_state
        self.action = action
        self.reward = reward
        self.next_observation = next_observation
        self.next_hidden_state = next_hidden_state
        self.done = done


class ReplayBuffer:
    """
    Experience replay buffer for stable learning.

    Stores recent experiences and allows sampling for learning updates.
    """

    def __init__(self, capacity: int = 1000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, experience: Experience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def clear(self) -> None:
        """Clear all experiences."""
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


class RewardShaper:
    """
    Shapes rewards to encourage survival behaviors.

    Provides intrinsic rewards for:
    - Finding and eating food (with energy-level bonuses)
    - Active exploration and movement
    - Moving toward food (dense reward)
    - Penalizing excessive waiting
    - Planting seeds (future benefit)
    - Heavy penalty for EAT spamming when no food present
    """

    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize reward shaper with tracking for distance rewards.

        Args:
            config: Reward diet (preset + magnitudes). Defaults to the active
                module-level config (legacy unless the run set it otherwise).
        """
        self.config = config if config is not None else get_active_reward_config()
        self.last_food_distance = None
        self.last_actions = []  # Track recent actions to detect spinning
        self.last_position = None  # Track position to detect movement
        self.recent_positions = deque(
            maxlen=24
        )  # Track recent positions to discourage loops
        self.forward_run_length = 0  # Consecutive successful forward moves
        self.blocked_forward_cooldown = (
            0  # Short memory of recent blocked-forward event
        )
        self.consecutive_waits = 0  # Track consecutive wait actions
        self.positions_visited = set()  # Track visited positions for exploration
        self.steps_without_movement = 0  # Track steps without position change
        self.consecutive_same_action = 0  # Track repeated same actions
        self.last_action = None  # Track last action for repetition detection
        self.consecutive_failed_eats = 0  # Track failed EAT attempts for spam detection
        self._recently_dropped_ids: set = set()  # Anti pick/drop spam
        self.last_food_dir_match = (
            0.5  # Track food direction for turn-toward-food reward
        )

    def calculate_reward(
        self,
        action: "Action",
        action_result: "ActionResult",
        energy_before: float,
        energy_after: float,
        agent: "Agent",
        world: "World",
    ) -> float:
        """
        Calculate shaped reward for an action.

        Uses DENSE REWARDS with EXPLORATION INCENTIVES:
        - Base survival reward per step
        - STRONG exploration bonus for movement
        - Energy-aware eating rewards (critical energy = big reward)
        - Distance-based reward (moving toward food)
        - Heavy penalty for excessive waiting

        Args:
            action: The action taken
            action_result: Result of the action
            energy_before: Energy before action
            energy_after: Energy after action
            agent: The agent
            world: The world

        Returns:
            Shaped reward value
        """
        from agents.actions import Action

        # ===== MINIMAL diet (W6c) =====
        # eat / death / energy-delta only — no dense shaping. The world's own
        # pressures (and curiosity, if enabled) must carry the learning signal.
        if self.config.preset == "minimal":
            energy_gain = energy_after - energy_before
            reward = 0.0
            if action == Action.EAT and action_result.success and energy_gain > 0:
                reward += (
                    self.config.eat_base
                    + energy_gain * self.config.eat_energy_gain_coef
                )
            if energy_gain < 0 and action != Action.EAT:
                reward -= abs(energy_gain) * self.config.metabolism_penalty_coef
            if not agent.alive:
                reward -= self.config.death_penalty
            return reward

        reward = 0.0

        prev_forward_run = self.forward_run_length

        # Track straight-run and blocked-forward context before shaping terms.
        if action == Action.MOVE_FORWARD:
            if action_result.success:
                self.forward_run_length += 1
            else:
                self.forward_run_length = 0
                self.blocked_forward_cooldown = 4
        elif action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            self.forward_run_length = 0
        elif action != Action.WAIT:
            self.forward_run_length = 0

        if action != Action.MOVE_FORWARD and self.blocked_forward_cooldown > 0:
            self.blocked_forward_cooldown -= 1

        # ===== DENSE REWARD #1: Base survival =====
        reward += 0.0  # Neutral baseline to avoid implicit idling incentive

        # ===== ANTI-SPAM: Mild penalty for repeating the same action =====
        if action == self.last_action:
            self.consecutive_same_action += 1
            if self.consecutive_same_action > 4:
                # Gentle progressive penalty (capped at -0.50)
                reward -= min(0.10 * (self.consecutive_same_action - 4), 0.50)
        else:
            self.consecutive_same_action = 0
        self.last_action = action

        # ===== EXPLORATION BONUS (REDUCED for balance) =====
        current_position = (agent.x, agent.y)
        prev_position = self.last_position
        moved_this_step = (
            prev_position is not None and current_position != prev_position
        )

        # Track if agent moved to a new position
        if prev_position is not None:
            if moved_this_step:
                # Agent moved! Reward active behavior
                self.steps_without_movement = 0
                self._recently_dropped_ids.clear()  # new tile, different objects

                # Movement should be mildly positive, but not so strong that
                # it dominates TURN decisions when no food signal exists.
                reward += 0.03

                # Extra bonus for visiting new tiles
                if current_position not in self.positions_visited:
                    reward += 0.04
                    self.positions_visited.add(current_position)

                    # Keep visited set from growing too large
                    if len(self.positions_visited) > 100:
                        self.positions_visited = set(list(self.positions_visited)[-50:])
            else:
                # Agent didn't move
                self.steps_without_movement += 1

                # Mild penalty for staying in place (only after many steps)
                if self.steps_without_movement > 8:
                    reward -= 0.03 * min(self.steps_without_movement - 8, 10)

        self.last_position = current_position

        # ===== ANTI-LOOP: discourage tight movement cycles =====
        # Only apply when we actually moved (avoid punishing WAIT/turn-in-place).
        if moved_this_step:
            immediate_backtrack = (
                len(self.recent_positions) >= 2
                and current_position == self.recent_positions[-2]
            )

            if immediate_backtrack:
                # Immediate A→B→A backtrack (common in rock-bounce loops)
                reward -= 0.10
            elif current_position in self.recent_positions:
                # Penalise short-horizon revisits more strongly than long-horizon
                # revisits to reduce repeated path cycling.
                recent_list = list(self.recent_positions)
                steps_ago = None
                for idx in range(len(recent_list) - 1, -1, -1):
                    if recent_list[idx] == current_position:
                        steps_ago = len(recent_list) - idx
                        break

                if steps_ago is None:
                    reward -= 0.04
                elif steps_ago <= 4:
                    reward -= 0.10
                elif steps_ago <= 8:
                    reward -= 0.07
                else:
                    reward -= 0.04

            # Low spatial novelty penalty: if we keep moving but only through
            # a tiny set of cells, nudge the policy to break route cycles.
            # This captures patterns that are not strict ABA/ABAB loops.
            recent_list = list(self.recent_positions)
            if len(recent_list) >= 10:
                recent_window = recent_list[-10:] + [current_position]
                unique_recent = len(set(recent_window))
                if unique_recent <= 4:
                    reward -= 0.08
                elif unique_recent <= 6:
                    reward -= 0.04

            self.recent_positions.append(current_position)

        # ===== WAIT: very gentle discouragement =====
        # Economic pressure comes from escalating energy cost in agent.py;
        # keep reward signal near-zero so WAIT remains a viable choice (~32-35%).
        if action == Action.WAIT:
            self.consecutive_waits += 1

            # Only penalise prolonged idling (capped at -0.15 extra)
            if self.consecutive_waits > 6:
                reward -= min(0.03 * (self.consecutive_waits - 6), 0.15)
        else:
            self.consecutive_waits = 0

        # ===== DENSE REWARD #2: Moving toward food =====
        nearest_food_dist = self._find_nearest_food_distance(agent, world)

        if nearest_food_dist is not None:
            # Reward for getting closer to food
            if self.last_food_distance is not None:
                distance_change = self.last_food_distance - nearest_food_dist
                if distance_change > 0:
                    # Got closer - reward scales with urgency
                    urgency_multiplier = 1.0
                    if agent.energy < agent.max_energy * 0.3:
                        urgency_multiplier = 3.0  # Critical energy = 3x reward
                    elif agent.energy < agent.max_energy * 0.5:
                        urgency_multiplier = 2.0  # Low energy = 2x reward

                    reward += 2.0 * distance_change * urgency_multiplier
                elif distance_change < 0:
                    # Got farther - penalty
                    reward -= 0.3 * abs(distance_change)

            # Proximity bonus (being close to food)
            if nearest_food_dist < 3.0:
                reward += 1.5 / (nearest_food_dist + 0.5)

            # On food tile bonus
            if nearest_food_dist < 1.0:
                reward += 3.0  # Strong signal: you're on food!

            self.last_food_distance = nearest_food_dist

        # ===== REST reward: WAIT when nothing salient nearby =====
        # When no food is in local range and energy isn't critical, a brief
        # WAIT can be a reasonable choice. This helps push WAIT frequency
        # toward the v3.1 target without incentivizing long idling streaks.
        if action == Action.WAIT and nearest_food_dist is None:
            energy_ratio = agent.energy / agent.max_energy if agent.max_energy else 0.0
            if energy_ratio > 0.6 and self.consecutive_waits <= 3:
                reward += 0.02

        # ===== TURN-TO-EXPLORE reward (when no food is visible) =====
        # When food isn't in local range, periodically turning helps avoid
        # long straight runs and reduces the "only turn when masked" behavior.
        if nearest_food_dist is None and action in [
            Action.TURN_LEFT,
            Action.TURN_RIGHT,
        ]:
            # Reward a turn if we've been mostly moving forward recently.
            recent = self.last_actions[-6:]
            if recent.count(Action.MOVE_FORWARD) >= 5:
                reward += 0.03

            # Stronger proactive turn reward after straight runs.
            # Only counts as proactive when there isn't a very recent blocked-forward event.
            if prev_forward_run >= 3 and self.blocked_forward_cooldown == 0:
                reward += min(0.12, 0.03 * (prev_forward_run - 2))

        # ===== STRAIGHT-RUN DAMPING =====
        # Mildly discourage very long forward runs when no food is visible.
        # This nudges agents to sample turns before hitting obstacles.
        if (
            nearest_food_dist is None
            and action == Action.MOVE_FORWARD
            and action_result.success
        ):
            if prev_forward_run >= 6:
                reward -= min(0.08, 0.01 * (prev_forward_run - 5))
        # ===== ENERGY-AWARE EATING REWARDS =====
        energy_gain = energy_after - energy_before

        if action == Action.EAT:
            if action_result.success:
                # Reset failed eat counter on success
                self.consecutive_failed_eats = 0

                # The eating bonuses are keyed off the *realised* energy gain,
                # not the food's identity: a successful eat that nets energy is
                # rewarded; eating something that nets zero/negative energy
                # (e.g. a toxic food, W3) only gets the proportional energy
                # term — which is negative — so the discrimination pressure is
                # emergent, never a scripted "poison = bad" rule (guideline §8).
                if energy_gain > 0:
                    # Base eating reward
                    reward += 5.0

                    # ENERGY-LEVEL MULTIPLIER for eating
                    energy_ratio = energy_before / agent.max_energy

                    if energy_ratio < 0.2:  # Critical (red) - HUGE reward
                        reward += 20.0
                    elif energy_ratio < 0.4:  # Low (orange/red)
                        reward += 15.0
                    elif energy_ratio < 0.6:  # Medium (yellow)
                        reward += 10.0
                    elif energy_ratio < 0.8:  # Good (green)
                        reward += 5.0
                    else:  # Full - still reward but less
                        reward += 2.0

                # Additional reward proportional to energy gained (negative
                # when the food was a net loss)
                reward += energy_gain * 0.2
            else:
                # FAILED EAT - Track and heavily penalize spam
                self.consecutive_failed_eats += 1

                # Escalating penalty for EAT spam
                # 1st fail: -2.5, 2nd: -3.5, 3rd: -4.5, etc.
                base_penalty = -2.5
                spam_multiplier = min(self.consecutive_failed_eats, 10)
                reward += base_penalty - (0.5 * (spam_multiplier - 1))

                # Extra harsh penalty if agent is just standing and spamming EAT
                if self.steps_without_movement > 2:
                    reward -= 1.0  # Should move to find food, not spam EAT
        else:
            # Reset failed eat counter when doing other actions
            self.consecutive_failed_eats = 0

        # Energy loss penalty (metabolism)
        if energy_gain < 0 and action != Action.EAT:
            reward -= abs(energy_gain) * 0.01  # Slight penalty for metabolism
        # ===== SUCCESS BONUSES =====
        if action_result.success:
            if "Picked up" in action_result.message:
                # Anti pick/drop spam: penalize re-picking an item we just dropped
                picked_id = getattr(action_result, "object_id", -1)
                if picked_id >= 0 and picked_id in self._recently_dropped_ids:
                    reward -= 0.80  # discourage pick/drop cycling
                    self._recently_dropped_ids.discard(picked_id)
                else:
                    reward += 1.0  # Found and picked up food/item!
            elif "Dropped" in action_result.message:
                dropped_id = getattr(action_result, "object_id", -1)
                if dropped_id >= 0:
                    self._recently_dropped_ids.add(dropped_id)
            elif "Planted" in action_result.message:
                reward += 2.0  # Good for ecosystem
            # (MOVE_FORWARD success is already rewarded by exploration bonus)
        else:
            # Penalty for failed actions (EAT handled separately above)
            if action == Action.PICK_UP:
                reward -= 0.8  # Increased penalty for failed pickup (was 0.5)
            elif action == Action.MOVE_FORWARD:
                reward -= 0.05  # Tiny penalty for bumping into walls
            elif action == Action.DROP:
                reward -= 0.3
            else:
                reward -= 0.05  # Small penalty for other failures        # ===== Inventory and hunger interaction =====
        from world.objects import EdibleComponent

        has_food_in_inventory = False
        food_count = 0

        if agent.inventory:
            for obj_id in agent.inventory:
                obj = world.objects.get(obj_id)
                if obj is not None and obj.has_component(EdibleComponent):
                    has_food_in_inventory = True
                    food_count += 1

        # Small bonus for having food (insurance)
        reward += food_count * 0.1

        # ===== CRITICAL: Encourage eating from inventory when hungry =====
        energy_ratio = agent.energy / agent.max_energy

        if has_food_in_inventory and energy_ratio < 0.5:
            # Agent has food and is hungry!
            if action == Action.EAT:
                # HUGE bonus for eating when you have food and need it
                if action_result.success:
                    if energy_ratio < 0.2:
                        reward += 15.0  # Critical: massive reward
                    elif energy_ratio < 0.35:
                        reward += 10.0  # Low: big reward
                    else:
                        reward += 5.0  # Medium: good reward
            else:
                # Penalty for NOT eating when you have food and need energy
                if energy_ratio < 0.2:
                    reward -= 2.0  # Critical: should be eating!
                elif energy_ratio < 0.35:
                    reward -= 1.0  # Low: really should eat
                else:
                    reward -= 0.3  # Medium: consider eating

        # ===== Anti-spinning penalty =====
        self.last_actions.append(action)
        if len(self.last_actions) > 8:
            self.last_actions.pop(0)

        if len(self.last_actions) >= 8:
            recent = self.last_actions[-8:]
            turn_count = sum(
                1 for a in recent if a in [Action.TURN_LEFT, Action.TURN_RIGHT]
            )
            move_count = sum(1 for a in recent if a == Action.MOVE_FORWARD)

            # Moderate anti-spin: capped to keep value function stable.
            # Heavy economic pressure already comes from escalating turn
            # energy cost in agent.execute_action.
            if turn_count >= 5 and move_count <= 1:
                reward -= 0.60
            elif turn_count >= 4:
                reward -= 0.30

        # ===== TURN-BALANCE regularization =====
        # Prevent the population from locking into a single turn
        # direction through evolutionary drift.  If recent turns are
        # heavily skewed L or R, mildly penalise the dominant one.
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            recent_turns = [
                a
                for a in self.last_actions
                if a in [Action.TURN_LEFT, Action.TURN_RIGHT]
            ]
            if len(recent_turns) >= 4:
                left_ct = recent_turns.count(Action.TURN_LEFT)
                right_ct = recent_turns.count(Action.TURN_RIGHT)
                # Penalise the dominant direction (ratio >= 4:1)
                if left_ct >= 4 * max(right_ct, 1) and action == Action.TURN_LEFT:
                    reward -= 0.06
                elif right_ct >= 4 * max(left_ct, 1) and action == Action.TURN_RIGHT:
                    reward -= 0.06

        # ===== TURN-TOWARD-FOOD reward =====
        # Reward turns that improve alignment with nearest food.
        # This gives turns a positive reward comparable to forward
        # exploration, preventing the "go straight until wall" pattern.
        food_dir_match = self._compute_food_dir_match(agent, world)
        if action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            dir_improvement = food_dir_match - self.last_food_dir_match
            if dir_improvement > 0.05:
                # Turned toward food — reward proportional to improvement
                reward += min(0.30, dir_improvement * 1.5)
        self.last_food_dir_match = food_dir_match

        # ===== Critical energy state urgency =====
        if agent.energy < agent.max_energy * 0.2:
            # Mild nudge against waiting at critical energy
            if action == Action.WAIT:
                reward -= 0.10

        # Death penalty
        if not agent.alive:
            reward -= 10.0

        return reward

    def _find_nearest_food_distance(
        self, agent: "Agent", world: "World"
    ) -> Optional[float]:
        """
        Find distance to nearest food.

        Args:
            agent: The agent
            world: The world

        Returns:
            Distance to nearest food, or None if no food exists.

        Notes:
            This function is on the reward hot-path. It must not scan all
            objects once the world has thousands of items.

            Implementation: bounded local tile scan around the agent.
            Returns a Manhattan distance (cheap) or None if no food is found.
        """
        if not hasattr(world, "tiles") or not hasattr(world, "nearest_edible"):
            return None

        # 10 => 21x21 = 441 tiles max per call. Delegated to the World helper,
        # which uses the W6a spatial index when present (O(objects nearby)) and
        # an identical bounded tile scan otherwise.
        scan_r = 10
        nearest = world.nearest_edible(agent.x, agent.y, scan_r)
        return None if nearest is None else nearest[0]

    def _compute_food_dir_match(self, agent: "Agent", world: "World") -> float:
        """
        Compute cosine match between agent facing direction and nearest food.

        Returns 0.5 (neutral) when no food is nearby.
        """
        import math

        if not hasattr(world, "tiles") or not hasattr(world, "nearest_edible"):
            return 0.5  # minimal/mock world

        scan_r = 5
        nearest = world.nearest_edible(agent.x, agent.y, scan_r)
        if nearest is None:
            return 0.5  # neutral — no food visible

        _, best_fx, best_fy = nearest
        diff_x = best_fx - agent.x
        diff_y = best_fy - agent.y
        mag = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        if mag == 0:
            return 1.0  # standing on food
        ndx, ndy = diff_x / mag, diff_y / mag
        dx, dy = agent.direction
        dot = dx * ndx + dy * ndy  # range [-1, 1]
        return (dot + 1.0) / 2.0  # remap to [0, 1]

    def reset(self):
        """Reset tracking for new episode."""
        self.last_food_distance = None
        self.last_actions = []
        self.last_position = None
        self.recent_positions.clear()
        self.forward_run_length = 0
        self.blocked_forward_cooldown = 0
        self.consecutive_waits = 0
        self.positions_visited.clear()
        self.steps_without_movement = 0
        self.consecutive_same_action = 0
        self.last_action = None
        self.consecutive_failed_eats = 0
        self._recently_dropped_ids.clear()
        self.last_food_dir_match = 0.5


class BestAgentTracker:
    """
    Tracks and saves the best performing agents' weights.

    This allows starting subsequent runs with pre-trained weights
    from agents that performed well in previous runs.
    """

    def __init__(self, save_dir: str = "data/weights"):
        """
        Initialize the tracker.

        Args:
            save_dir: Directory to save weight files
        """
        self.save_dir = save_dir
        self.best_agents = []  # List of (fitness, weights, metadata)
        self.max_saved = 5  # Keep top 5 agents

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def update(self, agent: "Agent", world: "World") -> bool:
        """
        Check if agent should be added to best agents list.

        Args:
            agent: Agent to evaluate
            world: Current world state

        Returns:
            True if agent was added to best list
        """
        # Calculate comprehensive fitness score
        fitness = self._calculate_fitness(agent)

        # Check if this agent qualifies for the best list
        if len(self.best_agents) < self.max_saved:
            self._add_agent(agent, fitness)
            return True
        elif fitness > self.best_agents[-1][0]:  # Better than worst in list
            self._add_agent(agent, fitness)
            return True

        return False

    def _calculate_fitness(self, agent: "Agent") -> float:
        """
        Calculate comprehensive fitness score.

        Args:
            agent: Agent to evaluate

        Returns:
            Fitness score
        """
        # Base fitness from agent's own fitness calculation
        fitness = agent.fitness

        # Bonus for longevity
        fitness += agent.age * 0.1

        # Bonus for remaining energy
        fitness += agent.energy * 0.05

        # Bonus for generation (evolved agents are likely better)
        fitness += agent.genome.generation * 2.0

        return fitness

    def _add_agent(self, agent: "Agent", fitness: float) -> None:
        """
        Add agent to the best list.

        Args:
            agent: Agent to add
            fitness: Pre-calculated fitness score
        """
        metadata = {
            "generation": agent.genome.generation,
            "age": agent.age,
            "energy": agent.energy,
            "fitness": agent.fitness,
            "traits": {k: float(v) for k, v in agent.genome.traits.items()},
        }

        # Get weights from genome
        weights = agent.genome.weights.copy()

        self.best_agents.append((fitness, weights, metadata))

        # Sort by fitness (descending) and keep only top agents
        self.best_agents.sort(key=lambda x: x[0], reverse=True)
        self.best_agents = self.best_agents[: self.max_saved]

    def save_best_weights(self, filename: str = "best_weights.npz") -> str:
        """
        Save best agent weights to file.

        Args:
            filename: Name of the file to save

        Returns:
            Path to saved file
        """
        if not self.best_agents:
            print("No best agents to save")
            return ""

        filepath = os.path.join(self.save_dir, filename)

        # Save weights as numpy arrays
        weights_dict = {}
        for i, (fitness, weights, metadata) in enumerate(self.best_agents):
            weights_dict[f"weights_{i}"] = weights
            weights_dict[f"fitness_{i}"] = np.array([fitness])

        np.savez(filepath, **weights_dict)

        # Also save metadata as JSON
        metadata_path = filepath.replace(".npz", "_metadata.json")
        metadata_list = [
            {"rank": i, "fitness": f, "metadata": m}
            for i, (f, _, m) in enumerate(self.best_agents)
        ]
        with open(metadata_path, "w") as f:
            json.dump(metadata_list, f, indent=2)

        print(f"Saved {len(self.best_agents)} best agent weights to {filepath}")
        return filepath

    @staticmethod
    def load_best_weights(
        filepath: str = "data/weights/best_weights.npz",
    ) -> Optional[np.ndarray]:
        """
        Load best weights from file.

        Args:
            filepath: Path to weight file

        Returns:
            Best agent's weights, or None if file doesn't exist
        """
        if not os.path.exists(filepath):
            print(f"No saved weights found at {filepath}")
            return None

        try:
            data = np.load(filepath)
            # Return the best (first) weights
            if "weights_0" in data:
                print(f"Loaded best weights from {filepath}")
                return data["weights_0"]
        except Exception as e:
            print(f"Error loading weights: {e}")

        return None

    @staticmethod
    def initialize_agent_from_weights(
        agent: "Agent", weights: np.ndarray, mutation_rate: float = 0.01
    ) -> None:
        """
        Initialize an agent's brain with pre-trained weights.

        Adds small mutations to create variation.

        Args:
            agent: Agent to initialize
            weights: Pre-trained weights
            mutation_rate: Standard deviation of mutations to add
        """
        # Apply small mutations for variation
        mutated_weights = weights + np.random.normal(0, mutation_rate, weights.shape)
        agent.genome.weights = mutated_weights.astype(np.float32)

        # Re-bind brain parameter views to the new weights
        # (architecture-agnostic: works for Brain v2 and v3)
        agent.brain.rebind(agent.genome)
        # Reset hidden state
        agent.h = agent.brain.initial_state()
