"""
Agent class for the emergent world simulation.

Agents are autonomous entities that:
- Perceive their environment through observations
- Make decisions using neural network brains
- Execute primitive actions to survive and reproduce
- Evolve over generations through genetic algorithms
- Learn during their lifetime through reinforcement learning

Author: Karan Vasa
Date: November 14, 2025
"""

from typing import TYPE_CHECKING, Optional, List, ClassVar
import numpy as np

from agents.actions import Action, ActionResult, DIRECTIONS
from agents.brain import Brain, create_brain  # noqa: F401 (Brain re-exported)
from agents.brain.instincts import InstinctModule
from agents.genome import Genome
from utils.agents.learning_utils import get_active_reward_config
from agents.scoring import (
    FLAT_ACTION_ENERGY_COST,
    LEGACY_BASE_COST,
    get_active_scoring_config,
)
import utils.agents.agent_utils as agent_utils

if TYPE_CHECKING:
    from world.world import World
    from utils.data.agent_logger import AgentLogger, WorldModelLogger
    from agents.learning import AgentLearner


class Agent:
    """
    An autonomous agent in the simulation.

    Agents must:
    - Maintain energy by consuming food
    - Navigate the world using primitive actions
    - Learn behaviors through evolutionary processes
    - Survive and reproduce to pass on genes

    Attributes:
        id (int): Unique identifier
        x (int): X coordinate in world
        y (int): Y coordinate in world
        direction (tuple[int, int]): Facing direction vector
        energy (float): Current energy level
        max_energy (float): Maximum energy capacity
        age (int): Age in ticks
        max_age (int): Maximum lifespan
        alive (bool): Whether agent is alive
        inventory (List[int]): Object IDs held by agent
        inventory_size (int): Maximum inventory capacity
        genome (Genome): Genetic information
        brain (Brain): Neural network policy
        traits (dict[str, float]): Phenotypic traits
        fitness (float): Fitness score for evolution
        metabolism_rate (float): Energy consumed per tick

    Class Attributes:
        logger (AgentLogger): Optional logger for tracking actions/states
        world_model_logger: Optional logger for world model training data
        instinct_config: Optional ``brain.instincts`` config dict applied
            to every newly created agent (set once from YAML in main.py)
        brain_config: Optional ``brain`` config dict selecting the brain
            architecture (version 2 or 3) and its sizes; None → v2 defaults
    """

    _next_id = 0
    logger: ClassVar[Optional["AgentLogger"]] = None
    world_model_logger: ClassVar[Optional["WorldModelLogger"]] = None
    instinct_config: ClassVar[Optional[dict]] = None
    brain_config: ClassVar[Optional[dict]] = None

    def __init__(
        self,
        x: int,
        y: int,
        genome: Genome,
        max_energy: float = 200.0,
        max_age: int = 1000,
        inventory_size: int = 5,
        metabolism_rate: float = 0.5,
    ):
        """
        Initialize a new agent.

        Args:
            x: Starting x coordinate
            y: Starting y coordinate
            genome: Genetic information
            max_energy: Maximum energy capacity
            max_age: Maximum lifespan in ticks
            inventory_size: Maximum inventory slots
            metabolism_rate: Base energy consumption per tick
        """
        self.id = Agent._next_id
        Agent._next_id += 1

        self.x = x
        self.y = y
        # Randomize initial facing direction to prevent population-wide
        # turn bias from correlated starting orientations.
        _cardinal = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.direction = _cardinal[np.random.randint(4)]

        self.energy = max_energy  # Start with full energy
        self.max_energy = max_energy
        self.age = 0
        self.max_age = max_age
        self.alive = True

        self.inventory: List[int] = []
        self.inventory_size = inventory_size
        # Evolutionary components
        self.genome = genome
        # Brain architecture (v2 or v3) and instincts are configured once
        # (class-level, from YAML) so offspring inherit the same setup.
        # Instincts fade with age — see agents/brain/instincts.py.
        self.brain = create_brain(
            genome,
            Agent.brain_config,
            instincts=InstinctModule.from_config(Agent.instinct_config),
        )
        # Remember the exact brain config this agent was built with, so
        # offspring can breed true even in a mixed-architecture (cohort)
        # population where the class-level Agent.brain_config differs from
        # this agent's. ``cohort`` is a free-text label for A/B comparison
        # (e.g. "v3" vs "v2-old"); it has no effect on the simulation.
        self.brain_config_used = Agent.brain_config
        self.cohort = "default"
        self.traits = genome.traits.copy()
        self.fitness = 0.0
        # Reproduction bookkeeping for the `reproduction` fitness model
        # (agents/scoring.py): who my parent is, how many children I have had,
        # and how many of them reached reproductive age themselves.
        self.parent_agent_id: Optional[int] = None
        self.offspring_count = 0
        self.offspring_matured = 0
        self._credited_parent = False

        # GRU hidden state (memory)
        self.h = self.brain.initial_state()

        # World-model extras (only active when the brain has a dynamics
        # head): optional latent rollout planner; curiosity is attached
        # by enable_learning since it shapes the learning reward.
        self.planner = None
        wm_cfg = (Agent.brain_config or {}).get("world_model", {}) or {}
        planner_cfg = wm_cfg.get("planner", {}) or {}
        if self.brain.has_world_model and planner_cfg.get("enabled", False):
            from agents.planner import LatentPlanner

            self.planner = LatentPlanner.from_config(planner_cfg)
        self.curiosity = None

        # Learning components
        self.learner: Optional["AgentLearner"] = None
        self.last_observation: Optional[np.ndarray] = None
        self.last_hidden_state: Optional[np.ndarray] = None
        self.learning_enabled = True  # Can be disabled for pure evolution
        self.temperature = 1.0  # Sampling temperature for exploration

        # Action-pattern tracking for energy shaping
        self._previous_action: Optional[Action] = None
        self._consecutive_turns = 0
        self._consecutive_waits = 0
        self._last_move_succeeded = False
        # Previous tick's energy, for the v4 `social` drive (a neighbour's
        # energy delta is the only thing an agent can "care" about).
        self._energy_last_tick = self.energy
        # Apply trait-based modifications
        self.metabolism_rate = metabolism_rate * self.traits.get("metabolism_rate", 1.0)
        self.vision_radius = int(self.traits.get("vision_radius", 5.0))

    def update(self, world: "World") -> None:
        """
        Update agent state each tick.

        Args:
            world: The world the agent exists in
        """
        if not self.alive:
            return

        # Current world tick — used by planner/learner warmup schedules so the
        # model-based planner (P2/P3) only kicks in once the world model has had
        # time to train (see docs/PLANNING_PROPOSAL.md "warmup").
        self._world_tick = int(getattr(world, "tick", 0))

        # Age the agent
        self.age += 1

        # Fitness bookkeeping under the `reproduction` model: once this agent
        # reaches reproductive age it counts toward its parent's fitness, and
        # its own fitness is recomputed from descendants + lifespan.
        self._refresh_fitness(world)

        # Consume energy for metabolism (temperature extremes cost more
        # when the environment engine is enabled — W1)
        energy_before = self.energy
        self.energy -= self.metabolism_rate * world.environment.metabolism_multiplier

        # Check for death conditions
        if self.energy <= 0 or self.age >= self.max_age:
            death_reason = "starvation" if self.energy <= 0 else "old_age"

            # Compute terminal observation ONCE for both loggers
            terminal_obs = None
            if (
                Agent.world_model_logger is not None
                and self.last_observation is not None
            ) or (
                self.learning_enabled
                and self.learner
                and self.last_observation is not None
            ):
                terminal_obs = self.observe(world)

            # Log terminal transition for world model
            if (
                Agent.world_model_logger is not None
                and self.last_observation is not None
                and terminal_obs is not None
            ):
                Agent.world_model_logger.log_transition(
                    tick=world.tick,
                    agent=self,
                    action=Action.WAIT,  # Placeholder
                    result=ActionResult(False, 0.0, f"Died: {death_reason}"),
                    reward=-1.0,
                    obs_before=self.last_observation,
                    obs_after=terminal_obs,
                    world=world,
                    x_before=self.x,
                    y_before=self.y,
                    energy_before=energy_before,
                    done=True,
                    death_reason=death_reason,
                )

            self.die(world)

            # Store terminal experience if learning
            if self.learning_enabled and self.learner:
                if getattr(self.learner, "wants_sequences", False):
                    # PPO path: flag the last stored step as terminal
                    if terminal_obs is None:
                        terminal_obs = self.observe(world)
                    self.learner.mark_done(terminal_obs)
                elif (
                    self.last_observation is not None
                    and self.last_hidden_state is not None
                ):
                    if terminal_obs is None:
                        terminal_obs = self.observe(world)
                    terminal_h = self.brain.initial_state()  # Dead state
                    self.learner.store_experience(
                        self.last_observation,
                        self.last_hidden_state,
                        0,  # Action doesn't matter
                        -1.0,  # Death penalty
                        terminal_obs,
                        terminal_h,
                        True,  # Episode done
                    )
            return  # Get observation and decide action
        observation = self.observe(world)

        action_mask = self.get_action_mask(world)
        action, h_before_step, step_logprob, uses_sequences = self.choose_action(
            observation, action_mask
        )

        # Store observation before action for logging
        obs_before = observation  # no copy needed — not modified before use
        x_before_action = self.x
        y_before_action = self.y

        # Execute action
        result = self.execute_action(action, world)

        # Get observation after action
        obs_after = self.observe(world)

        # Calculate reward (needed for both learning and logging)
        reward = 0.0
        if self.learning_enabled and self.learner:
            reward = self.compute_reward(
                action, result, energy_before, obs_after, world
            )

        # World model logging (captures full transitions for training)
        if Agent.world_model_logger is not None:
            Agent.world_model_logger.log_transition(
                tick=world.tick,
                agent=self,
                action=action,
                result=result,
                reward=reward,
                obs_before=obs_before,
                obs_after=obs_after,
                world=world,
                x_before=x_before_action,
                y_before=y_before_action,
                energy_before=energy_before,
                done=False,
                death_reason="",
            )

        # Advance the v4 episodic memory (path integration + write). Done
        # before the step is stored, so the state carried into the NEXT step
        # is the one the learner will reproduce during replay.
        self.advance_memory(observation, action, reward)
        self._energy_last_tick = self.energy

        # Learning step
        if self.learning_enabled and self.learner:
            if uses_sequences:
                # PPO path: time-ordered step with behaviour log-prob and
                # action mask (decision-time observation, not the stale
                # previous-tick one)
                self.learner.store_step(
                    observation=observation,
                    hidden_before=h_before_step,
                    action=action.value,
                    reward=reward,
                    next_observation=obs_after,
                    done=False,
                    logprob=step_logprob,
                    action_mask=action_mask,
                    moved=self._last_move_succeeded,
                )
            elif (
                self.last_observation is not None and self.last_hidden_state is not None
            ):
                # Legacy A2C path: single transitions
                # Note: self.h has already been updated to h_next by brain.decide above
                self.learner.store_experience(
                    self.last_observation,
                    self.last_hidden_state,
                    action.value,
                    reward,
                    obs_after,
                    self.h,  # Current (next) hidden state
                    False,  # Not done yet
                )

            # Learn when world scheduler grants a training slot (staggered + capped)
            has_enough_experience = (
                len(self.learner.replay_buffer) >= self.learner.batch_size
            )
            can_train_now = False

            if has_enough_experience:
                if hasattr(world, "try_acquire_learning_slot"):
                    can_train_now = world.try_acquire_learning_slot(self.id, self.age)
                else:
                    can_train_now = self.age % 3 == 0

            if can_train_now:
                # let the learner gate imagination by world tick (warmup)
                setattr(self.learner, "current_tick", getattr(self, "_world_tick", 0))
                self.learner.learn(self.brain)

            # Store current observation and hidden state for next step
            self.last_observation = obs_after.copy()
            self.last_hidden_state = self.h.copy()

    def _refresh_fitness(self, world: "World") -> None:
        """
        Recompute fitness under the `reproduction` scoring model.

        Fitness is the number of offspring that themselves reached
        reproductive age, plus a small per-tick lifespan tiebreak. Nothing
        about *what* the agent did enters it — that is the point (see
        agents/scoring.py and docs/BRAIN_V4_PROPOSAL.md §4.5).

        Crediting is done by the child, not the parent: the first time a
        child reaches `offspring_maturity_ticks` it increments its parent's
        `offspring_matured`. A parent that has already died is not credited —
        its fitness is no longer read by anything.

        Args:
            world: The world (used to find the parent agent)
        """
        scoring = get_active_scoring_config()
        if not scoring.reproduction_fitness:
            return

        if (
            not self._credited_parent
            and self.age >= scoring.offspring_maturity_ticks
            and self.parent_agent_id is not None
        ):
            self._credited_parent = True
            parent = world.agents.get(self.parent_agent_id)
            if parent is not None and parent.alive:
                parent.offspring_matured += 1
                parent.fitness = float(parent.offspring_matured) + (
                    scoring.fitness_lifespan_coef * parent.age
                )

        self.fitness = float(self.offspring_matured) + (
            scoring.fitness_lifespan_coef * self.age
        )

    def choose_action(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> tuple[Action, Optional[np.ndarray], float, bool]:
        """
        Decide this tick's action. SINGLE source of decision logic —
        used by both Agent.update and the parallel pipeline
        (utils/parallel.py), so the two paths cannot drift.

        Handles:
        - fading instinct strength (1.0 at birth → 0.0 at fade_age)
        - the PPO path (behaviour log-prob + pre-step hidden state)
        - optional model-based planning over imagined latent rollouts

        Updates self.h as a side effect.

        Args:
            observation: Current observation vector
            action_mask: Valid-action mask

        Returns:
            (action, h_before_step, logprob, uses_sequences) where
            h_before_step/logprob are only meaningful when
            uses_sequences is True (PPO learner attached)
        """
        instinct_strength = self.brain.instincts.strength_at(self.age)
        uses_sequences = self.learner is not None and getattr(
            self.learner, "wants_sequences", False
        )
        h_before = self.h.copy() if uses_sequences else None
        logprob = 0.0

        if self.planner is not None and self.brain.has_world_model:
            # Model-based: run the policy forward (memory must advance
            # regardless), then override the sampled choice with the
            # best imagined rollout's first action.
            probs, _, h_next = self.brain.forward(
                observation,
                self.h,
                action_mask=action_mask,
                temperature=self.temperature,
                instinct_strength=instinct_strength,
            )
            self.h = h_next
            action_idx = self.planner.plan(
                self.brain,
                self.h,
                action_mask,
                tick=getattr(self, "_world_tick", 0),
                model_error=getattr(self.learner, "wm_rollout_error_ema", None),
            )
            # Log-prob of the planner's choice under the policy — keeps
            # PPO's importance ratio meaningful (clipping bounds the rest)
            logprob = float(np.log(max(probs[action_idx], 1e-8)))
            action = Action(action_idx)
        elif uses_sequences:
            action, self.h, _, logprob = self.brain.decide_with_logprob(
                observation,
                self.h,
                action_mask=action_mask,
                temperature=self.temperature,
                instinct_strength=instinct_strength,
            )
        else:
            action, self.h, _ = self.brain.decide(
                observation,
                self.h,
                action_mask=action_mask,
                temperature=self.temperature,
                instinct_strength=instinct_strength,
            )

        return action, h_before, logprob, uses_sequences

    def compute_reward(
        self,
        action: Action,
        result: ActionResult,
        energy_before: float,
        obs_after: np.ndarray,
        world: "World",
    ) -> float:
        """
        Shaped reward for this tick, plus the curiosity bonus when a
        world model is present. Single source for both update paths.

        Curiosity = normalised error of the dynamics head's prediction
        of the post-action latent (see agents/curiosity.py).

        Args:
            action: Action taken
            result: Execution result
            energy_before: Energy before the tick
            obs_after: Observation after the action
            world: The world (reward shaper context)

        Returns:
            Total reward (extrinsic + intrinsic)
        """
        cfg = get_active_reward_config()
        if cfg.preset == "drives" and hasattr(self.brain, "drive_weights"):
            return self._drive_reward(energy_before, obs_after, world, cfg)

        reward = self.learner.reward_shaper.calculate_reward(
            action, result, energy_before, self.energy, self, world
        )
        if self.curiosity is not None and self.brain.has_world_model:
            z_pred, _ = self.brain.predict_next_latent(self.h, action.value)
            reward += self.curiosity.intrinsic_reward(z_pred, self._latent(obs_after))
        return reward

    def _latent(self, observation: np.ndarray) -> np.ndarray:
        """
        The latent the dynamics head predicts, for this observation.

        v2/v3 encode the observation alone. A v4 latent also carries the
        episodic memory read, so it needs the current recurrent state; the
        advanced state ``core_step`` returns is discarded here — this is a
        read-only measurement of "what does the world look like now".

        Args:
            observation: Observation vector

        Returns:
            Latent of the same width as the dynamics head's output
        """
        step = getattr(self.brain, "core_step", None)
        if step is None:
            return self.brain.encode(observation)
        return step(observation, self.h)[0]

    def _drive_reward(
        self,
        energy_before: float,
        obs_after: np.ndarray,
        world: "World",
        cfg,
    ) -> float:
        """
        The v4 evolved-motivation reward (docs/BRAIN_V4_PROPOSAL.md §4.5).

            r = lambda . ( homeostasis, empowerment, curiosity, social )

        with ``lambda`` read from the genome, so the objective is selected on
        reproductive success rather than written down. The four terms:

        * **homeostasis** — ``d_{t-1} - d_t`` with ``d = (1 - e/e_max)^2``.
          The sum telescopes to ``d_0 - d_T``, so this is exactly a
          potential-based shaping function with ``Phi = -d``, which leaves the
          optimal policy of the underlying MDP unchanged (Ng, Harada & Russell
          1999). It says having energy is good and nothing whatever about how
          to get it — the difference between a drive and a strategy.
        * **empowerment** — the spread of predicted next latents across
          actions: "how much does my choice matter here".
        * **curiosity** — the existing normalised dynamics prediction error.
        * **social** — the summed energy change of neighbours within
          ``drive_social_radius``. Deliberately *unsigned*: the weight may
          evolve positive (altruism), negative (spite) or zero. We do not
          choose; Hamilton's rule makes a prediction and the run tests it.

        Args:
            energy_before: Energy at the start of this tick
            obs_after: Observation after the action
            world: The world
            cfg: Active RewardConfig

        Returns:
            The weighted drive reward
        """
        lam = np.asarray(self.brain.drive_weights, dtype=np.float32)

        inv = 1.0 / max(self.max_energy, 1e-6)
        d_prev = (1.0 - min(1.0, max(0.0, energy_before * inv))) ** 2
        d_now = (1.0 - min(1.0, max(0.0, self.energy * inv))) ** 2
        r_homeo = d_prev - d_now
        if not self.alive:
            r_homeo -= 1.0

        r_emp = 0.0
        if self.brain.has_world_model:
            r_emp = cfg.drive_empowerment_scale * self.brain.action_latent_spread(
                self.h
            )

        r_cur = 0.0
        if self.curiosity is not None and self.brain.has_world_model:
            z_pred, _ = self.brain.predict_next_latent(self.h, 0)
            r_cur = self.curiosity.intrinsic_reward(z_pred, self._latent(obs_after))

        r_soc = 0.0
        radius = cfg.drive_social_radius
        if radius > 0 and abs(float(lam[3])) > 1e-6:
            total = 0.0
            for other in world.agents.values():
                if other is self or not other.alive:
                    continue
                if abs(other.x - self.x) + abs(other.y - self.y) > radius:
                    continue
                total += other.energy - getattr(
                    other, "_energy_last_tick", other.energy
                )
            r_soc = cfg.drive_social_scale * total

        return float(
            lam[0] * r_homeo + lam[1] * r_emp + lam[2] * r_cur + lam[3] * r_soc
        )

    def advance_memory(
        self, observation: np.ndarray, action: Action, reward: float
    ) -> None:
        """
        Advance the v4 episodic place memory by one tick.

        No-op for brains without one. Called after the action's reward is
        known, because the reward's magnitude is the write salience.

        Args:
            observation: The decision-time observation of this step
            action: The action taken
            reward: The reward received
        """
        update = getattr(self.brain, "update_memory", None)
        if update is None:
            return
        self.h = update(
            self.h, observation, action.value, reward, self._last_move_succeeded
        )

    def get_action_mask(self, world: "World") -> np.ndarray:
        """
        Binary mask over Action enum (1 = valid, 0 = invalid).
        Delegates to agent_utils.
        """
        return agent_utils.get_action_mask(self, world)

    def observe(self, world: "World") -> np.ndarray:
        """
        Build observation vector from world state.

        Args:
            world: The world to observe

        Returns:
            Normalized observation vector"""
        from utils.agents import build_observation

        return build_observation(self, world)

    def execute_action(self, action: Action, world: "World") -> ActionResult:
        """
        Execute a primitive action in the world.

        Args:
            action: The action to execute
            world: The world to act in

        Returns:
            Result of the action execution
        """
        if not self.alive:
            return ActionResult(False, 0.0, "Agent is dead")

        # Store state before action for logging
        x_before = self.x
        y_before = self.y
        energy_before = self.energy

        result = ActionResult(True, 0.0)

        if action == Action.MOVE_FORWARD:
            result = agent_utils.execute_move_forward(self, world)
        elif action == Action.TURN_LEFT:
            result = agent_utils.execute_turn_left(self)
        elif action == Action.TURN_RIGHT:
            result = agent_utils.execute_turn_right(self)
        elif action == Action.PICK_UP:
            result = agent_utils.execute_pick_up(self, world)
        elif action == Action.DROP:
            result = agent_utils.execute_drop(self, world)
        elif action == Action.EAT:
            result = agent_utils.execute_eat(self, world)
        elif action == Action.USE:
            result = agent_utils.execute_use(self, world)
        elif action == Action.WAIT:
            result = agent_utils.execute_wait(self)
        elif action == Action.SIGNAL:
            result = agent_utils.execute_signal(self, world)

        # Action cost model (agents/scoring.py).
        #   flat   — the cost is a constant of the action plus whatever the
        #            world charged (slope, hazard). Nothing depends on what
        #            the agent did last tick.
        #   legacy — the shipped "behaviour economics" below.
        scoring = get_active_scoring_config()
        effective_energy_cost = result.energy_cost

        if scoring.flat_costs:
            base = FLAT_ACTION_ENERGY_COST.get(action)
            if base is not None and result.success:
                # Keep the world-derived surcharge (slope climb, hazard
                # contact) that the executor added on top of its own base.
                surcharge = max(0.0, result.energy_cost - LEGACY_BASE_COST[action])
                effective_energy_cost = base + surcharge
            self._consecutive_turns = 0
            self._consecutive_waits = 0
        elif action in [Action.TURN_LEFT, Action.TURN_RIGHT]:
            self._consecutive_turns += 1
            self._consecutive_waits = 0
            # Mild escalating turn cost — only punishes extended spin loops.
            # IMPORTANT: don't penalize the first couple of turns; otherwise
            # agents learn "always MOVE_FORWARD until masked".
            extra_turn_penalty = max(0, self._consecutive_turns - 2)
            effective_energy_cost += min(0.04 * extra_turn_penalty, 0.20)
        elif action == Action.WAIT:
            self._consecutive_waits += 1
            self._consecutive_turns = 0
            # Very gentle escalating wait cost — WAIT should remain affordable
            extra_wait_penalty = max(0, self._consecutive_waits - 2)
            effective_energy_cost += min(0.02 * extra_wait_penalty, 0.10)
        elif action == Action.MOVE_FORWARD:  # noqa: E501 — legacy branch
            # Reward turn->move transition with a tiny cost discount
            if result.success and self._previous_action in [
                Action.TURN_LEFT,
                Action.TURN_RIGHT,
            ]:
                effective_energy_cost = max(0.10, effective_energy_cost - 0.02)
            self._consecutive_turns = 0
            self._consecutive_waits = 0
        else:
            self._consecutive_turns = 0
            self._consecutive_waits = 0

        # Use effective cost for state update and downstream logging
        result = result._replace(energy_cost=round(effective_energy_cost, 3))

        # Did this step actually displace the agent? The v4 episodic memory
        # path-integrates on the OUTCOME, not the intent — a blocked move
        # must not shift every stored place by one tile.
        self._last_move_succeeded = (self.x != x_before) or (self.y != y_before)

        # Deduct energy cost
        self.energy -= result.energy_cost

        # Track action for next-step energy shaping
        self._previous_action = action

        # Fitness model (agents/scoring.py). Under `reproduction`, fitness
        # counts descendants, not actions, so nothing accrues here — see
        # _refresh_fitness().
        if not scoring.reproduction_fitness:
            # Update fitness based on action outcomes.
            # Successful turns should not be penalized; otherwise the policy
            # is structurally biased toward MOVE_FORWARD.
            if result.success:
                if action == Action.WAIT:
                    # WAIT is neutral from a fitness perspective.
                    self.fitness += 0.0
                else:
                    self.fitness += 0.1  # Small reward for successful action
            else:
                self.fitness -= 0.05  # Small penalty for failed action

        # Log action if logger is enabled
        if Agent.logger is not None:
            Agent.logger.log_action(
                world.tick, self, action, result, x_before, y_before, energy_before
            )

        return result

    def die(self, world: "World") -> None:
        """
        Handle agent death.

        Args:
            world: The world the agent exists in
        """
        import random

        self.alive = False

        # Reset hidden state (agent's memory is lost on death)
        self.h = self.brain.initial_state()

        # Death penalty to fitness (legacy fitness model only). Under the
        # `reproduction` model, dying young already costs fitness — you stop
        # accruing lifespan and stop producing offspring — so no extra
        # hand-written penalty is applied (agents/scoring.py).
        if not get_active_scoring_config().reproduction_fitness:
            # proportional to how early the death was:
            # dying young = big penalty, dying old = small penalty
            age_ratio = self.age / self.max_age
            death_penalty = 10.0 * (
                1.0 - age_ratio
            )  # Max -10 for instant death, 0 for old age
            self.fitness -= death_penalty

            # Extra penalty for starvation (should have eaten!)
            if self.energy <= 0:
                self.fitness -= 5.0  # Starvation penalty

        # Drop all inventory items with stacking configuration check
        for obj_id in self.inventory:
            obj = world.objects.get(obj_id)
            if obj is not None:
                tile = world.tiles[self.y][self.x]

                # Check stacking configuration
                if world.allow_stacking or not tile.object_ids:
                    # Stacking allowed OR tile is empty - drop here
                    tile.object_ids.add(obj_id)
                    obj.x = self.x
                    obj.y = self.y
                else:
                    # Stacking disabled and tile occupied - try nearby tiles
                    directions = [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]
                    nearby_positions = [
                        (self.x + dx, self.y + dy) for dx, dy in directions
                    ]
                    random.shuffle(nearby_positions)

                    placed = False
                    for nx, ny in nearby_positions:
                        if 0 <= nx < world.width and 0 <= ny < world.height:
                            nearby_tile = world.tiles[ny][nx]
                            if not nearby_tile.object_ids:
                                # Found empty spot
                                nearby_tile.object_ids.add(obj_id)
                                obj.x = nx
                                obj.y = ny
                                placed = True
                                break

                    # If no space found, remove object from world
                    if not placed:
                        world.remove_object(obj_id)

        self.inventory.clear()

    def can_reproduce(self, config: dict = None) -> bool:
        """
        Check if agent has sufficient energy to reproduce.

        Reproduction requires:
        - Agent must be alive
        - Energy >= threshold (from config)
        - Age >= minimum (from config)

        Args:
            config: Optional reproduction config dict

        Returns:
            True if agent can reproduce, False otherwise
        """
        if not self.alive:
            return False

        # Default values if no config provided
        if config is None:
            energy_threshold_pct = 0.6
            min_age = 100
        else:
            energy_threshold_pct = config.get("energy_threshold", 0.6)
            min_age = config.get("min_age", 100)

        energy_threshold = self.max_energy * energy_threshold_pct

        return self.energy >= energy_threshold and self.age >= min_age

    def reproduce(self, world: "World", config: dict = None) -> Optional["Agent"]:
        """
        Reproduce via fission, creating an offspring.

        Uses config for reproduction parameters or defaults if not provided.

        Args:
            world: The world to spawn offspring in
            config: Optional reproduction config dict

        Returns:
            Offspring agent if successful, None if reproduction failed
        """
        if not self.can_reproduce(config):
            return None

        # Get config values or use defaults
        if config is None:
            energy_split = 0.6
            mutation_std = 0.02
        else:
            energy_split = config.get("energy_split", 0.6)
            mutation_std = config.get("mutation_std", 0.02)

        # Import here to avoid circular dependency
        from agents.evolution import clone_agent

        # Create offspring using evolution system
        offspring = clone_agent(parent=self, mutate=True, mutation_std=mutation_std)

        # Give offspring FULL energy for best survival chance
        # Parent loses energy based on split ratio
        energy_cost = self.energy * energy_split
        self.energy -= energy_cost
        offspring.energy = offspring.max_energy  # Start with FULL energy!

        # Find nearby empty position for offspring
        spawn_positions = [
            (self.x + dx, self.y + dy)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            if (dx != 0 or dy != 0)  # Not same position
        ]
        # Try positions in random order
        import random

        random.shuffle(spawn_positions)

        for x, y in spawn_positions:
            if world.is_valid_position(x, y):
                # Check if position is empty (no other agents)
                occupied = any(
                    agent.x == x and agent.y == y
                    for agent in world.agents.values()
                    if agent.alive
                )

                if not occupied:
                    offspring.x = x
                    offspring.y = y

                    # Offspring starts with fresh memory (no inherited hidden state)
                    offspring.h = offspring.brain.initial_state()

                    # Enable learning if parent has it (same algorithm)
                    if self.learner:
                        offspring.enable_learning(
                            learning_rate=self.learner.learning_rate,
                            discount_factor=self.learner.discount_factor,
                            batch_size=self.learner.batch_size,
                            buffer_capacity=1000,  # Use default capacity
                            compute_backend=self.learner.compute_backend,
                            compute_device=self.learner.compute_device,
                            algorithm=getattr(self.learner, "algorithm", "a2c"),
                            ppo_config=getattr(self, "_ppo_config", None),
                            curiosity_config=getattr(self, "_curiosity_config", None),
                        )

                    # Inherit parent's temperature
                    offspring.temperature = self.temperature

                    # Reproduction fitness bookkeeping (agents/scoring.py)
                    offspring.parent_agent_id = self.id
                    offspring.offspring_count = 0
                    offspring.offspring_matured = 0
                    offspring._credited_parent = False
                    self.offspring_count += 1

                    return offspring

        # No valid position found - reproduction fails
        # Refund energy
        self.energy += energy_cost
        return None

    def enable_learning(
        self,
        learning_rate: float = 0.001,
        discount_factor: float = 0.95,
        batch_size: int = 32,
        buffer_capacity: int = 1000,
        compute_backend: str = "auto",
        compute_device: str = "auto",
        algorithm: str = "a2c",
        ppo_config: Optional[dict] = None,
        curiosity_config: Optional[dict] = None,
    ) -> None:
        """
        Enable reinforcement learning for this agent.

        Args:
            learning_rate: Learning rate for gradient updates
            discount_factor: Discount factor for future rewards
            batch_size: Batch size for learning updates
            buffer_capacity: Size of experience replay buffer
            compute_backend: 'auto', 'numpy', or 'torch'
            compute_device: 'auto', 'cpu', 'cuda', or 'mps'
            algorithm: 'a2c' (legacy heads-only updates) or 'ppo'
                (full-network backprop, sequence replay, GAE + clipping;
                requires torch — falls back to a2c without it)
            ppo_config: Optional ``learning.ppo`` config dict (seq_len,
                gae_lambda, clip_epsilon, value_coef, entropy_coef,
                epochs, grad_clip, chunk_buffer, learning_rate)
            curiosity_config: Optional ``learning.curiosity`` config dict
                (enabled, weight, decay, clip, warmup); only active when
                the brain has a world model (brain.world_model.enabled)
        """
        # Curiosity rides on the world model's prediction error
        self._curiosity_config = curiosity_config
        if (
            curiosity_config
            and curiosity_config.get("enabled", False)
            and self.brain.has_world_model
        ):
            from agents.curiosity import CuriosityModule

            self.curiosity = CuriosityModule.from_config(curiosity_config)
        if algorithm == "ppo":
            from agents.ppo import TORCH_AVAILABLE, PPOSequenceLearner

            if TORCH_AVAILABLE:
                ppo = ppo_config or {}
                self.learner = PPOSequenceLearner(
                    learning_rate=ppo.get("learning_rate", 3e-4),
                    discount_factor=discount_factor,
                    batch_size=ppo.get("batch_size", 8),
                    seq_len=ppo.get("seq_len", 8),
                    gae_lambda=ppo.get("gae_lambda", 0.95),
                    clip_epsilon=ppo.get("clip_epsilon", 0.2),
                    value_coef=ppo.get("value_coef", 0.5),
                    entropy_coef=ppo.get("entropy_coef", 0.01),
                    epochs=ppo.get("epochs", 2),
                    grad_clip=ppo.get("grad_clip", 0.5),
                    chunk_capacity=ppo.get("chunk_buffer", 64),
                    compute_device=(
                        "cpu" if compute_device == "auto" else compute_device
                    ),
                    world_model_coef=ppo.get("world_model_coef", 1.0),
                    imagination=ppo.get("imagination", None),
                    world_model_multistep=ppo.get("world_model_multistep", None),
                    rollout_metric_k=ppo.get("rollout_metric_k", 3),
                    long_gamma=ppo.get("long_gamma", 0.999),
                    advantage_mix=ppo.get("advantage_mix", 0.0),
                    return_scale=ppo.get("return_scale", True),
                )
                self._ppo_config = ppo_config
                self.learning_enabled = True
                self.last_observation = None
                return
            print("  [LEARN] torch unavailable — falling back to a2c")

        from agents.learning import AgentLearner

        self.learner = AgentLearner(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            compute_backend=compute_backend,
            compute_device=compute_device,
        )
        self.learning_enabled = True
        self.last_observation = None

    def disable_learning(self) -> None:
        """Disable learning for this agent (use pure evolution)."""
        self.learning_enabled = False
        self.learner = None
        self.last_observation = None

    def get_learned_knowledge(self) -> Optional[np.ndarray]:
        """
        Extract learned weights from brain.

        Returns:
            Flattened weight array or None if no learning
        """
        if not self.learning_enabled or self.learner is None:
            return None

        # Get current brain weights (already synced by learner)
        return self.genome.weights.copy()

    def inherit_knowledge(self, parent_knowledge: np.ndarray) -> None:
        """
        Initialize brain with knowledge from parent.

        This allows offspring to start with learned behaviors
        from their parents, combining evolution with learning.

        Args:
            parent_knowledge: Flattened weight array from parent
        """
        # Update genome with parent's learned weights
        self.genome.weights = parent_knowledge.copy()

        # Re-bind the brain's parameter views to the new weights
        # (architecture and instinct configuration are preserved)
        self.brain.rebind(self.genome)

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, pos=({self.x},{self.y}), "
            f"energy={self.energy:.1f}, age={self.age}, alive={self.alive})"
        )
