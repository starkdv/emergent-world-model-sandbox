"""
Parallel execution utilities for the simulation engine.

Provides a phased agent-update pipeline and parallel world-system
helpers that keep the GIL-releasing numpy work off the main thread.

Architecture
------------
Agent ticks are split into three phases:

  Phase 1 — OBSERVE + DECIDE  (parallelisable)
      Each agent builds its observation vector (read-only world access)
      and runs the neural-net forward pass.  No world mutation.

  Phase 2 — EXECUTE  (serial)
      Each agent executes its chosen action, mutating world state
      (position, tiles, objects, energy).  Must be serial because
      agents share the world.

  Phase 3 — LEARN  (parallelisable)
      Each agent stores its experience and (if scheduled) runs
      a gradient update on its own brain.  Fully isolated per-agent.

Threading model
---------------
Uses ``concurrent.futures.ThreadPoolExecutor`` with a shared pool.
NumPy releases the GIL during matrix multiplications, so threads
give real speed-up for the neural-net heavy phases (1 & 3) on CPython.

Author: Karan Vasa
Date: February 20, 2026
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, Future
from typing import TYPE_CHECKING, List, Tuple, Optional

import numpy as np

if TYPE_CHECKING:
    from agents.agent import Agent
    from world.world import World

# ---------------------------------------------------------------------------
# Shared thread pool  (created lazily, max 4 workers by default)
# ---------------------------------------------------------------------------

_pool: Optional[ThreadPoolExecutor] = None
_io_pool: Optional[ThreadPoolExecutor] = None

# Cap workers — too many threads add context-switch overhead for the
# relatively short-lived tasks we submit.
_MAX_WORKERS = min(4, max(1, (os.cpu_count() or 2)))


def get_pool() -> ThreadPoolExecutor:
    """Return (or create) the shared thread pool."""
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="sim-worker",
        )
    return _pool


def get_io_pool() -> ThreadPoolExecutor:
    """Return (or create) the single-thread I/O pool.

    Used for append-only logging so that file writes never overlap while still
    keeping I/O off the simulation hot-path.
    """
    global _io_pool
    if _io_pool is None:
        _io_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="sim-io",
        )
    return _io_pool


def shutdown_pool() -> None:
    """Shutdown any shared thread pools (call at simulation exit)."""
    global _pool, _io_pool
    if _pool is not None:
        _pool.shutdown(wait=True)
        _pool = None
    if _io_pool is not None:
        _io_pool.shutdown(wait=True)
        _io_pool = None


# ---------------------------------------------------------------------------
# Phase 1:  observe + decide  (parallel, read-only world access)
# ---------------------------------------------------------------------------


def _observe_and_decide(agent: "Agent", world: "World") -> Optional[tuple]:
    """
    Run one agent's observe → decide pipeline.

    Returns
    -------
    tuple or None
        (agent, action, obs_before, action_mask, energy_before) on success.
        None if the agent died during the metabolism/age check.
    """
    from agents.actions import Action, ActionResult
    from world.objects import EdibleComponent

    if not agent.alive:
        return None

    # --- age & metabolism ---
    agent.age += 1
    energy_before = agent.energy
    agent.energy -= agent.metabolism_rate

    # --- death check ---
    if agent.energy <= 0 or agent.age >= agent.max_age:
        # Store info for serial phase to handle death properly
        death_reason = "starvation" if agent.energy <= 0 else "old_age"
        return ("DEATH", agent, energy_before, death_reason)

    # --- observation (read-only world access) ---
    observation = agent.observe(world)

    # --- action mask (read-only world access) ---
    action_mask = agent.get_action_mask(world)

    # --- auto-eat check ---
    has_food = False
    if agent.inventory:
        for obj_id in agent.inventory:
            obj = world.objects.get(obj_id)
            if obj is not None and obj.has_component(EdibleComponent):
                has_food = True
                break

    if agent.energy < agent.max_energy * 0.5 and has_food:
        action = Action.EAT
        _, _, agent.h = agent.brain.forward(
            observation,
            agent.h,
            action_mask=action_mask,
            temperature=agent.temperature,
        )
    else:
        action, agent.h, _ = agent.brain.decide(
            observation,
            agent.h,
            action_mask=action_mask,
            temperature=agent.temperature,
        )

    return ("OK", agent, action, observation, action_mask, energy_before)


# ---------------------------------------------------------------------------
# Phase 2:  execute  (SERIAL — mutates world)
# ---------------------------------------------------------------------------


def _execute_and_log(
    agent: "Agent",
    action,
    obs_before: np.ndarray,
    energy_before: float,
    world: "World",
) -> Optional[tuple]:
    """
    Execute the chosen action, compute reward, log transition.

    Returns
    -------
    tuple or None
        (agent, action, reward, obs_after) for the learning phase.
    """
    from agents.agent import Agent

    x_before = agent.x
    y_before = agent.y

    result = agent.execute_action(action, world)

    obs_after = agent.observe(world)

    reward = 0.0
    if agent.learning_enabled and agent.learner:
        reward = agent.learner.reward_shaper.calculate_reward(
            action,
            result,
            energy_before,
            agent.energy,
            agent,
            world,
        )

    # World-model logging
    if Agent.world_model_logger is not None:
        Agent.world_model_logger.log_transition(
            tick=world.tick,
            agent=agent,
            action=action,
            result=result,
            reward=reward,
            obs_before=obs_before,
            obs_after=obs_after,
            world=world,
            x_before=x_before,
            y_before=y_before,
            energy_before=energy_before,
            done=False,
            death_reason="",
        )

    return (agent, action, reward, obs_after)


# ---------------------------------------------------------------------------
# Phase 3:  learn  (parallel, fully isolated per-agent)
# ---------------------------------------------------------------------------


def _learn_step(
    agent: "Agent",
    action,
    reward: float,
    obs_after: np.ndarray,
    world: "World",
) -> None:
    """Store experience and (maybe) run a gradient update."""
    if not agent.learning_enabled or agent.learner is None:
        return

    if agent.last_observation is not None and agent.last_hidden_state is not None:
        agent.learner.store_experience(
            agent.last_observation,
            agent.last_hidden_state,
            action.value,
            reward,
            obs_after,
            agent.h,
            False,
        )

    has_enough = len(agent.learner.replay_buffer) >= agent.learner.batch_size
    can_train = False
    if has_enough:
        if hasattr(world, "try_acquire_learning_slot"):
            can_train = world.try_acquire_learning_slot(agent.id, agent.age)
        else:
            can_train = agent.age % 3 == 0

    if can_train:
        agent.learner.learn(agent.brain)

    agent.last_observation = obs_after.copy()
    agent.last_hidden_state = agent.h.copy()


# ---------------------------------------------------------------------------
# Phase 2-death:  handle agent death  (SERIAL)
# ---------------------------------------------------------------------------


def _handle_death(
    agent: "Agent",
    energy_before: float,
    death_reason: str,
    world: "World",
) -> None:
    """Process a dying agent — logging, terminal experience, die()."""
    from agents.actions import Action, ActionResult
    from agents.agent import Agent

    terminal_obs = None
    if (
        Agent.world_model_logger is not None and agent.last_observation is not None
    ) or (
        agent.learning_enabled and agent.learner and agent.last_observation is not None
    ):
        terminal_obs = agent.observe(world)

    if (
        Agent.world_model_logger is not None
        and agent.last_observation is not None
        and terminal_obs is not None
    ):
        Agent.world_model_logger.log_transition(
            tick=world.tick,
            agent=agent,
            action=Action.WAIT,
            result=ActionResult(False, 0.0, f"Died: {death_reason}"),
            reward=-1.0,
            obs_before=agent.last_observation,
            obs_after=terminal_obs,
            world=world,
            x_before=agent.x,
            y_before=agent.y,
            energy_before=energy_before,
            done=True,
            death_reason=death_reason,
        )

    agent.die(world)

    if (
        agent.learning_enabled
        and agent.learner
        and agent.last_observation is not None
        and agent.last_hidden_state is not None
    ):
        if terminal_obs is None:
            terminal_obs = agent.observe(world)
        terminal_h = agent.brain.initial_state()
        agent.learner.store_experience(
            agent.last_observation,
            agent.last_hidden_state,
            0,
            -1.0,
            terminal_obs,
            terminal_h,
            True,
        )


# ---------------------------------------------------------------------------
# Public API — called by World._update_agents()
# ---------------------------------------------------------------------------


def update_agents_parallel(world: "World") -> None:
    """
    Three-phase parallel agent update.

    Phase 1 — observe + decide  (thread pool)
    Phase 2 — execute + log     (serial main thread)
    Phase 3 — learn             (thread pool)
    """
    pool = get_pool()
    alive_agents = [a for a in world.agents.values() if a.alive]

    if not alive_agents:
        return

    # Get max population from config (default: unlimited)
    max_population = None
    if world.reproduction_config:
        max_population = world.reproduction_config.get("max_population", None)

    # ================================================================
    # PHASE 1:  observe + decide  (parallel)
    # ================================================================
    futures: List[Future] = []
    for agent in alive_agents:
        futures.append(pool.submit(_observe_and_decide, agent, world))

    phase1_results = [f.result() for f in futures]

    # ================================================================
    # PHASE 2:  execute actions  (serial — mutates world)
    # ================================================================
    learn_items: List[tuple] = []  # (agent, action, reward, obs_after)
    new_offspring: List = []

    for res in phase1_results:
        if res is None:
            continue

        tag = res[0]

        if tag == "DEATH":
            _, agent, energy_before, death_reason = res
            _handle_death(agent, energy_before, death_reason, world)
            continue

        # tag == "OK"
        _, agent, action, obs_before, action_mask, energy_before = res

        exec_result = _execute_and_log(agent, action, obs_before, energy_before, world)
        if exec_result is not None:
            learn_items.append(exec_result)

        # Reproduction check (after execute, same as original)
        if agent.alive and agent.can_reproduce(world.reproduction_config):
            if max_population is not None:
                current_pop = len(world.agents) + len(new_offspring)
                if current_pop >= max_population:
                    continue
            offspring = agent.reproduce(world, world.reproduction_config)
            if offspring is not None:
                new_offspring.append(offspring)

    # Add offspring
    for offspring in new_offspring:
        world.add_agent(offspring)

    # ================================================================
    # PHASE 3:  learn  (parallel)
    # ================================================================
    if learn_items:
        learn_futures: List[Future] = []
        for agent, action, reward, obs_after in learn_items:
            learn_futures.append(
                pool.submit(_learn_step, agent, action, reward, obs_after, world)
            )
        # Wait for all learning to complete
        for f in learn_futures:
            f.result()
