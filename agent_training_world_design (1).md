# Agent Training & Survival Balancing Guide

This document explains how to balance the world, rewards, and agent behaviour so agents **live long enough to learn**, **produce useful training data**, and **respond well to evolution or reinforcement learning**.

Use this as a reference for implementing a trainable agent system.

---

# 1. Overview
In early simulations, agents often die too fast due to:
- Harsh world settings
- Sparse rewards
- Random initial policy
- Strong mutations
- No curriculum or warm-up

This guide provides the **environment design**, **reward shaping**, and **training flow** needed to stabilize learning.

---

# 2. Core Principles

## ✔ Make the world *forgiving first*
Agents should be able to survive long enough to explore and accidentally obtain rewards.

## ✔ Provide *dense* rewards
Agents must receive feedback every few steps, not only when rare events occur.

## ✔ Use *curriculum learning*
Start with easy worlds; increase challenge gradually.

## ✔ Stabilize evolution / RL
Avoid chaotic mutations and full-population resets.

---

# 3. Environment Tuning (Make Survival Easier)
Adjust these until a random agent lives a reasonable number of steps (e.g., 200–500).

### **Parameters to loosen**:
- `starting_energy` ↑
- `energy_decay_per_step` ↓
- `food_spawn_rate` ↑
- `food_value` ↑
- `collision_damage` ↓
- `hazard_damage` ↓
- `max_steps_per_episode` fixed instead of instant death

### **Goal:**
Even a random policy should:
- survive awhile
- occasionally find food
- produce non-zero rewards

---

# 4. Enhanced Reward System
Agents must not rely only on rare events like eating. Use a **three-part reward**:

```text
reward_t = r_alive + r_approach_food + r_eat + r_penalty
```

## 4.1 Survival Reward
Small reward per step alive:

- `r_alive = +0.01` per step while `agent.alive == True`

This encourages agents to avoid instant death.

## 4.2 Approaching Food Reward
Reward agents for moving closer to edible food tiles.

**Distance-based shaping:**

- Let `d_t`   = distance to nearest edible food at time t
- Let `d_t+1` = distance at time t+1
- Define:

```text
Δd = d_t - d_t+1
if Δd > 0 and d_t > min_distance_threshold:
    r_approach_food = k * Δd
else:
    r_approach_food = 0
```

Where:
- `k` is a scale factor (e.g., `0.05`)
- `min_distance_threshold` (e.g., 1 tile) to stop rewarding "hovering" on top of food

**Alternative (grid-based):**

```python
if agent is adjacent (up/down/left/right) to an edible food tile:
    r_approach_food = +0.1
else:
    r_approach_food = 0
```

## 4.3 Eating Reward
Main objective reward when consuming food:

- `r_eat = +1.0` (or larger, depending on scaling)

This keeps eating as the primary learning signal.

## 4.4 Penalties (Optional)
Penalize clearly bad behaviour:

- `r_penalty = -0.1` when walking into walls/hazards
- `r_penalty = -X` when performing invalid actions

Keep penalties moderate so they do not overpower positive rewards.

## 4.5 Example `compute_reward` Logic

```python
def compute_reward(agent, prev_state, curr_state, ate_food, hit_hazard):
    reward = 0.0

    # Survival reward
    if agent.alive:
        reward += 0.01

    # Approach food (distance-based)
    prev_dist = distance_to_nearest_food(prev_state, agent)
    curr_dist = distance_to_nearest_food(curr_state, agent)

    if prev_dist is not None and curr_dist is not None:
        delta = prev_dist - curr_dist
        if delta > 0 and prev_dist > 1.0:
            reward += 0.05 * delta

    # Eating reward
    if ate_food:
        reward += 1.0

    # Penalties
    if hit_hazard:
        reward -= 0.1

    return reward
```

---

# 5. Agent Observation Design
Agents need useful inputs.
Minimum recommended features:
- own position `(x, y)`
- own energy
- vector to nearest food `(dx, dy)`
- optional: a small egocentric grid (local map)

If they can’t sense the world, they can’t learn about it.

---

# 6. Fission and Mutation (Evolution Layer)

This section defines how surviving agents create offspring across generations.

## 6.1 Terminology
- **Generation**: one full cycle where all agents live out an episode (or fixed steps).
- **Parent**: an agent from the current generation used to create offspring.
- **Offspring**: new agents in the next generation, created by cloning and mutating parents.
- **Elitism**: keeping top-performing agents unchanged into the next generation.

## 6.2 Fitness Definition
Define a fitness score per agent (per episode). Example:

```python
fitness = steps_survived + 5.0 * food_eaten
```

You can adjust the weight on `food_eaten` and optionally include other metrics.

## 6.3 Selection
After all agents finish an episode:

1. Compute fitness for each agent.
2. Sort agents by fitness (descending).
3. Choose top `K` as parents (e.g., `K = 3`).
4. Optionally, mark top `E` as elites (e.g., `E = 1 or 2`).

## 6.4 Fission-Based Reproduction
Use **fission**: parents simply replicate into children.

High-level idea:

```python
POP_SIZE = 10

# 1) Elites: copy best agents without mutation
elites = best_agents[:E]
new_population = [clone_agent(e, mutate=False) for e in elites]

# 2) Fill remaining slots with children from top-K parents
while len(new_population) < POP_SIZE:
    parent = random_choice(best_agents[:K])
    apply_mutation = should_mutate(len(new_population))
    new_population.append(clone_agent(parent, mutate=apply_mutation))
```

Where `clone_agent` copies:
- network architecture
- trained weights (end-of-life weights)
- resets episodic stats (energy, age, etc.)

## 6.5 Mutation Strategy
Apply small Gaussian noise to parent weights to produce variation.

```python
def mutate_weights(parameters, sigma=0.02):
    for p in parameters:
        if not p.requires_grad:
            continue
        noise = torch.normal(mean=0.0, std=sigma, size=p.data.shape, device=p.device)
        p.data += noise
```

Guidelines:
- Start with `sigma` around `0.01–0.05`.
- If population collapses / becomes unstable → reduce `sigma`.
- If population stagnates → carefully increase `sigma`.

## 6.6 Lamarckian Evolution
Use the **final trained weights** of the parent (after its RL training) as the base for offspring.

In pseudocode:

```python
def clone_agent(parent, mutate: bool):
    child = Agent(brain_architecture=parent.brain_architecture)

    # inherit trained weights
    child.brain.load_state_dict(parent.brain.state_dict())

    # optional mutation
    if mutate:
        mutate_weights(child.brain.parameters(), sigma=0.02)

    # reset life-specific values
    child.reset_life_state()
    return child
```

This is Lamarckian evolution: learning within an agent’s lifetime is passed to offspring.

## 6.7 Generation Loop Skeleton

```python
def run_generation(population, env, max_steps):
    for agent in population:
        state = env.reset_for_agent(agent)
        total_reward = 0.0

        for step in range(max_steps):
            obs = get_observation(state, agent)
            action = agent.brain(obs)
            next_state, step_reward, done, info = env.step(agent, action)

            # RL update for the agent (if any)
            agent.update_from_experience(obs, action, step_reward, next_state, done)

            total_reward += step_reward
            state = next_state

            if done:
                break

        agent.fitness = compute_fitness(agent, total_reward)

    # After all agents finish
    new_population = next_generation(population)
    return new_population


def next_generation(population):
    # sort by fitness
    sorted_agents = sorted(population, key=lambda a: a.fitness, reverse=True)

    # select parents & elites
    parents = sorted_agents[:K]
    elites  = sorted_agents[:E]

    new_population = [clone_agent(e, mutate=False) for e in elites]

    while len(new_population) < len(population):
        parent = random_choice(parents)
        mutate = (len(new_population) % 2 == 0)  # example pattern
        new_population.append(clone_agent(parent, mutate=mutate))

    return new_population
```

---

# 7. Curriculum Learning
Introduce difficulty in stages.

### Stage 1 – Baby Mode
- many food items
- flat terrain
- low decay
- zero hazards

### Stage 2 – Intermediate
- moderate food
- add walls
- small hazards

### Stage 3 – Final
- sparse food
- energy decay normal
- real challenge

Agents graduate when avg. fitness passes a threshold.

---

# 8. Debugging Trick: Hand-Coded Agent
Before training, test a simple heuristic agent:
- move toward nearest food
- avoid hazards

If even this agent dies quickly, the world is too harsh.

---

# 9. Training Flow (for RL + Evolution)

### Simplified loop:
```python
population = init_population(POP_SIZE)

for generation in range(max_generations):
    population = run_generation(population, env, max_steps)
    log_generation_stats(population, generation)
```

Where `run_generation` implements the per-agent RL loop, fitness calculation, and the call to `next_generation`.

---

# 10. Checklist for Agent Learnability & Evolution
- [ ] Agents survive at least 200+ steps randomly
- [ ] Agents frequently see food
- [ ] Reward is not zero most of the time
- [ ] Observations include cues for food/objects
- [ ] Mutation strength is small and controlled
- [ ] Elitism enabled
- [ ] Difficulty increases gradually (curriculum)
- [ ] Fission + mutation correctly create a new generation

If all are satisfied, learning and evolution should begin to show clear improvement over generations.

---

# 11. Next Steps
After stabilizing learning and evolution, you can:
- add crossover mating (combine two parents instead of just cloning)
- log full transition data `(state_t, action_t, state_t+1)` for training a world model
- add planning via a learned world model
- experiment with different reward structures and environments

This file should be used by GitHub Copilot to generate code for:
- reward shaping
- fission-based reproduction
- mutation
- and the overall evolutionary RL training loop.

