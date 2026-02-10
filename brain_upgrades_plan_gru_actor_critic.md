# Brain Upgrades Plan (GRU + Actor–Critic)

Goal: enable **ecological engineering / long-horizon behaviors** (returning to patches, delayed payoff strategies) by upgrading the agent brain from a **stateless MLP + epsilon-greedy** to a **memoryful Actor–Critic**.

---

## Why we are upgrading the Brain first
Your current Brain is a **memoryless MLP** that maps the current 64-d observation → 8 action probabilities. This works for reactive foraging, but it cannot represent:
- “I planted / used something here earlier”
- “This region is a good patch; return later”
- any delayed payoff strategy with multi-step commitment

To unlock ecological engineering, we need:
1) **Memory** (so strategies can persist over time)
2) **Temporal credit assignment** (so delayed rewards can propagate)

---

## Upgrades we are doing

### Upgrade 1 — Add Memory (GRU)
**Change:** Brain becomes `Encoder MLP → GRU → policy/value heads`.

**Effect:**
- Policies can now condition on **recent history**, not just the current observation.
- Agents can learn persistent behaviors (patrol, revisit, “stay near fertile patch”, etc.).

### Upgrade 2 — Add Critic (Actor–Critic)
**Change:** Add a second head that predicts state value **V(s)**.

**Effect:**
- Learning becomes far more stable than plain REINFORCE.
- Enables long-horizon behaviors because advantage uses `r + γV(s') − V(s)`.

### Upgrade 3 — Replace epsilon-greedy exploration with policy sampling (+ entropy)
**Change:** Remove epsilon coin-flip; always sample from policy distribution.

**Effect:**
- Exploration becomes **state-dependent** and consistent with policy gradients.
- Less random “suicidal” actions; cleaner learning signal.

### Upgrade 4 — (Optional, recommended) Action masking
**Change:** Mask invalid actions before softmax.

**Effect:**
- Prevents wasting probability mass on impossible actions (EAT w/o food, PICK_UP on empty tile).
- Reduces the need for harsh reward shaping penalties.

---

## Target Brain v2 architecture
- **Input:** 64-d observation
- **Encoder:** small MLP (default `[32]`)
- **Memory:** GRU hidden size 32
- **Outputs:**
  - Policy logits → softmax over 8 actions
  - Value head → scalar V(s)

All parameters still live in **Genome.weights (flat vector)** to keep evolution + Lamarckian inheritance working.

---

## Implementation plan (in sequence)

### Step A — Replace `agents/brain.py` with Brain v2
Key API changes:
- `Brain.initial_state() -> np.ndarray`
- `Brain.forward(obs, h, action_mask=None, temperature=1.0) -> (probs, value, h_next)`
- `Brain.decide(obs, h, action_mask=None, temperature=1.0) -> (Action, h_next, value)`

**Important:** Brain no longer exposes `weights/biases` as before; it unpacks parameters into structured dicts.

### Step B — Update weight_count in `main.py`
Your current code uses `Brain.calculate_weight_count(input_size, hidden_layers, output_size)`.

Update to:
- `encoder_layers` (list)
- `gru_hidden_size` (int)

And update YAML:
```yaml
brain:
  input_size: 64
  encoder_layers: [32]
  gru_hidden_size: 32
  output_size: 8
```

### Step C — Minimal glue changes in `agents/agent.py`
Add GRU state to Agent:
- In `__init__`: `self.h = self.brain.initial_state()`
- In `update()`: replace `brain.decide(observation, epsilon=...)` with:
  - `action, h_next, value = self.brain.decide(observation, h=self.h, temperature=1.0)`
  - `self.h = h_next`
- Reset `self.h`:
  - on death
  - on reproduction (offspring starts with fresh hidden state)

### Step D — Update `agents/learning.py` to Actor–Critic
Your current learner does manual backprop on `brain.weights`/`brain.biases` (MLP only).

Update learning rule to:
- Store transitions: `(obs, h, action, reward, next_obs, next_h, done)`
- Compute advantage:
  - `A = r + γ * V(next) * (1-done) - V(curr)`
- Loss:
  - Policy loss: `-logπ(a|s) * A`
  - Value loss: `0.5 * (V(curr) - target)^2`
  - Entropy bonus: `-β * H(π)`

This keeps everything compatible with your logger + Lamarckian sync.

---

## Expected behavior changes (what you should observe)
After Brain v2 (even before changing world/actions):
- Agents show **less twitchy**, more consistent action sequences.
- Increased persistence: “stay near a patch”, “return to areas”, “patrol loops”.
- Better survival with fewer random failures.

After Actor–Critic learning update:
- Faster learning
- More stable policies
- Less dependence on heavy reward shaping

Once we later tune world physics and incentives, this brain will support **ecological engineering** behaviors.

---

## Copilot tasks checklist
1) Replace `agents/brain.py` with Brain v2 (Encoder+GRU+ActorCritic)
2) Update `Brain.calculate_weight_count()` usage in `main.py`
3) Update YAML configs (`brain.encoder_layers`, `brain.gru_hidden_size`)
4) Update `agents/agent.py` to maintain GRU hidden state `self.h`
5) Update `agents/learning.py` to Actor–Critic (TD advantage + entropy)

---

## Notes / guardrails
- Keep the genome flat vector representation to preserve evolution.
- Do not change action space or world physics until Brain v2 + Actor–Critic is stable.
- First validate a headless run compiles; then enable learning.

