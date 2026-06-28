# Planner P3 — learn the policy *in imagination* (measured)

Phase **P3** of [`../PLANNING_PROPOSAL.md`](../PLANNING_PROPOSAL.md): instead of
planning at decision time, **train the actor/critic on rollouts imagined in the
latent world model** (Dreamer-style), so a good policy emerges and the per-tick
planner can be turned **off** — decision time goes back to one forward pass.

Implemented in `agents/ppo.py` (`TorchBrainMirror.imagine_loss`, a config-gated
auxiliary loss; default **off**). From a batch of detached hidden states it rolls
the actor forward in the dynamics head, scores with TD(λ) returns from the critic,
and updates the actor (REINFORCE + value baseline) and critic. The dynamics/
encoder get no gradient from it (returns detached) — they are trained by the
world-model loss; imagination only distils planning into the policy. Unit tests:
`tests/test_imagination.py`.

## Matched A/B (same seed 42, 64×64, v3.5 + PPO, 2000 ticks) — **planner OFF in both**

Identical brains (world-model head present in both), no runtime planner; the only
difference is whether the actor is also trained in imagination.

| Metric | no imagination | **+ imagination (P3)** | Δ |
|---|---|---|---|
| Avg peak fitness | 32.6 | **40.6** | **+25%** |
| Mean fitness (final gen) | 45.9 | **48.2** | +5% |
| Avg lifespan | 403 | **450** | +11% |
| Seeds planted | 61 | **84** | +38% |
| WAIT share (idling) | 31.0% | **15.4%** | −16pt |
| EAT attempts | 458 | 187 | fewer |
| Tiles / agent | 52.2 | 24.6 | fewer |
| ticks/s (with learning) | 9.05 | 7.22 | −20% (training only) |

## Finding

- **Imagination training lifts the policy without any planner**: peak fitness
  **+25%**, longer lifespan, more planting, and far less idle waiting — at full
  inference speed (no per-tick rollouts). The policy became more *efficient*
  (less aimless eating/wandering, more deliberate) rather than more active.
- **This is the P3 promise realised**: it nearly matches the best decision-time
  planner from P2 (CEM peak fitness 44.5) **without CEM's per-tick cost** — CEM
  ran at 5.19 ticks/s *with* the planner, P3 at 7.22 ticks/s and the slowdown is
  a one-time **training** cost (imagination backprop), not a per-decision tax.
  With learning frozen at deployment, the P3 policy costs one forward pass.
- Trade-off: the imagination-trained policy explores/eats less. Whether that is
  net-good depends on the task; here it coincided with higher fitness and
  survival.

Recommended config: [`config/planning_p3_v35.yaml`](../../config/planning_p3_v35.yaml)
(`learning.ppo.imagination.enabled: true`, `horizon: 5`, `weight: 0.5`,
`lambda: 0.95`). The default everywhere is **off**.

## Caveats

Single seed, 2000 ticks, 64×64, population 30, in-silico fitness — effect sizes,
not significance. Imagination trains the critic on **model-predicted** returns,
so it is only as trustworthy as the world model (the P2 model-error-discipline
item, deferred, would harden this). EXPERIMENTAL; default off.

## Reproduce

```bash
python main.py --no-viz --config config/planning_p3_base_v35.yaml \
    --learning --mode rl --seed 42 --generations 2 --log --log-dir noimag
python main.py --no-viz --config config/planning_p3_v35.yaml \
    --learning --mode rl --seed 42 --generations 2 --log --log-dir imag
python scripts/analyze_logs.py --file noimag/agent_actions_*.csv
python scripts/analyze_logs.py --file imag/agent_actions_*.csv
```
