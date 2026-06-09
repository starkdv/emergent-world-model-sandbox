# Emergent World-Model Sandbox

### Technical Overview & Research Aspiration

**Author:** Karan Vasa
**Prepared for:** Review by an AI / ML faculty member
**Date:** June 2026

> *A note on scope:* This document is a conceptual overview intended to convey the
> research framing, methodological stance, and trajectory of the project. It intentionally
> stays at the level of methods and design rationale rather than implementation specifics
> (exact hyperparameters, reward formulations, and architectural internals are omitted).
> I'm happy to go deeper on any part that is of interest.

---

## 1. One-Paragraph Summary

This is a multi-agent artificial-life sandbox in which embodied agents survive in a
physically consistent, resource-driven 2D world using only a small set of primitive
actions. Each agent is a compact recurrent **Actor–Critic** policy. The system supports
two evolutionary regimes under one engine — **pure neuroevolution** (genome-only, no
gradients) and **online reinforcement learning with Lamarckian inheritance** (lifetime
learning whose weights are written back to the genome and passed to offspring). The
central methodological commitment is **emergence-first**: no high-level behaviour is
hand-coded, so any complex strategy that appears — foraging, cultivation, spatial
routine — must arise from selection and learning pressure alone. The current system is a
working, tested **proof of the core loop**; the longer aspiration is to grow it into a
shared engine for two applications: **sim-to-real robotics** and a **persistent online
ALife world**.

---

## 2. Research Framing

The project is organised around a few questions I find genuinely open and interesting:

- **Emergence of complex behaviour.** Under what conditions do non-trivial strategies
  (cultivation rather than mere foraging, spatial routines, eventually coordination)
  emerge from primitive action repertoires under survival pressure — with nothing
  high-level specified in advance?
- **Learning × evolution interaction.** What is gained by coupling *lifetime* learning
  (online RL) with *generational* search (neuroevolution) via Lamarckian weight
  inheritance — versus either alone? This is essentially a Baldwin-effect / Lamarckian
  question made explicit and manipulable.
- **Open-endedness.** Over very long runs, does the system keep producing genuinely novel
  behaviour, or does it converge? How should "interestingness" even be measured?
- **From reactive to model-based.** Can agents move from reactive policies to maintaining
  an internal predictive *world model* that supports anticipation and planning — and does
  that qualitatively change the behaviours that emerge?

---

## 3. System at a Glance (Methods)

| Aspect | Approach |
|---|---|
| **Agent policy** | Compact recurrent (GRU) **Actor–Critic** — shared recurrent core with policy and value heads. A few thousand parameters, deliberately small enough for embedded deployment and for thousands of concurrent agents. |
| **Observation** | Low-dimensional, **egocentric** encoding: agent-internal state, a small agent-aligned local vision field, task-relevant stimulus signals, and inventory state. |
| **Actions** | A handful of **discrete primitives** only (movement, turning, pick up / drop, consume, plant, wait). Invalid actions are masked. No composite or high-level actions exist. |
| **Lifetime learning** | Online **Actor–Critic** with TD-based advantage, entropy regularisation, and a small replay buffer. NumPy core with an optional accelerated backend. |
| **Evolution** | Continuous, overlapping-generation **reproduction** (fitness-gated) with small Gaussian mutation; tournament/crossover machinery available. **Lamarckian**: learned weights are synced to the genome and inherited. |
| **Dual mode** | A single switch selects **pure neuroevolution** (no gradients) vs **RL + Lamarckian inheritance**, enabling clean ablation of learning's contribution. |
| **Bootstrapping** | Lightweight, **fading instinct biases** scaffold early survival, then yield to learned weights — so they aid exploration without becoming hardcoded behaviour. |
| **World** | A genuine environment model — resource spawning, plant growth and decay, soil/moisture dynamics, terrain hazards, and periodic calamities — providing real, non-stationary selection pressure rather than a static reward surface. |

---

## 4. Methodological Positioning

I want to be candid about what is standard and what is the interesting part, since that's
usually the most useful thing for a reviewer to react to.

**Standard, well-understood building blocks:** recurrent Actor–Critic RL, TD advantage,
entropy bonuses, Gaussian-mutation neuroevolution, masked discrete action spaces.

**The combination I find research-interesting:**

1. **Lamarckian inheritance coupled to online RL in an open population.** Agents learn
   within a lifetime and pass *learned* weights (plus mutation) to offspring through
   continuous, overlapping-generation reproduction — rather than discrete generational
   resets. This makes the learning/evolution interaction directly observable and tunable.
2. **A toggle between pure neuroevolution and RL+Lamarckian under identical conditions**,
   which turns "does lifetime learning help, and how?" into a controlled experiment rather
   than a comparison across codebases.
3. **A self-sustaining ecosystem as the selection mechanism.** Pressure comes from a
   simulated environment with its own dynamics and shocks, not a fixed task reward — which
   is closer to the conditions under which biological behaviour evolved and a better
   substrate for studying open-endedness.
4. **An emergence-first discipline enforced throughout**: the absence of hardcoded
   high-level behaviour is treated as the project's core integrity constraint and the bar
   its results are judged against.

---

## 5. Current Status (Phase 0)

The system is built, runnable, and under automated test. In its current form it
demonstrates the core loop end-to-end:

- Agents reliably learn to locate and consume resources, manage energy, and survive long
  horizons; **foraging-to-cultivation behaviour emerges** without being programmed.
- Populations **self-stabilise** and form divergent lineages with distinguishable
  strategies.
- The dual-mode design already supports **controlled comparison** of evolution with and
  without lifetime learning.
- Infrastructure for the next research step is in place: full **transition logging**
  `(state, action, reward, next_state)` is captured in a form designed for training a
  predictive world model — though that model is **not yet trained**, only logged.

I'd characterise the present state honestly as *promising and operational, not yet a
result*: the instrument works and produces the expected qualitative phenomena; the
rigorous measurement program is the next phase.

---

## 6. Aspiration

The long-term vision (detailed in the project roadmap) is to treat the current sandbox as
**Phase 0** of a shared engine — neuroevolution + recurrent brains + Lamarckian
inheritance + learned world models — that powers two application tracks.

### 6.1 The linchpin: learned world models
The single most important next capability is a **learned dynamics model** — per-agent (or
shared) predictors of `(next observation, reward)` from `(observation, action)`. This
unlocks three things at once:
- **Intrinsic motivation** — prediction error as a curiosity signal, reducing reliance on
  hand-crafted exploration incentives.
- **Imagination / planning** — Dreamer-style rollouts and model-based action selection.
- **Dream-based evolution** — running evolutionary episodes *inside* the learned model for
  large speedups over full environment simulation, with periodic grounding in the real
  environment to prevent model exploitation.

### 6.2 Track A — Sim-to-real robotics
Generalise from the grid world to **continuous physics** (2D first, then 3D via standard
engines), swap the discrete head for a **continuous (Gaussian) policy** over joint
torques, accept user-defined **morphologies**, and use the world model for
sample-efficient training. The compact brains are deliberately small enough to **export to
physical hardware**, with **domain randomisation** and a reality-grounding loop to address
the sim-to-real gap.

### 6.3 Track B — Persistent online ALife world
An always-on, continuously evolving world that doubles as a **research platform and a
public-facing demonstration** — spectating, lineage tracking, and genome
submission/competition. Beyond the engagement angle, its real scientific value is as a
source of **massive evolutionary runs and genome diversity** — exactly the data that feeds
better world models and, in turn, Track A.

### 6.4 Research directions I most want to pursue
- **Open-ended evolution**: behavioural-novelty metrics over very long runs.
- **Emergent communication & division of labour**: adding a signalling primitive and
  asking whether proto-coordination and role specialisation arise, quantified with role /
  signalling-entropy measures.
- **Interpretability**: probing recurrent hidden states for behavioural structure
  ("hungry", "returning to food"), and policy distillation to human-readable rules.

---

## 7. What I'm Seeking Feedback On

I'd value a researcher's candid view on:

- **Novelty & positioning.** Is the learning-×-evolution / Lamarckian framing genuinely
  interesting relative to the current literature, or largely settled? Where does it sit
  with respect to work I should be reading (open-ended evolution, quality-diversity,
  Baldwin-effect studies, world models, multi-agent emergence)?
- **Rigour & claims.** What measurements would make claims of "emergent" behaviour
  credible and falsifiable — appropriate baselines, ablations, statistical treatment, and
  control conditions? Where is this kind of work most often unconvincing?
- **The dual-mode comparison.** Is the neuroevolution-vs-RL+Lamarckian toggle a sound
  basis for a controlled study, and what confounds should I guard against?
- **Prioritisation.** Of the directions above, which would most likely yield a defensible,
  publishable result — and which are seductive but scientifically thin?
- **Scope realism.** The two-track ambition is large. Where should I narrow to do one
  thing well?

Encouraging, critical, or redirecting feedback is equally welcome.

---

*Thank you for taking the time to look at this — even a few pointers on framing or
relevant literature would be enormously helpful.*
