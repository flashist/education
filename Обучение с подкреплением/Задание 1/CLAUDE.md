# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Homework for a university course on Reinforcement Learning. Implements multi-armed bandit simulations in two parts:

- **`1-different-agents/`** — Part 1 (complete): three agents (A/B test, ε-greedy, Thompson Sampling) in a stationary Bernoulli environment.
- **`2-different-agents-and-environments/`** — Part 2 (in progress): same agents but across multiple environment types (stationary / depletion / depletion+recovery).

## Running

```bash
cd 1-different-agents   # or 2-different-agents-and-environments
pip install numpy pandas matplotlib
python run_experiment.py
```

No tests, no build step.

## Architecture

Each part has identical module layout under `banditsim/`:

```
banditsim/
  environment.py        # Environment class: holds payouts, runs simulation loop
  experiment.py         # run_many() + mean_curves(): multi-run aggregation -> DataFrames
  plotting.py           # matplotlib helpers consuming the DataFrames from experiment.py
  samplers/
    base.py             # BaseSampler: pulls/successes/theta_hat, logging, regret calc
    ab_test.py          # ABTestSampler: round-robin explore, then exploit best arm
    egreedy.py          # EGreedy: epsilon-greedy with optional warm-up (n_learning)
    thompson.py         # ThompsonSampler: Bayesian Beta-posterior per arm
    random_sampler.py   # RandomSampler: uniform random baseline
```

**Data flow:** `run_experiment.py` → `experiment.run_many()` → `Environment.run(agent)` per run → `agent.logs` (list of `AgentLog`) → converted to `steps_df` (long DataFrame) → `plotting.*` functions consume it.

**Agent contract** (enforced by `AgentProtocol` in `environment.py`):
- `choose_k() -> int`: sets `self.last_action`, returns arm index
- `update(reward: int)`: called with 0/1 after each step; `BaseSampler.update()` does pulls/successes/regret/logging; subclasses call `super().update(reward)` after their own update
- `finalize()`: called once after the run

**Regret** is computed inside `BaseSampler.update()` as `max(payouts) - payouts[chosen_arm]`, requires `payouts` to be passed to the agent constructor (for simulation purposes only — agents don't use it for decisions).

## Part 2: What Needs to Be Added

The task requires simulating environments with:
1. **No depletion** — already exists (`Environment` with `drift_std=0`)
2. **Depletion** — payout of an arm decreases as it is pulled
3. **Depletion + Recovery** — payout decreases when pulled, recovers over time when not pulled

New environment types should either subclass `Environment` or be separate classes implementing the same `run()` interface. The `experiment.py` machinery (`run_many`, `mean_curves`) can be reused as-is if the environment exposes the same interface.
