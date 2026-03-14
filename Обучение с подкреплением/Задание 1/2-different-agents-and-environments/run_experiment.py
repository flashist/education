"""run_experiment.py

Запуск экспериментов для многорукого бандита в трёх типах сред:
1. Стационарная — payout'ы не меняются.
2. Истощение   — каждое нажатие уменьшает payout руки.
3. Истощение + восстановление — истощение + постепенный возврат к базовым значениям.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from banditsim.environment import DepletionEnvironment, DepletionRecoveryEnvironment
from banditsim.experiment import mean_curves, run_many
from banditsim.plotting import (
    plot_actions_scatter,
    plot_mean_cumulative_regret,
    plot_mean_cumulative_reward,
    plot_total_reward_hist,
)
from banditsim.samplers import ABTestSampler, EGreedy, ThompsonSampler


def run_scenario(label, payouts, n_trials, n_runs, base_seed, agent_factories, env_factory=None):
    """Запустить один сценарий и показать все 6 графиков в одном окне."""
    steps_df, summary_df = run_many(
        payouts=payouts,
        n_trials=n_trials,
        agent_factories=agent_factories,
        n_runs=n_runs,
        base_seed=base_seed,
        env_factory=env_factory,
    )

    mean_df = mean_curves(steps_df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(label, fontsize=14, fontweight="bold")

    example_seed = base_seed
    plot_mean_cumulative_reward(mean_df, title="Средняя кумулятивная награда", ax=axes[0, 0])
    plot_mean_cumulative_regret(mean_df, title="Среднее кумулятивное сожаление", ax=axes[0, 1])
    plot_total_reward_hist(summary_df, title="Распределение суммарной награды", ax=axes[0, 2])
    plot_actions_scatter(steps_df, agent="ab-test", seed=example_seed,
                         title="Действия A/B-теста", ax=axes[1, 0])
    plot_actions_scatter(steps_df, agent="e-greedy", seed=example_seed,
                         title="Действия ε-жадного", ax=axes[1, 1])
    plot_actions_scatter(steps_df, agent="thompson", seed=example_seed,
                         title="Действия Thompson", ax=axes[1, 2])

    fig.tight_layout()

    print(f"\n=== {label} — Summary (first 10 rows) ===")
    print(summary_df.head(10).to_string(index=False))


def main() -> None:
    payouts = [0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11]

    N_TRIALS = 10_000
    N_RUNS = 50
    BASE_SEED = 42

    agent_factories = {
        "ab-test": lambda n_arms, p, seed: ABTestSampler(
            n_arms=n_arms, payouts=p, n_explore=50, rng_seed=seed
        ),
        "e-greedy": lambda n_arms, p, seed: EGreedy(
            n_arms=n_arms,
            payouts=p,
            epsilon=0.1,
            n_learning=500,
            rng_seed=seed,
        ),
        "thompson": lambda n_arms, p, seed: ThompsonSampler(
            n_arms=n_arms,
            payouts=p,
            prior_alpha=1.0,
            prior_beta=1.0,
            rng_seed=seed,
        ),
    }

    # Сценарий 1: Стационарная среда
    run_scenario(
        label="Стационарная среда",
        payouts=payouts,
        n_trials=N_TRIALS,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        agent_factories=agent_factories,
        env_factory=None,
    )

    # Сценарий 2: Истощение
    run_scenario(
        label="Истощение",
        payouts=payouts,
        n_trials=N_TRIALS,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        agent_factories=agent_factories,
        env_factory=lambda p, n, s: DepletionEnvironment(
            p, n, rng_seed=s, depletion_rate=0.01
        ),
    )

    # Сценарий 3: Истощение + восстановление
    run_scenario(
        label="Истощение + восстановление",
        payouts=payouts,
        n_trials=N_TRIALS,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        agent_factories=agent_factories,
        env_factory=lambda p, n, s: DepletionRecoveryEnvironment(
            p, n, rng_seed=s, depletion_rate=0.01, recovery_rate=0.003
        ),
    )

    plt.show()


if __name__ == "__main__":
    main()
