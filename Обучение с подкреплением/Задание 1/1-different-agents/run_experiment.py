"""run_experiment.py

Запуск демо-эксперимента для многорукого бандита.

Что делает скрипт:
1) задаёт истинные payout'ы (вероятности успеха) для K рук;
2) запускает несколько агентов на множестве прогонов;
3) строит графики:
   - средняя кумулятивная награда;
   - среднее кумулятивное сожаление;
   - пример поведения на одном прогоне (actions scatter);
   - распределение total_reward по прогонам.

Как запускать
-----------
1) Установите зависимости:
   pip install numpy pandas matplotlib

2) Запустите:
   python run_experiment.py

Подсказки
---------
- Хотите больше/меньше шагов? Измените N_TRIALS.
- Хотите больше прогонов для более гладких кривых? Измените N_RUNS.
- Хотите "нестабильную" среду? Поставьте DRIFT_STD > 0.
"""

from __future__ import annotations

import numpy as np

from banditsim.experiment import mean_curves, run_many
from banditsim.plotting import (
    plot_actions_scatter,
    plot_mean_cumulative_regret,
    plot_mean_cumulative_reward,
    plot_total_reward_hist,
)
from banditsim.samplers import ABTestSampler, EGreedy, ThompsonSampler


def main() -> None:
    # Пример payout'ов (как в учебном примере: один вариант сильно лучше)
    payouts = [0.023, 0.03, 0.029, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.11]

    N_TRIALS = 10_000
    N_RUNS = 50
    BASE_SEED = 42

    # Если > 0, то на каждый прогон payout'ы будут слегка шуметь (среда меняется)
    DRIFT_STD = 0.0

    # Фабрики агентов: чтобы experiment.run_many мог создавать новых агентов на каждый прогон
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

    steps_df, summary_df = run_many(
        payouts=payouts,
        n_trials=N_TRIALS,
        agent_factories=agent_factories,
        n_runs=N_RUNS,
        base_seed=BASE_SEED,
        drift_std=DRIFT_STD,
    )

    mean_df = mean_curves(steps_df)

    # --- Графики средних кривых ---
    plot_mean_cumulative_reward(mean_df)
    plot_mean_cumulative_regret(mean_df)

    # --- Пример одного прогона (поведение стратегии) ---
    example_seed = BASE_SEED  # первый прогон
    plot_actions_scatter(steps_df, agent="ab-test", seed=example_seed)
    plot_actions_scatter(steps_df, agent="e-greedy", seed=example_seed)
    plot_actions_scatter(steps_df, agent="thompson", seed=example_seed)

    # --- Распределение total reward ---
    plot_total_reward_hist(summary_df)

    # Показать всё
    import matplotlib.pyplot as plt

    plt.show()

    # Печать сводки (быстро посмотреть в консоли)
    print("\nSummary (first 10 rows):")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
