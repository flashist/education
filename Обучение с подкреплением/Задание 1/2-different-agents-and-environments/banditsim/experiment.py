"""banditsim.experiment

Утилиты для проведения экспериментов и подготовки данных для графиков.

Здесь мы отделяем:
- *симуляцию* (Environment + agent)
- *агрегацию результатов* (таблицы, усреднения по многим запускам)

Так легче:
- сравнивать несколько агентов;
- делать много прогонов для получения "средних" кривых.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .environment import Environment
from .samplers.base import BaseSampler


@dataclass
class RunSummary:
    """Короткая сводка по одному прогону."""

    agent_name: str
    seed: int
    total_reward: int
    total_regret: float


def run_single(
    payouts: Sequence[float],
    n_trials: int,
    agent_factory: Callable[[int, np.ndarray, int], BaseSampler],
    seed: int,
    drift_std: float = 0.0,
    env_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[pd.DataFrame, RunSummary]:
    """Один прогон: создать среду и агента, выполнить n_trials шагов.

    Parameters
    ----------
    payouts:
        Истинные вероятности успеха для рук.
    n_trials:
        Длина симуляции.
    agent_factory:
        Функция, которая создаёт агента. Сигнатура:
            agent_factory(n_arms, payouts_array, seed) -> BaseSampler
        Seed передаём, чтобы обеспечить воспроизводимость.
    seed:
        Seed конкретного прогона.
    drift_std:
        Шум (дрейф) payout'ов внутри среды.
    env_factory:
        Опциональная фабрика среды: (payouts_arr, n_trials, seed) -> Environment.
        Если None, используется стандартный Environment.

    Returns
    -------
    df:
        Таблица по шагам (t, action, reward, cumulative_reward, regret, cumulative_regret).
    summary:
        Итог по прогону.
    """

    payouts_arr = np.asarray(payouts, dtype=float)
    if env_factory is not None:
        env = env_factory(payouts_arr, n_trials, seed)
    else:
        env = Environment(payouts=payouts_arr, n_trials=n_trials, rng_seed=seed, drift_std=drift_std)

    agent = agent_factory(env.n_arms, env.payouts, seed)
    env.run(agent)

    # Конвертируем logs -> DataFrame
    df = pd.DataFrame([log.__dict__ for log in agent.logs])
    df["agent"] = agent.name
    df["seed"] = seed

    summary = RunSummary(
        agent_name=agent.name,
        seed=seed,
        total_reward=agent.total_reward,
        total_regret=agent.total_regret,
    )

    return df, summary


def run_many(
    payouts: Sequence[float],
    n_trials: int,
    agent_factories: Dict[str, Callable[[int, np.ndarray, int], BaseSampler]],
    n_runs: int = 50,
    base_seed: int = 123,
    drift_std: float = 0.0,
    env_factory: Optional[Callable[..., Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Запустить несколько агентов много раз и вернуть данные.

    Возвращаем:
    - "long" DataFrame по шагам для всех прогонов
    - summary DataFrame по каждому прогону

    Seeds делаем детерминированно: seed = base_seed + run_id.
    """

    all_steps: List[pd.DataFrame] = []
    all_summaries: List[Dict[str, object]] = []

    for run_id in range(n_runs):
        seed = base_seed + run_id
        for _name, factory in agent_factories.items():
            df, summary = run_single(
                payouts=payouts,
                n_trials=n_trials,
                agent_factory=factory,
                seed=seed,
                drift_std=drift_std,
                env_factory=env_factory,
            )
            all_steps.append(df)
            all_summaries.append(summary.__dict__)

    steps_df = pd.concat(all_steps, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)

    return steps_df, summary_df


def mean_curves(steps_df: pd.DataFrame) -> pd.DataFrame:
    """Посчитать средние кривые по шагам для каждого агента.

    Вход: long-таблица (как из run_many).
    Выход: таблица со средними значениями cumulative_reward и cumulative_regret
    по каждому t и agent.
    """

    required_cols = {"t", "agent", "cumulative_reward", "cumulative_regret"}
    if not required_cols.issubset(set(steps_df.columns)):
        missing = required_cols - set(steps_df.columns)
        raise ValueError(f"steps_df missing columns: {sorted(missing)}")

    grouped = (
        steps_df.groupby(["agent", "t"], as_index=False)[["cumulative_reward", "cumulative_regret"]]
        .mean()
        .sort_values(["agent", "t"])
    )
    return grouped
