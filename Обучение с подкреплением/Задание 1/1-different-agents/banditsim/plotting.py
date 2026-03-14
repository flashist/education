"""banditsim.plotting

Функции для построения графиков результатов.

Особенности этой версии:
- Подписи и заголовки — по-русски.
- На графиках показываем *итоговое количество очков* (суммарная награда) для каждого агента.

Функции здесь принимают pandas.DataFrame из banditsim.experiment.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Чтобы кириллица отображалась корректно на большинстве систем
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _final_totals_from_mean_df(mean_df: pd.DataFrame) -> dict[str, float]:
    """Достаём итоговую среднюю награду (в последнем t) для каждого агента."""
    totals: dict[str, float] = {}
    for agent, sub in mean_df.groupby("agent"):
        # mean_df уже усреднён по прогонам, поэтому последнее значение = средний итог.
        last_row = sub.loc[sub["t"].idxmax()]
        totals[str(agent)] = float(last_row["cumulative_reward"])
    return totals


def plot_mean_cumulative_reward(mean_df: pd.DataFrame, title: str = "Средняя суммарная награда") -> None:
    """Средняя кумулятивная награда по агентам.

    В легенде и на графике показываем итоговые очки (среднее по прогонам).
    """
    totals = _final_totals_from_mean_df(mean_df)

    plt.figure()
    for agent, sub in mean_df.groupby("agent"):
        a = str(agent)
        plt.plot(sub["t"], sub["cumulative_reward"], label=f"{a} (итого ≈ {totals[a]:.0f})")

    plt.xlabel("Шаг t")
    plt.ylabel("Средняя суммарная награда (очки)")
    plt.title(title)
    plt.legend()

    # Компактный блок с итогами (дублируем, чтобы было видно даже при длинной легенде)
    lines = [f"{a}: ≈ {totals[a]:.0f}" for a in sorted(totals.keys())]
    plt.gca().text(
        0.02,
        0.98,
        "Итоговые очки (среднее):\n" + "\n".join(lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()


def plot_mean_cumulative_regret(mean_df: pd.DataFrame, title: str = "Среднее кумулятивное сожаление") -> None:
    """Среднее кумулятивное сожаление по агентам."""
    plt.figure()
    for agent, sub in mean_df.groupby("agent"):
        plt.plot(sub["t"], sub["cumulative_regret"], label=str(agent))

    plt.xlabel("Шаг t")
    plt.ylabel("Среднее кумулятивное сожаление")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_actions_scatter(steps_df: pd.DataFrame, agent: str, seed: int, title: Optional[str] = None) -> None:
    """Точечный график: какой action выбирался на каждом шаге.

    В заголовке показываем *итоговые очки* для этого конкретного прогона.
    """
    sub = steps_df[(steps_df["agent"] == agent) & (steps_df["seed"] == seed)].copy()
    if sub.empty:
        raise ValueError(f"Нет данных для agent={agent!r}, seed={seed}")

    # Итоговые очки = последняя cumulative_reward
    total_points = float(sub.loc[sub["t"].idxmax(), "cumulative_reward"])

    plt.figure()
    plt.scatter(sub["t"], sub["action"], s=6)
    plt.xlabel("Шаг t")
    plt.ylabel("Выбранная рука (action)")
    plt.title(title or f"Выборы действий: {agent} (seed={seed}), итого очков = {total_points:.0f}")
    plt.tight_layout()


def plot_total_reward_hist(summary_df: pd.DataFrame, title: str = "Распределение итоговых очков") -> None:
    """Гистограмма итоговой награды по многим прогонам.

    В легенде добавляем среднее итоговых очков по каждому агенту.
    """
    if not {"agent_name", "total_reward"}.issubset(summary_df.columns):
        raise ValueError("summary_df должен содержать колонки: agent_name, total_reward")

    plt.figure()
    agents = list(summary_df["agent_name"].unique())

    for agent in agents:
        vals = summary_df.loc[summary_df["agent_name"] == agent, "total_reward"].to_numpy()
        mean_val = float(np.mean(vals)) if len(vals) else float("nan")
        plt.hist(vals, bins=20, alpha=0.5, label=f"{agent} (ср. ≈ {mean_val:.0f})")

    plt.xlabel("Итоговые очки (суммарная награда)")
    plt.ylabel("Количество прогонов")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
