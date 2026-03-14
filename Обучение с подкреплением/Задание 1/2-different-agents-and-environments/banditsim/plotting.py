"""banditsim.plotting

Функции для построения графиков результатов.

Особенности этой версии:
- Подписи и заголовки — по-русски.
- На графиках показываем *итоговое количество очков* (суммарная награда) для каждого агента.
- Каждая функция принимает необязательный параметр `ax`. Если передан — рисует в него;
  если нет — создаёт новый figure (обратная совместимость).

Функции здесь принимают pandas.DataFrame из banditsim.experiment.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd

# Чтобы кириллица отображалась корректно на большинстве систем
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _final_totals_from_mean_df(mean_df: pd.DataFrame) -> dict[str, float]:
    """Достаём итоговую среднюю награду (в последнем t) для каждого агента."""
    totals: dict[str, float] = {}
    for agent, sub in mean_df.groupby("agent"):
        last_row = sub.loc[sub["t"].idxmax()]
        totals[str(agent)] = float(last_row["cumulative_reward"])
    return totals


def plot_mean_cumulative_reward(
    mean_df: pd.DataFrame,
    title: str = "Средняя суммарная награда",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """Средняя кумулятивная награда по агентам."""
    totals = _final_totals_from_mean_df(mean_df)

    if ax is None:
        plt.figure()
        ax = plt.gca()

    for agent, sub in mean_df.groupby("agent"):
        a = str(agent)
        ax.plot(sub["t"], sub["cumulative_reward"], label=f"{a} (итого ≈ {totals[a]:.0f})")

    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Средняя суммарная награда (очки)")
    ax.set_title(title)
    ax.legend()

    lines = [f"{a}: ≈ {totals[a]:.0f}" for a in sorted(totals.keys())]
    ax.text(
        0.02, 0.98,
        "Итоговые очки (среднее):\n" + "\n".join(lines),
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=8,
    )


def plot_mean_cumulative_regret(
    mean_df: pd.DataFrame,
    title: str = "Среднее кумулятивное сожаление",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """Среднее кумулятивное сожаление по агентам."""
    if ax is None:
        plt.figure()
        ax = plt.gca()

    for agent, sub in mean_df.groupby("agent"):
        ax.plot(sub["t"], sub["cumulative_regret"], label=str(agent))

    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Среднее кумулятивное сожаление")
    ax.set_title(title)
    ax.legend()


def plot_actions_scatter(
    steps_df: pd.DataFrame,
    agent: str,
    seed: int,
    title: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """Точечный график: какой action выбирался на каждом шаге."""
    sub = steps_df[(steps_df["agent"] == agent) & (steps_df["seed"] == seed)].copy()
    if sub.empty:
        raise ValueError(f"Нет данных для agent={agent!r}, seed={seed}")

    total_points = float(sub.loc[sub["t"].idxmax(), "cumulative_reward"])

    if ax is None:
        plt.figure()
        ax = plt.gca()

    ax.scatter(sub["t"], sub["action"], s=6)
    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Выбранная рука (action)")
    ax.set_title(title or f"Выборы действий: {agent} (seed={seed}), итого очков = {total_points:.0f}")


def plot_total_reward_hist(
    summary_df: pd.DataFrame,
    title: str = "Распределение итоговых очков",
    ax: Optional[matplotlib.axes.Axes] = None,
) -> None:
    """Гистограмма итоговой награды по многим прогонам."""
    if not {"agent_name", "total_reward"}.issubset(summary_df.columns):
        raise ValueError("summary_df должен содержать колонки: agent_name, total_reward")

    if ax is None:
        plt.figure()
        ax = plt.gca()

    agents = list(summary_df["agent_name"].unique())
    for agent in agents:
        vals = summary_df.loc[summary_df["agent_name"] == agent, "total_reward"].to_numpy()
        mean_val = float(np.mean(vals)) if len(vals) else float("nan")
        ax.hist(vals, bins=20, alpha=0.5, label=f"{agent} (ср. ≈ {mean_val:.0f})")

    ax.set_xlabel("Итоговые очки (суммарная награда)")
    ax.set_ylabel("Количество прогонов")
    ax.set_title(title)
    ax.legend()
