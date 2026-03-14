"""banditsim.samplers.base

Базовый класс для агентов (samplers) в задаче многорукого бандита.

Зачем он нужен:
- хранить общие поля (оценки, счётчики, логи);
- дать единый формат данных для дальнейшего построения графиков;
- упростить написание новых стратегий.

Важное соглашение:
- Environment вызывает:
    k = agent.choose_k()
    agent.update(reward)
  на каждом шаге.

Здесь мы выбираем максимально прозрачную реализацию:
- поддерживаем "оценку вероятности успеха" theta_hat[k] = successes[k] / pulls[k]
  (для Bernoulli-наград);
- считаем regret относительно истинного оптимального payout (если он известен).

Если вы захотите другие награды/метрики — этот базовый класс легко адаптируется.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class AgentLog:
    """Лог одного шага (что выбрали и что получили)."""

    t: int
    action: int
    reward: int
    cumulative_reward: int
    regret: float
    cumulative_regret: float


class BaseSampler:
    """Базовый агент для Bernoulli-armed bandit.

    Parameters
    ----------
    n_arms:
        Количество рук (вариантов).
    payouts:
        Истинные вероятности успеха (если известны). Нужны для вычисления regret.
        В учебных симуляциях мы их знаем, но в реальном мире — нет.
    rng_seed:
        Seed для генератора случайных чисел (используется, например, в random-политиках).

    Attributes
    ----------
    pulls:
        Сколько раз каждая рука была выбрана.
    successes:
        Сколько успехов (reward=1) по каждой руке.
    theta_hat:
        Текущая оценка вероятности успеха для каждой руки.
    logs:
        Список AgentLog, заполняется при update().

    Notes
    -----
    - BaseSampler не реализует choose_k(): это задача конкретной стратегии.
    - update() реализован и одинаков для большинства простых стратегий:
      обновляем pulls/successes/theta_hat и пишем лог.
    - Некоторые стратегии (например, Thompson sampling) поддерживают
      дополнительные параметры (alpha/beta). Их можно хранить в наследниках.
    """

    name: str = "base"

    def __init__(
        self,
        n_arms: int,
        payouts: Optional[np.ndarray] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")

        self.n_arms = int(n_arms)
        self.payouts = None if payouts is None else np.asarray(payouts, dtype=float)

        # Генератор для случайностей внутри агента
        self.rng = np.random.default_rng(rng_seed)

        # Статистика
        self.pulls = np.zeros(self.n_arms, dtype=int)
        self.successes = np.zeros(self.n_arms, dtype=int)
        self.theta_hat = np.zeros(self.n_arms, dtype=float)

        # Текущее действие (важно: update() будет считать, что self.last_action задан)
        self.last_action: Optional[int] = None

        # Метрики
        self.total_reward = 0
        self.total_regret = 0.0

        # Логи по шагам
        self.logs: List[AgentLog] = []

    # ---------- Интерфейс, который ждёт Environment ----------
    def choose_k(self) -> int:
        """Выбрать руку.

        Должен быть переопределён в наследнике.
        """
        raise NotImplementedError

    def update(self, reward: int) -> None:
        """Обновить состояние агента после получения награды.

        reward должен быть 0 или 1 (Bernoulli).

        Алгоритм обновления:
        1) увеличиваем pulls[action]
        2) если reward=1, увеличиваем successes[action]
        3) пересчитываем theta_hat[action]
        4) считаем regret (если известны истинные payouts)
        5) пишем AgentLog

        Важно: метод предполагает, что choose_k() уже записал выбранную руку
        в self.last_action.
        """
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1 for Bernoulli bandit")
        if self.last_action is None:
            raise RuntimeError("last_action is None; choose_k() must set self.last_action")

        a = int(self.last_action)

        self.pulls[a] += 1
        self.successes[a] += int(reward)

        # Обновление оценки вероятности (MLE для Bernoulli)
        self.theta_hat[a] = self.successes[a] / max(1, self.pulls[a])

        # Кумулятивная награда
        self.total_reward += int(reward)

        # Regret: разница между оптимальным ожидаемым выигрышем и ожидаемым выигрышем выбранной руки
        regret = 0.0
        if self.payouts is not None:
            optimal = float(np.max(self.payouts))
            regret = optimal - float(self.payouts[a])

        self.total_regret += regret

        self.logs.append(
            AgentLog(
                t=len(self.logs),
                action=a,
                reward=int(reward),
                cumulative_reward=self.total_reward,
                regret=regret,
                cumulative_regret=self.total_regret,
            )
        )

    def finalize(self) -> None:
        """Хук после прогона (можно переопределять).

        По умолчанию ничего не делает.
        """
        return

    # ---------- Утилиты для анализа ----------
    def to_dict(self) -> Dict[str, Any]:
        """Сериализовать ключевые поля агента (для печати/логирования)."""
        return {
            "name": self.name,
            "n_arms": self.n_arms,
            "total_reward": self.total_reward,
            "total_regret": self.total_regret,
            "pulls": self.pulls.copy(),
            "successes": self.successes.copy(),
            "theta_hat": self.theta_hat.copy(),
        }
