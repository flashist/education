"""banditsim.environment

Этот модуль содержит класс Environment — "окружающую среду" для задачи
многорукого бандита.

Идея (как в конспекте/статье из вашего PDF):
- Есть K вариантов (рук/баннеров/действий).
- У каждого варианта есть скрытая вероятность успеха (payout rate).
- На каждом шаге агент выбирает руку k, среда возвращает награду (0/1).

Класс Environment отвечает за:
- хранение истинных payout'ов;
- генерацию наград (Бернулли/биномиальное с n=1);
- прогон симуляции: многократно вызвать agent.choose_k(), затем agent.update().

Файл ориентирован на учебные эксперименты и удобную отладку.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import numpy as np


class AgentProtocol(Protocol):
    """Минимальный интерфейс агента, который понимает Environment.

    Environment не должен знать, как именно агент устроен внутри.
    Ему достаточно двух методов:
    - choose_k(): выбрать индекс руки (int)
    - update(reward): обновить своё состояние по наблюдаемой награде

    Дополнительно мы договоримся, что у агента есть метод finalize(), который
    вызывается один раз после окончания прогона (например, чтобы собрать DataFrame).
    """

    def choose_k(self) -> int: ...

    def update(self, reward: int) -> None: ...

    def finalize(self) -> None: ...


@dataclass
class StepResult:
    """Результат одного шага (trial) симуляции."""

    t: int
    action: int
    reward: int


class Environment:
    """Окружающая среда для многорукого бандита.

    Parameters
    ----------
    payouts:
        Истинные вероятности успеха (Бернулли) для каждой руки.
        Длина массива = K.
    n_trials:
        Сколько шагов симуляции выполнить.
    rng_seed:
        Seed для генератора случайных чисел (повторяемость экспериментов).
    drift_std:
        Если задано (>0), можно включить "дрейф" payout'ов:
        на каждый прогон будет добавлен шум N(0, drift_std) к исходным payouts,
        затем значения будут обрезаны в [0, 1].

    Notes
    -----
    - Здесь награда: reward ~ Bernoulli(p[action]).
    - Если вы захотите непрерывные награды (например, нормальные),
      достаточно заменить метод sample_reward().
    """

    def __init__(
        self,
        payouts: Sequence[float],
        n_trials: int,
        rng_seed: Optional[int] = None,
        drift_std: float = 0.0,
    ) -> None:
        self.base_payouts = np.asarray(payouts, dtype=float)
        if self.base_payouts.ndim != 1:
            raise ValueError("payouts must be a 1D sequence")
        if np.any(self.base_payouts < 0) or np.any(self.base_payouts > 1):
            raise ValueError("all payouts must be in [0, 1]")

        if n_trials <= 0:
            raise ValueError("n_trials must be positive")

        self.n_trials = int(n_trials)
        self.n_arms = int(self.base_payouts.shape[0])

        # Генератор случайных чисел (лучше, чем глобальный np.random)
        self.rng = np.random.default_rng(rng_seed)

        # Включаем "дрейф" при создании среды (один раз на прогон)
        self.drift_std = float(drift_std)
        self.payouts = self._maybe_drift(self.base_payouts)

        # Лог прогона
        self.history: List[StepResult] = []

    def reset(self) -> None:
        """Сбросить историю и (опционально) пересэмплировать дрейф payout'ов."""
        self.history = []
        self.payouts = self._maybe_drift(self.base_payouts)

    def _maybe_drift(self, payouts: np.ndarray) -> np.ndarray:
        """Применить шум (дрейф) к payout'ам, если drift_std > 0."""
        if self.drift_std <= 0:
            return payouts.copy()
        noisy = payouts + self.rng.normal(loc=0.0, scale=self.drift_std, size=payouts.shape)
        return np.clip(noisy, 0.0, 1.0)

    def sample_reward(self, action: int) -> int:
        """Сэмплировать награду для выбранного действия.

        В классическом Bernoulli-bandit:
            reward = 1 с вероятностью p[action], иначе 0.
        """
        p = float(self.payouts[action])
        # биномиальное распределение с n=1 эквивалентно Бернулли
        return int(self.rng.binomial(n=1, p=p))

    def optimal_action(self) -> int:
        """Индекс руки с максимальным истинным payout."""
        return int(np.argmax(self.payouts))

    def run(self, agent: AgentProtocol) -> List[StepResult]:
        """Запустить симуляцию.

        Возвращает список StepResult (историю), а также пишет её в self.history.

        Важно: среда не "учит" агента. Она только:
        1) просит выбрать действие,
        2) генерирует награду,
        3) сообщает награду агенту.
        """
        self.history = []

        for t in range(self.n_trials):
            action = int(agent.choose_k())
            if action < 0 or action >= self.n_arms:
                raise ValueError(
                    f"agent returned invalid action={action}; must be in [0, {self.n_arms-1}]"
                )

            reward = self.sample_reward(action)
            agent.update(reward)

            self.history.append(StepResult(t=t, action=action, reward=reward))

        agent.finalize()
        return self.history


class DepletionEnvironment(Environment):
    """Среда с истощением: каждое нажатие на руку уменьшает её payout.

    Parameters
    ----------
    depletion_rate:
        Мультипликативный коэффициент истощения. На каждом шаге после выбора
        руки k: payouts[k] *= (1 - depletion_rate). Значение остаётся в [0,1].
    """

    def __init__(
        self,
        payouts: Sequence[float],
        n_trials: int,
        rng_seed: Optional[int] = None,
        drift_std: float = 0.0,
        depletion_rate: float = 0.005,
    ) -> None:
        super().__init__(payouts, n_trials, rng_seed=rng_seed, drift_std=drift_std)
        self.depletion_rate = float(depletion_rate)

    def run(self, agent: AgentProtocol) -> List[StepResult]:
        self.history = []

        for t in range(self.n_trials):
            action = int(agent.choose_k())
            if action < 0 or action >= self.n_arms:
                raise ValueError(
                    f"agent returned invalid action={action}; must be in [0, {self.n_arms-1}]"
                )

            # Сохраняем payouts до истощения, чтобы regret считался корректно
            payouts_before = self.payouts.copy()
            reward = self.sample_reward(action)

            # Применяем истощение
            self.payouts[action] *= (1 - self.depletion_rate)

            # Агент считает regret по payouts ДО истощения
            agent.payouts = payouts_before
            agent.update(reward)

            self.history.append(StepResult(t=t, action=action, reward=reward))

        agent.finalize()
        return self.history


class DepletionRecoveryEnvironment(DepletionEnvironment):
    """Среда с истощением и восстановлением.

    После истощения выбранной руки все руки экспоненциально восстанавливаются
    к своим базовым значениям:
        payouts[k] += recovery_rate * (base_payouts[k] - payouts[k])

    Parameters
    ----------
    recovery_rate:
        Скорость восстановления (шаг экспоненциального приближения к базовому).
    """

    def __init__(
        self,
        payouts: Sequence[float],
        n_trials: int,
        rng_seed: Optional[int] = None,
        drift_std: float = 0.0,
        depletion_rate: float = 0.005,
        recovery_rate: float = 0.003,
    ) -> None:
        super().__init__(
            payouts, n_trials, rng_seed=rng_seed, drift_std=drift_std,
            depletion_rate=depletion_rate,
        )
        self.recovery_rate = float(recovery_rate)

    def run(self, agent: AgentProtocol) -> List[StepResult]:
        self.history = []

        for t in range(self.n_trials):
            action = int(agent.choose_k())
            if action < 0 or action >= self.n_arms:
                raise ValueError(
                    f"agent returned invalid action={action}; must be in [0, {self.n_arms-1}]"
                )

            payouts_before = self.payouts.copy()
            reward = self.sample_reward(action)

            # Истощение выбранной руки
            self.payouts[action] *= (1 - self.depletion_rate)

            # Восстановление всех рук к базовым значениям
            self.payouts += self.recovery_rate * (self.base_payouts - self.payouts)

            agent.payouts = payouts_before
            agent.update(reward)

            self.history.append(StepResult(t=t, action=action, reward=reward))

        agent.finalize()
        return self.history
