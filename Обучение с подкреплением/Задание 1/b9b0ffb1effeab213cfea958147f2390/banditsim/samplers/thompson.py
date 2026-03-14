"""banditsim.samplers.thompson

Томпсоновское сэмплирование (Thompson Sampling) для Bernoulli multi-armed bandit.

Идея (простыми словами)
-----------------------
Для каждой руки k мы поддерживаем *апостериорное* распределение вероятности успеха p_k.
Если награда Бернулли (0/1), то естественная сопряжённая априорная/апостериорная пара —
распределение Бета:

    p_k ~ Beta(alpha_k, beta_k)

После наблюдения reward ∈ {0,1} параметры обновляются так:

    alpha_k <- alpha_k + reward
    beta_k  <- beta_k + (1 - reward)

На каждом шаге мы делаем:
1) для каждой руки сэмплируем "возможную истинную" вероятность успеха:
       theta_k ~ Beta(alpha_k, beta_k)
2) выбираем руку с максимальным theta_k

Таким образом стратегия автоматически балансирует exploration/exploitation:
- руки с малым числом наблюдений имеют широкое распределение → иногда "выстреливают";
- руки с хорошей статистикой получают высокие сэмплы чаще → их выбирают чаще.

Ссылки на терминологию:
- alpha, beta иногда называют "псевдо-успехи" и "псевдо-неудачи" (prior counts).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .base import BaseSampler


class ThompsonSampler(BaseSampler):
    """Thompson Sampling для Bernoulli bandit.

    Parameters
    ----------
    n_arms:
        Количество рук.
    payouts:
        Истинные payout'ы (вероятности успеха) — только для подсчёта regret в симуляции.
    prior_alpha, prior_beta:
        Параметры Beta-априора для каждой руки.
        - prior_alpha = prior_beta = 1 соответствует равномерному распределению Beta(1,1).
        - Можно поставить, например, 0.5 для более "острого" априора (Jeffreys prior).
    rng_seed:
        Seed для генератора случайных чисел.

    Notes
    -----
    - Работает корректно именно для наград {0,1}. Для других наград нужен другой байесовский
      апдейт (например, Normal-Normal для гауссовских).
    """

    def __init__(
        self,
        n_arms: int,
        payouts: Optional[np.ndarray] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        rng_seed: Optional[int] = None,
        name: str = "thompson",
    ) -> None:
        super().__init__(n_arms=n_arms, payouts=payouts, rng_seed=rng_seed)
        # Переопределяем "человеческое" имя стратегии (будет в DataFrame/графиках)
        self.name = name

        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be > 0")

        # Для удобства держим параметры постериора по каждой руке отдельными массивами
        self.prior_alpha = float(prior_alpha)
        self.prior_beta = float(prior_beta)

        self.alpha = np.full(self.n_arms, self.prior_alpha, dtype=float)
        self.beta = np.full(self.n_arms, self.prior_beta, dtype=float)

    def choose_k(self) -> int:
        """Выбрать руку: сэмплируем вероятности из Beta и берём argmax."""
        # Важно: используем self.rng, чтобы эксперименты были воспроизводимыми.
        # rng.beta умеет принимать массивы параметров и возвращает массив сэмплов.
        samples = self.rng.beta(self.alpha, self.beta)

        a = int(np.argmax(samples))
        self.last_action = a
        return a

    def update(self, reward: int) -> None:
        """Обновить постериор Beta и общий лог.

        Здесь два слоя обновления:
        1) Байесовский апдейт alpha/beta (это "сердце" Thompson Sampling)
        2) Общий учёт pulls/successes/logs (делает BaseSampler.update)

        Порядок можно менять — главное, чтобы оба слоя случились.
        """
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1 for Bernoulli bandit")
        if self.last_action is None:
            raise RuntimeError("last_action is None; choose_k() must set self.last_action")

        a = int(self.last_action)

        # 1) Байесовский апдейт
        self.alpha[a] += float(reward)
        self.beta[a] += float(1 - reward)

        # 2) Общий лог/метрики
        super().update(reward)

    def to_dict(self) -> Dict[str, Any]:
        """Добавим alpha/beta к стандартной сериализации (удобно для дебага)."""
        d = super().to_dict()
        d.update(
            {
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "alpha": self.alpha.copy(),
                "beta": self.beta.copy(),
            }
        )
        return d
