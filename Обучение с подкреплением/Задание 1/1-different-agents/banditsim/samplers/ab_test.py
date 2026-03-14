"""banditsim.samplers.ab_test

A/B тестирование: фаза исследования (round-robin) + фаза эксплуатации (лучшая рука).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseSampler


class ABTestSampler(BaseSampler):
    """Классический A/B тест.

    Фаза исследования: перебирает все руки по кругу (round-robin),
    каждую руку тянем n_explore раз.

    Фаза эксплуатации: фиксируется на руке с наибольшим theta_hat.

    Parameters
    ----------
    n_arms:
        Количество рук.
    payouts:
        Истинные вероятности успеха (для расчёта regret).
    n_explore:
        Количество шагов на каждую руку в фазе исследования.
    rng_seed:
        Seed генератора случайных чисел.
    """

    name: str = "ab-test"

    def __init__(
        self,
        n_arms: int,
        payouts: Optional[np.ndarray] = None,
        n_explore: int = 50,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_arms=n_arms, payouts=payouts, rng_seed=rng_seed)
        self.n_explore = int(n_explore)
        self._best_arm: Optional[int] = None

    def choose_k(self) -> int:
        t = len(self.logs)
        explore_steps = self.n_arms * self.n_explore

        if t < explore_steps:
            arm = t % self.n_arms
        else:
            if self._best_arm is None:
                self._best_arm = int(np.argmax(self.theta_hat))
            arm = self._best_arm

        self.last_action = arm
        return arm
