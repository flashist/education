"""banditsim.samplers.random_sampler

RandomSampler — базовый (baseline) агент, который просто выбирает руку случайно.

Он полезен как:
- нижняя граница качества (все остальные стратегии должны быть лучше);
- тест корректности среды/логирования.

Важно:
- Этот агент НЕ использует информацию о прошлых наградах для выбора.
- Но мы всё равно ведём статистику pulls/successes/theta_hat для анализа.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseSampler


class RandomSampler(BaseSampler):
    """Случайный агент."""

    name = "random"

    def __init__(
        self,
        n_arms: int,
        payouts: Optional[np.ndarray] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_arms=n_arms, payouts=payouts, rng_seed=rng_seed)

    def choose_k(self) -> int:
        # Выбираем руку равновероятно из {0, ..., n_arms-1}
        action = int(self.rng.integers(low=0, high=self.n_arms))

        # Обязательно сохраняем, чтобы BaseSampler.update() знал, что обновлять
        self.last_action = action
        return action
