"""banditsim.samplers.egreedy

ε-жадная (epsilon-greedy) стратегия.

Интуиция (как в классическом описании):
- с вероятностью (1 - ε) эксплуатируем: выбираем руку с максимальной оценкой theta_hat;
- с вероятностью ε исследуем: выбираем случайную руку.

Проблема cold-start:
- Если theta_hat изначально все нули, то argmax всегда 0.
- Поэтому часто делают "стартовое обучение" (warm-up): первые n_learning шагов
  выбирать случайно.

Этот класс реализует оба механизма:
- warm-up первые n_learning шагов (по умолчанию 0);
- epsilon exploration после warm-up.

Замечание:
- В отличие от кода из статьи, здесь мы НЕ храним массив ep заранее.
  Мы сэмплируем случайность на каждом шаге через rng.random().
  Это проще и воспроизводимо при заданном seed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import BaseSampler


class EGreedy(BaseSampler):
    """ε-жадный агент."""

    name = "e-greedy"

    def __init__(
        self,
        n_arms: int,
        payouts: Optional[np.ndarray] = None,
        epsilon: float = 0.1,
        n_learning: int = 0,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__(n_arms=n_arms, payouts=payouts, rng_seed=rng_seed)

        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        if n_learning < 0:
            raise ValueError("n_learning must be >= 0")

        self.epsilon = float(epsilon)
        self.n_learning = int(n_learning)

    def choose_k(self) -> int:
        t = len(self.logs)  # номер текущего шага

        # 1) Warm-up: первые n_learning шагов выбираем случайно
        if t < self.n_learning:
            action = int(self.rng.integers(low=0, high=self.n_arms))
            self.last_action = action
            return action

        # 2) После warm-up используем ε-жадное правило
        explore = self.rng.random() < self.epsilon

        if explore:
            # Исследуем: выбираем случайную руку
            action = int(self.rng.integers(low=0, high=self.n_arms))
        else:
            # Эксплуатируем: выбираем руку с максимальной оценкой theta_hat
            # Если есть несколько одинаковых максимумов, случайно выбираем среди них.
            max_val = float(np.max(self.theta_hat))
            candidates = np.flatnonzero(self.theta_hat == max_val)
            action = int(self.rng.choice(candidates))

        self.last_action = action
        return action
