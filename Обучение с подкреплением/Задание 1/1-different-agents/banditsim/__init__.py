"""banditsim

Мини-библиотека для симуляции задачи многорукого бандита.

Состав:
- Environment (environment.py)
- Samplers (папка samplers/): BaseSampler, RandomSampler, EGreedy
- Утилиты для экспериментов и графиков (experiment.py, plotting.py)

Быстрый старт смотрите в run_experiment.py (в корне архива).
"""

from .environment import Environment

__all__ = ["Environment"]
