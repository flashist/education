"""banditsim.samplers

Пакет со стратегиями (агентами) для многорукого бандита.
"""

from .base import BaseSampler, AgentLog
from .random_sampler import RandomSampler
from .egreedy import EGreedy
from .thompson import ThompsonSampler

__all__ = [
    "BaseSampler",
    "AgentLog",
    "RandomSampler",
    "EGreedy",
    "ThompsonSampler",
]
