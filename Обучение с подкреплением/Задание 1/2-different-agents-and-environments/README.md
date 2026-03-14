# BanditSim (Environment + BaseSampler + RandomSampler + ε-greedy)

Мини-набор Python-файлов для симуляции **многорукого бандита**:

- `banditsim/environment.py` — среда (генерирует награды и управляет прогоном)
- `banditsim/samplers/base.py` — базовый класс агента с логированием
- `banditsim/samplers/random_sampler.py` — baseline: случайный выбор
- `banditsim/samplers/egreedy.py` — ε-жадная стратегия с warm-up (`n_learning`)
- `banditsim/experiment.py` — многократные прогоны + агрегация
- `banditsim/plotting.py` — графики (matplotlib)
- `run_experiment.py` — пример запуска и построения графиков

## Установка зависимостей

```bash
pip install numpy pandas matplotlib
```

## Запуск

```bash
python run_experiment.py
```

После запуска откроются графики:
- Mean cumulative reward
- Mean cumulative regret
- Actions over time (random)
- Actions over time (e-greedy)
- Total reward distribution

## Как добавить новую стратегию

1. Создайте файл в `banditsim/samplers/`.
2. Наследуйтесь от `BaseSampler`.
3. Реализуйте `choose_k()` (и при необходимости переопределите `update()`).
4. Добавьте фабрику в `run_experiment.py`.

---

Основа терминов и примеров соответствует описанию из вашего материала про сравнение ε-greedy и Thompson sampling.
