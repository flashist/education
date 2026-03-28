import random

# ============================================================
# Учебный пример: оценка функции ценности состояния V(s)
# методом TD(0) в небольшой дискретной среде GridWorld.
#
# Идея примера:
# 1. У нас есть среда из клеток.
# 2. Агент действует по фиксированной стратегии.
# 3. Мы НЕ улучшаем стратегию, а только оцениваем,
#    насколько полезно каждое состояние.
# 4. Для оценки используем TD(0):
#       V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
# ============================================================

class GridWorld:
    def __init__(self, width=4, height=4, goal=(3, 3)):
        self.width = width
        self.height = height
        self.goal = goal
        self.start = (0, 0)

    def reset(self):
        # Возвращаем стартовое состояние эпизода
        return self.start

    def is_terminal(self, state):
        # Терминальное состояние: цель
        return state == self.goal

    def step(self, state, action):
        # Делаем переход по действию.
        # Если выходим за границу поля, остаемся на месте.
        x, y = state

        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.height - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(self.width - 1, x + 1)

        next_state = (x, y)

        # Награда:
        # - небольшой штраф за каждый обычный шаг;
        # - положительная награда за достижение цели.
        if next_state == self.goal:
            reward = 10.0
        else:
            reward = -0.2

        done = self.is_terminal(next_state)
        return next_state, reward, done


def policy(state, env):
    # Фиксированная стратегия поведения.
    # Мы не выбираем оптимальное действие,
    # а просто задаем вероятностное правило.
    # Небольшое смещение в сторону цели помогает
    # быстрее собрать осмысленные данные.
    actions = ["up", "down", "left", "right"]
    x, y = state
    goal_x, goal_y = env.goal

    preferred = []
    if x < goal_x:
        preferred.append("right")
    if y < goal_y:
        preferred.append("down")

    # С вероятностью 0.7 выбираем одно из "желательных" действий,
    # если оно существует; иначе действуем случайно.
    if preferred and random.random() < 0.7:
        return random.choice(preferred)
    return random.choice(actions)


def td0_value_estimation(
    env,
    episodes=500,
    alpha=0.1,
    gamma=0.95,
    max_steps_per_episode=100
):
    # Таблица оценок V(s).
    # Для простоты и наглядности храним значения в словаре.
    V = {
        (x, y): 0.0
        for y in range(env.height)
        for x in range(env.width)
    }

    history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps_per_episode):
            action = policy(state, env)
            next_state, reward, done = env.step(state, action)

            # TD-ошибка:
            # delta = r + gamma * V(s') - V(s)
            # Для терминального состояния будущая ценность не учитывается.
            target = reward
            if not done:
                target += gamma * V[next_state]

            td_error = target - V[state]

            # Обновление оценки состояния
            V[state] += alpha * td_error

            total_reward += reward
            state = next_state

            if done:
                break

        history.append(total_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = sum(history[-50:]) / 50
            print(f"Эпизод {episode + 1:3d} | средняя награда за последние 50 = {avg_reward:7.3f}")

    return V, history


def print_value_table(V, env):
    print("\nОценка функции ценности V(s):")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            state = (x, y)
            if state == env.goal:
                row.append(" GOAL ")
            else:
                row.append(f"{V[state]:6.2f}")
        print(" | ".join(row))


if __name__ == "__main__":
    random.seed(42)

    env = GridWorld(width=4, height=4, goal=(3, 3))
    V, history = td0_value_estimation(
        env,
        episodes=500,
        alpha=0.1,
        gamma=0.95,
        max_steps_per_episode=100
    )

    print_value_table(V, env)
