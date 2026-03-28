import random

# ============================================================
# Задание 9:
# Провести серию запусков при разных γ: 0.7, 0.9, 0.99.
# Объяснить, как меняется «дальновидность» агента.
#
# Среда: GridWorld 4×4, цель в правом нижнем углу (3, 3).
# Алгоритм: TD(0) для оценки V(s) при фиксированной стратегии.
#
# Формула TD(0):
#   V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
#
# γ (gamma) — коэффициент дисконтирования:
#   - при малом γ агент ценит только ближайшие награды;
#   - при γ близком к 1 агент учитывает долгосрочные последствия.
# ============================================================


class GridWorld:
    def __init__(self, width=4, height=4, goal=(3, 3)):
        self.width = width
        self.height = height
        self.goal = goal
        self.start = (0, 0)

    def reset(self):
        return self.start

    def is_terminal(self, state):
        return state == self.goal

    def step(self, state, action):
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

        if next_state == self.goal:
            reward = 10.0
        else:
            reward = -0.2

        done = self.is_terminal(next_state)
        return next_state, reward, done


def policy(state, env):
    # Фиксированная стратегия со смещением в сторону цели.
    actions = ["up", "down", "left", "right"]
    x, y = state
    goal_x, goal_y = env.goal

    preferred = []
    if x < goal_x:
        preferred.append("right")
    if y < goal_y:
        preferred.append("down")

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
    V = {
        (x, y): 0.0
        for y in range(env.height)
        for x in range(env.width)
    }

    for episode in range(episodes):
        state = env.reset()

        for step in range(max_steps_per_episode):
            action = policy(state, env)
            next_state, reward, done = env.step(state, action)

            target = reward
            if not done:
                target += gamma * V[next_state]

            V[state] += alpha * (target - V[state])
            state = next_state

            if done:
                break

    return V


def print_value_table(V, env, gamma):
    print(f"\nγ = {gamma} | Таблица V(s):")
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
    gammas = [0.7, 0.9, 0.99]
    env = GridWorld(width=4, height=4, goal=(3, 3))

    print("=" * 60)
    print("Сравнение V(s) при разных значениях γ (gamma)")
    print("Среда: GridWorld 4×4, цель (3,3), alpha=0.1, episodes=500")
    print("=" * 60)

    results = {}
    for gamma in gammas:
        # Фиксируем seed для воспроизводимости при каждом γ.
        random.seed(42)
        V = td0_value_estimation(env, episodes=500, alpha=0.1, gamma=gamma)
        results[gamma] = V
        print_value_table(V, env, gamma)

    # Вывод разброса значений для наглядности.
    print("\n" + "=" * 60)
    print("V(s) по полю (min / max / разница мин-макс, без клетки цели):")
    for gamma in gammas:
        V = results[gamma]
        non_goal = [v for (s, v) in V.items() if s != env.goal]
        print(f"  γ = {gamma}: min = {min(non_goal):6.2f}, max = {max(non_goal):6.2f}, "
              f"разница мин-макс = {max(non_goal) - min(non_goal):.2f}")

    print("\nВывод:")
    print("  γ = 0.7  — дальние клетки получают сильно заниженные оценки V(s),")
    print("             агент «близорукий» — почти не учитывает будущие награды.")
    print("  γ = 0.9  — умеренная дальновидность.")
    print("  γ = 0.99 — все клетки получают близкие высокие оценки V(s),")
    print("             агент «дальновидный» — хорошо видит ценность даже далёких состояний.")
