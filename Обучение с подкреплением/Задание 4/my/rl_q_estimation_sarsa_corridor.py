import random

# ============================================================
# Отдельная задача:
# Оценка функции ценности действия Q(s, a) методом SARSA.
#
# Среда:
# - прямоугольное поле 6x4;
# - есть цель и несколько "опасных" клеток;
# - агент стартует в левом нижнем углу;
# - за шаг начисляется -1, за ловушку -15, за цель +20.
#
# Смысл примера:
# 1. Мы оцениваем не V(s), а Q(s, a).
# 2. Это позволяет понять, какое действие лучше в каждом состоянии.
# 3. SARSA обновляет оценку по реально выбранному следующему действию:
#    Q(s, a) <- Q(s, a) + alpha * [r + gamma*Q(s', a') - Q(s, a)]
# ============================================================

class CorridorEnv:
    def __init__(self, width=6, height=4):
        self.width = width
        self.height = height
        self.start = (0, height - 1)
        self.goal = (width - 1, height - 1)
        self.traps = {(2, height - 1), (3, height - 1), (4, height - 1)}

    def reset(self):
        return self.start

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

        if next_state in self.traps:
            reward = -15.0
            done = True
        elif next_state == self.goal:
            reward = 20.0
            done = True
        else:
            reward = -1.0
            done = False

        return next_state, reward, done


ACTIONS = ["up", "down", "left", "right"]


def epsilon_greedy(Q, state, epsilon):
    # epsilon-жадная стратегия:
    # иногда исследуем случайные действия,
    # а обычно берем действие с наибольшей Q-оценкой.
    if random.random() < epsilon:
        return random.choice(ACTIONS)

    values = [Q[(state, a)] for a in ACTIONS]
    best_value = max(values)

    # Если несколько действий одинаково хороши,
    # выбираем одно из них случайно.
    best_actions = [a for a in ACTIONS if Q[(state, a)] == best_value]
    return random.choice(best_actions)


def sarsa_train(
    env,
    episodes=600,
    alpha=0.1,
    gamma=0.95,
    epsilon=0.2,
    max_steps_per_episode=100
):
    # Таблица Q-оценок.
    Q = {}
    for y in range(env.height):
        for x in range(env.width):
            for action in ACTIONS:
                Q[((x, y), action)] = 0.0

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        episode_reward = 0.0

        for step in range(max_steps_per_episode):
            next_state, reward, done = env.step(state, action)
            episode_reward += reward

            if done:
                # Если эпизод завершился, будущая Q-ценность равна 0.
                target = reward
                Q[(state, action)] += alpha * (target - Q[(state, action)])
                break
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                target = reward + gamma * Q[(next_state, next_action)]
                Q[(state, action)] += alpha * (target - Q[(state, action)])

                state = next_state
                action = next_action

        rewards_history.append(episode_reward)

        # Постепенно уменьшаем исследование.
        epsilon = max(0.03, epsilon * 0.995)

        if (episode + 1) % 50 == 0:
            avg_reward = sum(rewards_history[-50:]) / 50
            print(f"Эпизод {episode + 1:3d} | средняя награда за последние 50 = {avg_reward:7.3f}")

    return Q, rewards_history


def print_policy(Q, env):
    print("\nПриближенная стратегия по Q(s, a):")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            state = (x, y)

            if state == env.goal:
                row.append(" GOAL ")
            elif state in env.traps:
                row.append(" TRAP ")
            else:
                best_action = max(ACTIONS, key=lambda a: Q[(state, a)])
                row.append(best_action.center(6))

        print(" | ".join(row))


if __name__ == "__main__":
    random.seed(7)

    env = CorridorEnv(width=6, height=4)
    Q, rewards = sarsa_train(
        env,
        episodes=600,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        max_steps_per_episode=100
    )

    print_policy(Q, env)
