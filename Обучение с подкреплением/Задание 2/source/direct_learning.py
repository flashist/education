import random
from collections import defaultdict


class LinearWorld:
    """
    Простая эпизодическая среда.

    Состояния: 0, 1, 2, 3, 4
    Старт: 2
    Терминальные состояния:
        0 -> проигрыш, награда -10
        4 -> цель, награда +10

    Действия:
        0 -> влево
        1 -> вправо

    За каждый обычный шаг: -1
    """

    def __init__(self):
        self.start_state = 2
        self.terminal_left = 0
        self.terminal_right = 4
        self.state = self.start_state

    def reset(self):
        """
        Сброс среды в начальное состояние.
        Возвращает стартовое состояние.
        """
        self.state = self.start_state
        return self.state

    def step(self, action):
        """
        Выполнить действие и вернуть:
        next_state, reward, done

        action:
            0 - влево
            1 - вправо
        """
        if self.state in (self.terminal_left, self.terminal_right):
            raise ValueError("Эпизод уже завершён. Сначала вызовите reset().")

        # Переход в новое состояние
        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += 1
        else:
            raise ValueError("Неизвестное действие. Допустимы только 0 и 1.")

        # Ограничение на диапазон состояний
        self.state = max(0, min(4, self.state))

        # Проверка завершения эпизода
        if self.state == self.terminal_left:
            return self.state, -10, True
        elif self.state == self.terminal_right:
            return self.state, 10, True
        else:
            return self.state, -1, False


class MonteCarloAgent:
    """
    Агент, обучающийся методом Monte Carlo по полным эпизодам.

    Он хранит:
    - Q(s, a): оценку ценности действия a в состоянии s
    - epsilon: вероятность случайного действия
    - alpha: скорость обучения
    - gamma: коэффициент дисконтирования
    """

    def __init__(self, actions=(0, 1), alpha=0.1, gamma=1.0, epsilon=0.2):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-таблица: для каждого состояния храним словарь по действиям
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def choose_action(self, state):
        """
        ε-жадный выбор действия.

        С вероятностью epsilon агент исследует среду
        и выбирает случайное действие.
        С вероятностью 1 - epsilon выбирает лучшее действие по Q.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = self.Q[state]
        max_q = max(q_values.values())

        # Если несколько действий одинаково хороши, выбираем случайно из лучших
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def generate_episode(self, env, max_steps=100):
        """
        Сгенерировать один полный эпизод.

        Возвращает список шагов:
        episode = [(state, action, reward), ...]

        Здесь reward — это награда, полученная ПОСЛЕ выполнения action в state.
        """
        episode = []
        state = env.reset()

        for _ in range(max_steps):
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)

            episode.append((state, action, reward))
            state = next_state

            if done:
                break

        return episode

    def update_from_episode_every_visit(self, episode):
        """
        Every-Visit Monte Carlo update.

        Для каждого шага эпизода вычисляем возврат G_t
        и обновляем Q(s, a) по формуле:

            Q(s, a) <- Q(s, a) + alpha * (G_t - Q(s, a))

        Возвраты считаются с конца эпизода.
        """
        returns = [0.0] * len(episode)
        G = 0.0

        # Идём с конца к началу и считаем возвраты
        for t in reversed(range(len(episode))):
            _, _, reward = episode[t]
            G = reward + self.gamma * G
            returns[t] = G

        # Every-Visit: обновляем для каждого появления пары (s, a)
        for t, (state, action, _) in enumerate(episode):
            G_t = returns[t]
            old_q = self.Q[state][action]
            self.Q[state][action] = old_q + self.alpha * (G_t - old_q)

    def greedy_policy(self):
        """
        Вернуть текущую жадную стратегию:
        для каждого состояния указывается лучшее действие.
        """
        policy = {}
        for state, q_values in self.Q.items():
            best_action = max(q_values, key=q_values.get)
            policy[state] = best_action
        return policy


def print_q_table(agent):
    """
    Красивый вывод текущих оценок Q(s, a).
    """
    print("Текущая таблица Q(s, a):")
    for state in sorted(agent.Q.keys()):
        left_q = agent.Q[state][0]
        right_q = agent.Q[state][1]
        print(f"  Состояние {state}: LEFT={left_q:7.3f}, RIGHT={right_q:7.3f}")
    print()


def print_policy(agent):
    """
    Вывод текущей жадной стратегии.
    """
    action_names = {0: "LEFT", 1: "RIGHT"}
    policy = agent.greedy_policy()

    print("Текущая жадная стратегия:")
    for state in sorted(policy.keys()):
        print(f"  Состояние {state}: {action_names[policy[state]]}")
    print()


def main():
    random.seed(42)

    env = LinearWorld()
    agent = MonteCarloAgent(
        actions=(0, 1),
        alpha=0.1,
        gamma=1.0,
        epsilon=0.2
    )

    episodes_count = 500

    for episode_num in range(1, episodes_count + 1):
        episode = agent.generate_episode(env)
        agent.update_from_episode_every_visit(episode)

        # Периодически смотрим, чему научился агент
        if episode_num % 100 == 0:
            print(f"===== После {episode_num} эпизодов =====")
            print_q_table(agent)
            print_policy(agent)

    print("===== Финальный результат =====")
    print_q_table(agent)
    print_policy(agent)


if __name__ == "__main__":
    main()