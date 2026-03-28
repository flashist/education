"""Учебный пример кооперативного мультиагентного обучения.

Два агента движутся по решетке 4x4 и должны оказаться в общей целевой клетке.
Пример показывает идею полной координации: выбирается совместное действие
(joint action), а Q-функция хранится для пары действий одновременно.
"""

import random
from collections import defaultdict

ACTIONS = ['up', 'down', 'left', 'right', 'stay']


class MeetingWorld:
    def __init__(self, size=4, goal=(2, 2)):
        self.size = size
        self.goal = goal
        self.reset()

    def reset(self):
        # Агенты стартуют в противоположных углах.
        self.a1 = (0, 0)
        self.a2 = (self.size - 1, self.size - 1)
        return self.state()

    def state(self):
        # Глобальное состояние включает позиции обоих агентов.
        return (self.a1, self.a2)

    def _apply(self, pos, action):
        r, c = pos
        if action == 'up':
            r = max(0, r - 1)
        elif action == 'down':
            r = min(self.size - 1, r + 1)
        elif action == 'left':
            c = max(0, c - 1)
        elif action == 'right':
            c = min(self.size - 1, c + 1)
        return (r, c)

    def step(self, action1, action2):
        # Оба агента совершают ход одновременно.
        self.a1 = self._apply(self.a1, action1)
        self.a2 = self._apply(self.a2, action2)

        reward = -1
        done = False
        if self.a1 == self.goal and self.a2 == self.goal:
            reward = 30
            done = True
        return self.state(), reward, done


def all_joint_actions():
    return [(a1, a2) for a1 in ACTIONS for a2 in ACTIONS]


def epsilon_greedy_joint(Q, state, epsilon=0.2):
    # При полной координации система выбирает совместное действие сразу.
    if random.random() < epsilon:
        return random.choice(all_joint_actions())
    return max(all_joint_actions(), key=lambda act: Q[(state, act)])


def train_joint_q(episodes=300, alpha=0.2, gamma=0.95, epsilon=0.2):
    env = MeetingWorld()
    Q = defaultdict(float)

    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            joint_action = epsilon_greedy_joint(Q, state, epsilon)
            next_state, reward, done = env.step(*joint_action)

            best_next = max(Q[(next_state, act)] for act in all_joint_actions())
            td_target = reward + gamma * best_next
            td_error = td_target - Q[(state, joint_action)]
            Q[(state, joint_action)] += alpha * td_error

            state = next_state
            steps += 1

    return Q


def greedy_rollout(Q):
    # Демонстрация того, чему научилась joint-policy.
    env = MeetingWorld()
    state = env.reset()
    done = False
    history = [(env.a1, env.a2)]
    steps = 0

    while not done and steps < 30:
        action = max(all_joint_actions(), key=lambda act: Q[(state, act)])
        state, reward, done = env.step(*action)
        history.append((env.a1, env.a2))
        steps += 1

    return history, done


if __name__ == '__main__':
    Q = train_joint_q()
    history, success = greedy_rollout(Q)
    print('=== Кооперативная мультиагентная задача ===')
    print('Успех:', success)
    for i, positions in enumerate(history):
        print(f'Шаг {i:2d}: агент1={positions[0]}, агент2={positions[1]}')