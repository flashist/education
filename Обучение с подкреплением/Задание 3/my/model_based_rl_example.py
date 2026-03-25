import random
from collections import defaultdict

# -----------------------------
# 1. ОПИСАНИЕ СРЕДЫ GRIDWORLD
# -----------------------------
class GridWorld:
    def __init__(self, width=5, height=5):
        # Размеры поля
        self.width = width
        self.height = height

        # Стартовое и целевое состояние
        self.start = (0, 0)
        self.goal = (4, 4)

        # Набор препятствий: в эти клетки заходить нельзя
        self.walls = {(1, 1), (1, 2), (3, 2)}

        # Все допустимые действия агента
        self.actions = ["up", "down", "left", "right"]

        # Текущее состояние среды
        self.state = self.start

    def reset(self):
        """Возврат среды к начальному состоянию."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Выполнить действие и вернуть:
        next_state, reward, done
        """
        x, y = self.state

        # Вычисляем кандидат на следующее состояние
        if action == "up":
            candidate = (x, y - 1)
        elif action == "down":
            candidate = (x, y + 1)
        elif action == "left":
            candidate = (x - 1, y)
        else:
            candidate = (x + 1, y)

        # Проверяем границы поля и препятствия.
        # Если переход недопустим, агент остается на месте.
        nx, ny = candidate
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height or candidate in self.walls:
            candidate = self.state

        self.state = candidate

        # Награда за обычный шаг
        reward = -1
        done = False

        # Если достигнута цель, завершаем эпизод
        if self.state == self.goal:
            reward = 20
            done = True

        return self.state, reward, done


# -----------------------------------------
# 2. АГЕНТ С МОДЕЛЬЮ СРЕДЫ И ПЛАНИРОВАНИЕМ
# -----------------------------------------
class ModelBasedAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.95, epsilon=0.2, planning_steps=15):
        self.actions = actions

        # Параметры обучения
        self.alpha = alpha      # скорость обучения
        self.gamma = gamma      # коэффициент дисконтирования
        self.epsilon = epsilon  # вероятность случайного действия
        self.planning_steps = planning_steps  # сколько шагов планирования делать после реального шага

        # Q-таблица: оценка полезности действия a в состоянии s
        self.Q = defaultdict(float)

        # МОДЕЛЬ СРЕДЫ:
        # model[(s, a)] = (predicted_next_state, predicted_reward)
        self.model = {}

        # Счетчики посещений для усреднения наград
        self.reward_sum = defaultdict(float)
        self.reward_count = defaultdict(int)

        # Предшественники для эвристического поиска:
        # predecessors[s_next] = set of (s, a)
        # Нужно, чтобы после изменения ценности состояния
        # можно было быстро понять, какие состояния "зависят" от него.
        self.predecessors = defaultdict(set)

        # Список пар (state, action) с приоритетами.
        # Для простоты вместо настоящей очереди с приоритетами
        # используем словарь: pair -> priority.
        self.priority = defaultdict(float)

    def get_q(self, state, action):
        """Получить значение Q(s, a)."""
        return self.Q[(state, action)]

    def best_action_value(self, state):
        """Максимальное значение Q по всем действиям в состоянии state."""
        return max(self.get_q(state, a) for a in self.actions)

    def choose_action(self, state):
        """
        epsilon-greedy выбор действия:
        иногда исследуем случайно,
        иначе берем действие с максимальным Q.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        values = [self.get_q(state, a) for a in self.actions]
        max_value = max(values)

        # Может быть несколько лучших действий - выбираем случайно среди них
        best_actions = [a for a in self.actions if self.get_q(state, a) == max_value]
        return random.choice(best_actions)

    def update_from_real_experience(self, state, action, reward, next_state):
        """
        Обычное Q-learning обновление на реальном опыте.
        """
        old_q = self.get_q(state, action)
        target = reward + self.gamma * self.best_action_value(next_state)
        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)

        # Вернем абсолютную величину изменения — это пригодится
        # как простая эвристика важности обновления.
        return abs(target - old_q)

    def update_model(self, state, action, reward, next_state):
        """
        Обновить внутреннюю модель среды.
        Здесь модель детерминированная:
        для пары (s, a) храним наиболее свежий next_state
        и среднюю награду.
        """
        key = (state, action)

        # Обновляем среднюю награду
        self.reward_sum[key] += reward
        self.reward_count[key] += 1
        avg_reward = self.reward_sum[key] / self.reward_count[key]

        # Запоминаем прогноз модели
        self.model[key] = (next_state, avg_reward)

        # Запоминаем, что состояние next_state может быть
        # достигнуто из (state, action)
        self.predecessors[next_state].add((state, action))

    def push_priority(self, state, action, amount):
        """
        Повысить приоритет пары (state, action).
        Чем больше amount, тем раньше эту пару
        стоит рассмотреть в фазе планирования.
        """
        if amount > self.priority[(state, action)]:
            self.priority[(state, action)] = amount

    def planning_step(self):
        """
        Один шаг планирования по модели.
        Выбираем пару (s, a) с максимальным приоритетом,
        затем обновляем Q как будто этот переход реально произошел.
        """
        if not self.priority:
            return

        # Находим пару с максимальным приоритетом
        state_action = max(self.priority, key=self.priority.get)
        max_priority = self.priority[state_action]

        # Если приоритет слишком мал, смысла обновлять почти нет
        if max_priority < 1e-8:
            return

        # Удаляем выбранную пару из очереди приоритетов
        del self.priority[state_action]

        state, action = state_action

        # Если модели еще нет, планировать нельзя
        if (state, action) not in self.model:
            return

        next_state, predicted_reward = self.model[(state, action)]

        # "Воображаемое" обновление Q по модели
        old_q = self.get_q(state, action)
        target = predicted_reward + self.gamma * self.best_action_value(next_state)
        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)

        # После обновления состояния next_state может измениться важность его предшественников.
        # Поэтому поднимаем приоритеты для всех (s_prev, a_prev), ведущих в state.
        for prev_state, prev_action in self.predecessors[state]:
            if (prev_state, prev_action) in self.model:
                model_next_state, model_reward = self.model[(prev_state, prev_action)]
                estimate = model_reward + self.gamma * self.best_action_value(model_next_state)
                diff = abs(estimate - self.get_q(prev_state, prev_action))
                self.push_priority(prev_state, prev_action, diff)

    def train_episode(self, env, max_steps=200):
        """
        Один эпизод взаимодействия агента со средой.
        """
        state = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            # 1. Выбираем действие
            action = self.choose_action(state)

            # 2. Получаем реальный переход
            next_state, reward, done = env.step(action)
            total_reward += reward

            # 3. Обновляем Q по реальному опыту
            change = self.update_from_real_experience(state, action, reward, next_state)

            # 4. Обновляем модель среды
            self.update_model(state, action, reward, next_state)

            # 5. Помещаем текущую пару в приоритетную "очередь"
            self.push_priority(state, action, change)

            # 6. Выполняем несколько шагов планирования по модели
            for _ in range(self.planning_steps):
                self.planning_step()

            state = next_state

            if done:
                break

        return total_reward


# -----------------------------------------
# 3. ЗАПУСК ОБУЧЕНИЯ
# -----------------------------------------
if __name__ == "__main__":
    random.seed(42)

    env = GridWorld()
    agent = ModelBasedAgent(
        actions=env.actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.2,
        planning_steps=15
    )

    episodes = 120
    rewards = []

    for episode in range(episodes):
        ep_reward = agent.train_episode(env)
        rewards.append(ep_reward)

        # Печатаем прогресс каждые 10 эпизодов
        if (episode + 1) % 10 == 0:
            avg_last_10 = sum(rewards[-10:]) / 10
            print(f"Эпизод {episode + 1:3d} | средняя награда за последние 10 = {avg_last_10:6.2f}")

    # Покажем найденную стратегию в каждой клетке
    print("\nПриближенная стратегия:")
    for y in range(env.height):
        row = []
        for x in range(env.width):
            state = (x, y)

            if state in env.walls:
                row.append("####")
            elif state == env.goal:
                row.append("GOAL")
            else:
                best_a = max(env.actions, key=lambda a: agent.get_q(state, a))
                row.append(best_a[:4])
        print(" | ".join(row))
"""
Приближенная стратегия:
down | right | down | down | left
down | ####  | down | left | left
down | ####  | down | #### | down
right| down  | right| down | down
right| right | right| right| GOAL

Таблица приближенной стратегии показывает предпочтительное действие агента в каждом состоянии среды 
после обучения. Видно, что агент формирует маршрут в направлении целевого состояния GOAL, 
учитывая расположение препятствий ####. 
В большинстве верхних и левых клеток стратегия направляет агента вниз, а в нижней части поля — вправо, 
что соответствует движению к цели. Наличие локально неочевидных действий объясняется приближенным 
характером оценки модели и ценности состояний, а также конечным числом эпизодов обучения.
"""        