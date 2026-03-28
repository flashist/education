"""Учебный пример иерархического обучения с подкреплением в духе MAXQ.

В примере агент должен:
1. добраться до ключа,
2. получить ключ,
3. добраться до двери,
4. открыть дверь.

Сценарий сделан максимально простым, чтобы было видно разделение на:
- корневую задачу;
- составные подзадачи;
- навигационный навык;
- примитивные действия среды.
"""

class KeyDoorGrid:
    def __init__(self, size=5):
        self.size = size
        self.key_pos = (0, 4)
        self.door_pos = (4, 4)
        self.start_pos = (4, 0)
        self.reset()

    def reset(self):
        # Агент стартует без ключа и с закрытой дверью.
        self.agent = self.start_pos
        self.has_key = False
        self.done = False
        return self.state()

    def state(self):
        # Для простоты состояние описывается координатами и флагом наличия ключа.
        return (self.agent[0], self.agent[1], int(self.has_key))

    def move(self, action):
        # Примитивный уровень управления.
        r, c = self.agent
        if action == 'up':
            r = max(0, r - 1)
        elif action == 'down':
            r = min(self.size - 1, r + 1)
        elif action == 'left':
            c = max(0, c - 1)
        elif action == 'right':
            c = min(self.size - 1, c + 1)
        self.agent = (r, c)

        # Небольшой штраф за каждый шаг мотивирует решать задачу быстрее.
        reward = -1

        # Если дошли до ключа, отмечаем получение ресурса.
        if self.agent == self.key_pos:
            self.has_key = True

        # Если есть ключ и достигнута дверь, эпизод успешно завершается.
        if self.agent == self.door_pos and self.has_key:
            reward = 50
            self.done = True

        return self.state(), reward, self.done


def navigate(env, target):
    """Переиспользуемый навык навигации.

    Это и есть пример суб-компоненты: ее можно вызывать из разных задач.
    В настоящем HRL здесь могла бы быть отдельная обучаемая политика,
    но для учебной цели используем простой жадный маршрут.
    """
    total_reward = 0
    path = [env.agent]

    while env.agent != target and not env.done:
        ar, ac = env.agent
        tr, tc = target

        # Жадно двигаемся по Манхэттенскому расстоянию.
        if ar < tr:
            action = 'down'
        elif ar > tr:
            action = 'up'
        elif ac < tc:
            action = 'right'
        else:
            action = 'left'

        _, reward, _ = env.move(action)
        total_reward += reward
        path.append(env.agent)

    return total_reward, path


def task_get_key(env):
    # Составная подзадача 1: добраться до ключа.
    return navigate(env, env.key_pos)


def task_open_door(env):
    # Составная подзадача 2: добраться до двери.
    return navigate(env, env.door_pos)


def solve_episode(env):
    """Корневая задача.

    Если ключа нет, сначала выполняем подзадачу получения ключа.
    Затем выполняем подзадачу открытия двери.
    """
    env.reset()
    reward_sum = 0
    full_path = [env.agent]

    if not env.has_key:
        reward, path = task_get_key(env)
        reward_sum += reward
        full_path.extend(path[1:])

    reward, path = task_open_door(env)
    reward_sum += reward
    full_path.extend(path[1:])

    return reward_sum, env.done, full_path


if __name__ == '__main__':
    env = KeyDoorGrid(size=5)
    total_reward, success, path = solve_episode(env)

    print('=== Учебный пример MAXQ-подобной декомпозиции ===')
    print('Успешное завершение:', success)
    print('Суммарная награда:', total_reward)
    print('Траектория агента:')
    for step, pos in enumerate(path):
        print(f'  Шаг {step:2d}: {pos}')