# ============================================================
# Задание 9: Реализация ограничения ALISP
#
# Запрет на выполнение open_door до получения ключа.
#
# Среда: KeyDoorGrid 5×5
#   - агент стартует в (4, 0) — правый нижний угол
#   - ключ в (0, 0)           — левый верхний угол (рядом со стартом)
#   - дверь в (4, 4)          — правый нижний угол
#
# Сравниваются два подхода:
#   1. Наивный — агент сначала идёт к двери (без ключа), потом за ключом,
#      потом снова к двери. Лишний крюк через всё поле.
#   2. ALISP — ограничение запрещает идти к двери до получения ключа.
#      Агент сразу берёт ключ и идёт к двери — оптимальный маршрут.
# ============================================================


class KeyDoorGrid:
    def __init__(self, size=5):
        self.size = size
        self.key_pos = (0, 0)    # левый верхний угол
        self.door_pos = (4, 4)   # правый нижний угол
        self.start_pos = (4, 0)  # нижний левый угол
        self.reset()

    def reset(self):
        self.agent = self.start_pos
        self.has_key = False
        self.done = False
        return self.state()

    def state(self):
        return (self.agent[0], self.agent[1], int(self.has_key))

    def move(self, action):
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
        reward = -1

        if self.agent == self.key_pos:
            self.has_key = True

        if self.agent == self.door_pos and self.has_key:
            reward = 50
            self.done = True

        return self.state(), reward, self.done


def navigate(env, target):
    """Переиспользуемый навык навигации — жадный маршрут по Манхэттену."""
    total_reward = 0
    path = [env.agent]

    while env.agent != target and not env.done:
        ar, ac = env.agent
        tr, tc = target

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


# ------------------------------------------------------------------
# Подход 1: Наивный (без ограничения ALISP)
#
# Агент сначала идёт к двери, не имея ключа — дверь не открывается.
# Затем идёт за ключом и снова возвращается к двери.
# Это приводит к лишним шагам и меньшей суммарной награде.
# ------------------------------------------------------------------

def solve_naive(env):
    env.reset()
    reward_sum = 0
    full_path = [env.agent]

    # Шаг 1: идём к двери без ключа (нарушение порядка)
    r, path = navigate(env, env.door_pos)
    reward_sum += r
    full_path.extend(path[1:])

    # Шаг 2: дверь не открылась — идём за ключом
    if not env.has_key:
        r, path = navigate(env, env.key_pos)
        reward_sum += r
        full_path.extend(path[1:])

    # Шаг 3: возвращаемся к двери
    if not env.done:
        r, path = navigate(env, env.door_pos)
        reward_sum += r
        full_path.extend(path[1:])

    return reward_sum, env.done, full_path


# ------------------------------------------------------------------
# Подход 2: ALISP-ограничение
#
# SafeKeyDoorPolicy — программный каркас поведения:
# пока ключа нет, действие open_door недопустимо.
# Агент обязан сначала получить ключ.
# ------------------------------------------------------------------

class SafeKeyDoorPolicy:
    """ALISP-каркас: запрещает переход к двери без ключа."""

    def choose_action(self, state):
        _, _, has_key = state
        # Ограничение ALISP: open_door запрещён до получения ключа.
        if not has_key:
            return 'go_to_key'
        return 'go_to_door'


def solve_alisp(env, policy):
    state = env.reset()
    reward_sum = 0
    full_path = [env.agent]

    while not env.done:
        action = policy.choose_action(state)
        target = env.key_pos if action == 'go_to_key' else env.door_pos
        r, path = navigate(env, target)
        reward_sum += r
        full_path.extend(path[1:])
        state = env.state()

    return reward_sum, env.done, full_path


# ------------------------------------------------------------------
# Запуск и сравнение
# ------------------------------------------------------------------

if __name__ == '__main__':
    env = KeyDoorGrid(size=5)

    # Наивный подход
    reward_naive, success_naive, path_naive = solve_naive(env)
    print('=== Наивный подход (без ограничения ALISP) ===')
    print(f'Успех: {success_naive}')
    print(f'Число шагов: {len(path_naive) - 1}')
    print(f'Суммарная награда: {reward_naive}')
    print('Траектория:')
    for i, pos in enumerate(path_naive):
        marker = ''
        if pos == env.key_pos:
            marker = ' <- ключ'
        elif pos == env.door_pos:
            marker = ' <- дверь'
        elif pos == env.start_pos:
            marker = ' <- старт'
        print(f'  Шаг {i:2d}: {pos}{marker}')

    print()

    # ALISP-подход
    policy = SafeKeyDoorPolicy()
    reward_alisp, success_alisp, path_alisp = solve_alisp(env, policy)
    print('=== ALISP-подход (запрет open_door без ключа) ===')
    print(f'Успех: {success_alisp}')
    print(f'Число шагов: {len(path_alisp) - 1}')
    print(f'Суммарная награда: {reward_alisp}')
    print('Траектория:')
    for i, pos in enumerate(path_alisp):
        marker = ''
        if pos == env.key_pos:
            marker = ' <- ключ'
        elif pos == env.door_pos:
            marker = ' <- дверь'
        elif pos == env.start_pos:
            marker = ' <- старт'
        print(f'  Шаг {i:2d}: {pos}{marker}')

    print()
    print('=== Итог сравнения ===')
    print(f'  Наивный:  шагов={len(path_naive)-1:2d}, награда={reward_naive}')
    print(f'  ALISP:    шагов={len(path_alisp)-1:2d}, награда={reward_alisp}')
    saved = (len(path_naive) - 1) - (len(path_alisp) - 1)
    print(f'  ALISP сэкономил {saved} лишних шагов')
