import random
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ── Константы действий ─────────────────────────────────────────────────────
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
ACTION_NAMES   = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
ACTION_VECTORS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class GridWorldTeleport:
    """
    Эпизодическая среда — сетка 3×3 с телепортом.

    Карта:
        S  .  T
        .  .  .
        .  .  G

    Состояния: (row, col), от (0,0) до (2,2).
    Стартовое состояние : (0, 0) — S
    Телепорт             : (0, 2) — T
    Цель (терминал)      : (2, 2) — G, награда +10
    Обычный шаг          : -1

    teleport_mode:
        "fixed"  — всегда переносит в TELEPORT_DEST = (2, 0)
        "random" — переносит в случайную клетку (не цель, не сам телепорт)
    """

    START         = (0, 0)
    TELEPORT      = (0, 2)
    TELEPORT_DEST = (2, 0)   # назначение фиксированного телепорта
    GOAL          = (2, 2)

    ROWS = 3
    COLS = 3

    def __init__(self, teleport_mode="fixed"):
        assert teleport_mode in ("fixed", "random"), \
            "teleport_mode должен быть 'fixed' или 'random'"
        self.teleport_mode = teleport_mode

        # Все клетки, куда может перенести случайный телепорт
        self._random_dests = [
            (r, c)
            for r in range(self.ROWS)
            for c in range(self.COLS)
            if (r, c) not in (self.TELEPORT, self.GOAL)
        ]
        self.state = self.START

    def reset(self):
        """Сброс среды в начальное состояние."""
        self.state = self.START
        return self.state

    def step(self, action):
        """
        Выполнить действие и вернуть: next_state, reward, done.

        Действия: UP=0, DOWN=1, LEFT=2, RIGHT=3.
        При выходе за границы агент остаётся на месте.
        При попадании на T агент переносится согласно teleport_mode.
        """
        if self.state == self.GOAL:
            raise ValueError("Эпизод уже завершён. Сначала вызовите reset().")

        row, col = self.state

        if action == UP:
            new_row, new_col = row - 1, col
        elif action == DOWN:
            new_row, new_col = row + 1, col
        elif action == LEFT:
            new_row, new_col = row, col - 1
        elif action == RIGHT:
            new_row, new_col = row, col + 1
        else:
            raise ValueError(f"Неизвестное действие {action}. Допустимы 0..3.")

        # Проверка границ — остаёмся на месте если за полем
        if not (0 <= new_row < self.ROWS and 0 <= new_col < self.COLS):
            new_row, new_col = row, col

        next_state = (new_row, new_col)

        # Логика телепорта
        if next_state == self.TELEPORT:
            if self.teleport_mode == "fixed":
                next_state = self.TELEPORT_DEST
            else:
                next_state = random.choice(self._random_dests)

        self.state = next_state

        if next_state == self.GOAL:
            return next_state, 10, True
        else:
            return next_state, -1, False


class MonteCarloAgent:
    """
    Агент, обучающийся методом Monte Carlo по полным эпизодам.

    Хранит:
    - Q(s, a): оценку ценности действия a в состоянии s
    - epsilon : вероятность случайного действия (ε-жадная стратегия)
    - alpha   : скорость обучения
    - gamma   : коэффициент дисконтирования
    """

    def __init__(self, actions=(UP, DOWN, LEFT, RIGHT),
                 alpha=0.1, gamma=1.0, epsilon=0.2):
        self.actions = actions
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def choose_action(self, state):
        """ε-жадный выбор действия."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = self.Q[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def generate_episode(self, env, max_steps=200):
        """Сгенерировать один полный эпизод. Возвращает [(state, action, reward), ...]"""
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

            Q(s, a) ← Q(s, a) + alpha × (G_t − Q(s, a))
        """
        G = 0.0
        returns = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            _, _, reward = episode[t]
            G = reward + self.gamma * G
            returns[t] = G
        for t, (state, action, _) in enumerate(episode):
            self.Q[state][action] += self.alpha * (returns[t] - self.Q[state][action])

    def greedy_policy(self):
        """Вернуть текущую жадную стратегию: {state: best_action}."""
        return {s: max(q, key=q.get) for s, q in self.Q.items()}


# ── Текстовый вывод ────────────────────────────────────────────────────────

def print_q_table(agent, label=""):
    print(f"Таблица Q(s, a) [{label}]:")
    env = GridWorldTeleport
    for row in range(env.ROWS):
        for col in range(env.COLS):
            s = (row, col)
            if s == env.GOAL:
                continue
            q = agent.Q[s]
            tag = " T" if s == env.TELEPORT else "  "
            print(f"  {tag}({row},{col})  "
                  f"↑={q[UP]:6.2f}  ↓={q[DOWN]:6.2f}  "
                  f"←={q[LEFT]:6.2f}  →={q[RIGHT]:6.2f}")
    print()


def print_policy_grid(agent, label=""):
    policy = agent.greedy_policy()
    env = GridWorldTeleport
    print(f"Стратегия агента [{label}]:")
    for row in range(env.ROWS):
        row_str = "  "
        for col in range(env.COLS):
            s = (row, col)
            if s == env.GOAL:
                row_str += " G "
            elif s == env.TELEPORT:
                row_str += " T "
            elif s == env.START:
                row_str += f"[{ACTION_NAMES.get(policy.get(s), '?')}]"
            elif s in policy:
                row_str += f" {ACTION_NAMES[policy[s]]} "
            else:
                row_str += " ? "
        print(row_str)
    print()


# ── Подсчёт статистики ─────────────────────────────────────────────────────

def count_successes(agent, env, n_test=200):
    """Запустить n_test жадных эпизодов, вернуть (доля успехов, ср. длина пути)."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    successes, total_steps = 0, 0
    for _ in range(n_test):
        episode = agent.generate_episode(env, max_steps=200)
        _, _, last_reward = episode[-1]
        if last_reward == 10:
            successes += 1
            total_steps += len(episode)
    agent.epsilon = saved_epsilon
    avg_steps = total_steps / successes if successes > 0 else float("inf")
    return successes / n_test, avg_steps


# ── Графики ────────────────────────────────────────────────────────────────

def _draw_policy_ax(ax, agent, title):
    """Рисует стратегию агента на переданном Axes."""
    env = GridWorldTeleport
    policy = agent.greedy_policy()

    ax.set_xlim(0, env.COLS)
    ax.set_ylim(0, env.ROWS)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    for row in range(env.ROWS):
        for col in range(env.COLS):
            s = (row, col)
            cx, cy = col + 0.5, row + 0.5

            if s == env.GOAL:
                color = "#90EE90"
            elif s == env.TELEPORT:
                color = "#ADD8E6"
            elif s == env.START:
                color = "#FFD700"
            else:
                best_q = max(agent.Q[s].values()) if s in agent.Q else 0.0
                norm = min(max((best_q + 3) / 13, 0), 1)
                color = plt.cm.YlOrRd(norm)

            ax.add_patch(plt.Rectangle(
                (col, row), 1, 1,
                facecolor=color, edgecolor="black", linewidth=1.5
            ))

            if s == env.GOAL:
                ax.text(cx, cy, "G\n+10", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            elif s == env.TELEPORT:
                ax.text(cx, cy, "T", ha="center", va="center", fontsize=10)
            elif s == env.START:
                act = policy.get(s)
                ax.text(cx, cy, f"S\n{ACTION_NAMES.get(act, '?')}",
                        ha="center", va="center", fontsize=10, fontweight="bold")
            else:
                act = policy.get(s)
                if act is not None:
                    dr, dc = ACTION_VECTORS[act]
                    ax.annotate(
                        "", xy=(cx + dc * 0.35, cy + dr * 0.35),
                        xytext=(cx - dc * 0.2, cy - dr * 0.2),
                        arrowprops=dict(arrowstyle="->", color="black", lw=2)
                    )

    legend_items = [
        mpatches.Patch(color="#FFD700", label="S — старт"),
        mpatches.Patch(color="#ADD8E6", label="T — телепорт"),
        mpatches.Patch(color="#90EE90", label="G — цель"),
    ]
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)


def _draw_block(sf, title, returns, lengths, agent, color):
    """
    Рисует один блок (subfigure) с заголовком.

    Layout (4 строки × 4 колонки):
      width_ratios  = [1, 1, 1, 1]   — col 1 (стратегия) вдвое шире Q-карт
      height_ratios = [1, 1, 1, 1]   — 4 равные строки

      col 0 (узкая):   строки 0,1,2 — три графика; строка 3 — пустая
      col 1 (широкая): строки 0:4   — стратегия (весь блок)
      cols 2-3:         строки 0:2  — Q↑, Q↓
                        строки 2:4  — Q←, Q→
    """
    env = GridWorldTeleport
    window = 50
    win_success = 100
    action_titles = {UP: "↑ UP", DOWN: "↓ DOWN",
                     LEFT: "← LEFT", RIGHT: "→ RIGHT"}

    sf.suptitle(title, fontsize=11, fontweight="bold", y=0.96)

    gs = sf.add_gridspec(
        4, 4,
        width_ratios=[1, 1, 1, 1],
        height_ratios=[1, 1, 1.99, 0.01],
        top=0.88, bottom=0.11,
        hspace=0.6, wspace=0.35
    )

    # ── Награда (col 0, row 0) ─────────────────────────────────────────────
    ax_r = sf.add_subplot(gs[0, 0])
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(returns) + 1)
    ax_r.plot(x, smooth, color=color)
    ax_r.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_r.set_title(f"Награда (окно {window})", fontsize=8)
    ax_r.set_ylabel("Награда", fontsize=7)
    ax_r.tick_params(labelbottom=False, labelsize=7)
    ax_r.grid(alpha=0.3)

    # ── Длина эпизода (col 0, row 1) ──────────────────────────────────────
    ax_l = sf.add_subplot(gs[1, 0])
    smooth_l = np.convolve(lengths, np.ones(window) / window, mode="valid")
    ax_l.plot(x, smooth_l, color=color)
    ax_l.set_title(f"Длина эпизода (окно {window})", fontsize=8)
    ax_l.set_ylabel("Шагов", fontsize=7)
    ax_l.tick_params(labelbottom=False, labelsize=7)
    ax_l.grid(alpha=0.3)

    # ── Успех по окнам (col 0, row 2) ─────────────────────────────────────
    ax_s = sf.add_subplot(gs[2, 0])
    successes = [1 if r > 0 else 0 for r in returns]
    rates = [
        np.mean(successes[j: j + win_success])
        for j in range(0, len(successes) - win_success + 1, win_success)
    ]
    xs = np.array([j * win_success + win_success // 2
                   for j in range(len(rates))])
    ax_s.bar(xs, rates, width=win_success * 0.8, color=color, alpha=0.8)
    ax_s.set_ylim(0, 1.1)
    ax_s.set_title(f"Успех при обучении, ε=0.2 ({win_success} эп.)", fontsize=8)
    ax_s.set_xlabel("Эпизод", fontsize=7)
    ax_s.set_ylabel("Доля успехов", fontsize=7)
    ax_s.tick_params(labelsize=7)
    ax_s.grid(axis="y", alpha=0.3)

    # ── Строка 3, col 0 — пустая ──────────────────────────────────────────
    sf.add_subplot(gs[3, 0]).set_visible(False)

    # ── Стратегия (col 1, все строки) ─────────────────────────────────────
    ax_p = sf.add_subplot(gs[0:4, 1])
    _draw_policy_ax(ax_p, agent, "Стратегия")

    # ── Q-карты (cols 2-3, по 2 строки каждая) ────────────────────────────
    q_positions = [(0, 2, UP), (0, 3, DOWN), (2, 2, LEFT), (2, 3, RIGHT)]
    for row, col, action in q_positions:
        ax = sf.add_subplot(gs[row:row + 2, col])
        grid = np.zeros((env.ROWS, env.COLS))
        for r in range(env.ROWS):
            for c in range(env.COLS):
                s = (r, c)
                if s != env.GOAL and s in agent.Q:
                    grid[r, c] = agent.Q[s][action]

        im = ax.imshow(grid, cmap="RdYlGn", vmin=-3, vmax=10)
        ax.set_title(action_titles[action], fontsize=8)
        ax.set_yticks(range(env.ROWS))
        ax.tick_params(labelsize=7)
        # У верхних карт убираем подписи оси X чтобы не перекрывать заголовки нижних
        if row == 0:
            ax.set_xticks(range(env.COLS))
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticks(range(env.COLS))

        for r in range(env.ROWS):
            for c in range(env.COLS):
                s = (r, c)
                lbl = "G" if s == env.GOAL else "T" if s == env.TELEPORT else ""
                val = "" if s == env.GOAL else f"{grid[r, c]:.1f}"
                ax.text(c, r, f"{lbl}\n{val}" if lbl else val,
                        ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_comparison(fixed_returns, fixed_lengths, fixed_agent, fixed_stats,
                    random_returns, random_lengths, random_agent, random_stats):
    """
    Два блока (subfigures) — сверху фиксированный телепорт, снизу случайный.
    Каждый блок: 3 графика | стратегия | 2×2 Q-карты.
    """
    FIXED_COLOR  = "steelblue"
    RANDOM_COLOR = "tomato"

    fig = plt.figure(figsize=(14, 14))
    outer_gs = fig.add_gridspec(2, 1, hspace=0.18, top=0.99, bottom=0.01)
    sf_fixed  = fig.add_subfigure(outer_gs[0])
    sf_random = fig.add_subfigure(outer_gs[1])

    _draw_block(sf_fixed,  "Фиксированный телепорт",
                fixed_returns,  fixed_lengths,  fixed_agent,  FIXED_COLOR)
    _draw_block(sf_random, "Случайный телепорт",
                random_returns, random_lengths, random_agent, RANDOM_COLOR)


# ── Главный цикл обучения ─────────────────────────────────────────────────

def train(env, episodes_count, label):
    """Обучить агента на среде env, вернуть (агент, returns, lengths)."""
    agent = MonteCarloAgent(
        actions=(UP, DOWN, LEFT, RIGHT),
        alpha=0.1,
        gamma=1.0,
        epsilon=0.2
    )
    episode_returns = []
    episode_lengths = []

    print(f"\n{'─'*50}")
    print(f"  Обучение: {label}  ({episodes_count} эпизодов)")
    print(f"{'─'*50}")

    for ep in range(1, episodes_count + 1):
        episode = agent.generate_episode(env)
        agent.update_from_episode_every_visit(episode)
        episode_returns.append(sum(r for _, _, r in episode))
        episode_lengths.append(len(episode))

        if ep in (100, 500, 1000, episodes_count):
            success_rate, avg_steps = count_successes(agent, env)
            print(f"  После {ep:>4} эп. │ Успех (ε=0): {success_rate*100:.0f}%"
                  f"  │ Ср. длина: {avg_steps:.1f} шагов")

    return agent, episode_returns, episode_lengths


def main():
    random.seed(42)
    episodes_count = 2000

    print("=" * 50)
    print("  Вариант 9. GridWorld 3×3 с телепортом")
    print("  Карта:  S . T  /  . . .  /  . . G")
    print("=" * 50)

    # ── Фиксированный телепорт ─────────────────────────────────────────────
    env_fixed  = GridWorldTeleport(teleport_mode="fixed")
    fixed_agent, fixed_returns, fixed_lengths = train(
        env_fixed, episodes_count, f"фиксированный телепорт → {GridWorldTeleport.TELEPORT_DEST}"
    )
    fixed_stats = count_successes(fixed_agent, env_fixed)

    print("\n  Финальная стратегия:")
    print_q_table(fixed_agent, "фиксированный")
    print_policy_grid(fixed_agent, "фиксированный")

    q_01 = fixed_agent.Q[(0, 1)]
    print("Анализ телепорта [фиксированный]:")
    print(f"  Q((0,1), →)={q_01[RIGHT]:.2f}  Q((0,1), ↓)={q_01[DOWN]:.2f}")
    if q_01[RIGHT] > q_01[DOWN]:
        print("  Вывод: агент предпочитает использовать телепорт — он сокращает путь к цели.")
    else:
        print("  Вывод: агент предпочитает обходить телепорт — он уводит дальше от цели.")
    print()

    # ── Случайный телепорт ─────────────────────────────────────────────────
    random.seed(42)
    env_random = GridWorldTeleport(teleport_mode="random")
    random_agent, random_returns, random_lengths = train(
        env_random, episodes_count, "случайный телепорт"
    )
    random_stats = count_successes(random_agent, env_random)

    print("\n  Финальная стратегия:")
    print_q_table(random_agent, "случайный")
    print_policy_grid(random_agent, "случайный")

    q_01 = random_agent.Q[(0, 1)]
    print("Анализ телепорта [случайный]:")
    print(f"  Q((0,1), →)={q_01[RIGHT]:.2f}  Q((0,1), ↓)={q_01[DOWN]:.2f}")
    if q_01[RIGHT] > q_01[DOWN]:
        print("  Вывод: агент предпочитает использовать телепорт — он сокращает путь к цели.")
    else:
        print("  Вывод: агент предпочитает обходить телепорт — он уводит дальше от цели.")
    print()

    # ── Итоговое сравнение в консоли ──────────────────────────────────────
    print("=" * 50)
    print("  Сравнение режимов телепорта")
    print("=" * 50)
    print(f"  Фиксированный: успех (ε=0) {fixed_stats[0]*100:.0f}%,"
          f"  ср. путь {fixed_stats[1]:.1f} шагов")
    print(f"  Случайный    : успех (ε=0) {random_stats[0]*100:.0f}%,"
          f"  ср. путь {random_stats[1]:.1f} шагов")

    # ── Графики ────────────────────────────────────────────────────────────
    plot_comparison(
        fixed_returns,  fixed_lengths,  fixed_agent,  fixed_stats,
        random_returns, random_lengths, random_agent, random_stats,
    )
    plt.show()


if __name__ == "__main__":
    main()
