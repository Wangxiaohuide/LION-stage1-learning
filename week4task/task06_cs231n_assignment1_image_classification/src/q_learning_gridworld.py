import numpy as np


ACTION_NAMES = ["up", "right", "down", "left"]
ACTIONS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])


def step(state, action, grid_shape=(4, 4), goal=(3, 3), wall=(1, 1)):
    if state == goal:
        return state, 0.0, True

    candidate = tuple(np.array(state) + ACTIONS[action])
    row_ok = 0 <= candidate[0] < grid_shape[0]
    col_ok = 0 <= candidate[1] < grid_shape[1]
    if not row_ok or not col_ok or candidate == wall:
        candidate = state

    reward = 1.0 if candidate == goal else -0.04
    return candidate, reward, candidate == goal


def train_q_learning(episodes=600, alpha=0.25, gamma=0.95, epsilon=0.18, seed=3):
    rng = np.random.default_rng(seed)
    grid_shape = (4, 4)
    goal = (3, 3)
    wall = (1, 1)
    Q = np.zeros(grid_shape + (len(ACTION_NAMES),))

    for _ in range(episodes):
        state = (0, 0)
        for _ in range(60):
            if rng.random() < epsilon:
                action = rng.integers(len(ACTION_NAMES))
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done = step(state, action, grid_shape, goal, wall)
            target = reward if done else reward + gamma * np.max(Q[next_state])
            Q[state + (action,)] += alpha * (target - Q[state + (action,)])
            state = next_state
            if done:
                break

    return Q


def render_policy(Q, goal=(3, 3), wall=(1, 1)):
    arrows = {"up": "^", "right": ">", "down": "v", "left": "<"}
    rows = []
    for r in range(Q.shape[0]):
        row = []
        for c in range(Q.shape[1]):
            if (r, c) == goal:
                row.append("G")
            elif (r, c) == wall:
                row.append("#")
            else:
                row.append(arrows[ACTION_NAMES[int(np.argmax(Q[r, c]))]])
        rows.append(" ".join(row))
    return "\n".join(rows)


def greedy_rollout(Q):
    state = (0, 0)
    path = [state]
    total_reward = 0.0
    for _ in range(20):
        action = int(np.argmax(Q[state]))
        state, reward, done = step(state, action)
        total_reward += reward
        path.append(state)
        if done:
            break
    return path, total_reward


def main():
    Q = train_q_learning()
    path, total_reward = greedy_rollout(Q)
    print("CS229 late lectures mini demo: Q-learning")
    print("Learned greedy policy:")
    print(render_policy(Q))
    print(f"Greedy path: {path}")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
