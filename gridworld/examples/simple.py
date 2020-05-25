from gridworld.gridworld import GridWorld
from gridworld.agent import Agent
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

HEIGHT = 10
WIDTH = 10
TERM_POS = [(0, 9), (9, 0), (9, 9)]
ACTIONS = {'pawn':   {'up': (0, 1),   'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)},
           'bishop': {'ur': (1, 1),   'dr': (1, -1),   'dl': (-1, -1),  'ul': (-1, 1)},
           'knight': {'uur': (1, 2),  'urr': (2, 1),   'drr': (2, -1),  'ddr': (1, -2),
                      'uul': (-1, 2), 'ull': (-2, 1),  'dll': (-2, -1), 'ddl': (-1, -2)}}
AGENT_TYPE = 'knight'


def main():
    # Initialize simple GridWorld
    world = GridWorld(height=HEIGHT, width=WIDTH, term_pos=TERM_POS)

    # Define agent
    agent = Agent(start_pos=(0, 0), actions=ACTIONS[AGENT_TYPE])

    # Run agent through gridworld with uniform random policy (n=10)
    while not agent.reached_end(world.term_pos):
        agent.step(world)

    print(F"Agent reached terminal state after {len(agent.history) - 1} steps.")

    # Plot heatmap of visited states
    counter = collections.Counter(agent.history)
    freq = np.zeros((HEIGHT, WIDTH), dtype=int)
    for x, f in counter.items():
        freq[x[0], x[1]] = f

    f, ax = plt.subplots()
    sns.heatmap(freq, annot=True, fmt="d", linewidths=1, ax=ax)
    ax.invert_yaxis()
    ax.set_title(F'Visited states for {AGENT_TYPE}')
    plt.show()

    # Iterative policy evaluation to find optimal policy
    # world.show()


def update_pos(idx, history):
    return history[idx]


if __name__ == "__main__":
    main()
