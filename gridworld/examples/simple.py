from gridworld.gridworld import GridWorld
from gridworld.agent import Agent
from gridworld.game import Game
from tqdm import trange
import numpy as np
import collections
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define game specs
HEIGHT = 10
WIDTH = 10
TERM_POS = [(9, 9)]
ACTIONS = {'pawn':   {'up': (0, 1),   'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)},
           'bishop': {'ur': (1, 1),   'dr': (1, -1),   'dl': (-1, -1),  'ul': (-1, 1)},
           'knight': {'uur': (1, 2),  'urr': (2, 1),   'drr': (2, -1),  'ddr': (1, -2),
                      'uul': (-1, 2), 'ull': (-2, 1),  'dll': (-2, -1), 'ddl': (-1, -2)}}
AGENTS = ['pawn', 'bishop', 'knight']
N = 1000


def main():
    # Define a simple gridworld
    world = GridWorld(height=HEIGHT, width=WIDTH, term_pos=TERM_POS)

    # Define an agent
    agent_idx = 0
    agent = Agent(start_pos=(0, 0), actions=ACTIONS[AGENTS[agent_idx]])

    # Create a game
    game = Game(world, agent)
    # game.run(n=100)

    # Run agent through gridworld with uniform random policy (n=10)
    freq = np.zeros((HEIGHT, WIDTH))
    nsteps = np.zeros(N)

    for i in trange(N):
        # Let agent walk around
        while not agent.reached_end(world.term_pos):
            agent.step(world)

        # Track number of times states visited
        nsteps[i] = len(agent.history)
        counter = collections.Counter(agent.history)
        for x, f in counter.items():
            freq[x[0], x[1]] += f

        # Reset agent position
        agent.reset()

    # Average number of times visited
    freq /= N

    # Plot heatmap of state visitation rate
    f, ax = plt.subplots()
    sns.heatmap(freq, annot=True, fmt=".1f", linewidths=1, square=True, ax=ax)

    # Highlight starting grid
    ax.axvline(x=agent.start_pos[0], ymin=0, ymax=1.0 / world.height, color='g', linewidth=2)
    ax.axvline(x=agent.start_pos[0] + 1, ymin=0, ymax=1.0 / world.height, color='g', linewidth=2)
    ax.axhline(y=agent.start_pos[1], xmin=0, xmax=1.0 / world.width, color='g', linewidth=2)
    ax.axhline(y=agent.start_pos[1] + 1, xmin=0, xmax=1.0 / world.width, color='g', linewidth=2)

    # Highlight terminal grid(s)
    for tp in world.term_pos:
        ax.axvline(x=tp[0], ymin=0, ymax=1.0 / world.height, color='r', linewidth=2)
        ax.axvline(x=tp[0] + 1, ymin=0, ymax=1.0 / world.height, color='r', linewidth=2)
        ax.axhline(y=tp[1], xmin=0, xmax=1.0 / world.width, color='r', linewidth=2)
        ax.axhline(y=tp[1] + 1, xmin=0, xmax=1.0 / world.width, color='r', linewidth=2)

    ax.invert_yaxis()
    ax.set_title(F'Average number of visits ({AGENTS[agent_idx]}) (n={N})')
    plt.show()

    # Histogram of steps taken
    n_bins = 20
    plt.hist(nsteps, bins=n_bins)
    plt.title(F"Number of steps taken (n={N})")
    plt.show()

    # Iterative policy evaluation to find optimal policy
    # world.show()


def update_pos(idx, history):
    return history[idx]


if __name__ == "__main__":
    main()
