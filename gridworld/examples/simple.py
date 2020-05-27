from gridworld.examples import *

# Define game specs
HEIGHT = 10
WIDTH = 10
TERM_POS = [(0, 9), (9, 9), (9, 0)]
BLOCKED_POS = [(5, 5)]
ACTIONS = {'pawn':   {'up': (0, 1),   'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)},
           'bishop': {'ur': (1, 1),   'dr': (1, -1),   'dl': (-1, -1),  'ul': (-1, 1)},
           'queen':  {'up': (0, 1),   'ur': (1, 1),    'rt': (1, 0),    'dr': (1, -1),
                      'dn': (0, -1),  'dl': (-1, -1),  'lf': (-1, 0),   'ul': (-1, 1)},
           'knight': {'uur': (1, 2),  'urr': (2, 1),   'drr': (2, -1),  'ddr': (1, -2),
                      'uul': (-1, 2), 'ull': (-2, 1),  'dll': (-2, -1), 'ddl': (-1, -2)}}
AGENTS = ['pawn', 'bishop', 'queen', 'knight']
AGENT_IDX = 2
N = 100


def main():
    # Define a simple gridworld
    world = GridWorld(height=HEIGHT, width=WIDTH, term_pos=TERM_POS,
                      blocked_pos=BLOCKED_POS)

    # Define an agent
    agent = Agent(start_pos=(0, 0), actions=ACTIONS[AGENTS[AGENT_IDX]])

    # Create a game
    game = Game(world, agent)
    # game.run(n=100)

    # Run agent through gridworld with uniform random policy (n=10)
    freq = np.zeros((HEIGHT, WIDTH))
    n_steps = np.zeros(N)

    for i in trange(N):
        # Let agent walk around
        while not agent.reached_end(world.term_pos):
            agent.step(world)

        # Track number of times states visited
        n_steps[i] = len(agent.history)
        counter = collections.Counter(agent.history)
        for x, f in counter.items():
            freq[x[0], x[1]] += f

        # Reset agent position
        agent.reset()

    # Average number of times visited for each grid
    freq /= N

    # Mask for blocked sites
    mask = np.zeros((HEIGHT, WIDTH), dtype='bool')
    for obs in world.blocked_pos:
        mask[obs[0], obs[1]] = True

    # Plot heatmap of state visitation rate
    f, ax = plt.subplots()
    g = sns.heatmap(freq, mask=mask, annot=True, fmt=".1f", linewidths=1, square=True, ax=ax)
    g.set_facecolor('grey')

    # Highlight starting grid
    ax.axvline(x=agent.start_pos[0], ymin=0, ymax=1.0 / world.height, color='g', linewidth=2)
    ax.axvline(x=agent.start_pos[0] + 1, ymin=0, ymax=1.0 / world.height, color='g', linewidth=2)
    ax.axhline(y=agent.start_pos[1], xmin=0, xmax=1.0 / world.width, color='g', linewidth=2)
    ax.axhline(y=agent.start_pos[1] + 1, xmin=0, xmax=1.0 / world.width, color='g', linewidth=2)

    # Highlight terminal grid(s)
    for tp in world.term_pos:
        xmin = tp[0] / world.width
        xmax = (tp[0] + 1) / world.width
        ymin = tp[1] / world.height
        ymax = (tp[1] + 1) / world.height

        ax.axvline(x=tp[0], ymin=ymin, ymax=ymax, color='r', linewidth=2)
        ax.axvline(x=tp[0] + 1, ymin=ymin, ymax=ymax, color='r', linewidth=2)
        ax.axhline(y=tp[1], xmin=xmin, xmax=xmax, color='r', linewidth=2)
        ax.axhline(y=tp[1] + 1, xmin=xmin, xmax=xmax, color='r', linewidth=2)

    ax.invert_yaxis()
    ax.set_title(F'Average number of visits ({AGENTS[AGENT_IDX]}) (n={N})')
    plt.show()

    # Histogram of steps taken
    n_bins = 20
    plt.hist(n_steps, bins=n_bins)
    plt.title(F"Number of steps taken ({AGENTS[AGENT_IDX]}) (n={N})")
    plt.grid(True)
    plt.show()

    # Iterative policy evaluation to find optimal policy
    world.show()


def update_pos(idx, history):
    return history[idx]


if __name__ == "__main__":
    main()
