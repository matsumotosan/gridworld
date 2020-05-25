import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    def __init__(self, height=4, width=4, term_pos=((0, 0), (3, 3)), blocked_pos=None,
                 term_reward=1, reward=-1, value=None):

        self.height = height
        self.width = width
        self.term_pos = term_pos
        self.blocked_pos = blocked_pos
        self.term_reward = term_reward
        self.reward = reward

        if value is None:
            self.value = np.zeros((height, width))
        else:
            self.value = value

        self.value[self.blocked_pos] = 0.0
        self.value[[[x[0], x[1]] for x in self.term_pos]] = term_reward

    def show(self, policy=None):
        # Calculate grid centers
        x_ctr = np.arange(0.5, self.width + 0.5)
        y_ctr = np.arange(0.5, self.height + 0.5)

        # Plot colored grid for values at each cell
        plt.pcolormesh(self.value, color='w', linewidth=2)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for i, x in enumerate(x_ctr):
            for j, y in enumerate(y_ctr):
                # Annotate value
                ax.text(x, y, "(%d,%d)\nv=%.1f" % (i, j, self.value[i, j]),
                        ha='center', va='center', color='w')
                # Plot policy arrow

        ax.set_title('GridWorld Value Function')
        ax.set_aspect('equal')
        plt.colorbar()
        plt.show()

    def is_valid_action(self, pos, action):
        new_pos = [sum(x) for x in zip(pos, action)]
        return 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height
