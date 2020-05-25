import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Game:
    def __init__(self, world, agent):
        self.world = world
        self.agent = agent
        self.freq = np.zeros((self.world.height, self.world.width))

    def run(self, n=100):
        for i in range(n):
            # Let agent walk around
            while not self.agent.reached_end(self.world.term_pos):
                self.agent.step(self.world)

            print(F"Agent reached terminal state after {len(self.agent.history) - 1} steps.")

            # Track number of times states visited
            counter = collections.Counter(self.agent.history)
            for x, f in counter.items():
                self.freq[x[0], x[1]] += f

            # Reset agent position
            self.agent.reset()

    def reset(self):
        self.freq = np.zeros((self.world.height, self.world.width))
