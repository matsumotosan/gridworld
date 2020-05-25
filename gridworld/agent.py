import numpy as np


class Agent:
    def __init__(self, start_pos=(0, 0), actions=None, policy=None):
        self.start_pos = start_pos
        self.pos = start_pos
        self.history = [self.pos]
        self.points = 0.0

        if actions is None:
            # default movement
            self.actions = dict(up=(0, 1), down=(0, -1), left=(-1, 0), right=(1, 0))
        else:
            self.actions = actions

        if policy is None:
            # uniform random policy
            self.policy = np.ones(len(self.actions), dtype=float) / len(self.actions)
        else:
            self.policy = policy

    def step(self, gridworld):
        move = self.actions[np.random.choice(list(self.actions.keys()),
                                             p=self.policy)]

        if gridworld.is_valid_action(self.pos, move):
            self.pos = (self.pos[0] + move[0], self.pos[1] + move[1])
        self.history.append(self.pos)
        self.points += gridworld.reward

    def reached_end(self, term_pos):
        return self.pos in term_pos

    def reset(self):
        self.pos = self.start_pos
        self.policy = dict(up=0.25, down=0.25, left=0.25, right=0.25)
