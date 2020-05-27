import numpy as np
from scipy.ndimage import convolve
from tqdm import trange


def policy_iteration(gridworld, max_iter=20):

    # Initialize value (pad each side with zeros)
    v = np.zeros((gridworld.height + 2, gridworld.width + 2))

    # Act greedy wrt to each iteration of v
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    p0 = np.zeros((gridworld.height, gridworld.width))

    for itr in trange(max_iter):
        v = convolve(v, kernel, mode='constant', cval=0.0)
        for pos in zip(gridworld.term_pos, gridworld.blocked_pos):
            v[pos[0] + 1, pos[1] + 1] = 0.0

        p = greedy_policy(v[1:-1, 1:-1])
        if converged(p, p0):
            print(F"Policy converged after {itr} iterations.")
            return v[1:-1, 1:-1], p
        p0 = p

    print(F"Policy did not converge.")
    return p


def greedy_policy(v):
    return 0


def converged(p, p0):
    return False
