from typing import Any, Union

import numpy as np


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """

    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon=0.05, selector=None):
        self.epsilon = epsilon
        self.selector = selector if selector is not None else ArgmaxActionSelector()

    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        batch_size, n_actions = scores.shape
        actions = self.selector(scores)
        mask = np.random.random(size=batch_size) < self.epsilon
        rand_actions = np.random.choice(n_actions, np.sum(mask))
        actions[mask] = rand_actions
        return actions


class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """

    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            action = np.random.choice(len(prob), p=prob)
            actions.append(action)
        return np.array(actions)


class EpsilonTracker:
    """
    Updates epsilon according to linear schedule
    """

    def __init__(
        self,
        selector: EpsilonGreedyActionSelector,
        eps_start: Union[int, float],
        eps_final: Union[int, float],
        eps_frames: int,
    ):
        self.selector = selector
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_frames = eps_frames

    def frame(self, frame: int):
        eps = self.eps_start - frame / self.eps_frames
        self.selector.epsilon = max(self.eps_final, eps)


if __name__ == "__main__":
    scores = np.array([[1, 2, 3], [1, -1, 0], [0, 0, 0], [1, 1, 1]])
    argmax_selector = ArgmaxActionSelector()
    print("Argmax:", argmax_selector(scores))
    # Argmax: [2 0 0 0]

    greedy_selector = EpsilonGreedyActionSelector(epsilon=0.6, selector=argmax_selector)
    print("Greedy:", greedy_selector(scores))

    prob_selector = ProbabilityActionSelector()
    probs = np.array([[0.1, 0.8, 0.1], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    print("Prob:", prob_selector(probs))
