## Rohan Banerjee
## Variant of CarRacing-obstacles-Psi with modified (Dict) observation space,
## where environments are chosen from a fixed evaluation set.

from car_racing_obstacles import CarRacingObstacles
from gym import spaces
import numpy as np

import random
random.seed(0)
TURNRATES=[0.31,0.41,0.51,0.61,0.71]
PROBS=[0.05,0.07,0.09,0.11,0.13]

TRACK_TURN_RATE_MIN = 0.31
TRACK_TURN_RATE_MAX = 0.71
OBSTACLE_PROB_MIN = 0.05
OBSTACLE_PROB_MAX = 0.13

class CarRacingObstaclesPsiKPEval(CarRacingObstacles):
    """
    CarRacingObstaclesPsiKPEval is a modified version of CarRacingObstacles with modified (Dict) observation space,
    and with the ability to optionally normalize the observation space values to be between [0,1].

    The main difference is that each observation from step() is a dictionary with the following keys:
    - 'psi': the environment state (currently, psi = [K,p] = [TRACK_TURN_RATE, OBSTACLE_PROB])
    - 'img': the current image from the environment

    Another diference is that the environment parameters K and p are sampled from a set of possible values for each episode.
    Unlike CarRacingObstaclesKP, the environment set is fixed globally and doesn't change within the episode.

    Also takes in a flag called mode - one of "turn_rate","obs_prob","both" - indicating which parameter(s) to vary.
    """
    def __init__(self, verbose=1, mode="both"):
        # Call superclass constructor
        super().__init__(verbose=verbose)
        # Create a modified Dict observation space
        # Assumes that turn rate and obstacle probability lie in [0,1]
        img_observation_space = self.observation_space
        self.observation_space = spaces.Dict({"image": img_observation_space, \
                                              "psi": spaces.Box(low=np.array([0,0]), high=np.array([1,1]), shape=(2,), dtype=np.float32)})
        # Set self.turnrates and self.probs (env sampling set) based on mode
        self.mode = mode
        if self.mode == "turn_rate":
            self.turnrates = TURNRATES
            self.probs = [0]
            print("Eval environment: Varying only turn rate. Keeping obstacle prob fixed at 0.")
        elif self.mode == "obs_prob":
            self.turnrates = [TRACK_TURN_RATE_MIN]
            self.probs = PROBS
            print(f"Eval environment: Varying only obstacle prob. Keeping turn rate fixed at {TRACK_TURN_RATE_MIN}.")
        elif self.mode == "both":
            self.turnrates = TURNRATES
            self.probs = PROBS
            print("Eval environment: Varying both turn rate and obstacle prob.")

    def reset(self):
        """
        Resamples a new environment from the environment set. Modifies
        the given environment in-place.
        """
        # Sample a new environment parameter set from the environment set
        [K,p] = [random.choice(self.turnrates), random.choice(self.probs)]
        # Set the environment parameters
        print(f"Eval environment: Resetting [K,p] in env.reset() to: {[K,p]}")
        self.TRACK_TURN_RATE = K
        self.OBSTACLE_PROB = p
        # Call the superclass reset() method
        return super().reset()

    def step(self, action):
        # Call the superclass step() method first, and get return values
        obs, reward, done, info = super().step(action)
        # Return the environment parameter values as part of the observation
        track_turn_rate = self.TRACK_TURN_RATE
        obstacle_prob = self.OBSTACLE_PROB
        obs_dict = {"image": obs, "psi": np.array([track_turn_rate, obstacle_prob])}
        # Return the modified observation
        return obs_dict, reward, done, info
