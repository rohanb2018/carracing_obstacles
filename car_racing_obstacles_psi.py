## Rohan Banerjee
## Version of CarRacing-obstacles with modified (Dict) observation space.

import car_racing_obstacles
from car_racing_obstacles import CarRacingObstacles
from gym import spaces
import numpy as np

TRACK_TURN_RATE_MIN = 0.31
TRACK_TURN_RATE_MAX = 0.71
OBSTACLE_PROB_MIN = 0.05
OBSTACLE_PROB_MAX = 0.13

class CarRacingObstaclesPsiKP(CarRacingObstacles):
    """
    CarRacingObstaclesPsiKP is a modified version of CarRacingObstacles with modified (Dict) observation space,
    and with the ability to optionally normalize the observation space values to be between [0,1].

    The main difference is that each observation from step() is a dictionary with the following keys:
    - 'psi': the environment state (currently, psi = [K,p] = [TRACK_TURN_RATE, OBSTACLE_PROB])
    - 'img': the current image from the environment

    Another diference is that the environment parameters K and p are sampled from a set of possible values for each episode.
    """
    def __init__(self, verbose=1, normalize_obs=False, turn_rate=TRACK_TURN_RATE_MIN, obstacle_prob=OBSTACLE_PROB_MIN, \
                 env_set=np.array([[TRACK_TURN_RATE_MIN, OBSTACLE_PROB_MIN]]), env_rng=np.random.default_rng(),STATE_W=64,STATE_H=64):
        # Call superclass constructor
        super().__init__(verbose=verbose,STATE_W=STATE_W,STATE_H=STATE_H)
        # Create a modified Dict observation space
        # NOTE: Assumes that turn rate and obstacle probability lie in [0,1]
        img_observation_space = self.observation_space
        self.observation_space = spaces.Dict({"image": img_observation_space, \
                                              "psi": spaces.Box(low=np.array([0,0]), high=np.array([1,1]), shape=(2,), dtype=np.float32)})
        # Set the normalization flag
        self.normalize_obs = normalize_obs
        print(f"Normalizing CarRacingPsiKP observations: {self.normalize_obs}")
        # Set the initial turn rate and obstacle probability
        self.TRACK_TURN_RATE = turn_rate
        self.OBSTACLE_PROB = obstacle_prob
        print(f"Setting turn rate to: {self.TRACK_TURN_RATE}, obstacle prob to: {self.OBSTACLE_PROB}")
        # Set the initial environment set (from which we sample environment parameters)
        # Np array of shape (*,2)
        self.env_set = env_set
        # Set the environment random number generator
        self.env_rng = env_rng


    def change_env_set(self, env_set):
        """
        Sets the environment set to the given env_set.
        """
        self.env_set = env_set

    def get_env_set(self):
        """
        Returns the current environment set.
        """
        return self.env_set

    def reset(self):
        """
        Resamples a new environment from the environment set. Modifies
        the given environment in-place.
        """
        # Sample a new environment parameter set from the environment set
        [K,p] = self.env_set[self.env_rng.integers(0, self.env_set.shape[0]),:]
        # Set the environment parameters
        print(f"Resetting [K,p] in env.reset() to: {[K,p]}")
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
        # Optionally normalize the observations
        if self.normalize_obs:
            obs = (obs - self.observation_space["image"].low) / (self.observation_space["image"].high - self.observation_space["image"].low)
            track_turn_rate = (track_turn_rate - TRACK_TURN_RATE_MIN)/(TRACK_TURN_RATE_MAX-TRACK_TURN_RATE_MIN)
            obstacle_prob = (obstacle_prob - OBSTACLE_PROB_MIN)/(OBSTACLE_PROB_MAX-OBSTACLE_PROB_MIN)
        obs_dict = {"image": obs, "psi": np.array([track_turn_rate, obstacle_prob])}
        # Return the modified observation
        return obs_dict, reward, done, info

if __name__ == "__main__":
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, mod):
        if k == key.LEFT and a[0] == -1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == +1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingObstaclesPsiKP(normalize_obs=False)
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                print(f"Info dict {info}")
                print(s["psi"])
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
