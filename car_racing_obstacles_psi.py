## Rohan Banerjee
## Version of CarRacing-obstacles with modified (Dict) observation space.

import car_racing_obstacles
from car_racing_obstacles import CarRacingObstacles
from gym import spaces
import numpy as np

class CarRacingObstaclesPsiKP(CarRacingObstacles):
    """
    CarRacingObstaclesPsiKP is a modified version of CarRacingObstacles with modified (Dict) observation space.

    The main difference is that each observation from step() is a dictionary with the following keys:
    - 'psi': the environment state (currently, psi = [K,p] = [TRACK_TURN_RATE, OBSTACLE_PROB])
    - 'img': the current image from the environment
    """
    def __init__(self, *args, **kwargs):
        # Call superclass constructor
        super().__init__(*args, **kwargs)
        # Create a modified Dict observation space
        # NOTE: Assumes that turn rate and obstacle probability lie in [0,1]
        img_observation_space = self.observation_space
        self.observation_space = spaces.Dict({"image": img_observation_space, \
                                              "psi": spaces.Box(low=np.array([0,0]), high=np.array([1,1]), shape=(2,), dtype=np.float32)})


    def step(self, action):
        # Call the superclass step() method first, and get return values
        obs, reward, done, info = super().step(action)
        # Return the environment parameter values as part of the observation
        track_turn_rate = car_racing_obstacles.TRACK_TURN_RATE
        obstacle_prob = car_racing_obstacles.OBSTACLE_PROB
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

    env = CarRacingObstaclesPsiKP()
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
