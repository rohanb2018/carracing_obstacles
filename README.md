# carracing_obstacles
Variant of OpenAI Gym CarRacing environment with static obstacles

## Environment Description

This environment is a modified version of the [CarRacing environment](https://www.gymlibrary.dev/environments/box2d/car_racing/) from OpenAI Gym, and is also inspired by NotAnyMike's CarRacing variant as well [here](https://github.com/NotAnyMike/gym). It has the following features:
- Obstacles are randomly placed in a particular road tile with probability `OBSTACLE_PROB`, and are randomly placed on either the left or right side of the road.
- Obstacles are spaced apart by a minimum of `OBSTACLE_SPACING` tiles.
- Obstacles incur a fixed penalty (given by `OBSTACLE_PENALTY`)


## Requirements
- Gym 0.21.0

## Usage
To test the environment using human keyboard inputs, simply run:
```
python car_racing_obstacles.py
```

To use the environment in OpenAI Gym RL scenarios, move `car_racing_obstacles.py` to your working directory (or add
the directory containing this repo to your PYTHONPATH)
and do the following:

```
from car_racing_obstacles import CarRacingObstacles
from gym.wrappers.time_limit import TimeLimit
env = TimeLimit(CarRacingObstacles(),max_episode_steps=1000)
...
```

By default, `OBSTACLE_PROB` will be set to 0.05. Note that the time limit of 1000 timesteps is to ensure that behavior is identical to that of the
built-in CarRacing-v0 environment.


When we call `env.step(action)`, the returned info dictionary includes the following keys:
- `num_obstacles`: total number of obstacles in the track
- `num_collisions`: total number of collisions with obstacles (note that we can collide with the same obstacle more than once)
