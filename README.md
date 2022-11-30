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

To use the environment in OpenAI Gym RL scenarios, move `car_racing_obstacles.py` to your working directory,
and do the following:

```
from car_racing_obstacles import CarRacingObstacles
env = CarRacingObstacles()
...
```