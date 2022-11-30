# carracing_obstacles
Variant of OpenAI Gym CarRacing environment with static obstacles

## Requirements:
- Gym 0.21.0

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