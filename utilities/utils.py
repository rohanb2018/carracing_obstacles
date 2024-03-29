## Author: Rohan Banerjee (and Prishita Ray)
## Utilities file (contains useful methods for CarRacing)

import numpy as np

def check_if_car_on_grass(car):
    """
    Checks to see if car is on the grass, which is the case if at least one of the car's wheels
    is not in contact with any tiles (i.e. the car is not in contact with any road or obstacle tiles).

    Note that in some cases, even if one of the wheels is grazing the grass,
    the car may not be considered to be on the grass if that wheel is still in contact with a road or obstacle tile.
    (so there is a "buffer region" around the road where the car is not considered to be on the grass).

    Args:
        car (car_racing.Car)

    Return:
        true if car is on the grass, false otherwise
    """

    cnt=0
    
    for w in car.wheels:
        if len(w.tiles)==0:
            cnt+=1
            # wheel is on the grass (not in contact with any tiles, either road or obstacle)
    if cnt==4:
        return True
    else:
        return False

def check_if_car_on_obstacle(car):
    """
    Checks to see if car is on an obstacle, which is the case if at least one of the car's wheels
    is in contact with an obstacle tile.

    Args:
        car (car_racing.Car)

    Return:
        true if car is on an obstacle, false otherwise
    """
    for w in car.wheels:
        for tile in w.tiles:
            if tile.road_friction > 2.0:    # (indicates that a tile is obstacle: see FrictionDetector in car_racing_obstacles.py)
                return True
    return False

def get_nearest_obstacle_distance(car, obstacle_centroids):
    """
    Returns the distance to the nearest obstacle from the car's position.
    Args:
        car (car_racing.Car)
        obstacle_centroids (list): list of (x, y) coordinates of the centroids of the obstacle tiles

    Return:
        distance to the nearest obstacle from the car's position
    """
    # Compute relative vector from car to obstacle centroids
    car_x, car_y = car.hull.position
    car_position = np.array([car_x, car_y])
    obstacle_centroids_arr = np.array(obstacle_centroids)
    relative_positions = obstacle_centroids_arr - car_position
    # Compute distances to obstacles
    distances = np.linalg.norm(relative_positions, axis=1)
    return np.min(distances)

    ## FUTURE CODE (to filter whether obstacle is in front of the car - doesn't seem to work properly at the moment)
    # Get heading of the car
    # car_heading = car.hull.angle
    # car_heading_vector = np.array([np.cos(car_heading), np.sin(car_heading)])
    # # Only keep obstacles that are in front of the car
    # dot_products = np.dot(relative_positions, car_heading_vector)
    # in_front_mask = dot_products > 0
    # if np.all(~in_front_mask):
    #     # No obstacles in front of the car
    #     return np.inf
    # else:
    #     # Compute distances to obstacles that are in front of the car
    #     distances = np.linalg.norm(relative_positions, axis=1)
    #     distances = distances[in_front_mask]
    #     return np.min(distances)

def evaluate_best_model(best_model, eval_env, num_episodes=500):
    """
    Evaluates a policy on an evaluation CarRacing environment.
    (based on code from: Prishita Ray)

    Prints the following metrics:
    - Episode scores (for every episode)
    - Average number of tiles covered (averaged over all episodes)
    - Average time taken (averaged over all episodes)
    - Average proportion of time spent on grass (averaged over all episodes)

    Args:
        best_model (stable_baselines3.PPO): best policy
        eval_env (car_racing.CarRacing): evaluation environment
    """
    # To calculate number of tiles covered and time taken in default environment
    frames = []
    tiles=0
    times=0
    total_road_or_obstacle_timesteps = 0
    total_grass_timesteps = 0
    for episode in range(1, num_episodes+1):
        obs = eval_env.reset()  #state = env.reset()
        done = False
        score = 0
        info={}
        while not done:
            # Check if car is on the grass
            if check_if_car_on_grass(eval_env.car):
                total_grass_timesteps += 1
            else:
                total_road_or_obstacle_timesteps += 1
            # frames.append(env.render(mode='rgb_array'))
            action , _ = best_model.predict(obs.copy())
            obs, reward, done, info = eval_env.step(action)
            score += reward
        tiles+=eval_env.tile_visited_count
        times+=eval_env.t
        print("Episode:{} Score:{}".format(episode,score))
    print("Number of tiles:",tiles/num_episodes)
    print("Time taken:",times/num_episodes)
    print("Proportion of time spent on grass:", \
          total_grass_timesteps/(total_grass_timesteps+total_road_or_obstacle_timesteps))
    eval_env.close()
