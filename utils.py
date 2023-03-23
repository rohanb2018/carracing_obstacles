## Author: Rohan Banerjee
## Utilities file (contains useful methods for CarRacing)

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

    for w in car.wheels:
        if len(w.tiles)==0:
            # wheel is on the grass (not in contact with any tiles, either road or obstacle)
            return True
    return False
