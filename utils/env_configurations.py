from utils.constants import VEHICLE_COUNT

kinematics_config = {
    "observation": {
        "normalize": False,
        "type": "Kinematics",
        "vehicles_count": VEHICLE_COUNT,  # number of visible vehicles to observation
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "action": {"type": "Discrete"},
    },
    "duration": 50,
    "vehicles_count": 20,
    "vehicles_density": 2
}

occupancy_config = {
    "observation": {
        "normalize": False,
        "type": "OccupancyGrid",
        "vehicles_count": VEHICLE_COUNT,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
    },
    "duration": 50,
    "vehicles_count": 20,
    "vehicles_density": 2,
}

ttc_config = {
    "observation": {
        "type": "TimeToCollision",
        "horizon": 10
    }
}
