from gym.envs.registration import register
from highway_env import utils
from highway_env.envs import HighwayEnvFast, HighwayEnv, IntersectionEnv, RoundaboutEnv, UTurnEnv, TwoWayEnv, MergeEnv
from highway_env.vehicle.controller import ControlledVehicle
import constants
from classic_control_env.cartpole import CartPoleEnv


class CartPoleImageWithCost(CartPoleEnv):
    def __init__(self):
        super().__init__()

    def _cost(self, state):
        pole_angle = state[2]
        cart_position = state[0]
        return 1 if abs(pole_angle) >= 0.2095 or abs(cart_position) >= 2.4 else 0


class CartPoleWithCost(CartPoleEnv):
    def __init__(self):
        super().__init__()

    def _cost(self, state):
        pole_angle = state[2]
        cart_position = state[0]
        return 1 if abs(pole_angle) >= 0.2095 or abs(cart_position) >= 2.4 else 0


class HighwayEnvFastNoNormalization(HighwayEnvFast):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "normalize": False,
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            },
            "duration": 50,
            "vehicles_count": 20,
            "vehicles_density": 2,
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class HighwayEnvFastNoNormalizationOccupancyGrid(HighwayEnvFast):
    """
    A variant of highway-v0 with different observation type (OccupancyGrid):
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "normalize": False,
                "type": "OccupancyGrid",
                "vehicles_count": constants.VEHICLE_COUNT,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

            },
            "duration": 50,
            "vehicles_count": 20,
            "vehicles_density": 2,
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class HighwayEnvFastNoNormalizationGrayscale(HighwayEnvFast):
    """
    A variant of highway-v0 with different observation type (Grayscale):
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "normalize": False,
                "type": "GrayscaleObservation",
                "observation_shape": (constants.HW_IMAGE_WIDTH, constants.HW_IMAGE_HEIGHT),
                "stack_size": 1,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
            "duration": 50,
            "vehicles_count": 20,
            "vehicles_density": 2,
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class IntersectionEnvNoNormalization(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "flatten": False,
                "observe_intentions": False,
                "see_behind": True
            }
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class RoundaboutEnvNoNormalization(RoundaboutEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {"x": [-100, 100], "y": [-100, 100], "vx": [-15, 15], "vy": [-15, 15]},
                "see_behind": True
            }
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class UTurnEnvNoNormalization(UTurnEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

class TwoWayEnvNoNormalization(TwoWayEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return config

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        if len(neighbours) == 1:
            return 0
        reward = self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1) \
            + self.config["left_lane_reward"] \
                * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        return reward

class MergeEnvNoNormalization(MergeEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class EgoMergeEnvNoNormalization(MergeEnv):

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "observation": {
                "type": "Kinematics",
                "normalize": False,
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "see_behind": True
            },
        })
        return cfg

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)

    def _make_vehicles(self) -> None:

        road = self.road
        ego_vehicle = self.action_type.vehicle_class(road,
                                                     road.network.get_lane(("j", "k", 0)).position(110, 0),
                                                     speed=30)
        road.vehicles.append(ego_vehicle)

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(90, 0), speed=29))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(70, 0), speed=31))
        road.vehicles.append(other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(5, 0), speed=31.5))

        merging_v = other_vehicles_type(road, road.network.get_lane(("a", "b", 1)).position(30, 0), speed=20)
        merging_v.target_speed = 30
        road.vehicles.append(merging_v)
        self.vehicle = ego_vehicle


    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        action_reward = {0: self.config["lane_change_reward"],
                         1: 0,
                         2: self.config["lane_change_reward"],
                         3: 0,
                         4: 0}
        reward = self.config["collision_reward"] * self.vehicle.crashed \
            + self.config["right_lane_reward"] * self.vehicle.lane_index[2] / 1 \
            + self.config["high_speed_reward"] * self.vehicle.speed_index / (self.vehicle.target_speeds.size - 1)

        # Altruistic penalty
        for vehicle in self.road.vehicles:
            if vehicle.lane_index == ("b", "c", 2) and isinstance(vehicle, ControlledVehicle):
                speed_reward = 0 if vehicle.target_speed == 0 else (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
                reward += self.config["merging_speed_reward"] * speed_reward

        return utils.lmap(action_reward[action] + reward,
                          [self.config["collision_reward"] + self.config["merging_speed_reward"],
                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                          [0, 1])