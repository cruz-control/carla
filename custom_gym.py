import gym
from gym import spaces
import numpy as np
import carla
import torch
import random

HEIGHT = 256
WIDTH = 256

class CustomEnv(gym.Env):    
    metadata = {'render.modes': ['human']}

    def __init__(self, device, port=2000):
        super(CustomEnv, self).__init__()
        self.device = device
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape= (HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.sensor_data = {"image": None}

        self.actors = []
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(10.0)
        print("Loading")

        self.world = self.client.get_world()

        self.blueprint_library = self.world.get_blueprint_library()

        self.bp = self.blueprint_library.filter('model3')[0]
        print(self.bp)

        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(self.bp, self.spawn_point)
        # self.vehicle.set_autopilot(True)

        self.actors.append(self.vehicle)

        self.blueprint = self.blueprint_library.find('sensor.camera.rgb')
        self.blueprint.set_attribute('image_size_x', f'{WIDTH}')
        self.blueprint.set_attribute('image_size_y', f'{HEIGHT}')
        self.blueprint.set_attribute('fov', '110')

        self.spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = self.world.spawn_actor(self.blueprint, self.spawn_point, attach_to=self.vehicle)

        self.actors.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((HEIGHT, WIDTH, 4))
        i3 = i2[:, :, :3]
        self.sensor_data["image"] = i3

    def step(self, action):

        print(action)

        # self.world.tick()
        # waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving))
        # transform = waypoint.transform
        # loc = transform.location
        # vehicle_loc = self.vehicle.get_transform().location
        # l2_dist = np.sqrt((loc.x - vehicle_loc.x)**2 + (loc.y - vehicle_loc.y)**2)
        # image = self.sensor_data["image"]
        # control = self.vehicle.get_control()
        # steering = control.steer
        # steering = torch.tensor(steering).float().to(self.device)
        # # Execute one time step within the environment

    def reset(self):
        # Reset the state of the environment to an initial state
        for actor in self.actors:
            actor.destroy()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return