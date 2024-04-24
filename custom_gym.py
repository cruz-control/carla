import gym
from gym import spaces
import numpy as np
import carla
import random

HEIGHT = 256
WIDTH = 256

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        # They must be gym.spaces objects   
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape= (HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.actors = []
        self.client = carla.Client('localhost', 2000)
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
        

        


    def step(self, action):
        # Execute one time step within the environment

    def reset(self):
        # Reset the state of the environment to an initial state
        for actor in self.actors:
            actor.destroy()

    def render(self, mode='human', close=False):
        # Render the environment to the screen