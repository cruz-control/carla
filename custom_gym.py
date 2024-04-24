import gym
from gym import spaces
import numpy as np
import carla

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

        


    def step(self, action):
        # Execute one time step within the environment

    def reset(self):
        # Reset the state of the environment to an initial state
        for actor in self.actors:
            actor.destroy()

    def render(self, mode='human', close=False):
        # Render the environment to the screen