import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import custom_gym
from stable_baselines3 import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = custom_gym.CustomEnv(device, port=2000)

model = SAC("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

model.save("sac_carla")

del model

model = SAC.load("sac_carla")

env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    print(reward)

    if terminated or truncated:
        obs, info = env.reset()
