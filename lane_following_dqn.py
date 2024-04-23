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

import SAC.buffer as buffer
import SAC.networks as networks
import SAC.sac_torch as sac_torch

IM_WIDTH = 1080
IM_HEIGHT = 720

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sensor_data = {"image": None}

def process_img(image, sensor_data):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    sensor_data["image"] = i3

actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    print("Loading")

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    sensor.listen(lambda data: process_img(data, sensor_data))

    while True:
        world.tick()
        if sensor_data["image"] is not None:
            cv2.imshow("rgb", sensor_data["image"])
            cv2.waitKey(1)
            waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, 
                                                    lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
            transform = waypoint.transform
            loc = transform.location
            vehicle_loc = vehicle.get_transform().location
            l2_dist = np.sqrt((loc.x - vehicle_loc.x)**2 + (loc.y - vehicle_loc.y)**2)
            # Get the image and flatten it
            image = sensor_data["image"]
            image = cv2.resize(image, (IM_WIDTH, IM_HEIGHT))
            image = np.array(image).flatten()
            image = torch.tensor(image).float().to(device)
            # Get the steering angle
            control = vehicle.get_control()
            steering = control.steer
            steering = torch.tensor(steering).float().to(device)
            # Get the predicted steering angle
            pred_steering = 
            # Reward is negative of the absolute distance from the center of the lane
            reward = -l2_dist

            reward = reward * 10
            print("Reward: ", reward)

            




finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')