import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2
import PIL.Image as Image

IM_WIDTH = 1080
IM_HEIGHT = 720

sensor_data = {"rgb": None, "dvs": None}

def process_rgb(image, sensor_data):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    sensor_data["rgb"] = i3

def process_dvs(image, sensor_data):
    # Example of converting the raw_data from a carla.DVSEventArray
    # sensor into a NumPy array and using it as an image
    dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
        ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
    dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    # Blue is positive, red is negative
    dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
    sensor_data["dvs"] = dvs_img


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

    rgb_blueprint = blueprint_library.find('sensor.camera.rgb')

    # change the dimensions of the image
    rgb_blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    rgb_blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    rgb_blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    rgb_sensor = world.spawn_actor(rgb_blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(rgb_sensor)

    # do something with this sensor
    rgb_sensor.listen(lambda data: process_rgb(data, sensor_data))

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    dvs_blueprint = blueprint_library.find('sensor.camera.dvs')

    # change the dimensions of the image
    dvs_blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    dvs_blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    dvs_blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    dvs_sensor = world.spawn_actor(dvs_blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(dvs_sensor)

    # do something with this sensor
    dvs_sensor.listen(lambda data: process_dvs(data, sensor_data))


    while True:
        world.tick()
        if sensor_data["rgb"] is not None and sensor_data["dvs"] is not None:
            cv2.imshow("rgb", sensor_data["rgb"])
            cv2.imshow("dvs", sensor_data["dvs"])
            cv2.waitKey(1)


finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')