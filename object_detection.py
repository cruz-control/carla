# Edit this to avoid obstacles
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
     # Generate extra vehicles
    spawn_point.location += carla.Location(x=40, y=-3.2)
    spawn_point.rotation.yaw = -180.0
    for _ in range(0, 30):
        spawn_point.location.x += 8.0

        bp = random.choice(blueprint_library.filter('vehicle'))

        # This time we are using try_spawn_actor. If the spot is already
        # occupied by another object, the function will return None.
        npc = world.try_spawn_actor(bp, spawn_point)
        if npc is not None:
            actor_list.append(npc)
            npc.set_autopilot(True)
            print('created %s' % npc.type_id)

    # Create a transform for the spectator
    spectator = world.get_spectator()

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

    # Obstacle Detector
    obs_bp = world.get_blueprint_library().find('sensor.other.obstacle')
    obs_bp.set_attribute("only_dynamics",str(True))
    obs_bp.set_attribute("debug_linetrace",str(True))  # This currently doesn't visualize anything

    # Uncommenting the following attributes currently leads to the script instantly ending

    # obs_bp.set_attribute("distance", float(10))
    # obs_bp.set_attribute("hit_radius", float(3))
    obs_location = carla.Location(0,0,0)
    obs_rotation = carla.Rotation(0,0,0)
    obs_transform = carla.Transform(obs_location,obs_rotation)
    ego_obs = world.spawn_actor(obs_bp,obs_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
    actor_list.append(ego_obs)
    def obs_callback(obs):
        print("Obstacle detected:\n"+str(obs)+'\n')
        print(obs.distance)
    ego_obs.listen(lambda obs: obs_callback(obs))

    while True:
        world.tick()
        if sensor_data["image"] is not None:
            cv2.imshow("rgb", sensor_data["image"])
            cv2.waitKey(1)
            waypoint = world.get_map().get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder | carla.LaneType.Sidewalk))
            transform = waypoint.transform
            loc = transform.location
            vehicle_loc = vehicle.get_transform().location
            l2_dist = np.sqrt((loc.x - vehicle_loc.x)**2 + (loc.y - vehicle_loc.y)**2)
            spec_trans = carla.Transform(vehicle.get_transform().transform(carla.Location(x = -6, z = 2.5)), vehicle.get_transform().rotation)
            spectator.set_transform(spec_trans)

            # print("Distance: ", l2_dist)


finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')