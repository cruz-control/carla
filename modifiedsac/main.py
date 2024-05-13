from sac_torch import Agent
import carla
import random
import numpy as np
import cv2
from torchvision import transforms

HEIGHT = 224
WIDTH = 224
CHANNELS = 3

sensor_data = {"image": None}


agent = Agent(input_dims=(HEIGHT, WIDTH, CHANNELS), n_actions=1, max_size=1000)


def process_img(image, sensor_data):
    i = np.array(image.raw_data)
    i2 = i.reshape((HEIGHT, WIDTH, 4))
    i3 = i2[:, :, :3]
    sensor_data["image"] = i3

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

actor_list = []

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    print("Loading")

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive.

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find("sensor.camera.rgb")
    # change the dimensions of the image
    blueprint.set_attribute("image_size_x", f"{WIDTH}")
    blueprint.set_attribute("image_size_y", f"{HEIGHT}")
    blueprint.set_attribute("fov", "110")

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

            image = sensor_data["image"]
            observation = image

            action = agent.choose_action(observation)

            steering = action[0]
            throttle = 0.5
            brake = 0.0

            control = carla.VehicleControl(throttle, steering, brake)
            vehicle.apply_control(control)

            world.tick()

            observation_ = image

            waypoint = world.get_map().get_waypoint(
                vehicle.get_location(),
                project_to_road=True,
                lane_type=(
                    carla.LaneType.Driving
                    | carla.LaneType.Shoulder
                    | carla.LaneType.Sidewalk
                ),
            )

            transform = waypoint.transform
            loc = transform.location
            vehicle_loc = vehicle.get_transform().location

            l2_dist = np.sqrt(
                (loc.x - vehicle_loc.x) ** 2 + (loc.y - vehicle_loc.y) ** 2
            )

            reward = -l2_dist
            done = False

            agent.remember(observation, action, reward, observation_, done)
            agent.learn()


finally:
    print("destroying actors")
    for actor in actor_list:
        actor.destroy()
    print("done.")
