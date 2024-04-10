import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time


def main():
    actor_list = []

    try:
        # Create World
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        # Get Blueprints
        blueprint_library = world.get_blueprint_library()

        # Select Vehicle
        bp = blueprint_library.find('vehicle.dodge.charger_2020')

        # Random colors (it's funny)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Random Spawn Point Generation
        transform = random.choice(world.get_map().get_spawn_points())

        vehicle = world.spawn_actor(bp, transform)

        # Create a transform for the spectator
        spectator = world.get_spectator()
        spec_trans = carla.Transform(vehicle.get_transform().transform(carla.Location(x = -4, z = 2.5)), vehicle.get_transform().rotation)
        spectator.set_transform(spec_trans)

        # Update actor list for future destruction
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let vehicle drive freely
        vehicle.set_autopilot(True)

        # Generate RGB Camera and attach it to vehicle
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)

        # Update location (idk why this is here it was in tutorial.py)
        location = vehicle.get_location()
        location.x += 40
        vehicle.set_location(location)
        print('moved vehicle to %s' % location)

        # Save images from camera to disk
        camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame))

        # Generate extra vehicles
        transform.location += carla.Location(x=40, y=-3.2)
        transform.rotation.yaw = -180.0
        for _ in range(0, 10):
            transform.location.x += 8.0

            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)

        # Get vehicle control attributes (try to sync the data collection with the image generation if possible)
        for i in range(20):
            control = vehicle.get_control()
            print(control.steer)
            print(control.throttle)
            print(control.brake)
            print("\n")
            time.sleep(2)

    finally:
        # Destroy actors
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()