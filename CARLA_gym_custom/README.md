These files consist of a CARLA gym environment, based on the gym_carla library. Currently, the gym environment is customized to provide observation state details using a front, rear, and 2 side cameras. A top down point-cloud based view is also made possible through LIDAR.

The run.py file runs a DQN algorithm from stable_baselines, using the gym environment. Steps to use this environment:

1. Download and run the latest version of CARLA. This environment was tested on CARLA v0.9.15
2. Create and activate a conda environment. (This environment was tested on python 3.7)
3. Clone the repo and cd into the gym-carla folder.
Run the following:
4. pip3 install -r requirements.txt
5. pip3 install -e .
6. export PYTHONPATH=$PYTHONPATH:<path to CARLA installation folder/PythonAPI/carla/dist/carla-0.9.15-py3...>
7. Modify run.py with the port number you are running CARLA on, as well as any other parameters you would like to change.
8. python3 run.py

If all steps were successful, you should see a Pygame window visualizing the RL algorithm running.