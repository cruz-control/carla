import gymnasium as gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
# "meow meow meow meow" - Sanya

if __name__ == '__main__':
    env = gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)

    print("action space shape:", env.action_space.shape)
    print("observation space shape:", env.observation_space.shape)

    input = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]

    agent = Agent(input_dims=input, n_actions=env.action_space.shape[0], max_size=10000)
    n_games = 2503

    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    # env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)

    filename = '2d_car.png'

    figure_file = 'plots/' + filename

    best_score = 0
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            print(observation[0].shape)
            img = np.array(observation[0])
            print(img.shape)
            flat_img = img.flatten()
            print("HERE2")
            print(flat_img.shape)

            action = agent.choose_action(flat_img)
            observation_, reward, terminated, truncated, info = env.step(action)
            
            done = False

            if terminated or truncated:
                done = True

            score += reward

            img_ = np.array(observation_[0])
            flat_img_ = img_.flatten()

            agent.remember(flat_img, action, reward, flat_img_, done)

            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)