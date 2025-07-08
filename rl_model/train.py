import logging
import time

import ale_py
import gymnasium as gym
import numpy as np

from rl_model.agent import Agent
from rl_model.utils import save_screenshot, save_array

gym.register_envs(ale_py)


def train_model():
    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array")
    n_games = 100

    actions_map = {
        0: "NOOP",
        1: "UP",
        2: "RIGHT",
        3: "LEFT",
        4: "DOWN",
        5: "UPRIGHT",
        6: "UPLEFT",
        7: "DOWNRIGHT",
        8: "DOWNLEFT",
    }

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        alpha=0.0005,
        input_dims=(210, 160, 3),
        n_actions=len(actions_map.keys()),
        mem_size=2048,
        batch_size=128,
        epsilon_end=0.01,
        model_verbose=False
    )
    agent.load_model()

    DEBUG = False

    scores = []
    eps_history = []
    time_per_game = []
    for i in range(n_games):
        start_time = time.time()
        done = False
        score = 0
        observation = env.reset()
        screen = 0
        observation, reward, done, truncated, info = env.step(0)
        action_count = 0
        while not done:
            action = agent.choose_action(observation)
            action_count += 1
            logging.info(f"Action >> original: {action}")
            logging.info(f"Action >> mapped: {actions_map[int(action)]}")
            # Se calcula el siguiente estado, su recompensa, si ya se legó a un estado terminal, etc
            observation_, reward, done, truncated, info = env.step(action)

            score += reward
            if screen >= 250:
                prev_screen = env.render()
                print(
                    f"Game information: Game: {i}, Action: {actions_map[int(action)]}, Score: {score}, Reward: {reward}, Done: {done}, Truncated: {truncated}")
                save_screenshot("100_game/screenshots", f"game_{i}_score_{int(score)}", prev_screen)
                print("Saving Screen...")
                screen = 0
            if screen > 0:
                agent.remember(observation, action, reward, observation_, done, truncated)
            observation = observation_
            agent.learn()
            screen += 1
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - n_games):(i + 1)])
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        time_per_game.append(elapsed_time)
        print(
            f"Game Finished! duration: {elapsed_time} minutes, episode: {i}, score: {score}, average_score: {avg_score}")

        if i % 10 == 0 and i > 0:
            print("Saving Model...")
            agent.save_model()

    save_array("datasets/pacman_screenshots/100_game/data/games_scores.txt", np.array(scores))
    save_array("datasets/pacman_screenshots/100_game/data/games_epsilon.txt", np.array(eps_history))

    print("Finished!")
