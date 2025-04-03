import gymnasium as gym
import numpy as np
import cv2
from bindsnet.encoding import poisson

# Load Space Invaders

def run():
    env = gym.make("SpaceInvaders-v0", render_mode='human')

    state = env.reset()[0]

    terminated = False
    truncated = False

    while(not truncated and not terminated):
        action = env.action_space.sample()

        new_state, reward, terminated, truncated, _ = env.step(action)


        state = new_state

    env.close()

def

if __name__ == '__main__':
    run()

