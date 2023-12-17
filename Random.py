import gym
import retro
import numpy as np
import time


#env = gym.make("ALE/DonkeyKong-v5")
env = retro.make("SuperMarioBros-Nes")
obs = env.reset()




print(env.observation_space.shape)

done = False
if isinstance(env.action_space, gym.spaces.Discrete):
    num_actions = env.action_space.n
    print(f"Number of discrete actions: {num_actions}")
    possible_actions = list(range(num_actions))
    print(f"Possible discrete actions: {possible_actions}")
else:
    print("The action space is not discrete.")
    num_actions = env.action_space.n
    print(f"Number of indiscrete actions: {num_actions}")
    possible_actions = list(range(num_actions))
    print(f"Possible indiscrete actions: {possible_actions}")

while not done:
    time.sleep(0.01)
    obs, rew, done, info = env.step(env.action_space.sample())
    print(env)
    env.render()

env.close()