import retro
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import time


#model = PPO.load("tmp/best_model.zip")
model = PPO.load("bestDK/best_model.zip")
#model = PPO.load("bestMario/best_model1.zip")

def main():
    steps = 0

    #env = retro.make(game='SuperMarioBros-Nes')
    env = gym.make('ALE/DonkeyKong-v5')

    env = MaxAndSkipEnv(env, 4)

    obs = env.reset()
    done = False

    while not done:
        time.sleep(0.03)
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        print(info)
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Steps completed:",steps)
    print("Final Info")
    print(info)
    env.close()


if __name__ == "__main__":
    main()