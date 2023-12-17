import os
import retro
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

def make_env(env_id, rank, seed=0):
    def _init():
        env = retro.make(env_id) # , render_mode='human'
        env = MaxAndSkipEnv(env, 4) #Frame skip, 4fps skip
        env.seed = seed + rank
        return env
    
    set_random_seed(seed)
    return _init

    
if __name__ == '__main__':
    log_dir = "tmp/"
    tensor_log = "./board/"
    os.makedirs(log_dir, exist_ok=True)

    # Setteo del ambiente de entrenamiento
    env_id = "SuperMarioBros-Nes"
    num_cpu = 4
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu") # Especifica el dispositivo utilizado
    
    #Variables de entrenamiento

    LR = 0.00003 # learning rate
    episodes = 1000000 # episodios totales de entrenamiento
    freq = 1000 # chequeo periodico de valores para elegir los mas optimos hasta el momento
    

    # En el caso de cpu, realiza entrenamiento paralelo segun los hilos especificados
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]), "tmp/monitor")

    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=tensor_log,device=device, learning_rate=0.00003) # Cnn = Convolutional Neural Network
    #model = PPO.load('bestMario/best_model4', env=env)

    print("---------------------Entrenando con ",device," ----------------------------")
    print("------------------Learning Start---------------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=freq, log_dir=log_dir) # Permite revisar episodios anteriores (ver como funciona)
    model.learn(total_timesteps=episodes, callback=callback, tb_log_name="Mario_T3_PPO") # Nombre de tensorboard y config del entrenamiento
    model.save(env_id)
    print("------------------Learning Done--------------------")