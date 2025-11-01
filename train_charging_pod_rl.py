from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from loop_1_MAIN_RL import ChargingPodEnv
import sys
import traci
import os
#
# Custom Callback to save model at the end of each episode
class EpisodeCheckpointCallback(BaseCallback):
    """
    Save model checkpoints at the end of each episode,
    and log performance (zero_range, eff).
    """
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # SB3 increments num_timesteps each step
        # Use info dicts to detect episode end
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_counter += 1
                zero_range = info.get("zero_range", None)
                eff = info.get("eff", None)
                reward = info["episode"]["r"]

                if self.verbose > 0:
                    print(f"[Checkpoint] Ep={self.episode_counter}, "
                          f"Reward={reward:.2f}, ZeroRange={zero_range}, Eff={eff:.2f}")

                # Save checkpoint at the requested frequency
                if self.episode_counter % self.save_freq == 0:
                    filename = os.path.join(self.save_path, f"ep{self.episode_counter}")
                    self.model.save(filename)

        return True

if __name__ == "__main__":
    STATE_FILENAME = "SAVE_state_reset.xml.gz"
    warmup_time = 0
    done_time=700

    if len(sys.argv) > 1:
        sumo_config_file = sys.argv[1]
    else:
        sumo_config_file = "loop1.sumocfg"

    log_dir = "./ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    # Now wrap environment AFTER warm-up
    def make_env():
        env = ChargingPodEnv(sumo_config_file, ["pa_0", "pa_4", "pa_5", "pa_7", "pa_8", "pa_10", "pa_11", "pa_3", "pa_0", "pa_4", "pa_5", "pa_6","pa_2", "pa_3","pa_0", "pa_1","pa_2","pa_3", "pa_0", "pa_1","pa_9","pa_10", "pa_11", "pa_3"], load_state_file=STATE_FILENAME)
        env = Monitor(env, filename="training_logs/monitor.csv",  info_keywords=("zero_range", "eff"))  # logs rewards
        return env

    env = DummyVecEnv([make_env])
    # Either load normalization stats if they exist, or create new
    normalize_path = "models/vecnormalize.pkl"
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        print("[Info] Loaded VecNormalize stats")
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Path to pretrained model
    pretrained_model_path = "models/charging_pod_ppo_model_pretrained.zip"

    if os.path.exists(pretrained_model_path):
        print(f"[Info] Continuing training from {pretrained_model_path}")
        model = PPO.load(pretrained_model_path, env=env)  # reload with new env
    else:
        print("[Info] No pretrained model found, starting from scratch.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=8e-4,#3e-4
            n_steps=1024,#2048
            batch_size=64,
            n_epochs=5,#10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./ppo_tensorboard/"
        )
    num_episodes = 20
    timesteps_per_episode = done_time - warmup_time  # ~7600
    total_timesteps = num_episodes * timesteps_per_episode
    model.learn(total_timesteps=total_timesteps)
    model.save("models/charging_pod_ppo_model")

    # vecnormalize is not part of the model; save separately:
    env.save("models/vecnormalize.pkl")
    env.close()
    traci.close()

    # return ChargingPodEnv(sumo_config_file, ["pa_0", "pa_1", "pa_2", "pa_3"], load_state_file=STATE_FILENAME)
    # return Monitor(env, filename="training_logs/monitor.csv")
    # log_dir = "./dqn_tensorboard/"

    # model = DQN(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     learning_starts=1000,
    #     buffer_size=100000,
    #     batch_size=64,
    #     gamma=0.99,
    #     target_update_interval=250,
    #     train_freq=1,
    #     gradient_steps=1,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.02,
    #     tensorboard_log="./dqn_tensorboard/"
    # )
    # model.save("models/charging_pod_dqn_model")
    # model = DQN.load("models/charging_pod_dqn_model_v2", env=env)