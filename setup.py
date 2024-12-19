import gymnasium as gym
from stable_baselines3 import A2C, SAC, PPO
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime

MODELS_DIR = "trained_models"
LOGS_DIR = "logs"

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

class RLEnsemble:
    def __init__(self, env_name="LunarLander-v3"):
        # Create separate environments for different algorithms
        self.env_discrete = gym.make(env_name, render_mode="rgb_array")
        self.env_continuous = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
        
        # Initialize models with appropriate environments
        self.models = {
            'a2c': A2C("MlpPolicy", self.env_discrete, verbose=1, tensorboard_log=LOGS_DIR),
            'sac': SAC("MlpPolicy", self.env_continuous, verbose=1, tensorboard_log=LOGS_DIR),
            'ppo': PPO("MlpPolicy", self.env_discrete, verbose=1, tensorboard_log=LOGS_DIR)
        }
        
        # Keep track of which models use which environments
        self.model_envs = {
            'a2c': self.env_discrete,
            'sac': self.env_continuous,
            'ppo': self.env_discrete
        }
        
        self.session_dir = None
    
    def __del__(self):
        # Check if environments are initialized before trying to close them
        if hasattr(self, 'env_discrete'):
            self.env_discrete.close()
        if hasattr(self, 'env_continuous'):
            self.env_continuous.close()


    def create_session_dir(self):
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(MODELS_DIR, session_name)
        os.makedirs(self.session_dir)
        for model_name in self.models.keys():
            os.makedirs(os.path.join(self.session_dir, model_name))
        return self.session_dir

    def train_models(self, episodes=50):
        if not self.session_dir:
            self.create_session_dir()
        
        print(f"Training models will be saved in: {self.session_dir}")
        
        reward_logs = {model_name: [] for model_name in self.models.keys()}
        
        for episode in range(1, episodes + 1):
            print(f"\nTraining Episode: {episode}/{episodes}")
            
            for model_name, model in self.models.items():
                print(f"Training {model_name.upper()}...")
                model.learn(total_timesteps=1000)
                
                model_path = os.path.join(self.session_dir, model_name, f"episode_{episode}")
                model.save(model_path)
                
                total_reward = self.evaluate_model(
                    model, 
                    self.model_envs[model_name],
                    visual=True, 
                    episode_num=episode,
                    model_name=model_name
                )
                reward_logs[model_name].append(total_reward)
        
        self.plot_training_rewards(reward_logs)
        return self.session_dir

    def evaluate_model(self, model, env, attempts=1, visual=False, episode_num=0, model_name=""):
        total_rewards = []
        
        for _ in range(attempts):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

                if visual:
                    frame = env.render()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"Model: {model_name.upper()}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Episode: {episode_num}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Reward: {episode_reward:.2f}",
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow(f"Lunar Lander - {model_name.upper()}", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            total_rewards.append(episode_reward)
        
        if visual:
            cv2.destroyAllWindows()
        
        avg_reward = np.mean(total_rewards)
        print(f"{model_name.upper()} Average Reward: {avg_reward}")
        return avg_reward

    def plot_training_rewards(self, reward_logs):
        plt.figure(figsize=(12, 6))
        for model_name, rewards in reward_logs.items():
            plt.plot(range(1, len(rewards) + 1), rewards, label=model_name.upper())
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Performance Comparison")
        plt.legend()
        plt.savefig(os.path.join(self.session_dir, "training_comparison.png"))
        plt.show()

    def test_models(self, epoch_time_test=100):
        if not self.session_dir:
            latest_session = self.get_latest_session_dir()
            if not latest_session:
                print("No trained models found. Please train models first.")
                return
            self.session_dir = os.path.join(MODELS_DIR, latest_session)
        
        print(f"Loading models from: {self.session_dir}")
        rewards = {model_name: [] for model_name in self.models.keys()}
        
        for model_name in self.models.keys():
            model_dir = os.path.join(self.session_dir, model_name)
            model_paths = sorted([f for f in os.listdir(model_dir) if f.startswith("episode_")])
            if not model_paths:
                continue
                
            final_model_path = os.path.join(model_dir, model_paths[-1])
            print(f"Testing {model_name.upper()}: {final_model_path}")
            
            if model_name == 'a2c':
                model = A2C.load(final_model_path)
            elif model_name == 'sac':
                model = SAC.load(final_model_path)
            else:  # PPO
                model = PPO.load(final_model_path)
            
            env = self.model_envs[model_name]
            for episode in range(epoch_time_test):
                print(f"Testing {model_name.upper()} Episode {episode + 1}")
                reward = self.evaluate_model(
                    model, 
                    env,
                    visual=True, 
                    episode_num=episode + 1,
                    model_name=model_name
                )
                rewards[model_name].append(reward)
        
        self.plot_testing_rewards(rewards)

    def plot_testing_rewards(self, rewards):
        plt.figure(figsize=(12, 6))
        for model_name, reward_list in rewards.items():
            if reward_list:  # Only plot if we have data
                plt.plot(range(1, len(reward_list) + 1), reward_list, label=model_name.upper())
        
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Testing Performance Comparison")
        plt.legend()
        plt.savefig(os.path.join(self.session_dir, "testing_comparison.png"))
        plt.show()

    @staticmethod
    def get_latest_session_dir():
        if not os.path.exists(MODELS_DIR):
            return None
        sessions = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
        return max(sessions) if sessions else None

def main():
    ensemble = RLEnsemble()
    
    while True:
        choice = input("Choose an option:\n1. Train new models\n2. Test latest models\n3. Exit\nYour choice: ")
        
        if choice == "1":
            session_dir = ensemble.train_models(episodes=5)
            test_choice = input("Training complete. Would you like to test the models now? (y/n): ")
            if test_choice.lower() == 'y':
                print("Press 1 to enter epoch time")
                print("Press 2 to enter for default epoch time (which is 100)")
                input_choice = int(input(""))
                
                epoch_time_test = 100
                if input_choice == 1:
                    epoch_time_test = int(input("Enter an epoch time: "))
                else:
                    print("Default epoch time applying...")
                
                ensemble.test_models(epoch_time_test)
        
        elif choice == "2":
            try:
                print("Press 1 to enter epoch time")
                print("Press 2 to enter for default epoch time (which is 100)")
                input_choice = int(input(""))
                
                epoch_time_test = 100
                if input_choice == 1:
                    epoch_time_test = int(input("Enter an epoch time: "))
                else:
                    print("Default epoch time applying...")
                
                ensemble.test_models(epoch_time_test)
            except Exception as e:
                print(f"\nNo trained models found or error occurred: {str(e)}")
        
        elif choice == "3":
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()