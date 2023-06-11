from utils import evaluate_best_model
import gym

class RandomPolicy:
    def __init__(self, env):
        self.env = env

    def predict(self, obs):
        # Randomly sample action from env
        action = self.env.action_space.sample()
        action[1] = 0.5 # full acceleration
        action[2] = 0.0 # no brake
        return action, None

if __name__=="__main__":
    # Load evaluation environment
    eval_env = gym.make("CarRacing-v0")
    # Evaluate random policy
    evaluate_best_model(RandomPolicy(eval_env), eval_env, num_episodes=10)