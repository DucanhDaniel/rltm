import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype = np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        
        return batches
    
    def get_arrays(self):
        return np.array(self.states), \
                        np.array(self.actions),\
                        np.array(self.probs),\
                        np.array(self.vals),\
                        np.array(self.rewards),\
                        np.array(self.dones)

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, 
                 hidden_size1 = 256, hidden_size2 = 256, chkpt_dir = 'checkpoints/ppo'):
        super().__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, n_actions),
            nn.Softmax(dim = -1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # Return distribution over actions
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only = True))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, hidden_size1 = 256, hidden_size2 = 256, 
                 chkpt_dir = "checkpoints/ppo"):
        super().__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, "critic_torch_ppo")
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr = alpha)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))
     
class Agent:
    def __init__(self, n_actions, input_dims, gamma = 0.99, alpha = 0.0003, gae_lambda = 0.95,
                 policy_clip = 0.2, 
                 # Hyperparameters for continous actions
                 batch_size = 64, N = 2048, n_epochs = 10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print("Saving models...")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        print("Model saved successfully!")

    def load_models(self):
        print("Loading models...")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        print("Model loaded successfully!")

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype = torch.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        log_probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, log_probs, value
    
    def learn(self):
        # Calculate advantage 
        state_arr, action_arr, old_probs_arr, values,\
        reward_arr, done_arr = self.memory.get_arrays()
        advantage = np.zeros(len(reward_arr), dtype = np.float32)

        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount*(reward_arr[k] + self.gamma*values[k + 1]*\
                                     (1 - int(done_arr[k])) - values[k])
                discount *= self.gamma*self.gae_lambda

            advantage[t] = a_t
        advantage = torch.tensor(advantage).to(self.actor.device)
        values = torch.tensor(values).to(self.actor.device)


        for _ in range(self.n_epochs):
            batches = self.memory.generate_batches()

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype = torch.float, device = self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch], device = self.actor.device, dtype = torch.float)
                actions = torch.tensor(action_arr[batch], device = self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * advantage[batch]

                # Calculate actor loss
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Calculate critic loss
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

