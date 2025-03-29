import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from animator import Animator
from policy import Policy_VAE
from environment import Environment


def reinforce(policy, env : Environment, optimizer, n_training_episodes, max_t, gamma, print_every):
    anim = Animator(xlabel = "n_training_episodes", ylabel = "avg_score", legend = ["score"])
    scores_deque = deque(maxlen = 100)
    scores = []

    for i_episode in range(1, n_training_episodes + 1):
        saved_log_prods = []
        rewards = []
        state = env.reset()

        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_prods.append(log_prob)
            state, reward, done = env.step(action)
            rewards.append(reward)
            if done:
                break
        
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        print(rewards)
        returns = deque(maxlen = max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]:
            disc_return_t = (returns[0] if len(returns) > 0 else 0)
            returns.appendleft(gamma * disc_return_t + rewards[t])

        # eps is the smallest repersentable float, which is added
        # to the standard devitation of the returns to avoid numerical instabilities
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        policy_loss = torch.zeros(size = (1,))
        for log_prob, disc_return in zip(saved_log_prods, returns):
            policy_loss += -log_prob * disc_return
        # print(policy_loss)
        # policy_loss = torch.sum(torch.tensor(policy_loss))

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if (i_episode % print_every == 0):
            
            anim.add([i_episode], [np.mean(scores_deque)])
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

"""This class sucks"""
# class ReinforceTrainer():
#     def __init__(self, policy, env, optimizer, max_t = 20, print_every = 10, gamma=0.99, num_epochs=1000):
#         self.policy = policy
#         self.env = env
#         self.optimizer = optimizer
#         self.gamma = gamma
#         self.num_epochs = num_epochs
#         self.max_t = max_t
#         self.print_every = print_every
#         # self.animator = Animator(
#         #     xlabel = "Epochs", ylabel = None, 
#         #     legend = ["avg reward", "avg loss", "avg net loss"]
#         # )
#         self.animator_reward = Animator(
#             xlabel = "Epochs", ylabel = None, 
#             legend = ["avg reward"]
#         )
#         self.animator_loss = Animator(
#             xlabel = "Epochs", ylabel = None, 
#             # legend = ["avg loss", "avg net loss"]
#             legend = ["avg loss"]
#         )

#     def compute_returns(self, rewards):
#         returns = deque(maxlen = self.max_t)
#         n_steps = len(rewards)

#         for t in range(n_steps)[::-1]:
#             disc_return_t = (returns[0] if len(returns) > 0 else 0)
#             returns.appendleft(self.gamma * disc_return_t + rewards[t])

#         return list(returns)
    
#     def sample_trajectory(self):
#         rewards = []
#         log_probs = []
#         total_net_loss = torch.zeros(1, requires_grad=True)
#         state = self.env.reset()
#         for _ in range(self.max_t):    
#             # print("state in trainer: ", state)
#             action, log_prob, policy_loss = self.policy.act(state)
#             next_state, reward, done = self.env.step(action)
#             # print("done: ", done)
#             rewards.append(reward)
#             log_probs.append(log_prob)
#             total_net_loss = total_net_loss + policy_loss
#             state = next_state
#             if done:
#                 break

#         return rewards, log_probs, total_net_loss

#     def train(self):
#         for epoch in range(self.num_epochs):
#             # Sample trajectory
#             rewards, log_probs, total_net_loss = self.sample_trajectory()

#             # Compute returns and advantages
#             returns = self.compute_returns(rewards)

#             # Update policy
#             policy_loss = self.update_policy(log_probs, returns, total_net_loss)

#             # Print and plot results
#             if epoch % self.print_every == 0:
#                 rewards = torch.stack(rewards).mean()
#                 print(f'Epoch {epoch}, Average Reward: { rewards:.2f}, Policy Loss: {policy_loss.item():.2f}, Net Loss: {total_net_loss.item():.2f}')
#                 self.plot_rewards(epoch + 1, rewards, policy_loss, total_net_loss)

#     def update_policy(self, log_probs, returns, total_net_loss):
#         # Compute policy loss
#         policy_loss  = torch.zeros(size = (1,))
#         for log_prob, ret in zip(log_probs, returns):
#             # print("log_prob: ", log_prob)
#             policy_loss += -log_prob * ret
#         # policy_loss += total_net_loss / len(log_probs)
        
#         # Backpropagation
#         self.optimizer.zero_grad()
#         policy_loss.backward()
#         self.optimizer.step()

#         return policy_loss

#     def plot_rewards(self, epoch, rewards, policy_loss, total_net_loss):
#         # rewards = np.array(rewards)
#         avg_reward = rewards

#         avg_loss = policy_loss.item()
#         avg_net_loss = total_net_loss.item()

#         # self.animator.add(epoch, (avg_reward, avg_loss, avg_net_loss))
#         # self.animator_loss.add(epoch, [avg_loss, avg_net_loss])
#         self.animator_loss.add(epoch, [avg_loss])
#         self.animator_reward.add(epoch, [avg_reward])
