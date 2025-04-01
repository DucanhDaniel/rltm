import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(MLP, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_hiddens = hidden_size
        
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, vocab_size)
        self.critic = nn.Linear(hidden_size, 1)  # Critic outputs a scalar value
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class Policy(nn.Module):
    def __init__(self, vocab, env, mlp: MLP):
        super(Policy, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.env = env
        self.mlp = mlp

    def forward(self, state):
        doc_embed = state["doc_embed"]
        logits, value = self.mlp(doc_embed)
        probs = F.softmax(logits, dim=-1)
        return probs, value

    def act(self, state):
        probs, value = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        word = self.vocab[action.item()]
        return word, m.log_prob(action), value

    def compute_loss(self, log_probs, values, rewards, gamma=0.99):
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # Compute actor and critic losses
        advantages = discounted_rewards - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        return actor_loss + critic_loss