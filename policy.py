import numpy as np
from torch import nn
from environment import Environment
import torch
import random
from torch.distributions import Categorical
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, vocab_size, embed_size=768, hidden_size=1024, dropout=0., pretrained_WE=None):
        super(VAE, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        if pretrained_WE is None:
            self.construct = nn.Parameter(torch.randn((self.vocab_size, embed_size)))
        else: 
            self.construct = nn.Parameter(torch.from_numpy(pretrained_WE).float())
            
        # Encoder 
        self.encoder = nn.Module(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Latent space
        self.fc1 = nn.Linear(hidden_size, vocab_size) # Mean
        self.fc2 = nn.Linear(hidden_size, vocab_size) # Log variance
        
    def encode(self, x):
        z = self.encoder(x)
        return self.fc1(z), self.fc2(z)
    
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
        
        

class Policy(nn.Module):
    def __init__(self, vocab, env: Environment, vae: VAE):
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.env = env
        self.vae = vae
        self.debug_mode = False

        
    def merge_fn(self, state):
        doc_embed = state["doc_embed"]
        mean_embed = state["mean_topic_embeds"]
        return 0.5 * (doc_embed + mean_embed)
    
    def forward(self, state):
        x = self.merge_fn(state)

        x_norm = x / x.sum(1, keepdim = True)
        mu, logvar = self.vae.encode(x_norm)

        wd = torch.nn.functional.softmax(self.reparameterize(mu, logvar), dim = 1)

        recon_x = torch.matmul(wd, self.vae.construct)

        loss = self.loss_vae(x, recon_x, mu, logvar)

        return wd, loss
    
    def act(self, state):
        probs, loss_vae = self.forward(state)
        # print("state in act policy: ", state)
        m = Categorical(probs)
        action = m.sample()
        
        word = self.vocab[action.item()]
        
        # loss = loss_vae + self.loss_contrastive(self.env.get_embedding(word), state)
        
        return word, m.log_prob(action)

    def loss_contrastive(self, z, state, temperature = 0.07):
        # print("state: ", state)
        # print("doc_embed shape:", state["doc_embed"].shape)
        # print("mean_topic_embeds shape:", state["mean_topic_embeds"].shape)
        z_pos = state["doc_embed"]
        z_neg = state["mean_topic_embeds"]

        

        sim_pos = F.cosine_similarity(z, z_pos, dim=-1) / temperature
        sim_neg = F.cosine_similarity(z, z_neg, dim=-1) / temperature

        loss = -torch.log(torch.exp(sim_pos) / (torch.exp(sim_pos) + torch.exp(sim_neg)))

        return loss
    
    def loss_vae(self, x, recon_x, mu, logvar):
        recon_loss = F.mse_loss(x, recon_x)
        # print(((x - recon_x)*(x - recon_x)).sum() / 768)
        KLD = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)

        if self.debug_mode:
            print("X: ", x)
            print("recon_x: ", recon_x)
            print("recon_loss: ", recon_loss)
            print("KLD: ", KLD)

            print("X shape: ", x.shape)
            print("X_recon shape: ", recon_x.shape)

        loss = recon_loss + KLD

        return loss
