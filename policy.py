import numpy as np
from torch import nn
from environment import Environment
import torch
import random
from torch.distributions import Categorical
import torch.nn.functional as F

class Policy_VAE(nn.Module):
    def __init__(self, vocab, env: Environment, pretrained_WE = None, n_hiddens = 1024, lazy_linear = True, embed_size = 768, dropout = 0.):
        super().__init__()
        self.n_hiddens = n_hiddens
        self.lazy_linear = lazy_linear
        self.embed_size = embed_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.env = env
        self.debug_mode = False

        # Need to be replace with BERT embeddings
        if pretrained_WE is None:
            self.construct = nn.Parameter(torch.randn((self.vocab_size, embed_size)))
        else: 
            self.construct = nn.Parameter(torch.from_numpy(pretrained_WE).float())

        self.initialize_model(dropout)

    def initialize_model(self, dropout):
        if self.lazy_linear:
            self.encoder = nn.Sequential(
                nn.LazyLinear(self.n_hiddens),
                nn.ReLU(),
                nn.LazyLinear(self.n_hiddens),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.embed_size, self.n_hiddens),
                nn.ReLU(),
                nn.Linear(self.n_hiddens, self.n_hiddens),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        self.fc1 = nn.Linear(self.n_hiddens, self.vocab_size)
        self.fc2 = nn.Linear(self.n_hiddens, self.vocab_size)

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu
        
    def encode(self, x):
        z = self.encoder(x)

        return self.fc1(z), self.fc2(z)
        
    def merge_fn(self, state):
        doc_embed = state["doc_embed"]
        mean_embed = state["mean_topic_embeds"]
        return 0.5 * (doc_embed + mean_embed)
    
    def forward(self, state):
        x = self.merge_fn(state)

        x_norm = x / x.sum(1, keepdim = True)
        mu, logvar = self.encode(x_norm)

        wd = torch.nn.functional.softmax(self.reparameterize(mu, logvar), dim = 1)

        recon_x = torch.matmul(wd, self.construct)

        loss = self.loss_vae(x, recon_x, mu, logvar)

        return wd, loss
    
    def act(self, state):
        probs, loss_vae = self.forward(state)
        # print("state in act policy: ", state)
        m = Categorical(probs)
        action = m.sample()
        
        word = self.vocab[action.item()]
        
        loss = loss_vae + self.loss_contrastive(self.env.get_embedding(word), state)
        
        return self.vocab[action.item()], m.log_prob(action), loss

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
    
