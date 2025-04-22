import topmost.utils
import topmost.utils._utils
from topmost import eva
import numpy as np

class Beta_env():
    def __init__(self, dataset, max_steps, num_topics, embed_size, reference_corpus, num_top_words = 5, random = False):

        self.num_topics = num_topics
        self.embed_size = embed_size
        self.random_init = random
        self.step_counter = 0
        self.max_steps = max_steps
        self.num_top_words = num_top_words
        self.reference_corpus = reference_corpus
        self.dataset = dataset
        self.vocab = dataset.vocab
        self.vocab_size = len(self.vocab)
        # self.vocab_size = 100

        self.str_to_embeds = {
            dataset.vocab[i]:dataset.pretrained_WE[i]
            for i in range(0, self.vocab_size)
        }
        
        self.beta, self.topic_embeds = self.get_starting_state()


    def get_starting_state(self):
        if self.random_init:
            beta = np.random.normal(loc = 0, scale=1, size = (self.num_topics, self.vocab_size))
            topic_embeds = np.random.normal(loc = 0, scale = 1, size = (self.num_topics, self.embed_size))
        else:
            beta, topic_embeds = None, None
        return beta, topic_embeds

    def reset(self):
        self.step_counter = 0
        self.beta, self.topic_embeds = self.get_starting_state()
        return np.concatenate((self.beta, self.topic_embeds), axis = 1)
    
    def cosine_similarity(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return np.asarray([0])
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def compute_cosine_similarity(self, list_txt):
        result = 0
        for i in range(self.num_topics):
            for word in list_txt[i].split():
                we = self.str_to_embeds[word]
                result += self.cosine_similarity(we, self.topic_embeds[i, :]).item()
        return result
        
    def calculate_reward(self):
        top_words = self.get_top_words(False)
        total_TD =  eva.topic_diversity._diversity(top_words)

        total_cosine_similarity = self.compute_cosine_similarity(top_words)
        return (total_TD + total_cosine_similarity / (self.num_topics * self.num_top_words)) / 2
        

    def get_top_words(self, verbose = True):
        top_word_list = topmost.utils._utils.get_top_words(self.beta, self.vocab, num_top_words=self.num_top_words, verbose=verbose)
        return top_word_list

    def step(self, action):
        self.beta += action
        new_state = np.concatenate((self.beta, self.topic_embeds), axis = 1)
        reward = self.calculate_reward()
        self.step_counter += 1
        done = False
        if self.step_counter == self.max_steps:
            done = True
            self.step_counter = 0
        
        self.state = new_state
        return self.state, reward, done