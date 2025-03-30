import torch
from transformers import BertTokenizer, BertModel
import topmost
from topmost.utils import _utils
from topmost import Preprocess
import numpy as np
import random

class Environment():
    def __init__(self, dataset : topmost.data.basic_dataset.BasicDataset, topic_model: topmost.ETM, num_top_words, device):
        self.list_train_text = dataset.train_texts
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.topic_model = topic_model
        # self.topic_model_trainer = topic_model_trainer
        self.dataset = dataset
        self.num_top_words = num_top_words
        self.top_words_list = self.get_top_words(verbose=False)
        self.device = device
        self.batch_size = 200
        self.preprocess = Preprocess(verbose=False)

    def get_beta(self):
        beta = self.topic_model.get_beta().detach().cpu().numpy()
        return beta

    def get_top_words(self, num_top_words=None, verbose = False):
        if num_top_words is None:
            num_top_words = self.num_top_words
        beta = self.get_beta()
        top_words = _utils.get_top_words(beta, self.dataset.vocab, num_top_words, verbose=False)
        return top_words

    def test(self, bow):
        data_size = bow.shape[0]
        theta = list()
        all_idx = torch.split(torch.arange(data_size), self.batch_size)

        with torch.no_grad():
            self.topic_model.eval()
            for idx in all_idx:
                batch_input = bow[idx]
                batch_theta = self.topic_model.get_theta(batch_input)
                theta.extend(batch_theta.cpu().tolist())

        theta = np.asarray(theta)
        return theta
        
    def get_embedding(self, inputs):
        # Tokenize và tạo tensor đầu vào
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)

        # Lấy embedding từ BERT
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Lấy embedding của toàn bộ câu (dùng CLS token hoặc mean pooling)
        # cls_embedding = outputs.last_hidden_state[:, 0, :]  # Lấy embedding của [CLS] token
        mean_embedding = outputs.last_hidden_state.mean(dim=1)  # Trung bình tất cả token
        return mean_embedding # shape: (1, 768)
    
    def get_k_top_words(self, input, testing = False):
        new_docs = [input]
        parsed_new_docs, new_bow = self.preprocess.parse(new_docs, vocab=self.dataset.vocab)

        # Chuyển đổi sparse matrix sang dense NumPy array
        new_bow_dense = new_bow.toarray()  # Hoặc .todense() cũng được

        # Chuyển sang PyTorch tensor
        new_theta = self.test(torch.tensor(new_bow_dense, device=self.device).float())

        tmp = [self.top_words_list[i] for i in new_theta.argmax(1)]
        
        self.k_word_list = tmp[0].split()

        if testing:
            return self.k_word_list
        
        return [self.get_embedding(word) for word in self.k_word_list]
    
    def cosine_similarity(self, A, B):
        return torch.nn.functional.cosine_similarity(A, B)

    def get_text(self):
        text = random.choice(self.list_train_text)
        return text
    
    def reset(self, current_text = None):
        self.idx = 0

        if current_text is None:
            self.current_text = self.get_text()
            
        doc_embeds = self.get_embedding(self.current_text)
        self.k_top_words = self.get_k_top_words(self.current_text)

        self.state = {
            "doc_embed": doc_embeds, 
            "mean_topic_embeds": torch.zeros_like(doc_embeds)
        }
        return self.state
    
    def calculate_reward(self, topic_embed):
        reward = torch.zeros((1,), dtype = torch.float32)
        for item in self.k_top_words:
            reward -= self.cosine_similarity(topic_embed, item)
        reward += self.cosine_similarity(topic_embed, self.state["doc_embed"]) * len(self.k_top_words)
        return reward
    
    def step(self, action): # action is a word
        topic_embed = self.get_embedding(action)
        
        # Create new state
        old_mte = self.state["mean_topic_embeds"]
        new_mte = (old_mte * self.idx + topic_embed) / (self.idx + 1)
        self.state = {
            "doc_embed": self.state["doc_embed"],
            "mean_topic_embeds": new_mte
        }

        done = False
        self.idx += 1
        if (self.idx == len(self.k_top_words)):
            # self.idx = 0
            done = True
        reward = self.calculate_reward(topic_embed)

        return self.state, reward.item(), done
    
    def print_current_state(self):
        print("### ENVIRONMENT STATE ###")
        print(f"Current time step: {self.idx}")
        print("Document: ", self.current_text)
        print("K-Top words: ", self.k_word_list)
        