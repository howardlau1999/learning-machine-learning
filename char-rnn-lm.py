import torchtext
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.cuda
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


CORPUS_PATH = "tang.txt"
TEXT = torchtext.data.Field(init_token="<SOP>", eos_token="<EOP>")
train_dataset = torchtext.datasets.LanguageModelingDataset(CORPUS_PATH, TEXT, newline_eos=False)
TEXT.build_vocab(train_dataset)

class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, latent_dim, dropout=0.5):
        super(BasicRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)
        self.decode = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.init_weights()
        
    def forward(self, x, hidden_states):
        embedding = self.embed(x)
        output, (h_n, c_n) = self.lstm(embedding, hidden_states)
        output = self.dropout(output)
        bsz = output.size(1)
        decoded = F.softmax(self.tanh(self.decode(output.view(-1, output.size(2)))), dim=1)
        decoded = decoded.view(-1, bsz, self.vocab_size)
        return decoded, (h_n, c_n)
    
    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decode.bias.data.zero_()
        self.decode.weight.data.uniform_(-initrange, initrange)
    
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.hidden_dim),
                weight.new_zeros(self.num_layers, bsz, self.hidden_dim))
                
def generate_poem(model):
    idx = TEXT.vocab.stoi["<SOP>"]
    x = torch.Tensor([idx]).view(1, 1).long().to(device)
    poem = []
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for _ in range(128):
            output, hidden = model(x, hidden)
            idx = torch.argmax(output)
            if idx == TEXT.vocab.stoi["<EOP>"]: break
            poem.append(TEXT.vocab.itos[idx])
            x = torch.Tensor([idx]).view(1, 1).long().to(device)
    return poem
    
def detach_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)
    
def train(model, dataset, lr=1e-3, epochs=10):
    model.train()
    train_iter = torchtext.data.BPTTIterator(
            dataset,
            batch_size=128,
            bptt_len=32,
            device=device,
            repeat=False
        )
    vocab_size = len(dataset.fields['text'].vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hidden = None
    total_loss = 0.
    for epoch in range(epochs):
        epoch_loss = 0.
        train_iter.init_epoch()
        for i, batch in enumerate(tqdm(train_iter)):
            if hidden is None:
                hidden = model.init_hidden(batch.batch_size)
            else:
                hidden = detach_hidden(hidden)
            
            text, target = batch.text, batch.target
            output, hidden = model(text, hidden)
            optimizer.zero_grad()
            loss = criterion(output.view(-1, vocab_size), target.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * output.size(0) * output.size(1)
            if i % 100 == 0: print(''.join(generate_poem(model)))
        epoch_loss /= len(dataset.examples[0].text)
        
        print("Epoch Loss: %f" % epoch_loss)
        
