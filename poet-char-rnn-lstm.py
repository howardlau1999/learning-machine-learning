
# coding: utf-8

# In[1]:


import torchtext
import pandas as pd
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.cuda
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

TEXT = torchtext.data.Field(init_token="<SOP>", eos_token="<EOP>")
train_dataset = torchtext.datasets.LanguageModelingDataset("tang.txt", TEXT, newline_eos=False)
TEXT.build_vocab(train_dataset)

class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(BasicRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.decode = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.init_weights()
        
    def forward(self, x, hidden_states):
        embedding = self.dropout(self.embed(x))
        output, (h_n, c_n) = self.lstm(embedding, hidden_states)
        output = self.dropout(output)
        bsz = output.size(1)
        decoded = self.decode(output.view(-1, output.size(2)))
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

def generate_poem(model, sample=False):
    model.eval()
    idx = TEXT.vocab.stoi["<SOP>"]
    x = torch.Tensor([idx]).view(1, 1).long().to(device)
    poem = []
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for _ in range(128):
            output, hidden = model(x, hidden)
            output = output.view(model.vocab_size)
            if sample:
                probs = F.softmax(output, dim=0).cpu().numpy()
                probs /= probs.sum()
                idx = np.random.choice(range(model.vocab_size), p=probs)
            else:
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
    
def train(model, dataset, lr=1e-3, epochs=10, debug=False):
    train_iter = torchtext.data.BPTTIterator(
            dataset,
            batch_size=2048,
            bptt_len=33,
            device=device,
            repeat=False
        )
    vocab_size = len(dataset.fields['text'].vocab)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    hidden = None
    total_loss = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = []
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
            epoch_loss.append(loss.item())
            
        epoch_loss = np.mean(epoch_loss)
        total_loss.append(epoch_loss)
        if debug:
            print("Epoch %d Loss: %f" % (epoch, epoch_loss))
            print(''.join(generate_poem(model)))
        elif (epoch + 1) % 10 == 0: 
            print("Epoch %d Loss: %f" % (epoch, epoch_loss))
            print(''.join(generate_poem(model)))
            print(''.join(generate_poem(model, True)))
            with open("loss.log", "a") as f:
                f.write("Epoch %d Loss: %f\n" % (epoch, epoch_loss))
                f.write(''.join(generate_poem(model)) + '\n')
                f.write(''.join(generate_poem(model, True)) + '\n')
        if (epoch + 1) % 1000 == 0 or epoch == 0 and not debug:
            torch.save(model.state_dict(), "model_{0:d}.pth".format(epoch))
    return total_loss

model = BasicRNN(len(train_dataset.fields['text'].vocab), 300, 1024, 2, 0.5).to(device)
train(model, train_dataset, lr=1e-2, epochs=20000, debug=False)
generate_poem(model, False)

