import os,sys
sys.path.append('../')

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

import torchtext
from torchtext.data import Field, Dataset, Example

from gensim.models import KeyedVectors

from src.model import Model
from src.util import Util

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerClassifier(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class TorchDataset(torchtext.data.Dataset):
    def __init__(self, text_list, label_list, fields):
        self.examples = [ Example.fromlist([text, label], fields) for text,label in zip(text_list, label_list) ]

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return self.examples[i]

class ModelTransformer(Model):
    def __init__(self, run_fold_name, **params):
        super().__init__(run_fold_name, params)

    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        """ 
            tr_x : List[str] (example.) [ "I am happy", "hello" ]
            tr_y : List[label]
            embedding_model : gensim.models.KeyedVectors Object
        """
        validation = va_x is not None

        nb_classes = 5
        batch_size = int(self.params['batch_size'])
        embedding_vector = self.params['embedding_vector']
        use_pre_embedding = not (embedding_vector is None)

        self.max_len = 70
        self.TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split(","),use_vocab=True,
            fix_length=self.max_len, batch_first=True,include_lengths=True)
        self.LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

        fields = [('Text', self.TEXT), ('Label', self.LABEL)]
        train_ds = TorchDataset(tr_x, tr_y, fields)
        if validation:
            val_ds = TorchDataset(va_x, va_y, fields)
        
        ## DEBUG
        print(len(train_ds))
        print(vars(train_ds[0]))
        
        if validation:
            if use_pre_embedding:
                self.TEXT.build_vocab(train_ds, val_ds, vectors=embedding_vector)
            else:
                self.TEXT.build_vocab(train_ds, val_ds)
        else:
            if use_pre_embedding:
                self.TEXT.build_vocab(train_ds, vectors=embedding_vector)
            else:
                self.TEXT.build_vocab(train_ds)

        ## DEBUG
        print(self.TEXT.vocab.stoi)

        # パラメータ
        train_dl = torchtext.data.Iterator(train_ds, batch_size=batch_size, train=True)
        if validation:
            val_dl = torchtext.data.Iterator(val_ds, batch_size=batch_size, train=False, sort=False)
        
        ## DEBUG
        batch = next(iter(val_dl))
        print(batch.Text)
        print(batch.Label)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Use : ", device)

        ntokens = len(TEXT.vocab.stoi) # the size of vocabulary
        emsize = 200 # embedding dimension
        nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
        nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead = 2 # the number of heads in the multiheadattention models
        dropout = 0.2 # the dropout value
        epoch = 10
        model = TransformerClassifier(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
        
        criterion = nn.CrossEntropyLoss()
        lr = 5.0 # learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        import time
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, np.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


        

    def predict(self, te_x):
        xtest_seq = self.token.texts_to_sequences(te_x)
        te_x = pad_sequences(xtest_seq, maxlen=self.max_len)
        y_pred = self.model.predict(te_x)
        return y_pred

    def score(self, te_x, te_y):
        y_pred = self.predict(te_x)
        return f1_score(np.identity(5)[te_y], np.identity(5)[np.argmax(y_pred, axis=1)], average='samples')

    def save_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)

    def load_model(self, feature):
        model_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}.h5')
        scaler_path = os.path.join(f'../model/model/{feature}', f'{self.run_fold_name}-scaler.pkl')
        self.model = load_model(model_path)

