import pandas as pd
import numpy as np
import os

import spacy

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

stance_to_label = {'unrelated': 0, 'discuss': 1, 'disagree': 2, 'agree': 3}
label_to_stance = {v:k for (k,v) in stance_to_label.items()}

spacy_eng = spacy.load('en')

class News(Dataset) :
    def __init__(self, stances_df, body_df, tokenizer, return_tokens = False):
        self.stances_df = stances_df
        self.headings = self.stances_df['Headline']
        self.stances = self.stances_df['Stance'] 
        self.body_df = body_df
        self.bodies = self.body_df['articleBody']
        self.tokenizer = tokenizer
        self.return_tokens = return_tokens
        
    def __len__(self) :
        return len(self.stances_df)

    def bert_tokens(self, head, body):
        head_tokens = self.tokenizer.encode(head, add_special_tokens = True, truncation = True, max_length = 512)
        body_tokens = self.tokenizer.encode(body, add_special_tokens = True, truncation = True, max_length = 512)
        if self.return_tokens: # to return tokens instead of token_ids
            head_tokens = np.array(self.tokenizer.convert_ids_to_tokens(head_tokens))
            body_tokens = np.array(self.tokenizer.convert_ids_to_tokens(body_tokens))
        
        return head_tokens, body_tokens
        
    def __getitem__(self, index) :
        heading = self.headings[index]
        stance = self.stances[index]
        label = stance_to_label[stance]
        bdid = self.stances_df['Body ID'][index]
        body = self.body_df[self.body_df['Body ID'] == bdid].articleBody.item()   

        head_tokens, body_tokens = self.bert_tokens(heading, body)

        if self.return_tokens:#Since we cant make tensor of strings
            return head_tokens, body_tokens, torch.tensor(label)
        else: 
            return torch.tensor(head_tokens), torch.tensor(body_tokens), torch.tensor(label)

class MyCollate :
    def __init__(self, pad_idx) :
        self.pad_idx = pad_idx
        
    def __call__(self, batch) :
        heads = [item[0] for item in batch]
        heads = pad_sequence(heads, batch_first = True, padding_value = self.pad_idx)
        
        bodies = [item[1] for item in batch]
        bodies = pad_sequence(bodies, batch_first = True, padding_value = self.pad_idx)
        
        targets = torch.Tensor([item[2] for item in batch]).t()
        return heads, bodies, targets
    

class NewsCollate :
    def __init__(self) :
        self.pad_idx = 0
        
    def __call__(self, batch) :
        heads = [item[0] for item in batch]
        heads = pad_sequence(heads, batch_first = True, padding_value = self.pad_idx)
        
        bodies = [item[1] for item in batch]
        bodies = pad_sequence(bodies, batch_first = True, padding_value = self.pad_idx)
        
        targets = torch.Tensor([item[2] for item in batch]).t()
        return heads, bodies, targets

class CLFDCollate:
    def __init__(self, head_vocab, body_vocab):
        self.head_vocab = head_vocab
        self.body_vocab = body_vocab
        self.pad_idx = '0'

    def custom_padding(self, sequences, vocab, batch_first = True, padding_value = '0'):
        
        max_len = max([s.shape[0] for s in sequences])

        # important to specify dtype as "object" for arbitrary length strings in numpy
        if batch_first:
            output = np.full((len(sequences), max_len), padding_value, dtype=object)
        else:
            output = np.full((max_len, len(sequences)), padding_value, dtype=object)
        
        for item in sequences: 
            for i, token in enumerate(item):
                if token in vocab: 
                    item[i] = vocab[token]
                else :
                    item[i] = '0.0'
        
        for i, item in enumerate(sequences):
            length_of_list = item.shape[0]
            if batch_first:
                output[i, :length_of_list] = item
                # print(output[i, :length_of_list])
            else:
                output[:length_of_list, i] = item

        return output.astype('float64')

    def __call__(self, batch):
        heads = [item[0] for item in batch]
        heads = self.custom_padding(heads, self.head_vocab, batch_first = True, padding_value = self.pad_idx)

        bodies = [item[1] for item in batch]
        bodies = self.custom_padding(bodies, self.body_vocab, batch_first = True, padding_value = self.pad_idx)
        
        targets = torch.Tensor([item[2] for item in batch]).t()
        return torch.tensor(heads).unsqueeze_(1), torch.tensor(bodies).unsqueeze_(1), targets

def get_loader(dataset_inst,
               batch_size = 4,
               num_workers = 4,
               shuffle = True,
               pin_memory = True) :
    dataset = dataset_inst
    
    pad_idx = dataset.head_vocab.stoi['<PAD>']
    
    loader = DataLoader(dataset = dataset,
                       batch_size = batch_size,
                       num_workers = num_workers,
                       shuffle = shuffle,
                       pin_memory = pin_memory,
                       collate_fn = MyCollate(pad_idx = pad_idx))
    return loader

def transformer_loader(dataset_inst,
               batch_size = 4,
               num_workers = 4,
               shuffle = True,
               pin_memory = True) :
    dataset = dataset_inst

    loader = DataLoader(dataset = dataset,
                       batch_size = batch_size,
                       num_workers = num_workers,
                       shuffle = shuffle,
                       pin_memory = pin_memory,
                       collate_fn = NewsCollate())
    return loader


def clfd_loader(dataset_inst,
               head_vocab,
               body_vocab,
               batch_size = 4,
               num_workers = 4,
               shuffle = True,
               pin_memory = True) :
    dataset = dataset_inst

    loader = DataLoader(dataset = dataset,
                       batch_size = batch_size,
                       num_workers = num_workers,
                       shuffle = shuffle,
                       pin_memory = pin_memory,
                       collate_fn = CLFDCollate(head_vocab, body_vocab))
    return loader

class Vocabulary :
    def __init__(self, freq_threshold) :
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.freq_threshold = freq_threshold
    
    @staticmethod
    def tokenizer_eng(text) :
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def __len__(self) :
        return len(self.itos)
    
    def build_vocabulary(self, sentence_list) :
        frequencies = {}
        idx = 4
        
        for sentence in sentence_list :
            for word in self.tokenizer_eng(sentence) :
                if word not in frequencies :
                    frequencies[word] = 1
                else :
                    frequencies[word] += 1
                    
                if frequencies[word] == self.freq_threshold :
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text) :
        tokenized_text = self.tokenizer_eng(text) 
        return [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text]
    


    '''
    def apply_padding(self, tokenized):
        for i in tokenized.values:
            if len(i) > max_len: 
                max_len = len(i)
        
        padded = np.array([i+[0]*max_len-len(i) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        return padded, attention_mask
    '''

    '''
    def bert_tokens(self, head, body):
        head_tokens = self.tokenizer.encode(head, add_special_tokens = True)
        body_tokens = self.tokenizer.encode(body, add_special_tokens = True)

        head_attention_mask, padded = self.apply_padding(head_tokens)
        head_ids = torch.tensor(np.array(padded))
        head_attention_mask = torch.tensor(head_attention_mask)

        body_attention_mask, padded = self.apply_padding(body_tokens)
        body_ids = torch.tensor(np.array(padded))
        body_attention_mask = torch.tensor(body_attention_mask)

        return head_ids, head_attention_mask, body_ids, body_attention_mask
    '''
 




if __name__ == '__main__' :
    print('processing module')
