import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HeadingsLSTM(nn.Module) :
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) :
        super(HeadingsLSTM, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_size, sparse = True)
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, headings) :
        embedds = self.dropout(self.embeddings(headings))
        outs,ht = self.lstm(embedds)
        outputs = self.linear(ht[-1])
        return outputs

class BodyLSTM(nn.Module) :
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) :
        super(BodyLSTM, self).__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_size, sparse = True)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, bodies) :
        embedds = self.dropout(self.embeddings(bodies))
        #outs,(ht, ct) = self.lstm(embedds)
        outs, ht = self.lstm(embedds)
        outputs = self.linear(ht[-1])
        return outputs
    

class Classifier(nn.Module) :
    def __init__(self, embed_size, hidden_size, head_vocab_size, body_vocab_size, num_layers) :
        super(Classifier, self).__init__()
        self.headLSTM = HeadingsLSTM(embed_size, hidden_size, head_vocab_size, num_layers)
        self.bodyLSTM = BodyLSTM(embed_size, hidden_size, body_vocab_size, num_layers)
        self.linear = nn.Linear(2*hidden_size, 4)
        

    # want [4, 256*2] 
    # what i have [4, 18, 256] + [4, 346, 256] --> [4, 364, 256]
    # need to reduce to [4, 256] + [4, 256] --> [4, 256*2]
    def forward(self, headings, bodies) :
        head_features = self.headLSTM(headings)
        body_features = self.bodyLSTM(bodies)
        #print(head_features.shape)
        #print(body_features.shape)
        features = torch.cat((head_features, body_features), dim = -1)
        #print(features.shape)
        outputs = self.linear(features)
        #print(outputs.shape)
        return outputs

class BertClassifier(nn.Module):
    def __init__(self, head_transformer, body_transformer, classifier): 
        super(BertClassifier, self).__init__()
        self.head_transformer = head_transformer
        self.body_transformer = body_transformer
        self.linear = nn.Linear(768*2, 4)
        self.classifier = classifier

    def apply_padding(self, tokenized):
        for i in tokenized.values:
            if len(i) > max_len: 
                max_len = len(i)
        
        padded = np.array([i+[0]*max_len-len(i) for i in tokenized.values])
        attention_mask = np.where(padded != 0, 1, 0)
        return padded, attention_mask

    def forward(self, head_ids, body_ids) :
        head_feat = self.head_transformer(head_ids)
        body_feat = self.body_transformer(body_ids)

        features = torch.cat((head_feat, body_feat), dim = -1)
        return self.linear(features)

"""
head --> head featuers
body --> body features

"""


class CLFDClassifier(nn.Module):
    def __init__(self, kernel_size = 5, out_c = 100, head_pool = 500, body_pool = 500, drop = 0.5):
        super(CLFDClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_c, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=out_c, kernel_size=kernel_size)
        self.pool_head = nn.AdaptiveAvgPool1d(head_pool)
        self.pool_body = nn.AdaptiveAvgPool1d(body_pool)
        self.linear = nn.Linear(in_features= (head_pool+body_pool)*out_c, out_features = 4)
        self.bn = nn.BatchNorm1d(out_c)
        self.dropout = nn.Dropout(p = drop)
        
    def forward(self, head, body):
        heads = F.relu(self.conv2(F.relu(self.conv1(head))))
        bodies = F.relu(self.conv2(F.relu(self.conv1(body))))

        pooled_head = self.pool_head(heads)
        pooled_body = self.pool_body(bodies)

        # should have dimensions (n_c, feats) ...n_c --> 1
        # concat along feats
        comb_features = torch.cat((pooled_head, pooled_body), dim = -1)
        outputs = self.dropout(self.linear(self.bn(comb_features)))
        # outputs are logits --> use CrossEntropyLoss
        return comb_features, outputs


if __name__ == '__main__' :
    print('Model module!')