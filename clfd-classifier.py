import pandas as pd
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import transformers as hf
from utils.processsing import News, clfd_loader, save_checkpoint, load_checkpoint
from utils.score import report_score, LABELS
from utils.generate_splits import generate_train_valid_splits
from model import CLFDClassifier

'''
Getting training data
'''
#train_stances_path = 'fnc-1/train_stances.csv'
train_stances_path = 'fnc-1/small_stances.csv'
train_bodies_path = 'fnc-1/train_bodies.csv'
base_dir = 'body-keys'

stances_df = pd.read_csv(train_stances_path)
body_df = pd.read_csv(train_bodies_path)

'''
Splitting to train / valid data
'''
BATCH_SIZE = 8
NUM_WORKERS = 8


# Import CLFD Values and convert into dictionary
clfd_head_vocab = pd.read_csv('clfd/heading-tf_clfd.csv')
clfd_head_vocab = clfd_head_vocab.set_index('Unnamed: 0')
clfd_head_vocab = clfd_head_vocab.T
clfd_head_vocab = clfd_head_vocab.iloc[0].to_dict()

clfd_body_vocab = pd.read_csv('clfd/heading-tf_clfd.csv')
clfd_body_vocab = clfd_body_vocab.set_index('Unnamed: 0')
clfd_body_vocab = clfd_body_vocab.T
clfd_body_vocab = clfd_body_vocab.iloc[0].to_dict()


# Get tokenizer
tokenizer_class, pretrained_weights = (hf.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Generate split ideas
train_ids, valid_ids = generate_train_valid_splits(stances_df, 
                                                   training = 0.8)


"""
labels_unique, counts = np.unique(df['labels'], return_counts = True)
print(f'Unique Labels: {labels_unique}')
class_weights = [sum(counts) / c for c in counts] 

example_weights = [class_weights[e] for e in df['labels]]
sampler = WeightedRandomSampler(example_weights, len(df['labels']))
train_data_loader = DataLoader(sampler=sampler)
"""



# Load Dataset
train_dataset = News(stances_df.iloc[train_ids].reset_index(drop=True), body_df, tokenizer, return_tokens=True)
train_dataloader = clfd_loader(train_dataset,
                        clfd_head_vocab,
                        clfd_body_vocab,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True,)

valid_dataset = News(stances_df.iloc[valid_ids].reset_index(drop=True), body_df, tokenizer, return_tokens = True)
valid_dataloader = clfd_loader(valid_dataset,
                        clfd_head_vocab,
                        clfd_body_vocab,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True)


# use label.item()
# import IPython ; IPython.embed() ; exit(1)

'''
Define training and testing functions
'''

def train(dataloader, learning_rate = 3e-3, load_model = False, save_model = True, num_epochs = 2) :

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device set to: {device}')
    load_model = load_model
    save_model = save_model

    learning_rate = learning_rate
    num_epochs = num_epochs

    # For tensorboard
    writer = SummaryWriter('runs/clfd')
    step = 0

    # Initialize Model
    model = CLFDClassifier(kernel_size = 5, out_c = 1, 
                           head_pool = 200, body_pool = 500, 
                           drop = 0.5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model :
        model, optimizer, step = load_checkpoint(torch.load('clfd_chkpnt/my_checkpoint.pth.tar'), model, optimizer)
        return model

    for epoch in range(num_epochs) :
        if save_model :
            checkpoint = {
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'step' : step
            }
            save_checkpoint(checkpoint)

        epoch_loss = 0.0
        running_loss = 0.0
        running_accuracy = 0.0

        loop = tqdm(enumerate(dataloader), total = len(dataloader), leave = False)

        for batch, (head, body, stance) in loop :
            
            _, outputs = model(head.float().to(device), body.float().to(device))
            outputs = outputs.squeeze(1)
            loss = criterion(outputs.float(), stance.to(device).long())

            writer.add_scalar('Training Loss', loss.item(), step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            running_loss += loss.item()

            # Update progress bar
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = epoch_loss)

            running_accuracy += ((torch.argmax(outputs, dim = 1) == stance.to(device)).sum().item()) / BATCH_SIZE
            if (batch+1) % 10 == 0 :
                writer.add_scalar('Running Batch Loss', running_loss / 10, epoch * len(dataloader) + batch)
                writer.add_scalar('Running Batch Accuracy', running_accuracy / 10, epoch * len(dataloader) + batch)
                
                running_loss = 0.0
                running_accuracy = 0
        writer.add_scalar('Epoch Loss', epoch_loss / len(dataloader), epoch)
   
    return model


def get_predictions(dataloader, model) :
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter('runs/clfd-evals')
    step = 0

    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    # Calculating validation loss:
    loop = tqdm(enumerate(dataloader), total = len(dataloader),leave = True)
    for batch, (head, body, stance) in loop :

        _, outputs = model(head.float().to(device), body.float().to(device))
        outputs = outputs.squeeze(1)
        loss = criterion(outputs.float(), stance.to(device).long())

        writer.add_scalar('Validation Batch Loss', loss.item(), step)
        val_loss += loss.item()
        step += 1
    print(f'Validation Loss: {val_loss / len(dataloader)}')

    # Calculating score
    loop = tqdm(enumerate(dataloader), total = len(dataloader),leave = True)
    all_preds, all_actual = np.array([]), np.array([])
    model.eval()
    for batch, (head, body, stance) in loop :
        _, predictions = model(head.float().to(device), body.float().to(device))
        predictions = predictions.squeeze(1)
        preds = torch.argmax(predictions, dim = 1)

        all_preds = np.append(all_preds, preds.cpu().detach().numpy())
        all_actual = np.append(all_actual, stance.cpu().detach().numpy())
        
    get_scores(all_preds, all_actual)
    

def get_scores(predicted, actual) :

    get_labels = np.vectorize(lambda t: LABELS[int(t)])
    # breakpoint()
    predicted = get_labels(predicted)
    actual = get_labels(actual)

    print("Scores on the val set")
    report_score(actual,predicted)
    print("")


if __name__ == '__main__' :
  
    model = train(train_dataloader, 
                  learning_rate = 5e-3,
                  load_model = False, 
                  save_model = True, 
                  num_epochs = 12)
    get_predictions(valid_dataloader, model)