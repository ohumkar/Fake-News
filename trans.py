import pandas as pd
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import transformers as hf
from sklearn.linear_model import LogisticRegression

from utils.processsing import News, transformer_loader, save_checkpoint, load_checkpoint
from utils.score import report_score, LABELS
from utils.generate_splits import generate_train_valid_splits
from model import BertClassifier

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
Initialize model and tokenizer
'''
# Get tokenizer
tokenizer_class, pretrained_weights = (hf.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

# Get head and body transformers
head_trans_class, body_trans_class = (hf.DistilBertModel, hf.DistilBertModel)
head_trans = head_trans_class.from_pretrained(pretrained_weights)
body_trans = body_trans_class.from_pretrained(pretrained_weights)

# Get LR classifier
logistic_classifier = LogisticRegression()


'''
Splitting to train / valid data
'''
BATCH_SIZE = 8
NUM_WORKERS = 8

train_ids, valid_ids = generate_train_valid_splits(stances_df, training = 0.1)

train_dataset = News(stances_df.iloc[train_ids].reset_index(drop=True), body_df, tokenizer)
train_dataloader = transformer_loader(train_dataset,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True)

valid_dataset = News(stances_df.iloc[valid_ids].reset_index(drop=True), body_df, tokenizer)
valid_dataloader = transformer_loader(valid_dataset,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True)


# use label.item()
# import IPython ; IPython.embed() ; exit(1)

'''
Define training and testing functions
'''

def train(dataloader, head_trans, body_trans, classifier, load_model = False, save_model = True, num_epochs = 2) :

    torch.backends.cudnn.benchmark = True
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)
    load_model = load_model
    save_model = save_model

    learning_rate = 3e-3
    num_epochs = num_epochs

    # For tensorboard
    writer = SummaryWriter('runs/bert')
    step = 0

    # Initialize Model
    model = BertClassifier(head_trans, body_trans, classifier).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model :
        model, optimizer, step = load_checkpoint(torch.load('bert_chkpnt/my_checkpoint.pth.tar'), model, optimizer)
        return model

    for epoch in range(num_epochs) :
        if save_model :
            checkpoint = {
                'state_dict' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'step' : step
            }
            save_checkpoint(checkpoint)

        loop = tqdm(enumerate(dataloader), total = len(dataloader), leave = False)

        for batch, (head, body, stance) in loop :

            outputs = model(head.to(device), body.to(device))
            breakpoint()
            loss = criterion(outputs.float(), stance.to(device).long())

            writer.add_scalar('Training Loss', loss.item(), step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss = loss.item())
        
            running_loss += loss.item()
            running_accuracy += ((torch.argmax(outputs, dim = 1) == stance.to(device)).sum().item()) / BATCH_SIZE
            if (batch+1) % 10 == 0 :
                writer.add_scalar('Running Loss', running_loss / 10, epoch * len(dataloader) + batch)
                writer.add_scalar('Running Accuracy', running_accuracy / 10, epoch * len(dataloader) + batch)
                
                running_loss = 0.0
                running_accuracy = 0
   
    return model


def get_predictions(dataloader, model) :
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    writer = SummaryWriter('runs/bert-evals')
    step = 0

    criterion = nn.CrossEntropyLoss()

    # Calculating validation loss:
    loop = tqdm(enumerate(dataloader), total = len(dataloader),leave = True)
    for batch, (head, body, stance) in loop :

        outputs = model(head.to(device), body.to(device))
        loss = criterion(outputs.float(), stance.to(device).long())

        writer.add_scalar('Validation Loss', loss.item(), step)
        step += 1

    # Calculating score
    all_preds, all_actual = np.array([]), np.array([])
    model.eval()
    for batch, (head_ids, head_attention_mask, body_ids, body_attention_mask, stance) in loop :

        predictions = model(head_ids.to(device), head_attention_mask.to(device), body_ids.to(device), body_attention_mask.to(device))
        preds = torch.argmax(predictions, dim = 1)

        all_preds = np.append(all_preds, preds.cpu().detach().numpy())
        all_actual = np.append(all_actual, stance.cpu().detach().numpy())
        print(predictions)
        print(preds)
        break

    get_scores(all_preds, all_actual)
    

def get_scores(predicted, actual) :

    get_labels = np.vectorize(lambda t: LABELS[int(t)])
    predicted = get_labels(predicted)
    actual = get_labels(actual)

    print("Scores on the val set")
    report_score(actual,predicted)
    print("")


if __name__ == '__main__' :
  
    model = train(train_dataloader, head_trans, body_trans, logistic_classifier, load_model = False, save_model = True, num_epochs = 2)
    get_predictions(valid_dataloader, model)