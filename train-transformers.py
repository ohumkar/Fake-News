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

from utils.dataset import News, transformer_loader
from utils.helpers import generate_train_valid_splits, save_checkpoint, load_checkpoint, report_score, LABELS
from utils.models import BertClassifier

# Defining Constants
EMBED_SIZE = 10
NUM_WORKERS = 4
BATCH_SIZE = 4
NUM_EPOCHS = 4
LOAD_MODEL = False
SAVE_MODEL = True
LR = 3e-2
MODEL_PATH = 'chkpnts/'

def train_model(train_loader, 
                valid_loader, 
                head_trans, 
                body_trans, 
                classifier, 
                load_model = LOAD_MODEL, 
                save_model = SAVE_MODEL, 
                model_path = MODEL_PATH+'model.pth.tar', 
                num_epochs = NUM_EPOCHS, 
                lr = LR) :

    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(device)

    load_model = load_model
    save_model = save_model

    learning_rate = lr
    num_epochs = num_epochs

    # For tensorboard
    writer = SummaryWriter('runs/vpp')
    step = 0
  
    # Initialize Model
    model = BertClassifier(head_trans, body_trans, classifier).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model :
        model, optimizer, step = load_checkpoint(torch.load(MODEL_PATH), model, optimizer)
    
    for epoch in range(num_epochs) :
        running_loss = 0.0
        for mode in ['train', 'eval']: 

            # Setting Necessary Data Loader
            if mode == 'train' :
                dataloader = train_loader
            else :
                dataloader = valid_loader
                all_preds, all_actual = np.array([]), np.array([])

            # Creating Loop
            loop = tqdm(enumerate(dataloader), total = len(dataloader), leave = False)
            if mode == 'train' :
                model.train()
            else :
                model.eval()
                eval_loss = 0.0

            # Iterating over batches
            for batch, (head, body, stance) in loop :

                outputs = model(head.to(device), body.to(device))
                loss = criterion(outputs.float(), stance.to(device).long())

                if mode == 'train':
                    writer.add_scalar('Training Epoch Loss', loss.item(), step)
                    step += 1

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update progress bar
                    loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                    loop.set_postfix(loss = loss.item())
                
                    running_loss += loss.item()
                    """
                    #running_accuracy += ((torch.argmax(outputs, dim = 1) == stance.to(device)).sum().item()) / BATCH_SIZE
                    if batch %10 == 0:
                        bt_loss = running_loss
                        writer.add_scalar('Running Loss', running_loss / 10, epoch * len(dataloader) + batch)
                        bt_loss = 0.0
                        #writer.add_scalar('Running Accuracy', running_accuracy / 10, epoch * len(dataloader) + batch)
                        
                    #running_accuracy = 0
                    """
                else : # for evaluation mode
                    eval_loss += loss.item()
                    predictions = torch.argmax(outputs, dim = 1)
                    all_preds = np.append(all_preds, predictions.cpu().detach().numpy())
                    all_actual = np.append(all_actual, stance.cpu().detach().numpy())
            if mode == 'train':
                if save_model :
                            checkpoint = {
                                'state_dict' : model.state_dict(),
                                'optimizer' : optimizer.state_dict(),
                                'step' : step
                            }
                            save_checkpoint(checkpoint, model_path)
                epoch_loss = running_loss / len(dataloader)
                writer.add_scalar('Epoch Loss', epoch_loss, epoch+1)
                epoch_loss = 0.0
            else : # for evaluation mode
                epoch_eval_loss = eval_loss / len(dataloader)
                print(f'Evaluation Loss for epoch {epoch+1} : {epoch_eval_loss}')
                writer.add_scalar('Evaluation Loss', epoch_eval_loss, epoch+1)
                epoch_eval_loss = 0.0
                get_scores(all_preds, all_actual)
    print("\n")

def get_scores(predicted, actual) :

    get_labels = np.vectorize(lambda t: LABELS[int(t)])
    predicted = get_labels(predicted)
    actual = get_labels(actual)

    print("Scores on the val set")
    report_score(actual,predicted)
    print("")

if __name__ == '__main__' :

    # Getting Training Data
    #train_stances_path = 'fnc-1/train_stances.csv'
    train_stances_path = 'fnc-1/small_stances.csv'
    train_bodies_path = 'fnc-1/train_bodies.csv'
    base_dir = 'body-keys'

    stances_df = pd.read_csv(train_stances_path)
    body_df = pd.read_csv(train_bodies_path)

    # Get tokenizer
    tokenizer_class, pretrained_weights = (hf.DistilBertTokenizer, 'distilbert-base-uncased')
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    # Get head and body transformers
    head_trans_class, body_trans_class = (hf.DistilBertModel, hf.DistilBertModel)
    head_trans = head_trans_class.from_pretrained(pretrained_weights)
    body_trans = body_trans_class.from_pretrained(pretrained_weights)

    # Get LR classifier
    logistic_classifier = LogisticRegression()

    # Splitting into training and validation data
    train_ids, valid_ids = generate_train_valid_splits(stances_df, training = 0.05)

    train_dataset = News(stances_df.iloc[train_ids].reset_index(drop=True), body_df, tokenizer)
    valid_dataset = News(stances_df.iloc[valid_ids].reset_index(drop=True), body_df, tokenizer)

    train_dataloader = transformer_loader(train_dataset,
                            batch_size = BATCH_SIZE, 
                            num_workers = NUM_WORKERS,
                            shuffle = True,
                            pin_memory = True)

    valid_dataloader = transformer_loader(valid_dataset,
                            batch_size = BATCH_SIZE, 
                            num_workers = NUM_WORKERS,
                            shuffle = True,
                            pin_memory = True)


    # use label.item()
    # import IPython ; IPython.embed() ; exit(1)
    
    train_model(train_dataloader, valid_dataloader, head_trans, body_trans, logistic_classifier, load_model = LOAD_MODEL, save_model = SAVE_MODEL, num_epochs = NUM_EPOCHS)
