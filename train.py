import pandas as pd
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.processsing import FakeNews, get_loader, save_checkpoint, load_checkpoint
from utils.score import report_score, LABELS
from utils.generate_splits import generate_train_valid_splits
from model import Classifier



#train_stances_path = 'fnc-1/train_stances.csv'
train_stances_path = 'fnc-1/small_stances.csv'
train_bodies_path = 'fnc-1/train_bodies.csv'
base_dir = 'body-keys'

stances_df = pd.read_csv(train_stances_path)
body_df = pd.read_csv(train_bodies_path)

train_ids, valid_ids = generate_train_valid_splits(stances_df, training = 0.1)


BATCH_SIZE = 8
NUM_WORKERS = 8



train_dataset = FakeNews(stances_df.iloc[train_ids].reset_index(drop=True), body_df, freq_threshold = 5)
train_dataloader = get_loader(train_dataset,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True)

valid_dataset = FakeNews(stances_df.iloc[valid_ids].reset_index(drop=True), body_df, freq_threshold = 5)
valid_dataloader = get_loader(valid_dataset,
                        batch_size = BATCH_SIZE, 
                        num_workers = 8,
                        shuffle = True,
                        pin_memory = True)


# use label.item()

def train(dataloader, load_model = False, save_model = True, num_epochs = 2) :
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    load_model = load_model
    save_model = save_model

    embed_size = 256
    hidden_size = 256
    head_vocab_size = len(train_dataset.head_vocab)
    body_vocab_size = len(train_dataset.body_vocab)
    num_layers = 1
    learning_rate = 9e-4
    num_epochs = num_epochs

    # For tensorboard
    writer = SummaryWriter('runs/full')
    step = 0

    # Initialize Model
    model = Classifier(embed_size, hidden_size
    , head_vocab_size, body_vocab_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index= train_dataset.head_vocab.stoi['<PAD>'])
    optimizer = optim.Adagrad(model.parameters(), lr = learning_rate)

    # sample data for plotting model graph
    examples = iter(dataloader)
    ex_head, ex_body, _ = examples.next()


    if load_model :
        model, optimizer, step = load_checkpoint(torch.load('my_checkpoint.pth.tar'), model, optimizer)
        return model

    # Plotting model graph on tensorboard
    # writer.add_graph(model, [ex_head.to(device), ex_body.to(device)])
    # writer.close()
    # sys.exit()

    model.train()


    running_loss = 0.0
    running_accuracy = 0
    for epoch in range(num_epochs) :
        if save_model :
            checkpoint = {
                'state_dict' : model.state_dict(),
                'optmizer' : optimizer.state_dict(),
                'step' : step
            }
            save_checkpoint(checkpoint)

        loop = tqdm(enumerate(dataloader), total = len(dataloader), leave = False)

        

        for batch, (head, body, stance) in loop :
            outputs = model(head.to(device), body.to(device))

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

    writer = SummaryWriter('runs/evals')
    step = 0

    criterion = nn.CrossEntropyLoss(ignore_index= train_dataset.head_vocab.stoi['<PAD>'])

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
    for batch, (head, body, stance) in enumerate(dataloader) :
        predictions = model(head.to(device), body.to(device))
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
  
    model = train(train_dataloader, load_model = False, save_model = True, num_epochs = 2)
    get_predictions(valid_dataloader, model)