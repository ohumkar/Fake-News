import random 
import pandas as pd
import os

def change_index(dir = '/fnc-1/train_stances.csv', base_dir = 'body-keys') :
    filename = dir.split('/')[-1]

    dataframe = pd.read_csv(dir)
    print(dataframe.head())
    dataframe.set_index('Body ID', inplace = True)
    file_path = base_dir+"/"+filename.split('.')[0]
    if not (os.path.exists(file_path)) :
        os.mkdir(base_dir)
        dataframe.to_csv(base_dir+"/"+filename.split('.')[0])
    return dataframe





def generate_splits(dataset, training = 0.8, base_dir = "splits") :
    r = random.Random()
    r.seed(777)

    all_ids = dataset
