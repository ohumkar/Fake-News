import pandas as pd
import numpy as np
import os
from tokenizers import (BertWordPieceTokenizer)

# !python3 -m pip install tokenizers
# !wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

# from tokenizers import (BertWordPieceTokenizer)

tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
# tokenizer.encode(text).tokens # --> list of tokens



def create_splits(dataframe, split_path, n_splits = 10) :
    """
    Should i reset index ?
    """
    length = int(dataframe.shape[0] / int(n_splits))
    for i in range(n_splits) :
        frame = dataframe.iloc[i*length:(i+1)*length]
        if i == n_splits-1 :
            frame = dataframe.iloc[i*length:]
        name = split_path + f'split-{i}.csv.gz'
        if os.path.exists(name) :
            print(f'File {name} already exits! Exiting...')
            return 0
        print(frame.columns)
        frame = frame.drop(['Unnamed: 0'], axis = 1)
        frame.to_csv(name, index = False, compression = 'gzip')
    
def get_tokens(text) :
    return np.array(tokenizer.encode(text).tokens) # --> list of tokens

def get_term_frequencies(token_list) :
    term_freqs = {}
    for token in token_list :
        if token in term_freqs :
            term_freqs[token] += 1
        else :
            term_freqs[token] = 1
    return term_freqs

def class_term_occ(dataframe, column_name = 'Headline'):
    dataframe['Tokens'] = dataframe[column_name].apply(get_tokens)
    dataframe = dataframe.reset_index()
    all_tokens = []
    for i in range(dataframe.shape[0]) :
        all_tokens.extend(dataframe['Tokens'][i])   # set of all tokens for class i
    term_occ = get_term_frequencies(all_tokens)
    tots = sum(term_occ.values())
    return term_occ, tots

def calculate_clfr(self_total, rem_total, class_tok, rem_class_tok, all_toks): 
    total = self_total
    total_rem = sum(rem_total)
    rem_class_term_occ = rem_class_tok # list of remaining class term occs
    class_clfr = dict.fromkeys(all_toks.keys()) # class specific clfr scores for all tokens
   
    for tok in all_toks:
        try: 
            term_occ = class_tok[tok]
        except KeyError:
            term_occ = 0
        
        rem_term_occ = 0
        for class_term_occ in rem_class_term_occ :
            if tok in class_term_occ:
                rem_term_occ += class_term_occ[tok]
        clfr = np.log((((1 + term_occ) * total_rem) / ((1 + rem_term_occ) * total)) +1)
        class_clfr[tok] = clfr
    
    return class_clfr

def term_freq_calculator(dataframe, column_name) :
    
    dataframe['Tokens'] = dataframe[column_name].apply(get_tokens)
    dataframe = dataframe.reset_index()
    all_token_text = []
    for i in range(dataframe.shape[0]) :
        all_token_text.extend(dataframe['Tokens'][i]) 
    return get_term_frequencies(all_token_text)

        
def get_clfd(clfr_list, all_class_clfr) :
    clfd = {}
    for tok in clfr_list :
        clfd_val = max([x[tok] for x in all_class_clfr]) - min([x[tok] for x in all_class_clfr])
        clfd[tok] = clfd_val
    return clfd


def calculate_clfd_vec(dataframe, column_name = 'Headline') : 
    unrelated_dataframe = dataframe[dataframe['Stance'] == 0]
    discuss_dataframe = dataframe[dataframe['Stance'] == 1]
    disagree_dataframe = dataframe[dataframe['Stance'] == 2]
    agree_dataframe = dataframe[dataframe['Stance'] == 3] 

    unrelated_occ, unrelated_tots = class_term_occ(unrelated_dataframe, column_name)
    discuss_occ, discuss_tots = class_term_occ(discuss_dataframe, column_name)
    disagree_occ, disagree_tots = class_term_occ(disagree_dataframe, column_name)
    agree_occ, agree_tots = class_term_occ(agree_dataframe, column_name)

    
    # all_tokens = set(unrelated_occ.keys()).update(set(discuss_occ.keys())).update(set(disagree_occ.keys())).update(set(agree_occ.keys()))
    all_tokens = set(unrelated_occ.keys())
    all_tokens.update(discuss_occ.keys())
    all_tokens.update(set(disagree_occ.keys()))
    all_tokens.update(set(agree_occ.keys()))
    all_class_tokens = dict.fromkeys(list(all_tokens))
       

    unrelated_clfr = calculate_clfr(unrelated_tots, 
                                    (discuss_tots, disagree_tots, agree_tots), 
                                    unrelated_occ, 
                                    [discuss_occ, disagree_occ, agree_occ],
                                    all_class_tokens)   
    discuss_clfr = calculate_clfr(discuss_tots, 
                                    (unrelated_tots, disagree_tots, agree_tots), 
                                    discuss_occ, 
                                    [unrelated_occ, disagree_occ, agree_occ],
                                    all_class_tokens)
    disagree_clfr = calculate_clfr(disagree_tots, 
                                    (discuss_tots, unrelated_tots, agree_tots), 
                                    disagree_occ, 
                                    [discuss_occ, unrelated_occ, agree_occ],
                                    all_class_tokens)
    agree_clfr = calculate_clfr(agree_tots, 
                                    (discuss_tots, disagree_tots, unrelated_tots), 
                                    agree_occ, 
                                    [discuss_occ, disagree_occ, unrelated_occ],
                                    all_class_tokens)
                                                            
    clfd_vec = get_clfd(unrelated_clfr, [unrelated_clfr, discuss_clfr, disagree_clfr, agree_clfr])

    return clfd_vec

def calculate_tf_clfd(clfd, tf) :
    tf_clfd = {}
    for tok in clfd :
        tf_clfd[tok] = clfd[tok] * tf[tok]
    return tf_clfd

def save_csv(clfd_dict, path) :
    clfd_df = pd.Series(clfd_dict).to_frame()
    clfd_df = pd.DataFrame(clfd_df)
    clfd_df[0].to_csv(path)


if __name__ == '__main__': 
    print('Done')
    headings_path = 'frames/headings/'
    bodies_path = 'frames/bodies/'
    store_headings_path = 'clfd-frames/headings'
    store_bodies_path = 'clfd-frames/bodies'

    """
    # To create dataframe splits
    create_splits(pd.read_csv('fnc-1/changed/ntrain_headings.csv'), 'frames/headings/', n_splits = 5)
    create_splits(pd.read_csv('fnc-1/changed/ntrain_bodies.csv'), 'frames/bodies/', n_splits = 5)
    """

    """
    # FOR HEADINGS get relevent file names 
    heading_files = os.listdir(headings_path)
    heading_paths = [os.path.join(headings_path, x) for x in heading_files]
    """

    # FOR BODIES get relevent file names 
    body_files = os.listdir(bodies_path)
    body_paths = [os.path.join(bodies_path, x) for x in body_files]

    xx = pd.read_csv('frames/bodies/split-2.csv.gz')
    
    # dictionary to store all clfd values
    clfd_vec = {}
    term_freqs = {}
    for file_path in body_paths :
        file_df = pd.read_csv(file_path)
        tf = term_freq_calculator(file_df, 'Body')
        vec = calculate_clfd_vec(file_df, 'Body')

        # Average out clfd values across datasets
        for token in vec :
            if token in clfd_vec :
                clfd_vec[token] = (clfd_vec[token] + vec[token]) / 2
            else : 
                clfd_vec[token] = vec[token]
        
        # Average out tf values across datasets
        for token in tf :
            if token in term_freqs :
                term_freqs[token] = (term_freqs[token] + tf[token]) / 2
            else : 
                term_freqs[token] = tf[token]
    

    tf_clfd = calculate_tf_clfd(clfd_vec, term_freqs)
    save_csv(tf_clfd, 'clfd/body-tf_clfd.csv')

    """
    TO OPEN AS DICTIONARY AFTER SAVING AS CSV
    
    new_dict = clfd_df[0].T.to_dict()
    print(new_dict)
    """
    # clfd_df.to_csv('clfd/heading-clfd.csv.gz', compression = 'gzip')

    