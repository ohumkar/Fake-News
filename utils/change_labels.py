import pandas as pd


train_stances = pd.read_csv('fnc-1/train_stances.csv')
train_bodies = pd.read_csv('fnc-1/train_bodies.csv')



LABELS = {'unrelated': 0, 'discuss': 1, 'disagree': 2, 'agree': 3}

def get_val(row): 
    #bid = row['Body ID']
    val = train_stances['Stance'][train_stances['Body ID'] == row]
    print('heey')
    print(val)
    return val

# train_stances['Stance'] = train_stances['Stance'].apply(lambda x: LABELS[x])
# # train_bodies['Stance'] = train_bodies.apply(get_val)
# train_bodies['Stance'] = train_bodies['Body ID'].apply(lambda x: get_val(x))
# train_stances.to_csv('fnc-1/changed/ntrain_headings.csv')
# train_bodies.to_csv('fnc-1/changed/ntrain_bodies.csv')

def get_body(bid) :
    body  = train_bodies['articleBody'][train_bodies['Body ID'] == bid].item()
    return body

train_stances['Stance'] = train_stances['Stance'].apply(lambda x: LABELS[x])
print(train_bodies.columns)
train_stances['Body'] = train_stances['Body ID'].apply(lambda x: get_body(x))
print(train_stances.head())
# train_stances.to_csv('fnc-1/train_data.csv')
data = pd.DataFrame({'Body ID': train_stances['Body ID'], 'Body': train_stances['Body'], 'Stance': train_stances['Stance']})
train_stances = train_stances.drop(['Body'], axis = 1)

train_stances['Headline'] = train_stances['Headline'].astype(str)
train_stances['Body ID'] = pd.to_numeric(train_stances['Body ID'], downcast='integer')
train_stances['Stance'] = pd.to_numeric(train_stances['Stance'], downcast='integer')
# train_stances.to_csv('fnc-1/changed/ntrain_headings.csv')

data['Body'] = data['Body'].astype(str)
data['Body ID'] = pd.to_numeric(data['Body ID'], downcast='integer')
data['Stance'] = pd.to_numeric(data['Stance'], downcast='integer')
data.to_csv('fnc-1/changed/ntrain_bodies.csv.gz', compression = 'gzip')