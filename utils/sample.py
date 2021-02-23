import pandas as pd

headings = pd.read_csv('fnc-1/train_stances.csv')
bodies = pd.read_csv('fnc-1/train_bodies.csv')

print(headings.head())
print('\n\n')
print(headings.iloc[:20])
# print(bodies.head())

# print(headings.shape)
# print(bodies.shape)

# new = headings[11000:12000]
# new.to_csv('fnc-1/small_test.csv')