import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('hiring.csv')
df['experience'].fillna(0, inplace = True)
df['test_score'].fillna(df['test_score'].mean(), inplace = True)
# print(df)
X = df.iloc[:, :3] # zmienne niezależne, objasniajace
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
                 'six':6, 'seven':7, 'eight':8, 'nine':9, 'nan':0,
                 'zero':0, 'eleven':11, 'twelve':12, 'ten':10, 0:0}
    return word_dict[word]
X['experience'] = X['experience'].apply(lambda x: convert_to_int(x))
# print(X['experience'])
y=df.iloc[:,-1] # zmienna zależna

regressor = LinearRegression()
# dopasowujemy
regressor.fit(X, y)
# zamykamy w pliku pickle
pickle.dump(regressor, open('model.pkl', 'wb'))

