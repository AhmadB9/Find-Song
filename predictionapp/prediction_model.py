import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.api.models import load_model
# Load the Excel file
dataset= pd.read_csv('dataset.csv')
dataset.head()

song_list=list(set(dataset['Song']))
labelencoder=LabelEncoder()
labelencoder.fit_transform(song_list)


model =load_model('CNN_BLSTM_model_0.9.h5',compile=False)

