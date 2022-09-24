import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
import math
import mplfinance as mpf

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

########     Getting the Data      ######
csvFileName="DataMinuteTwttr.csv"

df=pd.read_csv(csvFileName, index_col="Unnamed: 0")
#Separate dates for future plotting
Data_dates = df.index
Data_dates=pd.to_datetime(Data_dates,utc=True)
Data_dates=Data_dates.tz_localize(None)
#....... dates .....#
Dates_To_Use_To_Forcast=Data_dates[Data_dates.shape[0]-40:]

#Getting the columns name
cols = list(df)[0:5]
#New dataframe with only training data - 5 columns
df_forcasting = df[cols].astype(float)


#####       Scaling data     #####

scaler = MinMaxScaler()

scaler = scaler.fit(df_forcasting)
DS_raw_scaled = scaler.transform(df_forcasting)

####   getting the 40 most present data  ####

Batch_to_predict=DS_raw_scaled[DS_raw_scaled.shape[0]-40:]
#....... databatch .....#
Batch_to_predict=np.reshape(Batch_to_predict,(1,40,5))


##############      Retriving model       ########


model = keras.models.load_model("Model/Model_Twttr_0_0")


##########################################
#           Model Forcasting             #
##########################################


N_Days_to_predict=14
Prediction_Saved=[]
#testingX=np.array(testingX)
######    Generating forcast data   ######
for i in range(N_Days_to_predict):
  prediction = model.predict(Batch_to_predict) #the input is a 30 units of time batch
  prediction_Reshaped=np.reshape(prediction,(1,1,5))
  Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
  Batch_to_predict=np.delete(Batch_to_predict,0,1)
  #print(Batch_to_predict.shape)
  Prediction_Saved.append(prediction_Reshaped[0])


predict_Open=[]
predict_High=[]
predict_Low=[]
predict_Close=[]
predict_Volume=[]

#Splitting data with scaling back
for i in range(N_Days_to_predict):
  y_pred_future = Prediction_Saved[i]
  predict_Open.append(y_pred_future[0][0])
  predict_High.append(y_pred_future[0][1])
  predict_Low.append(y_pred_future[0][2])
  predict_Close.append(y_pred_future[0][3])
  predict_Volume.append(y_pred_future[0][4])
  
####################################### 
#      Getting the candle chart       #
#######################################
#######    Generating forcasted dates    #######

lastTimedate=Dates_To_Use_To_Forcast[Dates_To_Use_To_Forcast.shape[0]-1:]
lastTimedate=str(lastTimedate[0])

lastTimedate=pd.Timestamp(str(lastTimedate))

Forcasted_Dates=[]
for i in range(0,N_Days_to_predict):
  lastTimedate=np.datetime64(lastTimedate) + np.timedelta64(1, 'm')
  Forcasted_Dates.append(pd.Timestamp(np.datetime64(lastTimedate)))
  


#####       Scaling Back     #####
AllPrediction_DS_scaled_Back=[]
for i in Prediction_Saved:
  AllPrediction_DS_scaled_Back.append(scaler.inverse_transform(i))
  
  
predict_Open=[]
predict_High=[]
predict_Low=[]
predict_Close=[]
predict_Volume=[]

#Splitting data with scaling back
for i in range(N_Days_to_predict):
  y_pred_future = AllPrediction_DS_scaled_Back[i]
  predict_Open.append(y_pred_future[0][0])
  predict_High.append(y_pred_future[0][1])
  predict_Low.append(y_pred_future[0][2])
  predict_Close.append(y_pred_future[0][3])
  predict_Volume.append(y_pred_future[0][4])
  
  
#--------  data shape it's (x days, 5 columns)
# Convert timestamp to date
df_forecast = pd.DataFrame({'Open':predict_Open,'High':predict_High, 'Low':predict_Low,'Close':predict_Close,'Volume':predict_Volume}, index=Forcasted_Dates)

df_forecast.index.name="Date"

title_chart="Twtr"
mpf.plot(df_forecast, type='candle',title=title_chart, style='charles')