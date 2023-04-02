from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


class Forcast_Data:
  def __init__(self,Model_Path,data_frame_Path):
    self.csvFileName=data_frame_Path
    self.model = keras.models.load_model(Model_Path)
    
  def ToForcast(self,n_units_to_predict,dateFromForcast):
  ########     Getting the Data     ######
    N_units_to_predict=n_units_to_predict
    #Model_Path=model_Path
    df=pd.read_csv(self.csvFileName,index_col=0)
    backDaysRef=20
    #Separate dates for future plotting
    Data_dates = df.index
    Data_dates=pd.to_datetime(Data_dates,utc=True)
    Data_dates=Data_dates.tz_localize(None)
    #....... dates .....#
    Dates_To_Use_To_Forcast=Data_dates[Data_dates.get_loc(dateFromForcast)-(backDaysRef-1):Data_dates.get_loc(dateFromForcast)+1]
    
    #print(Dates_To_Use_To_Forcast)
    
    Columns_N=df.shape[1]
    ColumToforcast=0
    #Getting the columns name
    cols = list(df)[0:Columns_N]
    #New dataframe with only training data - 5 columns
    df_forcasting = df[cols].astype(float)

    #####       Scaling data     #####

    scaler = MinMaxScaler()

    scaler = scaler.fit(df_forcasting)
    DS_raw_scaled = scaler.transform(df_forcasting)
    
    ####    Scaling only the close colum   ####
    print(dateFromForcast)
    df_forcasting_close=df_forcasting[cols[ColumToforcast]].to_numpy()
    #df_forcasting_close=df_forcasting[cols[3]].to_numpy()
    #df_forcasting_close=df_forcasting[cols[8]].to_numpy()
    df_forcasting_close=df_forcasting_close.reshape(len(df_forcasting[cols[ColumToforcast]].to_numpy()),-1)
    #df_forcasting_close=df_forcasting_close.reshape(len(df_forcasting[cols[3]].to_numpy()),-1)
    #df_forcasting_close=df_forcasting_close.reshape(len(df_forcasting[cols[8]].to_numpy()),-1)
    
    scaler_Close = MinMaxScaler()

    scaler_Close = scaler_Close.fit(df_forcasting_close)

    

    ####   getting the 120 most present data  ####

    Batch_to_predict=DS_raw_scaled[df.index.get_loc(dateFromForcast)-(backDaysRef-1):df.index.get_loc(dateFromForcast)+1]
    #Batch_Real_Y_NonScaled=df_forcasting[df.index.get_loc(dateFromForcast)-1:df.shape[0]-1]
    Batch_Real_Y_NonScaled=df_forcasting[df.index.get_loc(dateFromForcast)-(backDaysRef-2):df.index.get_loc(dateFromForcast)+2]
    #print("Batch_to_predict_Y_NonScaled: {}".format(Batch_to_predict))
    #print("Batch_Real_Y_NonScaled: {}".format(Batch_Real_Y_NonScaled))
    
    Batch_Real_Y_NonScaled=np.array(Batch_Real_Y_NonScaled)
    #....... databatch .....#
    Batch_to_predict=np.reshape(Batch_to_predict,(1,backDaysRef,Columns_N))


    ##############      Retriving model       ########


    #model = keras.models.load_model(Model_Path)


    ##########################################
    #           Model Forcasting             #
    ##########################################


    N_Days_to_predict=N_units_to_predict
    Prediction_Saved=[]
    #testingX=np.array(testingX)
    ######    Generating forcast data   ######
    for i in range(N_Days_to_predict):
      prediction = self.model.predict(Batch_to_predict) #the input is a 120 units of time batch
      #print(Batch_to_predict.shape)
      Prediction_Saved.append(prediction)

    #####       Scaling Back close before prediction    #####
    #print("this is Batch_to_predict.shape"+str(Batch_to_predict.shape))
    AllPrediction_DS_scaled_Back_1=scaler.inverse_transform(Batch_to_predict[0])
    #print(AllPrediction_DS_scaled_Back_1[119][3])
    
    #####       Scaling Back     #####
    AllPrediction_DS_scaled_Back=[]
    for i in Prediction_Saved:
      AllPrediction_DS_scaled_Back.append(scaler_Close.inverse_transform(i))
    
    
    Forcast_Close=0
    
    #Splitting data with scaling back
    for i in range(N_Days_to_predict):
      y_pred_future = AllPrediction_DS_scaled_Back[i]
      Forcast_Close=y_pred_future[0][0]
      
    #######    Generating forcasted dates    #######

    lastTimedate=Dates_To_Use_To_Forcast[Dates_To_Use_To_Forcast.shape[0]-1:]
    lastTimedate=str(lastTimedate[0])

    lastTimedate=pd.Timestamp(str(lastTimedate))

    Forcasted_Dates=[]
    for i in range(0,N_Days_to_predict):
      lastTimedate=np.datetime64(lastTimedate) + np.timedelta64(1, 'D')
      Forcasted_Dates.append(pd.Timestamp(np.datetime64(lastTimedate)))

      
    #####        splitting Real y    #####
    
    
    Real_Y_Close=0
    Real_Y_current=0
    
    #Splitting data  real Y
    #print("The shape of Batch_Real_Y_NonScaled: " + str(Batch_Real_Y_NonScaled.shape))
    try:
      Real_Y_Close=df_forcasting[df.index.get_loc(dateFromForcast)+1:df.index.get_loc(dateFromForcast)+2]
      Real_Y_Close=Real_Y_Close["Close"][0]
    except:
      Real_Y_Close=df_forcasting[df.index.get_loc(dateFromForcast):df.index.get_loc(dateFromForcast)+1]
      Real_Y_Close=Real_Y_Close["Close"][0]
    
    Real_Y_current=df_forcasting[df.index.get_loc(dateFromForcast):df.index.get_loc(dateFromForcast)+1]
    Real_Y_current=Real_Y_current["Close"][0]
    #Real_Y_current=Batch_Real_Y_NonScaled[Batch_Real_Y_NonScaled.shape[0]-1][ColumToforcast]

    print(type(Forcast_Close))
    
    return Real_Y_current,Forcast_Close,Real_Y_Close
    
  
  def to_forcastInLoop(self):
    ##########################################
    #           Model Forcasting             #
    ##########################################


    N_Days_to_predict=N_units_to_predict
    Prediction_Saved=[]
    #testingX=np.array(testingX)
    ######    Generating forcast data   #####
    for i in DS_finished_X:
      BackUnits_Reshaped=np.reshape(i,(1,120,4))
      prediction = model.predict(BackUnits_Reshaped) #the input is a 30 units of time batch
      prediction_Reshaped=np.reshape(prediction,(1,1,5))
      #Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
      #Batch_to_predict=np.delete(Batch_to_predict,0,1)
      #print(Batch_to_predict.shape)
      Prediction_Saved.append(prediction_Reshaped[0])