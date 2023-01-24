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
  def __init__(self,data_frame_Path):
    self.csvFileName=data_frame_Path
    
  def ToForcast(self,n_units_to_predict,model_Path,dateFromForcast):
  ########     Getting the Data      ######
    N_units_to_predict=n_units_to_predict
    Model_Path=model_Path
    df=pd.read_csv(self.csvFileName,index_col=0)
    backDaysRef=120
    #Separate dates for future plotting
    Data_dates = df.index
    print(Data_dates[0:5])
    Data_dates=pd.to_datetime(Data_dates,utc=True)
    Data_dates=Data_dates.tz_localize(None)
    #....... dates .....#
    #Dates_To_Use_To_Forcast=Data_dates[Data_dates.shape[0]-40:]
    Dates_To_Use_To_Forcast=Data_dates[df.index.get_loc(dateFromForcast)-backDaysRef:df.index.get_loc(dateFromForcast)]

    Columns_N=df.shape[1]
    #Getting the columns name
    cols = list(df)[0:Columns_N]
    #New dataframe with only training data - 5 columns
    df_forcasting = df[cols].astype(float)


    #####       Scaling data     #####

    scaler = MinMaxScaler()

    scaler = scaler.fit(df_forcasting)
    DS_raw_scaled = scaler.transform(df_forcasting)

    ####   getting the 40 most present data  ####

    Batch_to_predict=DS_raw_scaled[df.index.get_loc(dateFromForcast)-backDaysRef:df.index.get_loc(dateFromForcast)]
    Batch_Real_Y_NonScaled=df_forcasting[df.index.get_loc(dateFromForcast)-1:df.shape[0]-1]
    print("Batch_Real_Y_NonScaled: {}".format(Batch_Real_Y_NonScaled))
    
    Batch_Real_Y_NonScaled=np.array(Batch_Real_Y_NonScaled)
    #....... databatch .....#
    Batch_to_predict=np.reshape(Batch_to_predict,(1,backDaysRef,Columns_N))


    ##############      Retriving model       ########


    model = keras.models.load_model(Model_Path)


    ##########################################
    #           Model Forcasting             #
    ##########################################


    N_Days_to_predict=N_units_to_predict
    Prediction_Saved=[]
    #testingX=np.array(testingX)
    ######    Generating forcast data   ######
    for i in range(N_Days_to_predict):
      prediction = model.predict(Batch_to_predict) #the input is a 30 units of time batch
      prediction_Reshaped=np.reshape(prediction,(1,1,Columns_N))
      Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
      Batch_to_predict=np.delete(Batch_to_predict,0,1)
      #print(Batch_to_predict.shape)
      Prediction_Saved.append(prediction_Reshaped[0])

    #####       Scaling Back     #####
    AllPrediction_DS_scaled_Back=[]
    for i in Prediction_Saved:
      AllPrediction_DS_scaled_Back.append(scaler.inverse_transform(i))
    
    predict_Open= []
    predict_High= []
    predict_Low= []
    predict_Close= []
    predict_Volume= []
    predict_DayNumber= []
    predict_FFT_Mag_Open_10= []
    predict_FFT_Angl_Open_10= []
    predict_FFT_Mag_Open_50= []
    predict_FFT_Angl_Open_50= []
    predict_FFT_Mag_Open_100= []
    predict_FFT_Angl_Open_100= []
    predict_FFT_Mag_High_10= []
    predict_FFT_Angl_High_10= []
    predict_FFT_Mag_High_50= []
    predict_FFT_Angl_High_50= []
    predict_FFT_Mag_High_100= []
    predict_FFT_Angl_High_100= []
    predict_FFT_Mag_Low_10= []
    predict_FFT_Angl_Low_10= []
    predict_FFT_Mag_Low_50= []
    predict_FFT_Angl_Low_50= []
    predict_FFT_Mag_Low_100= []
    predict_FFT_Angl_Low_100= []
    predict_FFT_Mag_Close_10= []
    predict_FFT_Angl_Close_10= []
    predict_FFT_Mag_Close_50= []
    predict_FFT_Angl_Close_50= []
    predict_FFT_Mag_Close_100= []
    predict_FFT_Angl_Close_100= []
    predict_FFT_Mag_Volume_10= []
    predict_FFT_Angl_Volume_10= []
    predict_FFT_Mag_Volume_50= []
    predict_FFT_Angl_Volume_50= []
    predict_FFT_Mag_Volume_100= []
    predict_FFT_Angl_Volume_100= []

    #Splitting data with scaling back
    for i in range(N_Days_to_predict):
      y_pred_future = AllPrediction_DS_scaled_Back[i]
      predict_Open.append(y_pred_future[0][0])
      predict_High.append(y_pred_future[0][1])
      predict_Low.append(y_pred_future[0][2])
      predict_Close.append(y_pred_future[0][3])
      predict_Volume.append(y_pred_future[0][4])
      predict_DayNumber.append(y_pred_future[0][5])
      predict_FFT_Mag_Open_10.append(y_pred_future[0][6])
      predict_FFT_Angl_Open_10.append(y_pred_future[0][7])
      predict_FFT_Mag_Open_50.append(y_pred_future[0][8])
      predict_FFT_Angl_Open_50.append(y_pred_future[0][9])
      predict_FFT_Mag_Open_100.append(y_pred_future[0][10])
      predict_FFT_Angl_Open_100.append(y_pred_future[0][11])
      predict_FFT_Mag_High_10.append(y_pred_future[0][12])
      predict_FFT_Angl_High_10.append(y_pred_future[0][13])
      predict_FFT_Mag_High_50.append(y_pred_future[0][14])
      predict_FFT_Angl_High_50.append(y_pred_future[0][15])
      predict_FFT_Mag_High_100.append(y_pred_future[0][16])
      predict_FFT_Angl_High_100.append(y_pred_future[0][17])
      predict_FFT_Mag_Low_10.append(y_pred_future[0][18])
      predict_FFT_Angl_Low_10.append(y_pred_future[0][19])
      predict_FFT_Mag_Low_50.append(y_pred_future[0][20])
      predict_FFT_Angl_Low_50.append(y_pred_future[0][21])
      predict_FFT_Mag_Low_100.append(y_pred_future[0][22])
      predict_FFT_Angl_Low_100.append(y_pred_future[0][23])
      predict_FFT_Mag_Close_10.append(y_pred_future[0][24])
      predict_FFT_Angl_Close_10.append(y_pred_future[0][25])
      predict_FFT_Mag_Close_50.append(y_pred_future[0][26])
      predict_FFT_Angl_Close_50.append(y_pred_future[0][27])
      predict_FFT_Mag_Close_100.append(y_pred_future[0][28])
      predict_FFT_Angl_Close_100.append(y_pred_future[0][29])
      predict_FFT_Mag_Volume_10.append(y_pred_future[0][30])
      predict_FFT_Angl_Volume_10.append(y_pred_future[0][31])
      predict_FFT_Mag_Volume_50.append(y_pred_future[0][32])
      predict_FFT_Angl_Volume_50.append(y_pred_future[0][33])
      predict_FFT_Mag_Volume_100.append(y_pred_future[0][34])
      predict_FFT_Angl_Volume_100.append(y_pred_future[0][35])
      
    ####################################### 
    #      Getting the candle chart       #
    #######################################
    #######    Generating forcasted dates    #######

    lastTimedate=Dates_To_Use_To_Forcast[Dates_To_Use_To_Forcast.shape[0]-1:]
    lastTimedate=str(lastTimedate[0])

    lastTimedate=pd.Timestamp(str(lastTimedate))

    Forcasted_Dates=[]
    for i in range(0,N_Days_to_predict):
      lastTimedate=np.datetime64(lastTimedate) + np.timedelta64(1, 'D')
      Forcasted_Dates.append(pd.Timestamp(np.datetime64(lastTimedate)))

      
    #####        splitting Real y    #####
    
    Real_Y_Open=[]
    Real_Y_High=[]
    Real_Y_Low=[]
    Real_Y_Close=[]
    Real_Y_Volume=[]
    
    #Splitting data  real Y
    
    for i in Batch_Real_Y_NonScaled:
      Real_Y_Close.append(i[1])
      
      
    print(predict_Close)
    print(Real_Y_Close)
    plt.plot(predict_Close,'--g*')
    plt.plot(Real_Y_Close, '--ro')
    #plt.plot(predict_Close,'g', Real_Y_Close, 'r')
    plt.show()
      
      
    """"
    #--------  data shape it's (x days, 5 columns)
    # Convert timestamp to date
    df_forecast = pd.DataFrame({'Open':predict_Open,'High':predict_High, 'Low':predict_Low,'Close':predict_Close,'Volume':predict_Volume}, index=Forcasted_Dates)

    df_forecast.index.name="Date"

    title_chart="Twtr bid"
    return mpf.plot(df_forecast, type='candle',title=title_chart, style='charles')"""
  
  def to_forcast_close_true_and_forcasted(self,n_units_to_predict,model_Path,dateFromForcast):
    ########     Getting the Data      ######
    n_past=60
    N_units_to_predict=n_units_to_predict
    Model_Path=model_Path
    df=pd.read_csv(self.csvFileName, index_col="Unnamed: 0")
    print(df.shape)
    #Separate dates for future plotting
    Data_dates = df.index
    Data_dates=pd.to_datetime(Data_dates,utc=True)
    Data_dates=Data_dates.tz_localize(None)
    #....... dates .....#
    #Dates_To_Use_To_Forcast=Data_dates[Data_dates.shape[0]-40:]
    Dates_To_Use_To_Forcast=Data_dates[df.index.get_loc(dateFromForcast)-n_past:df.index.get_loc(dateFromForcast)]

    #Getting the columns name
    cols = list(df)[0:6]
    #New dataframe with only training data - 6 columns
    df_forcasting = df[cols].astype(float)


    #####       Scaling data     #####

    scaler = MinMaxScaler()

    scaler = scaler.fit(df_forcasting)
    DS_raw_scaled = scaler.transform(df_forcasting)

    ####   getting the 40 most present data  ####

    Batch_to_predict=DS_raw_scaled[df.index.get_loc(dateFromForcast)-n_past:df.shape[0]]
    Batch_Real_Y_NonScaled=df_forcasting[df.index.get_loc(dateFromForcast)-1:df.shape[0]-1]
    
    print("This is the df.shape[0]:{}".format(df.shape[0]))
    #....... databatch .....#
    #Batch_to_predict=np.reshape(Batch_to_predict,(1,60,6))
    #####      creating batches   ###### 
    DS_finished_X=[]
    
    for i in range(n_past, len(Batch_to_predict) - N_units_to_predict +1):
      DS_finished_X.append(Batch_to_predict[i - n_past:i, 0:Batch_to_predict.shape[1]])
      
      
    
    
    DS_finished_X, Batch_Real_Y_NonScaled=np.array(DS_finished_X), np.array(Batch_Real_Y_NonScaled)
    print("this DS_finished : {}".format(DS_finished_X.shape))
    print("this DS_finished : {}".format(Batch_Real_Y_NonScaled.shape))
    ##############      Retriving model       ########

    model = keras.models.load_model(Model_Path)


    ##########################################
    #           Model Forcasting             #
    ##########################################


    N_Days_to_predict=N_units_to_predict
    Prediction_Saved=[]
    #testingX=np.array(testingX)
    ######    Generating forcast data   #####
    for i in DS_finished_X:
      BackUnits_Reshaped=np.reshape(i,(1,60,6))
      prediction = model.predict(BackUnits_Reshaped) #the input is a 30 units of time batch
      prediction_Reshaped=np.reshape(prediction,(1,1,6))
      #Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
      #Batch_to_predict=np.delete(Batch_to_predict,0,1)
      #print(Batch_to_predict.shape)
      Prediction_Saved.append(prediction_Reshaped[0])

      
    ####################################### 
    #      Getting the candle chart       #
    #######################################
    #######    Generating forcasted dates    #######

    lastTimedate=Dates_To_Use_To_Forcast[Dates_To_Use_To_Forcast.shape[0]-1:]
    lastTimedate=str(lastTimedate[0])

    lastTimedate=pd.Timestamp(str(lastTimedate))

    Forcasted_Dates=[]
    for i in range(0,N_Days_to_predict):
      lastTimedate=np.datetime64(lastTimedate) + np.timedelta64(1, 'h')
      Forcasted_Dates.append(pd.Timestamp(np.datetime64(lastTimedate)))
      


    #####       Scaling Back  predicted    #####
    AllPrediction_DS_scaled_Back=[]
    for i in Prediction_Saved:
      AllPrediction_DS_scaled_Back.append(scaler.inverse_transform(i))
      
      
    predict_Open=[]
    predict_High=[]
    predict_Low=[]
    predict_Close=[]
    predict_Volume=[]

    #Splitting data with scaling back
    for i in range(DS_finished_X.shape[0]):
      y_pred_future = AllPrediction_DS_scaled_Back[i]
      predict_Open.append(y_pred_future[0][0])
      predict_High.append(y_pred_future[0][1])
      predict_Low.append(y_pred_future[0][2])
      predict_Close.append(y_pred_future[0][3])
      predict_Volume.append(y_pred_future[0][4])
      
    #####        splitting Real y    #####
    
    Real_Y_Open=[]
    Real_Y_High=[]
    Real_Y_Low=[]
    Real_Y_Close=[]
    Real_Y_Volume=[]
    
      #Splitting data  real Y
    for i in Batch_Real_Y_NonScaled:
      Real_Y_Close.append(i[1])
    
      
    print(predict_Close)
    print(Real_Y_Close)
    plt.plot(predict_Close,'--g*', label="Predict_Close")
    plt.plot(Real_Y_Close, '--ro',label="Real_Close")
    plt.show()
    
    Real_Y_Close=np.array(Real_Y_Close)
    predict_Close=np.array(predict_Close)
    pd.DataFrame(Real_Y_Close).to_csv(path_or_buf="/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/Study_Data/Real_Y_Close.csv")
    pd.DataFrame(predict_Close).to_csv(path_or_buf="/Users/eduardo/Desktop/LSTM_Capital_API_220922/with6ColumsPerHour/Study_Data/predict_Close.csv")  
    #--------  data shape it's (x days, 5 columns)
    # Convert timestamp to date
   