from unittest import result
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

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

class Model_Trainer:

  def __init__(self):
    pass
  
  def to_train(self,modelPath,DatasetPath,ThepercentageTrainingData):

    ########     Getting the Data      ######
    csvFileName=DatasetPath

    df=pd.read_csv(csvFileName)
    #Separate dates for future plotting
    Data_dates = df.index
    Data_dates=pd.to_datetime(Data_dates,utc=True)
    Data_dates=Data_dates.tz_localize(None)
    
    df.pop('Date')
    Columns_N=df.shape[1]
    #Getting the columns name
    cols = list(df)[0:Columns_N]
    print(cols)


    #New dataframe with only training data - 5 columns
    df_for_training = df[cols].astype(float)
    print(type(df_for_training))
    print(df_for_training.shape)


    #####       Scaling and splitting data    #####
    scaler = MinMaxScaler()

    scaler = scaler.fit(df_for_training)
    DS_raw_scaled = scaler.transform(df_for_training)

    #Scaling for OutPuts "Y" colums
    DS_raw_Open_scaled=DS_raw_scaled[:,[0]]
    DS_raw_High_scaled=DS_raw_scaled[:,[1]]
    DS_raw_Low_scaled=DS_raw_scaled[:,[2]]
    DS_raw_Close_scaled=DS_raw_scaled[:,[3]]
    DS_raw_Volume_scaled=DS_raw_scaled[:,[4]]
    DS_raw_DayNumber_scaled=DS_raw_scaled[:,[5]]
    DS_raw_FFT_Mag_Open_10_scaled=DS_raw_scaled[:,[6]]
    DS_raw_FFT_Angl_Open_10_scaled=DS_raw_scaled[:,[7]]
    DS_raw_FFT_Mag_Open_50_scaled=DS_raw_scaled[:,[8]]
    DS_raw_FFT_Angl_Open_50_scaled=DS_raw_scaled[:,[9]]
    DS_raw_FFT_Mag_Open_100_scaled=DS_raw_scaled[:,[10]]
    DS_raw_FFT_Angl_Open_100_scaled=DS_raw_scaled[:,[11]]
    DS_raw_FFT_Mag_High_10_scaled=DS_raw_scaled[:,[12]]
    DS_raw_FFT_Angl_High_10_scaled=DS_raw_scaled[:,[13]]
    DS_raw_FFT_Mag_High_50_scaled=DS_raw_scaled[:,[14]]
    DS_raw_FFT_Angl_High_50_scaled=DS_raw_scaled[:,[15]]
    DS_raw_FFT_Mag_High_100_scaled=DS_raw_scaled[:,[16]]
    DS_raw_FFT_Angl_High_100_scaled=DS_raw_scaled[:,[17]]
    DS_raw_FFT_Mag_Low_10_scaled=DS_raw_scaled[:,[18]]
    DS_raw_FFT_Angl_Low_10_scaled=DS_raw_scaled[:,[19]]
    DS_raw_FFT_Mag_Low_50_scaled=DS_raw_scaled[:,[20]]
    DS_raw_FFT_Angl_Low_50_scaled=DS_raw_scaled[:,[21]]
    DS_raw_FFT_Mag_Low_100_scaled=DS_raw_scaled[:,[22]]
    DS_raw_FFT_Angl_Low_100_scaled=DS_raw_scaled[:,[23]]
    DS_raw_FFT_Mag_Close_10_scaled=DS_raw_scaled[:,[24]]
    DS_raw_FFT_Angl_Close_10_scaled=DS_raw_scaled[:,[25]]
    DS_raw_FFT_Mag_Close_50_scaled=DS_raw_scaled[:,[26]]
    DS_raw_FFT_Angl_Close_50_scaled=DS_raw_scaled[:,[27]]
    DS_raw_FFT_Mag_Close_100_scaled=DS_raw_scaled[:,[28]]
    DS_raw_FFT_Angl_Close_100_scaled=DS_raw_scaled[:,[29]]
    DS_raw_FFT_Mag_Volume_10_scaled=DS_raw_scaled[:,[30]]
    DS_raw_FFT_Angl_Volume_10_scaled=DS_raw_scaled[:,[31]]
    DS_raw_FFT_Mag_Volume_50_scaled=DS_raw_scaled[:,[32]]
    DS_raw_FFT_Angl_Volume_50_scaled=DS_raw_scaled[:,[33]]
    DS_raw_FFT_Mag_Volume_100_scaled=DS_raw_scaled[:,[34]]
    DS_raw_FFT_Angl_Volume_100_scaled=DS_raw_scaled[:,[35]]

    
    #Empty lists to be populated using formatted training data
    DS_finished_X = []
    DS_finished_Open_Y = []
    DS_finished_High_Y = []
    DS_finished_Low_Y = []
    DS_finished_Close_Y = []
    DS_finished_Volume_Y = []
    DS_finished_DayNumber_Y = []
    DS_finished_FFT_Mag_Open_10_Y = []
    DS_finished_FFT_Angl_Open_10_Y = []
    DS_finished_FFT_Mag_Open_50_Y = []
    DS_finished_FFT_Angl_Open_50_Y = []
    DS_finished_FFT_Mag_Open_100_Y = []
    DS_finished_FFT_Angl_Open_100_Y = []
    DS_finished_FFT_Mag_High_10_Y = []
    DS_finished_FFT_Angl_High_10_Y = []
    DS_finished_FFT_Mag_High_50_Y = []
    DS_finished_FFT_Angl_High_50_Y = []
    DS_finished_FFT_Mag_High_100_Y = []
    DS_finished_FFT_Angl_High_100_Y = []
    DS_finished_FFT_Mag_Low_10_Y = []
    DS_finished_FFT_Angl_Low_10_Y = []
    DS_finished_FFT_Mag_Low_50_Y = []
    DS_finished_FFT_Angl_Low_50_Y = []
    DS_finished_FFT_Mag_Low_100_Y = []
    DS_finished_FFT_Angl_Low_100_Y = []
    DS_finished_FFT_Mag_Close_10_Y = []
    DS_finished_FFT_Angl_Close_10_Y = []
    DS_finished_FFT_Mag_Close_50_Y = []
    DS_finished_FFT_Angl_Close_50_Y = []
    DS_finished_FFT_Mag_Close_100_Y = []
    DS_finished_FFT_Angl_Close_100_Y = []
    DS_finished_FFT_Mag_Volume_10_Y = []
    DS_finished_FFT_Angl_Volume_10_Y = []
    DS_finished_FFT_Mag_Volume_50_Y = []
    DS_finished_FFT_Angl_Volume_50_Y = []
    DS_finished_FFT_Mag_Volume_100_Y = []
    DS_finished_FFT_Angl_Volume_100_Y = []

    n_future = 1   # Number of units(day, min, hour, etc..) we want to look into the future based on the past days.
    n_past =120


    # Creatig the data batches, each one with 30d days

    for i in range(n_past, len(DS_raw_scaled) - n_future +1):
      DS_finished_X.append(DS_raw_scaled[i - n_past:i, 0:DS_raw_scaled.shape[1]])
      DS_finished_Open_Y.append(DS_raw_Open_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_High_Y.append(DS_raw_High_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_Low_Y.append(DS_raw_Low_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_Close_Y.append(DS_raw_Close_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_Volume_Y.append(DS_raw_Volume_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_DayNumber_Y.append(DS_raw_DayNumber_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Open_10_Y.append(DS_raw_FFT_Mag_Open_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Open_10_Y.append(DS_raw_FFT_Angl_Open_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Open_50_Y.append(DS_raw_FFT_Mag_Open_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Open_50_Y.append(DS_raw_FFT_Angl_Open_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Open_100_Y.append(DS_raw_FFT_Mag_Open_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Open_100_Y.append(DS_raw_FFT_Angl_Open_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_High_10_Y.append(DS_raw_FFT_Mag_High_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_High_10_Y.append(DS_raw_FFT_Angl_High_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_High_50_Y.append(DS_raw_FFT_Mag_High_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_High_50_Y.append(DS_raw_FFT_Angl_High_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_High_100_Y.append(DS_raw_FFT_Mag_High_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_High_100_Y.append(DS_raw_FFT_Angl_High_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Low_10_Y.append(DS_raw_FFT_Mag_Low_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Low_10_Y.append(DS_raw_FFT_Angl_Low_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Low_50_Y.append(DS_raw_FFT_Mag_Low_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Low_50_Y.append(DS_raw_FFT_Angl_Low_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Low_100_Y.append(DS_raw_FFT_Mag_Low_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Low_100_Y.append(DS_raw_FFT_Angl_Low_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Close_10_Y.append(DS_raw_FFT_Mag_Close_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Close_10_Y.append(DS_raw_FFT_Angl_Close_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Close_50_Y.append(DS_raw_FFT_Mag_Close_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Close_50_Y.append(DS_raw_FFT_Angl_Close_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Close_100_Y.append(DS_raw_FFT_Mag_Close_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Close_100_Y.append(DS_raw_FFT_Angl_Close_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Volume_10_Y.append(DS_raw_FFT_Mag_Volume_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Volume_10_Y.append(DS_raw_FFT_Angl_Volume_10_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Volume_50_Y.append(DS_raw_FFT_Mag_Volume_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Volume_50_Y.append(DS_raw_FFT_Angl_Volume_50_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Mag_Volume_100_Y.append(DS_raw_FFT_Mag_Volume_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])
      DS_finished_FFT_Angl_Volume_100_Y.append(DS_raw_FFT_Angl_Volume_100_scaled[i:i+1, 0:DS_raw_scaled.shape[1]])

    DS_finished_X,DS_finished_Open_Y,DS_finished_High_Y,DS_finished_Low_Y,DS_finished_Close_Y,DS_finished_Volume_Y,DS_finished_DayNumber_Y,DS_finished_FFT_Mag_Open_10_Y,DS_finished_FFT_Angl_Open_10_Y,DS_finished_FFT_Mag_Open_50_Y,DS_finished_FFT_Angl_Open_50_Y,DS_finished_FFT_Mag_Open_100_Y,DS_finished_FFT_Angl_Open_100_Y,DS_finished_FFT_Mag_High_10_Y,DS_finished_FFT_Angl_High_10_Y,DS_finished_FFT_Mag_High_50_Y,DS_finished_FFT_Angl_High_50_Y,DS_finished_FFT_Mag_High_100_Y,DS_finished_FFT_Angl_High_100_Y,DS_finished_FFT_Mag_Low_10_Y,DS_finished_FFT_Angl_Low_10_Y,DS_finished_FFT_Mag_Low_50_Y,DS_finished_FFT_Angl_Low_50_Y,DS_finished_FFT_Mag_Low_100_Y,DS_finished_FFT_Angl_Low_100_Y,DS_finished_FFT_Mag_Close_10_Y,DS_finished_FFT_Angl_Close_10_Y,DS_finished_FFT_Mag_Close_50_Y,DS_finished_FFT_Angl_Close_50_Y,DS_finished_FFT_Mag_Close_100_Y,DS_finished_FFT_Angl_Close_100_Y,DS_finished_FFT_Mag_Volume_10_Y,DS_finished_FFT_Angl_Volume_10_Y,DS_finished_FFT_Mag_Volume_50_Y,DS_finished_FFT_Angl_Volume_50_Y,DS_finished_FFT_Mag_Volume_100_Y,DS_finished_FFT_Angl_Volume_100_Y=np.array(DS_finished_X), np.array(DS_finished_Open_Y),np.array(DS_finished_High_Y),np.array(DS_finished_Low_Y),np.array(DS_finished_Close_Y),np.array(DS_finished_Volume_Y),np.array(DS_finished_DayNumber_Y),np.array(DS_finished_FFT_Mag_Open_10_Y),np.array(DS_finished_FFT_Angl_Open_10_Y),np.array(DS_finished_FFT_Mag_Open_50_Y),np.array(DS_finished_FFT_Angl_Open_50_Y),np.array(DS_finished_FFT_Mag_Open_100_Y),np.array(DS_finished_FFT_Angl_Open_100_Y),np.array(DS_finished_FFT_Mag_High_10_Y),np.array(DS_finished_FFT_Angl_High_10_Y),np.array(DS_finished_FFT_Mag_High_50_Y),np.array(DS_finished_FFT_Angl_High_50_Y),np.array(DS_finished_FFT_Mag_High_100_Y),np.array(DS_finished_FFT_Angl_High_100_Y),np.array(DS_finished_FFT_Mag_Low_10_Y),np.array(DS_finished_FFT_Angl_Low_10_Y),np.array(DS_finished_FFT_Mag_Low_50_Y),np.array(DS_finished_FFT_Angl_Low_50_Y),np.array(DS_finished_FFT_Mag_Low_100_Y),np.array(DS_finished_FFT_Angl_Low_100_Y),np.array(DS_finished_FFT_Mag_Close_10_Y),np.array(DS_finished_FFT_Angl_Close_10_Y),np.array(DS_finished_FFT_Mag_Close_50_Y),np.array(DS_finished_FFT_Angl_Close_50_Y),np.array(DS_finished_FFT_Mag_Close_100_Y),np.array(DS_finished_FFT_Angl_Close_100_Y),np.array(DS_finished_FFT_Mag_Volume_10_Y),np.array(DS_finished_FFT_Angl_Volume_10_Y),np.array(DS_finished_FFT_Mag_Volume_50_Y),np.array(DS_finished_FFT_Angl_Volume_50_Y),np.array(DS_finished_FFT_Mag_Volume_100_Y),np.array(DS_finished_FFT_Angl_Volume_100_Y)
    
    DS_dates_finished_X = []

    ### Creating the data dates batches
    for a in range(n_past, len(Data_dates) - n_future +1):
      DS_dates_finished_X.append(Data_dates[a - n_past:a])
      #DS_dates_finished_X.append(Data_dates[a - n_past:a, 0:Data_dates.shape[1]])
    DS_dates_finished_X = np.array(DS_dates_finished_X)
    DS_dates_finished_X.shape


    #######    Let's split data     #######
    #It's going to be used 70% of data to train and 30% to test the model

    #This Function split dataset with shape (z,x, y) and is splited in z
    def Split3DimData(DataSet,percentageTrainig):
      percentageTrainDataset = 0
      percentageTrainDataset=int((DataSet.shape[0]*percentageTrainig)/100)
      DataSetSplittedTraining=DataSet[0:percentageTrainDataset]
      DataSetSplittedTesting= DataSet[percentageTrainDataset:]

      return DataSetSplittedTraining, DataSetSplittedTesting

    percentageTrainingData= ThepercentageTrainingData
    #####   training data, testing data  ####
    trainX, testingX = Split3DimData(DS_finished_X,percentageTrainingData)
    trainY_Open,testingY_Open=Split3DimData(DS_finished_Open_Y,percentageTrainingData)
    trainY_High,testingY_High=Split3DimData(DS_finished_High_Y,percentageTrainingData)
    trainY_Low,testingY_Low=Split3DimData(DS_finished_Low_Y,percentageTrainingData)
    trainY_Close,testingY_Close=Split3DimData(DS_finished_Close_Y,percentageTrainingData)
    trainY_Volume,testingY_Volume=Split3DimData(DS_finished_Volume_Y,percentageTrainingData)
    trainY_DayNumber,testingY_DayNumber=Split3DimData(DS_finished_DayNumber_Y,percentageTrainingData)
    trainY_FFT_Mag_Open_10,testingY_FFT_Mag_Open_10=Split3DimData(DS_finished_FFT_Mag_Open_10_Y,percentageTrainingData)
    trainY_FFT_Angl_Open_10,testingY_FFT_Angl_Open_10=Split3DimData(DS_finished_FFT_Angl_Open_10_Y,percentageTrainingData)
    trainY_FFT_Mag_Open_50,testingY_FFT_Mag_Open_50=Split3DimData(DS_finished_FFT_Mag_Open_50_Y,percentageTrainingData)
    trainY_FFT_Angl_Open_50,testingY_FFT_Angl_Open_50=Split3DimData(DS_finished_FFT_Angl_Open_50_Y,percentageTrainingData)
    trainY_FFT_Mag_Open_100,testingY_FFT_Mag_Open_100=Split3DimData(DS_finished_FFT_Mag_Open_100_Y,percentageTrainingData)
    trainY_FFT_Angl_Open_100,testingY_FFT_Angl_Open_100=Split3DimData(DS_finished_FFT_Angl_Open_100_Y,percentageTrainingData)
    trainY_FFT_Mag_High_10,testingY_FFT_Mag_High_10=Split3DimData(DS_finished_FFT_Mag_High_10_Y,percentageTrainingData)
    trainY_FFT_Angl_High_10,testingY_FFT_Angl_High_10=Split3DimData(DS_finished_FFT_Angl_High_10_Y,percentageTrainingData)
    trainY_FFT_Mag_High_50,testingY_FFT_Mag_High_50=Split3DimData(DS_finished_FFT_Mag_High_50_Y,percentageTrainingData)
    trainY_FFT_Angl_High_50,testingY_FFT_Angl_High_50=Split3DimData(DS_finished_FFT_Angl_High_50_Y,percentageTrainingData)
    trainY_FFT_Mag_High_100,testingY_FFT_Mag_High_100=Split3DimData(DS_finished_FFT_Mag_High_100_Y,percentageTrainingData)
    trainY_FFT_Angl_High_100,testingY_FFT_Angl_High_100=Split3DimData(DS_finished_FFT_Angl_High_100_Y,percentageTrainingData)
    trainY_FFT_Mag_Low_10,testingY_FFT_Mag_Low_10=Split3DimData(DS_finished_FFT_Mag_Low_10_Y,percentageTrainingData)
    trainY_FFT_Angl_Low_10,testingY_FFT_Angl_Low_10=Split3DimData(DS_finished_FFT_Angl_Low_10_Y,percentageTrainingData)
    trainY_FFT_Mag_Low_50,testingY_FFT_Mag_Low_50=Split3DimData(DS_finished_FFT_Mag_Low_50_Y,percentageTrainingData)
    trainY_FFT_Angl_Low_50,testingY_FFT_Angl_Low_50=Split3DimData(DS_finished_FFT_Angl_Low_50_Y,percentageTrainingData)
    trainY_FFT_Mag_Low_100,testingY_FFT_Mag_Low_100=Split3DimData(DS_finished_FFT_Mag_Low_100_Y,percentageTrainingData)
    trainY_FFT_Angl_Low_100,testingY_FFT_Angl_Low_100=Split3DimData(DS_finished_FFT_Angl_Low_100_Y,percentageTrainingData)
    trainY_FFT_Mag_Close_10,testingY_FFT_Mag_Close_10=Split3DimData(DS_finished_FFT_Mag_Close_10_Y,percentageTrainingData)
    trainY_FFT_Angl_Close_10,testingY_FFT_Angl_Close_10=Split3DimData(DS_finished_FFT_Angl_Close_10_Y,percentageTrainingData)
    trainY_FFT_Mag_Close_50,testingY_FFT_Mag_Close_50=Split3DimData(DS_finished_FFT_Mag_Close_50_Y,percentageTrainingData)
    trainY_FFT_Angl_Close_50,testingY_FFT_Angl_Close_50=Split3DimData(DS_finished_FFT_Angl_Close_50_Y,percentageTrainingData)
    trainY_FFT_Mag_Close_100,testingY_FFT_Mag_Close_100=Split3DimData(DS_finished_FFT_Mag_Close_100_Y,percentageTrainingData)
    trainY_FFT_Angl_Close_100,testingY_FFT_Angl_Close_100=Split3DimData(DS_finished_FFT_Angl_Close_100_Y,percentageTrainingData)
    trainY_FFT_Mag_Volume_10,testingY_FFT_Mag_Volume_10=Split3DimData(DS_finished_FFT_Mag_Volume_10_Y,percentageTrainingData)
    trainY_FFT_Angl_Volume_10,testingY_FFT_Angl_Volume_10=Split3DimData(DS_finished_FFT_Angl_Volume_10_Y,percentageTrainingData)
    trainY_FFT_Mag_Volume_50,testingY_FFT_Mag_Volume_50=Split3DimData(DS_finished_FFT_Mag_Volume_50_Y,percentageTrainingData)
    trainY_FFT_Angl_Volume_50,testingY_FFT_Angl_Volume_50=Split3DimData(DS_finished_FFT_Angl_Volume_50_Y,percentageTrainingData)
    trainY_FFT_Mag_Volume_100,testingY_FFT_Mag_Volume_100=Split3DimData(DS_finished_FFT_Mag_Volume_100_Y,percentageTrainingData)
    trainY_FFT_Angl_Volume_100,testingY_FFT_Angl_Volume_100=Split3DimData(DS_finished_FFT_Angl_Volume_100_Y,percentageTrainingData)

    ##Validated data set, im getting the spected result

    print(testingX.shape)
    print(trainX.shape)

    train_Dates, testing_Dates = Split3DimData(Data_dates,percentageTrainingData)

    print(train_Dates.shape)
    print(testing_Dates.shape)


    ##############      Retriving model       ########


    model = keras.models.load_model(modelPath)



    #######          Model Training      ################


    #--------------------------- Assing Y data to losses dictionary -----
    y_data={ 
      "dense": trainY_Open,
      "dense_1": trainY_High,
      "dense_2": trainY_Low,
      "dense_3": trainY_Close,
      "dense_4": trainY_Volume,
      "dense_5": trainY_DayNumber,
      "dense_6": trainY_FFT_Mag_Open_10,
      "dense_7": trainY_FFT_Angl_Open_10,
      "dense_8": trainY_FFT_Mag_Open_50,
      "dense_9": trainY_FFT_Angl_Open_50,
      "dense_10": trainY_FFT_Mag_Open_100,
      "dense_11": trainY_FFT_Angl_Open_100,
      "dense_12": trainY_FFT_Mag_High_10,
      "dense_13": trainY_FFT_Angl_High_10,
      "dense_14": trainY_FFT_Mag_High_50,
      "dense_15": trainY_FFT_Angl_High_50,
      "dense_16": trainY_FFT_Mag_High_100,
      "dense_17": trainY_FFT_Angl_High_100,
      "dense_18": trainY_FFT_Mag_Low_10,
      "dense_19": trainY_FFT_Angl_Low_10,
      "dense_20": trainY_FFT_Mag_Low_50,
      "dense_21": trainY_FFT_Angl_Low_50,
      "dense_22": trainY_FFT_Mag_Low_100,
      "dense_23": trainY_FFT_Angl_Low_100,
      "dense_24": trainY_FFT_Mag_Close_10,
      "dense_25": trainY_FFT_Angl_Close_10,
      "dense_26": trainY_FFT_Mag_Close_50,
      "dense_27": trainY_FFT_Angl_Close_50,
      "dense_28": trainY_FFT_Mag_Close_100,
      "dense_29": trainY_FFT_Angl_Close_100,
      "dense_30": trainY_FFT_Mag_Volume_10,
      "dense_31": trainY_FFT_Angl_Volume_10,
      "dense_32": trainY_FFT_Mag_Volume_50,
      "dense_33": trainY_FFT_Angl_Volume_50,
      "dense_34": trainY_FFT_Mag_Volume_100,
      "dense_35": trainY_FFT_Angl_Volume_100
    }
    testing_y_data={
      "dense": testingY_Open,
      "dense_1": testingY_High,
      "dense_2": testingY_Low,
      "dense_3": testingY_Close,
      "dense_4": testingY_Volume,
      "dense_5": testingY_DayNumber,
      "dense_6": testingY_FFT_Mag_Open_10,
      "dense_7": testingY_FFT_Angl_Open_10,
      "dense_8": testingY_FFT_Mag_Open_50,
      "dense_9": testingY_FFT_Angl_Open_50,
      "dense_10": testingY_FFT_Mag_Open_100,
      "dense_11": testingY_FFT_Angl_Open_100,
      "dense_12": testingY_FFT_Mag_High_10,
      "dense_13": testingY_FFT_Angl_High_10,
      "dense_14": testingY_FFT_Mag_High_50,
      "dense_15": testingY_FFT_Angl_High_50,
      "dense_16": testingY_FFT_Mag_High_100,
      "dense_17": testingY_FFT_Angl_High_100,
      "dense_18": testingY_FFT_Mag_Low_10,
      "dense_19": testingY_FFT_Angl_Low_10,
      "dense_20": testingY_FFT_Mag_Low_50,
      "dense_21": testingY_FFT_Angl_Low_50,
      "dense_22": testingY_FFT_Mag_Low_100,
      "dense_23": testingY_FFT_Angl_Low_100,
      "dense_24": testingY_FFT_Mag_Close_10,
      "dense_25": testingY_FFT_Angl_Close_10,
      "dense_26": testingY_FFT_Mag_Close_50,
      "dense_27": testingY_FFT_Angl_Close_50,
      "dense_28": testingY_FFT_Mag_Close_100,
      "dense_29": testingY_FFT_Angl_Close_100,
      "dense_30": testingY_FFT_Mag_Volume_10,
      "dense_31": testingY_FFT_Angl_Volume_10,
      "dense_32": testingY_FFT_Mag_Volume_50,
      "dense_33": testingY_FFT_Angl_Volume_50,
      "dense_34": testingY_FFT_Mag_Volume_100,
      "dense_35": testingY_FFT_Angl_Volume_100
    }

    #------------------------- Training model --------------------------------
    
    early_stop= EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
    
    model.fit(x=trainX,y=y_data, epochs=40, batch_size=15, validation_data=(testingX,testing_y_data),callbacks=[early_stop])
    #history = model.fit(trainX,y=y_data, epochs=125, batch_size=15)


    losses = pd.DataFrame(model.history.history)

    losses.plot()

    model.save(modelPath)
    
    
    Training_result="done... ;)"
    
    return Training_result


    #######    Model evaluation      ########
    def Model_evaluation(self):
      from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
      import math

      starBatch=1
      endbatch=0

      Batch_to_predict=testingX[testingX.shape[0]-starBatch:testingX.shape[0]-endbatch]

      print(Batch_to_predict.shape)

      print(testingX.shape)


      #########        Evaluation funtion  #########3


      #Mean square error of testing data

      #DataSet unseen before

      #Nota: Se obtuvo un error del 15% aprox. para los 5 features para el dataset no visto y ya visto

      y_data_testing={ 
        "dense": testingY_Open,
        "dense_1": testingY_High,
        "dense_2": testingY_Low,
        "dense_3": testingY_Close,
        "dense_4": testingY_Volume,
      }

      mean_square_error_testin_DS=model.evaluate(testingX,y_data_testing,verbose=0)

      mean_square_error_testing_DS_nparray=np.array(mean_square_error_testin_DS)
      mean_square_error_testing_DS_nparray=mean_square_error_testing_DS_nparray[0:5]
      mean_square_error_testing_DS_nparray_reshaped=np.reshape(mean_square_error_testing_DS_nparray,(1,5))

      #mean_square_error_training_DS_reshaped=np.array(mean_square_error_training_DS)
      testing_DS_scaled_Back = scaler.inverse_transform(mean_square_error_testing_DS_nparray_reshaped)

      for i in testing_DS_scaled_Back[0]:
        print(math.sqrt(i))
        
      y_data_taining_eval={ 
        "dense": trainY_Open,
        "dense_1": trainY_High,
        "dense_2": trainY_Low,
        "dense_3": trainY_Close,
        "dense_4": trainY_Volume,
      }

      #######          Evaluation Using the model      ##############


      N_Days_to_predict=14
      Prediction_Saved=[]
      #testingX=np.array(testingX)

      for i in range(N_Days_to_predict):
        prediction = model.predict(Batch_to_predict) #the input is a 30 units of time batch
        prediction_Reshaped=np.reshape(prediction,(1,1,5))
        Batch_to_predict=np.append(Batch_to_predict,prediction_Reshaped, axis=1)
        Batch_to_predict=np.delete(Batch_to_predict,0,1)
        print(Batch_to_predict.shape)
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
        
        
      ####      Getting the candle chart   #####

      BatchUsed_Dates=DS_dates_finished_X[len(DS_dates_finished_X)-starBatch:len(DS_dates_finished_X)-endbatch]

      lastTimedate=BatchUsed_Dates[0][BatchUsed_Dates.shape[1]-1:]
      lastTimedate=lastTimedate[0]


      import pandas as pd
      import numpy as np

      BatchForcasted_Dates=[]
      for i in range(0,N_Days_to_predict):
        BatchForcasted_Dates.append(pd.Timestamp(np.datetime64(lastTimedate)))
        lastTimedate=np.datetime64(lastTimedate) + np.timedelta64(1, 'h')


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
        
        
      #--------  data shape it's (x days, 6 columns)
      # Convert timestamp to date
      df_forecast = pd.DataFrame({'Open':predict_Open,'High':predict_High, 'Low':predict_Low,'Close':predict_Close,'Volume':predict_Volume}, index=BatchForcasted_Dates)

      df_forecast.index.name="Date"



      import mplfinance as mpf
      title_chart="Twtr"
      mpf.plot(df_forecast, type='candle',title=title_chart, style='charles')