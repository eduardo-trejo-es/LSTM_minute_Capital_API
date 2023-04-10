import sys
sys.path.append("/Users/eduardo/Desktop/LSTM_Capital_API_220922/FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/Pakages/DataSetgenPacks")
sys.path.append("/Users/eduardo/Desktop/LSTM_Capital_API_220922/FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/Pakages/ForcastingPacks")
from Trainer_Predicting_Esamble import Model_Trainer
#from Forcaster_Model import Forcast_Data
from Forcaster_Model_DateFromToForcast import Forcast_Data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


inverseModel=0

if inverseModel:
    Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/ModelSigmoid_Tanh/Model_LSTM_FFT_43_sigmoid_Inversed"
    Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/Inversed_direcPrice/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
    percentageData=95
    forcastPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Forcasts/Focast15_02_2023inversed.csv"
else:
    #Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/ModelSigmoid_Tanh/Model_LSTM_FFT_43_PReLU_RealCloseValue"
    Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/Models_fewColums/Model_LSTM_DayMonth20BackDlastFFTCloseValum100FFT"
    #Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/ModelSigmoid_Tanh/Model_LSTM_FFT_43_sigmoid_RightSence"
    #Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
    Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_CloseFFT_100.csv"
    all_colums_Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv"
    percentageData=95
    forcastPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Forcasts/Focast_CloseDayMonth20backdayslastFFT10030_03_2023.csv"

trainer_model = Model_Trainer()
forcaster =Forcast_Data(Model_Path,Data_CSV)
forcaster2 =Forcast_Data(Model_Path,Data_CSV)

#training_result=trainer_model.to_train(Model_Path,Data_CSV,percentageData)
#forcaster.ToForcastfrom("2023-03-24 00:00:00")
date_from="2023-03-24 00:00:00"
date_from2="2023-03-23 00:00:00"
BaseDataSet=""
NewForcast=""

#This method returns the the last prediction made of the n asked forcasted.
#Also during process a new csv data file is generated, it contains the whole real data + the appended n forcasted data
forcaster.RecurrentForcasting(1,date_from,BaseDataSet,NewForcast)
print("---------------------------------------")

forcaster2.RecurrentForcasting(1,date_from2,BaseDataSet,NewForcast)
