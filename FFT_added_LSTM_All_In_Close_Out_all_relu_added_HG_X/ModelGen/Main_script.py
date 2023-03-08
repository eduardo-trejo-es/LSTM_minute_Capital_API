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
    Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/Models_fewColums/Model_LSTM_CloseDayMonthYearFFT_only1800FFT"
    #Model_Path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Model/ModelSigmoid_Tanh/Model_LSTM_FFT_43_sigmoid_RightSence"
    #Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
    Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_CloseFFT_1800.csv"
    all_colums_Data_CSV="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv"
    percentageData=95
    forcastPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/ModelGen/Forcasts/Focast_CloseDayMonthYearvolumFFT180003_03_2023.csv"

trainer_model = Model_Trainer()
forcaster = Forcast_Data(Model_Path,Data_CSV)

#training_result=trainer_model.to_train(Model_Path,Data_CSV,percentageData)
"""
Real_Y_current,Real_Y_Forcast,Real_Y_Close=forcaster.ToForcast(1,"2023-02-27")
print(Real_Y_current)
print(Real_Y_Forcast)
print(Real_Y_Close) 
"""
########## forcasting instuctions below ########
saveAllandforcast=pd.DataFrame({})
fd_ColumnForcast_Close_Day=pd.DataFrame({})
all_df=pd.read_csv(all_colums_Data_CSV,index_col=0)


df=pd.read_csv(Data_CSV,index_col=0)


backdaysConsidered=10
locpercentage=0
ColumnCurrent_Close_Day=[]
Real_Y_current=0
ColumnForcast_Close_Day=[]
Real_Y_Forcast=0
ColumnReal_Close_Day=[]
Real_Y_Close=0
Forcast_Dates=[]
Forcast_Dates_toshow=[]

ensambly=[]

indexDates=df.index

locpercentage=int((indexDates.shape[0]*percentageData)/100)

#datefiltredPercentage=indexDates[locpercentage:]
datefiltredPercentage=indexDates[indexDates.shape[0]-backdaysConsidered:]
for i in datefiltredPercentage:
    Real_Y_current,Real_Y_Forcast,Real_Y_Close=forcaster.ToForcast(1,str(i))
    ColumnCurrent_Close_Day.append(Real_Y_current)
    ColumnForcast_Close_Day.append(Real_Y_Forcast)
    ColumnReal_Close_Day.append(Real_Y_Close)
    Forcast_Dates.append(str(i))
    
    #if i == datefiltredPercentage[len(datefiltredPercentage)-2]: break
Forcast_Dates_toshow=Forcast_Dates

print(ColumnForcast_Close_Day)
print("---------------------------------------------------")
print(ColumnReal_Close_Day)


### Below df only has close forcast data and dates forcast data
fd_ColumnForcast_Close_Day=pd.DataFrame({'Forcast':ColumnForcast_Close_Day})
#fd_ColumnForcast_Close_Day=pd.DataFrame({'Forcast':ColumnForcast_Close_Day,'ForcastDateTShow':Forcast_Dates_toshow})
fd_ColumnForcast_Close_Day['Dates']=Forcast_Dates
fd_ColumnForcast_Close_Day=fd_ColumnForcast_Close_Day.set_index('Dates')

### Below df has all origianl colums and dates
Allandforcast=all_df[all_df.shape[0]-backdaysConsidered:]

frames = [Allandforcast, fd_ColumnForcast_Close_Day]

Final_Allandforcast = pd.concat(frames,axis=1)
print(Final_Allandforcast)


plt.plot(ColumnForcast_Close_Day,label='ColumnForcast_Close_Day',color='orange', marker='o')
plt.plot(ColumnReal_Close_Day,label='ColumnReal_Close_Day',color='green', marker='*')
    #plt.plot([1,2,3,4])
plt.show()
#np.insert(a, a.shape[1], np.array((10, 10, 10, 10)), 1)
#print(ensambly_np.shape)
#print(ensambly_np.shape)
    # to convert to CSV

Final_Allandforcast.to_csv(path_or_buf=forcastPath,index=True)
