from Trainer_Predicting_Esamble import Model_Trainer
#from Forcaster_Model import Forcast_Data
from Forcaster_Model_DateFromToForcast import Forcast_Data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


Model_Path="FFT_added_LSTM_All_In_Close_Out_tath_added/ModelGen/Model/Model_LSTM_31_FFT_32_in_1_out_tanh_added"
Data_CSV="FFT_added_LSTM_All_In_Close_Out_tath_added/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
percentageData=85
forcastPath="FFT_added_LSTM_All_In_Close_Out_tath_added/ModelGen/Forcasts/Focast28_01_2023.csv"

trainer_model = Model_Trainer()
forcaster = Forcast_Data(Data_CSV)

#training_result=trainer_model.to_train(Model_Path,Data_CSV,percentageData)

########## forcasting instuctions below ########


df=pd.read_csv(Data_CSV,index_col=0)

locpercentage=0
ColumnCurrent_Close_Day=[]
Real_Y_current=0
ColumnForcast_Close_Day=[]
Real_Y_Forcast=0
ColumnReal_Close_Day=[]
Real_Y_Close=0

ensambly=[]

indexDates=df.index

locpercentage=int((indexDates.shape[0]*percentageData)/100)

#datefiltredPercentage=indexDates[locpercentage:]
datefiltredPercentage=indexDates[indexDates.shape[0]-150:]
for i in datefiltredPercentage:
    print(i)
    Real_Y_current,Real_Y_Forcast,Real_Y_Close=forcaster.ToForcast(1,Model_Path,str(i))
    ColumnCurrent_Close_Day.append([Real_Y_current])
    ColumnForcast_Close_Day.append([Real_Y_Forcast])
    ColumnReal_Close_Day.append([Real_Y_Close])
    
    print(ColumnCurrent_Close_Day)
    print(ColumnForcast_Close_Day)
    print(ColumnReal_Close_Day)
    
    if i == datefiltredPercentage[len(datefiltredPercentage)-2]: break
"""
ColumnCurrent_Close_Day_np=np.array(ColumnCurrent_Close_Day)
ColumnCurrent_Close_Day_np=np.reshape(ColumnCurrent_Close_Day_np, (1, ColumnCurrent_Close_Day_np.shape[0]))
ColumnForcast_Close_Day_np=np.array(ColumnForcast_Close_Day)
ColumnForcast_Close_Day_np=np.reshape(ColumnForcast_Close_Day_np, (1, ColumnForcast_Close_Day_np.shape[0]))
ColumnReal_Close_Day_np=np.array(ColumnReal_Close_Day)
ColumnReal_Close_Day_np=np.reshape(ColumnReal_Close_Day_np, (1, ColumnReal_Close_Day_np.shape[0]))


ensambly=np.concatenate((ColumnCurrent_Close_Day_np, ColumnForcast_Close_Day_np,ColumnReal_Close_Day_np))


plt.plot(ColumnForcast_Close_Day,label='ColumnForcast_Close_Day')
plt.plot(ColumnReal_Close_Day,label='ColumnReal_Close_Day')
    #plt.plot([1,2,3,4])
#plt.show()


ColumnCurrent_Close_Day_trans=np.reshape(ColumnCurrent_Close_Day_np, (ColumnCurrent_Close_Day_np.shape[1], 1))
ColumnForcast_Close_Day_trans=np.reshape(ColumnForcast_Close_Day_np, (ColumnForcast_Close_Day_np.shape[1], 1))
ColumnReal_Close_Day_trans=np.reshape(ColumnReal_Close_Day_np, (ColumnReal_Close_Day_np.shape[1], 1))"""

ensambly_fin=pd.DataFrame({'Current':ColumnCurrent_Close_Day,'Forcast':ColumnForcast_Close_Day,'Real':ColumnReal_Close_Day})

#np.insert(a, a.shape[1], np.array((10, 10, 10, 10)), 1)
#print(ensambly_np.shape)
#print(ensambly_np.shape)
    # to convert to CSV

ensambly_fin.to_csv(path_or_buf=forcastPath,index=False)

