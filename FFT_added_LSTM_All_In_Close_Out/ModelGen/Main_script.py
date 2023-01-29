from Trainer_Predicting_Esamble import Model_Trainer
#from Forcaster_Model import Forcast_Data
from Forcaster_Model_DateFromToForcast import Forcast_Data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


Model_Path="FFT_added_LSTM_All_In_Close_Out/ModelGen/Model/Model_LSTM_31_FFT_32_in_1_out"
Data_CSV="FFT_added_LSTM_All_In_Close_Out/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
percentageData=98

trainer_model = Model_Trainer()
forcaster = Forcast_Data(Data_CSV)


training_result=trainer_model.to_train(Model_Path,Data_CSV,percentageData)

"""
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
datefiltredPercentage=indexDates[indexDates.shape[0]-100:]
for i in datefiltredPercentage:
    print(i)
    Real_Y_current,Real_Y_Forcast,Real_Y_Close=forcaster.ToForcast(1,Model_Path,str(i))
    ColumnCurrent_Close_Day.append(Real_Y_current)
    ColumnForcast_Close_Day.append(Real_Y_Forcast)
    ColumnReal_Close_Day.append(Real_Y_Close)
    
    print(ColumnCurrent_Close_Day)
    print(ColumnForcast_Close_Day)
    print(ColumnReal_Close_Day)
    
    if i == datefiltredPercentage[len(datefiltredPercentage)-2]: break
ensambly.append(ColumnCurrent_Close_Day)
ensambly.append(ColumnForcast_Close_Day)
ensambly.append(ColumnReal_Close_Day)
    
plt.plot(ColumnForcast_Close_Day,label='ColumnForcast_Close_Day')
plt.plot(ColumnReal_Close_Day,label='ColumnReal_Close_Day')
    #plt.plot([1,2,3,4])
plt.show()
    
print(ensambly)
    # to convert to CSV
"""