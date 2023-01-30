from Retriver_and_Processor_Dataset import *

dataSet_Gen= DatasetGenerator()



#dataSet_Gen.RetivingDataPrices_Yahoo('2000-08-23', '2023-01-20',"FFT_added_LSTM/DatasetGen/CRUDE_OIL/CRUDE_OIL_DataCSV","FFT_added_LSTM/DatasetGen/CRUDE_OIL/CRUDE_OIL_DataCSV")

#dataSet_Gen.UpdateToday("FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv")



#dataSet_Gen.AddColumnWeekDay("FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv", "FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_DayNum.csv",False)

#dataSet_Gen.AddColumnMoth("FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_DayNum.csv", "FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_month.csv",False)

#dataSet_Gen.AddColumnYear("FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_month.csv","FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_year.csv")

#Generate new FFT columns done :)



Column=['Open','High','Low','Close','Volume']
frec=[10,50,100]

inicialPath="FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_year.csv"
FFTNew_FileData="FFT_added_LSTM_All_In_Close_Out_tanh_Month_year/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"
for i in Column:
    for j in frec:
        if i == Column[0] and j == frec[0]:
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,inicialPath,FFTNew_FileData)
        else:   
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,FFTNew_FileData,FFTNew_FileData)


