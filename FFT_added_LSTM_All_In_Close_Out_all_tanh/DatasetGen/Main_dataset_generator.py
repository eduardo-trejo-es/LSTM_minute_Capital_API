from Retriver_and_Processor_Dataset import *

dataSet_Gen= DatasetGenerator()




#######______________ CRUDE_Oil 'CL=F' _____________  ################
dateStart='2001-01-01'
dateEnd= '2023-02-01'
itemName='CL=F'
Original_Path_Retiving="FFT_added_LSTM_All_In_Close_Out_all_tanh/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv"
DayNumAddedPath="FFT_added_LSTM_All_In_Close_Out_all_tanh/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_DayNum.csv"
MonthAddedPath="FFT_added_LSTM_All_In_Close_Out_all_tanh/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_month.csv"
yearAddedPath="FFT_added_LSTM_All_In_Close_Out_all_tanh/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_year.csv"
FFTAddedPath="FFT_added_LSTM_All_In_Close_Out_all_tanh/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv"




dataSet_Gen.RetivingDataPrices_Yahoo(dateStart, dateEnd,Original_Path_Retiving,Original_Path_Retiving)

#dataSet_Gen.UpdateToday(Original_Path_Retiving)

dataSet_Gen.AddColumnWeekDay(Original_Path_Retiving, DayNumAddedPath,False)

dataSet_Gen.AddColumnMoth(DayNumAddedPath, MonthAddedPath,False)

dataSet_Gen.AddColumnYear(MonthAddedPath,yearAddedPath)

#Generate new FFT columns done :)



Column=['Open','High','Low','Close','Volume']
frec=[10,50,100]

inicialPath=yearAddedPath
FFTNew_FileData=FFTAddedPath
for i in Column:
    for j in frec:
        if i == Column[0] and j == frec[0]:
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,inicialPath,FFTNew_FileData)
        else:   
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,FFTNew_FileData,FFTNew_FileData)

