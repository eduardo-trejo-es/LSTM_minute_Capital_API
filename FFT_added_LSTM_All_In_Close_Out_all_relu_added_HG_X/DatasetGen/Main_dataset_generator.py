from Retriver_and_Processor_Dataset import *

dataSet_Gen= DatasetGenerator()



dateStart='2001-01-01'
dateEnd= '2023-03-02'


#######______________ CRUDE_Oil 'CL=F' _____________  ################
itemName='CL=F'

Original_Path_Retiving="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data.csv"
Onlyonecolumn="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_onlyClose.csv"
LastOnetwice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_LastOneTwice.csv"
DirectionPrice="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_DirePrice.csv"
DayNumAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_DayNum.csv"
MonthAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_month.csv"
yearAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Data_And_year.csv"
FFTAddedPath="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_CloseFFT_1800.csv"




#dataSet_Gen.RetivingDataPrices_Yahoo(itemName,dateStart, dateEnd,Original_Path_Retiving,Original_Path_Retiving)
dataSet_Gen.UpdateToday(itemName,Original_Path_Retiving)
#columns to pop up
dataSet_Gen.AddRepeatedLastOne(Original_Path_Retiving, LastOnetwice)

columns=['Open','High','Low','Volume']
#columns=['Open','High','Low']
#columns=['Volume']

dataSet_Gen.PopListdf(columns,LastOnetwice,Onlyonecolumn)
#dataSet_Gen.AddColumnPRCNTG(Original_Path_Retiving,PRCNTGAddedPath)
#if inversed:
#    dataSet_Gen.AddColumnInverseDirePrice(Original_Path_Retiving,DirectionPrice)
#else: 
#    dataSet_Gen.AddColumnDirePrice(Original_Path_Retiving,DirectionPrice)



dataSet_Gen.AddColumnWeekDay(Onlyonecolumn, DayNumAddedPath,False)

dataSet_Gen.AddColumnMoth(DayNumAddedPath, MonthAddedPath,False)

dataSet_Gen.AddColumnYear(MonthAddedPath,yearAddedPath)
#
#Generate new FFT columns done :)



#Column=['Open_CL=F','High_CL=F','Low_CL=F','Close_CL=F','Volume_CL=F','Open_HG=F','High_HG=F','Low_HG=F','Close_HG=F','Volume_HG=F','Open_X','High_X','Low_X','Close_X','Volume_X']
#Column=['Open','High','Low','Close']
Column=['Close']
frec=[1800]
#frec=[10,50,100]

inicialPath=yearAddedPath
FFTNew_FileData=FFTAddedPath
for i in Column:
    for j in frec:
        if i == Column[0] and j == frec[0]:
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,inicialPath,FFTNew_FileData)
        else:   
            dataSet_Gen.Add_ColumsFourier_Transform(j,i,FFTNew_FileData,FFTNew_FileData)
"""
if inversed:
    path_CSV_df=["FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/Inversed_direcPrice/CRUDE_OIL_Dataand_FFT_10_50_100.csv",
                "FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/Copper_GH_F/Inversed_direcPrice/Copper_Dataand_FFT_10_50_100.csv"]
    combined_path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/Combined_GH_F_CL_F_X/Inversed_direcPrice/CombinedGH_F_CL_F_X.csv"
else:
    path_CSV_df=["FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/CRUDE_OIL/CRUDE_OIL_Dataand_FFT_10_50_100.csv",
                "FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/Copper_GH_F/Copper_Dataand_FFT_10_50_100.csv"]
    combined_path="FFT_added_LSTM_All_In_Close_Out_all_relu_added_HG_X/DatasetGen/Combined_GH_F_CL_F_X/CombinedGH_F_CL_F_X.csv"
            
dataSet_Gen.dfCombiner(path_CSV_df,combined_path) 
"""


