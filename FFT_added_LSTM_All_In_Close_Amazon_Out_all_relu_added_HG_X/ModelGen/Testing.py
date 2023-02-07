
import pandas as pd
import numpy as np

csvFileName="FFT_added_LSTM_All_In_Close_Amazon_Out_all_relu_added_HG_X/DatasetGen/Combined_Amazon_GH_F_CL_F_X/Combined_AMZN_GH_F_CL_F_X.csv"
df=pd.read_csv(csvFileName,index_col=0)


print(df.shape)