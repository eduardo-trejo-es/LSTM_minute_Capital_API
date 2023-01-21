###   Model creation

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation



keras.backend.clear_session()  # Reseteo sencillo

#---------Layes are created

n_future = 1   # Number of units(day, min, hour, etc..) we want to look into the future based on the past days.
n_past =60

inputs=keras.Input(shape=(n_past,6))


LSTM_Layer1=keras.layers.LSTM(60, input_shape=(n_past,6), return_sequences=True, activation='sigmoid')(inputs)

Dropout_layer2=keras.layers.Dropout(0.6)(LSTM_Layer1)# modify
#x=Dropout_layer1=keras.layers.Dropout(0.2)(x)
LSTM_Layer2=keras.layers.LSTM(90, return_sequences=False)(Dropout_layer2)

Dropout_layer3=keras.layers.Dropout(0.6)(LSTM_Layer2)# modify


#---------------------------Outputs
dense2=keras.layers.Dense(1)(Dropout_layer3)
dense2_2=keras.layers.Dense(1)(Dropout_layer3)
dense2_3=keras.layers.Dense(1)(Dropout_layer3)
dense2_4=keras.layers.Dense(1)(Dropout_layer3)
dense2_5=keras.layers.Dense(1)(Dropout_layer3)
dense2_6=keras.layers.Dense(1)(Dropout_layer3)


#-------Layers outputs are linked

outputs=dense2
outputs2=dense2_2
outputs3=dense2_3
outputs4=dense2_4
outputs5=dense2_5
outputs6=dense2_6


#-----The model it's created

model=keras.Model(inputs=inputs, outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Forcasting')
#model=keras.Model(inputs=[inputs,None], outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')

#keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


#------------------- Loss and optimizer ----------------------------------------
#got to ensure MeanAbsoluteError it's the good one for our data
loss1 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss2 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss3 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss4 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss5 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
loss6 = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
#optim=keras.optimizers.Adam(1e-3)
optim=keras.optimizers.Adam(1e-3)
metrics=["accuracy"]

losses={
    "dense": loss1,
    "dense_1": loss2,
    "dense_2": loss3,
    "dense_3": loss4,
    "dense_4": loss5,
    "dense_5": loss6,
}

model.compile(loss=losses, optimizer=optim, metrics=metrics)

print(model.summary())

#keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

model.save("Model/Model_Twttr_6colum_0_0")