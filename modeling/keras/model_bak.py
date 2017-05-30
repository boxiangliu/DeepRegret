from keras.models import Sequential
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l1

reg_model=Sequential()
for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
    conv_height = 4 if i == 0 else 1
    reg_model.add(Convolution2D(
        nb_filter=nb_filter, nb_row=conv_height,
        nb_col=nb_col, activation='relu',
        init='he_normal', input_shape=(1, 4, seq_length),
        W_regularizer=l1(L1), b_regularizer=l1(L1)))

reg_model.add(Flatten())

# Add expression network:
expr_model=Sequential()
for units in num_units:
	expr_model.add(Dense(units=units,
		activation='relu',input_shape=num_tf,
		kernel_regularizer=l1(L1),bias_regularizer=l1(L1)))

