from keras.models import Model
from keras.layers import Input, Dense, Conv2D, concatenate, Flatten, Dropout, Concatenate
from keras.regularizers import l1
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
import numpy as np
import sys
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/keras/')
import input_data

num_reg=472
seq_length=1000
L1=0.01

units=512
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
x=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(reg_input)
x=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
reg_output=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)


filters=[32,64,128]
kernel_width=15
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Conv2D(filters=filters[0],kernel_size=(4,kernel_width),strides=1,padding='valid',activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(seq_input)
x=Conv2D(filters=filters[1],kernel_size=(1,kernel_width),strides=1,padding='valid',activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
x=Conv2D(filters=filters[2],kernel_size=(1,kernel_width),strides=1,padding='valid',activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
seq_output=Flatten()(x)


units=512
x=concatenate([reg_output,seq_output])
x=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
x=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
x=Dense(units,activation='relu',kernel_regularizer=l1(L1),bias_regularizer=l1(L1))(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',kernel_regularizer=l1(L1),bias_regularizer=l1(L1),name='rgs_output')(x)


model=Model(inputs=[reg_input,seq_input],outputs=[rgs_output])

early_stopping=EarlyStopping(monitor='val_loss',patience=2)
model.compile(loss={'rgs_output':'mean_squared_error'},optimizer='adam')
plot_model(model, show_shapes=True,to_file='../figures/keras/model.eps')
train,val,test=input_data.read_data_sets()
model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),epochs=10,batch_size=100,callbacks=[early_stopping],verbose=1)
model.evaluate({'seq_input':test['seq'],'reg_input':test['reg']},{'rgs_output':test['expr']},batch_size=100)
model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
