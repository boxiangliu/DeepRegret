from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D, Reshape
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot

import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K

import numpy as np
import sys,os
sys.path.append('modeling/maxpool_outer_product/')
import input_data



num_reg=472
seq_length=1000
fig_dir='../figures/maxpool_outer_product/model/'
log_dir='../logs/maxpool_outer_product/model/'
out_dir='../processed_data/maxpool_outer_product/model/'
if not os.path.exists(fig_dir): os.mkdir(fig_dir)
if not os.path.exists(log_dir): os.mkdir(log_dir)
if not os.path.exists(out_dir): os.mkdir(out_dir)

print('INFO - %s'%('building regression model.'))
units=512
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
reg_output=Reshape((num_reg,1))(reg_input)

print('INFO - %s'%('building sequence model.'))
filters=[256]
kernel_width=[19]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid')(seq_input)
x=MaxPooling2D(pool_size=(1,982))(x)
x=Activation('relu')(x)
seq_output=Reshape((1,filters[0]))(x)



print('INFO - %s'%('building concatenate model.'))
units=512
x=Merge(mode=lambda x: x[0] * x[1],output_shape=(num_reg,filters[0]))([reg_output,seq_output])
x=Reshape((num_reg,filters[0],1))(x)
x=MaxPooling2D(pool_size=(1,256))(x)
x=Flatten()(x)
x=Dense(units,activation='relu')(x)
x=Dropout(0.5,seed=42)(x)
x=Dense(units,activation='relu')(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)


model=Model(input=[reg_input,seq_input],output=[rgs_output])
model.compile(loss={'rgs_output':'mean_squared_error'},optimizer='sgd')
plot(model, show_shapes=True,to_file='%s/model.eps'%(fig_dir))

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets()


print('INFO - %s'%('training model.'))
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
checkpoint=ModelCheckpoint(filepath="%s/model.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir), monitor='val_loss',save_best_only=True)
history=model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),nb_epoch=30,batch_size=100,callbacks=[early_stopping,checkpoint],verbose=1)
with open('%s/history'%(log_dir),'wb') as f:
	pickle.dump(history.history,f)

with open('%s/history'%(log_dir),'rb') as f:
	history=pickle.load(f)
plt.ion()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['train','val'])
plt.savefig('%s/learning_curve.png'%(fig_dir))
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.scatter(pred,test['expr'])
plt.savefig("%s/pred_vs_obs.png"%(fig_dir))
output=np.column_stack((test['expr'], pred[:,0]))
np.savetxt("%s/prediction.txt"%(out_dir), output,delimiter='\t')


