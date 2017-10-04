from keras.models import Model
from keras.layers import Input, Dense, \
	RevCompConv1D, Flatten, Dropout, Merge, \
	Activation, MaxPooling1D, DenseAfterRevcompConv1D, \
	RevCompConv1DBatchNorm, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, \
	ModelCheckpoint, ReduceLROnPlateau
from keras.utils.visualize_util import plot


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from ggplot import *
import pandas as pd

import sys
sys.path.append('utils/')
import pickle
import numpy as np
import os
import input_data_config as input_data
from callbacks import BatchHistory


num_reg=472
seq_length=1000
run_number=sys.argv[1]
dir_suffix='/revcomp/regression/'
fig_dir='../figures/%s/%s'%(dir_suffix,run_number)
log_dir='../logs/%s/%s'%(dir_suffix,run_number)
out_dir='../processed_data/%s/%s'%(dir_suffix,run_number)

if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)


print('INFO - %s'%('building regression model.'))
units=512
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
reg_output=Dense(units,activation='relu')(reg_input)


print('INFO - %s'%('building sequence model.'))
units=512
filters=[50,50]
kernel_width=[9,9]
pool_size=[4,4]

seq_input=Input(shape=(seq_length,4),dtype='float32',name='seq_input')
x=seq_input
for i in range(2):
	x=RevCompConv1D(nb_filter=filters[i],filter_length=kernel_width[i],border_mode='valid')(x)
	x=RevCompConv1DBatchNorm()(x)
	x=Activation('relu')(x)
	x=MaxPooling1D(pool_length=pool_size[i],border_mode='valid')(x)
seq_output=DenseAfterRevcompConv1D(output_dim=units)(x)


print('INFO - %s'%('building concatenate model.'))
units=[512,512]
x=Merge(mode='concat',concat_axis=1)([reg_output,seq_output])
for i in range(2):
	x=Dense(units[i])(x)
	x=BatchNormalization()(x)
	x=Activation('relu')(x)
	x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)


model=Model(input=[reg_input,seq_input],output=[rgs_output])
model.compile(loss={'rgs_output':'mean_squared_error'},optimizer=SGD(lr=0.01,momentum=0.5))
plot(model, show_shapes=True, to_file='%s/model.eps'%(fig_dir))

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10,conv1d=True)


print('INFO - %s'%('training model.'))
reduce_lr=ReduceLROnPlateau(verbose=1,factor=0.5, patience=5)
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
checkpoint=ModelCheckpoint(filepath="%s/model.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir), monitor='val_loss')
history=model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},
	validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),
	nb_epoch=200,
	batch_size=100,
	callbacks=[early_stopping,checkpoint,reduce_lr],
	verbose=1)
with open('%s/history.pkl'%(log_dir),'wb') as f:
	pickle.dump([history.history],f)
with open('%s/history.pkl'%(log_dir),'rb') as f:
	x=pickle.load(f)


# Plot the learning curve:
history=pd.DataFrame(x[0])
history['epoch']=(range(1,history.shape[0]+1))
history_melt=pd.melt(history,id_vars=['epoch'],value_vars=['loss','val_loss'],var_name='type',value_name='loss')

p1=ggplot(history_melt,aes('epoch','loss',color='type'))+geom_line()+theme_bw()
p1.save(filename='%s/learning_curve.png'%(fig_dir))


# Plot prediction vs ground truth: 
pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.scatter(pred,test['expr'])
plt.savefig("%s/pred_vs_obs.png"%(fig_dir))
output=np.column_stack((test['expr'], pred[:,0]))
np.savetxt("%s/prediction.txt"%(out_dir), output,delimiter='\t')
