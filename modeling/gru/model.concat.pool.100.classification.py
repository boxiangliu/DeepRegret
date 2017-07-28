from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D, Reshape, GRU, Bidirectional,RepeatVector
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.utils.visualize_util import plot

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
import sys
import os
sys.path.append('utils/')
import input_data_config as input_data





num_reg=472
seq_length=1000
dir_suffix='gru/concat.pool.100.classification/'
fig_dir='../figures/%s'%(dir_suffix)
log_dir='../logs/%s'%(dir_suffix)
out_dir='../processed_data/%s'%(dir_suffix)

if not os.path.exists(fig_dir): os.makedirs(fig_dir)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(out_dir): os.makedirs(out_dir)


print('INFO - %s'%('building sequence model.'))
filters=[256]
kernel_width=[19]
seq_input=Input(shape=(4,seq_length,1),dtype='float32',name='seq_input')
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid')(seq_input)
x=MaxPooling2D(pool_size=(1,100),border_mode='same')(x)
x=Activation('relu')(x)
length=int(x.shape[2])
seq_output=Reshape((length,filters[0]))(x)

print('INFO - %s'%('building regression model.'))
reg_input=Input(shape=(num_reg,),dtype='float32',name='reg_input')
reg_output=RepeatVector(length)(reg_input)


print('INFO - %s'%('building concatenate model.'))
units=512
x=Merge(mode='concat',concat_axis=-1)([reg_output,seq_output])
x=Bidirectional(GRU(units,dropout_W=0.5,dropout_U=0.5))(x)
x=Dense(units,activation='relu')(x)
x=Dropout(0.5,seed=42)(x)
x=Dense(units,activation='relu')(x)
x=Dropout(0.5,seed=42)(x)
cls_output=Dense(3,activation='sigmoid',name='cls_output')(x)



model=Model(input=[reg_input,seq_input],output=[cls_output])
model.compile(loss={'cls_output':'categorical_crossentropy'},optimizer='adam')
plot(model, show_shapes=True,to_file='%s/model.eps'%(fig_dir))

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets(train_pct=80,val_pct=10,test_pct=10)



print('INFO - %s'%('training model.'))
reduce_lr=ReduceLROnPlateau(verbose=1,factor=0.5, patience=5)
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
checkpoint=ModelCheckpoint(filepath="%s/model.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir), monitor='val_loss',save_best_only=True)
history=model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'cls_output':to_categorical(train['class'])},
	validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'cls_output':to_categorical(val['class'])}),
	nb_epoch=100,
	batch_size=100,
	callbacks=[early_stopping,checkpoint,reduce_lr],
	verbose=1)
with open('%s/history.pkl'%(log_dir),'wb') as f:
	pickle.dump(history.history,f)

pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
pred_class=pred.argmax(axis=1)

cm=confusion_matrix(test['class'], pred_class)
np.savetxt('%s/confusion_matrix.txt'%(out_dir),cm,fmt='%i',delimiter='\t')
np.savetxt('%s/confusion_matrix_pct.txt'%(out_dir),cm/float(cm.sum())*100,fmt='%.10f',delimiter='\t')

output=np.column_stack((test['class'], pred_class))
np.savetxt("%s/prediction.txt"%(out_dir), output,delimiter='\t')

