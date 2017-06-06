from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, Flatten, Dropout, Merge, Activation, MaxPooling2D, Reshape
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.utils.visualize_util import plot
from keras.regularizers import l1l2

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import sys,os
import argparse
from datetime import datetime
sys.path.append('/srv/scratch/bliu2/deepregret/scripts/modeling/single_layer/')
import input_data



# Args:
parser = argparse.ArgumentParser(description='Tune regularization param.')
parser.add_argument('--l1', default=0.01,type=float, help='l1 regularization')
parser.add_argument('--l2', default=0.01,type=float, help='l2 regularization')
parser.add_argument('--epochs', default=30,type=int, help='number of epochs')
args = parser.parse_args()

print(args)


# Variables:
num_reg=472
seq_length=1000
time=datetime.now().strftime("%Y%m%d_%H%M%S")
fig_dir='../figures/single_layer/%s_l1_%s_l2_%s'%(time,args.l1,args.l2)
log_dir='../logs/single_layer/%s_l1_%s_l2_%s'%(time,args.l1,args.l2)
out_dir='../processed_data/single_layer/%s_l1_%s_l2_%s'%(time,args.l1,args.l2)
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
x=Convolution2D(nb_filter=filters[0],nb_row=4,nb_col=kernel_width[0],subsample=(1,1),border_mode='valid',W_regularizer=l1l2(l1=args.l1, l2=args.l1),b_regularizer=l1l2(l1=args.l1, l2=args.l1))(seq_input)
x=MaxPooling2D(pool_size=(1,982))(x)
x=Activation('relu')(x)
seq_output=Reshape((1,filters[0]))(x)



print('INFO - %s'%('building concatenate model.'))
units=512
x=Merge(mode=lambda x: x[0] * x[1],output_shape=(num_reg,filters[0]))([reg_output,seq_output])
x=Flatten()(x)
x=Dense(units,activation='relu',W_regularizer=l1l2(l1=args.l1, l2=args.l1),b_regularizer=l1l2(l1=args.l1, l2=args.l1))(x)
x=Dropout(0.5,seed=42)(x)
x=Dense(units,activation='relu',W_regularizer=l1l2(l1=args.l1, l2=args.l1),b_regularizer=l1l2(l1=args.l1, l2=args.l1))(x)
x=Dropout(0.5,seed=42)(x)
rgs_output=Dense(1,activation='linear',name='rgs_output')(x)


model=Model(input=[reg_input,seq_input],output=[rgs_output])
model.compile(loss={'rgs_output':'mean_squared_error'},optimizer='sgd')
plot(model, show_shapes=True,to_file='%s/model.eps'%(fig_dir))

print('INFO - %s'%('loading data.'))
train,val,test=input_data.read_data_sets()


print('INFO - %s'%('training model.'))
early_stopping=EarlyStopping(monitor='val_loss',patience=10)
checkpoint=ModelCheckpoint(filepath="%s/model.{epoch:02d}-{val_loss:.4f}.hdf5"%(log_dir), monitor='val_loss')
history=History()
model.fit({'seq_input':train['seq'],'reg_input':train['reg']},{'rgs_output':train['expr']},validation_data=({'seq_input':val['seq'],'reg_input':val['reg']},{'rgs_output':val['expr']}),nb_epoch=args.epochs,batch_size=100,callbacks=[early_stopping,checkpoint,history],verbose=1)


pred=model.predict({'seq_input':test['seq'],'reg_input':test['reg']},batch_size=100,verbose=1)
plt.scatter(pred,test['expr'])
plt.savefig("%s/pred_vs_obs.png"%(fig_dir))
output=np.column_stack((test['expr'], pred[:,0]))
np.savetxt("%s/prediction.txt"%(out_dir), output,delimiter='\t')


